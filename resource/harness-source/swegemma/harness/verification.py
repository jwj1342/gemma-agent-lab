"""Phase 2 evaluation container verification and test execution."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import tempfile
import time
import uuid
from pathlib import Path

from adk_eval_core.tracing import SessionTrace

from swegemma.config import EvalConfig
from swegemma.harness.container_setup import (
    ContainerSetupError,
    apply_patch_in_container,
    extract_snapshot,
    install_editable_package,
    install_test_dependencies,
    setup_baseline_commit,
    setup_container_wheels,
    setup_git_exclude,
    setup_synthetic_hardware,
    setup_workspace_test_config,
)
from swegemma.models import Task, TaskResult, extract_test_files_from_patch
from swegemma.sandbox import (
    BaseSandboxManager,
    sandbox_exec,
    sandbox_start,
    sandbox_stop,
)

logger = logging.getLogger(__name__)


def save_trace_artifact(
    trace: SessionTrace | None,
    results_dir: Path | None,
    instance_id: str,
) -> str | None:
    """Saves a SessionTrace to results_dir/traces/trace_<instance_id>.json if available."""
    if trace is None or results_dir is None:
        return None
    try:
        trace_dir = results_dir / 'traces'
        trace_dir.mkdir(parents=True, exist_ok=True)
        safe_id = instance_id.replace('/', '__')
        trace_path = trace_dir / f'trace_{safe_id}.json'
        trace.save(trace_path)
        return str(trace_path)
    except Exception as tr_err:
        logger.warning('Failed to save trace for task %s: %s', instance_id, tr_err)
        return None


def _is_protected_test_or_config_path(rel_path: str) -> bool:
    """Returns True if rel_path is a test file, conftest, or runner config that agent_patch must not tamper with."""
    p = Path(rel_path)
    name = p.name
    if name in {
        'conftest.py',
        'pytest.ini',
        'pyproject.toml',
        'tox.ini',
        'setup.cfg',
        '.pytest.ini',
        'sitecustomize.py',
        'usercustomize.py',
        '_swegemma_stubs.py',
    }:
        return True
    if name.endswith('.pth'):
        return True
    if name.startswith('test_') and name.endswith('.py'):
        return True
    if name.endswith('_test.py'):
        return True
    parts = {part.lower() for part in p.parts[:-1]}
    return bool({'tests', 'test', 'testing'} & parts and name.endswith('.py'))


def _extract_test_functions_from_patch(test_patch: str) -> list[str]:
    """Extract added or modified test_* function names from a unified diff test_patch."""
    if not test_patch:
        return []
    added = re.findall(
        r'^\+\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(',
        test_patch,
        re.MULTILINE,
    )
    if added:
        return list(dict.fromkeys(added))
    removed = set(
        re.findall(
            r'^-\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(',
            test_patch,
            re.MULTILINE,
        )
    )
    hunk_headers = re.findall(
        r'^@@[^@]+@@\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\b',
        test_patch,
        re.MULTILINE,
    )
    context_defs = re.findall(
        r'^ \s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(',
        test_patch,
        re.MULTILINE,
    )
    context_set = set(context_defs)
    combined = list(dict.fromkeys(hunk_headers + context_defs))
    return [
        fn
        for fn in combined
        if fn not in removed
        and not (
            fn not in context_set
            and (
                any(r.startswith(fn) for r in removed)
                or any(c != fn and c.startswith(fn) for c in context_set)
            )
        )
    ]


def _node_matches_requirement(node_name: str, req: str, req_short: str, req_base: str) -> bool:
    """Check if a JUnit testcase identifier matches a required test specification."""
    if node_name in (req, req_short, req_base):
        return True
    short_node = node_name.rsplit('::', 1)[-1].rsplit('.', 1)[-1]
    if short_node in (req, req_short, req_base):
        return True
    if short_node.startswith(f'{req_base}_') or short_node.startswith(f'{req_base}['):
        return True
    if len(req_base) >= 68 and short_node.startswith(req_base):
        return True
    return short_node.startswith(req_base) and short_node[len(req_base) :].isdigit()


def _validate_junit_xml(
    xml_content: str,
    fail_to_pass: tuple[str, ...] = (),
    pass_to_pass: tuple[str, ...] = (),
    test_patch: str = '',
) -> tuple[bool, str | None]:
    """Validate JUnit XML report to prevent os._exit(0), conftest, or skip-all exploits."""
    import xml.etree.ElementTree as ET

    if not xml_content or not xml_content.strip():
        return False, 'Missing or empty JUnit XML report (possible premature exit)'
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return False, f'Malformed JUnit XML report: {e}'

    suites = [root] if root.tag == 'testsuite' else list(root.findall('.//testsuite'))
    if not suites:
        return False, 'No <testsuite> elements found in JUnit XML'

    total_tests = sum(int(s.attrib.get('tests', 0)) for s in suites)
    total_failures = sum(int(s.attrib.get('failures', 0)) for s in suites)
    total_errors = sum(int(s.attrib.get('errors', 0)) for s in suites)
    total_skipped = sum(int(s.attrib.get('skipped', 0)) for s in suites)
    passed_tests = total_tests - total_failures - total_errors - total_skipped

    if total_tests <= 0 or passed_tests <= 0:
        return (
            False,
            f'No passing tests recorded in JUnit XML (total={total_tests}, passed={passed_tests}, skipped={total_skipped})',
        )
    if total_failures > 0 or total_errors > 0:
        return (
            False,
            f'Test failures/errors recorded in JUnit XML (failures={total_failures}, errors={total_errors})',
        )

    required_nodes = list(fail_to_pass) + list(pass_to_pass)
    if not required_nodes and test_patch:
        required_nodes = _extract_test_functions_from_patch(test_patch)
    if required_nodes:
        passed_names: set[str] = set()
        failed_or_skipped_names: set[str] = set()
        for tc in root.findall('.//testcase'):
            name = tc.attrib.get('name', '')
            classname = tc.attrib.get('classname', '')
            candidates = {name}
            if '[' in name:
                candidates.add(name.split('[', 1)[0])
            if classname:
                candidates.add(f'{classname}::{name}')
                candidates.add(f'{classname}.{name}')
                mod_path = classname.replace('.', '/') + '.py'
                candidates.add(f'{mod_path}::{name}')
            if (
                tc.find('failure') is None
                and tc.find('error') is None
                and tc.find('skipped') is None
            ):
                passed_names.update(candidates)
            else:
                failed_or_skipped_names.update(candidates)
        for req in required_nodes:
            req_short = req.split('::')[-1]
            req_base = req_short.split('[', 1)[0]
            has_pass = any(
                _node_matches_requirement(p, req, req_short, req_base)
                for p in passed_names
            )
            has_non_pass = any(
                _node_matches_requirement(np, req, req_short, req_base)
                for np in failed_or_skipped_names
            )
            if not has_pass or has_non_pass:
                return False, f'Required test node did not pass: {req}'

    return True, None


async def verify_task(
    docker: BaseSandboxManager,
    config: EvalConfig,
    task: Task,
    snapshot_path: Path,
    *,
    base_snapshot_path: Path | None = None,
    patch_path: Path | None = None,
    fast_path: bool = True,
    agent_patch: str = '',
    agent_error: str | None = None,
    trace: SessionTrace | None = None,
    start_time: float = 0.0,
) -> TaskResult:
    """Verifies a generated patch inside the clean evaluation container (Container B)."""
    is_synthetic_mock_repo = (
        task.repo.startswith('test/mock_repo') and task.base_commit == '123456'
    )
    if (
        not task.test_patch.strip()
        and not getattr(task, 'FAIL_TO_PASS', ())
        and not getattr(task, 'PASS_TO_PASS', ())
        and not getattr(task, 'test_files', None)
        and not is_synthetic_mock_repo
    ):
        json_path = save_trace_artifact(trace, config.results_dir, task.instance_id)
        return TaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            resolved=False,
            agent_patch=agent_patch,
            test_output='',
            test_exit_code=-1,
            duration_seconds=time.perf_counter() - start_time,
            error='Missing test specification (empty test_patch and FAIL_TO_PASS; secret hydration failed or unavailable)',
            trace=trace,
            trace_json_path=json_path,
        )

    eval_id = await sandbox_start(docker)
    try:
        # Setup wheels in container
        await asyncio.to_thread(setup_container_wheels, docker, eval_id, config)

        # Copy snapshot into eval container and extract (supports base + patch)
        await asyncio.to_thread(
            extract_snapshot,
            docker,
            eval_id,
            snapshot_path,
            patch_path=patch_path,
            base_snapshot_path=base_snapshot_path,
        )

        # Setup synthetic environment stubs if provided by sandbox setup
        await asyncio.to_thread(setup_synthetic_hardware, docker, eval_id)

        # Configure git exclude to ignore bytecode and test cache artifacts
        await asyncio.to_thread(setup_git_exclude, docker, eval_id)

        # Install editable package
        await asyncio.to_thread(install_editable_package, docker, eval_id)

        # Install test dependencies (fast-path by default)
        await asyncio.to_thread(
            install_test_dependencies,
            docker,
            eval_id,
            task.repo,
            fast_path=fast_path,
            config=config,
        )

        # Configure pytest.ini and conftest.py for hermetic test discovery
        await asyncio.to_thread(
            setup_workspace_test_config,
            docker,
            eval_id,
            repo=task.repo,
        )

        # Create clean baseline commit so git checkout has a reliable HEAD
        await asyncio.to_thread(setup_baseline_commit, docker, eval_id, 'eval_baseline')

        # Apply agent patch if present
        if agent_patch.strip():
            with tempfile.NamedTemporaryFile('w', suffix='.patch', delete=False) as f:
                f.write(
                    agent_patch if agent_patch.endswith('\n') else agent_patch + '\n'
                )
                patch_host_path = Path(f.name)

            try:
                await asyncio.to_thread(
                    docker.copy_to, eval_id, patch_host_path, '/tmp/'
                )
                exit_code, stdout, stderr = await asyncio.to_thread(
                    apply_patch_in_container,
                    docker,
                    eval_id,
                    f'/tmp/{patch_host_path.name}',
                )
                if exit_code != 0:
                    err_detail = stderr or stdout
                    if 'Patch file not found:' in err_detail:
                        raise ContainerSetupError(
                            f'Agent patch file missing in container on task {task.instance_id}: {err_detail}'
                        )
                    logger.error(
                        'Failed to apply agent patch on task %s: %s',
                        task.instance_id,
                        err_detail,
                    )
                    agent_error = f'Failed to apply agent patch: {err_detail}'
                    json_path = save_trace_artifact(
                        trace, config.results_dir, task.instance_id
                    )
                    return TaskResult(
                        instance_id=task.instance_id,
                        repo=task.repo,
                        resolved=False,
                        agent_patch=agent_patch,
                        test_output=err_detail,
                        test_exit_code=exit_code,
                        duration_seconds=time.perf_counter() - start_time,
                        error=agent_error,
                        trace=trace,
                        trace_json_path=json_path,
                    )
            finally:
                patch_host_path.unlink(missing_ok=True)

        target_test_files = (
            list(task.test_files)
            if task.test_files
            else extract_test_files_from_patch(task.test_patch)
        )

        # Collect any test files, conftest.py, or runner config files touched by agent_patch (including deleted/renamed sources)
        agent_touched_files = extract_test_files_from_patch(agent_patch)
        if agent_patch.strip():
            for raw_line in agent_patch.splitlines():
                if raw_line.startswith('diff --git '):
                    m_git = re.match(r'^diff --git "?a/(.*?)"? "?b/(.*?)"?$', raw_line)
                    if m_git:
                        for grp in (m_git.group(1), m_git.group(2)):
                            if grp and grp not in agent_touched_files:
                                agent_touched_files.append(grp)
                elif raw_line.startswith(('--- a/', '+++ b/')):
                    rel_p = raw_line[6:].strip().strip('"')
                    if rel_p and rel_p != '/dev/null' and rel_p not in agent_touched_files:
                        agent_touched_files.append(rel_p)
                elif raw_line.startswith('rename from '):
                    rel_p = raw_line[len('rename from ') :].strip().strip('"')
                    if rel_p and rel_p not in agent_touched_files:
                        agent_touched_files.append(rel_p)
            git_status_res = await sandbox_exec(
                docker,
                eval_id,
                'cd /workspace && (git diff --name-only HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null) || true',
            )
            for line in (git_status_res.stdout or '').splitlines():
                rel_line = line.strip()
                if rel_line and rel_line not in agent_touched_files:
                    agent_touched_files.append(rel_line)

        protected_agent_files = [
            p for p in agent_touched_files if _is_protected_test_or_config_path(p)
        ]
        files_to_reset = list(
            dict.fromkeys(
                target_test_files
                + protected_agent_files
                + [
                    'conftest.py',
                    'pytest.ini',
                    'sitecustomize.py',
                    'usercustomize.py',
                    '_swegemma_stubs.py',
                ]
            )
        )

        # Reset test and config files to baseline to prevent test tampering or conflicts with test_patch
        if files_to_reset:
            test_files_quoted = [shlex.quote(p) for p in files_to_reset]
            test_files_str = ' '.join(test_files_quoted)
            checkout_cmd = f'cd /workspace && git checkout HEAD -- {test_files_str} 2>/dev/null || true'
            clean_cmd = (
                f'cd /workspace && git clean -f -- {test_files_str} 2>/dev/null || true'
            )
            await sandbox_exec(docker, eval_id, checkout_cmd)
            await sandbox_exec(docker, eval_id, clean_cmd)

        # Apply test patch (verification tests)
        if task.test_patch.strip():
            with tempfile.NamedTemporaryFile('w', suffix='.patch', delete=False) as f:
                f.write(
                    task.test_patch
                    if task.test_patch.endswith('\n')
                    else task.test_patch + '\n'
                )
                test_patch_host_path = Path(f.name)

            try:
                await asyncio.to_thread(
                    docker.copy_to, eval_id, test_patch_host_path, '/tmp/'
                )
                exit_code, stdout, stderr = await asyncio.to_thread(
                    apply_patch_in_container,
                    docker,
                    eval_id,
                    f'/tmp/{test_patch_host_path.name}',
                )
                if exit_code != 0:
                    err_detail = stderr or stdout
                    if 'Patch file not found:' in err_detail:
                        raise ContainerSetupError(
                            f'Test patch file missing in container on task {task.instance_id}: {err_detail}'
                        )
                    logger.error(
                        'Failed to apply test patch on task %s: %s',
                        task.instance_id,
                        err_detail,
                    )
                    json_path = save_trace_artifact(
                        trace, config.results_dir, task.instance_id
                    )
                    return TaskResult(
                        instance_id=task.instance_id,
                        repo=task.repo,
                        resolved=False,
                        agent_patch=agent_patch,
                        test_output='',
                        test_exit_code=-1,
                        duration_seconds=time.perf_counter() - start_time,
                        error=f'Failed to apply test_patch: {err_detail}',
                        trace_json_path=json_path,
                    )
            finally:
                test_patch_host_path.unlink(missing_ok=True)

        # Refresh workspace paths, stubs, and pytest config after patches add new files/imports
        await asyncio.to_thread(
            install_test_dependencies,
            docker,
            eval_id,
            task.repo,
            fast_path=fast_path,
            config=config,
        )
        await asyncio.to_thread(
            setup_workspace_test_config,
            docker,
            eval_id,
            repo=task.repo,
            overwrite=True,
        )

        # Run pytest on verification test files (excluding conftest.py from positional test args)
        pytest_targets = [
            p
            for p in target_test_files
            if Path(p).name != 'conftest.py' and p.endswith('.py')
        ] or target_test_files
        test_files_quoted = [shlex.quote(p) for p in pytest_targets]
        test_files_str = ' '.join(test_files_quoted) if test_files_quoted else '.'
        junit_xml_path = f'/tmp/_swegemma_junit_{uuid.uuid4().hex[:12]}.xml'
        await sandbox_exec(docker, eval_id, f'rm -f {junit_xml_path}')
        pytest_cmd = (
            f'cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest {test_files_str} '
            f'--junitxml={junit_xml_path} '
            '-p no:anyio -o timeout=0 '
            '-o python_classes="Test* *Test" -q'
        )
        test_res = await sandbox_exec(
            docker,
            eval_id,
            pytest_cmd,
            timeout=config.harness.command_timeout_seconds,
        )

        resolved = test_res.exit_code == 0
        junit_error: str | None = None
        if resolved:
            xml_res = await sandbox_exec(
                docker,
                eval_id,
                f'cat {junit_xml_path} 2>/dev/null || true',
            )
            xml_content = xml_res.stdout or ''
            is_real_sandbox = type(docker).__name__ in {
                'ContainerManager',
                'SubprocessManager',
                'DockerSandbox',
                'SubprocessSandbox',
            }
            stdout_text = (test_res.stdout or '').lower()
            stdout_invalid = (
                bool(re.search(r'\b0 passed\b', stdout_text))
                or 'no tests ran' in stdout_text
                or 'collected 0 items' in stdout_text
                or (stdout_text.strip() != '' and 'passed' not in stdout_text)
            )
            if xml_content.strip():
                junit_ok, junit_reason = _validate_junit_xml(
                    xml_content,
                    fail_to_pass=getattr(task, 'FAIL_TO_PASS', ()),
                    pass_to_pass=getattr(task, 'PASS_TO_PASS', ()),
                    test_patch=getattr(task, 'test_patch', ''),
                )
                if not junit_ok:
                    resolved = False
                    junit_error = junit_reason
                elif stdout_invalid:
                    resolved = False
                    junit_error = 'Pytest stdout summary indicates zero or no passing tests'
            elif is_real_sandbox or stdout_invalid or not stdout_text.strip():
                resolved = False
                junit_error = 'Missing JUnit XML report (possible premature os._exit(0))'

        duration = time.perf_counter() - start_time

        json_path = save_trace_artifact(trace, config.results_dir, task.instance_id)

        return TaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            resolved=resolved,
            agent_patch=agent_patch,
            test_output=f'STDOUT:\n{test_res.stdout}\nSTDERR:\n{test_res.stderr}',
            test_exit_code=test_res.exit_code,
            duration_seconds=duration,
            error=(junit_error or agent_error) if not resolved else None,
            trace=trace,
            trace_json_path=json_path,
        )
    except ContainerSetupError:
        raise
    except Exception as e:
        logger.exception(
            'Error during evaluation container run for task %s', task.instance_id
        )
        return TaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            resolved=False,
            agent_patch=agent_patch,
            test_output='',
            test_exit_code=-1,
            duration_seconds=time.perf_counter() - start_time,
            error=f'Evaluation error: {e}',
            trace=trace,
        )
    finally:
        await sandbox_stop(docker, eval_id)
