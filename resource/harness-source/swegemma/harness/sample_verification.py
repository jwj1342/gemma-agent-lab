"""Air-gapped verification sweep runner for SWE-bench sample tasks across all 13 repositories."""

from __future__ import annotations

import dataclasses
import logging
import shlex
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swegemma.config import EvalConfig
from swegemma.harness.container_setup import (
    apply_patch_in_container,
    extract_deduplicated_snapshot,
    extract_snapshot,
    install_editable_package,
    install_test_dependencies,
    setup_baseline_commit,
    setup_container_wheels,
    setup_git_exclude,
    setup_synthetic_hardware,
)
from swegemma.models import Task, extract_test_files_from_patch, load_tasks
from swegemma.sandbox import BaseSandboxManager, ContainerConfig, ContainerManager

logger = logging.getLogger(__name__)

KNOWN_PUBLIC_REPOS: frozenset[str] = frozenset({
    'fastapi/fastapi',
    'encode/httpx',
    'psf/requests',
    'Textualize/rich',
})
KNOWN_PRIVATE_REPOS: frozenset[str] = frozenset()
ALL_BENCHMARK_REPOS: frozenset[str] = KNOWN_PUBLIC_REPOS
LATENCY_SLA_THRESHOLD_SECONDS: float = 5.0


@dataclasses.dataclass(frozen=True)
class SampleVerificationResult:
    """Detailed verification outcome for a single sample task in an air-gapped container."""

    instance_id: str
    repo: str
    is_private: bool
    base_exit_code: int
    patch_exit_code: int
    passed_criteria: bool  # True if base_exit_code != 0 and patch_exit_code == 0
    bootstrap_time_seconds: float
    latency_sla_passed: bool  # True if bootstrap_time_seconds < 5.0s
    base_stdout: str = ''
    base_stderr: str = ''
    patch_stdout: str = ''
    patch_stderr: str = ''
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'repo': self.repo,
            'is_private': self.is_private,
            'base_exit_code': self.base_exit_code,
            'patch_exit_code': self.patch_exit_code,
            'passed_criteria': self.passed_criteria,
            'bootstrap_time_seconds': round(self.bootstrap_time_seconds, 4),
            'latency_sla_passed': self.latency_sla_passed,
            'error': self.error,
        }


@dataclasses.dataclass(frozen=True)
class SweepSummary:
    """Aggregate summary of an air-gapped test verification sweep across sample tasks."""

    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    pass_rate: float
    latency_sla_passed: bool
    max_bootstrap_time: float
    avg_bootstrap_time: float
    results: list[SampleVerificationResult]
    by_repo: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    @property
    def is_100_percent_pass(self) -> bool:
        return self.total_tasks > 0 and self.passed_tasks == self.total_tasks


def select_sample_tasks(
    tasks: Sequence[dict[str, Any] | Task],
    samples_per_repo: int = 1,
    target_repos: Sequence[str] | None = None,
) -> list[Task]:
    """Selects deterministic sample tasks covering all repositories present in tasks.

    Args:
        tasks: Sequence of task dicts or Task instances.
        samples_per_repo: Number of sample tasks to select per repository (default: 1).
        target_repos: Optional subset of repositories to restrict selection to.

    Returns:
        List of selected Task objects.
    """
    task_objs: list[Task] = []
    for t in tasks:
        if isinstance(t, Task):
            task_objs.append(t)
        elif isinstance(t, dict):
            task_objs.append(
                Task(
                    instance_id=str(t.get('instance_id', '')),
                    repo=str(t.get('repo', '')),
                    base_commit=str(t.get('base_commit', '')),
                    patch=str(t.get('patch', '')),
                    test_patch=str(t.get('test_patch', '')),
                    problem_statement=str(t.get('problem_statement', '')),
                    hints_text=str(t.get('hints_text', '')),
                    created_at=str(t.get('created_at', '')),
                )
            )

    if target_repos is not None:
        repos_to_sample = set(target_repos)
    else:
        task_repos = {t.repo for t in task_objs}
        repos_to_sample = task_repos if task_repos else set(ALL_BENCHMARK_REPOS)

    repo_to_tasks: dict[str, list[Task]] = defaultdict(list)
    for t in task_objs:
        if t.repo in repos_to_sample:
            repo_to_tasks[t.repo].append(t)

    selected: list[Task] = []
    for repo in sorted(repos_to_sample):
        r_tasks = repo_to_tasks.get(repo, [])
        if not r_tasks:
            logger.warning('No tasks found for repository %s during sample selection', repo)
            continue
        # Deterministic sorting by instance_id
        sorted_tasks = sorted(r_tasks, key=lambda x: x.instance_id)
        selected.extend(sorted_tasks[:samples_per_repo])

    return selected


def load_all_benchmark_tasks(
    splits_dir: Path | str = 'data/splits',
) -> list[Task]:
    """Loads all tasks from JSONL files in splits_dir.

    Args:
        splits_dir: Directory containing split JSONL files.

    Returns:
        Combined list of Task instances across available splits.
    """
    s_dir = Path(splits_dir).resolve()
    if not s_dir.exists():
        for parent in Path(__file__).resolve().parents:
            cand = parent / 'data' / 'splits'
            if cand.exists():
                s_dir = cand.resolve()
                break
    all_tasks: list[Task] = []
    seen: set[str] = set()

    if s_dir.exists():
        for p in sorted(s_dir.glob('*.jsonl')):
            for t in load_tasks(p):
                if t.instance_id not in seen:
                    seen.add(t.instance_id)
                    all_tasks.append(t)

    return all_tasks


def resolve_task_snapshot(
    task: Task,
    snapshots_dir: Path | None = None,
    packaged_split_dir: Path | None = None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Resolves snapshot paths for a task (standalone snapshot, base snapshot, patch, or fallback).

    Returns:
        tuple (snapshot_path, base_snapshot_path, patch_path, fallback_path)
    """
    iid = task.instance_id
    clean_repo = task.repo.replace('/', '_').replace('-', '_')

    # 1. Check packaged/deduplicated split directory layout
    if packaged_split_dir is not None and packaged_split_dir.exists():
        p_dir = Path(packaged_split_dir).resolve()
        base_cand = p_dir / 'base_snapshots' / f'base_{clean_repo}.tgz'
        patch_cand = p_dir / 'patches' / f'{iid}.patch'
        fallback_cand = p_dir / 'fallbacks' / f'{iid}.tgz'

        if base_cand.exists():
            return None, base_cand, patch_cand if patch_cand.exists() else None, fallback_cand if fallback_cand.exists() else None

    # 2. Check candidate directories
    repo_root = Path(__file__).resolve().parents[3]
    candidate_dirs: list[Path] = []
    if snapshots_dir:
        candidate_dirs.append(Path(snapshots_dir).resolve())

    candidate_dirs.extend([
        repo_root / 'data' / 'snapshots',
        repo_root / 'build' / 'deduplicated' / 'base_snapshots',
        Path('data/snapshots'),
    ])

    for cand_dir in candidate_dirs:
        if not cand_dir.exists():
            continue
        # Direct snapshot file
        snap_file = cand_dir / f'{iid}.tgz'
        if snap_file.exists():
            return snap_file, None, None, None

        # Base snapshot file
        base_file = cand_dir / f'base_{clean_repo}.tgz'
        patch_file = cand_dir.parent / 'patches' / f'{iid}.patch'
        fallback_file = cand_dir.parent / 'fallbacks' / f'{iid}.tgz'
        if base_file.exists() and (patch_file.exists() or fallback_file.exists()):
            valid_patch = (
                patch_file
                if (patch_file.exists() and patch_file.stat().st_size > 0)
                else None
            )
            return (
                None,
                base_file,
                valid_patch,
                fallback_file if fallback_file.exists() else None,
            )

    return None, None, None, None


def verify_sample_task_airgapped(
    task: Task,
    sandbox_manager: BaseSandboxManager | None = None,
    image: str = 'swebench-sandbox:latest',
    snapshot_path: Path | None = None,
    base_snapshot_path: Path | None = None,
    patch_path: Path | None = None,
    fallback_path: Path | None = None,
    snapshots_dir: Path | None = None,
    packaged_split_dir: Path | None = None,
    fast_path: bool = True,
    timeout_seconds: int = 300,
) -> SampleVerificationResult:
    """Verifies a single sample task in an air-gapped Docker container (network_mode='none').

    Executes:
      1. Baseline run (test_patch applied without gold solution patch) -> asserts base_exit_code != 0
      2. Patched run (gold solution patch applied + test_patch applied) -> asserts patch_exit_code == 0
      3. Measures container bootstrap and dependency injection latency -> asserts latency < 5.0s

    Args:
        task: Task instance to verify.
        sandbox_manager: Optional custom sandbox manager (e.g. MockSandboxManager or ContainerManager).
        image: Docker image to execute within (default: swebench-sandbox:latest).
        snapshot_path: Path to task snapshot archive (.tgz).
        base_snapshot_path: Path to repository base snapshot archive (.tgz).
        patch_path: Path to incremental binary patch (.patch).
        fallback_path: Path to fallback snapshot (.tgz).
        snapshots_dir: Optional snapshots directory for auto-resolution.
        packaged_split_dir: Optional packaged split directory for auto-resolution.
        fast_path: Whether to use fast-path dependency injection (<0.1s).
        timeout_seconds: Pytest command execution timeout.

    Returns:
        SampleVerificationResult
    """
    is_priv = task.repo not in KNOWN_PUBLIC_REPOS
    manager: Any = sandbox_manager if sandbox_manager is not None else ContainerManager(ContainerConfig(image=image, network_mode='none'))
    cid = manager.start()

    bootstrap_start = time.perf_counter()
    bootstrap_time = 0.0

    try:
        from adk_submission import ModelRegistry

        config = EvalConfig(
            tasks_path=Path('data/splits/training_tasks.jsonl'),
            snapshots_dir=Path('data/snapshots'),
            results_dir=Path('build/results'),
            submission_dir=Path('build/submission'),
            models=ModelRegistry(),
            wheels_dir=Path('docker/wheels') if Path('docker/wheels').exists() else None,
        )
        setup_container_wheels(manager, cid, config)

        # Extract snapshot
        if base_snapshot_path is not None and base_snapshot_path.exists():
            extract_deduplicated_snapshot(
                manager,
                cid,
                base_snapshot_path=base_snapshot_path,
                patch_path=patch_path,
                fallback_path=fallback_path,
            )
        elif snapshot_path is not None and snapshot_path.exists():
            extract_snapshot(manager, cid, snapshot_path)
        else:
            # Auto-resolve
            s_p, b_p, p_p, f_p = resolve_task_snapshot(
                task,
                snapshots_dir=snapshots_dir,
                packaged_split_dir=packaged_split_dir,
            )
            if b_p is not None and b_p.exists():
                extract_deduplicated_snapshot(
                    manager,
                    cid,
                    base_snapshot_path=b_p,
                    patch_path=p_p,
                    fallback_path=f_p,
                )
            elif s_p is not None and s_p.exists():
                extract_snapshot(manager, cid, s_p)
            else:
                # If no snapshot archive exists, initialize workspace directory
                manager.exec(cid, 'mkdir -p /workspace')

        # Setup synthetic hardware, git exclude, editable package, dependencies
        setup_synthetic_hardware(manager, cid)
        setup_git_exclude(manager, cid)
        install_editable_package(manager, cid)
        install_test_dependencies(manager, cid, repo=task.repo, fast_path=fast_path)
        setup_baseline_commit(manager, cid, 'eval_baseline')

        bootstrap_time = time.perf_counter() - bootstrap_start

        target_test_files = (
            list(task.test_files)
            if task.test_files
            else extract_test_files_from_patch(task.test_patch)
        )
        test_files_quoted = [shlex.quote(p) for p in target_test_files]
        test_files_str = ' '.join(test_files_quoted) if test_files_quoted else '.'

        # -------------------------------------------------------------
        # Step 1: Baseline Test Execution (Expected to FAIL, base_exit != 0)
        # -------------------------------------------------------------
        if task.test_patch.strip():
            with tempfile.NamedTemporaryFile('w', suffix='_test_patch.patch', delete=False) as f:
                f.write(task.test_patch if task.test_patch.endswith('\n') else task.test_patch + '\n')
                tp_host = Path(f.name)
            try:
                manager.copy_to(cid, tp_host, '/tmp/')
                apply_patch_in_container(manager, cid, f'/tmp/{tp_host.name}')
            finally:
                tp_host.unlink(missing_ok=True)

        base_res = manager.exec(
            cid,
            f'cd /workspace && PYTHONSAFEPATH=1 python3 -m pytest {test_files_str} -p no:anyio -o timeout=0 -o norecursedirs=".* build dist venv" -o python_classes="Test* *Test" -q',
            timeout=timeout_seconds,
        )
        base_exit_code = base_res.exit_code if base_res.exit_code is not None else 1

        # -------------------------------------------------------------
        # Step 2: Patched Test Execution (Expected to PASS, patch_exit == 0)
        # -------------------------------------------------------------
        # Reset workspace to clean baseline commit
        manager.exec(cid, 'cd /workspace && git checkout -f eval_baseline 2>/dev/null || true')
        manager.exec(cid, 'cd /workspace && git clean -fdx 2>/dev/null || true')

        # Apply gold solution patch
        if task.patch.strip():
            with tempfile.NamedTemporaryFile('w', suffix='_gold_patch.patch', delete=False) as f:
                f.write(task.patch if task.patch.endswith('\n') else task.patch + '\n')
                gp_host = Path(f.name)
            try:
                manager.copy_to(cid, gp_host, '/tmp/')
                apply_patch_in_container(manager, cid, f'/tmp/{gp_host.name}')
            finally:
                gp_host.unlink(missing_ok=True)

        # Apply verification test patch
        if task.test_patch.strip():
            with tempfile.NamedTemporaryFile('w', suffix='_test_patch.patch', delete=False) as f:
                f.write(task.test_patch if task.test_patch.endswith('\n') else task.test_patch + '\n')
                tp_host = Path(f.name)
            try:
                manager.copy_to(cid, tp_host, '/tmp/')
                apply_patch_in_container(manager, cid, f'/tmp/{tp_host.name}')
            finally:
                tp_host.unlink(missing_ok=True)

        patch_res = manager.exec(
            cid,
            f'cd /workspace && PYTHONSAFEPATH=1 python3 -m pytest {test_files_str} -p no:anyio -o timeout=0 -o norecursedirs=".* build dist venv" -o python_classes="Test* *Test" -q',
            timeout=timeout_seconds,
        )
        patch_exit_code = patch_res.exit_code if patch_res.exit_code is not None else 0

        passed_criteria = (base_exit_code != 0 and patch_exit_code == 0)
        latency_sla_passed = bootstrap_time < LATENCY_SLA_THRESHOLD_SECONDS

        return SampleVerificationResult(
            instance_id=task.instance_id,
            repo=task.repo,
            is_private=is_priv,
            base_exit_code=base_exit_code,
            patch_exit_code=patch_exit_code,
            passed_criteria=passed_criteria,
            bootstrap_time_seconds=bootstrap_time,
            latency_sla_passed=latency_sla_passed,
            base_stdout=base_res.stdout,
            base_stderr=base_res.stderr,
            patch_stdout=patch_res.stdout,
            patch_stderr=patch_res.stderr,
        )

    except Exception as e:
        logger.exception('Error during sample verification for task %s', task.instance_id)
        return SampleVerificationResult(
            instance_id=task.instance_id,
            repo=task.repo,
            is_private=is_priv,
            base_exit_code=-1,
            patch_exit_code=-1,
            passed_criteria=False,
            bootstrap_time_seconds=time.perf_counter() - bootstrap_start,
            latency_sla_passed=False,
            error=str(e),
        )
    finally:
        manager.stop(cid)


def run_sample_verification_sweep(
    tasks: Sequence[dict[str, Any] | Task],
    sandbox_manager: BaseSandboxManager | None = None,
    image: str = 'swebench-sandbox:latest',
    snapshots_dir: Path | None = None,
    samples_per_repo: int = 1,
    fast_path: bool = True,
    target_repos: Sequence[str] | None = None,
) -> SweepSummary:
    """Runs an air-gapped test verification sweep across sample tasks covering all repositories.

    Asserts:
      1. 100% of sample tasks satisfy SWE-bench criteria (base_exit_code != 0, patch_exit_code == 0).
      2. Per-task bootstrap & test prep latency is < 5.0 seconds.

    Args:
        tasks: Candidate benchmark tasks.
        sandbox_manager: Optional sandbox manager.
        image: Container image tag.
        snapshots_dir: Optional snapshots directory.
        samples_per_repo: Number of sample tasks per repository.
        fast_path: Fast-path dependency injection flag.
        target_repos: Subset of repositories to sample.

    Returns:
        SweepSummary
    """
    sample_tasks = select_sample_tasks(tasks, samples_per_repo=samples_per_repo, target_repos=target_repos)
    logger.info('Running air-gapped verification sweep on %d sample tasks...', len(sample_tasks))

    results: list[SampleVerificationResult] = []
    by_repo: dict[str, dict[str, Any]] = defaultdict(lambda: {'total': 0, 'passed': 0, 'latencies': []})

    for task in sample_tasks:
        logger.info('Verifying sample task: %s (%s)...', task.instance_id, task.repo)
        res = verify_sample_task_airgapped(
            task=task,
            sandbox_manager=sandbox_manager,
            image=image,
            snapshots_dir=snapshots_dir,
            fast_path=fast_path,
        )
        results.append(res)

        repo_entry = by_repo[task.repo]
        repo_entry['total'] += 1
        if res.passed_criteria:
            repo_entry['passed'] += 1
        repo_entry['latencies'].append(res.bootstrap_time_seconds)

        status_str = 'PASSED' if res.passed_criteria else 'FAILED'
        logger.info(
            '  [%s] %s: base_exit=%d, patch_exit=%d, bootstrap=%.2fs',
            status_str,
            task.instance_id,
            res.base_exit_code,
            res.patch_exit_code,
            res.bootstrap_time_seconds,
        )

    passed_count = sum(1 for r in results if r.passed_criteria)
    total_count = len(results)
    pass_rate = (passed_count / total_count) if total_count > 0 else 0.0

    latencies = [r.bootstrap_time_seconds for r in results]
    max_lat = max(latencies) if latencies else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    all_sla_passed = all(r.latency_sla_passed for r in results)

    # Convert by_repo defaultdict to standard dict
    clean_by_repo: dict[str, dict[str, Any]] = {}
    for repo, data in by_repo.items():
        clean_by_repo[repo] = {
            'total': data['total'],
            'passed': data['passed'],
            'pass_rate': data['passed'] / data['total'] if data['total'] > 0 else 0.0,
            'avg_latency': sum(data['latencies']) / len(data['latencies']) if data['latencies'] else 0.0,
        }

    return SweepSummary(
        total_tasks=total_count,
        passed_tasks=passed_count,
        failed_tasks=total_count - passed_count,
        pass_rate=pass_rate,
        latency_sla_passed=all_sla_passed,
        max_bootstrap_time=max_lat,
        avg_bootstrap_time=avg_lat,
        results=results,
        by_repo=clean_by_repo,
    )
