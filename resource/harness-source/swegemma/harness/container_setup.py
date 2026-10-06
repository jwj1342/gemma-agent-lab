"""Container preparation utilities for SWE-bench sandbox and evaluation environments."""

from __future__ import annotations

import logging
import os
import shlex
import threading
from pathlib import Path

from swegemma.config import EvalConfig
from swegemma.sandbox import BaseSandboxManager

logger = logging.getLogger(__name__)


class ContainerSetupError(RuntimeError):
    """Raised when container setup fails (snapshot, baseline patch, or setup.py)."""


def resolve_sandbox_setup_script(config: EvalConfig | None = None) -> Path | None:
    """Locates the sandbox setup.py script from task configuration or standard paths."""
    candidates: list[Path] = []
    env_sandbox = os.environ.get('KAGGLE_SANDBOX_DIR', '')
    if env_sandbox:
        candidates.append(Path(env_sandbox) / 'setup.py')
        candidates.append(Path(env_sandbox) / 'sandbox' / 'setup.py')
    if config:
        if config.tasks_path:
            candidates.append(config.tasks_path.parent / 'sandbox' / 'setup.py')
            candidates.append(config.tasks_path.parent / 'setup.py')
        if config.snapshots_dir:
            candidates.append(config.snapshots_dir.parent / 'sandbox' / 'setup.py')
            candidates.append(config.snapshots_dir.parent / 'setup.py')
        if config.submission_dir:
            candidates.append(config.submission_dir / 'setup.py')
            candidates.append(config.submission_dir / 'sandbox' / 'setup.py')

    candidates.extend([
        Path(
            '/kaggle/input/datasets/metric/gemma-4-developer-agent-metric-data/sandbox/setup.py'
        ),
        Path('/kaggle/input/gemma-4-developer-agent-metric-data/sandbox/setup.py'),
        Path('/kaggle/input/competition-data/sandbox/setup.py'),
        Path('/kaggle/input/competition-data/rerun/sandbox/setup.py'),
        Path('/kaggle/input/competition-data/secret/sandbox/setup.py'),
        Path('data/competition_data/secret/sandbox/setup.py'),
        Path('competition_data/rerun/sandbox/setup.py'),
        Path('/sandbox/setup.py'),
    ])

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


INSTALL_DEPS_SCRIPT = resolve_sandbox_setup_script()


def resolve_wheels_dir(config: EvalConfig | None = None) -> Path | None:
    """Finds the local wheels directory containing task dependencies."""
    if config and config.wheels_dir and config.wheels_dir.exists():
        return config.wheels_dir
    candidates = [
        config.tasks_path.parent / 'wheels' if (config and config.tasks_path) else None,
        config.tasks_path.parent / 'sandbox' / 'wheels'
        if (config and config.tasks_path)
        else None,
        config.snapshots_dir.parent / 'wheels'
        if (config and config.snapshots_dir)
        else None,
        Path(
            '/kaggle/input/datasets/metric/gemma-4-developer-agent-metric-data/sandbox/wheels'
        ),
        Path('/kaggle/input/gemma-4-developer-agent-metric-data/sandbox/wheels'),
        Path('/kaggle/input/competition-data/wheels'),
        Path('data/competition_data/secret/sandbox/wheels'),
        Path('wheels'),
        Path('/wheels'),
        Path('/tmp/wheels'),
    ]
    for cand in candidates:
        if cand and cand.is_dir() and any(f.endswith('.whl') for f in os.listdir(cand)):
            return cand
    return None


def _is_wheel_compatible_py313(filename: str) -> bool:
    """Checks if a wheel filename is compatible with CPython 3.13 on Linux x86_64 under PEP 427."""
    base = os.path.basename(filename)
    if not base.endswith('.whl'):
        return False
    parts = base[:-4].split('-')
    if len(parts) < 5:
        return False
    py_tag = parts[-3]
    abi_tag = parts[-2]
    if (
        abi_tag in ('cp38', 'cp39', 'cp310', 'cp311', 'cp312')
        and py_tag in ('cp38', 'cp39', 'cp310', 'cp311', 'cp312')
    ):
        return False
    return abi_tag == 'none' or 'abi3' in abi_tag or 'cp313' in py_tag


def _deduplicate_wheels(wheel_files: list[str]) -> list[Path]:
    """Deduplicates compatible wheels to the highest version per normalized distribution name."""
    import re

    from packaging.version import parse as parse_version

    latest: dict[str, tuple[object, Path]] = {}
    for w_str in wheel_files:
        if not _is_wheel_compatible_py313(w_str):
            continue
        p = Path(w_str)
        parts = p.name[:-4].split('-')
        pkg_norm = re.sub(r'[-_.]+', '_', parts[0]).lower()
        try:
            ver: object = parse_version(parts[1])
        except Exception:
            ver = parts[1]
        prev = latest.get(pkg_norm)
        if prev is None:
            latest[pkg_norm] = (ver, p)
        else:
            try:
                if ver > prev[0]:  # type: ignore[operator]
                    latest[pkg_norm] = (ver, p)
            except TypeError:
                if str(ver) > str(prev[0]):
                    latest[pkg_norm] = (ver, p)
    return [item[1] for _, item in sorted(latest.items())]


_LARGE_WHEEL_THRESHOLD_BYTES: int = 20 * 1024 * 1024
_WHEEL_INDEX_CACHE: dict[
    str,
    tuple[list[Path], dict[str, Path], dict[str, set[str]], dict[str, set[str]]],
] = {}


def _inspect_wheels_dir(
    wheels_dir: Path,
) -> tuple[list[Path], dict[str, Path], dict[str, set[str]], dict[str, set[str]]]:
    """Indexes wheels in wheels_dir into base (<16MB) and large (>=16MB) sets with module and dependency graphs."""
    import re
    import zipfile

    key = str(wheels_dir.resolve())
    if key in _WHEEL_INDEX_CACHE:
        return _WHEEL_INDEX_CACHE[key]

    skip_base_pkgs = {
        'pytest',
        'pytest_timeout',
        'pytest_asyncio',
        'pytest_httpbin',
        'pytest_xdist',
        'pluggy',
        'pip',
        'setuptools',
        'wheel',
    }

    all_wheels = [
        str(wheels_dir / f) for f in os.listdir(wheels_dir) if f.endswith('.whl')
    ]
    deduped = _deduplicate_wheels(all_wheels)

    base_wheels: list[Path] = []
    large_wheels: dict[str, Path] = {}
    module_to_pkg: dict[str, set[str]] = {}
    pkg_requires: dict[str, set[str]] = {}

    for w_path in deduped:
        pkg_norm = re.sub(r'[-_.]+', '_', w_path.name.split('-', 1)[0]).lower()
        if pkg_norm in skip_base_pkgs:
            continue
        if w_path.stat().st_size < _LARGE_WHEEL_THRESHOLD_BYTES:
            base_wheels.append(w_path)
        else:
            large_wheels[pkg_norm] = w_path

        module_to_pkg.setdefault(pkg_norm, set()).add(pkg_norm)
        try:
            with zipfile.ZipFile(w_path, 'r') as zf:
                names = zf.namelist()
                for n in names:
                    top = n.split('/', 1)[0]
                    if top.endswith('.dist-info') or top.endswith('.data'):
                        if n.endswith('/top_level.txt'):
                            tl_txt = zf.read(n).decode('utf-8', errors='ignore')
                            for line in tl_txt.splitlines():
                                mod = line.strip().split('/')[0].lower()
                                if mod:
                                    module_to_pkg.setdefault(mod, set()).add(pkg_norm)
                        elif n.endswith('/METADATA'):
                            meta_txt = zf.read(n).decode('utf-8', errors='ignore')
                            for line in meta_txt.splitlines():
                                if line.startswith('Requires-Dist:'):
                                    raw_dep = line.split(':', 1)[1].strip()
                                    if ';' in raw_dep and 'extra ==' in raw_dep:
                                        continue
                                    m_dep = re.match(r'^([a-zA-Z0-9_\-\.]+)', raw_dep)
                                    if m_dep:
                                        dep_norm = re.sub(
                                            r'[-_.]+', '_', m_dep.group(1)
                                        ).lower()
                                        pkg_requires.setdefault(pkg_norm, set()).add(
                                            dep_norm
                                        )
                    else:
                        mod_name = (
                            top[:-3].lower()
                            if top.endswith('.py')
                            else top.split('.')[0].lower()
                        )
                        if mod_name and not mod_name.startswith('_'):
                            module_to_pkg.setdefault(mod_name, set()).add(pkg_norm)
        except Exception:
            pass

    res = (base_wheels, large_wheels, module_to_pkg, pkg_requires)
    _WHEEL_INDEX_CACHE[key] = res
    return res


_WHEELS_TAR_LOCK = threading.Lock()


def _build_unpacked_wheels_tar(
    selected_wheels: list[Path], cache_name: str, marker_name: str
) -> Path:
    """Unpacks selected_wheels into a cached uncompressed tar archive with a marker file."""
    import shutil
    import subprocess
    import tempfile
    import uuid
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    cache_dir = Path(tempfile.gettempdir()) / 'swegemma_sp_cache_v8'
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / f'{cache_name}.tar'
    if tar_path.exists() and tar_path.stat().st_size > 0:
        return tar_path

    with _WHEELS_TAR_LOCK:
        if tar_path.exists() and tar_path.stat().st_size > 0:
            return tar_path

        uid = uuid.uuid4().hex[:8]
        staging_dir = cache_dir / f'staging_{cache_name}_{os.getpid()}_{uid}'
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        def _extract_one(w_path: Path) -> None:
            try:
                with zipfile.ZipFile(w_path, 'r') as zf:
                    members = [
                        m
                        for m in zf.infolist()
                        if not m.filename.endswith(('.pyc', '/RECORD'))
                    ]
                    zf.extractall(staging_dir, members=members)
            except Exception as e:
                logger.debug('Skipping wheel %s during staging: %s', w_path.name, e)

        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(_extract_one, selected_wheels))

            for data_dir in list(staging_dir.glob('*.data')):
                for sub_name in ('purelib', 'platlib'):
                    sub_dir = data_dir / sub_name
                    if sub_dir.is_dir():
                        for item in sub_dir.iterdir():
                            dest = staging_dir / item.name
                            if not dest.exists():
                                shutil.move(str(item), str(dest))
                shutil.rmtree(data_dir, ignore_errors=True)

            (staging_dir / marker_name).write_text('ok\n', encoding='utf-8')

            tmp_tar = cache_dir / f'{cache_name}_{os.getpid()}_{uid}.tar.tmp'
            subprocess.run(
                ['tar', '-cf', str(tmp_tar), '-C', str(staging_dir), '.'],
                check=True,
            )
            tmp_tar.replace(tar_path)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    return tar_path


def _extract_tar_into_site_packages(
    docker: BaseSandboxManager, container_id: str, tar_path: Path
) -> None:
    """Streams or copies a tar archive and extracts it into /usr/local/lib/python3.13/site-packages."""
    sp_dir = '/usr/local/lib/python3.13/site-packages'
    stream_fn = getattr(docker, '_stream_tar_via_exec_socket', None)
    if callable(stream_fn):
        with open(tar_path, 'rb') as f:
            if stream_fn(container_id, f, sp_dir):
                return
    staged = f'/var/tmp/{tar_path.name}'
    docker.copy_to(container_id, tar_path, staged)
    docker.exec(
        container_id,
        f'tar --no-same-owner -xf {shlex.quote(staged)} -C {sp_dir} && rm -f {shlex.quote(staged)}',
    )


def _ensure_container_site_packages(
    docker: BaseSandboxManager,
    container_id: str,
    repo: str,
    config: EvalConfig | None = None,
) -> None:
    """Dynamically injects cached unpacked wheels into container site-packages based on workspace imports."""
    if not hasattr(docker, 'extract_archive_to_container'):
        return

    wheels_dir = resolve_wheels_dir(config)
    if not wheels_dir:
        return

    base_wheels, large_wheels, module_to_pkg, pkg_requires = _inspect_wheels_dir(
        wheels_dir
    )
    sp_dir = '/usr/local/lib/python3.13/site-packages'

    # 1. Inject base wheels (<16MB) once per container
    base_marker = f'{sp_dir}/.swegemma_base_injected'
    if docker.exec(container_id, f'test -f {base_marker}').exit_code != 0:
        base_tar = _build_unpacked_wheels_tar(
            base_wheels, 'sp_base', '.swegemma_base_injected'
        )
        _extract_tar_into_site_packages(docker, container_id, base_tar)

    if not large_wheels:
        return

    # 2. Discover top-level imported modules in /workspace and resolve transitive large wheels (>=16MB)
    scan_cmd = (
        'python3 -S -c "'
        'import pathlib, re; mods = set(); '
        '[mods.update(m.group(1).lower() for m in re.finditer(r\'^\\s*(?:import|from)\\s+([a-zA-Z0-9_]+)\', p.read_text(errors=\'ignore\'), re.M)) '
        'for p in pathlib.Path(\'/workspace\').rglob(\'*.py\') if \'.git\' not in p.parts]; '
        'print(\' \'.join(sorted(mods)))" 2>/dev/null'
    )
    scan_res = docker.exec(container_id, scan_cmd)
    imported_mods = set((scan_res.stdout or '').split())
    if repo:
        repo_short = repo.rsplit('/', 1)[-1].lower()
        imported_mods.add(repo_short)

    needed_pkgs: set[str] = set()
    queue: list[str] = []
    for mod in imported_mods:
        for pkg_norm in module_to_pkg.get(mod, ()):
            if pkg_norm not in needed_pkgs:
                needed_pkgs.add(pkg_norm)
                queue.append(pkg_norm)

    while queue:
        curr = queue.pop()
        for dep in pkg_requires.get(curr, ()):
            if dep not in needed_pkgs:
                needed_pkgs.add(dep)
                queue.append(dep)
            for p_cand in large_wheels:
                if p_cand.startswith(curr + '_') and p_cand not in needed_pkgs:
                    needed_pkgs.add(p_cand)
                    queue.append(p_cand)

    needed_large = sorted(p for p in needed_pkgs if p in large_wheels)
    for pkg_norm in needed_large:
        marker = f'{sp_dir}/.swegemma_whl_{pkg_norm}'
        if docker.exec(container_id, f'test -f {marker}').exit_code != 0:
            w_tar = _build_unpacked_wheels_tar(
                [large_wheels[pkg_norm]],
                f'sp_whl_{pkg_norm}',
                f'.swegemma_whl_{pkg_norm}',
            )
            _extract_tar_into_site_packages(docker, container_id, w_tar)


def setup_container_wheels(
    docker: BaseSandboxManager, container_id: str, config: EvalConfig
) -> None:
    """Copies task wheels into the sandbox container if available on host and not already present."""
    check_res = docker.exec(container_id, 'ls /wheels/*.whl 2>/dev/null | wc -l')
    if check_res.exit_code == 0 and check_res.stdout.strip() not in ('0', ''):
        logger.debug(
            'Container %s already has wheels populated in /wheels, skipping copy.',
            container_id,
        )
        return

    # For real Docker containers with extract_archive_to_container, install_test_dependencies
    # streams unpacked wheels directly into site-packages, avoiding a 1.7 GB /wheels tar copy per task.
    if hasattr(docker, 'extract_archive_to_container'):
        return

    wheels_path = resolve_wheels_dir(config)
    if wheels_path:
        docker.exec(container_id, 'mkdir -p /wheels')
        docker.copy_to(container_id, wheels_path, '/wheels')


def extract_snapshot(
    docker: BaseSandboxManager,
    container_id: str,
    snapshot_path: Path,
    *,
    patch_path: Path | None = None,
    base_snapshot_path: Path | None = None,
) -> None:
    """Copies a snapshot tarball (or base snapshot + patch) into the container and extracts it into /workspace."""
    tar_to_copy = (
        base_snapshot_path if base_snapshot_path is not None else snapshot_path
    )
    extracted = False
    extract_fn = getattr(docker, 'extract_archive_to_container', None)
    if callable(extract_fn):
        extracted = bool(
            extract_fn(
                container_id, tar_to_copy, '/workspace'
            )
        )
        if not extracted:
            logger.warning(
                'Direct archive extraction of %s into /workspace failed on container %s; falling back to copy + tar',
                tar_to_copy.name,
                container_id,
            )
    if not extracted:
        docker.copy_to(container_id, tar_to_copy, '/tmp')
        ext_res = docker.exec(
            container_id,
            f'(tar --no-same-owner -I pigz -xf /tmp/{tar_to_copy.name} -C /workspace 2>/dev/null || tar --no-same-owner -xzf /tmp/{tar_to_copy.name} -C /workspace) && rm -f /tmp/{tar_to_copy.name}',
        )
        if ext_res.exit_code != 0:
            msg = f'Failed to extract snapshot {tar_to_copy.name} on container {container_id}: {ext_res.stderr or ext_res.stdout}'
            logger.error(msg)
            raise ContainerSetupError(msg)

    # Apply incremental patch if provided
    if patch_path is not None and patch_path.exists():
        docker.copy_to(container_id, patch_path, '/tmp')
        exit_code, stdout, stderr = apply_patch_in_container(
            docker,
            container_id,
            f'/tmp/{patch_path.name}',
            workspace_dir='/workspace',
        )
        docker.exec(container_id, f'rm -f /tmp/{shlex.quote(patch_path.name)}')
        if exit_code != 0:
            msg = f'Failed to apply incremental patch {patch_path.name} on container {container_id}: {stderr or stdout}'
            logger.error(msg)
            raise ContainerSetupError(msg)

    # Remove any circular symlinks from snapshot (pointing to /workspace or ancestors)
    # that cause setuptools find_packages() and pytest directory walkers to loop infinitely
    docker.exec(
        container_id,
        "python3 -S -c \"import pathlib; ws=pathlib.Path('/workspace').resolve(); "
        "[p.unlink() for p in list(ws.rglob('*')) if p.is_symlink() and (p.resolve() == ws or ws.is_relative_to(p.resolve()))]\" 2>/dev/null || true",
    )


def extract_deduplicated_snapshot(
    docker: BaseSandboxManager,
    container_id: str,
    base_snapshot_path: Path,
    patch_path: Path | None = None,
    fallback_path: Path | None = None,
) -> None:
    """Extracts a deduplicated snapshot: uses fallback archive if present, else base snapshot + incremental patch."""
    if fallback_path and fallback_path.exists():
        extract_snapshot(docker, container_id, fallback_path)
        return

    extract_snapshot(
        docker,
        container_id,
        base_snapshot_path,
        patch_path=patch_path,
        base_snapshot_path=base_snapshot_path,
    )


def setup_git_exclude(docker: BaseSandboxManager, container_id: str) -> None:
    """Configures git exclude rules in the container workspace."""
    docker.exec(
        container_id,
        'mkdir -p /workspace/.git/info && printf "__pycache__/\\n*.pyc\\n.pytest_cache/\\n*.egg-info/\\nbuild/\\ndist/\\n.coverage\\n.adk_exec_*.py\\n" >> /workspace/.git/info/exclude',
    )


def install_editable_package(docker: BaseSandboxManager, container_id: str) -> None:
    """Installs the workspace package in editable mode using pre-cached wheels.

    Uses --no-deps to prevent pip from resolving and potentially installing incompatible
    wheel versions over the container's pre-configured environment.
    """
    if type(docker).__name__ in {'SubprocessManager', 'SubprocessSandbox'}:
        return
    docker.exec(
        container_id,
        'pip install --no-index --find-links=/wheels --no-build-isolation --no-deps -e /workspace 2>/dev/null || true',
    )


def install_test_dependencies(
    docker: BaseSandboxManager,
    container_id: str,
    repo: str = '',
    *,
    fast_path: bool = True,
    config: EvalConfig | None = None,
) -> None:
    """Installs test dependencies discovered from pyproject.toml and requirements files or via fast-path."""
    if type(docker).__name__ in {'SubprocessManager', 'SubprocessSandbox'}:
        return
    _ensure_container_site_packages(docker, container_id, repo, config)

    setup_script = resolve_sandbox_setup_script(config)
    if setup_script and setup_script.exists():
        docker.copy_to(container_id, setup_script, '/tmp/setup.py')
        script_in_container = '/tmp/setup.py'
    else:
        check_res = docker.exec(container_id, 'test -f /sandbox/setup.py')
        if check_res.exit_code == 0:
            script_in_container = '/sandbox/setup.py'
        else:
            msg = f'setup.py not found on host or container for {container_id}'
            is_subprocess = (
                getattr(config, 'sandbox', 'docker') == 'subprocess'
                or type(docker).__name__ in {'SubprocessManager', 'SubprocessSandbox'}
            )
            if config is not None and not is_subprocess:
                logger.error(msg)
                raise ContainerSetupError(msg)
            logger.warning(msg)
            return

    fast_flag = '--fast-path' if fast_path else ''
    cmd = f'python3 {script_in_container} {fast_flag} {repo}'.strip()
    try:
        res = docker.exec(container_id, cmd)
        logger.debug('Install test deps stdout: %s', res.stdout)
        if res.exit_code != 0:
            details = f'stdout:\n{res.stdout}\nstderr:\n{res.stderr}'.strip()
            if 'No such file or directory' in (res.stderr or ''):
                msg = f'Failed to execute {script_in_container} on container {container_id}: {res.stderr}'
                logger.error(msg)
                raise ContainerSetupError(msg)
            logger.warning(
                'Failed to install test dependencies for %s (exit code %d):\n%s',
                repo,
                res.exit_code,
                details,
            )
    finally:
        if script_in_container == '/tmp/setup.py':
            docker.exec(container_id, 'rm -f /tmp/setup.py')


def setup_baseline_commit(
    docker: BaseSandboxManager, container_id: str, commit_msg: str = 'baseline'
) -> None:
    """Creates a clean git baseline commit for diff calculations."""
    docker.exec(
        container_id,
        f'cd /workspace && git add -A && git commit -m "{commit_msg}" --allow-empty -q && git tag -f _swegemma_baseline',
    )


def apply_patch_in_container(
    docker: BaseSandboxManager,
    container_id: str,
    patch_path_in_container: str,
    *,
    workspace_dir: str = '/workspace',
) -> tuple[int, str, str]:
    """Applies a patch inside the container using a robust multi-strategy pipeline.

    Strategies executed in order:
    1. Raw git apply with -p1 strategies ONLY (-3, whitespace tolerance, recount).
    2. Symlink path normalization with -p1 strategies ONLY (resolves symlink directory leading path
       components via os.path.realpath against workspace_dir and retries git apply).
    3. Prefixless git apply with -p0 strategies on raw and normalized patches (-p0, -3, whitespace tolerance, recount).
    4. Non-interactive fallback with --dry-run validation (patch -p1 then -p0 --batch --forward, including -l for whitespace).

    Returns:
        tuple[int, str, str]: (exit_code, stdout, stderr)
    """
    script = r'''
import os, sys, subprocess, re
from pathlib import Path

if len(sys.argv) < 3:
    sys.stderr.write("Usage: apply_patch.py <workspace_dir> <patch_path>\n")
    sys.exit(1)

workspace = Path(sys.argv[1]).resolve()
p_path = Path(sys.argv[2]).resolve()
if not p_path.exists():
    sys.stderr.write(f"Patch file not found: {p_path}\n")
    sys.exit(1)
if p_path.stat().st_size == 0:
    sys.exit(0)

def check_safe_rel_path(raw_rel: str) -> None:
    s = raw_rel.strip().strip('"')
    if not s or s == "/dev/null":
        return
    if s.startswith(("a/", "b/")):
        s = s[2:].strip()
    if not s or s == "/dev/null":
        return
    if s.startswith("/") or ".." in Path(s).parts:
        sys.stderr.write(f"Unsafe path traversal in patch: {raw_rel}\n")
        sys.exit(1)
    resolved_target = (workspace / s).resolve()
    if resolved_target != workspace and not resolved_target.is_relative_to(workspace):
        sys.stderr.write(f"Unsafe symlink/path escape in patch: {raw_rel}\n")
        sys.exit(1)

raw_patch_text = p_path.read_text(errors="replace")
for raw_line in raw_patch_text.splitlines():
    if raw_line.startswith("diff --git "):
        m = re.match(r'^diff --git\s+"?a/(.+?)"?\s+"?b/(.+?)"?$', raw_line)
        if m:
            check_safe_rel_path(m.group(1))
            check_safe_rel_path(m.group(2))
    elif raw_line.startswith(("--- ", "+++ ")):
        hdr = raw_line[4:].split("\t", 1)[0].strip()
        check_safe_rel_path(hdr)
    elif raw_line.startswith(("rename from ", "rename to ")):
        hdr = raw_line.split(" ", 2)[2].strip()
        check_safe_rel_path(hdr)
    elif raw_line.startswith("# empty_dir: "):
        hdr = raw_line.split(": ", 1)[-1].strip()
        check_safe_rel_path(hdr)

try:
    with open(p_path, 'rb') as pf:
        for line in pf:
            if line.startswith(b'# empty_dir: '):
                d = line.decode('utf-8', errors='ignore').strip().split(': ', 1)[-1]
                if d:
                    check_safe_rel_path(d)
                    (workspace / d).mkdir(parents=True, exist_ok=True)
            elif line.startswith(b'diff --git'):
                break
except Exception:
    pass

def normalize_patch(p: Path) -> Path:
    text = p.read_text(errors="replace")
    lines, mod = [], False

    def quote_path(prefix: str, path_str: str) -> str:
        if " " in path_str:
            return f'"{prefix}/{path_str}"'
        return f"{prefix}/{path_str}"

    for line in text.splitlines(keepends=True):
        nl = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
        content = line[:-len(nl)] if nl else line

        if content.startswith("diff --git "):
            matched = False
            # Case 1: Quoted diff --git "a/..." "b/..."
            m = re.match(r'^diff --git\s+"a/(.+?)"\s+"b/(.+?)"(.*)$', content)
            if m:
                p_a, p_b, rest = m.groups()
                matched = True
            else:
                # Case 2: Unquoted identical paths (with or without spaces)
                m = re.match(r'^diff --git\s+a/(.+)\s+b/\1(.*)$', content)
                if m:
                    p_a, p_b, rest = m.group(1), m.group(1), m.group(2)
                    matched = True
                else:
                    # Case 3: Standard unquoted diff --git a/... b/...
                    m = re.match(r'^diff --git\s+a/(\S+)\s+b/(\S+)(.*)$', content)
                    if m:
                        p_a, p_b, rest = m.groups()
                        matched = True
            if matched:
                try:
                    r_a = os.path.relpath(os.path.realpath(workspace / p_a), workspace)
                    r_b = os.path.relpath(os.path.realpath(workspace / p_b), workspace)
                    check_safe_rel_path(r_a)
                    check_safe_rel_path(r_b)
                    if r_a != p_a or r_b != p_b:
                        mod = True
                    qa = quote_path("a", r_a)
                    qb = quote_path("b", r_b)
                    lines.append(f"diff --git {qa} {qb}{rest}{nl}")
                    continue
                except Exception:
                    pass

        elif content.startswith("--- a/") or content.startswith('--- "a/'):
            try:
                if content.startswith('--- "a/'):
                    raw_after = content[7:]
                    p_sub, _, rest = raw_after.partition('"')
                    quoted = True
                else:
                    raw_after = content[6:]
                    p_sub, tab, timestamp = raw_after.partition("\t")
                    rest = f"\t{timestamp}" if tab else ""
                    quoted = False
                r_sub = os.path.relpath(os.path.realpath(workspace / p_sub), workspace)
                check_safe_rel_path(r_sub)
                if r_sub != p_sub:
                    mod = True
                q_sub = quote_path("a", r_sub) if (quoted or " " in r_sub) else f"a/{r_sub}"
                lines.append(f"--- {q_sub}{rest}{nl}")
                continue
            except Exception:
                pass

        elif content.startswith("+++ b/") or content.startswith('+++ "b/'):
            try:
                if content.startswith('+++ "b/'):
                    raw_after = content[7:]
                    p_sub, _, rest = raw_after.partition('"')
                    quoted = True
                else:
                    raw_after = content[6:]
                    p_sub, tab, timestamp = raw_after.partition("\t")
                    rest = f"\t{timestamp}" if tab else ""
                    quoted = False
                r_sub = os.path.relpath(os.path.realpath(workspace / p_sub), workspace)
                check_safe_rel_path(r_sub)
                if r_sub != p_sub:
                    mod = True
                q_sub = quote_path("b", r_sub) if (quoted or " " in r_sub) else f"b/{r_sub}"
                lines.append(f"+++ {q_sub}{rest}{nl}")
                continue
            except Exception:
                pass

        lines.append(line)

    if not mod:
        return p
    norm_p = p.with_suffix(".norm.patch")
    norm_p.write_text("".join(lines))
    return norm_p

errors = []
def try_cmd(c: list[str]) -> bool:
    try:
        r = subprocess.run(c, cwd=workspace, capture_output=True, text=True)
        if r.returncode == 0:
            sys.exit(0)
        errors.append(f"{' '.join(c)} failed ({r.returncode}):\n{r.stderr}\n{r.stdout}")
    except FileNotFoundError as fnf:
        errors.append(f"{c[0]} not found: {fnf}")
    return False

# Pass 1: Raw patch with -p1 strategies ONLY
for c in [
    ["git", "apply", str(p_path)],
    ["git", "apply", "-3", str(p_path)],
    ["git", "apply", "--ignore-space-change", "--ignore-whitespace", str(p_path)],
    ["git", "apply", "--recount", str(p_path)],
]:
    try_cmd(c)

# Pass 2: Normalized patch with -p1 strategies ONLY
norm_p = normalize_patch(p_path)
if norm_p != p_path:
    for c in [
        ["git", "apply", str(norm_p)],
        ["git", "apply", "-3", str(norm_p)],
        ["git", "apply", "--ignore-space-change", "--ignore-whitespace", str(norm_p)],
        ["git", "apply", "--recount", str(norm_p)],
    ]:
        try_cmd(c)

# Pass 3: -p0 strategies on raw and normalized patches ONLY if both -p1 passes fail (for prefixless patches)
p0_targets = [p_path, norm_p] if norm_p != p_path else [p_path]
for target in p0_targets:
    for c in [
        ["git", "apply", "-p0", str(target)],
        ["git", "apply", "-p0", "-3", str(target)],
        ["git", "apply", "-p0", "--ignore-space-change", "--ignore-whitespace", str(target)],
        ["git", "apply", "-p0", "--recount", str(target)],
    ]:
        try_cmd(c)

# Pass 4: Non-interactive patch fallback with --dry-run validation (try -p1 first, then -p0)
patch_targets = [norm_p, p_path] if norm_p != p_path else [p_path]
for target in patch_targets:
    for c in [
        ["patch", "-p1", "--batch", "--forward", "-i", str(target)],
        ["patch", "-p1", "-l", "--batch", "--forward", "-i", str(target)],
    ]:
        dry_cmd = c + ["--dry-run"]
        try:
            r_dry = subprocess.run(dry_cmd, cwd=workspace, capture_output=True, text=True)
            if r_dry.returncode == 0:
                r_apply = subprocess.run(c, cwd=workspace, capture_output=True, text=True)
                if r_apply.returncode == 0:
                    sys.exit(0)
                errors.append(f"{' '.join(c)} apply failed ({r_apply.returncode}):\n{r_apply.stderr}\n{r_apply.stdout}")
            else:
                errors.append(f"{' '.join(dry_cmd)} dry-run failed ({r_dry.returncode}):\n{r_dry.stderr}\n{r_dry.stdout}")
        except FileNotFoundError as fnf:
            errors.append(f"{c[0]} not found: {fnf}")

for target in patch_targets:
    for c in [
        ["patch", "-p0", "--batch", "--forward", "-i", str(target)],
        ["patch", "-p0", "-l", "--batch", "--forward", "-i", str(target)],
    ]:
        dry_cmd = c + ["--dry-run"]
        try:
            r_dry = subprocess.run(dry_cmd, cwd=workspace, capture_output=True, text=True)
            if r_dry.returncode == 0:
                r_apply = subprocess.run(c, cwd=workspace, capture_output=True, text=True)
                if r_apply.returncode == 0:
                    sys.exit(0)
                errors.append(f"{' '.join(c)} apply failed ({r_apply.returncode}):\n{r_apply.stderr}\n{r_apply.stdout}")
            else:
                errors.append(f"{' '.join(dry_cmd)} dry-run failed ({r_dry.returncode}):\n{r_dry.stderr}\n{r_dry.stdout}")
        except FileNotFoundError as fnf:
            errors.append(f"{c[0]} not found: {fnf}")

sys.stderr.write("\n".join(errors) + "\n")
sys.exit(1)
'''
    cmd = (
        f'python3 -S -c {shlex.quote(script)} {shlex.quote(workspace_dir)} '
        f'{shlex.quote(patch_path_in_container)}'
    )
    try:
        res = docker.exec(container_id, cmd)
        exit_code = res.exit_code if res.exit_code is not None else -1
        return exit_code, res.stdout, res.stderr
    finally:
        if patch_path_in_container.startswith('/tmp/'):
            docker.exec(container_id, f'rm -f {shlex.quote(patch_path_in_container)}*')


def setup_synthetic_hardware(docker: BaseSandboxManager, container_id: str) -> None:
    """Delegate synthetic environment preparation to the bundle's sandbox/setup.py script."""
    return


def setup_workspace_test_config(
    docker: BaseSandboxManager,
    container_id: str,
    repo: str = '',
    *,
    workspace_dir: str = '/workspace',
    overwrite: bool = False,
) -> None:
    """Configures baseline pytest.ini and conftest.py in the container workspace if not already configured by sandbox/setup.py."""
    script = r'''
import sys
from pathlib import Path

ws = Path(sys.argv[1]).resolve()
repo = sys.argv[2] if len(sys.argv) > 2 else ""
overwrite = sys.argv[3] == "1" if len(sys.argv) > 3 else False

# 1. Configure pytest.ini if not already configured by sandbox/setup.py
ini_path = ws / "pytest.ini"
existing_ini = ""
if ini_path.exists():
    try:
        existing_ini = ini_path.read_text(errors="replace")
    except Exception:
        existing_ini = ""

if overwrite or ("# Hermetic test discovery" not in existing_ini and "[pytest]" not in existing_ini):
    content = """[pytest]
addopts = -p no:anyio
norecursedirs = .* build dist venv
python_classes = Test* *Test
python_files = test_*.py *_test.py
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
"""
    try:
        ini_path.write_text(content)
    except Exception as e:
        sys.stderr.write(f"Warning: could not write {ini_path}: {e}\n")

# 2. Configure conftest.py if not already configured by sandbox/setup.py
conftest_path = ws / "conftest.py"
existing = ""
if conftest_path.exists():
    try:
        existing = conftest_path.read_text(errors="replace")
    except Exception:
        existing = ""

hook_header = "# Hermetic test discovery hook for SWE-gemma\n"
if overwrite or (
    "# Hermetic test discovery hook for SWE-gemma" not in existing
    and "# Standard test discovery hook for SWE-gemma" not in existing
):
    clean_existing = existing.replace(hook_header, "").strip()
    new_conftest = hook_header + ("\n" + clean_existing if clean_existing else "") + "\n"
    try:
        conftest_path.write_text(new_conftest)
    except Exception as e:
        sys.stderr.write(f"Warning: could not write {conftest_path}: {e}\n")
'''
    overwrite_flag = '1' if overwrite else '0'
    cmd = (
        f'python3 -S -c {shlex.quote(script)} {shlex.quote(workspace_dir)}'
        f' {shlex.quote(repo)} {overwrite_flag}'
    )
    res = docker.exec(container_id, cmd)
    if res.exit_code != 0:
        logger.warning(
            'Failed to setup workspace test config for %s (exit code %d): %s',
            repo,
            res.exit_code,
            res.stderr or res.stdout,
        )


def setup_container_workspace(
    docker: BaseSandboxManager,
    container_id: str,
    repo: str = '',
    *,
    fast_path: bool = True,
    commit_msg: str = 'eval_baseline',
    workspace_dir: str = '/workspace',
    config: EvalConfig | None = None,
) -> None:
    """Sets up the complete container environment for evaluation or agent discovery.

    Executes git exclude configuration, editable package installation,
    dependency/setup.py execution, workspace test discovery configuration, and baseline git commit.
    """
    setup_synthetic_hardware(docker, container_id)
    setup_git_exclude(docker, container_id)
    install_editable_package(docker, container_id)
    install_test_dependencies(
        docker, container_id, repo=repo, fast_path=fast_path, config=config
    )
    setup_workspace_test_config(
        docker, container_id, repo=repo, workspace_dir=workspace_dir
    )
    setup_baseline_commit(docker, container_id, commit_msg=commit_msg)


def disable_test_runners_in_sandbox(
    docker: BaseSandboxManager,
    container_id: str,
) -> None:
    """Masks pytest and suppresses unconstrained test discovery in the agent sandbox container.

    This prevents agents from running unconstrained test discovery loops or full-repo
    test suites that derail them into fixing pre-existing repository or environment test failures.
    Preserves unittest, unittest.mock, and TestCase so standard library dependencies
    load cleanly.
    Container B (evaluation) does not run this, preserving test frameworks for scoring.
    """
    from swegemma.sandbox.subprocess import SubprocessManager
    if isinstance(docker, SubprocessManager):
        logger.debug('Skipping disable_test_runners_in_sandbox for SubprocessManager to protect host environment.')
        return

    script = r'''
import os, sys

pytest_msg = "pytest is disabled in this environment to prevent slow full-repo test runs and pre-existing environment failures. Please verify your implementation directly using python3 -c or targeted test assertions.\n"

shim = f"""#!/bin/sh
echo "{pytest_msg.strip()}" >&2
exit 1
"""
for bin_name in ["/usr/local/bin/pytest", "/usr/local/bin/py.test"]:
    try:
        with open(bin_name, "w") as f:
            f.write(shim)
        os.chmod(bin_name, 0o755)
    except Exception as e:
        sys.stderr.write(f"Warning: could not write {bin_name}: {e}\n")

try:
    import pytest
    p = os.path.dirname(pytest.__file__)
    main_py = os.path.join(p, "__main__.py")
    with open(main_py, "w") as f:
        f.write(f'import sys\nsys.stderr.write("{pytest_msg}")\nsys.exit(1)\n')
except Exception:
    pass

unittest_discovery_msg = "unittest test discovery is disabled in this environment to prevent full-repo test discovery loops and pre-existing test breakages. Please verify your implementation by running targeted test files or methods directly (e.g. python3 path/to/test.py).\n"
for bin_name in ["/usr/local/bin/unittest"]:
    if os.path.exists(bin_name):
        try:
            with open(bin_name, "w") as f:
                f.write(f'#!/bin/sh\necho "{unittest_discovery_msg.strip()}" >&2\nexit 1\n')
            os.chmod(bin_name, 0o755)
        except Exception:
            pass

try:
    import unittest
    p = os.path.dirname(unittest.__file__)
    main_py = os.path.join(p, "__main__.py")
    with open(main_py, "w") as f:
        f.write(f"""import sys
if len(sys.argv) <= 1 or sys.argv[1] in ("discover", "-h", "--help"):
    sys.stderr.write("{unittest_discovery_msg}")
    sys.exit(1)
from unittest.main import main
main(module=None)
""")
except Exception as e:
    sys.stderr.write(f"Warning: could not shim unittest discovery: {e}\n")
'''
    cmd = f'python3 -S -c {shlex.quote(script)}'
    res = docker.exec(container_id, cmd)
    if res.exit_code != 0:
        logger.warning(
            'Failed to disable test runners in sandbox (exit code %d): %s',
            res.exit_code,
            res.stderr or res.stdout,
        )


# Backward-compatible alias
disable_pytest_in_sandbox = disable_test_runners_in_sandbox


