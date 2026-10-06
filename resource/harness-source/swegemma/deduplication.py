"""Snapshot deduplication tooling, binary patch generation, and reconstruction verification."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from swegemma.models.task import Task

logger = logging.getLogger(__name__)


def clean_repo_name(repo: str) -> str:
    """Sanitizes repository slug into a valid filesystem identifier.

    Replaces slashes, backslashes, and hyphens with underscores.

    Examples:
        'pallets/flask' -> 'pallets_flask'
        'fastapi/fastapi' -> 'fastapi_fastapi'
        'Textualize/rich' -> 'Textualize_rich'
        'psf/requests-oauthlib' -> 'psf_requests_oauthlib'
    """
    clean = re.sub(r'[/\\-]+', '_', repo.strip())
    return clean.strip('_')


def get_base_snapshot_filename(repo: str, suffix: str = '.tgz') -> str:
    """Generates standard base snapshot filename adhering to base_{repo_clean}.tgz convention."""
    if not suffix.startswith('.'):
        suffix = f'.{suffix}'
    clean = clean_repo_name(repo)
    return f'base_{clean}{suffix}'


def get_task_cl(task: dict[str, Any] | Task) -> int | None:
    """Extracts integer CL or numerical issue/PR ID from a task."""
    if isinstance(task, Task):
        iid = task.instance_id
        cl_val = getattr(task, 'cl', None)
    elif isinstance(task, dict):
        iid = str(task.get('instance_id', ''))
        cl_val = task.get('cl')
    else:
        return None

    if cl_val is not None:
        try:
            return int(cl_val)
        except (ValueError, TypeError):
            pass

    if '_' in iid:
        last_part = iid.rsplit('_', 1)[-1]
        if last_part.isdigit():
            return int(last_part)
    return None


def identify_base_snapshot(
    tasks: Sequence[dict[str, Any] | Task],
    repo: str | None = None,
) -> dict[str, Any] | Task:
    """Identifies the single earliest base snapshot task for a repository.

    Selection algorithm:
    1. Filter tasks for repo (if specified).
    2. Primary sort key: earliest CL (integer) if present.
    3. Secondary sort key: created_at timestamp string if present.
    4. Deterministic tiebreaker: instance_id alphabetically.

    Args:
        tasks: Sequence of task dictionaries or Task model instances.
        repo: Repository slug to filter by (optional).

    Returns:
        The selected base task object.

    Raises:
        ValueError: If no tasks match the repository.
    """
    if repo is not None:
        matched_tasks = [
            t
            for t in tasks
            if (t.repo if isinstance(t, Task) else t.get('repo')) == repo
        ]
    else:
        matched_tasks = list(tasks)

    if not matched_tasks:
        raise ValueError(f"No tasks found for repository '{repo}'")

    def sort_key(t: dict[str, Any] | Task) -> tuple[int, str, str]:
        cl = get_task_cl(t)
        cl_val = cl if cl is not None else 999_999_999_999
        if isinstance(t, Task):
            created_at = t.created_at or ''
            iid = t.instance_id
        else:
            created_at = str(t.get('created_at', ''))
            iid = str(t.get('instance_id', ''))
        return (cl_val, created_at, iid)

    return min(matched_tasks, key=sort_key)


def identify_base_snapshots(
    tasks: Sequence[dict[str, Any] | Task],
) -> dict[str, dict[str, Any] | Task]:
    """Identifies base snapshots for all unique repositories in the task sequence.

    Args:
        tasks: Sequence of task dicts or Task instances.

    Returns:
        Dict mapping repo name -> base task object.
    """
    repo_to_tasks: dict[str, list[dict[str, Any] | Task]] = defaultdict(list)
    for t in tasks:
        repo = t.repo if isinstance(t, Task) else str(t.get('repo', ''))
        if repo:
            repo_to_tasks[repo].append(t)

    base_snapshots: dict[str, dict[str, Any] | Task] = {}
    for repo, r_tasks in sorted(repo_to_tasks.items()):
        base_snapshots[repo] = identify_base_snapshot(r_tasks, repo)

    return base_snapshots


def clean_circular_symlinks(directory: Path | str) -> list[Path]:
    """Detects and removes circular, broken, or escaping symlinks in directory.

    Prevents infinite recursion during filesystem traversal in tools like pytest,
    find_packages(), or setuptools, and prevents external symlink escapes.

    Args:
        directory: Root directory to inspect.

    Returns:
        List of unlinked symlink paths.
    """
    root_path = Path(directory).resolve()
    unlinked: list[Path] = []

    if not root_path.exists():
        return unlinked

    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        current_dir = Path(dirpath)
        for d in list(dirnames):
            p = current_dir / d
            if p.is_symlink():
                try:
                    target = p.resolve(strict=True)
                    if (
                        target in (root_path, current_dir)
                        or root_path.is_relative_to(target)
                        or current_dir.is_relative_to(target)
                        or not target.is_relative_to(root_path)
                    ):
                        p.unlink()
                        dirnames.remove(d)
                        unlinked.append(p)
                except Exception:
                    p.unlink()
                    dirnames.remove(d)
                    unlinked.append(p)

        for f in filenames:
            p = current_dir / f
            if p.is_symlink():
                try:
                    target = p.resolve(strict=True)
                    if (
                        target == root_path
                        or root_path.is_relative_to(target)
                        or not target.is_relative_to(root_path)
                    ):
                        p.unlink()
                        unlinked.append(p)
                except Exception:
                    p.unlink()
                    unlinked.append(p)

    return unlinked


def _validate_tar_members(arch: Path, dest: Path) -> None:
    """Validates that all tar archive members and link targets remain within dest."""
    dest_resolved = dest.resolve()
    with tarfile.open(arch, 'r:*') as tar:
        for member in tar.getmembers():
            if member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError(
                    f'Malicious tar member escaping destination: {member.name}'
                )
            member_path = (dest_resolved / member.name).resolve()
            if not member_path.is_relative_to(dest_resolved):
                raise ValueError(
                    f'Malicious tar member escaping destination: {member.name}'
                )
            if member.issym() or member.islnk():
                linkname = member.linkname
                if os.path.isabs(linkname):
                    raise ValueError(
                        f'Malicious tar member escaping destination: {member.name}'
                    )
                if member.issym():
                    link_target = (member_path.parent / linkname).resolve()
                else:
                    link_target = (dest_resolved / linkname).resolve()
                if not link_target.is_relative_to(dest_resolved):
                    raise ValueError(
                        f'Malicious tar member escaping destination: {member.name}'
                    )


def extract_tar_archive(archive_path: Path | str, dest_dir: Path | str) -> None:
    """Extracts a tar/tgz archive into dest_dir safely."""
    arch = Path(archive_path).resolve()
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not arch.exists():
        raise FileNotFoundError(f'Archive not found: {arch}')

    _validate_tar_members(arch, dest)

    has_tar = shutil.which('tar') is not None
    has_pigz = shutil.which('pigz') is not None
    extracted = False
    if has_tar:
        cmd = ['tar', '--no-same-owner']
        if has_pigz:
            cmd.extend(['-I', 'pigz', '-xf', str(arch), '-C', str(dest)])
        else:
            cmd.extend(['-xzf', str(arch), '-C', str(dest)])
        res = subprocess.run(cmd, capture_output=True, check=False)
        if res.returncode == 0:
            extracted = True

    if not extracted:
        try:
            with tarfile.open(arch, 'r:*') as tar:
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(path=dest, filter='data')
                else:
                    tar.extractall(path=dest)
                extracted = True
        except tarfile.FilterError as e:
            raise ValueError(
                f'Malicious tar member escaping destination: {e}'
            ) from e
        except Exception:
            # Subprocess tar fallback (only after _validate_tar_members passed)
            subprocess.run(
                ['tar', '--no-same-owner', '-xzf', str(arch), '-C', str(dest)],
                check=True,
                capture_output=True,
            )

    clean_circular_symlinks(dest)


def pack_tar_archive(source_dir: Path | str, output_archive: Path | str) -> None:
    """Packs directory into a gzip .tgz archive."""
    src = Path(source_dir).resolve()
    out = Path(output_archive).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    has_tar = shutil.which('tar') is not None
    has_pigz = shutil.which('pigz') is not None
    temp_out = out.with_name(f'{out.name}.tmp')

    if has_tar:
        cmd = ['tar']
        if has_pigz:
            cmd.extend(['-I', 'pigz -1', '-cf', str(temp_out), '-C', str(src), '.'])
        else:
            cmd.extend(['-czf', str(temp_out), '-C', str(src), '.'])
        res = subprocess.run(cmd, capture_output=True, check=False)
        if res.returncode == 0:
            temp_out.rename(out)
            return

    with tarfile.open(temp_out, 'w:gz') as tar:
        tar.add(src, arcname='.')
    temp_out.rename(out)


def create_binary_patch(
    base_dir: Path | str,
    target_dir: Path | str,
    output_patch_path: Path | str | None = None,
) -> str:
    """Generates a git binary patch diff representing changes from base_dir to target_dir.

    Preserves directory symlinks (--no-dereference), captures binary assets (--binary),
    file additions, deletions, renames, empty directories, and permission changes.

    Args:
        base_dir: Directory containing baseline repository state.
        target_dir: Directory containing target repository state.
        output_patch_path: Optional path to write the generated .patch file.

    Returns:
        String containing the unified git diff with binary support.
    """
    base_path = Path(base_dir).resolve()
    target_path = Path(target_dir).resolve()

    base_empty_dirs = {
        str(d.relative_to(base_path))
        for d in base_path.rglob('*')
        if d.is_dir() and '.git' not in d.parts and not any(d.iterdir())
    }
    target_empty_dirs = [
        str(d.relative_to(target_path))
        for d in target_path.rglob('*')
        if d.is_dir() and '.git' not in d.parts and not any(d.iterdir())
    ]
    deleted_empty_dirs = [
        rel_d for rel_d in sorted(base_empty_dirs) if not (target_path / rel_d).exists()
    ]

    with tempfile.TemporaryDirectory(prefix='dedup_diff_') as tmpdir:
        work_dir = Path(tmpdir) / 'work'
        shutil.copytree(base_path, work_dir, symlinks=True)

        if not (work_dir / '.git').exists():
            subprocess.run(
                ['git', '--no-pager', 'init', '-q'], cwd=work_dir, check=True
            )
        subprocess.run(
            ['git', '--no-pager', 'config', 'user.email', 'dedup@gemma4swe'],
            cwd=work_dir,
            check=True,
        )
        subprocess.run(
            ['git', '--no-pager', 'config', 'user.name', 'Dedup Tool'],
            cwd=work_dir,
            check=True,
        )
        subprocess.run(
            ['git', '--no-pager', 'add', '-A', '--force'], cwd=work_dir, check=True
        )
        subprocess.run(
            [
                'git',
                '--no-pager',
                'commit',
                '-m',
                'baseline',
                '-q',
                '--allow-empty',
            ],
            cwd=work_dir,
            check=True,
        )

        # Sync target over work tree preserving symlinks, deleting files not in target, excluding .git
        subprocess.run(
            [
                'rsync',
                '-a',
                '-c',
                '--delete',
                '--exclude=.git',
                f'{target_path}/',
                f'{work_dir}/',
            ],
            check=True,
        )

        # Stage changes (including .gitignore'd files) and generate staged binary diff
        subprocess.run(
            ['git', '--no-pager', 'add', '-A', '--force'], cwd=work_dir, check=True
        )

        diff_res = subprocess.run(
            [
                'git',
                '--no-pager',
                'diff',
                '--staged',
                '--binary',
                '--src-prefix=a/',
                '--dst-prefix=b/',
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        header_lines: list[str] = []
        for ed in sorted(target_empty_dirs):
            if ed not in base_empty_dirs:
                header_lines.append(f'# empty_dir: {ed}\n')
        for ded in deleted_empty_dirs:
            header_lines.append(f'# delete_empty_dir: {ded}\n')
        diff_text = ''.join(header_lines) + diff_res.stdout

    if output_patch_path is not None:
        out_p = Path(output_patch_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(diff_text, encoding='utf-8')

    return diff_text


def create_patch_from_archives(
    base_archive: Path | str,
    target_archive: Path | str,
    output_patch_path: Path | str | None = None,
) -> str:
    """Extracts base and target archive files and generates a binary patch between them."""
    base_tar = Path(base_archive).resolve()
    target_tar = Path(target_archive).resolve()

    if not base_tar.exists():
        raise FileNotFoundError(f'Base archive not found: {base_tar}')
    if not target_tar.exists():
        raise FileNotFoundError(f'Target archive not found: {target_tar}')

    with tempfile.TemporaryDirectory(prefix='dedup_arch_') as tmpdir:
        tmp = Path(tmpdir)
        b_dir = tmp / 'base'
        t_dir = tmp / 'target'
        b_dir.mkdir()
        t_dir.mkdir()

        extract_tar_archive(base_tar, b_dir)
        extract_tar_archive(target_tar, t_dir)

        return create_binary_patch(b_dir, t_dir, output_patch_path)


def _check_safe_patch_path(ws: Path, raw_path: str) -> None:
    """Validates that a file path referenced in a patch header does not escape ws."""
    s = raw_path.strip()
    if not s:
        return
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    s = s.split('\t', 1)[0].strip()
    if s in ('/dev/null', 'a/dev/null', 'b/dev/null'):
        return
    if s.startswith('/') or '..' in Path(s).parts:
        raise ValueError(f'Unsafe path in patch: {raw_path}')
    clean = s[2:] if s.startswith(('a/', 'b/')) else s
    if clean.startswith('/') or '..' in Path(clean).parts:
        raise ValueError(f'Unsafe path in patch: {raw_path}')
    target_p = (ws / clean).resolve()
    if not target_p.is_relative_to(ws):
        raise ValueError(f'Unsafe path in patch: {raw_path}')


def apply_patch_to_dir(
    workspace_dir: Path | str,
    patch_path_or_text: Path | str,
) -> tuple[bool, str, str]:
    """Applies a patch onto a workspace directory using multi-strategy fallbacks.

    Strategies attempted in order:
    1. git apply --binary [--ignore-space-change --ignore-whitespace] [-3] [--recount]
    2. Prefixless -p0 git apply
    3. Non-interactive patch -p1 and -p0 with whitespace tolerance.

    Returns:
        tuple (success: bool, stdout: str, stderr: str)
    """
    ws = Path(workspace_dir).resolve()
    if not ws.exists():
        return False, '', f'Workspace directory not found: {ws}'

    temp_patch_dir: Path | None = None
    if isinstance(patch_path_or_text, Path) or (
        isinstance(patch_path_or_text, str) and os.path.exists(patch_path_or_text)
    ):
        patch_file = Path(patch_path_or_text).resolve()
        if not patch_file.exists():
            return False, '', f'Patch file not found: {patch_file}'
        if patch_file.stat().st_size == 0:
            return True, '', ''
    else:
        patch_content = str(patch_path_or_text)
        if not patch_content.strip():
            return True, '', ''
        temp_dir = Path(tempfile.mkdtemp(prefix='dedup_patch_'))
        patch_file = temp_dir / 'input.patch'
        if isinstance(patch_path_or_text, bytes):
            patch_file.write_bytes(patch_path_or_text)
        else:
            patch_file.write_text(patch_content, encoding='utf-8')
        temp_patch_dir = temp_dir

    try:
        empty_dirs_to_create: list[Path] = []
        empty_dirs_to_delete: list[Path] = []
        in_hunk = False
        rem_old = 0
        rem_new = 0

        with open(patch_file, 'rb') as pf:
            for line in pf:
                line_str = line.decode('utf-8', errors='ignore').rstrip('\r\n')
                if not in_hunk and line_str.startswith('# empty_dir: '):
                    rel_dir = line_str.split(': ', 1)[-1].strip()
                    if rel_dir:
                        if rel_dir.startswith('/') or '..' in Path(rel_dir).parts:
                            raise ValueError(f'Unsafe path in patch empty_dir: {rel_dir}')
                        target_empty = (ws / rel_dir).resolve()
                        if not target_empty.is_relative_to(ws):
                            raise ValueError(f'Unsafe path in patch empty_dir: {rel_dir}')
                        empty_dirs_to_create.append(target_empty)
                        target_empty.mkdir(parents=True, exist_ok=True)
                    continue
                if not in_hunk and line_str.startswith('# delete_empty_dir: '):
                    rel_dir = line_str.split(': ', 1)[-1].strip()
                    if rel_dir:
                        if rel_dir.startswith('/') or '..' in Path(rel_dir).parts:
                            raise ValueError(
                                f'Unsafe path in patch delete_empty_dir: {rel_dir}'
                            )
                        target_del = (ws / rel_dir).resolve()
                        if not target_del.is_relative_to(ws):
                            raise ValueError(
                                f'Unsafe path in patch delete_empty_dir: {rel_dir}'
                            )
                        empty_dirs_to_delete.append(target_del)
                    continue

                if line_str.startswith('diff --git '):
                    in_hunk = False
                    rem_old = 0
                    rem_new = 0
                    m_q = re.match(r'^diff --git\s+"a/(.+?)"\s+"b/(.+?)"$', line_str)
                    m_same = re.match(r'^diff --git\s+a/(.+)\s+b/\1$', line_str)
                    m_std = re.match(r'^diff --git\s+a/(\S+)\s+b/(\S+)$', line_str)
                    if m_q:
                        _check_safe_patch_path(ws, m_q.group(1))
                        _check_safe_patch_path(ws, m_q.group(2))
                    elif m_same:
                        _check_safe_patch_path(ws, m_same.group(1))
                    elif m_std:
                        _check_safe_patch_path(ws, m_std.group(1))
                        _check_safe_patch_path(ws, m_std.group(2))
                    continue

                if line_str.startswith('@@ '):
                    in_hunk = True
                    m = re.match(
                        r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line_str
                    )
                    if m:
                        rem_old = int(m.group(2)) if m.group(2) is not None else 1
                        rem_new = int(m.group(4)) if m.group(4) is not None else 1
                    continue

                if line_str.startswith(('--- a/', '--- "a/', '--- /dev/null')):
                    in_hunk = False
                    hdr_path = line_str[4:].split('\t', 1)[0].strip()
                    _check_safe_patch_path(ws, hdr_path)
                    continue

                if in_hunk:
                    if line_str.startswith(' '):
                        rem_old -= 1
                        rem_new -= 1
                    elif line_str.startswith('-'):
                        rem_old -= 1
                    elif line_str.startswith('+'):
                        rem_new -= 1
                    if rem_old <= 0 and rem_new <= 0:
                        in_hunk = False
                    continue

                if line_str.startswith('--- ') or line_str.startswith('+++ '):
                    _check_safe_patch_path(ws, line_str[4:])
                elif line_str.startswith('rename from '):
                    _check_safe_patch_path(ws, line_str[len('rename from ') :])
                elif line_str.startswith('rename to '):
                    _check_safe_patch_path(ws, line_str[len('rename to ') :])

        if not empty_dirs_to_create and not empty_dirs_to_delete:
            # Check if patch has no diff hunks (only comments/headers)
            pass

        if not (ws / '.git').exists():
            subprocess.run(
                ['git', '--no-pager', 'init', '-q'], cwd=ws, check=False
            )
            subprocess.run(
                ['git', '--no-pager', 'config', 'user.email', 'eval@test'],
                cwd=ws,
                check=False,
            )
            subprocess.run(
                ['git', '--no-pager', 'config', 'user.name', 'Eval'],
                cwd=ws,
                check=False,
            )
            subprocess.run(
                ['git', '--no-pager', 'add', '-A', '--force'], cwd=ws, check=False
            )
            subprocess.run(
                [
                    'git',
                    '--no-pager',
                    'commit',
                    '-m',
                    'init',
                    '-q',
                    '--allow-empty',
                ],
                cwd=ws,
                check=False,
            )

        # Check if patch contains any actual file diff (beyond # empty_dir / # delete_empty_dir headers)
        has_diff_body = False
        with open(patch_file, 'rb') as pf:
            for line in pf:
                if not line.startswith(b'#') and line.strip():
                    has_diff_body = True
                    break

        def _finalize_empty_dirs() -> None:
            for ed in empty_dirs_to_create:
                ed.mkdir(parents=True, exist_ok=True)
            for ded in sorted(
                empty_dirs_to_delete, key=lambda p: len(p.parts), reverse=True
            ):
                if ded.exists() and ded.is_dir() and not any(ded.iterdir()):
                    ded.rmdir()

        if not has_diff_body:
            _finalize_empty_dirs()
            return True, '', ''

        commands = [
            ['git', '--no-pager', 'apply', '--binary', str(patch_file)],
            [
                'git',
                '--no-pager',
                'apply',
                '--binary',
                '--ignore-space-change',
                '--ignore-whitespace',
                str(patch_file),
            ],
            [
                'git',
                '--no-pager',
                'apply',
                '--binary',
                '-3',
                str(patch_file),
            ],
            [
                'git',
                '--no-pager',
                'apply',
                '--binary',
                '--recount',
                str(patch_file),
            ],
            [
                'git',
                '--no-pager',
                'apply',
                '--binary',
                '-p0',
                str(patch_file),
            ],
            [
                'git',
                '--no-pager',
                'apply',
                '--binary',
                '-p0',
                '--ignore-space-change',
                '--ignore-whitespace',
                str(patch_file),
            ],
            [
                'patch',
                '-p1',
                '--batch',
                '--forward',
                '--ignore-whitespace',
                '-i',
                str(patch_file),
            ],
            [
                'patch',
                '-p0',
                '--batch',
                '--forward',
                '--ignore-whitespace',
                '-i',
                str(patch_file),
            ],
        ]

        errors: list[str] = []
        for cmd in commands:
            try:
                res = subprocess.run(
                    cmd, cwd=ws, capture_output=True, text=True, check=False
                )
                if res.returncode == 0:
                    _finalize_empty_dirs()
                    return True, res.stdout, ''
                errors.append(
                    f"{' '.join(cmd)} failed ({res.returncode}):\n{res.stderr}\n{res.stdout}"
                )
            except FileNotFoundError:
                continue

        return False, '', '\n'.join(errors)
    finally:
        if temp_patch_dir is not None and temp_patch_dir.exists():
            shutil.rmtree(temp_patch_dir, ignore_errors=True)


def verify_byte_identical_trees(
    dir_a: Path | str,
    dir_b: Path | str,
    exclude: Sequence[str] = ('.git',),
) -> tuple[bool, str]:
    """Verifies that two directory trees are 100% byte identical without dereferencing symlinks."""
    cmd = ['diff', '-r', '--no-dereference']
    for ex in exclude:
        cmd.append(f'--exclude={ex}')
    cmd.extend([str(dir_a), str(dir_b)])

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.returncode == 0, res.stdout


def validate_patch_reconstruction(
    base_archive_or_dir: Path | str,
    patch_path_or_text: Path | str,
    target_archive_or_dir: Path | str,
) -> tuple[bool, str]:
    """Validates that applying patch over base recreates target 100% byte-identically.

    Args:
        base_archive_or_dir: Base snapshot (.tgz file or extracted directory).
        patch_path_or_text: Patch diff (.patch file or diff text string).
        target_archive_or_dir: Target snapshot (.tgz file or extracted directory).

    Returns:
        tuple (is_valid: bool, error_message: str)
    """
    with tempfile.TemporaryDirectory(prefix='val_patch_') as tmpdir:
        tmp = Path(tmpdir)
        reconstruct_dir = tmp / 'reconstruct'
        target_dir = tmp / 'target'
        reconstruct_dir.mkdir()
        target_dir.mkdir()

        base_p = Path(base_archive_or_dir).resolve()
        if base_p.is_file():
            extract_tar_archive(base_p, reconstruct_dir)
        elif base_p.is_dir():
            if (base_p / '.git').exists():
                subprocess.run(
                    ['git', '--no-pager', 'clone', '--shared', '-q', str(base_p), str(reconstruct_dir)],
                    check=True,
                )
            else:
                shutil.copytree(
                    base_p, reconstruct_dir, dirs_exist_ok=True, symlinks=True
                )
        else:
            return False, f'Base snapshot not found: {base_p}'

        ok, _out, err = apply_patch_to_dir(reconstruct_dir, patch_path_or_text)
        if not ok:
            return False, f'Failed to apply patch: {err}'

        target_p = Path(target_archive_or_dir).resolve()
        if target_p.is_file():
            extract_tar_archive(target_p, target_dir)
            compare_target = target_dir
        elif target_p.is_dir():
            compare_target = target_p
        else:
            return False, f'Target snapshot not found: {target_p}'

        identical, diff_out = verify_byte_identical_trees(
            reconstruct_dir, compare_target
        )
        if not identical:
            return False, f'Reconstruction diff mismatch:\n{diff_out[:1000]}'

        return True, ''


def reconstruct_snapshot(
    base_snapshot_path: Path | str,
    patch_path: Path | str | None,
    output_dir: Path | str,
    fallback_path: Path | str | None = None,
) -> bool:
    """Unpacks base snapshot and applies patch to reconstruct task workspace at execution time.

    If patch fails and a fallback archive is provided, falls back to unpacking
    the standalone snapshot archive.

    Args:
        base_snapshot_path: Path to base snapshot archive (.tgz).
        patch_path: Path to per-task patch diff file (.patch) or None.
        output_dir: Destination directory where working tree is reconstructed.
        fallback_path: Optional path to standalone .tgz snapshot archive for fallback.

    Returns:
        bool: True if reconstruction or fallback succeeded, False otherwise.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_p = Path(base_snapshot_path).resolve()
    fallback_p = Path(fallback_path).resolve() if fallback_path else None

    # Case 1: Base missing but fallback available
    if not base_p.exists():
        if fallback_p and fallback_p.exists():
            logger.info(
                'Base snapshot missing at %s, using fallback archive %s',
                base_p,
                fallback_p,
            )
            try:
                extract_tar_archive(fallback_p, out_dir)
                clean_circular_symlinks(out_dir)
                return True
            except Exception as e:
                logger.error('Failed to extract fallback %s: %s', fallback_p, e)
                return False
        raise FileNotFoundError(f'Base snapshot missing: {base_p}')

    # Extract base snapshot
    try:
        extract_tar_archive(base_p, out_dir)
    except Exception as e:
        logger.error('Failed to extract base snapshot %s: %s', base_p, e)
        if fallback_p and fallback_p.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                extract_tar_archive(fallback_p, out_dir)
                clean_circular_symlinks(out_dir)
                return True
            except Exception:
                return False
        return False

    # If patch is None or empty, base snapshot IS the target state
    if patch_path is None:
        clean_circular_symlinks(out_dir)
        return True

    patch_p = Path(patch_path).resolve()
    if not patch_p.exists() or patch_p.stat().st_size == 0:
        clean_circular_symlinks(out_dir)
        return True

    # Apply patch
    ok, _out, err = apply_patch_to_dir(out_dir, patch_p)
    if ok:
        clean_circular_symlinks(out_dir)
        if (out_dir / '.git').exists():
            subprocess.run(
                ['git', '--no-pager', 'config', 'user.email', 'dedup@gemma4swe'],
                cwd=out_dir,
                check=False,
            )
            subprocess.run(
                ['git', '--no-pager', 'config', 'user.name', 'Dedup Tool'],
                cwd=out_dir,
                check=False,
            )
            subprocess.run(
                ['git', '--no-pager', 'add', '-A', '--force'],
                cwd=out_dir,
                check=False,
            )
            subprocess.run(
                [
                    'git',
                    '--no-pager',
                    'commit',
                    '-m',
                    'reconstruct baseline',
                    '-q',
                    '--allow-empty',
                ],
                cwd=out_dir,
                check=False,
            )
        return True

    logger.warning('Patch %s failed to apply over %s: %s', patch_p, base_p, err)
    if fallback_p and fallback_p.exists():
        logger.info('Falling back to standalone snapshot archive %s', fallback_p)
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            extract_tar_archive(fallback_p, out_dir)
            clean_circular_symlinks(out_dir)
            return True
        except Exception as e:
            logger.error('Failed to extract fallback %s: %s', fallback_p, e)
            return False

    return False


def purge_tar_gz_archives(
    directory: Path | str, dry_run: bool = False
) -> list[Path]:
    """Finds and deletes redundant .tar.gz archives, enforcing strictly .tgz archives.

    Args:
        directory: Directory to search.
        dry_run: If True, returns matches without deleting.

    Returns:
        List of purged (or found) .tar.gz file paths.
    """
    dir_path = Path(directory).resolve()
    purged: list[Path] = []
    if not dir_path.exists():
        return purged

    for p in dir_path.rglob('*.tar.gz'):
        if p.is_file():
            purged.append(p)
            if not dry_run:
                p.unlink()

    return purged


def validate_archive_extensions(
    directory: Path | str, allowed_extensions: Sequence[str] = ('.tgz',)
) -> list[Path]:
    """Finds any archive files in directory violating allowed extension conventions."""
    dir_path = Path(directory).resolve()
    violating: list[Path] = []
    if not dir_path.exists():
        return violating

    for p in dir_path.rglob('*'):
        if p.is_file() and (
            p.name.endswith('.tar.gz')
            or (
                p.suffix in ('.tar', '.gz', '.bz2', '.xz')
                and not p.name.endswith('.tgz')
            )
        ):
            violating.append(p)

    return violating


@dataclasses.dataclass(frozen=True)
class DeduplicationSummary:
    """Summary statistics of a snapshot deduplication run."""

    total_tasks: int
    base_snapshots: dict[str, str]
    patches_count: int
    fallbacks_count: int
    original_size_bytes: int
    deduplicated_size_bytes: int
    savings_bytes: int
    savings_percentage: float
    manifest: list[dict[str, Any]]


def deduplicate_snapshots(
    tasks: Sequence[dict[str, Any] | Task] | Path | str,
    source_snapshots_dir: Path | str,
    output_dir: Path | str,
    base_snapshots_dir: Path | str | None = None,
    fallback_on_error: bool = True,
    max_workers: int = 8,
    purge_tar_gz: bool = False,
) -> DeduplicationSummary:
    """Deduplicates a task dataset into base snapshots, patches, and fallbacks.

    Args:
        tasks: Sequence of task dicts/Tasks, or path to a JSONL tasks file.
        source_snapshots_dir: Directory containing source standalone snapshot .tgz files.
        output_dir: Destination directory for deduplicated artifacts.
        base_snapshots_dir: Optional custom directory to store/read base snapshots.
        fallback_on_error: If True, retains standalone archive for tasks whose patch fails validation.
        max_workers: Parallel worker threads for patch creation and validation.
        purge_tar_gz: If True, purges redundant .tar.gz archives from source directory first.

    Returns:
        DeduplicationSummary dataclass containing size metrics and manifest.
    """
    src_snaps = Path(source_snapshots_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if purge_tar_gz:
        purged = purge_tar_gz_archives(src_snaps)
        logger.info('Purged %d redundant .tar.gz files from %s', len(purged), src_snaps)

    # Load tasks if path given
    task_list: list[dict[str, Any] | Task] = []
    if isinstance(tasks, (Path, str)):
        t_path = Path(tasks)
        with open(t_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task_list.append(json.loads(line))
    else:
        task_list = list(tasks)

    # Base snapshots and output directories
    base_out_dir = Path(base_snapshots_dir).resolve() if base_snapshots_dir else out_dir / 'base_snapshots'
    patches_out_dir = out_dir / 'patches'
    fallbacks_out_dir = out_dir / 'fallbacks'

    base_out_dir.mkdir(parents=True, exist_ok=True)
    patches_out_dir.mkdir(parents=True, exist_ok=True)
    fallbacks_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base snapshots identification
    base_task_map = identify_base_snapshots(task_list)
    repo_base_files: dict[str, Path] = {}
    base_summary_map: dict[str, str] = {}

    for repo, base_task in base_task_map.items():
        base_iid = base_task.instance_id if isinstance(base_task, Task) else base_task['instance_id']
        base_fn = get_base_snapshot_filename(repo)
        base_dest = base_out_dir / base_fn

        src_base = src_snaps / f'{base_iid}.tgz'
        if not src_base.exists():
            src_base = src_snaps / f'{base_iid}.tar.gz'

        if not src_base.exists():
            raise FileNotFoundError(f'Source snapshot for base task {base_iid} not found at {src_base}')

        if not base_dest.exists():
            shutil.copy2(src_base, base_dest)

        repo_base_files[repo] = base_dest
        base_summary_map[repo] = base_fn

    # 2. Pre-extract base snapshots and initialize git once per repo for high-speed shared cloning
    repo_base_dirs: dict[str, Path] = {}
    temp_work_dir = Path(tempfile.mkdtemp(prefix='dedup_cache_', dir=out_dir.parent))
    manifest: list[dict[str, Any]] = []
    original_size_total = 0
    patches_count = 0
    fallbacks_count = 0

    try:
        logger.info('Pre-extracting and caching %d base repositories...', len(repo_base_files))
        for repo, base_dest in repo_base_files.items():
            base_extract_dir = temp_work_dir / f'base_{clean_repo_name(repo)}'
            extract_tar_archive(base_dest, base_extract_dir)
            if not (base_extract_dir / '.git').exists():
                subprocess.run(
                    ['git', '--no-pager', 'init', '-q'],
                    cwd=base_extract_dir,
                    check=True,
                )
            subprocess.run(
                ['git', '--no-pager', 'config', 'user.email', 'dedup@gemma4swe'],
                cwd=base_extract_dir,
                check=True,
            )
            subprocess.run(
                ['git', '--no-pager', 'config', 'user.name', 'Dedup Tool'],
                cwd=base_extract_dir,
                check=True,
            )
            subprocess.run(
                ['git', '--no-pager', 'add', '-A', '--force'],
                cwd=base_extract_dir,
                check=True,
            )
            subprocess.run(
                [
                    'git',
                    '--no-pager',
                    'commit',
                    '-m',
                    'baseline',
                    '-q',
                    '--allow-empty',
                ],
                cwd=base_extract_dir,
                check=True,
            )
            repo_base_dirs[repo] = base_extract_dir

        def process_single_task(t: dict[str, Any] | Task) -> dict[str, Any]:
            iid = t.instance_id if isinstance(t, Task) else t['instance_id']
            repo = t.repo if isinstance(t, Task) else t['repo']
            base_dir = repo_base_dirs[repo]

            src_snap = src_snaps / f'{iid}.tgz'
            if not src_snap.exists():
                src_snap = src_snaps / f'{iid}.tar.gz'

            if not src_snap.exists():
                raise FileNotFoundError(f'Snapshot not found for task {iid} at {src_snap}')

            src_size = src_snap.stat().st_size
            base_task = base_task_map[repo]
            base_iid = base_task.instance_id if isinstance(base_task, Task) else base_task['instance_id']

            patch_file = patches_out_dir / f'{iid}.patch'
            fallback_file = fallbacks_out_dir / f'{iid}.tgz'

            # If this task is the base snapshot itself
            if iid == base_iid:
                patch_file.write_text('', encoding='utf-8')
                return {
                    'instance_id': iid,
                    'repo': repo,
                    'is_base': True,
                    'is_fallback': False,
                    'source_size_bytes': src_size,
                    'patch_file': patch_file.name,
                    'patch_size_bytes': 0,
                }

            # Create patch diff against base archive using shared git clone
            try:
                with tempfile.TemporaryDirectory(dir=temp_work_dir) as task_tmp:
                    t_dir = Path(task_tmp) / 'target'
                    extract_tar_archive(src_snap, t_dir)

                    # Record empty directories in base and target (excluding .git)
                    base_empty_dirs = {
                        str(d.relative_to(base_dir))
                        for d in base_dir.rglob('*')
                        if d.is_dir() and '.git' not in d.parts and not any(d.iterdir())
                    }
                    empty_dirs = [
                        str(d.relative_to(t_dir))
                        for d in t_dir.rglob('*')
                        if d.is_dir() and '.git' not in d.parts and not any(d.iterdir())
                    ]
                    deleted_empty_dirs = [
                        rel_d for rel_d in sorted(base_empty_dirs)
                        if not (t_dir / rel_d).exists()
                    ]

                    work_dir = Path(task_tmp) / 'work'
                    subprocess.run(
                        ['git', '--no-pager', 'clone', '--shared', '-q', str(base_dir), str(work_dir)],
                        check=True,
                    )
                    subprocess.run(
                        ['rsync', '-a', '-c', '--delete', '--exclude=.git', f'{t_dir}/', f'{work_dir}/'],
                        check=True,
                    )
                    subprocess.run(
                        ['git', '--no-pager', 'add', '-A', '--force'],
                        cwd=work_dir,
                        check=True,
                    )
                    diff_res = subprocess.run(
                        [
                            'git',
                            '--no-pager',
                            'diff',
                            '--staged',
                            '--binary',
                            '--src-prefix=a/',
                            '--dst-prefix=b/',
                        ],
                        cwd=work_dir,
                        capture_output=True,
                        text=False,
                        check=True,
                    )
                    header_lines = [f'# empty_dir: {ed}\n' for ed in sorted(empty_dirs)]
                    header_lines.extend(f'# delete_empty_dir: {ded}\n' for ded in deleted_empty_dirs)
                    header = ''.join(header_lines).encode('utf-8')
                    patch_file.write_bytes(header + diff_res.stdout)

                    # Validate reconstruction
                    is_valid, err_msg = validate_patch_reconstruction(
                        base_dir, patch_file, t_dir
                    )

                if is_valid:
                    return {
                        'instance_id': iid,
                        'repo': repo,
                        'is_base': False,
                        'is_fallback': False,
                        'source_size_bytes': src_size,
                        'patch_file': patch_file.name,
                        'patch_size_bytes': patch_file.stat().st_size,
                    }
                else:
                    logger.warning('Task %s failed reconstruction: %s', iid, err_msg)
                    if fallback_on_error:
                        if patch_file.exists():
                            patch_file.unlink()
                        shutil.copy2(src_snap, fallback_file)
                        return {
                            'instance_id': iid,
                            'repo': repo,
                            'is_base': False,
                            'is_fallback': True,
                            'fallback_reason': f'Reconstruction validation failed: {err_msg}',
                            'source_size_bytes': src_size,
                            'fallback_file': fallback_file.name,
                            'fallback_size_bytes': fallback_file.stat().st_size,
                        }
                    else:
                        raise RuntimeError(f'Task {iid} failed reconstruction validation: {err_msg}')
            except Exception as exc:
                logger.error('Error generating/validating patch for task %s: %s', iid, exc)
                if fallback_on_error:
                    if patch_file.exists():
                        patch_file.unlink()
                    shutil.copy2(src_snap, fallback_file)
                    return {
                        'instance_id': iid,
                        'repo': repo,
                        'is_base': False,
                        'is_fallback': True,
                        'fallback_reason': str(exc),
                        'source_size_bytes': src_size,
                        'fallback_file': fallback_file.name,
                        'fallback_size_bytes': fallback_file.stat().st_size,
                    }
                else:
                    raise

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_task, t): t for t in task_list
            }
            for future in as_completed(future_to_task):
                record = future.result()
                manifest.append(record)
    finally:
        shutil.rmtree(temp_work_dir, ignore_errors=True)

    # Sort manifest deterministically by instance_id
    manifest.sort(key=lambda x: x['instance_id'])

    for rec in manifest:
        original_size_total += rec.get('source_size_bytes', 0)
        if rec.get('is_fallback'):
            fallbacks_count += 1
        else:
            patches_count += 1

    # Compute deduplicated size
    base_size_total = sum(f.stat().st_size for f in base_out_dir.glob('*.tgz'))
    patches_size_total = sum(f.stat().st_size for f in patches_out_dir.glob('*.patch'))
    fallbacks_size_total = sum(f.stat().st_size for f in fallbacks_out_dir.glob('*.tgz'))
    deduped_size_total = base_size_total + patches_size_total + fallbacks_size_total

    savings_bytes = max(0, original_size_total - deduped_size_total)
    savings_pct = (savings_bytes / original_size_total * 100.0) if original_size_total > 0 else 0.0

    # Save manifest.json
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    summary = DeduplicationSummary(
        total_tasks=len(task_list),
        base_snapshots=base_summary_map,
        patches_count=patches_count,
        fallbacks_count=fallbacks_count,
        original_size_bytes=original_size_total,
        deduplicated_size_bytes=deduped_size_total,
        savings_bytes=savings_bytes,
        savings_percentage=savings_pct,
        manifest=manifest,
    )

    return summary


def resolve_task_snapshot_paths(
    snapshots_dir: Path | str,
    instance_id: str,
    repo: str,
) -> tuple[Path, Path | None, Path | None]:
    """Resolves (effective_snapshot_path, base_snapshot_path, patch_path) for task execution.

    Resolution order:
    1. Check for fallback standalone archive: snapshots_dir/fallbacks/{instance_id}.tgz
    2. Check for deduplicated base + patch:
       Base candidates:
         - snapshots_dir/base_snapshots/base_{clean_repo}.tgz
         - snapshots_dir/base_{clean_repo}.tgz
       Patch candidates:
         - snapshots_dir/patches/{instance_id}.patch
         - snapshots_dir/{instance_id}.patch
    3. Check for standalone archive:
       - snapshots_dir/{instance_id}.tgz
       - snapshots_dir/{instance_id}.tar.gz
    4. Default missing path: snapshots_dir/{instance_id}.tgz

    Returns:
        tuple (effective_snapshot_path, base_snapshot_path, patch_path)
    """
    s_dir = Path(snapshots_dir).resolve()
    iid = instance_id.strip()

    # 1. Fallback archive
    fallback_cand = s_dir / 'fallbacks' / f'{iid}.tgz'
    if fallback_cand.exists():
        return fallback_cand, None, None

    # 2. Deduplicated base snapshot + patch
    base_candidates = [
        s_dir / 'base_snapshots' / get_base_snapshot_filename(repo),
        s_dir / get_base_snapshot_filename(repo),
    ]
    base_file = next((b for b in base_candidates if b.exists()), None)
    if base_file is not None:
        patch_candidates = [
            s_dir / 'patches' / f'{iid}.patch',
            s_dir / f'{iid}.patch',
        ]
        patch_file = next((p for p in patch_candidates if p.exists()), None)
        if patch_file is not None:
            valid_patch = (
                patch_file if patch_file.stat().st_size > 0 else None
            )
            return base_file, base_file, valid_patch

    # 3. Direct standalone archive
    for ext in ('.tgz', '.tar.gz'):
        direct = s_dir / f'{iid}{ext}'
        if direct.exists():
            return direct, None, None

    # 4. Not found fallback
    return s_dir / f'{iid}.tgz', None, None
