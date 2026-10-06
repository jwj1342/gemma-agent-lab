"""Process and filesystem-based sandbox lifecycle management with per-task venv isolation."""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import venv
from pathlib import Path

from adk_eval_core.sandbox import (
    BaseSandbox,
    ExecutionResult,
    build_sanitized_env,
    execute_subprocess_command,
)

logger = logging.getLogger(__name__)


class SubprocessManager(BaseSandbox):
    """Manages isolated local working directories and virtual environments directly conforming to BaseSandbox."""

    def __init__(
        self,
        timeout_seconds: int = 300,
        base_dir: Path | None = None,
        system_site_packages: bool = True,
    ):
        self.timeout_seconds = timeout_seconds
        self.base_dir = base_dir
        self.system_site_packages = system_site_packages
        self._sandboxes: dict[str, dict[str, Path]] = {}
        self._default_sandbox_id: str | None = None
        self._initial_files: dict[str, str | bytes] = {}
        self._lock = threading.RLock()

    @property
    def sandboxes(self) -> dict[str, dict[str, Path]]:
        with self._lock:
            return {k: dict(v) for k, v in self._sandboxes.items()}

    @property
    def work_dir(self) -> str:
        with self._lock:
            if self._default_sandbox_id and self._default_sandbox_id in self._sandboxes:
                return str(self._sandboxes[self._default_sandbox_id]['workspace'])
        return '/workspace'

    def initialize(self, files: dict[str, str | bytes] | None = None) -> None:
        """Start default sandbox and optionally write initial files."""
        with self._lock:
            if (
                self._default_sandbox_id is None
                or self._default_sandbox_id not in self._sandboxes
            ):
                self._default_sandbox_id = self.start()
        if files is not None:
            self._initial_files = dict(files)
            for rel_path, content in files.items():
                self.write_file(rel_path, content)

    def run_command(
        self,
        command: str,
        timeout: float | None = 300,
        work_dir: str | None = None,
    ) -> ExecutionResult:
        """Execute a command in the active sandbox workspace."""
        if self._default_sandbox_id is None:
            self.initialize()
        assert self._default_sandbox_id is not None
        int_timeout = int(timeout) if timeout is not None else self.timeout_seconds
        cmd = f'cd {shlex.quote(work_dir)} && {command}' if work_dir else command
        return self.exec(self._default_sandbox_id, cmd, timeout=int_timeout)

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file into the active sandbox workspace."""
        if self._default_sandbox_id is None:
            self.initialize()
        assert self._default_sandbox_id is not None
        paths = self._sandboxes[self._default_sandbox_id]
        root_dir = paths['root'].resolve()
        ws_dir = paths['workspace'].resolve()
        if not posixpath.isabs(path):
            target_dest = ws_dir / path
            allowed_root = ws_dir
        elif path == '/workspace' or path.startswith('/workspace/'):
            rel = path[len('/workspace/') :] if path.startswith('/workspace/') else ''
            target_dest = ws_dir / rel
            allowed_root = ws_dir
        elif path.startswith(str(root_dir)):
            target_dest = Path(path)
            allowed_root = root_dir
        else:
            target_dest = root_dir / path.lstrip('/')
            allowed_root = root_dir

        target_resolved = target_dest.resolve()
        if not target_resolved.is_relative_to(allowed_root):
            raise ValueError(
                f'Path traversal detected: {path} escapes sandbox root {allowed_root}'
            )

        target_resolved.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target_resolved.write_text(content, encoding='utf-8')
        else:
            target_resolved.write_bytes(content)

    def read_file(self, path: str) -> str:
        """Read a file from the active sandbox workspace."""
        if self._default_sandbox_id is None:
            self.initialize()
        assert self._default_sandbox_id is not None
        paths = self._sandboxes[self._default_sandbox_id]
        root_dir = paths['root'].resolve()
        ws_dir = paths['workspace'].resolve()
        if not posixpath.isabs(path):
            source_path = ws_dir / path
            allowed_root = ws_dir
        elif path == '/workspace' or path.startswith('/workspace/'):
            rel = path[len('/workspace/') :] if path.startswith('/workspace/') else ''
            source_path = ws_dir / rel
            allowed_root = ws_dir
        elif path.startswith(str(root_dir)):
            source_path = Path(path)
            allowed_root = root_dir
        else:
            source_path = root_dir / path.lstrip('/')
            allowed_root = root_dir

        source_resolved = source_path.resolve()
        if not source_resolved.is_relative_to(allowed_root):
            raise ValueError(
                f'Path traversal detected: {path} escapes sandbox root {allowed_root}'
            )

        if not source_resolved.exists():
            raise FileNotFoundError(f'File {path} not found in sandbox')
        return source_resolved.read_text(encoding='utf-8', errors='replace')

    def copy_to(
        self,
        src_or_sandbox: str | Path,
        dst_or_src: str | Path,
        dst: str | Path | None = None,
    ) -> None:
        """Copy a file or directory into the sandbox.

        Supports both BaseSandbox 2-argument signature:
            copy_to(host_path, container_path)
        and legacy manager 3-argument signature:
            copy_to(sandbox_id, src, dst)
        """
        if dst is not None:
            sandbox_id = str(src_or_sandbox)
            src = Path(dst_or_src)
            target_dst = str(dst)
            self._copy_to_sandbox(sandbox_id, src, target_dst)
        else:
            if self._default_sandbox_id is None:
                self.initialize()
            assert self._default_sandbox_id is not None
            host_p = Path(src_or_sandbox)
            target_dst = str(dst_or_src)
            self._copy_to_sandbox(self._default_sandbox_id, host_p, target_dst)

    def _copy_to_sandbox(self, sandbox_id: str, src: Path, dst: str) -> None:
        if not src.exists():
            raise FileNotFoundError(f'Source path {src} does not exist')
        with self._lock:
            if sandbox_id not in self._sandboxes:
                raise ValueError(f'Unknown sandbox_id: {sandbox_id}')
            paths = dict(self._sandboxes[sandbox_id])
        root_dir = paths['root'].resolve()

        if dst in ('/tmp/', '/tmp'):
            target_dest = paths['tmp'] / src.name
        elif dst.startswith('/tmp/'):
            rel = dst[len('/tmp/') :]
            target_dest = paths['tmp'] / rel
        elif dst in ('/workspace/', '/workspace'):
            target_dest = paths['workspace'] / src.name
        elif dst.startswith('/workspace/'):
            rel = dst[len('/workspace/') :]
            target_dest = paths['workspace'] / rel
        elif dst in ('/wheels/', '/wheels'):
            target_dest = paths['wheels'] / src.name
        elif dst.startswith('/wheels/'):
            rel = dst[len('/wheels/') :]
            target_dest = paths['wheels'] / rel
        elif dst in ('/usr/local/bin/', '/usr/local/bin'):
            target_dest = paths['venv'] / 'bin' / src.name
        elif dst.startswith('/usr/local/bin/'):
            rel = dst[len('/usr/local/bin/') :]
            target_dest = paths['venv'] / 'bin' / rel
        elif not posixpath.isabs(dst):
            target_dest = paths['workspace'] / dst
        else:
            rel = dst.lstrip('/')
            target_dest = paths['root'] / rel

        if src.is_file() and (
            target_dest.is_dir()
            or (
                dst.endswith('/')
                and dst not in ('/tmp/', '/workspace/', '/wheels/', '/usr/local/bin/')
            )
        ):
            target_dest = target_dest / src.name

        target_resolved = target_dest.resolve()
        if not target_resolved.is_relative_to(root_dir):
            raise ValueError(
                f"Path traversal detected: destination '{dst}' escapes sandbox root '{root_dir}'"
            )

        target_dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, target_dest)
        elif src.is_dir():
            shutil.copytree(src, target_dest, dirs_exist_ok=True)

    def copy_from(
        self,
        src_or_sandbox: str | Path,
        dst_or_src: str | Path,
        dst: str | Path | None = None,
    ) -> None:
        """Copy a file or directory from sandbox to host.

        Supports both BaseSandbox 2-argument signature:
            copy_from(container_path, host_path)
        and legacy manager 3-argument signature:
            copy_from(sandbox_id, src, dst)
        """
        if dst is not None:
            sandbox_id = str(src_or_sandbox)
            src = str(dst_or_src)
            dest_p = Path(dst)
            self._copy_from_sandbox(sandbox_id, src, dest_p)
        else:
            if self._default_sandbox_id is None:
                self.initialize()
            assert self._default_sandbox_id is not None
            src_p = str(src_or_sandbox)
            dest_p = Path(dst_or_src)
            self._copy_from_sandbox(self._default_sandbox_id, src_p, dest_p)

    def _copy_from_sandbox(self, sandbox_id: str, src: str, dst: Path) -> None:
        with self._lock:
            if sandbox_id not in self._sandboxes:
                raise ValueError(f'Unknown sandbox_id: {sandbox_id}')
            paths = dict(self._sandboxes[sandbox_id])
        root_dir = paths['root'].resolve()

        if src in ('/workspace', '/workspace/'):
            source_path = paths['workspace']
        elif src.startswith('/workspace/'):
            source_path = paths['workspace'] / src[len('/workspace/') :]
        elif src in ('/tmp', '/tmp/'):
            source_path = paths['tmp']
        elif src.startswith('/tmp/'):
            source_path = paths['tmp'] / src[len('/tmp/') :]
        elif src in ('/wheels', '/wheels/'):
            source_path = paths['wheels']
        elif src.startswith('/wheels/'):
            source_path = paths['wheels'] / src[len('/wheels/') :]
        elif src in ('/usr/local/bin', '/usr/local/bin/'):
            source_path = paths['venv'] / 'bin'
        elif src.startswith('/usr/local/bin/'):
            source_path = paths['venv'] / 'bin' / src[len('/usr/local/bin/') :]
        elif not posixpath.isabs(src):
            source_path = paths['workspace'] / src
        else:
            source_path = paths['root'] / src.lstrip('/')

        source_resolved = source_path.resolve()
        if not source_resolved.is_relative_to(root_dir):
            raise ValueError(
                f"Path traversal detected: source '{src}' escapes sandbox root '{root_dir}'"
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_file():
            shutil.copy2(source_path, dst)
        elif source_path.is_dir():
            for item in source_path.rglob('*'):
                if item.is_symlink() or item.exists():
                    item_resolved = item.resolve()
                    if not item_resolved.is_relative_to(root_dir):
                        raise ValueError(
                            f"Path traversal detected: symlink '{item}' escapes sandbox root '{root_dir}'"
                        )
            shutil.copytree(source_path, dst, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(
                f'Source path {src} not found in sandbox {sandbox_id}'
            )

    def start(self) -> str:
        """Create a new isolated sandbox working directory and virtual environment."""
        sandbox_id = str(uuid.uuid4())[:12]
        if self.base_dir:
            sandbox_root = self.base_dir / f'swegemma_sandbox_{sandbox_id}'
            sandbox_root.mkdir(parents=True, exist_ok=True)
        else:
            sandbox_root = Path(
                tempfile.mkdtemp(prefix=f'swegemma_sandbox_{sandbox_id}_')
            )

        workspace_dir = sandbox_root / 'workspace'
        tmp_dir = sandbox_root / 'tmp'
        wheels_dir = sandbox_root / 'wheels'
        venv_dir = sandbox_root / 'venv'

        workspace_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wheels_dir.mkdir(parents=True, exist_ok=True)

        venv_has_pip = False
        for use_symlinks in (True, False):
            try:
                venv.create(
                    venv_dir,
                    system_site_packages=self.system_site_packages,
                    with_pip=True,
                    symlinks=use_symlinks,
                    clear=True,
                )
                venv_has_pip = True
                break
            except Exception as e:
                if use_symlinks:
                    logger.debug(
                        'venv with_pip+symlinks failed for sandbox %s: %s',
                        sandbox_id,
                        e,
                    )
                    continue
                logger.warning(
                    'ensurepip unavailable in sandbox %s, creating venv without pip: %s',
                    sandbox_id,
                    e,
                )

        if not venv_has_pip:
            for use_symlinks in (True, False):
                try:
                    venv.create(
                        venv_dir,
                        system_site_packages=self.system_site_packages,
                        with_pip=False,
                        symlinks=use_symlinks,
                        clear=True,
                    )
                    break
                except Exception:
                    if not use_symlinks:
                        raise

            venv_python = venv_dir / 'bin' / 'python3'
            if venv_python.exists():
                try:
                    subprocess.run(
                        [str(venv_python), '-m', 'pip', '--version'],
                        capture_output=True,
                        timeout=30,
                        check=True,
                    )
                    venv_has_pip = True
                except Exception:
                    logger.warning(
                        'pip not available in sandbox %s venv; '
                        'task package installation may be skipped.',
                        sandbox_id,
                    )

        if self.system_site_packages:
            import sys

            host_site_packages = []
            for p in sys.path:
                if 'site-packages' in p and Path(p).exists() and Path(p).is_dir():
                    host_site_packages.append(p)
            for sp_dir in venv_dir.glob('lib/python*/site-packages'):
                pth_file = sp_dir / '_host_env.pth'
                pth_file.write_text(
                    '\n'.join(host_site_packages) + '\n', encoding='utf-8'
                )

        with self._lock:
            self._sandboxes[sandbox_id] = {
                'root': sandbox_root,
                'workspace': workspace_dir,
                'tmp': tmp_dir,
                'wheels': wheels_dir,
                'venv': venv_dir,
            }
            if self._default_sandbox_id is None:
                self._default_sandbox_id = sandbox_id
        return sandbox_id

    async def start_async(self) -> str:
        """Start a new sandbox instance asynchronously without blocking the event loop."""
        return await asyncio.to_thread(self.start)

    def exec(
        self, sandbox_id: str, command: str, *, timeout: int | None = None
    ) -> ExecutionResult:
        """Execute a command in the local sandbox workspace and isolated venv."""
        with self._lock:
            if sandbox_id not in self._sandboxes:
                raise ValueError(f'Unknown sandbox_id: {sandbox_id}')
            paths = dict(self._sandboxes[sandbox_id])
        workspace_dir = paths['workspace']
        tmp_dir = paths['tmp']
        wheels_dir = paths['wheels']
        venv_dir = paths['venv']
        timeout = timeout if timeout is not None else self.timeout_seconds

        root_dir_str = str(paths['root'])
        sentinel = '__SWEGEMMA_SANDBOX_ROOT__'
        translated_cmd = command.replace(root_dir_str, sentinel)
        translated_cmd = re.sub(
            r'(?<![\w.-])/(?:\.\.?/)+(tmp|workspace|usr/local/bin|wheels)(?![\w.-])',
            r'/\1',
            translated_cmd,
        )

        translated_cmd = re.sub(
            r'(?<![\w.-])/tmp(?![\w.-])', str(tmp_dir), translated_cmd
        )
        translated_cmd = re.sub(
            r'(?<![\w.-])/workspace(?![\w.-])',
            str(workspace_dir),
            translated_cmd,
        )

        bin_dir = paths['venv'] / 'bin'
        bin_dir.mkdir(parents=True, exist_ok=True)
        translated_cmd = re.sub(
            r'(?<![\w.-])/usr/local/bin(?![\w.-])',
            str(bin_dir),
            translated_cmd,
        )

        if wheels_dir.exists() and any(wheels_dir.glob('*.whl')):
            active_wheels = wheels_dir
        else:
            active_wheels = None
            for wh_cand in [
                Path('/kaggle/input/competition-data/wheels'),
                Path('/kaggle/input/datasets/ryanholbrook/gemma4swe-wheelhouses'),
                Path('build/wheelhouse'),
                Path('data/wheels'),
                Path('wheels'),
                Path('/tmp/wheelhouse'),
            ]:
                if wh_cand.exists() and any(wh_cand.glob('*.whl')):
                    active_wheels = wh_cand.resolve()
                    break

        if active_wheels:
            translated_cmd = re.sub(
                r'(?<![\w.-])/wheels(?![\w.-])',
                str(active_wheels),
                translated_cmd,
            )

        translated_cmd = translated_cmd.replace(sentinel, root_dir_str)

        venv_bin = venv_dir / 'bin'
        host_bin = str(Path(sys.executable).parent)
        extra_env = {
            'VIRTUAL_ENV': str(venv_dir),
            'PATH': f'{venv_bin}:{host_bin}:{os.environ.get("PATH", "/usr/bin:/bin")}',
            'PYTHONPATH': str(workspace_dir),
        }
        env = build_sanitized_env(
            work_dir=workspace_dir,
            tmp_dir=tmp_dir,
            extra_env=extra_env,
        )
        return execute_subprocess_command(
            command=['/bin/bash', '-c', translated_cmd],
            cwd=workspace_dir,
            env=env,
            timeout=timeout,
            shell=False,
        )

    async def exec_async(
        self, sandbox_id: str, command: str, *, timeout: int | None = None
    ) -> ExecutionResult:
        """Execute a command in the local sandbox workspace asynchronously without blocking the event loop."""
        return await asyncio.to_thread(self.exec, sandbox_id, command, timeout=timeout)

    def stop(self, sandbox_id: str) -> None:
        """Stop and cleanup the sandbox directory and virtual environment."""
        with self._lock:
            entry = self._sandboxes.pop(sandbox_id, None)
            if self._default_sandbox_id == sandbox_id:
                self._default_sandbox_id = next(iter(self._sandboxes.keys()), None)
        if entry is not None:
            shutil.rmtree(entry['root'], ignore_errors=True)

    async def stop_async(self, sandbox_id: str) -> None:
        """Stop and cleanup the sandbox directory asynchronously without blocking the event loop."""
        await asyncio.to_thread(self.stop, sandbox_id)

    def cleanup_all(self) -> None:
        """Stop and cleanup all active sandboxes."""
        with self._lock:
            sids = list(self._sandboxes.keys())
        for sid in sids:
            self.stop(sid)

    def reset(self) -> None:
        """Reset default sandbox workspace."""
        with self._lock:
            default_id = self._default_sandbox_id
            ws = (
                self._sandboxes[default_id]['workspace']
                if (default_id and default_id in self._sandboxes)
                else None
            )
        if ws is not None:
            for item in ws.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            if self._initial_files:
                for rel_path, content in self._initial_files.items():
                    self.write_file(rel_path, content)

    def close(self) -> None:
        """Clean up all sandboxes."""
        self.cleanup_all()
        with self._lock:
            self._default_sandbox_id = None

    def get_sandbox(self, sandbox_id: str, work_dir: str = '/workspace') -> BaseSandbox:
        """Return a BaseSandbox view of a specific managed sandbox instance."""
        from swegemma.sandbox.base import ManagedSandbox

        return ManagedSandbox(manager=self, sandbox_id=sandbox_id, _work_dir=work_dir)

