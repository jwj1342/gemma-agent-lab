"""Base interface and result models for sandbox execution."""

from __future__ import annotations

import asyncio
import inspect
import posixpath
import tempfile
from pathlib import Path
from typing import Any, Literal

from adk_eval_core.sandbox import (
    AdkSandboxCodeExecutor,
    BaseSandbox,
    ExecutionResult,
)


class ExecResult(ExecutionResult):
    """Deprecated alias for adk_eval_core.sandbox.ExecutionResult."""

    def __init__(
        self,
        exit_code: int = 0,
        stdout: str = '',
        stderr: str = '',
        duration_seconds: float = 0.0,
        *,
        timed_out: bool = False,
        status: Literal['ok', 'error', 'timeout'] | None = None,
    ) -> None:
        if status is None:
            status = 'timeout' if timed_out else ('ok' if exit_code == 0 else 'error')
        super().__init__(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
        )


class BaseSandboxManager(BaseSandbox):
    """Deprecated: BaseSandboxManager is superseded by adk_eval_core.sandbox.BaseSandbox."""

    def start(self) -> str:
        """Start a new sandbox instance and return its identifier."""
        raise NotImplementedError

    async def start_async(self) -> str:
        """Start a new sandbox instance asynchronously without blocking the event loop."""
        return await asyncio.to_thread(self.start)

    def exec(
        self, sandbox_id: str, command: str, *, timeout: int | None = None
    ) -> ExecutionResult:
        """Execute a shell command within the sandbox."""
        raise NotImplementedError

    async def exec_async(
        self, sandbox_id: str, command: str, *, timeout: int | None = None
    ) -> ExecutionResult:
        """Execute a shell command within the sandbox asynchronously."""
        return await asyncio.to_thread(self.exec, sandbox_id, command, timeout=timeout)

    def copy_to(
        self,
        src_or_container: str | Path,
        dst_or_src: str | Path,
        dst: str | Path | None = None,
    ) -> None:
        """Copy a file or directory into the sandbox."""
        raise NotImplementedError

    def copy_from(
        self,
        src_or_container: str | Path,
        dst_or_src: str | Path,
        dst: str | Path | None = None,
    ) -> None:
        """Copy a file or directory from the sandbox to host."""
        raise NotImplementedError

    def stop(self, sandbox_id: str) -> None:
        """Stop and cleanup the sandbox instance."""
        raise NotImplementedError

    async def stop_async(self, sandbox_id: str) -> None:
        """Stop and cleanup the sandbox instance asynchronously without blocking the event loop."""
        await asyncio.to_thread(self.stop, sandbox_id)

    def cleanup_all(self) -> None:
        """Stop and cleanup all managed sandboxes."""
        pass

    def get_sandbox(
        self, sandbox_id: str, work_dir: str = '/workspace'
    ) -> ManagedSandbox:
        """Return a BaseSandbox view of a specific managed sandbox instance."""
        return ManagedSandbox(manager=self, sandbox_id=sandbox_id, _work_dir=work_dir)


class ManagedSandbox(BaseSandbox):
    """Deprecated compatibility wrapper delegating to a sandbox instance."""

    def __init__(
        self,
        manager: Any,
        sandbox_id: str,
        _work_dir: str = '/workspace',
    ) -> None:
        self.manager: Any = manager
        self.sandbox_id = sandbox_id
        self._work_dir_path = _work_dir
        self._initial_files: dict[str, str | bytes] = {}

    @property
    def work_dir(self) -> str:
        return self._work_dir_path

    def initialize(self, files: dict[str, str | bytes] | None = None) -> None:
        if files:
            self._initial_files = dict(files)
            for rel_path, content in files.items():
                self.write_file(rel_path, content)

    def run_command(
        self,
        command: str,
        timeout: float | None = 300,
        work_dir: str | None = None,
    ) -> ExecutionResult:
        import shlex

        wd = work_dir or self._work_dir_path
        cmd = f'cd {shlex.quote(str(wd))} && {command}' if wd else command
        int_timeout = int(timeout) if timeout is not None else None
        if hasattr(self.manager, 'exec'):
            return self.manager.exec(self.sandbox_id, cmd, timeout=int_timeout)
        return self.manager.run_command(cmd, timeout=timeout, work_dir=wd)

    def exec(self, command: str, *, timeout: int | None = None) -> ExecutionResult:
        """Execute a command directly in this managed sandbox."""
        if hasattr(self.manager, 'exec'):
            return self.manager.exec(self.sandbox_id, command, timeout=timeout)
        return self.run_command(command, timeout=timeout)

    async def exec_async(
        self, command: str, *, timeout: int | None = None
    ) -> ExecutionResult:
        """Execute a command directly in this managed sandbox asynchronously."""
        if hasattr(self.manager, 'exec_async'):
            return await self.manager.exec_async(
                self.sandbox_id, command, timeout=timeout
            )
        return await asyncio.to_thread(self.exec, command, timeout=timeout)

    def write_file(self, path: str, content: str | bytes) -> None:
        target_path = (
            path if posixpath.isabs(path) else posixpath.join(self._work_dir_path, path)
        )
        with tempfile.NamedTemporaryFile('wb', delete=False) as tmp:
            if isinstance(content, str):
                tmp.write(content.encode('utf-8'))
            else:
                tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            self.manager.copy_to(self.sandbox_id, tmp_path, target_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def read_file(self, path: str) -> str:
        target_path = (
            path if posixpath.isabs(path) else posixpath.join(self._work_dir_path, path)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dst = Path(tmpdir) / 'downloaded_file'
            self.manager.copy_from(self.sandbox_id, target_path, local_dst)
            if not local_dst.exists():
                raise FileNotFoundError(f'File not found: {path}')
            return local_dst.read_text(encoding='utf-8', errors='replace')

    def copy_to(self, host_path: str | Path, container_path: str | Path) -> None:
        hp = Path(host_path)
        if not hp.exists():
            raise FileNotFoundError(f'Host path does not exist: {host_path}')
        target_path = (
            str(container_path)
            if posixpath.isabs(str(container_path))
            else posixpath.join(self._work_dir_path, str(container_path))
        )
        self.manager.copy_to(self.sandbox_id, hp, target_path)

    def copy_from(self, container_path: str | Path, host_path: str | Path) -> None:
        hp = Path(host_path)
        target_path = (
            str(container_path)
            if posixpath.isabs(str(container_path))
            else posixpath.join(self._work_dir_path, str(container_path))
        )
        self.manager.copy_from(self.sandbox_id, target_path, hp)

    def reset(self) -> None:
        if self._initial_files:
            for rel_path, content in self._initial_files.items():
                self.write_file(rel_path, content)

    def close(self) -> None:
        if hasattr(self.manager, 'stop'):
            self.manager.stop(self.sandbox_id)


async def sandbox_start(docker: Any) -> str:
    """Start sandbox asynchronously, supporting async managers, sync managers, and mocks."""
    if inspect.iscoroutinefunction(getattr(docker, 'start_async', None)):
        return await docker.start_async()
    start_fn = getattr(docker, 'start', None)
    if start_fn is None:
        raise AttributeError(f'{docker} has neither start_async nor start')
    if inspect.iscoroutinefunction(start_fn):
        return await start_fn()
    from unittest.mock import Mock

    if isinstance(docker, Mock) or isinstance(start_fn, Mock):
        val = start_fn()
        return await val if inspect.isawaitable(val) else val
    return await asyncio.to_thread(start_fn)


async def sandbox_stop(docker: Any, sandbox_id: str) -> None:
    """Stop sandbox asynchronously, supporting async managers, sync managers, and mocks."""
    if inspect.iscoroutinefunction(getattr(docker, 'stop_async', None)):
        await docker.stop_async(sandbox_id)
        return
    stop_fn = getattr(docker, 'stop', None)
    if stop_fn is None:
        return
    if inspect.iscoroutinefunction(stop_fn):
        await stop_fn(sandbox_id)
        return
    from unittest.mock import Mock

    if isinstance(docker, Mock) or isinstance(stop_fn, Mock):
        val = stop_fn(sandbox_id)
        if inspect.isawaitable(val):
            await val
        return
    await asyncio.to_thread(stop_fn, sandbox_id)


async def sandbox_exec(
    docker: Any, sandbox_id: str, command: str, *, timeout: int | None = None
) -> Any:
    """Execute command in sandbox asynchronously, supporting async managers, sync managers, and mocks."""
    if inspect.iscoroutinefunction(getattr(docker, 'exec_async', None)):
        return await docker.exec_async(sandbox_id, command, timeout=timeout)
    exec_fn = getattr(docker, 'exec', None)
    if exec_fn is None:
        raise AttributeError(f'{docker} has neither exec_async nor exec')
    if inspect.iscoroutinefunction(exec_fn):
        return await exec_fn(sandbox_id, command, timeout=timeout)
    from unittest.mock import Mock

    if isinstance(docker, Mock) or isinstance(exec_fn, Mock):
        val = exec_fn(sandbox_id, command, timeout=timeout)
        return await val if inspect.isawaitable(val) else val
    return await asyncio.to_thread(exec_fn, sandbox_id, command, timeout=timeout)


__all__ = [
    'AdkSandboxCodeExecutor',
    'BaseSandbox',
    'BaseSandboxManager',
    'ExecResult',
    'ExecutionResult',
    'ManagedSandbox',
    'sandbox_exec',
    'sandbox_start',
    'sandbox_stop',
]
