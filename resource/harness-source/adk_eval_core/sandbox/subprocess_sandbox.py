"""Subprocess sandbox implementation of BaseSandbox for local non-containerized execution."""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import time

from adk_eval_core.errors import (
    EvalCoreError,
    SandboxFileNotFoundError,
    SandboxPathTraversalError,
)
from adk_eval_core.sandbox.base import BaseSandbox, ExecutionResult

logger = logging.getLogger(__name__)


_SAFE_ENV_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TERM",
    "USER",
    "LOGNAME",
    "SYSTEMROOT",
    "LD_LIBRARY_PATH",
    "TMPDIR",
    "TEST_TMPDIR",
)


def build_sanitized_env(
    work_dir: pathlib.Path,
    tmp_dir: pathlib.Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a sanitized environment dictionary from _SAFE_ENV_KEYS."""
    eff_tmp = tmp_dir if tmp_dir is not None else work_dir
    sanitized_env: dict[str, str] = {
        k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ
    }
    sanitized_env["HOME"] = str(eff_tmp)
    sanitized_env["TMPDIR"] = str(eff_tmp)
    sanitized_env["TEST_TMPDIR"] = str(eff_tmp)
    sanitized_env["PYTHONNOUSERSITE"] = "1"
    if extra_env:
        sanitized_env.update(extra_env)
    return sanitized_env


def execute_subprocess_command(
    command: str | list[str],
    cwd: pathlib.Path,
    env: dict[str, str],
    timeout: float | None = 300,
    *,
    shell: bool | None = None,
    max_output_chars: int = 10 * 1024 * 1024,
) -> ExecutionResult:
    """Execute a command in a new process group with timeout and SIGKILL cleanup."""
    if shell is None:
        shell = isinstance(command, str)
    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            command,
            shell=shell,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except (RuntimeError, OSError, ValueError, EvalCoreError) as e:
        duration = time.perf_counter() - start
        return ExecutionResult(
            status="error",
            stdout="",
            stderr=str(e),
            exit_code=-1,
            duration_seconds=round(duration, 3),
        )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        duration = time.perf_counter() - start
        stdout = stdout or ""
        stderr = stderr or ""
        if max_output_chars and len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars] + "\n...[Output truncated at 10MB limit]\n"
        if max_output_chars and len(stderr) > max_output_chars:
            stderr = stderr[:max_output_chars] + "\n...[Output truncated at 10MB limit]\n"
        status = "ok" if proc.returncode == 0 else "error"
        return ExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            duration_seconds=round(duration, 3),
        )
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except (subprocess.TimeoutExpired, Exception):  # noqa: BLE001
            stdout, stderr = "", ""
        duration = time.perf_counter() - start
        return ExecutionResult(
            status="timeout",
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=124,
            duration_seconds=round(duration, 3),
        )
    except (RuntimeError, OSError, ValueError, EvalCoreError) as e:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        duration = time.perf_counter() - start
        return ExecutionResult(
            status="error",
            stdout="",
            stderr=str(e),
            exit_code=1,
            duration_seconds=round(duration, 3),
        )
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)


class SubprocessSandbox(BaseSandbox):
    """Local process sandbox operating within a host directory."""

    def __init__(
        self,
        work_dir: str | pathlib.Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        if work_dir is None:
            import tempfile
            self._temp_dir = tempfile.TemporaryDirectory(prefix="adk_sandbox_")
            self._work_dir = pathlib.Path(self._temp_dir.name).resolve()
        else:
            self._temp_dir = None
            self._work_dir = pathlib.Path(work_dir).resolve()
            self._work_dir.mkdir(parents=True, exist_ok=True)
        self._environment: dict[str, str] = dict(environment) if environment else {}
        self._initial_files: dict[str, str | bytes] = {}

    def initialize(self, files: dict[str, str | bytes] | None = None) -> None:
        """Ensure working directory exists and write initial files."""
        self._work_dir.mkdir(parents=True, exist_ok=True)
        if files is not None:
            self._initial_files = dict(files)
            self._write_files(files)

    def _write_files(self, files: dict[str, str | bytes]) -> None:
        for rel_path, content in files.items():
            self.write_file(rel_path, content)

    def run_command(
        self,
        command: str,
        timeout: float | None = 300,
        work_dir: str | None = None,
    ) -> ExecutionResult:
        """Execute a shell command via subprocess."""
        target_dir = self._resolve_sandbox_path(work_dir) if work_dir else self._work_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        sanitized_env = build_sanitized_env(
            work_dir=self._work_dir,
            extra_env=self._environment,
        )
        return execute_subprocess_command(
            command=command,
            cwd=target_dir,
            env=sanitized_env,
            timeout=timeout,
            shell=True,
        )

    def _resolve_sandbox_path(self, path: str | pathlib.Path) -> pathlib.Path:
        p = pathlib.Path(path)
        if not p.is_absolute():
            p = self._work_dir / p
        resolved = p.resolve()
        if not resolved.is_relative_to(self._work_dir.resolve()):
            raise SandboxPathTraversalError(f"Path escapes sandbox: {path!r}", path=str(path))
        return resolved

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write file to local workspace."""
        full_path = self._resolve_sandbox_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            full_path.write_text(content, encoding="utf-8")
        else:
            full_path.write_bytes(content)

    def read_file(self, path: str) -> str:
        """Read text file from local workspace."""
        full_path = self._resolve_sandbox_path(path)
        if not full_path.is_file():
            raise SandboxFileNotFoundError(f"File not found: {path}", path=str(path))
        return full_path.read_text(encoding="utf-8")

    def copy_to(self, host_path: str | pathlib.Path, container_path: str | pathlib.Path) -> None:
        """Copy a file or directory from the host into the sandbox."""
        host_p = pathlib.Path(host_path).resolve()
        if not host_p.exists():
            raise SandboxFileNotFoundError(f"Host path not found: {host_path}", path=str(host_path))

        target_p = self._resolve_sandbox_path(container_path)
        if host_p.is_file():
            if str(container_path).endswith(("/", "\\")) or target_p.is_dir():
                target_p.mkdir(parents=True, exist_ok=True)
                target_file = target_p / host_p.name
            else:
                target_p.parent.mkdir(parents=True, exist_ok=True)
                target_file = target_p
            shutil.copy2(host_p, target_file)
        elif host_p.is_dir():
            target_p.mkdir(parents=True, exist_ok=True)
            shutil.copytree(host_p, target_p, dirs_exist_ok=True)

    def copy_from(self, container_path: str | pathlib.Path, host_path: str | pathlib.Path) -> None:
        """Copy a file or directory from the sandbox to the host."""
        src_p = self._resolve_sandbox_path(container_path)
        if not src_p.exists():
            raise SandboxFileNotFoundError(f"Sandbox path not found: {container_path}", path=str(container_path))

        host_p = pathlib.Path(host_path).resolve()
        if src_p.is_file():
            if str(host_path).endswith(("/", "\\")) or host_p.is_dir():
                host_p.mkdir(parents=True, exist_ok=True)
                target_file = host_p / src_p.name
            else:
                host_p.parent.mkdir(parents=True, exist_ok=True)
                target_file = host_p
            shutil.copy2(src_p, target_file)
        elif src_p.is_dir():
            src_resolved = src_p.resolve()
            for root, dirs, files in os.walk(src_p, followlinks=False):
                root_p = pathlib.Path(root)
                for name in dirs + files:
                    item = root_p / name
                    if item.is_symlink():
                        target_resolved = item.resolve()
                        if not target_resolved.is_relative_to(src_resolved):
                            raise SandboxPathTraversalError(
                                f"Path escapes copied directory via symlink: {item}",
                                path=str(item),
                            )
                        if target_resolved.is_dir() and root_p.resolve().is_relative_to(target_resolved):
                            raise SandboxPathTraversalError(
                                f"Recursive directory symlink detected: {item} -> {target_resolved}",
                                path=str(item),
                            )
                    elif not item.resolve().is_relative_to(src_resolved):
                        raise SandboxPathTraversalError(
                            f"Path escapes sandbox via symlink: {item}", path=str(item)
                        )
            host_p.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_p, host_p, symlinks=True, dirs_exist_ok=True)

    def reset(self) -> None:
        """Reset sandbox workspace to its initial state."""
        if self._work_dir.exists():
            for item in self._work_dir.iterdir():
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            self._work_dir.mkdir(parents=True, exist_ok=True)

        if self._initial_files:
            self._write_files(self._initial_files)

    def close(self) -> None:
        """Cleanup temporary directory if created."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    @property
    def work_dir(self) -> str:
        return str(self._work_dir)
