"""Docker sandbox implementation of BaseSandbox."""

from __future__ import annotations

import logging
import posixpath
import shlex
import time
from pathlib import Path
from typing import Any

from adk_eval_core.errors import (
    EvalCoreError,
    SandboxPathTraversalError,
)
from adk_eval_core.sandbox.base import BaseSandbox, ExecutionResult
from adk_eval_core.sandbox.containers import DockerContainer

_DockerContainerCls = DockerContainer

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "gcr.io/kaggle-images/python"


class DockerSandbox(BaseSandbox):
    """Isolated Docker container sandbox."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        work_dir: str = "/work",
        environment: dict[str, str] | None = None,
        mem_limit: str = "4g",
    ) -> None:
        self._image = image
        self._work_dir = posixpath.normpath(work_dir) or "/"
        self._environment = environment or {}
        self._mem_limit = mem_limit
        self.container: DockerContainer | None = None
        self._initial_files: dict[str, str | bytes] = {}

    def initialize(self, files: dict[str, str | bytes] | None = None) -> None:
        """Start the Docker container and optionally write initial files."""
        if self.container is None:
            self.container = DockerContainer(
                image=self._image,
                work_dir=self._work_dir,
                environment=self._environment,
                mem_limit=self._mem_limit,
            )
            self.container.start()
            self.container.run_command(f"mkdir -p {shlex.quote(self._work_dir)}")

        if files is not None:
            self._initial_files = dict(files)
            self._write_files(files)

    def _write_files(self, files: dict[str, str | bytes]) -> None:
        for rel_path, content in files.items():
            self.write_file(rel_path, content)

    def _safe_path(self, path: str) -> str:
        safe_path = posixpath.normpath(posixpath.join(self._work_dir, path))
        if not (
            safe_path.startswith((self._work_dir + "/", "/tmp/"))
            or safe_path == self._work_dir
            or safe_path == "/tmp"
        ):
            raise SandboxPathTraversalError(f"Path escapes sandbox: {path!r}", path=path)
        return safe_path

    def run_command(
        self,
        command: str,
        timeout: float | None = 300,
        work_dir: str | None = None,
    ) -> ExecutionResult:
        """Run shell command inside Docker container."""
        if not self.container:
            self.initialize()

        assert self.container is not None
        eff_work_dir = self._safe_path(work_dir) if work_dir is not None else self._work_dir
        start = time.perf_counter()
        try:
            res: Any = self.container.run_command(
                command, timeout=timeout, work_dir=eff_work_dir
            )
            duration = time.perf_counter() - start
            if isinstance(res, tuple) and len(res) == 2:
                exit_code, output = res
                status = "ok" if exit_code == 0 else "error"
                return ExecutionResult(
                    status=status,
                    stdout=output if exit_code == 0 else "",
                    stderr=output if exit_code != 0 else "",
                    exit_code=exit_code,
                    duration_seconds=round(duration, 3),
                )
            else:
                exit_code, stdout_str, stderr_str = res
                status = "ok" if exit_code == 0 else "error"
                return ExecutionResult(
                    status=status,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=exit_code,
                    duration_seconds=round(duration, 3),
                )
        except TimeoutError as e:
            duration = time.perf_counter() - start
            return ExecutionResult(
                status="timeout",
                stdout=getattr(e, "stdout", None) or "",
                stderr=getattr(e, "stderr", None) or str(e),
                exit_code=124,
                duration_seconds=round(duration, 3),
            )
        except (RuntimeError, OSError, ValueError, EvalCoreError) as e:
            duration = time.perf_counter() - start
            return ExecutionResult(
                status="error",
                stdout="",
                stderr=str(e),
                exit_code=1,
                duration_seconds=round(duration, 3),
            )

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write file into container workspace."""
        if not self.container:
            self.initialize()
        assert self.container is not None
        safe = self._safe_path(path)
        self.container.write_file(safe, content)

    def read_file(self, path: str) -> str:
        """Read file from container workspace."""
        if not self.container:
            self.initialize()
        assert self.container is not None
        safe = self._safe_path(path)
        return self.container.read_file(safe)

    def copy_to(self, host_path: str | Path, container_path: str | Path) -> None:
        """Copy a file or directory tree from host into container."""
        if not self.container or self.container.container is None:
            self.initialize()
        assert self.container is not None
        raw_dst = str(container_path)
        safe_dst = self._safe_path(raw_dst)
        if raw_dst.endswith("/"):
            safe_dst = safe_dst.rstrip("/") + "/"
        if isinstance(self.container, _DockerContainerCls):
            self.container.copy_to(host_path, safe_dst)
        else:
            _DockerContainerCls.copy_to(self.container, host_path, safe_dst)

    def copy_from(self, container_path: str | Path, host_path: str | Path) -> None:
        """Copy a file or directory tree from container to host with Tar Slip protection."""
        if not self.container or self.container.container is None:
            self.initialize()
        assert self.container is not None
        safe_src = self._safe_path(str(container_path))
        if isinstance(self.container, _DockerContainerCls):
            self.container.copy_from(safe_src, host_path)
        else:
            _DockerContainerCls.copy_from(self.container, safe_src, host_path)

    def reset(self) -> None:
        """Reset container workspace to initial state."""
        if not self.container:
            return

        work_dir_quoted = shlex.quote(self._work_dir)
        self.container.run_command(f"rm -rf {work_dir_quoted} && mkdir -p {work_dir_quoted}")

        if self._initial_files:
            self._write_files(self._initial_files)

    def close(self) -> None:
        """Stop container and release resources."""
        if self.container:
            self.container.stop()
            self.container = None

    @property
    def work_dir(self) -> str:
        return self._work_dir
