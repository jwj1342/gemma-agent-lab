"""Abstract base interfaces and code executor integration for sandboxing."""

from __future__ import annotations

import posixpath
import shlex
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self

from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import (
    CodeExecutionInput,
    CodeExecutionResult,
)
from pydantic import ConfigDict
from typing_extensions import override


@dataclass
class ExecutionResult:
    """Result of a command execution in the sandbox."""

    status: Literal["ok", "error", "timeout"]
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_seconds: float = 0.0

    @property
    def timed_out(self) -> bool:
        """Backwards-compatible alias for checking if the command timed out."""
        return self.status == "timeout"

    def to_json(self, max_stdout_chars: int = 5000) -> str:
        """Serialize to JSON string for the agent."""
        data = asdict(self)
        if data["stdout"] and len(data["stdout"]) > max_stdout_chars:
            data["stdout"] = data["stdout"][:max_stdout_chars] + "... [truncated]"
        if data["stderr"] and len(data["stderr"]) > max_stdout_chars:
            data["stderr"] = data["stderr"][:max_stdout_chars] + "... [truncated]"
        import json
        filtered = {k: v for k, v in data.items() if v is not None and v != ""}
        return json.dumps(filtered)


class BaseSandbox(ABC):
    """Abstract Base Class for an isolated execution sandbox environment."""

    @abstractmethod
    def initialize(self, files: dict[str, str | bytes] | None = None) -> None:
        """Initialize the sandbox environment and optionally seed initial files.

        Args:
            files: Optional mapping of relative file paths to content (str or bytes)
                to write into the sandbox workspace upon initialization and restore
                upon reset.
        """

    @abstractmethod
    def run_command(
        self,
        command: str,
        timeout: float | None = 300,
        work_dir: str | None = None,
    ) -> ExecutionResult:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Max execution duration in seconds (defaults to 300).
            work_dir: Working directory inside sandbox. If None, uses self.work_dir.

        Returns:
            ExecutionResult containing status ("ok", "error", "timeout"),
            stdout, stderr, exit_code, and duration_seconds.
        """

    @abstractmethod
    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a single file into the sandbox workspace.

        Args:
            path: Relative or absolute path inside the sandbox.
            content: File content as UTF-8 string or raw bytes.
        """

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read a text file from the sandbox workspace.

        Args:
            path: Relative or absolute path inside the sandbox.

        Returns:
            Decoded UTF-8 string content.

        Raises:
            FileNotFoundError: If the file does not exist.
        """

    @abstractmethod
    def copy_to(self, host_path: str | Path, container_path: str | Path) -> None:
        """Copy a file or directory tree from the host into the sandbox.

        Args:
            host_path: Path on the host filesystem (must exist).
            container_path: Destination path inside the sandbox workspace.

        Raises:
            FileNotFoundError: If host_path does not exist.
            ValueError: If container_path escapes the sandbox workspace.
        """

    @abstractmethod
    def copy_from(self, container_path: str | Path, host_path: str | Path) -> None:
        """Copy a file or directory tree from the sandbox to the host.

        Args:
            container_path: Source path inside the sandbox (must exist).
            host_path: Destination path on the host filesystem.

        Raises:
            FileNotFoundError: If container_path does not exist in the sandbox.
            ValueError: If path traversal or tar slip attack is detected.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the sandbox workspace to its initial state.

        Clears all modified/created files and restores any initial files
        supplied during initialize().
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up and release all sandbox resources."""

    @property
    @abstractmethod
    def work_dir(self) -> str:
        """Return the root working directory inside the sandbox."""

    def __enter__(self) -> Self:
        """Context manager entry point: initializes the sandbox."""
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit point: releases resources via close()."""
        self.close()


class AdkSandboxCodeExecutor(BaseCodeExecutor):
    """Executes ADK code blocks and scripts inside a BaseSandbox instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sandbox: Any
    timeout_seconds: int | None = None
    start_time: float | None = None
    max_time_minutes: float | None = None
    budget_check_fn: Callable[[], str | None] | None = None

    @override
    def execute_code(
        self, invocation_context: Any, code_execution_input: CodeExecutionInput
    ) -> CodeExecutionResult:
        if self.budget_check_fn is not None:
            budget_err = self.budget_check_fn()
            if budget_err is not None:
                return CodeExecutionResult(
                    stdout="",
                    stderr=budget_err,
                    output_files=[],
                )
        fallback_timeout = self.timeout_seconds if self.timeout_seconds is not None else 300
        if self.start_time is not None and self.max_time_minutes is not None:
            elapsed = time.perf_counter() - self.start_time
            remaining = (self.max_time_minutes * 60) - elapsed
            if remaining <= 0:
                return CodeExecutionResult(
                    stdout="",
                    stderr="Execution time budget exceeded.",
                    output_files=[],
                )
            timeout = max(1, int(min(fallback_timeout, remaining)))
        else:
            timeout = fallback_timeout

        temp_filename = f".adk_exec_{uuid.uuid4().hex[:8]}.py"
        target_path = posixpath.join(self.sandbox.work_dir, temp_filename)
        self.sandbox.write_file(temp_filename, code_execution_input.code)

        wrapped_cmd = f"python3 {shlex.quote(target_path)}"

        try:
            res = self.sandbox.run_command(wrapped_cmd, timeout=timeout)
        finally:
            self.sandbox.run_command(f"rm -f {shlex.quote(target_path)}")

        stderr = res.stderr
        if not stderr:
            if res.status == "timeout":
                stderr = f"Command timed out after {timeout}s (exit_code={res.exit_code})"
            elif res.status != "ok" or (res.exit_code is not None and res.exit_code != 0):
                stderr = f"Command failed with status={res.status}, exit_code={res.exit_code}"

        return CodeExecutionResult(
            stdout=res.stdout,
            stderr=stderr,
            output_files=[],
        )
