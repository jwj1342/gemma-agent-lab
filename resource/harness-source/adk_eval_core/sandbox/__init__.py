"""Sandboxing and container management module for adk-eval-core."""

from adk_eval_core.sandbox.base import (
    AdkSandboxCodeExecutor,
    BaseSandbox,
    ExecutionResult,
)
from adk_eval_core.sandbox.containers import (
    DockerContainer,
    build_tar_stream,
    reset_tarinfo,
)
from adk_eval_core.sandbox.docker_sandbox import DockerSandbox
from adk_eval_core.sandbox.subprocess_sandbox import (
    SubprocessSandbox,
    build_sanitized_env,
    execute_subprocess_command,
)

__all__ = [
    "AdkSandboxCodeExecutor",
    "BaseSandbox",
    "DockerContainer",
    "DockerSandbox",
    "ExecutionResult",
    "SubprocessSandbox",
    "build_sanitized_env",
    "build_tar_stream",
    "execute_subprocess_command",
    "reset_tarinfo",
]
