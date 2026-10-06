"""Sandbox management interfaces and implementations for SWE-gemma."""

from swegemma.sandbox.base import (
    AdkSandboxCodeExecutor,
    BaseSandbox,
    BaseSandboxManager,
    ExecResult,
    ExecutionResult,
    ManagedSandbox,
    sandbox_exec,
    sandbox_start,
    sandbox_stop,
)
from swegemma.sandbox.docker import (
    SDG_IMAGE_CANDIDATES,
    ContainerConfig,
    ContainerManager,
    resolve_sandbox_image,
)
from swegemma.sandbox.subprocess import SubprocessManager

__all__ = [
    'SDG_IMAGE_CANDIDATES',
    'AdkSandboxCodeExecutor',
    'BaseSandbox',
    'BaseSandboxManager',
    'ContainerConfig',
    'ContainerManager',
    'ExecResult',
    'ExecutionResult',
    'ManagedSandbox',
    'SubprocessManager',
    'resolve_sandbox_image',
    'sandbox_exec',
    'sandbox_start',
    'sandbox_stop',
]

