"""SWE-bench evaluation harness sub-modules."""

from swegemma.harness.agent_runner import build_agent_prompt, run_agent_sandbox
from swegemma.harness.container_setup import (
    ContainerSetupError,
    apply_patch_in_container,
    extract_snapshot,
    install_editable_package,
    install_test_dependencies,
    resolve_sandbox_setup_script,
    resolve_wheels_dir,
    setup_baseline_commit,
    setup_container_wheels,
    setup_container_workspace,
    setup_git_exclude,
    setup_synthetic_hardware,
    setup_workspace_test_config,
)
from swegemma.harness.verification import verify_task

__all__ = [
    'ContainerSetupError',
    'apply_patch_in_container',
    'build_agent_prompt',
    'extract_snapshot',
    'install_editable_package',
    'install_test_dependencies',
    'resolve_sandbox_setup_script',
    'resolve_wheels_dir',
    'run_agent_sandbox',
    'setup_baseline_commit',
    'setup_container_wheels',
    'setup_container_workspace',
    'setup_git_exclude',
    'setup_synthetic_hardware',
    'setup_workspace_test_config',
    'verify_task',
]
