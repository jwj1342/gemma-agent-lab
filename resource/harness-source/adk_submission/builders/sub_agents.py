"""Sub-agent compilation for nested agent architectures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.agents.base_agent import BaseAgent

from ..context import CompilationContext
from ..errors import LimitExceededError
from ..paths import handle_schema_validation_error
from ..schema import SandboxedAgentConfig
from ..yaml_loader import load_yaml


def compile_sub_agents(
    refs: list[Any] | None,
    ctx: CompilationContext,
    compile_agent_fn: Callable[[Any, CompilationContext], BaseAgent] | None = None,
) -> list[BaseAgent]:
    """Recursively compile a list of sub-agent configuration references.

    Args:
        refs: A list of sub-agent reference objects containing config_path attributes, or None.
        ctx: The compilation context tracking mutable state, recursion depth, and limits.
        compile_agent_fn: Optional callable for recursive agent compilation.

    Returns:
        A list of compiled ADK BaseAgent instances.

    Raises:
        LimitExceededError: If the sub-agent nesting depth exceeds ``ctx.limits.max_sub_agent_depth``.
        PathTraversalError: If a sub-agent config_path escapes the submission root directory.
        SubmissionValidationError: If a sub-agent configuration file is not found.
        SubmissionSchemaError: If a sub-agent configuration fails schema validation.
        yaml.YAMLError: If a sub-agent YAML configuration file is malformed.
    """
    if not refs:
        return []

    ctx.depth += 1
    try:
        if ctx.depth > ctx.limits.max_sub_agent_depth:
            raise LimitExceededError(
                f"Sub-agent nesting too deep: {ctx.depth} "
                f"(max {ctx.limits.max_sub_agent_depth})"
            )

        if compile_agent_fn is None:
            from ..compiler import _compile_agent
            compile_agent_fn = _compile_agent

        agents: list[BaseAgent] = []
        for ref in refs:
            current_dir = ctx.current_dir or ctx.root_dir
            sub_path = ctx.resolve_sub_config_path(ref.config_path, error_prefix="Sub-agent")

            raw = load_yaml(sub_path, ctx.root_dir, limits=ctx.limits)
            try:
                sub_config = SandboxedAgentConfig.model_validate(raw)
            except Exception as e:  # noqa: BLE001
                handle_schema_validation_error(
                    e, context_desc=f"sub-agent config ({ref.config_path})"
                )
            prev_dir = current_dir
            ctx.current_dir = sub_path.parent
            try:
                agents.append(compile_agent_fn(sub_config.root, ctx))
            finally:
                ctx.current_dir = prev_dir

        return agents
    finally:
        ctx.depth -= 1


__all__ = ["compile_sub_agents"]
