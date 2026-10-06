"""Callback resolution for agents."""

from __future__ import annotations

from typing import Any

from ..context import CompilationContext
from ..registry import CallbackType
from ..schema import SandboxedLlmAgentConfig

# Maps YAML field name → (ADK kwarg name, CallbackType)
_LLM_CALLBACK_FIELDS: list[tuple[str, str, CallbackType]] = [
    ("before_agent_callbacks", "before_agent_callback", CallbackType.BEFORE_AGENT),
    ("after_agent_callbacks", "after_agent_callback", CallbackType.AFTER_AGENT),
    ("before_model_callbacks", "before_model_callback", CallbackType.BEFORE_MODEL),
    ("after_model_callbacks", "after_model_callback", CallbackType.AFTER_MODEL),
    ("before_tool_callbacks", "before_tool_callback", CallbackType.BEFORE_TOOL),
    ("after_tool_callbacks", "after_tool_callback", CallbackType.AFTER_TOOL),
]

_BASE_CALLBACK_FIELDS: list[tuple[str, str, CallbackType]] = [
    ("before_agent_callbacks", "before_agent_callback", CallbackType.BEFORE_AGENT),
    ("after_agent_callbacks", "after_agent_callback", CallbackType.AFTER_AGENT),
]


def resolve_callbacks(
    config: SandboxedLlmAgentConfig,
    ctx: CompilationContext,
) -> dict[str, Any]:
    """Resolve all LLM-specific callback fields for an LlmAgent configuration.

    Args:
        config: The sandboxed LLM agent configuration object.
        ctx: The compilation context containing the callback registry.

    Returns:
        A dictionary mapping ADK callback keyword arguments to lists of resolved callback callables.

    Raises:
        CallbackNotFoundError: If a referenced callback name is not found in the callback registry.
    """
    return resolve_callback_fields(config, ctx, _LLM_CALLBACK_FIELDS)


def resolve_base_callbacks(
    config: Any,
    ctx: CompilationContext,
) -> dict[str, Any]:
    """Resolve before/after agent callbacks for workflow agent configurations.

    Args:
        config: The sandboxed workflow agent configuration object (Sequential, Parallel, or Loop).
        ctx: The compilation context containing the callback registry.

    Returns:
        A dictionary mapping ADK callback keyword arguments to lists of resolved callback callables.

    Raises:
        CallbackNotFoundError: If a referenced callback name is not found in the callback registry.
    """
    return resolve_callback_fields(config, ctx, _BASE_CALLBACK_FIELDS)


def resolve_callback_fields(
    config: Any,
    ctx: CompilationContext,
    fields: list[tuple[str, str, CallbackType]],
) -> dict[str, Any]:
    """Resolve a specified list of YAML callback fields into ADK constructor keyword arguments.

    Args:
        config: The sandboxed agent configuration object.
        ctx: The compilation context containing the callback registry.
        fields: A list of tuples containing (yaml_field_name, adk_kwarg_name, CallbackType).

    Returns:
        A dictionary mapping ADK callback keyword arguments to lists of resolved callback callables.

    Raises:
        CallbackNotFoundError: If a referenced callback name is not found in the callback registry.
    """
    kwargs: dict[str, Any] = {}
    if ctx.callback_registry is None:
        return kwargs

    for yaml_field, adk_kwarg, cb_type in fields:
        names: list[str] | None = getattr(config, yaml_field, None)
        if names:
            resolved = [
                ctx.callback_registry.get(name, cb_type) for name in names
            ]
            kwargs[adk_kwarg] = resolved

    return kwargs


__all__ = [
    "_BASE_CALLBACK_FIELDS",
    "_LLM_CALLBACK_FIELDS",
    "resolve_base_callbacks",
    "resolve_callback_fields",
    "resolve_callbacks",
]
