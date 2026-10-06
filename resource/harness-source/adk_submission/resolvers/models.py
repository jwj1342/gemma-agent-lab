"""Model and adapter resolution for LLM agents."""

from __future__ import annotations

import copy
from typing import Any

from ..context import CompilationContext
from ..errors import AdapterNotFoundError, SubmissionValidationError
from ..schema import SandboxedLlmAgentConfig


def _clone_model(model: Any) -> Any:
    if isinstance(model, str):
        return model
    if hasattr(model, "model_copy") and callable(model.model_copy):
        try:
            return model.model_copy(deep=True)
        except Exception:  # noqa: BLE001
            pass
    cloned = copy.copy(model)
    if hasattr(model, "_additional_args"):
        cloned._additional_args = copy.deepcopy(model._additional_args)
    return cloned


def resolve_model(
    config: SandboxedLlmAgentConfig,
    ctx: CompilationContext,
) -> Any:
    """Resolve the model for an LLM agent, attaching an adapter if specified.

    Args:
        config: The sandboxed LLM agent configuration object.
        ctx: The compilation context tracking mutable state and registries.

    Returns:
        The resolved model object or string identifier, or None if neither
        model nor adapter is specified.

    Raises:
        AdapterNotFoundError: If the specified adapter is not found in discovered adapters.
        SubmissionValidationError: If an adapter is specified without a base model.
        ModelNotFoundError: If the specified base model alias is not in the model registry.
    """
    if config.adapter:
        if not ctx.adapter_manifest or config.adapter not in ctx.adapter_manifest.adapters:
            available = (
                sorted(ctx.adapter_manifest.adapters.keys())
                if ctx.adapter_manifest
                else []
            )
            raise AdapterNotFoundError(config.adapter, available=available)
        if not config.model:
            raise SubmissionValidationError(
                f"Agent '{config.name}' specifies adapter '{config.adapter}' "
                f"without a base model alias."
            )
        base_model = ctx.model_registry.get(config.model)
        adapter_info = ctx.adapter_manifest.adapters[config.adapter]
        if ctx.adapter_resolver_fn is not None:
            return ctx.adapter_resolver_fn(base_model, adapter_info)
        prefixed_key = f"adapter:{adapter_info.name}"
        if prefixed_key in ctx.model_registry:
            registered = ctx.model_registry.get(prefixed_key)
            return _clone_model(registered)
        if adapter_info.name in ctx.model_registry:
            registered = ctx.model_registry.get(adapter_info.name)
            return _clone_model(registered)
        if isinstance(base_model, str):
            return f"{base_model}:{adapter_info.name}"
        if hasattr(base_model, "model") and isinstance(base_model.model, str):
            cloned = _clone_model(base_model)
            cloned.model = f"{base_model.model}:{adapter_info.name}"
            return cloned
        return f"{base_model}:{adapter_info.name}"
    elif config.model:
        base_model = ctx.model_registry.get(config.model)
        return _clone_model(base_model)
    return None


__all__ = ["resolve_model"]
