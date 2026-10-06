"""General utility functions for ADK evaluation framework."""

from __future__ import annotations

import json
import logging
from typing import Any

from adk_eval_core.budget.budget import PricingTable

logger = logging.getLogger(__name__)


def unwrap_tool_response(fr: Any) -> dict[str, Any] | None:
    """Extract a parsed dict from an ADK FunctionResponse, raw dict, JSON string, or object with to_dict."""
    if fr is None:
        return None

    is_wrapped = hasattr(fr, "response")
    resp = getattr(fr, "response", fr)
    if resp is None:
        return None

    if isinstance(resp, dict):
        if "result" not in resp:
            return dict(resp)
        inner = resp["result"]
        siblings = {k: v for k, v in resp.items() if k != "result"}
        if isinstance(inner, str):
            try:
                parsed = json.loads(inner)
                inner_dict = parsed if isinstance(parsed, dict) else {"result": parsed}
            except (json.JSONDecodeError, TypeError):
                inner_dict = {"raw": inner}
            if siblings:
                merged = dict(siblings)
                merged.update(inner_dict)
                return merged
            return inner_dict
        if isinstance(inner, dict):
            if siblings:
                merged = dict(siblings)
                merged.update(inner)
                return merged
            return dict(inner)
        return dict(resp)

    if isinstance(resp, str):
        try:
            parsed = json.loads(resp)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except (json.JSONDecodeError, TypeError):
            return {"raw": resp}

    if hasattr(resp, "to_dict") and callable(resp.to_dict):
        try:
            d = resp.to_dict()
            if isinstance(d, dict):
                return d
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to unwrap response via to_dict: %s", e)

    if is_wrapped and isinstance(resp, (list, int, float, bool)):
        return {"result": resp}

    return None


def setup_model_registry(pricing_table: PricingTable | None = None) -> Any:
    """Populate ADK ModelRegistry / LLMRegistry with models based on PricingTable."""
    registry: Any
    try:
        from google.adk.models import LLMRegistry
        registry = LLMRegistry()
    except ImportError:
        try:
            from google.adk.models.model_registry import (  # type: ignore[import-not-found]
                ModelRegistry,
            )
            registry = ModelRegistry()
        except ImportError:
            logger.warning("Neither LLMRegistry nor ModelRegistry found in google.adk.models")
            return {}

    if pricing_table is None:
        pricing_table = PricingTable.from_yaml()

    registered_models: dict[str, Any] = {}

    try:
        from google.adk.models.lite_llm import LiteLlm as _LiteLlmCls
        has_litellm = True
    except ImportError:
        from dataclasses import dataclass

        @dataclass
        class _FallbackLiteLlm:
            model: str

        _LiteLlmCls = _FallbackLiteLlm  # type: ignore[assignment]
        has_litellm = False

    if has_litellm and hasattr(registry, "register"):
        try:
            registry.register(_LiteLlmCls)
        except TypeError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not register LiteLlm class on registry: %s", e)

    for model_id in pricing_table.model_ids:
        try:
            pricing = pricing_table.get(model_id) if hasattr(pricing_table, "get") else None
            model_path = (pricing.path if pricing and getattr(pricing, "path", None) else model_id) or model_id
            model_instance = _LiteLlmCls(model=model_path)
            registered_models[model_id] = model_instance
            if model_path != model_id:
                registered_models[model_path] = model_instance

            if hasattr(registry, "register_model"):
                registry.register_model(model_id, model_instance)
            elif hasattr(registry, "register"):
                try:
                    registry.register(model_id, model_instance)
                except TypeError:
                    pass
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not auto-register model %s in registry: %s", model_id, e)

    if isinstance(registry, dict):
        registry.update(registered_models)
    else:
        registry.registered_models = registered_models

    return registry
