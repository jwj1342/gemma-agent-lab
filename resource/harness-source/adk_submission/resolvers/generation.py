"""Generation configuration resolution and validation."""

from __future__ import annotations

from typing import Any

from google.genai import types as genai_types

from ..context import CompilationContext
from ..schema import SandboxedLlmAgentConfig


def _get_genai_types(ctx: CompilationContext | None = None) -> Any:
    if ctx is not None and ctx.agent_factory and ctx.agent_factory.genai_types is not None:
        return ctx.agent_factory.genai_types
    return genai_types


def resolve_generation_config(
    config: SandboxedLlmAgentConfig,
    ctx: CompilationContext,
) -> Any:
    """Resolve and validate the LLM generation configuration against organizer constraints.

    Args:
        config: The sandboxed LLM agent configuration object.
        ctx: The compilation context containing organizer-defined generation constraints.

    Returns:
        A GenerateContentConfig instance if configuration or defaults exist, otherwise None.

    Raises:
        SubmissionValidationError: If the generation configuration violates organizer-defined constraints.
    """
    types_mod = _get_genai_types(ctx)
    if config.generate_content_config is not None:
        config_dict = config.generate_content_config.model_dump(exclude_none=True)

        if ctx.generation_constraints is not None:
            config_dict = ctx.generation_constraints.validate_config(
                config_dict, config.name
            )

        if config_dict:
            return types_mod.GenerateContentConfig(**config_dict)
        return None

    # No user config — apply organizer defaults if any
    if (
        ctx.generation_constraints is not None
        and ctx.generation_constraints.defaults
    ):
        validated = ctx.generation_constraints.validate_config({}, config.name)
        return types_mod.GenerateContentConfig(
            **validated
        )

    return None


import functools
import inspect


def _normalize_reasoning_message(msg: Any) -> Any:
    """Ensure assistant messages with reasoning_content also populate reasoning for vLLM."""
    if isinstance(msg, dict):
        rc = msg.get("reasoning_content")
        if rc is not None and msg.get("reasoning") is None:
            msg["reasoning"] = rc
    elif isinstance(msg, list):
        for item in msg:
            if isinstance(item, dict):
                rc = item.get("reasoning_content")
                if rc is not None and item.get("reasoning") is None:
                    item["reasoning"] = rc
    return msg


def _normalize_schema_dict(schema_dict: Any, schema_to_dict_fn: Any = None) -> Any:
    """Flatten nullable any_of branches so top-level 'type' is populated for chat templates."""
    if not isinstance(schema_dict, dict):
        return schema_dict

    any_of = schema_dict.get("any_of")
    if isinstance(any_of, list):
        normalized_branches: list[dict[str, Any]] = []
        for branch in any_of:
            if isinstance(branch, dict):
                b_dict = (
                    schema_to_dict_fn(branch)
                    if callable(schema_to_dict_fn)
                    else dict(branch)
                )
                if "type" in b_dict and b_dict["type"] is not None:
                    t = b_dict["type"]
                    b_dict["type"] = (
                        t.value if hasattr(t, "value") else str(t)
                    ).lower()
                normalized_branches.append(b_dict)
            elif hasattr(branch, "model_dump") and callable(schema_to_dict_fn):
                normalized_branches.append(schema_to_dict_fn(branch))

        null_branches = [
            b for b in normalized_branches if b.get("type") == "null"
        ]
        non_null_branches = [
            b for b in normalized_branches if b.get("type") != "null"
        ]
        if null_branches:
            schema_dict["nullable"] = True

        if len(non_null_branches) == 1 and not schema_dict.get("type"):
            schema_dict.pop("any_of", None)
            for k, v in non_null_branches[0].items():
                schema_dict.setdefault(k, v)
        elif len(non_null_branches) > 1:
            schema_dict["any_of"] = normalized_branches
            first_type = non_null_branches[0].get("type")
            if first_type and not schema_dict.get("type"):
                schema_dict["type"] = first_type
        elif not non_null_branches and null_branches and not schema_dict.get("type"):
            schema_dict.pop("any_of", None)
            schema_dict["type"] = "null"

    return schema_dict


def install_litellm_reasoning_patch() -> bool:
    """Patch ADK LiteLlm message and tool schema serialization for vLLM and Gemma 4.

    1. Mirrors ``reasoning_content`` into ``reasoning`` on assistant messages so
       vLLM<=0.21.x preserves prior thoughts across multi-step tool calls.
    2. Flattens ``any_of: [{type: T}, {type: NULL}]`` in ``_schema_to_dict`` so
       optional tool parameters (e.g. ``int | None``) retain a top-level ``type``
       key required by Gemma 4's ``chat_template.jinja``.
    """
    try:
        from google.adk.models import lite_llm
    except ImportError:
        return False

    patched = False

    orig_schema_to_dict = getattr(lite_llm, "_schema_to_dict", None)
    if callable(orig_schema_to_dict) and not getattr(
        orig_schema_to_dict, "_adk_submission_schema_patched", False
    ):

        @functools.wraps(orig_schema_to_dict)
        def _patched_schema_to_dict(schema: Any) -> dict[str, Any]:
            res = orig_schema_to_dict(schema)
            return _normalize_schema_dict(res, _patched_schema_to_dict)

        _patched_schema_to_dict._adk_submission_schema_patched = True  # type: ignore[attr-defined]
        lite_llm._schema_to_dict = _patched_schema_to_dict
        patched = True

    orig_content_to_msg = getattr(lite_llm, "_content_to_message_param", None)
    if callable(orig_content_to_msg) and not getattr(
        orig_content_to_msg, "_adk_submission_reasoning_patched", False
    ):
        if inspect.iscoroutinefunction(orig_content_to_msg):

            @functools.wraps(orig_content_to_msg)
            async def _patched_content_to_message_param(*args: Any, **kwargs: Any) -> Any:
                res = await orig_content_to_msg(*args, **kwargs)
                return _normalize_reasoning_message(res)

        else:

            @functools.wraps(orig_content_to_msg)
            def _patched_content_to_message_param(*args: Any, **kwargs: Any) -> Any:
                res = orig_content_to_msg(*args, **kwargs)
                return _normalize_reasoning_message(res)

        _patched_content_to_message_param._adk_submission_reasoning_patched = True  # type: ignore[attr-defined]
        lite_llm._content_to_message_param = _patched_content_to_message_param
        patched = True

    orig_ensure_tools = getattr(lite_llm, "_ensure_tool_results", None)
    if callable(orig_ensure_tools) and not getattr(
        orig_ensure_tools, "_adk_submission_reasoning_patched", False
    ):

        @functools.wraps(orig_ensure_tools)
        def _patched_ensure_tool_results(messages: Any, *args: Any, **kwargs: Any) -> Any:
            res = orig_ensure_tools(messages, *args, **kwargs)
            _normalize_reasoning_message(messages)
            return res

        _patched_ensure_tool_results._adk_submission_reasoning_patched = True  # type: ignore[attr-defined]
        lite_llm._ensure_tool_results = _patched_ensure_tool_results
        patched = True

    orig_get_inputs = getattr(lite_llm, "_get_completion_inputs", None)
    if callable(orig_get_inputs) and not getattr(
        orig_get_inputs, "_adk_submission_reasoning_patched", False
    ):
        def _enrich_completion_inputs(res: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if not isinstance(res, tuple) or len(res) == 0:
                return res
            _normalize_reasoning_message(res[0])
            llm_request = args[0] if args else kwargs.get("llm_request")
            req_cfg = getattr(llm_request, "config", None) if llm_request is not None else None
            seed = getattr(req_cfg, "seed", None) if req_cfg is not None else None
            if seed is not None and not isinstance(seed, bool) and len(res) >= 4:
                gen_params = dict(res[3]) if isinstance(res[3], dict) else {}
                gen_params.setdefault("seed", int(seed))
                return (res[0], res[1], res[2], gen_params, *res[4:])
            return res

        if inspect.iscoroutinefunction(orig_get_inputs):

            @functools.wraps(orig_get_inputs)
            async def _patched_get_completion_inputs(*args: Any, **kwargs: Any) -> Any:
                res = await orig_get_inputs(*args, **kwargs)
                return _enrich_completion_inputs(res, args, kwargs)

        else:

            @functools.wraps(orig_get_inputs)
            def _patched_get_completion_inputs(*args: Any, **kwargs: Any) -> Any:
                res = orig_get_inputs(*args, **kwargs)
                return _enrich_completion_inputs(res, args, kwargs)

        _patched_get_completion_inputs._adk_submission_reasoning_patched = True  # type: ignore[attr-defined]
        lite_llm._get_completion_inputs = _patched_get_completion_inputs
        patched = True

    return patched or bool(
        getattr(
            getattr(lite_llm, "_content_to_message_param", None),
            "_adk_submission_reasoning_patched",
            False,
        )
        and getattr(
            getattr(lite_llm, "_schema_to_dict", None),
            "_adk_submission_schema_patched",
            False,
        )
    )


install_litellm_reasoning_patch()


def apply_thinking_config_to_model(model: Any, gen_config: Any) -> Any:
    """Attach thinking_config and seed from GenerateContentConfig to LiteLlm._additional_args.

    Because ADK's LiteLlm._get_completion_inputs() does not forward Google GenAI
    thinking_config or seed to OpenAI-compatible endpoints (like vLLM), this bridges
    thinking_level, thinking_budget, include_thoughts, and seed into
    extra_body.chat_template_kwargs.enable_thinking, extra_body.thinking_token_budget,
    reasoning_effort, and seed on a cloned model instance.
    """
    install_litellm_reasoning_patch()

    if model is None or gen_config is None:
        return model

    if not hasattr(model, "_additional_args") or not isinstance(model._additional_args, dict):
        return model

    if isinstance(gen_config, dict):
        seed = gen_config.get("seed")
        thinking_cfg = gen_config.get("thinking_config")
    else:
        seed = getattr(gen_config, "seed", None)
        thinking_cfg = getattr(gen_config, "thinking_config", None)

    has_valid_seed = seed is not None and not isinstance(seed, bool)

    enable_thinking: bool | None = None
    level_str: str | None = None
    thinking_budget: int | float | None = None

    if thinking_cfg is not None:
        if isinstance(thinking_cfg, dict):
            include_thoughts = thinking_cfg.get("include_thoughts")
            raw_level = thinking_cfg.get("thinking_level")
            thinking_budget = thinking_cfg.get("thinking_budget")
        else:
            include_thoughts = getattr(thinking_cfg, "include_thoughts", None)
            raw_level = getattr(thinking_cfg, "thinking_level", None)
            thinking_budget = getattr(thinking_cfg, "thinking_budget", None)

        if raw_level is not None:
            level_str = str(getattr(raw_level, "value", raw_level)).lower()

        if (
            include_thoughts is False
            or level_str == "none"
            or (
                thinking_budget is not None
                and not isinstance(thinking_budget, bool)
                and int(thinking_budget) <= 0
            )
        ):
            enable_thinking = False
        elif (
            include_thoughts is True
            or level_str in {"minimal", "low", "medium", "high"}
            or (
                thinking_budget is not None
                and not isinstance(thinking_budget, bool)
                and int(thinking_budget) > 0
            )
        ):
            enable_thinking = True

    if enable_thinking is None and not has_valid_seed:
        return model

    import copy

    cloned = copy.copy(model)
    additional_args = copy.deepcopy(model._additional_args)

    if has_valid_seed:
        additional_args["seed"] = int(seed)  # type: ignore[arg-type]

    if enable_thinking is not None:
        extra_body = dict(additional_args.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = enable_thinking
        extra_body["chat_template_kwargs"] = chat_template_kwargs

        if (
            enable_thinking
            and thinking_budget is not None
            and not isinstance(thinking_budget, bool)
            and int(thinking_budget) > 0
        ):
            extra_body["thinking_token_budget"] = int(thinking_budget)
        else:
            extra_body.pop("thinking_token_budget", None)

        additional_args["extra_body"] = extra_body

        if enable_thinking and level_str in {"low", "medium", "high"}:
            additional_args["reasoning_effort"] = level_str
        elif not enable_thinking:
            additional_args.pop("reasoning_effort", None)

    cloned._additional_args = additional_args
    return cloned


__all__ = [
    "_get_genai_types",
    "apply_thinking_config_to_model",
    "install_litellm_reasoning_patch",
    "resolve_generation_config",
]
