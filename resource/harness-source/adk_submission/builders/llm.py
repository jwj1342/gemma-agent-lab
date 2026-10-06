"""LLM agent builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.base_agent import BaseAgent

from ..context import (
    CompilationContext,
    check_description_length,
    check_instruction_length,
)
from ..resolvers.callbacks import resolve_callbacks
from ..resolvers.generation import (
    apply_thinking_config_to_model,
    resolve_generation_config,
)
from ..resolvers.models import resolve_model
from ..resolvers.tools import resolve_skills, resolve_tool_item
from ..schema import SandboxedLlmAgentConfig
from .sub_agents import compile_sub_agents


def _get_agent_cls(ctx: CompilationContext | None = None) -> Any:
    if ctx is not None and ctx.agent_factory and ctx.agent_factory.agent_cls is not None:
        return ctx.agent_factory.agent_cls
    return Agent


def compile_llm_agent(
    config: SandboxedLlmAgentConfig,
    ctx: CompilationContext,
    compile_agent_fn: Callable[[Any, CompilationContext], BaseAgent] | None = None,
) -> Agent:
    """Compile a sandboxed LLM agent configuration into an ADK Agent.

    Resolves tools, skills (via SkillToolset), models, callbacks, sub-agents,
    and generation configurations while enforcing instruction, description, and skill limits.

    Args:
        config: The sandboxed LLM agent configuration object.
        ctx: The compilation context tracking mutable state and limits.
        compile_agent_fn: Optional callable for recursive agent compilation.

    Returns:
        A fully configured ADK Agent instance.

    Raises:
        LimitExceededError: If instruction length, description length, or cumulative skill count limits are exceeded.
        PathTraversalError: If a referenced skill path escapes the submission root directory.
        SubmissionValidationError: If a skill directory is missing or fails to load.
        ToolNotFoundError: If a referenced tool is not found in the tool registry.
        ModelNotFoundError: If a referenced model alias is not found in the model registry.
        CallbackNotFoundError: If a referenced callback is not found in the callback registry.
    """
    check_instruction_length(config.instruction, config.name, ctx)
    check_description_length(config.description, config.name, ctx)

    tools = [
        resolve_tool_item(item, ctx, compile_agent_fn)
        for item in (config.tools or [])
    ]

    skill_toolset = resolve_skills(config, ctx, tools=tools)
    if skill_toolset is not None:
        tools.append(skill_toolset)

    model = resolve_model(config, ctx)
    callback_kwargs = resolve_callbacks(config, ctx)
    sub_agents = compile_sub_agents(config.sub_agents, ctx, compile_agent_fn)
    gen_config = resolve_generation_config(config, ctx)
    model = apply_thinking_config_to_model(model, gen_config)

    kwargs: dict[str, Any] = dict(
        name=config.name,
        description=config.description or "",
        model=model,
        instruction=config.instruction,
        tools=tools,
        sub_agents=sub_agents,
        output_key=config.output_key,
        include_contents=config.include_contents,
        generate_content_config=gen_config,
        **callback_kwargs,
    )

    if config.disallow_transfer_to_parent is not None:
        kwargs["disallow_transfer_to_parent"] = config.disallow_transfer_to_parent
    if config.disallow_transfer_to_peers is not None:
        kwargs["disallow_transfer_to_peers"] = config.disallow_transfer_to_peers

    agent_cls = _get_agent_cls(ctx)
    return agent_cls(**kwargs)


__all__ = [
    "_get_agent_cls",
    "compile_llm_agent",
]
