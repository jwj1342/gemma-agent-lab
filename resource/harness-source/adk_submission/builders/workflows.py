"""Workflow agent builders (SequentialAgent, ParallelAgent, LoopAgent)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.agents import LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent

from ..context import CompilationContext, check_description_length
from ..errors import LimitExceededError
from ..resolvers.callbacks import resolve_base_callbacks
from ..schema import (
    SandboxedLoopAgentConfig,
    SandboxedParallelAgentConfig,
    SandboxedSequentialAgentConfig,
)
from .sub_agents import compile_sub_agents


def _get_sequential_agent_cls(ctx: CompilationContext | None = None) -> Any:
    if ctx is not None and ctx.agent_factory and ctx.agent_factory.sequential_agent_cls is not None:
        return ctx.agent_factory.sequential_agent_cls
    return SequentialAgent


def _get_parallel_agent_cls(ctx: CompilationContext | None = None) -> Any:
    if ctx is not None and ctx.agent_factory and ctx.agent_factory.parallel_agent_cls is not None:
        return ctx.agent_factory.parallel_agent_cls
    return ParallelAgent


def _get_loop_agent_cls(ctx: CompilationContext | None = None) -> Any:
    if ctx is not None and ctx.agent_factory and ctx.agent_factory.loop_agent_cls is not None:
        return ctx.agent_factory.loop_agent_cls
    return LoopAgent


def compile_sequential_agent(
    config: SandboxedSequentialAgentConfig,
    ctx: CompilationContext,
    compile_agent_fn: Callable[[Any, CompilationContext], BaseAgent] | None = None,
) -> SequentialAgent:
    """Compile a sandboxed sequential agent configuration into an ADK SequentialAgent.

    Args:
        config: The sandboxed sequential workflow agent configuration object.
        ctx: The compilation context tracking mutable state and limits.
        compile_agent_fn: Optional callable for recursive agent compilation.

    Returns:
        A fully configured ADK SequentialAgent instance.

    Raises:
        LimitExceededError: If the description length limit is exceeded.
        CallbackNotFoundError: If a referenced callback is not found in the callback registry.
    """
    check_description_length(config.description, config.name, ctx)
    sub_agents = compile_sub_agents(config.sub_agents, ctx, compile_agent_fn)
    callback_kwargs = resolve_base_callbacks(config, ctx)
    cls = _get_sequential_agent_cls(ctx)
    return cls(
        name=config.name,
        description=config.description or "",
        sub_agents=sub_agents,
        **callback_kwargs,
    )


def compile_parallel_agent(
    config: SandboxedParallelAgentConfig,
    ctx: CompilationContext,
    compile_agent_fn: Callable[[Any, CompilationContext], BaseAgent] | None = None,
) -> ParallelAgent:
    """Compile a sandboxed parallel agent configuration into an ADK ParallelAgent.

    Args:
        config: The sandboxed parallel workflow agent configuration object.
        ctx: The compilation context tracking mutable state and limits.
        compile_agent_fn: Optional callable for recursive agent compilation.

    Returns:
        A fully configured ADK ParallelAgent instance.

    Raises:
        LimitExceededError: If the description length limit is exceeded.
        CallbackNotFoundError: If a referenced callback is not found in the callback registry.
    """
    check_description_length(config.description, config.name, ctx)
    sub_agents = compile_sub_agents(config.sub_agents, ctx, compile_agent_fn)
    callback_kwargs = resolve_base_callbacks(config, ctx)
    cls = _get_parallel_agent_cls(ctx)
    return cls(
        name=config.name,
        description=config.description or "",
        sub_agents=sub_agents,
        **callback_kwargs,
    )


def compile_loop_agent(
    config: SandboxedLoopAgentConfig,
    ctx: CompilationContext,
    compile_agent_fn: Callable[[Any, CompilationContext], BaseAgent] | None = None,
) -> LoopAgent:
    """Compile a sandboxed loop agent configuration into an ADK LoopAgent.

    Validates and caps the maximum loop iterations against configured submission limits.

    Args:
        config: The sandboxed loop workflow agent configuration object.
        ctx: The compilation context tracking mutable state and limits.
        compile_agent_fn: Optional callable for recursive agent compilation.

    Returns:
        A fully configured ADK LoopAgent instance.

    Raises:
        LimitExceededError: If description length or max_iterations limits are exceeded.
        CallbackNotFoundError: If a referenced callback is not found in the callback registry.
    """
    check_description_length(config.description, config.name, ctx)
    max_iter = config.max_iterations
    if max_iter is None:
        max_iter = ctx.limits.max_loop_iterations
    elif max_iter > ctx.limits.max_loop_iterations:
        raise LimitExceededError(
            f"Agent '{config.name}': max_iterations={max_iter} exceeds "
            f"limit of {ctx.limits.max_loop_iterations}"
        )

    sub_agents = compile_sub_agents(config.sub_agents, ctx, compile_agent_fn)
    callback_kwargs = resolve_base_callbacks(config, ctx)
    cls = _get_loop_agent_cls(ctx)
    return cls(
        name=config.name,
        description=config.description or "",
        sub_agents=sub_agents,
        max_iterations=max_iter,
        **callback_kwargs,
    )


__all__ = [
    "_get_loop_agent_cls",
    "_get_parallel_agent_cls",
    "_get_sequential_agent_cls",
    "compile_loop_agent",
    "compile_parallel_agent",
    "compile_sequential_agent",
]
