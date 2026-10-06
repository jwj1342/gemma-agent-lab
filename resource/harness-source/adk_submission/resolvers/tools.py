"""Tool and skill resolution for agents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from pathlib import Path
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.skill_toolset import SkillToolset

from ..context import CompilationContext, check_instruction_length
from ..errors import (
    LimitExceededError,
    PathTraversalError,
    SkillNotFoundError,
    SubmissionValidationError,
)
from ..paths import handle_schema_validation_error, validate_sandboxed_path
from ..schema import AgentToolEntry, SandboxedAgentConfig, SandboxedLlmAgentConfig
from ..yaml_loader import load_yaml


def _get_agent_tool_cls(ctx: CompilationContext | None = None) -> Any:
    if ctx is not None and ctx.agent_factory and ctx.agent_factory.agent_tool_cls is not None:
        return ctx.agent_factory.agent_tool_cls
    return AgentTool


def _get_skill_toolset_cls(ctx: CompilationContext | None = None) -> Any:
    if ctx is not None and ctx.agent_factory and ctx.agent_factory.skill_toolset_cls is not None:
        return ctx.agent_factory.skill_toolset_cls
    return SkillToolset


def resolve_tool_item(
    item: str | AgentToolEntry,
    ctx: CompilationContext,
    compile_agent_fn: Callable[[Any, CompilationContext], BaseAgent] | None = None,
) -> Any:
    """Resolve a single entry in the agent's tools list.

    - Plain ``str`` → look up in the tool registry.
    - :class:`AgentToolEntry` → compile the referenced agent and wrap it in an ADK ``AgentTool``.

    Args:
        item: Either a plain string representing a tool name in the registry, or an AgentToolEntry object referencing a sub-agent.
        ctx: The compilation context tracking mutable state and limits.
        compile_agent_fn: Optional callable for recursive agent compilation.

    Returns:
        The resolved tool callable or an ADK AgentTool instance wrapping a compiled sub-agent.

    Raises:
        ToolNotFoundError: If a plain string tool name is not found in the tool registry.
        PathTraversalError: If an AgentTool config_path escapes the submission root directory.
        SubmissionValidationError: If an AgentTool configuration file is not found.
        LimitExceededError: If sub-agent nesting depth via AgentTool exceeds ``ctx.limits.max_sub_agent_depth``.
        SubmissionSchemaError: If an AgentTool configuration fails schema validation.
        yaml.YAMLError: If an AgentTool YAML configuration file is malformed.
    """
    if isinstance(item, str):
        tool = ctx.tool_registry.get(item)
        if callable(tool):
            cur_name = getattr(tool, "__name__", "")
            if cur_name == "<lambda>":
                try:
                    tool.__name__ = item
                    tool._is_named_lambda = True  # type: ignore[attr-defined]
                except (AttributeError, TypeError):
                    @functools.wraps(tool)
                    def _wrapped_ro(*args: Any, **kwargs: Any) -> Any:
                        return tool(*args, **kwargs)

                    _wrapped_ro.__name__ = item
                    return _wrapped_ro
            elif getattr(tool, "_is_named_lambda", False) and cur_name != item:
                @functools.wraps(tool)
                def _wrapped_alias(*args: Any, **kwargs: Any) -> Any:
                    return tool(*args, **kwargs)

                _wrapped_alias.__name__ = item
                return _wrapped_alias
        return tool

    # --- AgentToolEntry -------------------------------------------------
    ref = item.agent_tool
    current_dir = ctx.current_dir or ctx.root_dir
    sub_path = ctx.resolve_sub_config_path(ref.config_path, error_prefix="AgentTool")

    ctx.depth += 1
    prev_dir = current_dir
    ctx.current_dir = sub_path.parent
    try:
        if ctx.depth > ctx.limits.max_sub_agent_depth:
            raise LimitExceededError(
                f"Sub-agent nesting too deep (via agent_tool): {ctx.depth} "
                f"(max {ctx.limits.max_sub_agent_depth})"
            )

        raw = load_yaml(sub_path, ctx.root_dir, limits=ctx.limits)
        try:
            agent_config = SandboxedAgentConfig.model_validate(raw)
        except Exception as e:  # noqa: BLE001
            handle_schema_validation_error(
                e, context_desc=f"AgentTool config ({ref.config_path})"
            )

        if compile_agent_fn is None:
            from ..compiler import _compile_agent
            compile_agent_fn = _compile_agent

        agent = compile_agent_fn(agent_config.root, ctx)
        tool_cls = _get_agent_tool_cls(ctx)
        agent_tool = tool_cls(agent=agent, skip_summarization=ref.skip_summarization)
        cache = getattr(ctx, "_resolved_agent_tools", None)
        if cache is None:
            cache = {}
            ctx._resolved_agent_tools = cache  # type: ignore[attr-defined]
        cache[id(item)] = agent_tool
        return agent_tool
    finally:
        ctx.current_dir = prev_dir
        ctx.depth -= 1


def resolve_skills(
    config: SandboxedLlmAgentConfig,
    ctx: CompilationContext,
    tools: Sequence[Any] | None = None,
) -> Any | None:
    """Resolve skills declared by an LLM agent into a SkillToolset.

    Args:
        config: The sandboxed LLM agent configuration object.
        ctx: The compilation context tracking cumulative skill counts and registries.
        tools: Optional sequence of tools resolved specifically for this agent.

    Returns:
        A SkillToolset instance if skills were declared, or None.

    Raises:
        LimitExceededError: If the cumulative skill count, skill byte size, or instruction length exceeds configured limits.
        PathTraversalError: If a skill reference escapes the sandbox boundary.
        SkillNotFoundError: If a skill reference is missing from a provided SkillRegistry.
        SubmissionValidationError: If loading a skill directory fails.
    """
    if not config.skills:
        return None

    ctx.total_skills_count += len(config.skills)
    if ctx.total_skills_count > ctx.limits.max_skills:
        raise LimitExceededError(
            f"Agent '{config.name}': cumulative skills count ({ctx.total_skills_count}) "
            f"exceeds submission limit of {ctx.limits.max_skills}"
        )

    loaded_skills = []
    for skill_ref in config.skills:
        skill_path: Path | None = None
        if ctx.skill_registry is not None:
            cand_keys = [
                skill_ref,
                Path(skill_ref).name,
                Path(skill_ref).name.replace("_", "-"),
                skill_ref.strip("/").replace("/", "-").replace("_", "-"),
            ]
            matched_key = next(
                (k for k in cand_keys if k in ctx.skill_registry),
                None,
            )
            if matched_key is not None:
                skill = ctx.skill_registry.get(matched_key)
            else:
                avail = (
                    ctx.skill_registry.list_skills()
                    if hasattr(ctx.skill_registry, "list_skills")
                    else ctx.skill_registry.list_available()
                )
                raise SkillNotFoundError(skill_ref, available=avail)
            loaded_skills.append(skill)
        else:
            skill_path = validate_sandboxed_path(
                path=skill_ref,
                base_dir=ctx.root_dir,
                allow_relative=True,
                must_exist=True,
                allow_symlinks=False,
                error_prefix="Skill",
            )
            skill_size = 0
            for child in sorted(skill_path.rglob("*")):
                if child.is_symlink():
                    raise PathTraversalError(
                        f"Skill path escapes submission directory: {child}"
                    )
                validate_sandboxed_path(
                    path=child,
                    base_dir=ctx.root_dir,
                    must_exist=True,
                    allow_symlinks=False,
                    error_prefix="Skill",
                )
                if child.is_file():
                    skill_size += child.stat().st_size
            if skill_size > ctx.limits.max_skill_size_bytes:
                raise LimitExceededError(
                    f"Skill '{skill_ref}' size {skill_size:,} bytes "
                    f"exceeds limit of {ctx.limits.max_skill_size_bytes:,} bytes"
                )
            try:
                from google.adk.skills import load_skill_from_dir

                skill = load_skill_from_dir(skill_path)
                loaded_skills.append(skill)
            except Exception as e:
                raise SubmissionValidationError(
                    f"Failed to load skill '{skill_ref}': {e}"
                ) from e

        skill_instructions = getattr(skill, "instructions", None)
        if isinstance(skill_instructions, str):
            inst_len = len(skill_instructions)
        elif skill_path is not None and (skill_path / "SKILL.md").is_file():
            inst_len = len(
                (skill_path / "SKILL.md").read_text(
                    encoding="utf-8", errors="replace"
                )
            )
        else:
            inst_len = 0

        if hasattr(ctx, "check_instruction_length"):
            ctx.check_instruction_length(inst_len, f"skill:{skill_ref}")  # type: ignore[attr-defined]
        else:
            check_instruction_length("x" * inst_len, f"skill:{skill_ref}", ctx)

    if tools is not None:
        resolved_tools = list(tools)
    else:
        resolved_tools = []
        cache = getattr(ctx, "_resolved_agent_tools", {})
        for item in config.tools or []:
            if isinstance(item, str):
                resolved_tools.append(resolve_tool_item(item, ctx))
            elif id(item) in cache:
                resolved_tools.append(cache[id(item)])
            else:
                resolved_tools.append(resolve_tool_item(item, ctx))

    skill_toolset_cls = _get_skill_toolset_cls(ctx)
    try:
        return skill_toolset_cls(
            skills=loaded_skills,
            code_executor=ctx.code_executor,
            script_timeout=ctx.script_timeout,
            additional_tools=list(resolved_tools),
        )
    except ValueError as e:
        raise SubmissionValidationError(
            f"Agent '{config.name}': {e}"
        ) from e


__all__ = [
    "_get_agent_tool_cls",
    "_get_skill_toolset_cls",
    "resolve_skills",
    "resolve_tool_item",
]
