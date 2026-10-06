"""Compilation context and limit tracking for sandboxed agent compilation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.adk.code_executors.base_code_executor import BaseCodeExecutor

from .discovery import AdapterInfo, AdapterManifest
from .errors import LimitExceededError
from .limits import GenerationConstraints, SubmissionLimits
from .registry import CallbackRegistry, ModelRegistry, SkillRegistry, ToolRegistry


@dataclass
class AgentClassFactory:
    """Factory registry holding agent constructors and ADK class references."""

    agent_cls: type[Any] | None = None
    sequential_agent_cls: type[Any] | None = None
    parallel_agent_cls: type[Any] | None = None
    loop_agent_cls: type[Any] | None = None
    agent_tool_cls: type[Any] | None = None
    skill_toolset_cls: type[Any] | None = None
    genai_types: Any = None


@dataclass
class CompilationContext:
    """Mutable state carried through the recursive compilation process.

    Tracks cumulative limits (agent count, nesting depth, instruction characters,
    and skill counts) and provides access to organizer-defined registries, limits,
    and sandboxed code execution environments.

    Attributes:
        root_dir: Absolute path to the submission root directory (sandbox boundary).
        tool_registry: Registry of pre-approved tools available to agents.
        model_registry: Registry mapping model aliases to LLM configurations.
        callback_registry: Optional registry of pre-approved agent, model, or tool callbacks.
        limits: Configurable structural limits and constraints for submission validation.
        generation_constraints: Optional organizer constraints on LLM generation parameters.
        code_executor: Optional sandboxed code executor for running skill scripts.
        script_timeout: Maximum execution time (in seconds) allowed for skill scripts.
        skill_registry: Optional registry of pre-approved skills.
        adapter_manifest: Optional manifest of discovered model adapters.
        adapter_resolver_fn: Optional callable mapping (base_model, adapter_info) to a model.
        agent_factory: Factory registry holding agent constructors and ADK class references.
        agent_count: Current cumulative count of compiled agents (enforces max_agents).
        depth: Current recursive nesting depth of sub-agents (enforces max_sub_agent_depth).
        total_instruction_chars: Current cumulative character count across all agent instructions (enforces max_total_instruction_chars).
        total_skills_count: Current cumulative count of loaded skills across all agents (enforces max_skills).
    """

    root_dir: Path
    tool_registry: ToolRegistry
    model_registry: ModelRegistry
    callback_registry: CallbackRegistry | None
    limits: SubmissionLimits
    generation_constraints: GenerationConstraints | None
    code_executor: BaseCodeExecutor | None = None
    script_timeout: int = 300
    skill_registry: SkillRegistry | None = None
    adapter_manifest: AdapterManifest | None = None
    adapter_resolver_fn: Callable[[Any, AdapterInfo], Any] | None = None
    agent_factory: AgentClassFactory = field(default_factory=AgentClassFactory)

    agent_count: int = 0
    depth: int = 0
    total_instruction_chars: int = 0
    total_skills_count: int = 0
    current_dir: Path | None = None

    def __post_init__(self) -> None:
        self._current_dir: Path = (
            Path(self.current_dir).resolve()
            if self.current_dir is not None
            else self.root_dir
        )
        self.current_dir = self._current_dir

    def check_instruction_length(
        self,
        instruction_or_len: str | int,
        agent_name: str = "agent",
    ) -> None:
        """Validate instruction length against per-agent and total limits."""
        text = (
            "x" * instruction_or_len
            if isinstance(instruction_or_len, int)
            else str(instruction_or_len)
        )
        check_instruction_length(text, agent_name, self)

    def resolve_sub_config_path(
        self,
        config_path: str,
        error_prefix: str = "Sub-agent",
    ) -> Path:
        """Resolve a relative sub-agent or AgentTool config path within the sandbox."""
        from .paths import ensure_no_traversal_components, validate_sandboxed_path

        ensure_no_traversal_components(config_path, context=f"in {error_prefix}")
        cur_dir = Path(self.current_dir).resolve() if self.current_dir is not None else self.root_dir
        candidate = cur_dir / config_path
        if cur_dir != self.root_dir and (candidate.exists() or candidate.is_symlink()):
            try:
                target_path: Any = str(candidate.relative_to(self.root_dir))
            except ValueError:
                target_path = candidate
        else:
            target_path = config_path
        return validate_sandboxed_path(
            path=target_path,
            base_dir=self.root_dir,
            allow_relative=True,
            must_exist=True,
            allow_symlinks=False,
            allowed_extensions={".yaml", ".yml"},
            error_prefix=error_prefix,
        )


def check_instruction_length(
    instruction: str,
    agent_name: str,
    ctx: CompilationContext,
) -> None:
    """Enforce per-agent and cumulative total instruction character limits.

    Args:
        instruction: The instruction string for the agent.
        agent_name: The name of the agent being compiled.
        ctx: The compilation context tracking cumulative instruction length and limits.

    Raises:
        LimitExceededError: If the individual instruction length or cumulative total instruction length exceeds configured limits.
    """
    n = len(instruction)
    if n > ctx.limits.max_instruction_chars:
        raise LimitExceededError(
            f"Agent '{agent_name}': instruction length {n:,} chars "
            f"exceeds limit of {ctx.limits.max_instruction_chars:,}"
        )
    ctx.total_instruction_chars += n
    if ctx.total_instruction_chars > ctx.limits.max_total_instruction_chars:
        raise LimitExceededError(
            f"Total instruction length {ctx.total_instruction_chars:,} chars "
            f"exceeds limit of {ctx.limits.max_total_instruction_chars:,}"
        )


def check_description_length(
    description: str | None,
    agent_name: str,
    ctx: CompilationContext,
) -> None:
    """Enforce per-agent description character limits.

    Note: Capped against ``ctx.limits.max_instruction_chars`` as a shared upper bound.

    Args:
        description: The description string for the agent, or None.
        agent_name: The name of the agent being compiled.
        ctx: The compilation context containing the submission limits.

    Raises:
        LimitExceededError: If the description length exceeds the configured limit.
    """
    if description:
        n = len(description)
        if n > ctx.limits.max_instruction_chars:
            raise LimitExceededError(
                f"Agent '{agent_name}': description length {n:,} chars "
                f"exceeds limit of {ctx.limits.max_instruction_chars:,}"
            )


__all__ = [
    "AgentClassFactory",
    "CompilationContext",
    "check_description_length",
    "check_instruction_length",
]
