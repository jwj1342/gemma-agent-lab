"""Restricted Pydantic schema for sandboxed agent configurations.

This module defines the Pydantic models used to parse and validate agent YAML
configurations submitted by participants. To ensure sandboxed execution, these models
mirror the standard ADK configuration structure but **replace all ``CodeConfig``-based
fields** with safe string-reference alternatives (e.g., file paths or registry keys).

Primary Models:
    * :class:`SandboxedLlmAgentConfig`: Configuration for LLM-backed agents.
    * :class:`SandboxedSequentialAgentConfig`: Configuration for sequential multi-agent workflows.
    * :class:`SandboxedParallelAgentConfig`: Configuration for parallel multi-agent workflows.
    * :class:`SandboxedLoopAgentConfig`: Configuration for iterative loop workflows.
    * :class:`SandboxedAgentConfig`: Root discriminated union model for parsing arbitrary agent configs.

Security & Validation:
    All file path references (such as sub-agent configs and skill directories) are strictly
    validated against path traversal attacks (e.g., absolute paths or ``..`` directory climbing).
    Generation parameters are parsed permissively here and subsequently enforced by
    :class:`~adk_submission.limits.GenerationConstraints` during compilation.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    RootModel,
    Tag,
    ValidationInfo,
    field_validator,
)

from .paths import ensure_no_traversal_components

# ---------------------------------------------------------------------------
# Sub-agent reference — config_path only, no code references
# ---------------------------------------------------------------------------


class SubAgentRef(BaseModel):
    """Reference to a sub-agent's YAML configuration file.

    This model is used within multi-agent workflows (Sequential, Parallel, Loop, and LLM sub-agents)
    to reference child agents. Only ``config_path`` is supported; the standard ADK ``code`` field
    (which resolves arbitrary Python imports) is intentionally excluded to prevent unsandboxed code execution.
    The ``config_path`` field is strictly validated to prevent path traversal attacks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: str = Field(
        description="Relative path to the sub-agent's YAML config within "
        "the submission directory."
    )

    @field_validator("config_path")
    @classmethod
    def validate_no_path_traversal(cls, v: str) -> str:
        """Validate that the configuration path does not attempt directory traversal.

        Args:
            v (str): The relative file path to validate.

        Returns:
            str: The unmodified file path if validation succeeds.

        Raises:
            PathTraversalError: If the path contains traversal components.
        """
        ensure_no_traversal_components(v)
        return v


# ---------------------------------------------------------------------------
# Agent-as-tool reference
# ---------------------------------------------------------------------------


class AgentToolRef(BaseModel):
    """Reference to an agent that should be wrapped and invoked as an ``AgentTool``.

    Allows an LLM agent to call another agent as a tool during its execution.
    The ``config_path`` field is strictly validated to prevent path traversal attacks.

    Attributes:
        skip_summarization (bool): If True, the raw output of the tool-agent is returned directly
            to the calling LLM without intermediate summarization. Defaults to False.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: str = Field(
        description="Relative path to the agent's YAML config within "
        "the submission directory."
    )
    skip_summarization: bool = False

    @field_validator("config_path")
    @classmethod
    def validate_no_path_traversal(cls, v: str) -> str:
        """Validate that the tool agent configuration path does not attempt directory traversal.

        Args:
            v (str): The relative file path to validate.

        Returns:
            str: The unmodified file path if validation succeeds.

        Raises:
            PathTraversalError: If the path contains traversal components.
        """
        ensure_no_traversal_components(v)
        return v


class AgentToolEntry(BaseModel):
    """A ``tools`` list entry that wraps an agent as a tool.

    Attributes:
        agent_tool (AgentToolRef): The reference configuration for the agent being wrapped as a tool.

    YAML example::

        tools:
          - agent_tool:
              config_path: sub_agents/summarizer.yaml
              skip_summarization: true
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_tool: AgentToolRef


# ---------------------------------------------------------------------------
# Generation config — permissive schema, validated at compile time
# ---------------------------------------------------------------------------


def _reject_bool_nan_inf(v: Any, field_name: str) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"{field_name} cannot be a boolean")
    if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
        raise ValueError(f"{field_name} must be a finite number, got {v!r}")
    return v


class ThinkingConfig(BaseModel):
    """Thinking configuration for supported LLMs (e.g., reasoning models).

    Attributes:
        thinking_budget (Optional[int]): The token budget allocated for the model's internal
            reasoning/thinking process. Must be greater than or equal to 0 (0 disables thinking).
            Defaults to None.
        include_thoughts (Optional[bool]): Whether to include the raw reasoning thoughts in the
            final model output. Defaults to None.
        thinking_level (Optional[str]): Presets for model thinking level (e.g., 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH'). Defaults to None.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    thinking_budget: int | None = Field(default=None, ge=0)
    include_thoughts: bool | None = None
    thinking_level: str | None = None

    @field_validator("thinking_budget", mode="before")
    @classmethod
    def _validate_thinking_budget(cls, v: Any) -> Any:
        return _reject_bool_nan_inf(v, "thinking_budget")


class GenerateContentConfig(BaseModel):
    """Generation parameters — validated against organizer constraints at compile time.

    This schema is intentionally permissive: it defines the *universe* of parameters a submission
    could set. Actual restrictions and bounding checks are applied by
    :class:`~adk_submission.limits.GenerationConstraints` during compilation.

    Fields that are **never** settable by submissions (``safety_settings``, ``tools``,
    ``system_instruction``, ``response_schema``) are hard-excluded from this schema.

    Attributes:
        temperature (Optional[float]): Sampling temperature. Must be >= 0.0. Defaults to None.
        top_p (Optional[float]): Nucleus sampling threshold. Must be between 0.0 and 1.0 inclusive. Defaults to None.
        top_k (Optional[int]): Top-k sampling cutoff. Must be >= 1. Defaults to None.
        max_output_tokens (Optional[int]): Maximum number of tokens to generate. Must be >= 1. Defaults to None.
        stop_sequences (Optional[list[str]]): List of strings that stop generation when encountered. Defaults to None.
        presence_penalty (Optional[float]): Penalty for repeated tokens based on presence. Defaults to None.
        frequency_penalty (Optional[float]): Penalty for repeated tokens based on frequency. Defaults to None.
        response_mime_type (Optional[str]): Expected MIME type for the model response (e.g., 'application/json'). Defaults to None.
        seed (Optional[int]): Random seed for deterministic generation. Defaults to None.
        thinking_config (Optional[ThinkingConfig]): Configuration for model reasoning/thinking. Defaults to None.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_mime_type: str | None = None
    seed: int | None = None
    thinking_config: ThinkingConfig | None = None

    @field_validator(
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        mode="before",
    )
    @classmethod
    def _validate_numeric_fields(cls, v: Any, info: ValidationInfo) -> Any:
        return _reject_bool_nan_inf(v, info.field_name or "numeric_field")


# ---------------------------------------------------------------------------
# Agent configs — shared base + specialized subclasses
# ---------------------------------------------------------------------------


class _BaseAgentFields(BaseModel):
    """Fields shared by all agent configuration types.

    Not part of the public API — use the concrete subclasses below.

    Attributes:
        name (str): The unique identifying name of the agent.
        description (str): Optional description of the agent's purpose. Defaults to an empty string.
        before_agent_callbacks (Optional[list[str]]): List of callback names to execute before the agent runs.
        after_agent_callbacks (Optional[list[str]]): List of callback names to execute after the agent completes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""

    # Callbacks available on every agent type
    before_agent_callbacks: list[str] | None = None
    after_agent_callbacks: list[str] | None = None


class SandboxedLlmAgentConfig(_BaseAgentFields):
    """Restricted ``LlmAgent`` configuration — replaces all code references with safe strings.

    Attributes:
        agent_class (Literal["LlmAgent"]): Literal identifier for the agent class. Defaults to "LlmAgent".
        model (Optional[str]): The model alias or name to be used by the agent. Defaults to None.
        adapter (Optional[str]): Name of a discovered adapter from adapters/ to attach to the base model. Defaults to None.
        output_key (Optional[str]): Key under which the agent's output is stored in the state. Defaults to None.
        include_contents (Literal["default", "none"]): Controls whether input contents are included in the prompt. Defaults to "default".
        disallow_transfer_to_parent (Optional[bool]): If True, prevents the agent from transferring control back to its parent. Defaults to None.
        disallow_transfer_to_peers (Optional[bool]): If True, prevents the agent from transferring control to peer agents. Defaults to None.
        tools (Optional[list[str | AgentToolEntry]]): List of tool names (strings) or wrapped agent tools (:class:`AgentToolEntry`). Defaults to None.
        sub_agents (Optional[list[SubAgentRef]]): List of sub-agents managed by this LLM agent. Defaults to None.
        generate_content_config (Optional[GenerateContentConfig]): Permissive generation parameters for the LLM. Defaults to None.
        before_model_callbacks (Optional[list[str]]): List of callback names to execute before model generation. Defaults to None.
        after_model_callbacks (Optional[list[str]]): List of callback names to execute after model generation. Defaults to None.
        before_tool_callbacks (Optional[list[str]]): List of callback names to execute before a tool is called. Defaults to None.
        after_tool_callbacks (Optional[list[str]]): List of callback names to execute after a tool completes. Defaults to None.
    """

    agent_class: Literal["LlmAgent"] = "LlmAgent"
    model: str | None = None
    adapter: str | None = None
    instruction: str = Field(
        description="Agent instruction text — inline or resolved from !include."
    )
    output_key: str | None = None
    include_contents: Literal["default", "none"] = "default"
    disallow_transfer_to_parent: bool | None = None
    disallow_transfer_to_peers: bool | None = None
    tools: list[str | AgentToolEntry] | None = None
    skills: list[str] | None = Field(
        default=None,
        description="List of relative paths to skill directories within the submission archive.",
    )
    sub_agents: list[SubAgentRef] | None = None
    generate_content_config: GenerateContentConfig | None = None

    # LLM-specific callbacks
    before_model_callbacks: list[str] | None = None
    after_model_callbacks: list[str] | None = None
    before_tool_callbacks: list[str] | None = None
    after_tool_callbacks: list[str] | None = None

    @field_validator("agent_class", mode="before")
    @classmethod
    def _default_null_agent_class(cls, v: Any) -> Any:
        return "LlmAgent" if v is None else v

    @field_validator("skills")
    @classmethod
    def validate_skills_paths(cls, v: list[str] | None) -> list[str] | None:
        """Validate that skill directory paths do not attempt directory traversal.

        Args:
            v (Optional[list[str]]): List of relative skill directory paths to validate.

        Returns:
            Optional[list[str]]: The unmodified list of skill paths if validation succeeds.

        Raises:
            PathTraversalError: If any path contains traversal components.
        """
        if v is not None:
            for p in v:
                ensure_no_traversal_components(p, context="in skill")
        return v


class SandboxedSequentialAgentConfig(_BaseAgentFields):
    """Restricted ``SequentialAgent`` configuration for running sub-agents in sequential order.

    Attributes:
        agent_class (Literal["SequentialAgent"]): Literal identifier for the agent class.
        sub_agents (list[SubAgentRef]): Ordered list of sub-agent references to execute sequentially.
    """

    agent_class: Literal["SequentialAgent"]
    sub_agents: list[SubAgentRef]


class SandboxedParallelAgentConfig(_BaseAgentFields):
    """Restricted ``ParallelAgent`` configuration for running sub-agents concurrently.

    Attributes:
        agent_class (Literal["ParallelAgent"]): Literal identifier for the agent class.
        sub_agents (list[SubAgentRef]): List of sub-agent references to execute in parallel.
    """

    agent_class: Literal["ParallelAgent"]
    sub_agents: list[SubAgentRef]


class SandboxedLoopAgentConfig(_BaseAgentFields):
    """Restricted ``LoopAgent`` configuration for executing sub-agents in an iterative loop.

    Attributes:
        agent_class (Literal["LoopAgent"]): Literal identifier for the agent class.
        sub_agents (list[SubAgentRef]): List of sub-agent references to execute in each loop iteration.
        max_iterations (Optional[int]): Maximum number of loop iterations allowed. Must be >= 1. Defaults to None.
    """

    agent_class: Literal["LoopAgent"]
    sub_agents: list[SubAgentRef]
    max_iterations: int | None = Field(None, ge=1)

    @field_validator("max_iterations", mode="before")
    @classmethod
    def _validate_max_iterations(cls, v: Any) -> Any:
        return _reject_bool_nan_inf(v, "max_iterations")


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


def agent_class_discriminator(v: Any) -> str:
    """Return the discriminator tag for the agent configuration union.

    Examines the input dictionary for an ``agent_class`` key to determine the concrete
    agent model to instantiate. Defaults to "LlmAgent" if the key is not present or is None.

    Args:
        v (Any): The raw configuration dictionary being parsed.

    Returns:
        str: The discriminator tag corresponding to the agent class (e.g., "LlmAgent", "SequentialAgent").

    Raises:
        ValueError: If the input ``v`` is not a dictionary.
    """
    if isinstance(v, dict):
        return v.get("agent_class") or "LlmAgent"
    raise ValueError(f"Invalid agent config (expected dict): {type(v).__name__}")


ConfigsUnion = Annotated[
    Annotated[SandboxedLlmAgentConfig, Tag("LlmAgent")] | Annotated[SandboxedSequentialAgentConfig, Tag("SequentialAgent")] | Annotated[SandboxedParallelAgentConfig, Tag("ParallelAgent")] | Annotated[SandboxedLoopAgentConfig, Tag("LoopAgent")],
    Discriminator(agent_class_discriminator),
]


class SandboxedAgentConfig(RootModel[ConfigsUnion]):
    """Root model for parsing a sandboxed agent YAML configuration.

    Acts as a discriminated union over :class:`SandboxedLlmAgentConfig`,
    :class:`SandboxedSequentialAgentConfig`, :class:`SandboxedParallelAgentConfig`,
    and :class:`SandboxedLoopAgentConfig`. The concrete model is dynamically selected
    based on the ``agent_class`` field in the input data.

    Example::

        config = SandboxedAgentConfig.model_validate(yaml_dict)
        agent_config = config.root  # Instance of SandboxedLlmAgentConfig, etc.
    """

