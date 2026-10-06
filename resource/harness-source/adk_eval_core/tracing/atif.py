"""ATIF (Agent Trajectory Interchange Format) v1.7 schema definitions and serializers.

Provides native data models and serialization methods compliant with RFC 0001 (ATIF v1.7)
for logging autonomous LLM agent interaction histories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallSchema:
    """Represents a single tool invocation within an agent step."""

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize ToolCallSchema to ATIF dictionary."""
        d: dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "function_name": self.function_name,
            "arguments": self.arguments,
        }
        if self.extra is not None:
            d["extra"] = self.extra
        return d


@dataclass
class ObservationResultSchema:
    """Represents an observation or tool result received within a step."""

    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize ObservationResultSchema to ATIF dictionary."""
        d: dict[str, Any] = {
            "content": self.content,
        }
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.extra is not None:
            d["extra"] = self.extra
        return d


@dataclass
class StepObject:
    """Represents a single interaction turn (user message, agent turn, or system observation)."""

    step_id: int
    source: str  # "user", "agent", "system"
    message: str | list[dict[str, Any]] | None = None
    model_name: str | None = None
    tool_calls: list[ToolCallSchema] = field(default_factory=list)
    observation: ObservationResultSchema | None = None
    metrics: dict[str, Any] | None = None
    llm_call_count: int = 1
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize StepObject to ATIF dictionary."""
        d: dict[str, Any] = {
            "step_id": self.step_id,
            "source": self.source,
        }
        if self.message is not None:
            d["message"] = self.message
        if self.model_name is not None:
            d["model_name"] = self.model_name
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.observation is not None:
            d["observation"] = self.observation.to_dict()
        if self.metrics is not None:
            d["metrics"] = self.metrics
        if self.llm_call_count != 1 or self.source == "agent":
            d["llm_call_count"] = self.llm_call_count
        if self.extra is not None:
            d["extra"] = self.extra
        return d


@dataclass
class AgentSchema:
    """Identifies the agent configuration used for the trajectory."""

    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize AgentSchema to ATIF dictionary."""
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        if self.model_name is not None:
            d["model_name"] = self.model_name
        if self.tool_definitions is not None:
            d["tool_definitions"] = self.tool_definitions
        if self.extra is not None:
            d["extra"] = self.extra
        return d


@dataclass
class Trajectory:
    """Root ATIF v1.7 Trajectory container representing a complete interaction history."""

    schema_version: str = "ATIF-v1.7"
    agent: AgentSchema = field(
        default_factory=lambda: AgentSchema(name="adk-eval-core", version="1.0")
    )
    steps: list[StepObject] = field(default_factory=list)
    session_id: str | None = None
    trajectory_id: str | None = None
    notes: str | None = None
    final_metrics: dict[str, Any] | None = None
    continued_trajectory_ref: str | None = None
    subagent_trajectories: list[Trajectory] = field(default_factory=list)
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize Trajectory to ATIF dictionary."""
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "agent": self.agent.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.trajectory_id is not None:
            d["trajectory_id"] = self.trajectory_id
        if self.notes is not None:
            d["notes"] = self.notes
        if self.final_metrics is not None:
            d["final_metrics"] = self.final_metrics
        if self.continued_trajectory_ref is not None:
            d["continued_trajectory_ref"] = self.continued_trajectory_ref
        if self.subagent_trajectories:
            d["subagent_trajectories"] = [
                sub.to_dict() for sub in self.subagent_trajectories
            ]
        if self.extra is not None:
            d["extra"] = self.extra
        return d
