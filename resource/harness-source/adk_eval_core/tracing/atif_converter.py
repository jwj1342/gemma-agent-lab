"""ATIF v1.7 trajectory conversion for SessionTrace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adk_eval_core.tracing.atif import (
    AgentSchema,
    ObservationResultSchema,
    StepObject,
    ToolCallSchema,
    Trajectory,
)

if TYPE_CHECKING:
    from adk_eval_core.tracing.trace import SessionTrace


def _merge_usage(dst: dict[str, Any] | None, src: dict[str, Any]) -> dict[str, Any]:
    """Sum numeric token fields from src into dst instead of overwriting."""
    if not dst:
        return dict(src)
    merged = dict(dst)
    for k, v in src.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(merged.get(k), (int, float)):
            merged[k] = merged[k] + v
        else:
            merged[k] = v
    return merged


def convert_trace_to_atif(
    trace: SessionTrace,
    agent_name: str = "adk-eval-core",
    agent_version: str = "1.0",
    model_name: str | None = None,
) -> Trajectory:
    """Convert the session trace into a canonical ATIF v1.7 Trajectory."""
    steps: list[StepObject] = []
    current_step_id = 1

    for i, entry in enumerate(trace.entries):
        if entry.event_type == "tool_call":
            call_id = f"call_{i}"
            tc_extra: dict[str, Any] = {
                "elapsed_s": entry.elapsed,
                "timestamp": entry.timestamp,
            }
            if entry.author:
                tc_extra["author"] = entry.author
            if entry.usage:
                tc_extra["usage"] = dict(entry.usage)
            if entry.model_version:
                tc_extra["model_version"] = entry.model_version
            tool_call = ToolCallSchema(
                tool_call_id=call_id,
                function_name=entry.tool_name,
                arguments=entry.tool_args or {},
                extra=tc_extra,
            )
            prev_event_type = (steps[-1].extra or {}).get("event_type") if steps else None
            if (
                steps
                and steps[-1].source == "agent"
                and steps[-1].observation is None
                and prev_event_type in (None, "thinking", "tool_call")
            ):
                steps[-1].tool_calls.append(tool_call)
                if entry.usage:
                    steps[-1].metrics = _merge_usage(steps[-1].metrics, entry.usage)
            else:
                step_extra: dict[str, Any] = {
                    "elapsed_s": entry.elapsed,
                    "timestamp": entry.timestamp,
                }
                if entry.author:
                    step_extra["author"] = entry.author
                steps.append(
                    StepObject(
                        step_id=current_step_id,
                        source="agent",
                        model_name=entry.model_version or model_name,
                        tool_calls=[tool_call],
                        metrics=dict(entry.usage) if entry.usage else None,
                        extra=step_extra,
                    )
                )
                current_step_id += 1
        elif entry.event_type == "tool_response":
            obs_extra: dict[str, Any] = {
                "tool_name": entry.tool_name,
                "elapsed_s": entry.elapsed,
                "timestamp": entry.timestamp,
            }
            if entry.author:
                obs_extra["author"] = entry.author
            obs = ObservationResultSchema(
                content=entry.tool_result,
                extra=obs_extra,
            )
            if steps and steps[-1].source == "agent" and steps[-1].observation is None and steps[-1].tool_calls:
                steps[-1].observation = obs
            else:
                steps.append(
                    StepObject(
                        step_id=current_step_id,
                        source="system",
                        observation=obs,
                        extra={
                            "elapsed_s": entry.elapsed,
                            "timestamp": entry.timestamp,
                        },
                    )
                )
                current_step_id += 1
        elif entry.event_type in (
            "thinking",
            "text",
            "final",
            "system_instruction",
            "task_prompt",
        ):
            if entry.event_type == "system_instruction":
                source = "system"
            elif entry.event_type == "task_prompt":
                source = "user"
            else:
                source = "agent"
            step_extra = {
                "event_type": entry.event_type,
                "elapsed_s": entry.elapsed,
                "timestamp": entry.timestamp,
            }
            if entry.author:
                step_extra["author"] = entry.author
            if entry.metadata is not None:
                step_extra["metadata"] = entry.metadata
            steps.append(
                StepObject(
                    step_id=current_step_id,
                    source=source,
                    message=entry.content,
                    model_name=entry.model_version or model_name,
                    metrics=dict(entry.usage) if entry.usage else None,
                    extra=step_extra,
                )
            )
            current_step_id += 1
        elif entry.event_type == "usage" and entry.usage:
            if steps:
                steps[-1].metrics = _merge_usage(steps[-1].metrics, entry.usage)
            else:
                steps.append(
                    StepObject(
                        step_id=current_step_id,
                        source="agent",
                        metrics=dict(entry.usage),
                        extra={
                            "event_type": "usage",
                            "elapsed_s": entry.elapsed,
                            "timestamp": entry.timestamp,
                        },
                    )
                )
                current_step_id += 1
        else:
            step_extra = {
                "event_type": entry.event_type,
                "elapsed_s": entry.elapsed,
                "timestamp": entry.timestamp,
            }
            if entry.author:
                step_extra["author"] = entry.author
            if entry.metadata is not None:
                step_extra["metadata"] = entry.metadata
            steps.append(
                StepObject(
                    step_id=current_step_id,
                    source="system",
                    message=entry.content,
                    model_name=entry.model_version or model_name,
                    metrics=dict(entry.usage) if entry.usage else None,
                    extra=step_extra,
                )
            )
            current_step_id += 1

    summary = trace.summarize()
    final_metrics = {
        "total_prompt_tokens": summary.get("total_prompt_tokens", 0),
        "total_completion_tokens": summary.get("total_completion_tokens", 0),
        "total_cached_tokens": summary.get("total_cached_prompt_tokens", 0),
        "total_tokens": summary.get("total_tokens", 0),
        "total_steps": len(steps),
    }

    return Trajectory(
        schema_version="ATIF-v1.7",
        agent=AgentSchema(
            name=agent_name, version=agent_version, model_name=model_name
        ),
        steps=steps,
        final_metrics=final_metrics,
    )
