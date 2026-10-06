"""Session trace recorder for ADK agent benchmark evaluations.

Captures every ADK event during an agent run and provides
structured output for debugging and analysis.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from adk_eval_core.utils.utils import unwrap_tool_response

if TYPE_CHECKING:
    from adk_eval_core.tracing.atif import Trajectory


@dataclass
class TraceEntry:
    """A single recorded event in the session trace.

    Attributes:
        timestamp: Absolute wall-clock time of the event (seconds since epoch).
        elapsed: Seconds elapsed since the trace started.
        event_type: Category of event (e.g., "tool_call", "tool_response",
            "thinking", "text", "final", "usage", or custom types).
        author: The agent or component that produced this event.
        content: Text content (for thinking, text, and final events).
        tool_name: Name of the tool (for tool_call and tool_response events).
        tool_args: Arguments passed to the tool (for tool_call events).
        tool_result: Serialized tool response (for tool_response events).
        usage: Token usage metadata (prompt, completion, total counts).
        model_version: Model identifier that produced this event.
        metadata: Arbitrary metadata for custom events.
    """

    timestamp: float
    elapsed: float  # seconds since trace start
    event_type: (
        str  # "thinking", "tool_call", "tool_response", "text", "final", "error"
    )
    author: str = ""
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] | None = None
    tool_result: str = ""
    usage: dict[str, Any] | None = None  # token counts
    model_version: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the trace entry to a dictionary."""
        d = {
            "timestamp": self.timestamp,
            "elapsed_s": round(self.elapsed, 3),
            "type": self.event_type,
        }
        if self.author:
            d["author"] = self.author
        if self.content:
            d["content"] = self.content
        if self.tool_name:
            d["tool"] = self.tool_name
        if self.tool_args is not None:
            d["args"] = self.tool_args
        if self.tool_result:
            d["result"] = self.tool_result
        if self.usage:
            d["usage"] = self.usage
        if self.model_version:
            d["model"] = self.model_version
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class SessionTrace:
    """Records and exports a complete session trace.

    Usage:
        trace = SessionTrace()
        trace.start()
        # ... during agent run, call trace.record_event(event) for each ADK event
        trace.save("trace.json")
    """

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []
        self._start_time: float | None = None
        self._on_entry: Callable[[TraceEntry], None] | None = None

    def start(self) -> None:
        """Mark the start of the trace."""
        self._start_time = time.time()
        self._entries.clear()

    @property
    def entries(self) -> list[TraceEntry]:
        """Return a copy of the recorded trace entries."""
        return list(self._entries)

    @property
    def duration(self) -> float:
        """Return the total elapsed duration of the trace in seconds."""
        if not self._entries:
            return 0.0
        return self._entries[-1].elapsed

    def set_callback(self, callback: Callable[[TraceEntry], None]) -> None:
        """Set a callback fired for each new entry (for live display)."""
        self._on_entry = callback

    def _add(self, entry: TraceEntry) -> None:
        """Append a trace entry and fire the callback if set."""
        self._entries.append(entry)
        if self._on_entry:
            self._on_entry(entry)

    def _extract_usage(self, event: Any) -> dict[str, Any] | None:
        """Extract usage metadata from an event if available."""
        if hasattr(event, "usage_metadata") and event.usage_metadata:
            um = event.usage_metadata
            usage = {}
            if hasattr(um, "prompt_token_count") and um.prompt_token_count is not None:
                usage["prompt_tokens"] = um.prompt_token_count
            if hasattr(um, "candidates_token_count") and um.candidates_token_count is not None:
                usage["completion_tokens"] = um.candidates_token_count
            if hasattr(um, "cached_content_token_count") and um.cached_content_token_count is not None:
                usage["cached_tokens"] = um.cached_content_token_count
            if hasattr(um, "total_token_count") and um.total_token_count is not None:
                usage["total_tokens"] = um.total_token_count
            if any(usage.values()):
                return usage
        return None

    def _record_compaction(
        self,
        event: Any,
        author: str,
        model: str,
        now: float,
        elapsed: float,
        consume_usage: Callable[[], dict[str, Any] | None],
    ) -> bool:
        """Record compaction action event if present."""
        if (
            hasattr(event, "actions")
            and event.actions
            and type(event.actions).__name__ != "MagicMock"
            and hasattr(event.actions, "compaction")
            and event.actions.compaction
        ):
            comp = event.actions.compaction
            meta = {
                "start_timestamp": comp.start_timestamp,
                "end_timestamp": comp.end_timestamp,
            }
            summary_text = ""
            if hasattr(comp, "compacted_content") and comp.compacted_content and hasattr(comp.compacted_content, "parts"):
                for part in comp.compacted_content.parts:
                    if hasattr(part, "text") and part.text:
                        summary_text += part.text
            self._add(
                TraceEntry(
                    timestamp=now,
                    elapsed=elapsed,
                    event_type="compaction",
                    author=author or "compactor",
                    content=summary_text.strip(),
                    usage=consume_usage(),
                    model_version=model,
                    metadata=meta,
                )
            )
            return True
        return False

    def _record_function_call(
        self,
        part: Any,
        author: str,
        model: str,
        now: float,
        elapsed: float,
        consume_usage: Callable[[], dict[str, Any] | None],
    ) -> None:
        """Record a tool function call event."""
        fc = part.function_call
        args = dict(fc.args) if fc.args else {}
        self._add(
            TraceEntry(
                timestamp=now,
                elapsed=elapsed,
                event_type="tool_call",
                author=author,
                tool_name=fc.name,
                tool_args=args,
                usage=consume_usage(),
                model_version=model,
            )
        )

    def _record_function_response(
        self,
        part: Any,
        author: str,
        model: str,
        now: float,
        elapsed: float,
        consume_usage: Callable[[], dict[str, Any] | None],
    ) -> None:
        """Record a tool function response event."""
        fr = part.function_response
        parsed = unwrap_tool_response(fr)
        resp_text = json.dumps(parsed, default=str) if parsed is not None else ""
        self._add(
            TraceEntry(
                timestamp=now,
                elapsed=elapsed,
                event_type="tool_response",
                author=author,
                tool_name=getattr(fr, "name", "") or "",
                tool_result=resp_text,
                usage=consume_usage(),
                model_version=model,
            )
        )

    def _record_thought(
        self,
        part: Any,
        author: str,
        model: str,
        now: float,
        elapsed: float,
        consume_usage: Callable[[], dict[str, Any] | None],
    ) -> None:
        """Record a thinking/thought event."""
        self._add(
            TraceEntry(
                timestamp=now,
                elapsed=elapsed,
                event_type="thinking",
                author=author,
                content=part.text.strip() if part.text else "",
                usage=consume_usage(),
                model_version=model,
            )
        )

    def _record_text(
        self,
        part: Any,
        event: Any,
        author: str,
        model: str,
        now: float,
        elapsed: float,
        consume_usage: Callable[[], dict[str, Any] | None],
    ) -> None:
        """Record a text or final response event."""
        is_final = (
            hasattr(event, "is_final_response") and event.is_final_response()
        )
        self._add(
            TraceEntry(
                timestamp=now,
                elapsed=elapsed,
                event_type="final" if is_final else "text",
                author=author,
                content=part.text.strip(),
                usage=consume_usage(),
                model_version=model,
            )
        )

    def record_event(self, event: Any) -> None:
        """Record an ADK Event into the trace."""
        now = time.time()
        elapsed = now - self._start_time if self._start_time is not None else 0.0
        author = getattr(event, "author", "")
        model = getattr(event, "model_version", "") or ""

        usage = self._extract_usage(event)

        def _consume_usage() -> dict[str, Any] | None:
            """Return usage and clear it so it's only attached once."""
            nonlocal usage
            result, usage = usage, None
            return result

        if self._record_compaction(event, author, model, now, elapsed, _consume_usage):
            return

        content = getattr(event, "content", None)
        if not content or not hasattr(content, "parts") or not content.parts:
            if usage:
                self._add(
                    TraceEntry(
                        timestamp=now,
                        elapsed=elapsed,
                        event_type="usage",
                        author=author,
                        usage=_consume_usage(),
                        model_version=model,
                    )
                )
            return

        for part in content.parts:
            if hasattr(part, "function_call") and part.function_call:
                self._record_function_call(part, author, model, now, elapsed, _consume_usage)
            elif hasattr(part, "function_response") and part.function_response:
                self._record_function_response(part, author, model, now, elapsed, _consume_usage)
            elif hasattr(part, "thought") and part.thought:
                self._record_thought(part, author, model, now, elapsed, _consume_usage)
            elif hasattr(part, "text") and part.text and part.text.strip():
                self._record_text(part, event, author, model, now, elapsed, _consume_usage)

        if usage:
            self._add(
                TraceEntry(
                    timestamp=now,
                    elapsed=elapsed,
                    event_type="usage",
                    author=author,
                    usage=_consume_usage(),
                    model_version=model,
                )
            )

    def record_custom(
        self,
        event_type: str,
        content: str = "",
        metadata: dict[str, Any] | None = None,
        author: str = "harness",
    ) -> None:
        """Record a custom trace entry (e.g., harness-level events)."""
        now = time.time()
        elapsed = now - self._start_time if self._start_time is not None else 0.0
        self._add(
            TraceEntry(
                timestamp=now,
                elapsed=elapsed,
                event_type=event_type,
                author=author,
                content=content,
                metadata=metadata,
            )
        )

    def to_atif(
        self,
        agent_name: str = "adk-eval-core",
        agent_version: str = "1.0",
        model_name: str | None = None,
    ) -> Trajectory:
        """Convert the session trace into a canonical ATIF v1.7 Trajectory."""
        from adk_eval_core.tracing.atif_converter import convert_trace_to_atif

        return convert_trace_to_atif(
            self,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=model_name,
        )

    def to_dict(self, format: str = "atif") -> dict[str, Any]:
        """Export the full trace as a dict (in either 'atif' or 'legacy' format)."""
        if format == "legacy":
            return {
                "trace_version": "1.0",
                "duration_s": round(self.duration, 3),
                "num_entries": len(self._entries),
                "entries": [e.to_dict() for e in self._entries],
                "summary": self.summarize(),
            }
        return self.to_atif().to_dict()

    @property
    def tool_call_breakdown(self) -> dict[str, int]:
        """Return a mapping of tool name → call count."""
        counts: dict[str, int] = {}
        for e in self._entries:
            if e.event_type == "tool_call":
                counts[e.tool_name] = counts.get(e.tool_name, 0) + 1
        return counts

    def summarize(self) -> dict[str, Any]:
        """Generate summary statistics."""
        breakdown = self.tool_call_breakdown

        total_prompt = sum(
            int(e.usage.get("prompt_tokens", 0) or 0) for e in self._entries if e.usage
        )
        total_cached = sum(
            int(e.usage.get("cached_tokens", 0) or 0) for e in self._entries if e.usage
        )
        total_completion = sum(
            int(e.usage.get("completion_tokens", 0) or 0) for e in self._entries if e.usage
        )
        total_tokens = sum(
            max(
                int(e.usage.get("prompt_tokens", 0) or 0) + int(e.usage.get("completion_tokens", 0) or 0),
                int(e.usage.get("total_tokens", 0) or 0),
            )
            for e in self._entries
            if e.usage
        )

        return {
            "total_events": len(self._entries),
            "tool_calls": sum(breakdown.values()),
            "tool_call_breakdown": breakdown,
            "thinking_entries": sum(
                1 for e in self._entries if e.event_type == "thinking"
            ),
            "compaction_entries": sum(
                1 for e in self._entries if e.event_type == "compaction"
            ),
            "text_entries": sum(1 for e in self._entries if e.event_type == "text"),
            "total_prompt_tokens": total_prompt,
            "total_cached_prompt_tokens": total_cached,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
        }

    def save(self, path: str | Path, format: str = "atif") -> None:
        """Save the trace to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(format=format), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionTrace:
        """Load a SessionTrace from either an ATIF v1.7 or legacy dictionary."""
        trace = cls()
        trace.start()

        if "schema_version" in data and str(data["schema_version"]).startswith("ATIF"):
            for step in data.get("steps", []):
                source = step.get("source", "agent")
                msg = step.get("message")
                extra = step.get("extra", {}) or {}
                author = extra.get("author", source)
                event_type = extra.get("event_type")
                step_elapsed = float(extra.get("elapsed_s", 0.0) or 0.0)
                step_ts = float(extra.get("timestamp", 0.0) or 0.0)
                step_meta = extra.get("metadata")
                step_metrics = step.get("metrics")
                metrics_consumed = False

                if msg is not None or (
                    event_type is not None
                    and not step.get("tool_calls")
                    and not step.get("observation")
                    and event_type != "usage"
                ):
                    if not event_type:
                        if source == "system":
                            event_type = "system_instruction"
                        elif source == "user":
                            event_type = "task_prompt"
                        else:
                            event_type = "text"
                    trace._add(
                        TraceEntry(
                            timestamp=step_ts,
                            elapsed=step_elapsed,
                            event_type=event_type,
                            author=author,
                            content=str(msg) if msg is not None else "",
                            usage=step_metrics,
                            model_version=step.get("model_name", "") or "",
                            metadata=step_meta,
                        )
                    )
                    metrics_consumed = True

                for idx, tc in enumerate(step.get("tool_calls", [])):
                    tc_extra = tc.get("extra", {}) or {}
                    elapsed = float(tc_extra.get("elapsed_s", step_elapsed) or 0.0)
                    ts = float(tc_extra.get("timestamp", step_ts) or 0.0)
                    tc_author = tc_extra.get("author", author)
                    if "usage" in tc_extra:
                        tc_usage = tc_extra.get("usage")
                        metrics_consumed = True
                    else:
                        tc_usage = step_metrics if (not metrics_consumed and idx == 0) else None
                        if tc_usage is not None:
                            metrics_consumed = True
                    tc_model = tc_extra.get("model_version") or step.get("model_name", "") or ""
                    trace._add(
                        TraceEntry(
                            timestamp=ts,
                            elapsed=elapsed,
                            event_type="tool_call",
                            author=tc_author,
                            tool_name=tc.get("function_name", ""),
                            tool_args=tc.get("arguments", {}),
                            usage=tc_usage,
                            model_version=tc_model,
                        )
                    )

                obs = step.get("observation")
                if obs:
                    obs_extra = obs.get("extra", {}) or {}
                    tool_name = obs_extra.get("tool_name", "")
                    elapsed = float(obs_extra.get("elapsed_s", step_elapsed) or 0.0)
                    ts = float(obs_extra.get("timestamp", step_ts) or 0.0)
                    obs_author = obs_extra.get("author") or "system"
                    trace._add(
                        TraceEntry(
                            timestamp=ts,
                            elapsed=elapsed,
                            event_type="tool_response",
                            author=obs_author,
                            tool_name=tool_name,
                            tool_result=str(obs.get("content", "")),
                        )
                    )

                if not metrics_consumed and step_metrics:
                    trace._add(
                        TraceEntry(
                            timestamp=step_ts,
                            elapsed=step_elapsed,
                            event_type=event_type or "usage",
                            author=author,
                            usage=step_metrics,
                            model_version=step.get("model_name", "") or "",
                            metadata=step_meta,
                        )
                    )
            return trace

        for e in data.get("entries", []):
            trace._add(
                TraceEntry(
                    timestamp=e.get("timestamp")
                    if e.get("timestamp") is not None
                    else 0.0,
                    elapsed=e.get("elapsed_s")
                    if e.get("elapsed_s") is not None
                    else 0.0,
                    event_type=e.get("type") if e.get("type") is not None else "custom",
                    author=e.get("author") if e.get("author") is not None else "",
                    content=e.get("content") if e.get("content") is not None else "",
                    tool_name=e.get("tool") if e.get("tool") is not None else "",
                    tool_args=e.get("args"),
                    tool_result=e.get("result")
                    if e.get("result") is not None
                    else "",
                    usage=e.get("usage"),
                    model_version=e.get("model") if e.get("model") is not None else "",
                    metadata=e.get("metadata"),
                )
            )
        return trace

    @classmethod
    def from_json(cls, path: str | Path) -> SessionTrace:
        """Load a SessionTrace from a JSON file (ATIF v1.7 or legacy)."""
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return cls.from_dict(data)

    def to_markdown(
        self,
        max_thought_len: int | None = 200,
        max_tool_args_len: int | None = 200,
        max_tool_result_len: int | None = 200,
        max_text_len: int | None = 200,
        max_final_len: int | None = 300,
        max_compaction_len: int | None = 300,
        include_events: set[str] | None = None,
        exclude_events: set[str] | None = None,
        show_usage: bool = True,
        show_summary_section: bool = True,
        show_tool_breakdown: bool = True,
        timestamp_mode: str = "elapsed",
        multiline_mode: str = "collapse",
        use_emojis: bool = True,
        show_author: bool = True,
    ) -> str:
        """Export the trace as a readable markdown document."""
        from adk_eval_core.tracing.markdown import render_trace_markdown

        return render_trace_markdown(
            self,
            max_thought_len=max_thought_len,
            max_tool_args_len=max_tool_args_len,
            max_tool_result_len=max_tool_result_len,
            max_text_len=max_text_len,
            max_final_len=max_final_len,
            max_compaction_len=max_compaction_len,
            include_events=include_events,
            exclude_events=exclude_events,
            show_usage=show_usage,
            show_summary_section=show_summary_section,
            show_tool_breakdown=show_tool_breakdown,
            timestamp_mode=timestamp_mode,
            multiline_mode=multiline_mode,
            use_emojis=use_emojis,
            show_author=show_author,
        )
