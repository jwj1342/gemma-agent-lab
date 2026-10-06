"""Post-session trace viewer for ADK evaluations.

Renders a self-contained HTML document (embedded CSS + vanilla JS) inside a
Jupyter notebook cell.  No widget dependencies — works in JupyterLab, Colab,
VS Code notebooks, and nbviewer.

Usage::

    from adk_eval_core.viewer import TraceViewer

    # From an evaluation result or TaskResult
    viewer = TraceViewer(result)
    viewer.show()

    # From a saved trace JSON
    viewer = TraceViewer.from_json("trace_instance_1.json")
    viewer.show()

    # Jupyter auto-display (just put viewer on the last line of a cell)
    viewer = TraceViewer(result)
    viewer  # renders automatically via _repr_html_
"""

from __future__ import annotations

import collections
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adk_eval_core.tracing.trace import SessionTrace
from adk_eval_core.ui.formatters import (
    ToolFormatterRegistry,
    get_default_formatter_registry,
)
from adk_eval_core.viewer.templates import (
    badge,
    detail_sections,
    esc,
    format_elapsed,
    render_full_viewer_html,
    render_summary_only_html,
    render_timeline_only_html,
    summary_line,
    token_badge,
)

# ---------------------------------------------------------------------------
# Result metadata model
# ---------------------------------------------------------------------------


@dataclass
class ViewerMetadata:
    """Structured container for result-level metadata used by the viewer."""

    instance_id: str = ""
    problem_id: str = ""
    metric: str = ""
    end_status: str = "unknown"
    resolved: bool | None = None
    wall_time_seconds: float = 0.0
    tool_calls_used: int = 0
    submissions_used: int = 0
    total_cost_usd: float = 0.0
    max_budget_usd: float = 0.0
    max_tool_calls: int = 0
    max_submissions: int = 0
    max_time_minutes: float = 0.0
    total_input_tokens: int = 0
    total_cached_input_tokens: int = 0
    last_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_calls: int = 0
    score: float = float("nan")
    final_score: float = float("nan")
    best_public_score: float = float("nan")
    best_private_score: float = float("nan")
    public_scores: dict[str, float] = field(default_factory=dict)
    private_scores: dict[str, float] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    auto_selected_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    higher_is_better: bool = True

    @classmethod
    def from_result(
        cls, result: Any, trace: SessionTrace | None = None
    ) -> ViewerMetadata:
        """Construct ViewerMetadata directly from a BaseTaskResult or benchmark result object."""
        m = cls()

        def _is_mock(val: Any) -> bool:
            return type(val).__name__ in ("MagicMock", "Mock", "NonCallableMagicMock")

        def _get_str(attr: str) -> str:
            val = getattr(result, attr, None)
            if isinstance(val, str) and not _is_mock(val):
                return val
            if isinstance(val, (int, float)) and not _is_mock(val):
                return str(val)
            return ""

        def _get_float(attr: str, default: float = 0.0) -> float:
            val = getattr(result, attr, None)
            if isinstance(val, (int, float)) and not _is_mock(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default
            return default

        def _get_int(attr: str, default: int = 0) -> int:
            val = getattr(result, attr, None)
            if isinstance(val, (int, float)) and not _is_mock(val):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default
            return default

        # Task and Problem Identifiers from BaseTaskResult or legacy aliases
        task_id = _get_str("task_id")
        inst_id = _get_str("instance_id")
        prob_id = _get_str("problem_id")

        if task_id:
            m.instance_id = task_id
            m.problem_id = task_id
        else:
            m.instance_id = inst_id or prob_id
            m.problem_id = prob_id or inst_id or "Trace"

        # Resolution and lifecycle status
        raw_res = getattr(result, "resolved", None)
        if raw_res is not None and not _is_mock(raw_res):
            m.resolved = bool(raw_res)
            raw_end = _get_str("end_status") or _get_str("status")
            if raw_end and raw_end.lower() not in ("success", "failed", "unknown"):
                m.end_status = raw_end.lower()
            else:
                m.end_status = "resolved" if m.resolved else "unresolved"
        else:
            raw_status = _get_str("end_status") or _get_str("status")
            m.end_status = raw_status.lower() if raw_status else "unknown"

        # Timing and execution duration
        m.wall_time_seconds = _get_float("wall_time_seconds", 0.0) or _get_float("duration_seconds", 0.0)

        # Financial cost
        m.total_cost_usd = _get_float("total_cost_usd", 0.0) or _get_float("cost_usd", 0.0)

        # Token & call counters
        m.llm_calls = _get_int("total_llm_calls", 0) or _get_int("llm_calls", 0)
        m.total_input_tokens = _get_int("total_input_tokens", 0)
        m.total_cached_input_tokens = _get_int("total_cached_input_tokens", 0)
        m.total_output_tokens = _get_int("total_output_tokens", 0)

        # Limits and caps
        m.max_budget_usd = _get_float("max_budget_usd", 0.0)
        m.max_tool_calls = _get_int("max_tool_calls", 0)
        m.max_submissions = _get_int("max_submissions", 0)
        m.max_time_minutes = _get_float("max_time_minutes", 0.0)
        m.tool_calls_used = _get_int("tool_calls_used", 0)
        m.submissions_used = _get_int("total_submissions", 0) or _get_int("submissions_used", 0)

        # Benchmark metadata dictionary
        raw_meta = getattr(result, "metadata", {})
        m.metadata = dict(raw_meta) if isinstance(raw_meta, dict) and not _is_mock(raw_meta) else {}

        # Domain metrics & scores
        raw_metrics = getattr(result, "metrics", {})
        m.metrics = dict(raw_metrics) if isinstance(raw_metrics, dict) and not _is_mock(raw_metrics) else {}

        raw_metric = _get_str("metric")
        if raw_metric:
            m.metric = raw_metric
        elif "metric" in m.metadata:
            m.metric = str(m.metadata["metric"])
        else:
            m.metric = ""

        # Score parsing
        score_val = _get_float("score", float("nan"))
        if math.isnan(score_val):
            score_val = _get_float("best_score", float("nan"))
        if not math.isnan(score_val):
            m.score = score_val
            m.final_score = score_val

        final_score_val = _get_float("final_score", float("nan"))
        if not math.isnan(final_score_val):
            m.final_score = final_score_val

        best_pub_val = _get_float("best_public_score", float("nan"))
        if math.isnan(best_pub_val):
            best_pub_val = _get_float("best_score", float("nan"))
        if not math.isnan(best_pub_val):
            m.best_public_score = best_pub_val

        best_priv_val = _get_float("best_private_score", float("nan"))
        if not math.isnan(best_priv_val):
            m.best_private_score = best_priv_val

        pub_scores = getattr(result, "public_scores", {})
        m.public_scores = dict(pub_scores) if isinstance(pub_scores, dict) and not _is_mock(pub_scores) else {}

        priv_scores = getattr(result, "private_scores", {})
        m.private_scores = dict(priv_scores) if isinstance(priv_scores, dict) and not _is_mock(priv_scores) else {}

        raw_sel = getattr(result, "selected_ids", [])
        m.selected_ids = list(raw_sel) if isinstance(raw_sel, (list, tuple)) and not _is_mock(raw_sel) else []

        raw_auto_sel = getattr(result, "auto_selected_ids", [])
        m.auto_selected_ids = list(raw_auto_sel) if isinstance(raw_auto_sel, (list, tuple)) and not _is_mock(raw_auto_sel) else []

        higher = getattr(result, "higher_is_better", True)
        m.higher_is_better = bool(higher) if not _is_mock(higher) else True

        # Trace context
        if trace and trace.entries:
            for e in reversed(trace.entries):
                if e.usage and "prompt_tokens" in e.usage:
                    m.last_input_tokens = e.usage["prompt_tokens"]
                    break

        return m

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        trace: SessionTrace | None = None,
        path: Path | None = None,
    ) -> ViewerMetadata:
        """Construct ViewerMetadata from serialized JSON dictionary."""
        meta = cls()
        summary = data.get("summary", {})
        final_metrics = data.get("final_metrics", {})
        if data.get("instance_id"):
            meta.instance_id = str(data["instance_id"])
        elif path is not None:
            meta.instance_id = path.stem.removeprefix("trace_")
        else:
            meta.instance_id = ""

        if data.get("problem_id"):
            meta.problem_id = str(data["problem_id"])
        else:
            meta.problem_id = meta.instance_id or "Trace"
        meta.wall_time_seconds = data.get("duration_s", 0.0) or (trace.duration if trace else 0.0)
        meta.total_input_tokens = summary.get(
            "total_prompt_tokens", final_metrics.get("total_prompt_tokens", 0)
        )
        meta.total_cached_input_tokens = summary.get(
            "total_cached_prompt_tokens", final_metrics.get("total_cached_tokens", 0)
        )
        meta.total_output_tokens = summary.get(
            "total_completion_tokens", final_metrics.get("total_completion_tokens", 0)
        )
        meta.llm_calls = summary.get("llm_calls", 0)

        explicit_status = data.get("end_status")
        if explicit_status:
            meta.end_status = explicit_status
        elif trace:
            for e in reversed(trace.entries):
                if e.event_type == "interrupted":
                    meta.end_status = "interrupted"
                    break
                elif e.event_type in ("error", "agent_error"):
                    meta.end_status = "agent_error"
                    break
                elif e.event_type == "budget_exceeded":
                    meta.end_status = "budget_exceeded"
                    break
                elif e.event_type == "final":
                    meta.end_status = "completed"
                    break
            else:
                meta.end_status = "completed" if trace.entries else "unknown"

        meta.resolved = data.get("resolved", None)
        meta.metric = data.get("metric", "")
        meta.metadata = data.get("metadata", {})
        meta.metrics = data.get("metrics", {})
        return meta


_ResultMeta = ViewerMetadata


# ---------------------------------------------------------------------------
# TraceViewer
# ---------------------------------------------------------------------------


class TraceViewer:
    """Post-session trace viewer that renders as self-contained HTML.

    Displays a summary card and an interactive, collapsible timeline of all
    trace entries including the system instruction, task prompt, and all
    model events.  Designed for use in Jupyter notebooks.

    Usage::

        # From an evaluation result or TaskResult
        viewer = TraceViewer(result)
        viewer.show()

        # From a saved trace JSON file
        viewer = TraceViewer.from_json("trace_instance_1.json")
        viewer.show()

        # Jupyter auto-display
        viewer  # triggers _repr_html_

    Args:
        result: An evaluation result, :class:`~adk_eval_core.runner.TaskResult`, or
            duck-typed benchmark result object. Either ``result`` or ``trace`` must be provided.
        trace: A bare :class:`~adk_eval_core.tracing.SessionTrace`. Use this
            when you have a trace but no result object (e.g. after loading
            from JSON via :meth:`from_json`).
        _meta: Internal result-level metadata. Populated automatically from
            ``result`` or from the JSON envelope when loading from file.
        formatter_registry: Pluggable formatter registry for tool call and response display.
    """

    def __init__(
        self,
        result: Any | None = None,
        *,
        trace: SessionTrace | None = None,
        _meta: ViewerMetadata | None = None,
        formatter_registry: ToolFormatterRegistry | None = None,
    ) -> None:
        self.formatter_registry = (
            formatter_registry
            if formatter_registry is not None
            else get_default_formatter_registry().copy()
        )

        if result is not None:
            self._trace = getattr(result, "trace", None) or trace or SessionTrace()
            self._meta = _meta or ViewerMetadata.from_result(result, trace=self._trace)
        elif trace is not None:
            self._trace = trace
            self._meta = _meta or ViewerMetadata()
        else:
            raise ValueError("Provide either result or trace.")

    @classmethod
    def from_json(cls, path: str | Path) -> TraceViewer:
        """Load a viewer from a saved trace JSON file.

        The JSON file must be in the format produced by
        :meth:`~adk_eval_core.tracing.SessionTrace.save`.

        Args:
            path: Path to the trace JSON file.

        Returns:
            A :class:`TraceViewer` ready to render.
        """
        path = Path(path)
        with open(path) as f:
            data: dict[str, Any] = json.load(f)

        trace = SessionTrace.from_dict(data)
        meta = ViewerMetadata.from_dict(data, trace=trace, path=path)
        return cls(trace=trace, _meta=meta)

    # ── Rendering ────────────────────────────────────────────────────

    def _repr_html_(self) -> str:
        """Return the full viewer HTML for Jupyter auto-display."""
        return self._render_html()

    def show(self) -> None:
        """Display the full viewer (summary + timeline) in a Jupyter cell."""
        try:
            from IPython.display import HTML, display  # type: ignore

            display(HTML(self._render_html()))
        except ImportError:
            print(self._trace.to_markdown())

    def summary(self) -> None:
        """Display the summary card only."""
        try:
            from IPython.display import HTML, display  # type: ignore

            html = render_summary_only_html(self._render_summary())
            display(HTML(html))
        except ImportError:
            m = self._meta
            print(f"Problem: {m.problem_id}  Metric: {m.metric}  Status: {m.end_status}")

    def timeline(self) -> None:
        """Display the timeline only (no summary card)."""
        try:
            from IPython.display import HTML, display  # type: ignore

            viewer_id = f"adk-viewer-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
            html = render_timeline_only_html(viewer_id, self._render_timeline())
            display(HTML(html))
        except ImportError:
            print(self._trace.to_markdown())

    def _render_html(self) -> str:
        """Render the complete viewer HTML string."""
        viewer_id = f"adk-viewer-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        return render_full_viewer_html(
            viewer_id=viewer_id,
            summary_html=self._render_summary(),
            timeline_html=self._render_timeline(),
        )

    # ── Summary card ─────────────────────────────────────────────────

    def _render_summary(self) -> str:
        m = self._meta
        trace = self._trace
        summary = trace.summarize()

        # Status badge
        status_cls = f"adk-viewer-status-{m.end_status.replace(' ', '_')}"
        status_icon = {
            "completed": "✓",
            "resolved": "✓",
            "unresolved": "✗",
            "interrupted": "⚠",
            "budget_exceeded": "💸",
            "agent_error": "✗",
            "error": "✗",
            "no_submissions": "—",
        }.get(m.end_status, "?")
        status_badge = (
            f'<span class="adk-viewer-status-badge {esc(status_cls)}">'
            f"{status_icon} {esc(m.end_status.replace('_', ' '))}"
            f"</span>"
        )

        header = (
            f'<div class="adk-viewer-summary-header">'
            f'<span class="adk-viewer-problem-id">{esc(m.instance_id or m.problem_id or "Trace")}</span>'
            + (
                f'<span class="adk-viewer-metric">{esc(m.metric)}</span>'
                if m.metric
                else ""
            )
            + status_badge
            + "</div>"
        )

        # Stats grid
        total_tokens = m.total_input_tokens + m.total_output_tokens
        wall_time = f"{m.wall_time_seconds:.1f}s" if m.wall_time_seconds else f"{trace.duration:.1f}s"

        def _stat(label: str, value: str, cls: str = "") -> str:
            cls_attr = f' class="adk-viewer-stat-value {cls}"' if cls else ' class="adk-viewer-stat-value"'
            return (
                f'<div class="adk-viewer-stat">'
                f'<span class="adk-viewer-stat-label">{esc(label)}</span>'
                f"<span{cls_attr}>{value}</span>"
                f"</div>"
            )

        score_val = m.score if not math.isnan(m.score) else m.final_score
        score_str = f"{score_val:.6f}" if not math.isnan(score_val) else ""
        best_public_str = (
            f"{m.best_public_score:.6f}" if not math.isnan(m.best_public_score) else ""
        )

        stats_items = []
        if score_str:
            stats_items.append(_stat("Score", score_str, "highlight"))
        if best_public_str:
            stats_items.append(_stat("Best Public", best_public_str))
        if m.resolved is not None:
            stats_items.append(
                _stat("Resolved", "YES" if m.resolved else "NO", "highlight" if m.resolved else "warn")
            )

        stats_items.append(
            _stat(
                "Wall Time",
                f"{wall_time} / {m.max_time_minutes}m" if m.max_time_minutes > 0 else wall_time,
                "warn" if m.max_time_minutes > 0 and m.wall_time_seconds / (m.max_time_minutes * 60) > 0.8 else "",
            )
        )

        if m.tool_calls_used > 0 or m.max_tool_calls > 0:
            stats_items.append(
                _stat(
                    "Tool Calls",
                    f"{m.tool_calls_used} / {m.max_tool_calls}" if m.max_tool_calls > 0 else f"{m.tool_calls_used}",
                    "warn" if m.max_tool_calls > 0 and m.tool_calls_used / m.max_tool_calls > 0.8 else "",
                )
            )

        if m.submissions_used > 0 or m.max_submissions > 0:
            stats_items.append(
                _stat(
                    "Submissions",
                    f"{m.submissions_used} / {m.max_submissions}" if m.max_submissions > 0 else f"{m.submissions_used}",
                    "warn" if m.max_submissions > 0 and m.submissions_used / m.max_submissions > 0.8 else "",
                )
            )

        stats_items.append(_stat("LLM Calls", f"{summary.get('llm_calls', m.llm_calls) or m.llm_calls}"))
        stats_items.append(_stat("Total Tokens", f"{total_tokens:,}" if total_tokens else f"{summary['total_tokens']:,}"))
        stats_items.append(
            _stat(
                "Input Tokens",
                f"{m.total_input_tokens:,}" + (f" ({m.total_cached_input_tokens:,} cached)" if m.total_cached_input_tokens > 0 else ""),
            )
        )
        stats_items.append(_stat("Output Tokens", f"{m.total_output_tokens:,}"))

        if m.max_budget_usd > 0 or m.total_cost_usd > 0:
            stats_items.append(
                _stat(
                    "Cost",
                    f"${m.total_cost_usd:.4f} / ${m.max_budget_usd:.2f}" if m.max_budget_usd > 0 else f"${m.total_cost_usd:.4f}",
                    "warn" if m.max_budget_usd > 0 and m.total_cost_usd / m.max_budget_usd > 0.8 else "",
                )
            )

        stats = f'<div class="adk-viewer-stats-grid">{"".join(stats_items)}</div>'

        # Score table
        score_table = ""
        if m.public_scores:
            selected_set = set(m.selected_ids)
            auto_set = set(m.auto_selected_ids)

            best_pub = m.best_public_score
            best_priv = m.best_private_score

            rows = ""
            for sub_id in m.public_scores:
                pub = m.public_scores.get(sub_id, float("nan"))
                priv = m.private_scores.get(sub_id, float("nan"))

                if sub_id in selected_set:
                    sel_html = '<span class="selected">← selected</span>'
                elif sub_id in auto_set:
                    sel_html = '<span class="auto-sel">← auto</span>'
                else:
                    sel_html = ""

                pub_cls = "best" if not math.isnan(pub) and not math.isnan(best_pub) and math.isclose(pub, best_pub) else ""
                priv_cls = "best" if not math.isnan(priv) and not math.isnan(best_priv) and math.isclose(priv, best_priv) else ""

                rows += (
                    f"<tr>"
                    f'<td class="sub-id">{esc(sub_id)}</td>'
                    f'<td class="{pub_cls}">{pub:.6f}</td>'
                    f'<td class="{priv_cls}">{priv:.6f}</td>'
                    f"<td>{sel_html}</td>"
                    f"</tr>"
                )

            score_table = (
                f'<table class="adk-viewer-scores-table">'
                f"<thead><tr>"
                f"<th>ID</th><th>Public</th><th>Private</th><th></th>"
                f"</tr></thead>"
                f"<tbody>{rows}</tbody>"
                f"</table>"
            )

        # Tool breakdown
        breakdown = summary.get("tool_call_breakdown", {})
        breakdown_html = ""
        if breakdown:
            chips = "".join(
                f'<span class="adk-viewer-tool-chip">{esc(name)}: {count}</span>'
                for name, count in sorted(breakdown.items())
            )
            breakdown_html = f'<div class="adk-viewer-tool-breakdown">{chips}</div>'

        return (
            '<div class="adk-viewer-summary">'
            + header
            + stats
            + score_table
            + breakdown_html
            + "</div>"
        )

    # ── Timeline ──────────────────────────────────────────────────────

    def _render_timeline(self) -> str:
        entries = self._trace.entries
        if not entries:
            return '<div class="adk-viewer-timeline-body" style="padding:16px;color:#6c7086;">No trace entries.</div>'

        non_agent_authors = {"harness", "system", "user", "tool", "compactor"}
        # Determine the root author from the first agent event (not harness/system/user/tool/compactor).
        root_author: str | None = None
        for e in entries:
            if e.author and e.author.lower() not in non_agent_authors:
                root_author = e.author
                break

        # Collect event types present in the trace for filter buttons.
        present_types: list[str] = []
        seen: set[str] = set()
        for e in entries:
            if e.event_type not in seen:
                seen.add(e.event_type)
                present_types.append(e.event_type)

        # Filter button strip.
        filter_buttons = "".join(
            f'<button class="adk-viewer-filter-btn" data-type="{esc(t)}">{esc(t.replace("_", " "))}</button>'
            for t in present_types
        )

        header = (
            '<div class="adk-viewer-timeline-header">'
            '<span class="adk-viewer-timeline-title">Timeline</span>'
            + filter_buttons
            + '<button class="adk-viewer-expand-all-btn">Expand all</button>'
            + "</div>"
        )

        # Identify paired tool_call / tool_response entries (by tool name, in FIFO order).
        paired_calls: set[int] = set()
        paired_responses: set[int] = set()
        pending_calls: dict[str, collections.deque[int]] = {}
        for i, e in enumerate(entries):
            if e.event_type == "tool_call":
                pending_calls.setdefault(e.tool_name, collections.deque()).append(i)
            elif e.event_type == "tool_response":
                q = pending_calls.get(e.tool_name)
                if q:
                    call_i = q.popleft()
                    paired_calls.add(call_i)
                    paired_responses.add(i)

        # Render each entry row.
        rows = ""
        for i, entry in enumerate(entries):
            is_sub = bool(
                entry.author
                and root_author
                and entry.author.lower() not in non_agent_authors
                and entry.author != root_author
            )

            entry_classes = ["adk-viewer-entry"]
            if is_sub:
                entry_classes.append("sub-agent")
            if i in paired_calls:
                entry_classes.append("tool-call-paired")
            if i in paired_responses:
                entry_classes.append("tool-response-paired")

            # Author tag (only for non-harness, non-root events)
            author_html = ""
            if entry.author and entry.author.lower() not in non_agent_authors:
                author_html = f'<span class="adk-viewer-author">{esc(entry.author)}</span>'

            # Detail sections
            sections = detail_sections(entry, self.formatter_registry)
            detail_html = ""
            if sections:
                detail_html = '<div class="adk-viewer-detail">'
                for label, content in sections:
                    detail_html += (
                        f'<div class="adk-viewer-detail-section">'
                        f'<div class="adk-viewer-detail-label">{esc(label)}</div>'
                        f'<div class="adk-viewer-detail-content">{esc(content)}</div>'
                        f"</div>"
                    )
                detail_html += "</div>"

            toggle = '<span class="adk-viewer-toggle">▶</span>' if sections else '<span class="adk-viewer-toggle"> </span>'

            rows += (
                f'<div class="{" ".join(entry_classes)}" data-type="{esc(entry.event_type)}">'
                f'<div class="adk-viewer-entry-row">'
                f'<span class="adk-viewer-ts">{esc(format_elapsed(entry.elapsed))}</span>'
                + badge(entry.event_type)
                + author_html
                + f'<span class="adk-viewer-summary-line">{summary_line(entry, self.formatter_registry)}</span>'
                + token_badge(entry)
                + toggle
                + "</div>"
                + detail_html
                + "</div>"
            )

        body = f'<div class="adk-viewer-timeline-body">{rows}</div>'
        return header + body


__all__ = ["TraceViewer", "ViewerMetadata"]
