"""Markdown rendering for SessionTrace."""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adk_eval_core.tracing.trace import SessionTrace


def _truncate(text: str, limit: int = 200) -> str:
    """Truncate text to *limit* characters, appending '...' if needed."""
    return text[:limit] + "..." if len(text) > limit else text


def _format_content(
    text: str, limit: int | None, multiline_mode: str, label: str = ""
) -> str:
    """Format text with optional truncation and multiline handling."""
    if not text:
        return ""
    if limit is not None and len(text) > limit:
        text = text[:limit] + "..."
    if "\n" in text:
        max_backticks = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
        fence = "`" * max(3, max_backticks + 1)
        safe_label = html.escape(label, quote=True)
        if multiline_mode == "collapse":
            safe_text = text.replace("<", "&lt;")
            return f"\n<details><summary>{safe_label}</summary>\n\n{fence}\n{safe_text}\n{fence}\n\n</details>"
        elif multiline_mode == "code":
            return f"\n{fence}\n{text}\n{fence}"
        elif multiline_mode == "quote":
            quoted = "\n".join(f"> {line}" for line in text.splitlines())
            return f"\n{quoted}"
    return text


def render_trace_markdown(
    trace: SessionTrace,
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
    lines = []
    if show_summary_section:
        lines.append("# Session Trace\n")
        summary = trace.summarize()
        lines.append(f"**Duration**: {trace.duration:.1f}s")
        lines.append(f"**Events**: {summary['total_events']}")
        lines.append(f"**Tool calls**: {summary['tool_calls']}")
        lines.append(f"**Tokens**: {summary['total_tokens']:,}")
        lines.append("")

    if show_tool_breakdown:
        summary = trace.summarize()
        if summary["tool_call_breakdown"]:
            lines.append("## Tool Call Breakdown\n")
            for name, count in sorted(summary["tool_call_breakdown"].items()):
                lines.append(f"- `{name}`: {count}")
            lines.append("")

    lines.append("## Timeline\n")
    for entry in trace.entries:
        if include_events is not None and entry.event_type not in include_events:
            continue
        if exclude_events is not None and entry.event_type in exclude_events:
            continue
        if entry.event_type == "usage" and not show_usage:
            continue

        if timestamp_mode == "elapsed":
            ts = f"[{entry.elapsed:7.2f}s]"
        elif timestamp_mode == "absolute":
            ts = f"[{datetime.fromtimestamp(entry.timestamp, tz=UTC).isoformat()}]"
        else:
            ts = ""
        author_str = f" ({entry.author})" if (entry.author and show_author) else ""
        prefix = f"{ts}{author_str}".lstrip()
        prefix_str = f"{prefix} " if prefix else ""

        if entry.event_type == "thinking":
            emoji = "💭 " if use_emojis else ""
            formatted = _format_content(entry.content, max_thought_len, multiline_mode, "Thinking")
            if "\n" in formatted:
                line = f"{prefix_str}{emoji}**Thinking**:{formatted}"
            else:
                line = f"{prefix_str}{emoji}**Thinking**: {formatted}"

        elif entry.event_type == "compaction":
            emoji = "🗜️ " if use_emojis else ""
            meta_info = (
                f" (range: {entry.metadata.get('start_timestamp', '?')} - {entry.metadata.get('end_timestamp', '?')})"
                if entry.metadata
                else ""
            )
            formatted = _format_content(entry.content, max_compaction_len, multiline_mode, "Compaction")
            if "\n" in formatted:
                line = f"{prefix_str}{emoji}**Compaction**{meta_info}:{formatted}"
            else:
                line = f"{prefix_str}{emoji}**Compaction**{meta_info}: {formatted}"

        elif entry.event_type == "tool_call":
            emoji = "🔧 " if use_emojis else ""
            args_str = json.dumps(entry.tool_args, default=str)
            formatted = _format_content(args_str, max_tool_args_len, multiline_mode, f"{entry.tool_name} args")
            line = f"{prefix_str}{emoji}**{entry.tool_name}**({formatted})"

        elif entry.event_type == "tool_response":
            emoji = "📤 " if use_emojis else ""
            formatted = _format_content(entry.tool_result, max_tool_result_len, multiline_mode, f"{entry.tool_name} result")
            if "\n" in formatted:
                line = f"{prefix_str}{emoji}**→ {entry.tool_name}**:{formatted}"
            else:
                line = f"{prefix_str}{emoji}**→ {entry.tool_name}**: `{formatted}`"

        elif entry.event_type == "text":
            emoji = "💬 " if use_emojis else ""
            formatted = _format_content(entry.content, max_text_len, multiline_mode, "Message")
            if "\n" in formatted:
                line = f"{prefix_str}{emoji}Message:{formatted}"
            else:
                line = f"{prefix_str}{emoji}{formatted}"

        elif entry.event_type == "final":
            emoji = "✅ " if use_emojis else ""
            formatted = _format_content(entry.content, max_final_len, multiline_mode, "Final Response")
            if "\n" in formatted:
                line = f"{prefix_str}{emoji}**Final**:{formatted}"
            else:
                line = f"{prefix_str}{emoji}**Final**: {formatted}"

        elif entry.event_type == "usage":
            if entry.usage and show_usage:
                emoji = "📊 " if use_emojis else ""
                line = f"{prefix_str}{emoji}Tokens: {entry.usage}"
            else:
                continue

        else:
            emoji = "📌 " if use_emojis else ""
            meta_str = f" | {entry.metadata}" if entry.metadata else ""
            formatted = _format_content(entry.content, max_text_len, multiline_mode, entry.event_type.capitalize())
            if "\n" in formatted:
                line = f"{prefix_str}{emoji}**{entry.event_type}**:{formatted}{meta_str}"
            else:
                line = f"{prefix_str}{emoji}**{entry.event_type}**: {formatted}{meta_str}"

        if entry.event_type != "usage" and entry.usage and show_usage:
            emoji = "📊 " if use_emojis else ""
            line += f" [{emoji}Tokens: {entry.usage}]"

        lines.append(line)

    lines.append("")
    return "\n".join(lines)
