"""Template assembly, asset loading, and HTML rendering helpers for TraceViewer."""

from __future__ import annotations

import functools
import importlib.resources
import json
import logging
from pathlib import Path

from adk_eval_core.tracing.trace import TraceEntry
from adk_eval_core.ui.formatters import (
    ToolFormatterRegistry,
    get_default_formatter_registry,
)

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent / "assets"


@functools.lru_cache(maxsize=1)
def load_viewer_css() -> str:
    """Load the TraceViewer CSS stylesheet."""
    try:
        if hasattr(importlib.resources, "files"):
            return (
                importlib.resources.files("adk_eval_core.viewer")
                .joinpath("assets/viewer.css")
                .read_text(encoding="utf-8")
            )
    except (ImportError, AttributeError, OSError, TypeError) as e:
        logger.debug("Could not load viewer.css via importlib.resources: %s", e)
    css_path = _ASSETS_DIR / "viewer.css"
    return css_path.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _load_viewer_js_template() -> str:
    """Load the TraceViewer JavaScript template."""
    try:
        if hasattr(importlib.resources, "files"):
            return (
                importlib.resources.files("adk_eval_core.viewer")
                .joinpath("assets/viewer.js")
                .read_text(encoding="utf-8")
            )
    except (ImportError, AttributeError, OSError, TypeError) as e:
        logger.debug("Could not load viewer.js via importlib.resources: %s", e)
    js_path = _ASSETS_DIR / "viewer.js"
    return js_path.read_text(encoding="utf-8")


def load_viewer_js(viewer_id: str) -> str:
    """Load and format the TraceViewer JavaScript with the given viewer_id."""
    raw_js = _load_viewer_js_template()
    return raw_js.replace("__VIEWER_ID__", viewer_id)


def esc(text: str) -> str:
    """Escape text for safe HTML embedding."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def badge(event_type: str) -> str:
    """Render an event type badge."""
    known = {
        "thinking": "thinking",
        "tool_call": "tool_call",
        "tool_response": "tool_response",
        "text": "text",
        "final": "final",
        "error": "error",
        "usage": "usage",
        "system_instruction": "system_instruction",
        "task_prompt": "task_prompt",
        "problem_start": "problem_start",
        "budget_exceeded": "budget_exceeded",
        "interrupted": "interrupted",
        "compaction": "compaction",
    }
    cls = known.get(event_type, "custom")
    label = event_type.replace("_", " ")
    return f'<span class="adk-viewer-badge badge-{cls}">{esc(label)}</span>'



def format_elapsed(elapsed: float) -> str:
    """Format elapsed seconds as a compact timestamp string."""
    if elapsed < 60:
        return f"{elapsed:5.1f}s"
    m, s = divmod(int(elapsed), 60)
    return f"{m}m{s:02d}s"


def summary_line(entry: TraceEntry, registry: ToolFormatterRegistry | None = None) -> str:
    """Produce a one-line summary of a trace entry for the timeline row."""
    t = entry.event_type
    content = entry.content or ""
    fmt_reg = registry or get_default_formatter_registry()

    if t == "thinking":
        n = len(content)
        preview = content[:120].replace("\n", " ")
        if len(content) > 120:
            preview += "…"
        return f"<em>({n} chars)</em> {esc(preview)}"

    if t == "tool_call":
        fmt = fmt_reg.get(entry.tool_name or "")
        call_disp = fmt.format_call(entry.tool_name or "", entry.tool_args or {})
        preview_str = f" {esc(call_disp.preview)}" if call_disp.preview else ""
        return f"{esc(str(call_disp.icon))} <strong>{esc(call_disp.title)}</strong>{preview_str}"

    if t == "tool_response":
        fmt = fmt_reg.get(entry.tool_name or "")
        res_disp = fmt.format_response(entry.tool_name or "", entry.tool_result or "")
        prefix = "❌" if res_disp.is_error else esc(str(res_disp.icon))
        preview_str = f": {esc(res_disp.preview)}" if res_disp.preview else ""
        return f"{prefix} <strong>{esc(res_disp.title)}</strong>{preview_str}"

    if t in ("text", "final"):
        preview = content[:160].replace("\n", " ")
        if len(content) > 160:
            preview += "…"
        return esc(preview)

    if t == "system_instruction":
        lines = content.splitlines()
        preview = lines[0][:120] if lines else ""
        return esc(preview)

    if t == "task_prompt":
        lines = content.splitlines()
        preview = lines[0][:120] if lines else ""
        return esc(preview)

    if t == "usage":
        if entry.usage:
            total = entry.usage.get("total_tokens", 0)
            prompt = entry.usage.get("prompt_tokens", 0)
            comp = entry.usage.get("completion_tokens", 0)
            cached = entry.usage.get("cached_tokens", 0)
            if cached > 0:
                return f"in:{prompt:,} [{cached:,} cached] + out:{comp:,} = {total:,} tokens"
            return f"in:{prompt:,} + out:{comp:,} = {total:,} tokens"
        return ""

    if t == "problem_start":
        return esc(content)

    # Generic fallback
    preview = content[:160].replace("\n", " ") if content else ""
    if len(content) > 160:
        preview += "…"
    return esc(preview)


def detail_sections(
    entry: TraceEntry, registry: ToolFormatterRegistry | None = None
) -> list[tuple[str, str]]:
    """Return (label, preformatted content) pairs for the expanded detail view."""
    t = entry.event_type
    fmt_reg = registry or get_default_formatter_registry()
    sections: list[tuple[str, str]] = []

    if t == "thinking":
        sections.append(("Thought", entry.content))

    elif t == "tool_call":
        fmt = fmt_reg.get(entry.tool_name or "")
        call_disp = fmt.format_call(entry.tool_name or "", entry.tool_args or {})
        if call_disp.details:
            sections.extend(call_disp.details)
        elif entry.tool_args is not None:
            sections.append(("Args", json.dumps(entry.tool_args, indent=2, default=str)))

    elif t == "tool_response":
        fmt = fmt_reg.get(entry.tool_name or "")
        res_disp = fmt.format_response(entry.tool_name or "", entry.tool_result or "")
        if res_disp.details:
            sections.extend(res_disp.details)
        if entry.tool_result:
            try:
                parsed = json.loads(entry.tool_result)
                label = "Raw Response" if res_disp.details else "Response"
                sections.append((label, json.dumps(parsed, indent=2, default=str)))
            except (json.JSONDecodeError, ValueError):
                if not res_disp.details:
                    sections.append(("Response", entry.tool_result))

    elif t in ("text", "final"):
        sections.append(("Message", entry.content))

    elif t == "system_instruction":
        sections.append(("System Instruction", entry.content))

    elif t == "task_prompt":
        sections.append(("Task Prompt", entry.content))

    elif t == "problem_start":
        if entry.metadata:
            sections.append(("Metadata", json.dumps(entry.metadata, indent=2, default=str)))

    elif entry.content:
        sections.append(("Content", entry.content))

    if entry.metadata and t not in ("problem_start",):
        sections.append(("Metadata", json.dumps(entry.metadata, indent=2, default=str)))

    if entry.usage:
        sections.append(("Token Usage", json.dumps(entry.usage, indent=2, default=str)))

    if entry.model_version:
        sections.append(("Model", entry.model_version))

    return sections


def token_badge(entry: TraceEntry) -> str:
    """Render a small token count badge if this entry has usage data."""
    if not entry.usage:
        return ""
    total = entry.usage.get("total_tokens", 0)
    if not total:
        return ""
    cached = entry.usage.get("cached_tokens", 0)
    if cached > 0:
        label = f"{total:,} tok ({cached:,} cached)"
    else:
        label = f"{total:,} tok"
    return f'<span class="adk-viewer-token-badge">{esc(label)}</span>'


def render_full_viewer_html(
    viewer_id: str,
    summary_html: str,
    timeline_html: str,
) -> str:
    """Assemble complete HTML document for TraceViewer."""
    css = load_viewer_css()
    js = load_viewer_js(viewer_id)
    return (
        f'<div class="adk-viewer" id="{viewer_id}">'
        f"<style>{css}</style>"
        f"{summary_html}"
        f"{timeline_html}"
        f"<script>{js}</script>"
        f"</div>"
    )


def render_summary_only_html(summary_html: str) -> str:
    """Assemble HTML document for summary card only."""
    css = load_viewer_css()
    return (
        f'<div class="adk-viewer" style="font-family: sans-serif;">'
        f"<style>{css}</style>"
        f"{summary_html}"
        f"</div>"
    )


def render_timeline_only_html(viewer_id: str, timeline_html: str) -> str:
    """Assemble HTML document for timeline only."""
    css = load_viewer_css()
    js = load_viewer_js(viewer_id)
    return (
        f'<div class="adk-viewer" id="{viewer_id}" style="font-family: sans-serif;">'
        f"<style>{css}</style>"
        f"{timeline_html}"
        f"<script>{js}</script>"
        f"</div>"
    )



__all__ = [
    "badge",
    "detail_sections",
    "esc",
    "format_elapsed",
    "load_viewer_css",
    "load_viewer_js",
    "render_full_viewer_html",
    "render_summary_only_html",
    "render_timeline_only_html",
    "summary_line",
    "token_badge",
]
