"""Terminal live-display manager coordinating StatusPanel and scrolling logs."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any, Self

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from adk_eval_core.ui.formatters import (
    ToolFormatterRegistry,
    get_default_formatter_registry,
)
from adk_eval_core.ui.hud import StatusPanel
from adk_eval_core.ui.protocol import EvaluationContextProtocol
from adk_eval_core.utils.utils import unwrap_tool_response

logger = logging.getLogger(__name__)

_DETAIL_MAX_LINES = 12


@dataclass
class _EventContext:
    """Per-event display context computed once in on_event."""

    ts: str
    indent: str
    author_tag: str
    is_sub: bool
    author: str


@dataclass
class _DetailEntry:
    """Stores full content for the auto-expanding detail panel."""

    icon: str
    title: str
    sections: list[tuple[str, str]] = field(default_factory=list)


class EventDisplay:
    """Live display with a persistent status HUD and scrolling event log.

    Uses ``rich.live.Live`` to keep the HUD fixed at the bottom of the
    terminal while event log lines scroll above it.
    """

    def __init__(
        self,
        console: Console | None = None,
        title: str = "ADK Evaluation",
        ctx: EvaluationContextProtocol | Any = None,
        formatter_registry: ToolFormatterRegistry | None = None,
        *,
        problem_id: str = "",
        metric: str = "",
        higher_is_better: bool = True,
        root_agent_name: str = "",
        enable_live: bool = True,
        **kwargs: Any,
    ) -> None:
        self.enable_live = enable_live
        self.console = console or Console()
        self._console = self.console
        self.title = title
        self.ctx = ctx
        self._ctx = ctx
        self.formatter_registry = (
            formatter_registry
            if formatter_registry is not None
            else get_default_formatter_registry().copy()
        )
        self.panel = StatusPanel(
            ctx=self.ctx,
            title=title,
            problem_id=problem_id,
            metric=metric,
            higher_is_better=higher_is_better,
            **kwargs,
        )
        self._panel = self.panel
        self._live: Live | None = None
        self._disp_id: str | None = None
        self._start: float | None = None
        self._detail: _DetailEntry | None = None
        self._root_author: str | None = root_agent_name or None
        self._log_buffer: list[str] = []
        self._persistent_logs: list[str] = []
        self._lock = threading.RLock()
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None

    @property
    def live(self) -> Live | None:
        return self._live

    @live.setter
    def live(self, val: Live | None) -> None:
        self._live = val

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Reset the elapsed timer."""
        self._start = time.time()
        self.panel.reset()
        self._detail = None

    def start(self) -> None:
        """Start live display."""
        self._start = time.time()
        self.panel.reset()
        if self.enable_live:
            if self._console.is_jupyter:
                import uuid

                from IPython.display import (  # type: ignore[import-not-found]
                    display as ipy_display,
                )

                self._disp_id = f"event_display_{uuid.uuid4().hex}"
                self._persistent_logs = []
                ipy_display(
                    self._get_jupyter_renderable(self._build_live_renderable()),
                    display_id=self._disp_id,
                )
            else:
                self._live = Live(
                    self._build_live_renderable(),
                    console=self._console,
                    refresh_per_second=4,
                )
                self._live.start()

        try:
            loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue()
            self._worker_task = loop.create_task(self._worker_loop())
        except RuntimeError:
            self._queue = None
            self._worker_task = None

    async def _worker_loop(self) -> None:
        """Background worker task that pulls events sequentially from the queue."""
        assert self._queue is not None
        while True:
            try:
                event = await self._queue.get()
                self._process_event(event)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001, S110
                pass

    def _drain_queue(self) -> None:
        """Synchronously process all pending events in the async queue."""
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    event = self._queue.get_nowait()
                    self._process_event(event)
                    self._queue.task_done()
                except Exception:  # noqa: BLE001
                    break

    def stop(self) -> None:
        """Stop live display."""
        if self._worker_task is not None:
            self._drain_queue()
            self._worker_task.cancel()
            self._worker_task = None
            self._queue = None

        if self._live:
            self._live.update(self._build_live_renderable())
            self._live.stop()
            self._live = None
        elif self._console.is_jupyter and self._disp_id:
            from IPython.display import update_display  # type: ignore[import-not-found]

            update_display(
                self._get_jupyter_renderable(self._build_live_renderable()),
                display_id=self._disp_id,
            )
            self._disp_id = None
        self._flush_logs()

    def log(self, text: str, style: str = "") -> None:
        """Log a line to the console live feed."""
        self._drain_queue()
        t = Text(text, style=style)
        if self._live:
            self._console.print(t)
            self._live.update(self._build_live_renderable())
        else:
            self._console.print(t)

    def log_system_instruction(self, instruction: str, author: str = "system") -> None:
        """Log system instruction / prompt to the live scrolling feed and detail panel."""
        self._drain_queue()
        with self._lock:
            if self._start is None:
                self._start = time.time()
            ectx = _EventContext(
                ts=self._ts(),
                indent="",
                author_tag="",
                is_sub=False,
                author=author,
            )
            self._handle_system_instruction(instruction, ectx)
            self._flush_logs()
            self._refresh_hud()

    log_system_prompt = log_system_instruction

    def log_task_prompt(self, prompt: str, author: str = "user") -> None:
        """Log task / user prompt to the live scrolling feed and detail panel."""
        self._drain_queue()
        with self._lock:
            if self._start is None:
                self._start = time.time()
            ectx = _EventContext(
                ts=self._ts(),
                indent="",
                author_tag="",
                is_sub=False,
                author=author,
            )
            self._handle_task_prompt(prompt, ectx)
            self._flush_logs()
            self._refresh_hud()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.stop()

    # -- output helpers -----------------------------------------------------

    def _ts(self) -> str:
        """Return a rich-formatted elapsed timestamp."""
        if self._start is None:
            self._start = time.time()
        elapsed = time.time() - self._start
        return f"[dim]\\[{elapsed:6.1f}s][/dim]"

    def _print(self, markup: str) -> None:
        """Buffer a log line to be printed later."""
        self._log_buffer.append(markup)
        if self._console.is_jupyter:
            self._persistent_logs.append(markup)

    def _flush_logs(self) -> None:
        """Print all buffered log lines at once and clear the buffer."""
        import contextlib

        with self._lock:
            logs = self._log_buffer
            self._log_buffer = []

        if logs and not self._console.is_jupyter:
            self._console.print(*logs, sep="\n", highlight=False)
        if hasattr(self._console, "file") and hasattr(self._console.file, "flush"):
            with contextlib.suppress(Exception):
                self._console.file.flush()

    def _get_jupyter_renderable(self, renderable: Any) -> Any:
        from rich.jupyter import JupyterRenderable, _render_segments

        segments = list(self._console.render(renderable))
        html = _render_segments(segments)
        text = self._console._render_buffer(segments)
        return JupyterRenderable(html, text)

    def _refresh_hud(self) -> None:
        """Repaint the HUD panel."""
        if self._live:
            self._live.update(self._build_live_renderable(), refresh=False)
        elif self._console.is_jupyter and self._disp_id:
            from IPython.display import update_display  # type: ignore[import-not-found]

            update_display(
                self._get_jupyter_renderable(self._build_live_renderable()),
                display_id=self._disp_id,
            )

    def _build_live_renderable(self) -> Group:
        """Build the full Live renderable: detail panel + HUD."""
        with self._lock:
            parts = []
            if self._console.is_jupyter and self._persistent_logs:
                log_text = Text()
                for line in self._persistent_logs:
                    log_text.append_text(Text.from_markup(line))
                    log_text.append("\n")
                parts.append(log_text)

            if self._detail:
                parts.append(self._render_detail())
            parts.append(self.panel.render())
            return Group(*parts)

    def _render_detail(self) -> Panel:
        """Render the detail panel for the most recent substantial event."""
        d = self._detail
        assert d is not None
        header = Text(f"  {d.icon} {d.title}", style="bold")

        all_lines: list[Text] = []
        for label, content in d.sections:
            all_lines.append(Text(f"\n  {label}:", style="bold dim"))
            content_str = str(content) if content is not None else ""
            for line in content_str.strip().splitlines():
                all_lines.append(Text(f"    {line}"))

        if len(all_lines) > _DETAIL_MAX_LINES:
            skipped = len(all_lines) - _DETAIL_MAX_LINES
            all_lines = [
                Text(f"    ... ({skipped} earlier lines)", style="dim")
            ] + all_lines[-_DETAIL_MAX_LINES:]

        return Panel(
            Group(header, *all_lines),
            title="[dim]Latest Detail[/dim]",
            border_style="dim",
            padding=(0, 1),
        )

    def _set_detail(
        self, icon: str, title: str, sections: list[tuple[str, str]]
    ) -> None:
        """Set the detail panel content."""
        self._detail = _DetailEntry(icon=icon, title=title, sections=sections)

    def _append_or_set_detail(
        self, icon: str, title: str, sections: list[tuple[str, str]]
    ) -> None:
        """Append sections to existing detail if title matches, else create new."""
        if self._detail and self._detail.title == title:
            self._detail.sections.extend(sections)
        else:
            self._detail = _DetailEntry(icon=icon, title=title, sections=sections)

    def _append_subagent_detail(self, author: str, action: str, content: str) -> None:
        """Append sub-agent activity to the current detail panel."""
        label = f"{author} → {action}"
        if self._detail:
            self._detail.sections.append((label, content))
        else:
            self._detail = _DetailEntry(
                icon="🤖", title=author, sections=[(label, content)]
            )

    def _update_detail(
        self,
        icon: str,
        title: str,
        sections: list[tuple[str, str]],
        *,
        is_sub: bool = False,
        author: str = "",
        append: bool = False,
    ) -> None:
        """Route detail updates: append for sub-agents, set/replace for root."""
        if is_sub:
            for label, content in sections:
                self._append_subagent_detail(author, label, content)
        elif append:
            self._append_or_set_detail(icon, title, sections)
        else:
            self._set_detail(icon, title, sections)

    # -- event callback -----------------------------------------------------

    def _is_subagent_event(self, event: Any) -> tuple[bool, str]:
        """Check if this event comes from a sub-agent."""
        author = getattr(event, "author", None) or ""
        non_agent_authors = {"harness", "system", "user", "tool", "compactor"}
        is_non_agent = author.lower() in non_agent_authors
        if self._root_author is None and author and not is_non_agent:
            self._root_author = author
        is_sub = bool(
            author
            and not is_non_agent
            and self._root_author
            and author != self._root_author
        )
        return is_sub, author

    def on_event(self, event: Any) -> None:
        """Callback to receive evaluation / runner events."""
        if self._queue is not None:
            self._queue.put_nowait(event)
        else:
            self._process_event(event)

    def _process_event(self, event: Any) -> None:
        """Actual event processing logic."""
        with self._lock:
            if self._start is None:
                self._start = time.time()

            is_sub, author = self._is_subagent_event(event)
            ectx = _EventContext(
                ts=self._ts(),
                indent="  [dim]\u2502[/dim] " if is_sub else "",
                author_tag=f"[dim]({escape(str(author))})[/dim] " if is_sub else "",
                is_sub=is_sub,
                author=author,
            )

            # Check for custom prompt event types (TraceEntry, custom dict, or duck-typed event)
            event_type = (
                getattr(event, "event_type", None)
                or getattr(event, "type", None)
                or (event.get("event_type") if isinstance(event, dict) else None)
            )
            if event_type in ("system_instruction", "task_prompt"):
                content_val = (
                    getattr(event, "content", None)
                    or (event.get("content") if isinstance(event, dict) else "")
                    or ""
                )
                if event_type == "system_instruction":
                    self._handle_system_instruction(str(content_val), ectx)
                else:
                    self._handle_task_prompt(str(content_val), ectx)
                self._flush_logs()
                self._refresh_hud()
                return

            # Check for compaction event
            actions = getattr(event, "actions", None)
            comp = (
                getattr(actions, "compaction", None)
                if (actions and type(actions).__name__ != "MagicMock")
                else None
            )
            if comp:
                summary_text = ""
                if (
                    hasattr(comp, "compacted_content")
                    and comp.compacted_content
                    and hasattr(comp.compacted_content, "parts")
                ):
                    for part in comp.compacted_content.parts:
                        if hasattr(part, "text") and part.text:
                            summary_text += part.text
                start_ts = getattr(comp, "start_timestamp", "?")
                end_ts = getattr(comp, "end_timestamp", "?")
                self._print(
                    f"  {ectx.ts} {ectx.indent}{ectx.author_tag}[bold yellow]🗜️ Compaction[/bold yellow] [dim](range: {escape(str(start_ts))} - {escape(str(end_ts))})[/dim]"
                )
                self._update_detail(
                    "🗜️",
                    "Compaction",
                    [
                        ("Range", f"{start_ts} - {end_ts}"),
                        ("Summary", summary_text.strip()),
                    ],
                    is_sub=ectx.is_sub,
                    author=ectx.author,
                )
                self._flush_logs()
                self._refresh_hud()
                return

            content = getattr(event, "content", None)
            if not content or not hasattr(content, "parts") or not content.parts:
                return

            for part in content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    self._handle_tool_call(part.function_call, ectx)

                elif hasattr(part, "function_response") and part.function_response:
                    self._handle_tool_response(part.function_response, content, ectx)

                elif (
                    hasattr(part, "thought")
                    and part.thought
                    and getattr(part, "text", None)
                ):
                    text = part.text.strip()
                    n = len(text)
                    preview = text[:120] + "..." if n > 120 else text
                    self._print(
                        f'  {ectx.ts} {ectx.indent}{ectx.author_tag}[dim]💭 ({n} chars) "{escape(preview)}"[/dim]'
                    )
                    self._update_detail(
                        "💭",
                        "Thinking",
                        [("Thought", text)],
                        is_sub=ectx.is_sub,
                        author=ectx.author,
                    )

                elif hasattr(part, "text") and part.text and part.text.strip():
                    self._handle_text(part, event, content, ectx)

        self._flush_logs()
        self._refresh_hud()

    def _handle_tool_call(self, fc: Any, ectx: _EventContext) -> None:
        """Format and print a tool call event using ToolFormatterRegistry."""
        name = getattr(fc, "name", "") or ""
        args = dict(fc.args) if getattr(fc, "args", None) else {}

        fmt = self.formatter_registry.get(name)
        call_disp = fmt.format_call(name, args)

        if call_disp.markup:
            self._print(f"  {ectx.ts} {ectx.indent}{ectx.author_tag}{call_disp.markup}")
        else:
            suffix = f" [dim]{escape(call_disp.preview)}[/dim]" if call_disp.preview else ""
            self._print(
                f"  {ectx.ts} {ectx.indent}{ectx.author_tag}[cyan]{call_disp.icon} {escape(call_disp.title)}[/cyan]{suffix}"
            )

        if call_disp.details:
            self._update_detail(
                call_disp.icon,
                call_disp.title,
                call_disp.details,
                is_sub=ectx.is_sub,
                author=ectx.author,
            )

    def _handle_tool_response(self, fr: Any, content: Any, ectx: _EventContext) -> None:
        """Format and print a tool response event using ToolFormatterRegistry."""
        name = getattr(fr, "name", "") or ""
        parsed = unwrap_tool_response(fr)
        if parsed is None:
            return

        fmt = self.formatter_registry.get(name)
        res_disp = fmt.format_response(name, parsed)

        if res_disp.markup:
            self._print(f"  {ectx.ts}   {ectx.indent}{ectx.author_tag}{res_disp.markup}")
        else:
            if res_disp.is_error:
                self._print(
                    f"  {ectx.ts}   {ectx.indent}{ectx.author_tag}[bold red]{res_disp.icon} {escape(res_disp.preview)}[/bold red]"
                )
            else:
                self._print(
                    f"  {ectx.ts}   {ectx.indent}{ectx.author_tag}[green]{res_disp.icon} {escape(res_disp.preview)}[/green]"
                )

        if res_disp.details:
            self._update_detail(
                res_disp.icon,
                res_disp.title,
                res_disp.details,
                is_sub=ectx.is_sub,
                author=ectx.author,
                append=True,
            )

    def _handle_text(
        self, part: Any, event: Any, content: Any, ectx: _EventContext
    ) -> None:
        """Format and print a text or final-response event."""
        role = getattr(content, "role", "") or ""
        author = (ectx.author or "").lower()
        if role == "user" or author in ("user", "harness"):
            self._handle_task_prompt(part.text, ectx)
            return

        is_final_raw = hasattr(event, "is_final_response") and event.is_final_response()
        has_fn_parts = any(
            getattr(p, "function_call", None) or getattr(p, "function_response", None)
            for p in content.parts
        )
        is_final = is_final_raw and not has_fn_parts

        limit = 200 if is_final else 150
        text = part.text.strip()
        preview = text[:limit] + "..." if len(text) > limit else text

        if is_final:
            self._print(
                f"  {ectx.ts} {ectx.indent}{ectx.author_tag}[bold green]✅ Agent finished:[/bold green]"
                f" [dim]{escape(preview)}[/dim]"
            )
            self._update_detail(
                "✅",
                "Agent Finished",
                [("Message", text)],
                is_sub=ectx.is_sub,
                author=ectx.author,
            )
        else:
            self._print(
                f"  {ectx.ts} {ectx.indent}{ectx.author_tag}[yellow]💬 {escape(preview)}[/yellow]"
            )
            self._update_detail(
                "💬",
                "Agent Message",
                [("Message", text)],
                is_sub=ectx.is_sub,
                author=ectx.author,
            )

    def _handle_system_instruction(self, text: str, ectx: _EventContext) -> None:
        """Format and print a system instruction event."""
        clean_text = text.strip()
        lines = clean_text.splitlines()
        first_line = lines[0] if lines else ""
        preview = first_line[:120] + "..." if len(first_line) > 120 else first_line
        self._print(
            f"  {ectx.ts} {ectx.indent}[bold magenta]📋 System Prompt:[/bold magenta] [dim]{escape(preview)}[/dim]"
        )
        self._update_detail(
            "📋",
            "System Instruction",
            [("System Instruction", clean_text)],
            is_sub=ectx.is_sub,
            author=ectx.author,
        )

    def _handle_task_prompt(self, text: str, ectx: _EventContext) -> None:
        """Format and print a task prompt event."""
        clean_text = text.strip()
        lines = clean_text.splitlines()
        first_line = lines[0] if lines else ""
        preview = first_line[:120] + "..." if len(first_line) > 120 else first_line
        self._print(
            f"  {ectx.ts} {ectx.indent}[bold cyan]🎯 Task Prompt:[/bold cyan] [dim]{escape(preview)}[/dim]"
        )
        self._update_detail(
            "🎯",
            "Task Prompt",
            [("Task Prompt", clean_text)],
            is_sub=ectx.is_sub,
            author=ectx.author,
        )


__all__ = ["EventDisplay"]

