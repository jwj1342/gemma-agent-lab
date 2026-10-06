"""ADK EventDisplayPlugin for real-time live terminal streaming."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from google.adk.plugins.base_plugin import BasePlugin

from adk_eval_core.ui.formatters import (
    ToolFormatterRegistry,
    get_default_formatter_registry,
)

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events.event import Event

    from adk_eval_core.ui.event_display import EventDisplay

logger = logging.getLogger(__name__)


class EventDisplayPlugin(BasePlugin):
    """ADK plugin intercepting agent events to render terminal UI updates."""

    def __init__(
        self,
        display: EventDisplay | Callable[[Any], None] | None = None,
        formatter_registry: ToolFormatterRegistry | None = None,
        name: str | None = None,
        on_event: Callable[[Any], None] | None = None,
        *,
        event_display: Any | None = None,
        display_manager: Any | None = None,
        trace_callback: Callable[[Any], None] | None = None,
    ) -> None:
        eff_name = (
            name
            if name is not None
            else (
                "swegemma_display"
                if (
                    trace_callback is not None
                    or display_manager is not None
                    or event_display is not None
                )
                else "event_display"
            )
        )
        super().__init__(name=eff_name)
        eff_display = display if display is not None else (event_display or display_manager)
        eff_on_event = on_event if on_event is not None else trace_callback

        if callable(eff_on_event):
            self._on_event: Callable[..., Any] | None = eff_on_event
            self.display: Any | None = (
                eff_display
                if (eff_display is not None and not callable(eff_display))
                else None
            )
        elif (
            callable(eff_display)
            and not hasattr(eff_display, "on_event")
            and not hasattr(eff_display, "log")
        ):
            self._on_event = eff_display
            self.display = None
        else:
            self._on_event = None
            self.display = eff_display

        self.display_manager = self.display
        self.trace_callback = self._on_event

        if formatter_registry is not None:
            self.formatter_registry = formatter_registry
        elif self.display and hasattr(self.display, "formatter_registry"):
            self.formatter_registry = self.display.formatter_registry
        else:
            self.formatter_registry = get_default_formatter_registry()

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Event | None:
        """Called when an ADK event occurs during agent execution."""
        if self.display is not None and hasattr(self.display, "on_event"):
            self.display.on_event(event)

        if self._on_event is not None and self._on_event != getattr(self.display, "on_event", None):
            self._on_event(event)

        if self.display is not None and hasattr(self.display, "log") and not hasattr(self.display, "on_event"):
            from rich.markup import escape

            event_type = getattr(event, "type", "event")
            content = getattr(event, "content", "")

            if event_type == "call":
                tool_name = getattr(event, "tool_name", "tool") or "tool"
                args = getattr(event, "tool_args", {}) or {}
                fmt = self.formatter_registry.get(tool_name)
                call_disp = fmt.format_call(tool_name, args)
                preview_str = f" [dim]{escape(str(call_disp.preview))}[/dim]" if call_disp.preview else ""
                self.display.log(
                    f"{call_disp.icon} [bold cyan]{escape(str(call_disp.title))}[/bold cyan]{preview_str}"
                )
            elif event_type == "response":
                tool_name = getattr(event, "tool_name", "tool") or "tool"
                raw_res = getattr(event, "tool_result", "")
                fmt = self.formatter_registry.get(tool_name)
                res_disp = fmt.format_response(tool_name, raw_res)
                if res_disp.is_error:
                    self.display.log(
                        f"{res_disp.icon} [bold red]{escape(str(res_disp.title))}: {escape(str(res_disp.preview))}[/bold red]"
                    )
                else:
                    self.display.log(
                        f"{res_disp.icon} [bold yellow]{escape(str(res_disp.title))}: {escape(str(res_disp.preview))}[/bold yellow]"
                    )
            elif event_type == "thought" and content:
                first_line = escape(str(content).split("\n")[0][:100])
                self.display.log(f"💭 [dim white]{first_line}...[/dim white]")
            elif event_type == "system_instruction" and content:
                first_line = escape(str(content).split("\n")[0][:100])
                self.display.log(f"📋 [bold magenta]System Prompt:[/bold magenta] {first_line}...")
            elif event_type == "task_prompt" and content:
                first_line = escape(str(content).split("\n")[0][:100])
                self.display.log(f"🎯 [bold cyan]Task Prompt:[/bold cyan] {first_line}...")
            elif event_type == "final":
                self.display.log("✅ [bold green]Final response received[/bold green]")

        return event


__all__ = ["EventDisplayPlugin"]
