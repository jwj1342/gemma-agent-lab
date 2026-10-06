"""Generic fallback tool formatter for arbitrary or unknown tools."""

from __future__ import annotations

import json
from typing import Any

from rich.markup import escape

from adk_eval_core.ui.formatters.models import (
    ToolCallDisplay,
    ToolResponseDisplay,
)


class GenericToolFormatter:
    """Fallback tool formatter for arbitrary, generic, or unknown tools."""

    def format_call(self, tool_name: str, args: dict[str, Any] | None) -> ToolCallDisplay:
        """Format a generic tool call with JSON arguments."""
        args = args or {}
        if "request" in args:
            req = str(args["request"])
            preview = req[:100] + "..." if len(req) > 100 else req
            markup = f"[cyan]🤖 {escape(tool_name)}[/cyan] [dim]{escape(preview)}[/dim]"
            details = [("Request", req)]
            body_str = req
        elif args:
            try:
                flat = json.dumps(args, default=str)
                preview = flat[:100] + "..." if len(flat) > 100 else flat
                body_str = json.dumps(args, indent=2, default=str)
            except (TypeError, ValueError):
                preview = str(args)[:100]
                body_str = str(args)
            markup = f"[cyan]🤖 {escape(tool_name)}[/cyan] [dim]{escape(preview)}[/dim]"
            details = [("Args", body_str)]
        else:
            preview = ""
            body_str = "{}"
            markup = f"[cyan]🤖 {escape(tool_name)}[/cyan]"
            details = [("Args", "{}")]

        return ToolCallDisplay(
            title=tool_name,
            preview=preview,
            body=body_str,
            syntax="json",
            icon="🤖",
            markup=markup,
            metadata=args or {},
            details=details,
        )

    def format_response(self, tool_name: str, response: Any) -> ToolResponseDisplay:
        """Format a generic tool response."""
        is_error = False
        parsed: Any = None

        if isinstance(response, dict):
            parsed = response
        elif isinstance(response, str):
            try:
                parsed = json.loads(response)
            except (json.JSONDecodeError, ValueError):
                parsed = None

        if isinstance(parsed, dict) and (
            parsed.get("status") == "error"
            or parsed.get("is_error") is True
            or bool(parsed.get("error"))
            or (parsed.get("exit_code") is not None and parsed.get("exit_code") != 0)
        ):
            is_error = True

        raw_str = response if isinstance(response, str) else str(response)
        if isinstance(parsed, (dict, list)):
            try:
                flat = json.dumps(parsed, default=str)
                preview = flat[:150] + "..." if len(flat) > 150 else flat
            except Exception:  # noqa: BLE001
                preview = raw_str[:150]
        else:
            preview = raw_str[:150] + "..." if len(raw_str) > 150 else raw_str

        details: list[tuple[str, str]] = []
        if parsed is not None:
            try:
                details.append(("Response", json.dumps(parsed, indent=2, default=str)))
            except (TypeError, ValueError):
                details.append(("Response", str(parsed)))
        elif raw_str:
            details.append(("Response", raw_str))

        icon = "❌" if is_error else "→"
        markup = f"[dim]→ {escape(tool_name)}: {escape(preview)}[/dim]"

        return ToolResponseDisplay(
            title=tool_name,
            preview=preview,
            body=raw_str,
            syntax="json" if parsed is not None else "text",
            icon=icon,
            markup=markup,
            is_error=is_error,
            metadata=parsed if isinstance(parsed, dict) else {},
            details=details,
        )


DefaultToolFormatter = GenericToolFormatter

__all__ = [
    "DefaultToolFormatter",
    "GenericToolFormatter",
]
