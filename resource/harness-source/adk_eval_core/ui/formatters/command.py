"""Formatter for shell command execution tools."""

from __future__ import annotations

import json
from typing import Any

from rich.markup import escape

from adk_eval_core.ui.formatters.models import (
    ToolCallDisplay,
    ToolResponseDisplay,
)


class CommandToolFormatter:
    """Formatter for terminal shell command tools (run_command, bash, shell, exec)."""

    def format_call(self, tool_name: str, args: dict[str, Any] | None) -> ToolCallDisplay:
        args = args or {}
        cmd = args.get("command") or args.get("cmd") or args.get("script") or ""
        cmd_str = str(cmd)
        lines = cmd_str.splitlines()
        n = len(lines)
        if not lines:
            preview = ""
            count = ""
        elif n == 1:
            preview = lines[0][:100] + ("…" if len(lines[0]) > 100 else "")
            count = ""
        else:
            first = lines[0][:80]
            preview = f"{first}… ({n} lines)"
            count = f" ({n} lines)"

        details = [("Command", cmd_str)] if cmd_str else []
        markup = (
            f"[bold cyan]🔧 {escape(tool_name)}[/bold cyan][dim]{count}[/dim] [grey50]{escape(lines[0][:80] if lines else '')}[/grey50]"
            if lines
            else f"[bold cyan]🔧 {escape(tool_name)}[/bold cyan]"
        )

        return ToolCallDisplay(
            title=tool_name,
            preview=preview,
            body=cmd_str,
            syntax="bash",
            icon="🔧",
            markup=markup,
            metadata=args or {},
            details=details,
        )

    def format_response(self, tool_name: str, response: Any) -> ToolResponseDisplay:
        parsed: dict[str, Any] | None = None
        raw_str = response if isinstance(response, str) else str(response)

        if isinstance(response, dict):
            parsed = response
        elif isinstance(response, str):
            try:
                loaded = json.loads(response)
                if isinstance(loaded, dict):
                    parsed = loaded
            except (json.JSONDecodeError, ValueError):
                parsed = None

        exit_code = 0
        stdout = ""
        stderr = ""
        is_error = False

        if parsed is not None:
            exit_code = parsed.get("exit_code", 0)
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            status = parsed.get("status", "")
            if exit_code != 0 or status == "error" or parsed.get("is_error") is True:
                is_error = True
        else:
            stdout = raw_str

        details: list[tuple[str, str]] = []
        if is_error:
            err_msg = stderr or stdout or "Command failed"
            preview = f"error (exit {exit_code}): {err_msg[:100]}"
            if stderr:
                details.append(("Stderr", stderr))
            if stdout:
                details.append(("Stdout", stdout))
            if not details:
                details.append(("Error", err_msg))
            icon = "❌"
            markup = f"[bold red]→ error (exit {escape(str(exit_code))}): {escape(str(stderr or stdout)[:120])}[/bold red]"
        else:
            first_line = stdout.splitlines()[0] if stdout.splitlines() else ""
            preview = first_line[:120] + ("…" if len(first_line) > 120 else "")
            if stdout:
                details.append(("Stdout", stdout))
            if stderr:
                details.append(("Stderr", stderr))
            if not details:
                details.append(("Output", raw_str))
            icon = "🔧"
            markup = f"[green]→ {escape(first_line[:120])}[/green]" if first_line else ""

        return ToolResponseDisplay(
            title=tool_name,
            preview=preview,
            body=raw_str,
            syntax="text",
            icon=icon,
            markup=markup,
            is_error=is_error,
            metadata=parsed if parsed is not None else {},
            details=details,
        )


__all__ = ["CommandToolFormatter"]
