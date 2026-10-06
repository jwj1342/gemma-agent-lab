"""Formatter for Python REPL and script execution tools."""

from __future__ import annotations

from typing import Any

from adk_eval_core.ui.formatters.models import (
    ToolCallDisplay,
    ToolResponseDisplay,
)


class PythonToolFormatter:
    """Formatter for Python REPL / script execution tools (python, python_repl, py_exec)."""

    def format_call(self, tool_name: str, args: dict[str, Any] | None) -> ToolCallDisplay:
        args = args or {}
        code = args.get("code") or args.get("script") or args.get("source") or ""
        code_str = str(code)
        lines = code_str.splitlines()
        first = lines[0][:80] if lines else ""
        preview = f"{first}…" if len(lines) > 1 or len(first) > 80 else first

        details = [("Code", code_str)] if code_str else []

        return ToolCallDisplay(
            title=tool_name,
            preview=preview,
            body=code_str,
            syntax="python",
            icon="🐍",
            metadata=args or {},
            details=details,
        )

    def format_response(self, tool_name: str, response: Any) -> ToolResponseDisplay:
        raw_str = response if isinstance(response, str) else str(response)
        is_error = (
            "Traceback (most recent call last)" in raw_str
            or "Error:" in raw_str
        )

        if isinstance(response, dict) and (
            response.get("status") == "error" or response.get("is_error")
        ):
            is_error = True

        first_line = raw_str.splitlines()[0] if raw_str.splitlines() else ""
        preview = first_line[:120] + ("…" if len(first_line) > 120 else "")

        details = [("Output", raw_str)] if raw_str else []
        icon = "❌" if is_error else "🐍"

        return ToolResponseDisplay(
            title=tool_name,
            preview=preview,
            body=raw_str,
            syntax="python" if not is_error else "text",
            icon=icon,
            is_error=is_error,
            metadata=response if isinstance(response, dict) else {},
            details=details,
        )


__all__ = ["PythonToolFormatter"]
