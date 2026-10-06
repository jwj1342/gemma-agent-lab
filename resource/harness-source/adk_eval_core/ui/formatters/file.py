"""Formatter for filesystem operations and file manipulation tools."""

from __future__ import annotations

import posixpath
from typing import Any

from rich.markup import escape

from adk_eval_core.ui.formatters.models import (
    ToolCallDisplay,
    ToolResponseDisplay,
)


class FileToolFormatter:
    """Formatter for file operations (write_file, read_file, view_file, edit_file, replace_file_content)."""

    def _infer_syntax(self, path: str) -> str:
        ext = posixpath.splitext(path)[1].lower()
        mapping = {
            ".py": "python",
            ".json": "json",
            ".md": "markdown",
            ".sh": "bash",
            ".bash": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".html": "html",
            ".css": "css",
            ".js": "javascript",
            ".ts": "typescript",
            ".sql": "sql",
            ".diff": "diff",
            ".patch": "diff",
        }
        return mapping.get(ext, "text")

    def format_call(self, tool_name: str, args: dict[str, Any] | None) -> ToolCallDisplay:
        args = args or {}
        path = (
            args.get("filepath")
            or args.get("path")
            or args.get("file_path")
            or args.get("TargetFile")
            or args.get("AbsolutePath")
            or ""
        )
        path_str = str(path)
        syntax = self._infer_syntax(path_str)

        # Determine sub-action
        is_write = any(kw in tool_name for kw in ("write", "create", "save"))
        is_edit = any(kw in tool_name for kw in ("edit", "replace", "patch", "modify"))
        is_read = any(kw in tool_name for kw in ("read", "view", "show", "cat"))

        content = (
            args.get("content")
            or args.get("CodeContent")
            or args.get("ReplacementContent")
            or ""
        )
        content_str = str(content)

        details: list[tuple[str, str]] = []
        if path_str:
            details.append(("Path", path_str))

        if is_write:
            icon = "📝"
            lines_count = len(content_str.splitlines()) if content_str else 0
            preview = (
                f"({path_str}, {lines_count} lines)"
                if lines_count
                else f"({path_str})"
            )
            markup = f"[cyan]📝 {escape(tool_name)}[/cyan] [dim]({escape(path_str)}, {lines_count} lines)[/dim]"
            truncated_content = "\n".join(content_str.splitlines()[:50])
            if lines_count > 50:
                truncated_content += f"\n... ({lines_count - 50} more lines omitted)"
            if content_str:
                details.append(("Content", truncated_content))
        elif is_edit:
            icon = "✏️"
            preview = f"({path_str})"
            markup = f"[cyan]✏️ {escape(tool_name)}[/cyan] [dim]({escape(path_str)})[/dim]"
            target = args.get("TargetContent") or args.get("target") or ""
            if target:
                details.append(("Target", str(target)))
            if content_str:
                details.append(("Replacement", content_str))
        elif is_read:
            icon = "📖"
            preview = f"({path_str})"
            markup = f"[cyan]📖 {escape(tool_name)}[/cyan] [dim]({escape(path_str)})[/dim]"
        else:
            icon = "📝"
            preview = f"({path_str})" if path_str else ""
            markup = (
                f"[cyan]📝 {escape(tool_name)}[/cyan] [dim]({escape(path_str)})[/dim]"
                if path_str
                else f"[cyan]📝 {escape(tool_name)}[/cyan]"
            )

        return ToolCallDisplay(
            title=tool_name,
            preview=preview,
            body=content_str,
            syntax=syntax,
            icon=icon,
            markup=markup,
            metadata=args or {},
            details=details,
        )

    def format_response(self, tool_name: str, response: Any) -> ToolResponseDisplay:
        raw_str = response if isinstance(response, str) else str(response)
        is_error = False

        if (
            isinstance(response, dict)
            and (
                response.get("status") == "error"
                or response.get("is_error") is True
                or bool(response.get("error"))
            )
        ) or (
            isinstance(response, str)
            and (
                any(
                    response.lower().startswith(p)
                    for p in (
                        "error:",
                        "failed",
                        "filenotfounderror:",
                        "permissionerror:",
                        "isadirectoryerror:",
                    )
                )
                or "traceback (most recent call last)" in response.lower()
                or (
                    "error" in response.lower()
                    and ("failed" in response.lower() or "exception" in response.lower())
                    and tool_name not in ("read_file", "view_file", "cat", "show")
                )
            )
        ):
            is_error = True

        first_line = raw_str.splitlines()[0] if raw_str.splitlines() else ""
        preview = first_line[:120] + ("…" if len(first_line) > 120 else "")

        details = [("Response", raw_str)] if raw_str else []
        icon = "❌" if is_error else "✓"

        return ToolResponseDisplay(
            title=tool_name,
            preview=preview,
            body=raw_str,
            syntax="text",
            icon=icon,
            is_error=is_error,
            metadata=response if isinstance(response, dict) else {},
            details=details,
        )


__all__ = ["FileToolFormatter"]
