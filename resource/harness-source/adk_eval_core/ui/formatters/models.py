"""Data models and protocol definitions for tool display formatters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCallDisplay:
    """Display representation of a tool call event.

    Attributes:
        title: Clean title or tool identifier (e.g., 'run_command', 'write_file').
        preview: Compact one-line summary string for HUDs and timeline rows.
        body: Full formatted body or content (e.g., full shell script, file body).
        syntax: Syntax highlighting language identifier (e.g., 'bash', 'python', 'json', 'text').
        icon: Display emoji or symbol (e.g., '🔧', '📝', '📖', '🐍', '🤖').
        markup: Pre-formatted Rich markup string for logging.
        metadata: Structured dictionary of auxiliary parameters.
        details: List of (section_label, formatted_content) pairs for expandable detail panels.
        headline: Alias for title for compatibility.
    """

    title: str = ""
    preview: str = ""
    body: str = ""
    syntax: str = ""
    icon: str = "🔧"
    markup: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    details: list[tuple[str, str]] = field(default_factory=list)

    def __init__(
        self,
        title: str = "",
        preview: str = "",
        body: str = "",
        syntax: str = "",
        icon: str = "🔧",
        markup: str = "",
        metadata: dict[str, Any] | None = None,
        details: list[tuple[str, str]] | None = None,
        *,
        headline: str | None = None,
    ) -> None:
        object.__setattr__(self, "title", headline if headline is not None else title)
        object.__setattr__(self, "preview", preview)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "syntax", syntax)
        object.__setattr__(self, "icon", icon)
        object.__setattr__(self, "markup", markup)
        object.__setattr__(self, "metadata", metadata if metadata is not None else {})
        object.__setattr__(self, "details", details if details is not None else [])

    @property
    def headline(self) -> str:
        """Alias for title for compatibility."""
        return self.title


@dataclass(frozen=True)
class ToolResponseDisplay:
    """Display representation of a tool execution response.

    Attributes:
        title: Clean headline or tool identifier.
        preview: Compact one-line summary string for HUDs and timeline rows.
        body: Full output or return value text.
        syntax: Syntax highlighting language identifier (e.g., 'text', 'json', 'diff').
        icon: Display emoji or symbol (e.g., '→', '✓', '❌').
        markup: Pre-formatted Rich markup string for logging.
        is_error: True if tool execution failed (non-zero exit, exception, error payload).
        metadata: Structured dictionary of parsed response data.
        details: List of (section_label, formatted_content) pairs for expandable detail panels.
        headline: Alias for title for compatibility.
    """

    title: str = ""
    preview: str = ""
    body: str = ""
    syntax: str = ""
    icon: str = "→"
    markup: str = ""
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    details: list[tuple[str, str]] = field(default_factory=list)

    def __init__(
        self,
        title: str = "",
        preview: str = "",
        body: str = "",
        syntax: str = "",
        icon: str | None = None,
        markup: str = "",
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
        details: list[tuple[str, str]] | None = None,
        *,
        headline: str | None = None,
    ) -> None:
        resolved_title = headline if headline is not None else title
        if icon is None:
            resolved_icon = "❌" if is_error else "→"
        else:
            resolved_icon = icon

        object.__setattr__(self, "title", resolved_title)
        object.__setattr__(self, "preview", preview)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "syntax", syntax)
        object.__setattr__(self, "icon", resolved_icon)
        object.__setattr__(self, "markup", markup)
        object.__setattr__(self, "is_error", is_error)
        object.__setattr__(self, "metadata", metadata if metadata is not None else {})
        object.__setattr__(self, "details", details if details is not None else [])

    @property
    def headline(self) -> str:
        """Alias for title for compatibility."""
        return self.title


@runtime_checkable
class ToolFormatter(Protocol):
    """Protocol for formatting tool calls and tool responses for terminal and HTML UI."""

    def format_call(self, tool_name: str, args: dict[str, Any]) -> ToolCallDisplay:
        """Format a tool invocation call."""
        ...

    def format_response(self, tool_name: str, response: Any) -> ToolResponseDisplay:
        """Format a tool execution response."""
        ...


__all__ = [
    "ToolCallDisplay",
    "ToolFormatter",
    "ToolResponseDisplay",
]
