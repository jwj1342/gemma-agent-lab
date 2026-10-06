"""Pluggable tool call and response formatting for terminal HUD and HTML viewers."""

from __future__ import annotations

from adk_eval_core.ui.formatters.command import CommandToolFormatter
from adk_eval_core.ui.formatters.file import FileToolFormatter
from adk_eval_core.ui.formatters.generic import (
    DefaultToolFormatter,
    GenericToolFormatter,
)
from adk_eval_core.ui.formatters.models import (
    ToolCallDisplay,
    ToolFormatter,
    ToolResponseDisplay,
)
from adk_eval_core.ui.formatters.python import PythonToolFormatter
from adk_eval_core.ui.formatters.registry import (
    ToolFormatterRegistry,
    create_default_tool_formatter_registry,
    get_default_formatter_registry,
    register_tool_formatter,
    set_default_formatter_registry,
)

__all__ = [
    "CommandToolFormatter",
    "DefaultToolFormatter",
    "FileToolFormatter",
    "GenericToolFormatter",
    "PythonToolFormatter",
    "ToolCallDisplay",
    "ToolFormatter",
    "ToolFormatterRegistry",
    "ToolResponseDisplay",
    "create_default_tool_formatter_registry",
    "get_default_formatter_registry",
    "register_tool_formatter",
    "set_default_formatter_registry",
]
