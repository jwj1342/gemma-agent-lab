"""UI, display, formatting, and presentation module for adk-eval-core."""

from adk_eval_core.ui.dashboard import EvaluationDashboard, SlotState
from adk_eval_core.ui.event_display import EventDisplay
from adk_eval_core.ui.formatters import (
    CommandToolFormatter,
    DefaultToolFormatter,
    FileToolFormatter,
    GenericToolFormatter,
    PythonToolFormatter,
    ToolCallDisplay,
    ToolFormatter,
    ToolFormatterRegistry,
    ToolResponseDisplay,
    create_default_tool_formatter_registry,
    get_default_formatter_registry,
    register_tool_formatter,
    set_default_formatter_registry,
)
from adk_eval_core.ui.hud import StatusPanel, gauge_color
from adk_eval_core.ui.presenter import (
    DefaultSummaryPresenter,
    ResultPresenter,
)
from adk_eval_core.ui.protocol import (
    EvaluationContextProtocol,
    TokenBudgetProtocol,
)
from adk_eval_core.ui.summary import print_evaluation_summary

__all__ = [
    "CommandToolFormatter",
    "DefaultSummaryPresenter",
    "DefaultToolFormatter",
    "EvaluationContextProtocol",
    "EvaluationDashboard",
    "EventDisplay",
    "FileToolFormatter",
    "GenericToolFormatter",
    "PythonToolFormatter",
    "ResultPresenter",
    "SlotState",
    "StatusPanel",
    "TokenBudgetProtocol",
    "ToolCallDisplay",
    "ToolFormatter",
    "ToolFormatterRegistry",
    "ToolResponseDisplay",
    "create_default_tool_formatter_registry",
    "gauge_color",
    "get_default_formatter_registry",
    "print_evaluation_summary",
    "register_tool_formatter",
    "set_default_formatter_registry",
]
