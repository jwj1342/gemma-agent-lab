"""Protocols for evaluation contexts, token budgets, formatters, and presenters in UI rendering."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rich.console import Console


@runtime_checkable
class TokenBudgetProtocol(Protocol):
    """Protocol for token budget counters and cost tracking."""

    @property
    def total_input_tokens(self) -> int:
        """Total input / prompt tokens consumed."""
        ...

    @property
    def total_output_tokens(self) -> int:
        """Total output / completion tokens generated."""
        ...

    @property
    def total_cached_input_tokens(self) -> int:
        """Cached prompt tokens consumed, if supported."""
        ...

    @property
    def last_input_tokens(self) -> int:
        """Input tokens consumed in the most recent LLM invocation."""
        ...

    @property
    def llm_calls(self) -> int:
        """Total number of LLM invocations."""
        ...

    @property
    def total_cost_usd(self) -> float:
        """Total accumulated cost in USD."""
        ...

    @property
    def max_budget_usd(self) -> float | None:
        """Cost ceiling in USD, or None if unconstrained."""
        ...


@runtime_checkable
class EvaluationContextProtocol(Protocol):
    """Runtime-checkable protocol defining the interface required by UI status HUDs.

    Allows benchmark and evaluation contexts (e.g. SwegemmaContext, KaggleKaggleContext)
    to drive the terminal StatusPanel and EventDisplay without tight coupling.
    """

    @property
    def task_id(self) -> str:
        """Identifier of the task / problem / instance being evaluated."""
        ...

    @property
    def candidate_id(self) -> str | None:
        """Identifier of the candidate model, agent, or submission under test."""
        ...

    @property
    def status(self) -> str:
        """Current lifecycle status (e.g. 'running', 'completed', 'budget_exceeded', 'error')."""
        ...

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed execution time in seconds."""
        ...

    @property
    def max_time_minutes(self) -> float | None:
        """Maximum allowed duration in minutes, or None if unlimited."""
        ...

    @property
    def tool_calls(self) -> int:
        """Total number of tool calls executed so far."""
        ...

    @property
    def max_tool_calls(self) -> int | None:
        """Maximum allowed tool calls, or None if unlimited."""
        ...

    @property
    def token_budget(self) -> TokenBudgetProtocol | Any | None:
        """Token budget tracker or object providing token usage and cost metrics."""
        ...

    @property
    def metrics(self) -> dict[str, Any]:
        """Dictionary of domain-specific metrics collected during execution."""
        ...

    def to_status_dict(self) -> dict[str, Any]:
        """Return a dictionary of HUD status indicators for UI rendering."""
        ...


from adk_eval_core.ui.formatters.models import ToolFormatter


@runtime_checkable
class ResultPresenter(Protocol):
    """Protocol for formatting and presenting benchmark evaluation results across diverse output formats."""

    def present(
        self,
        result: Any,
        console: Console | None = None,
        *,
        output_format: str = "rich",
        **kwargs: Any,
    ) -> str | Any:
        """Render evaluation results to console or string.

        Args:
            result: EvaluationResult, TaskResult, list of TaskResults, or custom benchmark result.
            console: Optional Rich Console instance for direct printing.
            output_format: Desired output format: 'rich', 'text', 'plain', 'markdown', or 'json'.
            **kwargs: Additional presentation options (e.g. title, group_by, show_metadata).

        Returns:
            Rendered string (for text/markdown/json) or Rich renderable (for rich).
        """
        ...


__all__ = [
    "EvaluationContextProtocol",
    "ResultPresenter",
    "TokenBudgetProtocol",
    "ToolFormatter",
]
