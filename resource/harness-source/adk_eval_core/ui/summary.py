"""Evaluation summary formatting functions for rich console display."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from adk_eval_core.ui.presenter import DefaultSummaryPresenter


def print_evaluation_summary(
    task_results: list[Any] | Any,
    console: Console | None = None,
) -> None:
    """Print a clean rich formatted summary table for SWE/Benchmark runs."""
    presenter = DefaultSummaryPresenter()
    presenter.present(task_results, console=console or Console(), output_format="rich")


__all__ = ["print_evaluation_summary"]
