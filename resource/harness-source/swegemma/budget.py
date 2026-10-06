"""Token counter, USD spending calculator, evaluation budgets, and harness operational limits."""

from __future__ import annotations

from dataclasses import dataclass

from adk_eval_core.budget import (
    ModelPricing,
    PricingTable,
    TokenBudget,
)


@dataclass(frozen=True)
class EvaluationBudget:
    """Consumable allowances that terminate or conclude a task session when exhausted."""

    time_minutes: float | None = 60.0  # Max session wall-clock time in minutes
    tool_calls: int | None = None  # Max total tool invocations
    turns: int | None = None  # Max LLM reasoning / loop turns
    cost_usd: float | None = None  # Max monetary spend in USD
    total_tokens: int | None = None  # Max total tokens consumed


@dataclass(frozen=True)
class HarnessLimits:
    """Operational constraints that govern individual tool and sandbox execution."""

    command_timeout_seconds: int | None = 300  # Timeout for a single shell command
    max_stdout_chars: int | None = 5000  # Max characters returned per command
    max_file_lines: int | None = 150  # Max lines returned per read_file
    max_file_chars: int | None = 10000  # Max characters returned per read_file


__all__ = [
    'EvaluationBudget',
    'HarnessLimits',
    'ModelPricing',
    'PricingTable',
    'TokenBudget',
]
