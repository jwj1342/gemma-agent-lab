"""Runner and evaluation base classes for ADK benchmarks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from adk_eval_core.models.base import BaseTaskResult

logger = logging.getLogger(__name__)


class TaskResult(BaseTaskResult):
    """Outcome of evaluating a single benchmark task/instance."""


@dataclass
class EvaluationResult:
    """Aggregated result container for a complete benchmark evaluation run."""

    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.task_results)

    @property
    def resolved(self) -> int:
        return sum(1 for r in self.task_results if r.resolved)

    @property
    def resolution_rate(self) -> float:
        return (self.resolved / self.total) if self.total > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        """Compute overall evaluation summary statistics."""
        return {
            "total": self.total,
            "resolved": self.resolved,
            "rate": self.resolution_rate,
        }

    def group_by(
        self,
        key: str | Callable[[TaskResult], str] = "default",
        default: str = "default",
    ) -> dict[str, dict[str, Any]]:
        """Group task results and compute summary statistics per group.

        Args:
            key: Either a metadata key string (or attribute name on TaskResult)
                or an extractor callable taking a TaskResult and returning a group name.
            default: Fallback group name if the key is missing or empty.

        Returns:
            Dictionary mapping group names to statistics:
            {"total": int, "resolved": int, "rate": float}
        """
        stats: dict[str, dict[str, Any]] = {}
        for tr in self.task_results:
            if callable(key):
                val = key(tr)
                group = str(val) if (val is not None and val != "") else default
            elif isinstance(key, str):
                if key in tr.metadata:
                    val = tr.metadata[key]
                    group = str(val) if (val is not None and val != "") else default
                elif hasattr(tr, key) and not callable(getattr(tr, key)):
                    val = getattr(tr, key)
                    group = str(val) if (val is not None and val != "") else default
                else:
                    group = default
            else:
                group = default

            if group not in stats:
                stats[group] = {"total": 0, "resolved": 0, "rate": 0.0}
            stats[group]["total"] += 1
            if tr.resolved:
                stats[group]["resolved"] += 1

        for group, s in stats.items():
            s["rate"] = (s["resolved"] / s["total"]) if s["total"] > 0 else 0.0
        return stats


class BaseEvaluator(ABC):
    """Abstract base runner for benchmark evaluations."""

    @abstractmethod
    def evaluate_task(self, task: Any, task_index: int = 1, total_tasks: int = 1) -> TaskResult:
        """Run evaluation on a single benchmark task."""

    @abstractmethod
    def run(self) -> EvaluationResult:
        """Run complete benchmark evaluation across all loaded tasks."""
