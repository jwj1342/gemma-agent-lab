"""Base protocols and abstract types for benchmark runners."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from adk_eval_core.models.base import BaseSuiteResult, BaseTaskResult, BenchmarkTask

TTask = TypeVar("TTask", bound=BenchmarkTask)
TResult = TypeVar("TResult", bound=BaseTaskResult)


@runtime_checkable
class AsyncEvaluatorProtocol(Protocol[TTask, TResult]):
    """Standard async evaluator interface for all domain benchmarks.

    Defines the contract for executing benchmark evaluation tasks asynchronously.
    """

    async def evaluate_task(self, task: TTask) -> TResult:
        """Run evaluation for a single task instance asynchronously.

        Args:
            task: The benchmark task to evaluate.

        Returns:
            The evaluation result for the single task.
        """
        ...

    async def run(self, tasks: list[TTask]) -> BaseSuiteResult[TResult]:
        """Run evaluation across a collection of tasks.

        Args:
            tasks: List of benchmark tasks to evaluate.

        Returns:
            Aggregated evaluation suite result.
        """
        ...


__all__ = ["AsyncEvaluatorProtocol"]
