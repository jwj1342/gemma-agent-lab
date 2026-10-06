"""Base data models for benchmark tasks, results, and execution suites."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class BenchmarkTask(BaseModel):
    """Base model for any evaluation task or benchmark problem instance."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="allow")

    task_id: str = Field(
        ...,
        validation_alias=AliasChoices("task_id", "instance_id", "problem_id"),
        description="Unique task or problem identifier.",
    )
    dataset_name: str = Field(
        default="benchmark",
        description="Dataset or benchmark suite name (e.g. swe-bench-lite, kaggle).",
    )
    prompt: str = Field(
        default="",
        description="Initial instruction or prompt presented to the agent.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary benchmark metadata dictionary.",
    )

    def __init__(
        self,
        task_id: str | None = None,
        instance_id: str | None = None,
        problem_id: str | None = None,
        **data: Any,
    ) -> None:
        tid = (
            task_id
            or instance_id
            or problem_id
            or data.pop("task_id", None)
            or data.pop("instance_id", None)
            or data.pop("problem_id", None)
        )
        if tid is not None:
            data["task_id"] = tid
        super().__init__(**data)

    @property
    def instance_id(self) -> str:
        """Backwards-compatibility alias for SWE-bench task identifier."""
        return self.task_id

    @property
    def problem_id(self) -> str:
        """Backwards-compatibility alias for Kaggle problem identifier."""
        return self.task_id


class BaseTaskResult(BaseModel):
    """Standard base model for single-task evaluation outcomes across all benchmarks."""

    model_config = ConfigDict(populate_by_name=True, extra="allow", arbitrary_types_allowed=True)

    task_id: str = Field(
        ...,
        validation_alias=AliasChoices("task_id", "instance_id", "problem_id"),
        description="Unique task identifier.",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the task was successfully resolved according to benchmark criteria.",
    )
    status: str = Field(
        default="SUCCESS",
        validation_alias=AliasChoices("status", "end_status"),
        description="Granular status string (e.g. SUCCESS, FAILED, TIMEOUT, ERROR, BUDGET_EXCEEDED).",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("duration_seconds", "wall_time_seconds"),
        description="Total task execution wall-clock time in seconds.",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("cost_usd", "total_cost_usd"),
        description="Total accumulated dollar cost of LLM tokens consumed.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total input and output tokens consumed.",
    )
    total_llm_calls: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("total_llm_calls", "llm_calls"),
        description="Total number of LLM inference requests made.",
    )
    error_message: str | None = Field(
        default=None,
        validation_alias=AliasChoices("error_message", "error"),
        description="Failure details or traceback if execution faulted.",
    )
    trace_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("trace_path", "trace_file", "trace_json_path"),
        description="Filesystem path to saved ATIF session trace.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific extensible metadata.",
    )

    def __init__(
        self,
        task_id: str | None = None,
        instance_id: str | None = None,
        problem_id: str | None = None,
        **data: Any,
    ) -> None:
        tid = (
            task_id
            or instance_id
            or problem_id
            or data.pop("task_id", None)
            or data.pop("instance_id", None)
            or data.pop("problem_id", None)
        )
        if tid is not None:
            data["task_id"] = tid
        super().__init__(**data)

    @field_validator("duration_seconds", "cost_usd", mode="before")
    @classmethod
    def _coerce_none_to_zero_float(cls, v: Any) -> Any:
        if v is None:
            return 0.0
        return v

    @field_validator("total_tokens", "total_llm_calls", mode="before")
    @classmethod
    def _coerce_none_to_zero_int(cls, v: Any) -> Any:
        if v is None:
            return 0
        return v

    @property
    def instance_id(self) -> str:
        """Backwards-compatibility alias for SWE-bench task identifier."""
        return self.task_id

    @instance_id.setter
    def instance_id(self, val: str) -> None:
        self.task_id = val

    @property
    def problem_id(self) -> str:
        """Backwards-compatibility alias for Kaggle problem identifier."""
        return self.task_id

    @problem_id.setter
    def problem_id(self, val: str) -> None:
        self.task_id = val

    @property
    def error(self) -> str | None:
        """Backwards-compatibility alias for error details."""
        return self.error_message

    @error.setter
    def error(self, val: str | None) -> None:
        self.error_message = val

    @property
    def total_cost_usd(self) -> float:
        """Backwards-compatibility alias for financial cost."""
        return self.cost_usd

    @total_cost_usd.setter
    def total_cost_usd(self, val: float) -> None:
        self.cost_usd = val

    @property
    def wall_time_seconds(self) -> float:
        """Backwards-compatibility alias for duration."""
        return self.duration_seconds

    @wall_time_seconds.setter
    def wall_time_seconds(self, val: float) -> None:
        self.duration_seconds = val

    @property
    def llm_calls(self) -> int:
        """Backwards-compatibility alias for LLM calls count."""
        return self.total_llm_calls

    @llm_calls.setter
    def llm_calls(self, val: int) -> None:
        self.total_llm_calls = val

    @property
    def trace_file(self) -> str | None:
        """Backwards-compatibility alias for session trace path."""
        return self.trace_path

    @trace_file.setter
    def trace_file(self, val: str | None) -> None:
        self.trace_path = val

    @property
    def trace_json_path(self) -> str | None:
        """Backwards-compatibility alias for JSON trace path in swegemma."""
        return self.trace_path

    @trace_json_path.setter
    def trace_json_path(self, val: str | None) -> None:
        self.trace_path = val

    @property
    def end_status(self) -> str:
        """Backwards-compatibility alias for status."""
        return self.status

    @end_status.setter
    def end_status(self, val: str) -> None:
        self.status = val


TTask = TypeVar("TTask", bound=BenchmarkTask)
TResult = TypeVar("TResult", bound=BaseTaskResult)


class BaseSuiteResult(BaseModel, Generic[TResult]):
    """Aggregated evaluation outcomes across an entire benchmark suite."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="allow")

    benchmark_name: str = Field(
        default="benchmark",
        description="Name of the benchmark suite.",
    )
    total_tasks: int = Field(
        default=0,
        ge=0,
        description="Total number of tasks evaluated.",
    )
    resolved_tasks: int = Field(
        default=0,
        ge=0,
        description="Total number of successfully resolved tasks.",
    )
    accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of resolved tasks (resolved / total).",
    )
    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total accumulated cost in USD.",
    )
    total_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock duration in seconds.",
    )
    results: list[TResult] = Field(
        default_factory=list,
        description="List of individual task result models.",
    )

    @classmethod
    def from_results(
        cls,
        results: list[TResult],
        benchmark_name: str = "benchmark",
    ) -> BaseSuiteResult[TResult]:
        """Factory method to construct BaseSuiteResult from a list of TaskResult models."""
        total = len(results)
        resolved = sum(1 for r in results if r.resolved)
        accuracy = (resolved / total) if total > 0 else 0.0
        total_cost = sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in results)
        total_duration = sum(getattr(r, "duration_seconds", 0.0) or 0.0 for r in results)
        return cls(
            benchmark_name=benchmark_name,
            total_tasks=total,
            resolved_tasks=resolved,
            accuracy=accuracy,
            total_cost_usd=total_cost,
            total_duration_seconds=total_duration,
            results=results,
        )

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary of suite evaluation metrics."""
        return {
            "benchmark_name": self.benchmark_name,
            "total": self.total_tasks,
            "resolved": self.resolved_tasks,
            "resolution_rate": self.accuracy,
            "total_cost_usd": self.total_cost_usd,
            "duration_seconds": self.total_duration_seconds,
        }
