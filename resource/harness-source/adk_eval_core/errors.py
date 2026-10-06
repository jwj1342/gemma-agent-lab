"""Custom exception hierarchy for the adk-eval-core package."""

from __future__ import annotations


class EvalCoreError(Exception):
    """Root base exception for all errors originating from adk-eval-core."""


# --- Sandbox Errors ---


class SandboxError(EvalCoreError):
    """Base exception for sandbox and container execution failures."""


class SandboxNotStartedError(SandboxError, RuntimeError):
    """Raised when an operation is attempted on an uninitialized/stopped sandbox."""

    def __init__(self, message: str = "Sandbox is not running", sandbox_id: str | None = None) -> None:
        super().__init__(message)
        self.sandbox_id = sandbox_id


class SandboxTimeoutError(SandboxError, TimeoutError):
    """Raised when a container startup or command execution times out."""

    def __init__(
        self,
        message: str,
        timeout: float | None = None,
        command: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.timeout = timeout
        self.command = command
        self.stdout = stdout
        self.stderr = stderr


class SandboxExecutionError(SandboxError, RuntimeError):
    """Raised when a command or archive extraction inside a sandbox fails."""

    def __init__(
        self,
        message: str,
        exit_code: int | None = None,
        command: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.command = command
        self.stderr = stderr


class SandboxFileNotFoundError(SandboxError, FileNotFoundError):
    """Raised when a file or directory is not found on host or inside sandbox."""

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.filename = path

    def __str__(self) -> str:
        return self.args[0] if self.args else super().__str__()


class SandboxIsADirectoryError(SandboxError, IsADirectoryError):
    """Raised when an operation expected a file but encountered a directory."""

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.filename = path

    def __str__(self) -> str:
        return self.args[0] if self.args else super().__str__()


class SandboxPathTraversalError(SandboxError, ValueError):
    """Raised when a file path escapes the sandbox or a tar slip is detected."""

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


# --- Budget Errors ---


class BudgetError(EvalCoreError):
    """Base exception for token accounting and budget enforcement."""


class BudgetExceededError(BudgetError):
    """Raised when an agent session exceeds configured budget or call limits."""

    def __init__(
        self,
        message: str = "Budget limit exceeded",
        total_cost_usd: float | None = None,
        max_budget_usd: float | None = None,
        llm_calls: int | None = None,
        max_llm_calls: int | None = None,
    ) -> None:
        super().__init__(message)
        self.total_cost_usd = total_cost_usd
        self.max_budget_usd = max_budget_usd
        self.llm_calls = llm_calls
        self.max_llm_calls = max_llm_calls


class InvalidBudgetError(BudgetError, ValueError):
    """Raised when token counts or pricing configurations are negative or invalid."""

    def __init__(
        self,
        message: str = "Invalid budget configuration or token count",
        model_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model_id = model_id


# --- Editing Errors ---


class EditError(EvalCoreError):
    """Base exception for code editing and replacement errors."""


class EditMatchError(EditError, ValueError):
    """Raised when target content cannot be matched in a file."""

    def __init__(self, message: str, filepath: str | None = None, old_string: str | None = None) -> None:
        super().__init__(message)
        self.filepath = filepath
        self.old_string = old_string


class AmbiguousMatchError(EditMatchError):
    """Raised when target content flexibly matches multiple locations."""

    def __init__(
        self,
        message: str,
        matches_count: int | None = None,
        filepath: str | None = None,
        old_string: str | None = None,
    ) -> None:
        super().__init__(message, filepath=filepath, old_string=old_string)
        self.matches_count = matches_count


# --- Scoring Errors ---


class ScoringError(EvalCoreError):
    """Base exception for metric resolution and scoring errors."""


class MetricResolutionError(ScoringError, ImportError):
    """Raised when a required scoring dependency (e.g. scikit-learn) is missing."""

    def __init__(self, message: str, metric_name: str | None = None) -> None:
        super().__init__(message)
        self.metric_name = metric_name
        self.name = metric_name


class InvalidScorerError(ScoringError, TypeError):
    """Raised when an invalid scorer object or function type is provided."""


__all__ = [
    "AmbiguousMatchError",
    "BudgetError",
    "BudgetExceededError",
    "EditError",
    "EditMatchError",
    "EvalCoreError",
    "InvalidBudgetError",
    "InvalidScorerError",
    "MetricResolutionError",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxFileNotFoundError",
    "SandboxIsADirectoryError",
    "SandboxNotStartedError",
    "SandboxPathTraversalError",
    "SandboxTimeoutError",
    "ScoringError",
]
