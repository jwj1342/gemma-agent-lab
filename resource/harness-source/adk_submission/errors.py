"""Custom exception hierarchy for the ``adk_submission`` package.

This module defines the complete exception hierarchy used throughout the
submission compilation, validation, and execution lifecycle. All custom
exceptions inherit from the base :class:`SubmissionError`.

Exception Hierarchy:
    * :class:`SubmissionError`: Root exception for all package errors.
        * :class:`SubmissionValidationError`: Structural or schema validation failure.
            * :class:`LimitExceededError`: Configured resource/execution limit exceeded.
            * :class:`PathTraversalError`: Attempted path traversal outside sandbox.
            * :class:`SubmissionSchemaError`: Pydantic YAML schema validation failure.
        * :class:`ToolNotFoundError`: Referenced tool missing from registry.
        * :class:`ModelNotFoundError`: Referenced model missing from registry.
        * :class:`AdapterNotFoundError`: Referenced adapter missing from discovered adapters.
        * :class:`CallbackNotFoundError`: Referenced callback missing or invalid type.
        * :class:`SkillNotFoundError`: Referenced skill missing from registry.
        * :class:`ServerStartupError`: Local model server startup or healthcheck failure.

Usage Example:
    When compiling or validating a submission, organizers can catch specific
    validation errors or handle all library errors generically::

        from adk_submission.errors import SubmissionValidationError, SubmissionError

        try:
            compile_submission("path/to/submission", limits=my_limits)
        except SubmissionValidationError as e:
            print(f"Validation failed: {e}")
        except SubmissionError as e:
            print(f"An unexpected ADK submission error occurred: {e}")
"""


class SubmissionError(Exception):
    """Base exception for all custom errors raised by the ``adk_submission`` package.

    This exception serves as the root of the package's exception hierarchy.
    Catching ``SubmissionError`` allows organizers and calling applications to trap
    all library-specific errors—including schema validation failures, limit
    violations, security constraints, and registry lookup failures—with a single
    exception handler.
    """


class SubmissionValidationError(SubmissionError, ValueError):
    """Exception raised when a submission fails structural, constraint, or schema validation.

    This class represents errors encountered during the validation phase of submission
    compilation. It is raised directly for general validation violations (such as
    disallowed generation parameters or invalid configuration combinations) and serves
    as the base class for more specialized validation exceptions:
    
    * :class:`LimitExceededError`: For violations of configured resource or execution limits.
    * :class:`PathTraversalError`: For security violations involving file path escaping.
    * :class:`SubmissionSchemaError`: For Pydantic schema parsing and validation failures.
    """


class LimitExceededError(SubmissionValidationError):
    """Exception raised when a submission exceeds a configured resource or execution limit.

    This error indicates a violation of constraints defined by the organizer in either
    :class:`~adk_submission.limits.SubmissionLimits` (e.g., total archive size, file count,
    agent count, instruction character length, or loop iteration caps) or
    :class:`~adk_submission.limits.GenerationConstraints` (e.g., parameter range bounds).

    The exception message provides specific details identifying the exceeded limit, the
    configured maximum, and the offending value or agent.
    """


class PathTraversalError(SubmissionValidationError):
    r"""Exception raised when a file path attempts to escape the sandboxed submission directory.

    This security exception is triggered during schema validation when a configuration path
    (such as a sub-agent reference, agent-tool reference, or skill directory path) contains
    forbidden traversal patterns. Specifically, it prevents sandboxed configurations from
    referencing absolute paths (starting with ``/`` or ``\``) or navigating upward in the
    directory hierarchy using parent references (``..``).
    """


class ToolNotFoundError(SubmissionError):
    """Exception raised when a referenced tool name cannot be resolved in the tool registry.

    This error occurs during submission compilation or agent loading when an agent's
    configuration specifies a tool name that has not been registered with the
    organizer's :class:`~adk_submission.registry.ToolRegistry`.

    Attributes:
        tool_name (str): The name of the tool that could not be found.
        available (list[str]): A list of tool names currently available in the registry.
    """

    def __init__(self, tool_name: str, available: list[str] | None = None):
        """Initialize the ToolNotFoundError with the missing tool and available options.

        Args:
            tool_name: The name of the tool that could not be found in the registry.
            available: An optional list of tool names currently registered. If ``None``
                or empty, the error message will indicate that no tools are available.
        """
        self.tool_name = tool_name
        self.available = available or []
        available_str = ", ".join(sorted(self.available)) if self.available else "none"
        super().__init__(
            f"Tool '{tool_name}' not found in registry. "
            f"Available tools: {available_str}"
        )


class ModelNotFoundError(SubmissionError):
    """Exception raised when a referenced model alias cannot be resolved in the model registry.

    This error occurs during submission compilation or agent loading when an agent's
    configuration specifies a model alias that has not been registered with the
    organizer's :class:`~adk_submission.registry.ModelRegistry`.

    Attributes:
        model_alias (str): The model alias that could not be found.
        available (list[str]): A list of model aliases currently available in the registry.
    """

    def __init__(self, model_alias: str, available: list[str] | None = None):
        """Initialize the ModelNotFoundError with the missing model and available options.

        Args:
            model_alias: The model alias that could not be found in the registry.
            available: An optional list of model aliases currently registered. If ``None``
                or empty, the error message will indicate that no models are available.
        """
        self.model_alias = model_alias
        self.available = available or []
        available_str = ", ".join(sorted(self.available)) if self.available else "none"
        super().__init__(
            f"Model alias '{model_alias}' not found in registry. "
            f"Available models: {available_str}"
        )


class AdapterNotFoundError(SubmissionError):
    """Exception raised when a referenced adapter name cannot be resolved in discovered adapters.

    This error occurs during submission compilation when an agent configuration specifies
    an ``adapter`` name that was not discovered in the submission's ``adapters/`` directory.

    Attributes:
        adapter_name (str): The adapter identifier name that could not be found.
        available (list[str]): A list of adapter names currently discovered in the submission.
    """

    def __init__(self, adapter_name: str, available: list[str] | None = None):
        """Initialize the AdapterNotFoundError with the missing adapter and available options.

        Args:
            adapter_name: The name of the adapter that could not be found.
            available: An optional list of discovered adapter names. If ``None``
                or empty, the error message will indicate that no adapters were discovered.
        """
        self.adapter_name = adapter_name
        self.available = available or []
        available_str = ", ".join(sorted(self.available)) if self.available else "none"
        super().__init__(
            f"Adapter '{adapter_name}' not found in discovered adapters. "
            f"Available adapters: {available_str}"
        )


class CallbackNotFoundError(SubmissionError):
    """Exception raised for missing callbacks or callback lifecycle position mismatches.

    This error occurs during agent loading and validation under two distinct conditions:
    1. **Missing Callback**: A callback name referenced in an agent's configuration is not
       present in the organizer's :class:`~adk_submission.registry.CallbackRegistry`.
    2. **Type/Position Mismatch**: A callback is registered, but its defined type is
       incompatible with the lifecycle hook where it is specified (e.g., attempting to use
       a tool callback in a model callback position).

    Attributes:
        callback_name (str): The name of the callback being resolved.
        callback_type (str | None): The lifecycle position or type requested (e.g.,
            ``'before_model'``), if applicable.
        available (list[str]): A list of callback names available in the registry.
        allowed_types (str | None): A string listing the allowed positions/types when the
            error is due to a position mismatch.
    """

    def __init__(
        self,
        callback_name: str,
        callback_type: str | None = None,
        available: list[str] | None = None,
        allowed_types: str | None = None,
    ):
        """Initialize the CallbackNotFoundError for missing callbacks or type mismatches.

        The exception message is dynamically generated based on the provided arguments:
        * If ``allowed_types`` is provided, the message highlights a position mismatch
          indicating that the callback is not allowed for the specified ``callback_type``.
        * Otherwise, the message indicates that the callback was not found in the registry
          and lists the ``available`` callbacks.

        Args:
            callback_name: The name of the callback that caused the error.
            callback_type: The lifecycle position or type context (e.g., ``'before_model'``)
                in which the callback was referenced.
            available: An optional list of callback names currently registered.
            allowed_types: A string describing the allowed positions/types for the callback.
                Providing this parameter switches the error mode to a position mismatch.
        """
        self.callback_name = callback_name
        self.callback_type = callback_type
        self.available = available or []
        self.allowed_types = allowed_types
        if allowed_types:
            super().__init__(
                f"Callback '{callback_name}' is not allowed for position '{callback_type}'. "
                f"Allowed positions: {allowed_types}"
            )
        else:
            available_str = ", ".join(sorted(self.available)) if self.available else "none"
            type_str = f" (type: {callback_type})" if callback_type else ""
            super().__init__(
                f"Callback '{callback_name}'{type_str} not found in registry. "
                f"Available callbacks: {available_str}"
            )


class SkillNotFoundError(SubmissionError):
    """Exception raised when a referenced skill name cannot be resolved in the skill registry.

    This error occurs during submission compilation or agent loading when an agent's
    configuration specifies a skill name that has not been registered with the
    organizer's :class:`~adk_submission.registry.SkillRegistry`.

    Attributes:
        skill_name (str): The name of the skill that could not be found.
        available (list[str]): A list of skill names currently available in the registry.
    """

    def __init__(self, skill_name: str, available: list[str] | None = None):
        """Initialize the SkillNotFoundError with the missing skill and available options.

        Args:
            skill_name: The name of the skill that could not be found in the registry.
            available: An optional list of skill names currently registered. If ``None``
                or empty, the error message will indicate that no skills are available.
        """
        self.skill_name = skill_name
        self.available = available or []
        available_str = ", ".join(sorted(self.available)) if self.available else "none"
        super().__init__(
            f"Skill '{skill_name}' not found in registry. "
            f"Available skills: {available_str}"
        )


class SubmissionSchemaError(SubmissionValidationError):
    """Exception raised when a submission configuration fails Pydantic schema validation.

    This exception is triggered during the YAML parsing and structural validation phase
    when an agent's configuration file violates the expected Pydantic schema definitions
    (e.g., :class:`~adk_submission.schema.SandboxedAgentConfig`).

    It encapsulates underlying Pydantic validation errors—such as missing required fields,
    invalid field types, or unrecognized agent classes—providing a clear, submission-specific
    error boundary for organizers.
    """


class ServerStartupError(SubmissionError):
    """Exception raised when a local model server fails to start or healthcheck times out.

    Attributes:
        message (str): Explanation of the failure condition.
        output (str | None): Last captured standard output/error from the server process.
    """

    def __init__(self, message: str, output: str | None = None):
        """Initialize the ServerStartupError with an explanatory message and optional server output.

        Args:
            message: Explanation of the failure condition.
            output: Optional captured process stdout/stderr from the terminated server.
        """
        self.message = message
        self.output = output
        full_msg = f"{message}\nServer output:\n{output}" if output else message
        super().__init__(full_msg)
