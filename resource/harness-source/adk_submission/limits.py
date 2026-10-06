"""Configurable limits and generation constraints for submissions.

This module defines the data structures and validation logic used to enforce
organizer-defined constraints on autonomous agent submissions. It separates
constraints into two primary categories:

1. Structural limits (:class:`SubmissionLimits`): Caps on archive size, file counts,
   agent nesting depth, loop iterations, and allowed file extensions.
2. Generation constraints (:class:`GenerationConstraints`): Restrictions on LLM
   generation parameters (e.g., temperature, token limits, stop sequences)
   configured via :class:`NumericRange`.

Violations of these limits during submission compilation or configuration validation
raise :class:`~adk_submission.errors.LimitExceededError` or
:class:`~adk_submission.errors.SubmissionValidationError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .errors import SubmissionValidationError

# ---------------------------------------------------------------------------
# Numeric range helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericRange:
    """Inclusive range for a numeric parameter.

    Used primarily to define valid bounding intervals for generation parameters
    (such as temperature, top_p, or token limits) within :class:`GenerationConstraints`.

    Attributes:
        min: The minimum allowed value (inclusive). Can be a float or an integer.
        max: The maximum allowed value (inclusive). Can be a float or an integer.

    Raises:
        ValueError: If ``min`` is strictly greater than ``max`` during initialization,
            or if ``min`` or ``max`` is a boolean, NaN, or infinite.
    """

    min: float | int
    max: float | int

    def __post_init__(self) -> None:
        """Validate that the minimum value is less than or equal to the maximum value.

        Raises:
            ValueError: If ``min`` or ``max`` is not a finite number, or if ``min`` is greater than ``max``.
        """
        for bound_name, bound_val in (("min", self.min), ("max", self.max)):
            if (
                isinstance(bound_val, bool)
                or not isinstance(bound_val, (int, float))
                or math.isnan(bound_val)
                or math.isinf(bound_val)
            ):
                raise ValueError(
                    f"NumericRange {bound_name} must be a finite number, got {bound_val!r}"
                )
        if self.min > self.max:
            raise ValueError(
                f"NumericRange min ({self.min}) must be <= max ({self.max})"
            )

    def check(self, value: float, field_name: str, agent_name: str) -> None:
        """Raise SubmissionValidationError if value is outside the inclusive range.

        Args:
            value: The numeric value to validate against the range.
            field_name: The name of the parameter being checked (used in error formatting).
            agent_name: The name of the agent associated with the configuration.

        Raises:
            SubmissionValidationError: If ``value`` is a boolean, NaN, infinite,
                less than ``min``, or greater than ``max``.
        """
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or math.isnan(value)
            or math.isinf(value)
        ):
            raise SubmissionValidationError(
                f"Agent '{agent_name}': {field_name}={value!r} is not a valid finite number"
            )
        if value < self.min or value > self.max:
            raise SubmissionValidationError(
                f"Agent '{agent_name}': {field_name}={value} is outside "
                f"allowed range [{self.min}, {self.max}]"
            )


# ---------------------------------------------------------------------------
# Submission limits
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".rst",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".ggml",
        ".model",
        ".tiktoken",
        ".jinja",
        ".json",
        ".jsonl",
        ".toml",
        ".cfg",
        ".ini",
        ".py",
        ".sh",
        ".sql",
        ".csv",
        ".tsv",
        ".patch",
        ".diff",
        ".html",
        ".xml",
    }
)

_DEFAULT_ADAPTER_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".ggml",
    }
)


@dataclass
class SubmissionLimits:
    """Configurable structural limits for submission validation.

    Defines the resource caps, file constraints, and structural boundaries enforced
    on an agent submission archive during compilation and validation.

    Attributes:
        max_total_size_bytes: Maximum allowable uncompressed size of the entire
            submission directory in bytes (including ``adapters/``). Defaults to
            3,221,225,472 (3 GiB).
        max_yaml_size_bytes: Maximum allowable byte size for any individual YAML file
            or cumulative !include expansion. Defaults to 52,428,800 (50 MiB).
        max_skill_size_bytes: Maximum allowable byte size for any individual skill
            directory. Defaults to 52,428,800 (50 MiB).
        max_file_count: Maximum total number of files permitted within the submission
            directory. Defaults to 10,000.
        max_yaml_files: Maximum number of YAML configuration files allowed. Defaults to 1,000.
        max_instruction_chars: Maximum number of characters allowed for an individual
            agent's instruction text. Defaults to 1,000,000.
        max_total_instruction_chars: Maximum cumulative characters allowed across all
            agent instructions in the submission. Defaults to 10,000,000.
        max_agents: Maximum total number of agent definitions permitted in the submission.
            Defaults to 500.
        max_sub_agent_depth: Maximum allowable nesting depth for sub-agent hierarchies.
            Defaults to 50.
        max_loop_iterations: Maximum allowable execution iterations configured for any
            LoopAgent. Defaults to 500.
        max_skills: Maximum number of distinct skills allowable per submission. Defaults to 1,000.
        allowed_file_extensions: Set of file extensions permitted within the submission
            archive. Defaults to a comprehensive set of config, prompt, code, and adapter extensions.
        adapter_extensions: Set of file extensions recognized as model adapter weights
            (e.g., LoRA weights). Defaults to ``{'.safetensors', '.bin', '.pt', '.pth', '.gguf', '.ggml'}``.
    """

    max_total_size_bytes: int = 3 * 1024 * 1024 * 1024  # 3 GiB (including adapters)
    max_yaml_size_bytes: int = 50 * 1024 * 1024  # 50 MiB per YAML / cumulative !include
    max_skill_size_bytes: int = 50 * 1024 * 1024  # 50 MiB per skill directory
    max_file_count: int = 10_000
    max_yaml_files: int = 1_000
    max_instruction_chars: int = 1_000_000  # Per agent
    max_total_instruction_chars: int = 10_000_000  # Across all agents
    max_agents: int = 500  # Total agent count
    max_sub_agent_depth: int = 50  # Nesting depth
    max_loop_iterations: int = 500  # LoopAgent cap
    max_skills: int = 1_000  # Max skills per submission
    allowed_file_extensions: frozenset[str] = _DEFAULT_ALLOWED_EXTENSIONS
    adapter_extensions: frozenset[str] = _DEFAULT_ADAPTER_EXTENSIONS

    def __post_init__(self) -> None:
        for attr_name in (
            "max_total_size_bytes",
            "max_yaml_size_bytes",
            "max_skill_size_bytes",
            "max_file_count",
            "max_yaml_files",
            "max_instruction_chars",
            "max_total_instruction_chars",
            "max_agents",
            "max_loop_iterations",
        ):
            val = getattr(self, attr_name)
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(
                    f"SubmissionLimits.{attr_name} must be a positive integer (> 0), got {val!r}"
                )
        for non_neg_name in ("max_sub_agent_depth", "max_skills"):
            val = getattr(self, non_neg_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(
                    f"SubmissionLimits.{non_neg_name} must be a non-negative integer (>= 0), got {val!r}"
                )


# ---------------------------------------------------------------------------
# Generation constraints
# ---------------------------------------------------------------------------

GENERATION_PARAM_FIELDS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
        "response_mime_type",
        "seed",
        "thinking_config",
    }
)
"""The complete universe of generation parameter fields supported by the submission schema.

These fields correspond to the configurable attributes on
:class:`~adk_submission.schema.GenerateContentConfig`.
"""


# Generation parameter fields that have NumericRange constraints.
# Used by GenerationConstraints.validate_config() to check bounds.
_RANGE_CONSTRAINED_FIELDS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "presence_penalty",
        "frequency_penalty",
    }
)


@dataclass
class GenerationConstraints:
    """Organizer-defined constraints on LLM generation parameters.

    Controls which generation parameters submissions are permitted to configure and
    enforces valid bounding ranges. If a ``GenerationConstraints`` instance is not
    provided during submission compilation, no generation parameters are restricted
    (all schema-level parameters are allowed).

    Attributes:
        allowed_fields: Set of parameter names submissions are permitted to configure.
            If ``None``, all schema-supported fields are allowed. If an empty ``frozenset``,
            submissions are prohibited from specifying any generation configuration.
        temperature: Bounding range constraint for the temperature parameter. If ``None``,
            no range restriction is enforced.
        top_p: Bounding range constraint for the top_p parameter. If ``None``, no range
            restriction is enforced.
        top_k: Bounding range constraint for the top_k parameter. If ``None``, no range
            restriction is enforced.
        max_output_tokens: Bounding range constraint for the max_output_tokens parameter.
            If ``None``, no range restriction is enforced.
        presence_penalty: Bounding range constraint for the presence_penalty parameter.
            If ``None``, no range restriction is enforced.
        frequency_penalty: Bounding range constraint for the frequency_penalty parameter.
            If ``None``, no range restriction is enforced.
        thinking_budget: Bounding range constraint for the model's thinking budget
            (within thinking_config). If ``None``, no range restriction is enforced.
        max_stop_sequences: Maximum allowable number of custom stop sequences. If ``None``,
            no cap on stop sequences is enforced.
        allowed_response_mime_types: Set of permitted MIME types for model responses
            (e.g., ``{'text/plain', 'application/json'}``). If ``None``, all MIME types are allowed.
        defaults: Dictionary of default generation parameter values applied when a submission
            does not explicitly specify a parameter. Submission-specified values take precedence.
    """

    allowed_fields: frozenset[str] | None = None
    temperature: NumericRange | None = None
    top_p: NumericRange | None = None
    top_k: NumericRange | None = None
    max_output_tokens: NumericRange | None = None
    presence_penalty: NumericRange | None = None
    frequency_penalty: NumericRange | None = None
    thinking_budget: NumericRange | None = None
    max_stop_sequences: int | None = None
    allowed_response_mime_types: frozenset[str] | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    def validate_config(
        self,
        config_dict: dict[str, Any],
        agent_name: str,
    ) -> dict[str, Any]:
        """Validate and merge a submission's generate_content_config against constraints.

        Performs multi-step validation on the submission's generation parameters:
        1. Verifies that all specified fields are present in ``allowed_fields``.
        2. Checks numeric parameters against configured ``NumericRange`` bounds.
        3. Validates that the number of ``stop_sequences`` does not exceed ``max_stop_sequences``.
        4. Verifies that ``response_mime_type`` is listed in ``allowed_response_mime_types``.
        5. Validates ``thinking_budget`` within ``thinking_config``. If ``thinking_config`` is
           provided without an explicit budget, it is automatically assigned the maximum
           allowed budget (``thinking_budget.max``).
        6. Merges the validated configuration with ``defaults``, where submission-specified
           values take precedence.

        Args:
            config_dict: The submission's generation configuration represented as a dictionary
                containing only explicitly set (non-modified) fields.
            agent_name: The name of the agent being validated, used for contextual error reporting.

        Returns:
            A new dictionary containing the validated generation configuration enriched with
            constraint defaults and fallback thinking budget values.

        Raises:
            SubmissionValidationError: If any parameter is disallowed, falls outside configured
                numeric ranges, exceeds sequence limits, or specifies an invalid MIME type.
        """
        # 1. Check each set field against allowed_fields
        if self.allowed_fields is not None:
            for key, val in config_dict.items():
                if val is not None and key not in self.allowed_fields:
                    allowed_str = (
                        ", ".join(sorted(self.allowed_fields))
                        if self.allowed_fields
                        else "none"
                    )
                    raise SubmissionValidationError(
                        f"Agent '{agent_name}': generation parameter "
                        f"'{key}' is not allowed. "
                        f"Allowed parameters: {allowed_str}"
                    )

        # 2. Check numeric range constraints and finite non-bool types
        for field_name in _RANGE_CONSTRAINED_FIELDS | {"seed"}:
            if field_name in config_dict and config_dict[field_name] is not None:
                val = config_dict[field_name]
                if (
                    isinstance(val, bool)
                    or not isinstance(val, (int, float))
                    or math.isnan(val)
                    or math.isinf(val)
                ):
                    raise SubmissionValidationError(
                        f"Agent '{agent_name}': {field_name}={val!r} is not a valid finite number"
                    )
                if field_name in _RANGE_CONSTRAINED_FIELDS:
                    range_constraint: NumericRange | None = getattr(self, field_name)
                    if range_constraint is not None:
                        range_constraint.check(val, field_name, agent_name)

        # 3. Check stop_sequences length
        if (
            self.max_stop_sequences is not None
            and "stop_sequences" in config_dict
            and config_dict["stop_sequences"] is not None
        ):
            n = len(config_dict["stop_sequences"])
            if n > self.max_stop_sequences:
                raise SubmissionValidationError(
                    f"Agent '{agent_name}': {n} stop sequences exceeds "
                    f"maximum of {self.max_stop_sequences}"
                )

        # 4. Check response_mime_type
        if (
            self.allowed_response_mime_types is not None
            and "response_mime_type" in config_dict
            and config_dict["response_mime_type"] is not None
        ):
            mime = config_dict["response_mime_type"]
            if mime not in self.allowed_response_mime_types:
                allowed_str = ", ".join(sorted(self.allowed_response_mime_types))
                raise SubmissionValidationError(
                    f"Agent '{agent_name}': response_mime_type '{mime}' "
                    f"is not allowed. Allowed types: {allowed_str}"
                )

        # 5. Merge defaults (submission values take priority, deep-merging thinking_config)
        merged = dict(self.defaults)
        default_tc = merged.get("thinking_config")
        merged.update(config_dict)
        if isinstance(default_tc, dict) and isinstance(config_dict.get("thinking_config"), dict):
            combined_tc = dict(default_tc)
            combined_tc.update(config_dict["thinking_config"])
            merged["thinking_config"] = combined_tc

        # 6. Check and enrich thinking_budget on merged config
        if "thinking_config" in merged and merged["thinking_config"] is not None:
            tc = merged["thinking_config"]
            budget = (
                tc.get("thinking_budget")
                if isinstance(tc, dict)
                else getattr(tc, "thinking_budget", None)
            )
            if budget is not None:
                if (
                    isinstance(budget, bool)
                    or not isinstance(budget, (int, float))
                    or math.isnan(budget)
                    or math.isinf(budget)
                ):
                    raise SubmissionValidationError(
                        f"Agent '{agent_name}': thinking_budget={budget!r} is not a valid finite number"
                    )
                if self.thinking_budget is not None:
                    self.thinking_budget.check(budget, "thinking_budget", agent_name)
            elif self.thinking_budget is not None:
                default_budget = self.thinking_budget.max
                if isinstance(tc, dict):
                    tc_copy = dict(tc)
                    tc_copy["thinking_budget"] = default_budget
                    merged["thinking_config"] = tc_copy
                else:
                    tc_copy = dict(tc.model_dump() if hasattr(tc, "model_dump") else tc.__dict__)
                    tc_copy["thinking_budget"] = default_budget
                    merged["thinking_config"] = tc_copy

        return merged
