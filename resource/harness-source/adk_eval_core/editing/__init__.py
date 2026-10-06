"""File editing module for adk-eval-core."""

from adk_eval_core.editing.edit import (
    EditResult,
    apply_indentation,
    apply_replacement,
    make_diff,
)

__all__ = [
    "EditResult",
    "apply_indentation",
    "apply_replacement",
    "make_diff",
]
