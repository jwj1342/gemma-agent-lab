"""Resilient file editing module delegating to adk_eval_core with diff feedback."""

from __future__ import annotations

import dataclasses
from typing import Literal

from adk_eval_core.editing import EditResult, apply_indentation, make_diff
from adk_eval_core.editing import apply_replacement as core_apply_replacement


@dataclasses.dataclass
class AppliedEdit:
    """Result of an applied edit with unified diff and strategy telemetry."""

    new_content: str
    diff: str
    occurrences: int = 1
    strategy: Literal['exact', 'flexible', 'regex', 'unescaped', 'none'] = 'exact'

    def __iter__(self):
        return iter((self.new_content, self.diff))

    def __getitem__(self, index: int) -> str:
        return (self.new_content, self.diff)[index]


def apply_replacement(
    content: str,
    old_string: str,
    new_string: str,
    filepath: str = 'file',
    allow_multiple: bool = False,
) -> AppliedEdit:
    """Applies replacement to content using tiered matching (exact -> flexible -> regex).

    Args:
        content: The original file text.
        old_string: The string to be replaced.
        new_string: Replacement content.
        filepath: Basename for diff output.
        allow_multiple: Whether to allow replacing multiple occurrences.

    Returns:
        An AppliedEdit instance (which unpacks as (new_content, diff) for backward compatibility).

    Raises:
        ValueError: If target string is empty, not found, or matches multiple locations when allow_multiple is False.
    """
    if not old_string:
        raise ValueError('old_string cannot be empty.')

    res = core_apply_replacement(
        content, old_string, new_string, allow_multiple=allow_multiple
    )
    if res.error_message:
        if res.occurrences > 1:
            if res.strategy == 'flexible':
                raise ValueError(
                    f'Target string flexibly matches {res.occurrences} locations in {filepath}. '
                    'Please provide more surrounding context.'
                )
            raise ValueError(
                f'Target string occurs {res.occurrences} times in {filepath}. '
                'Please provide more surrounding context.'
            )
        raise ValueError(res.error_message)

    diff = make_diff(content, res.new_content, filepath)
    return AppliedEdit(
        new_content=res.new_content,
        diff=diff,
        occurrences=res.occurrences,
        strategy=res.strategy,
    )


__all__ = [
    'AppliedEdit',
    'EditResult',
    'apply_indentation',
    'apply_replacement',
]
