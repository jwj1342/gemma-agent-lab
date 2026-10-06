"""Core replacement engine for file editing tool.

Implements tiered matching strategies (Exact -> Flexible -> Regex)
mirroring Gemini CLI's replacement logic.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from typing import Literal

from adk_eval_core.errors import AmbiguousMatchError


@dataclasses.dataclass
class EditResult:
    """Result of an edit replacement operation."""

    new_content: str
    occurrences: int
    strategy: Literal["exact", "flexible", "regex", "unescaped", "none"]
    error_message: str | None = None


def make_diff(old_content: str, new_content: str, filepath: str = "file") -> str:
    """Generates a unified diff snippet."""
    diff_lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff_lines)


def _find_flexible_match(
    lines: list[str], old_lines: list[str], filepath: str = "file"
) -> int | None:
    """Finds matching line index allowing for relaxed trailing whitespace and indentation differences.

    Raises ValueError if multiple ambiguous matches are found.
    """
    if not old_lines:
        return None

    n_old = len(old_lines)
    if len(lines) < n_old:
        return None

    # Try rstrip match first (trailing whitespace relaxed)
    norm_old_rstrip = [line.rstrip() for line in old_lines]
    rstrip_matches: list[int] = []
    for i in range(len(lines) - n_old + 1):
        candidate_window = lines[i : i + n_old]
        if [line.rstrip() for line in candidate_window] == norm_old_rstrip:
            rstrip_matches.append(i)

    if len(rstrip_matches) == 1:
        return rstrip_matches[0]
    if len(rstrip_matches) > 1:
        raise AmbiguousMatchError(
            f"Target string flexibly matches {len(rstrip_matches)} locations in {filepath}. "
            "Please provide more surrounding context.",
            matches_count=len(rstrip_matches),
            filepath=filepath,
            old_string="\n".join(old_lines),
        )

    # Fallback to full strip match (indentation relaxed)
    norm_old_strip = [line.strip() for line in old_lines]
    strip_matches: list[int] = []
    for i in range(len(lines) - n_old + 1):
        candidate_window = lines[i : i + n_old]
        if [line.strip() for line in candidate_window] == norm_old_strip:
            strip_matches.append(i)

    if len(strip_matches) == 1:
        return strip_matches[0]
    if len(strip_matches) > 1:
        raise AmbiguousMatchError(
            f"Target string flexibly matches {len(strip_matches)} locations in {filepath}. "
            "Please provide more surrounding context.",
            matches_count=len(strip_matches),
            filepath=filepath,
            old_string="\n".join(old_lines),
        )

    return None


def apply_indentation(lines: list[str], target_indent: str) -> list[str]:
    """Applies target indentation to a block of lines, preserving relative indents."""
    if not lines:
        return []

    ref_indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)

    result = []
    for line in lines:
        if not line.strip():
            result.append("")
        else:
            result.append(target_indent + line[ref_indent:])
    return result


def _apply_exact_match(
    current_content: str,
    old_string: str,
    new_string: str,
    allow_multiple: bool,
) -> EditResult | None:
    exact_count = current_content.count(old_string)
    if exact_count == 0:
        return None

    if not allow_multiple and exact_count > 1:
        return EditResult(
            new_content=current_content,
            occurrences=exact_count,
            strategy="exact",
            error_message=(
                f"Expected 1 occurrence but found {exact_count}. "
                "If you intended to replace multiple occurrences, set 'allow_multiple' to true."
            ),
        )

    new_content = current_content.replace(old_string, new_string)
    return EditResult(
        new_content=new_content,
        occurrences=exact_count,
        strategy="exact",
    )


def _apply_flexible_match(
    current_content: str,
    old_string: str,
    new_string: str,
    allow_multiple: bool,
) -> EditResult | None:
    """Applies flexible matching by comparing rstripped and stripped lines within a sliding window."""
    current_lines = current_content.splitlines(keepends=True)
    old_lines = old_string.splitlines()
    new_lines = new_string.splitlines()

    if not old_lines:
        return None

    # Pass 1: rstrip match
    norm_old_rstrip = [line.rstrip() for line in old_lines]
    matches: list[int] = []
    i = 0
    while i <= len(current_lines) - len(old_lines):
        window = current_lines[i : i + len(old_lines)]
        if [line.rstrip() for line in window] == norm_old_rstrip:
            matches.append(i)
            i += len(old_lines)
        else:
            i += 1

    # Pass 2: strip match fallback
    if not matches:
        norm_old_strip = [line.strip() for line in old_lines]
        i = 0
        while i <= len(current_lines) - len(old_lines):
            window = current_lines[i : i + len(old_lines)]
            if [line.strip() for line in window] == norm_old_strip:
                matches.append(i)
                i += len(old_lines)
            else:
                i += 1

    if not matches:
        return None

    if not allow_multiple and len(matches) > 1:
        return EditResult(
            new_content=current_content,
            occurrences=len(matches),
            strategy="flexible",
            error_message=(
                f"Expected 1 occurrence but found {len(matches)}. "
                "If you intended to replace multiple occurrences, set 'allow_multiple' to true."
            ),
        )

    line_sep = "\r\n" if "\r\n" in current_content else "\n"

    old_ref_indent = min((len(l) - len(l.lstrip()) for l in old_lines if l.strip()), default=0)
    new_ref_indent = min((len(l) - len(l.lstrip()) for l in new_lines if l.strip()), default=0)
    relative_indent_delta = max(0, new_ref_indent - old_ref_indent)

    # Apply replacements from bottom to top so line indices remain valid
    for match_idx in reversed(matches):
        if new_string == "":
            current_lines[match_idx : match_idx + len(old_lines)] = []
            continue

        window = current_lines[match_idx : match_idx + len(old_lines)]
        non_empty_window = [l for l in window if l.strip()]
        if non_empty_window:
            min_line = min(non_empty_window, key=lambda l: len(l) - len(l.lstrip()))
            indent_match = re.match(r"^([ \t]*)", min_line)
            indent = indent_match.group(1) if indent_match else ""
        else:
            first_line = current_lines[match_idx]
            indent_match = re.match(r"^([ \t]*)", first_line)
            indent = indent_match.group(1) if indent_match else ""

        target_indent = indent + (" " * relative_indent_delta)
        indented_new = apply_indentation(new_lines, target_indent)
        replacement_text = line_sep.join(indented_new)

        # Preserve trailing newline if the last line of the matched window had one
        last_window_line = current_lines[match_idx + len(old_lines) - 1]
        if last_window_line.endswith("\r\n"):
            if not replacement_text.endswith("\r\n"):
                replacement_text = replacement_text.rstrip("\r\n") + "\r\n"
        elif last_window_line.endswith("\n"):
            if not replacement_text.endswith("\n"):
                replacement_text += "\n"
        elif replacement_text.endswith(("\r\n", "\n")):
            replacement_text = replacement_text.rstrip("\r\n")

        replacement_lines = replacement_text.splitlines(keepends=True)
        current_lines[match_idx : match_idx + len(old_lines)] = replacement_lines

    new_content = "".join(current_lines)
    return EditResult(
        new_content=new_content,
        occurrences=len(matches),
        strategy="flexible",
    )


def _apply_regex_match(
    current_content: str,
    old_string: str,
    new_string: str,
    allow_multiple: bool,
) -> EditResult | None:
    token_pattern = re.compile(
        r"==|!=|<=|>=|->|:=|\+=|-=|\*\*|//|[():\[\]{},><=+\-*/]|\w+|\S+"
    )
    tokens = token_pattern.findall(old_string)
    if not tokens:
        return None

    parts: list[str] = []
    for idx, tok in enumerate(tokens):
        if idx > 0:
            prev_tok = tokens[idx - 1]
            if re.fullmatch(r"\w+", prev_tok) and re.fullmatch(r"\w+", tok):
                parts.append(r"\s+")
            else:
                parts.append(r"\s*")
        parts.append(re.escape(tok))

    pattern_str = "".join(parts)
    final_pattern = f"^([ \t]*){pattern_str}[ \t]*(?=\\r?\\n|$)"

    try:
        matches = list(re.finditer(final_pattern, current_content, flags=re.MULTILINE))
    except re.error:
        return None

    if not matches:
        return None

    if not allow_multiple and len(matches) > 1:
        return EditResult(
            new_content=current_content,
            occurrences=len(matches),
            strategy="regex",
            error_message=(
                f"Expected 1 occurrence but found {len(matches)}. "
                "If you intended to replace multiple occurrences, set 'allow_multiple' to true."
            ),
        )

    new_lines = new_string.splitlines()
    line_sep = "\r\n" if "\r\n" in current_content else "\n"

    def repl(match: re.Match) -> str:
        indent = match.group(1) or ""
        indented = apply_indentation(new_lines, indent)
        return line_sep.join(indented)

    count = len(matches)
    new_content = re.sub(
        final_pattern,
        repl,
        current_content,
        count=0 if allow_multiple else 1,
        flags=re.MULTILINE,
    )

    return EditResult(
        new_content=new_content,
        occurrences=count,
        strategy="regex",
    )


_UNICODE_ONLY_ESCAPE_RE = re.compile(r"(?<!\\)\\u([0-9a-fA-F]{4})")
_INLINE_JSON_ESCAPE_RE = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\])')
_FULL_JSON_ESCAPE_RE = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\/bfnrt])')

_FULL_JSON_ESCAPE_MAP: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _unescape_unicode_only(text: str) -> str:
    """Unescapes \\uXXXX sequences without altering quotes or other backslashes."""
    return _UNICODE_ONLY_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _unescape_inline_json_layer(text: str) -> str:
    """Unescapes one layer of inline JSON escapes (\\\", \\\\, \\uXXXX) while preserving \\n/\\r/\\t."""

    def _repl(match: re.Match[str]) -> str:
        seq = match.group(1)
        if seq.startswith("u"):
            return chr(int(seq[1:], 16))
        return seq

    return _INLINE_JSON_ESCAPE_RE.sub(_repl, text)


def _unescape_full_json_layer(text: str) -> str:
    """Unescapes one full layer of JSON string escapes including \\n, \\r, \\t."""

    def _repl(match: re.Match[str]) -> str:
        seq = match.group(1)
        if seq.startswith("u"):
            return chr(int(seq[1:], 16))
        return _FULL_JSON_ESCAPE_MAP.get(seq, seq)

    return _FULL_JSON_ESCAPE_RE.sub(_repl, text)


def _apply_unescaped_match(
    current_content: str,
    old_string: str,
    new_string: str,
    allow_multiple: bool,
) -> EditResult | None:
    """Fallback strategy that unescapes 1- or 2-level JSON-escaped strings when normal tiers fail."""
    if "\\" not in old_string:
        return None

    transforms = (
        _unescape_unicode_only,
        _unescape_inline_json_layer,
        lambda s: _unescape_inline_json_layer(_unescape_inline_json_layer(s)),
        _unescape_full_json_layer,
        lambda s: _unescape_full_json_layer(_unescape_full_json_layer(s)),
    )

    seen_old: set[str] = {old_string}
    for transform in transforms:
        candidate_old = transform(old_string)
        if not candidate_old or candidate_old in seen_old:
            continue
        seen_old.add(candidate_old)

        candidate_new = transform(new_string)
        if candidate_old == candidate_new:
            continue

        exact_res = _apply_exact_match(
            current_content, candidate_old, candidate_new, allow_multiple
        )
        if exact_res is not None:
            if exact_res.error_message is None:
                exact_res.strategy = "unescaped"
            return exact_res

        flex_res = _apply_flexible_match(
            current_content, candidate_old, candidate_new, allow_multiple
        )
        if flex_res is not None:
            if flex_res.error_message is None:
                flex_res.strategy = "unescaped"
            return flex_res

    return None


def apply_replacement(
    current_content: str,
    old_string: str,
    new_string: str,
    allow_multiple: bool = False,
) -> EditResult:
    """Executes tiered replacement strategies (Exact -> Flexible -> Regex -> Unescaped)."""
    if not old_string:
        return EditResult(
            new_content=current_content,
            occurrences=0,
            strategy="none",
            error_message="old_string cannot be empty for existing files.",
        )

    if old_string == new_string:
        return EditResult(
            new_content=current_content,
            occurrences=1,
            strategy="exact",
            error_message="No changes to apply. old_string and new_string are identical.",
        )

    # 1. Exact Match Strategy
    exact_res = _apply_exact_match(current_content, old_string, new_string, allow_multiple)
    if exact_res:
        return exact_res

    # 2. Flexible Match Strategy
    flex_res = _apply_flexible_match(current_content, old_string, new_string, allow_multiple)
    if flex_res:
        return flex_res

    # 3. Regex Match Strategy
    regex_res = _apply_regex_match(current_content, old_string, new_string, allow_multiple)
    if regex_res:
        return regex_res

    # 4. JSON-Unescaping Fallback Strategy
    unescaped_res = _apply_unescaped_match(
        current_content, old_string, new_string, allow_multiple
    )
    if unescaped_res:
        return unescaped_res

    return EditResult(
        new_content=current_content,
        occurrences=0,
        strategy="none",
        error_message=(
            "Failed to replace: old_string not found. "
            "Ensure you're not escaping content incorrectly and check whitespace, indentation, and context."
        ),
    )
