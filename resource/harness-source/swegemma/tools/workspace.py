"""Workspace file manipulation tools for SWE-gemma."""

from __future__ import annotations

import logging
import shlex
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from swegemma.tools.base import (
    budget_gated,
    error_response,
    ok_response,
    parse_tool_response,
)

logger = logging.getLogger(__name__)


def _resolve_workspace_path(
    filepath: str,
    workspace_root: Path = Path('/workspace'),
    *,
    allow_root: bool = True,
) -> Path:
    """Normalize and resolve relative paths strictly within the workspace directory."""
    workspace_root_resolved = workspace_root.resolve()
    ws_str = str(workspace_root_resolved)
    norm_filepath = filepath
    if norm_filepath == '/workspace':
        norm_filepath = ''
    elif norm_filepath.startswith('/workspace/'):
        norm_filepath = norm_filepath[len('/workspace/') :]
    elif ws_str != '/workspace':
        if norm_filepath == ws_str:
            norm_filepath = ''
        elif norm_filepath.startswith(ws_str + '/'):
            norm_filepath = norm_filepath[len(ws_str) + 1 :]
    if norm_filepath.startswith('/'):
        raise ValueError(
            f"Path traversal detected: '{filepath}' escapes workspace root."
        )
    clean_rel = Path(norm_filepath)
    resolved = (workspace_root_resolved / clean_rel).resolve()
    if not resolved.is_relative_to(workspace_root_resolved):
        raise ValueError(
            f"Path traversal detected: '{filepath}' escapes workspace root."
        )
    if not allow_root and resolved == workspace_root_resolved:
        raise ValueError(
            f"Invalid file path '{filepath}': refers to workspace root directory, not a file."
        )
    return resolved


def _coerce_optional_int(val: Any, param_name: str) -> int | None:
    """Coerce an optional integer tool parameter that may arrive as a string."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise TypeError(f'{param_name} must be an integer, got boolean {val}')
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ('none', 'null', 'nil'):
            return None
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
            raise ValueError(
                f'Invalid integer for {param_name}: {val!r}'
            ) from None
    raise ValueError(f'Invalid integer for {param_name}: {val!r}')


@budget_gated
def read_file(
    ctx: Any,
    filepath: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read the contents of a file from the repository workspace.

    Supports reading specific line ranges using start_line and end_line (1-indexed).
    To avoid overwhelming the context window, large files are automatically truncated
    to a maximum number of lines (default: 150 lines) and characters (default: 10,000 characters).

    Args:
        ctx: Context instance.
        filepath: Relative path within the workspace directory (e.g., "fastapi/main.py", "pyproject.toml").
        start_line: Optional starting line number (1-indexed, inclusive).
        end_line: Optional ending line number (1-indexed, inclusive).

    Returns:
        JSON string containing file content, line range, and truncation status.
    """
    try:
        coerced_start = _coerce_optional_int(start_line, 'start_line')
        coerced_end = _coerce_optional_int(end_line, 'end_line')

        dest_path = _resolve_workspace_path(filepath, allow_root=False)
        filename = dest_path.name

        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = Path(tmpdir) / filename
            ctx.docker.copy_from(ctx.container_id, str(dest_path), local_file)

            if not local_file.exists():
                raise FileNotFoundError(f"File '{filepath}' not found in workspace.")

            content = local_file.read_text(encoding='utf-8', errors='replace')

        lines = content.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return ok_response(
                filepath=filepath,
                content='',
                start_line=1,
                end_line=0,
                total_lines=0,
                is_truncated=False,
            )

        start = coerced_start if coerced_start is not None else 1
        end = coerced_end if coerced_end is not None else total_lines

        start = max(start, 1)
        end = min(end, total_lines)

        if start > end:
            return error_response(
                error_type='ValueError',
                error_message=f'start_line ({start}) cannot be greater than end_line ({end})',
                details={},
            )

        max_lines = getattr(ctx.harness, 'max_file_lines', 150)
        max_chars = getattr(ctx.harness, 'max_file_chars', 10000)
        is_truncated = False
        if max_lines is not None and (end - start + 1) > max_lines:
            end = start + max_lines - 1
            is_truncated = True

        selected_lines = lines[start - 1 : end]
        snippet = '\n'.join(selected_lines)

        if max_chars is not None and len(snippet) > max_chars:
            snippet = snippet[:max_chars]
            is_truncated = True

        return ok_response(
            filepath=filepath,
            content=snippet,
            start_line=start,
            end_line=end,
            total_lines=total_lines,
            is_truncated=is_truncated,
        )
    except Exception as e:
        return error_response(
            error_type='FileReadError',
            error_message=str(e),
        )


@budget_gated
def write_file(ctx: Any, filepath: str, content: str) -> str:
    """Create or overwrite a file in the workspace.

    Args:
        ctx: Context instance.
        filepath: Relative path to the file to create or overwrite.
        content: Full content of the file.
    """
    try:
        dest_path = _resolve_workspace_path(filepath, allow_root=False)
        quoted_parent = shlex.quote(str(dest_path.parent))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / dest_path.name
            tmp_file.write_text(content, encoding='utf-8')

            mkdir_res = ctx.docker.exec(ctx.container_id, f'mkdir -p {quoted_parent}')
            if mkdir_res.exit_code != 0:
                raise RuntimeError(
                    f'Failed to create directory {dest_path.parent} inside container: {mkdir_res.stderr}'
                )

            ctx.docker.copy_to(ctx.container_id, tmp_file, f'{dest_path.parent}/')

        return ok_response(
            filepath=filepath,
            size=len(content),
        )
    except Exception as e:
        return error_response(
            error_type='FileWriteError',
            error_message=str(e),
        )


@budget_gated
def edit_file(
    ctx: Any,
    filepath: str,
    old_string: str,
    new_string: str,
    allow_multiple: bool = False,
) -> str:
    """Replace a contiguous block of text in an existing file.

    More token-efficient than write_file for targeted changes. Provide 2-4 lines of unique
    surrounding context in old_string to ensure an unambiguous match.

    Args:
        ctx: Context instance.
        filepath: Relative path to the target file.
        old_string: Text block to replace.
        new_string: Replacement text block.
        allow_multiple: Replace all occurrences of old_string if True.
    """
    try:
        from swegemma.edit import apply_replacement

        dest_path = _resolve_workspace_path(filepath, allow_root=False)
        filename = dest_path.name

        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = Path(tmpdir) / filename
            ctx.docker.copy_from(ctx.container_id, str(dest_path), local_file)

            if not local_file.exists():
                raise FileNotFoundError(f"File '{filepath}' not found in workspace.")

            with open(local_file, encoding='utf-8', errors='replace', newline='') as f:
                content = f.read()

            if len(content) == 0:
                raise ValueError(
                    f"File '{filepath}' is empty. Use write_file to populate an empty or newly created file."
                )

            coerced_allow_multiple = (
                allow_multiple.strip().lower() in ('true', '1', 'yes')
                if isinstance(allow_multiple, str)
                else bool(allow_multiple)
            )

            res = apply_replacement(
                content,
                old_string,
                new_string,
                filepath=filepath,
                allow_multiple=coerced_allow_multiple,
            )

            with open(local_file, 'w', encoding='utf-8', newline='') as f:
                f.write(res.new_content)

            ctx.docker.copy_to(ctx.container_id, local_file, f'{dest_path.parent}/')

        max_chars = ctx.harness.max_stdout_chars
        diff_str = res.diff
        is_truncated = False
        if max_chars is not None and len(diff_str) > max_chars:
            diff_str = diff_str[:max_chars]
            is_truncated = True

        return ok_response(
            filepath=filepath,
            occurrences=res.occurrences,
            strategy=res.strategy,
            diff=diff_str,
            is_truncated=is_truncated,
        )
    except Exception as e:
        return error_response(
            error_type='FileEditError',
            error_message=str(e),
        )


def make_read_file(ctx: Any) -> Callable:
    def read_file_tool(
        filepath: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read the contents of a file from the repository workspace.

        Supports reading specific line ranges using start_line and end_line (1-indexed).
        To avoid overwhelming the context window, large files are automatically truncated
        to a maximum number of lines (default: 150 lines).

        Args:
            filepath: Relative path within the workspace directory (e.g., "fastapi/main.py", "pyproject.toml").
            start_line: Optional starting line number (1-indexed, inclusive).
            end_line: Optional ending line number (1-indexed, inclusive).

        Returns:
            Dictionary containing file content, line range, and truncation status.
        """
        return parse_tool_response(read_file(ctx, filepath, start_line, end_line))

    read_file_tool.__name__ = 'read_file'
    return read_file_tool


def make_write_file(ctx: Any) -> Callable:
    def write_file_tool(filepath: str, content: str) -> dict[str, Any]:
        """Create or overwrite a file in the workspace.

        Args:
            filepath: Relative path to the file to create or overwrite.
            content: Full content of the file.
        """
        return parse_tool_response(write_file(ctx, filepath, content))

    write_file_tool.__name__ = 'write_file'
    return write_file_tool


def make_edit_file(ctx: Any) -> Callable:
    def edit_file_tool(
        filepath: str,
        old_string: str,
        new_string: str,
        allow_multiple: bool = False,
    ) -> dict[str, Any]:
        """Replace a contiguous block of text in an existing file.

        More token-efficient than write_file for targeted changes. Provide 2-4 lines of unique
        surrounding context in old_string to ensure an unambiguous match.

        Args:
            filepath: Relative path to the target file.
            old_string: Text block to replace.
            new_string: Replacement text block.
            allow_multiple: Replace all occurrences of old_string if True.

        Returns:
            Dictionary containing file status, occurrences count, match strategy, and unified diff.
        """
        return parse_tool_response(
            edit_file(ctx, filepath, old_string, new_string, allow_multiple)
        )

    edit_file_tool.__name__ = 'edit_file'
    return edit_file_tool
