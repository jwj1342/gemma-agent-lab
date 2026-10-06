"""Sandboxed path validation and traversal prevention utilities.

This module provides centralized, robust functions to validate, resolve, and sanitize
file and directory paths against directory traversal attacks (such as absolute paths,
parent directory climbing with '..', or symlinks pointing outside the sandbox boundary).
"""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path
from typing import NoReturn

from pydantic import ValidationError

from .errors import PathTraversalError, SubmissionSchemaError, SubmissionValidationError

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:")


def ensure_no_traversal_components(
    path: str | Path,
    context: str = "",
) -> None:
    """Ensure a path specification contains no directory traversal indicators.

    Performs strict lexical validation to verify that:
    1. The path is not absolute (does not start with '/', '\\', or Windows drive letters like 'C:').
    2. The path contains no parent directory climbing segments ('..').

    Args:
        path: Path string or Path object to validate.
        context: Optional descriptive context for error messages (e.g., 'in skill').

    Raises:
        PathTraversalError: If the path is absolute or contains '..' components.
    """
    raw = str(path).strip()
    if not raw:
        return

    ctx_suffix = f" {context}" if context else ""

    # Check for absolute path indicators (POSIX and Windows)
    if (
        raw.startswith(("/", "\\"))
        or bool(_WINDOWS_DRIVE_PATTERN.match(raw))
        or Path(raw).is_absolute()
    ):
        raise PathTraversalError(f"Path traversal attempted{ctx_suffix}: {raw}")

    # Check for '..' components across all path separators
    normalized_parts = raw.replace("\\", "/").split("/")
    if ".." in normalized_parts:
        raise PathTraversalError(f"Path traversal attempted{ctx_suffix}: {raw}")


def validate_sandboxed_path(
    path: str | Path,
    base_dir: str | Path,
    allow_relative: bool = True,
    must_exist: bool = False,
    allow_symlinks: bool = False,
    allowed_extensions: Collection[str] | None = None,
    error_prefix: str = "",
) -> Path:
    """Resolve and validate a path strictly within a base directory sandbox.

    Args:
        path: The path to validate (relative or absolute).
        base_dir: The root sandbox directory that path must reside within.
        allow_relative: If True, relative paths are resolved against base_dir.
            If False, relative paths are rejected. Defaults to True.
        must_exist: If True, verifies that the target path exists on disk.
            Defaults to False.
        allow_symlinks: If False, rejects the target or any parent directory
            if it is a symbolic link. Defaults to False.
        allowed_extensions: Optional container of permitted file extensions
            (e.g., frozenset({'.md', '.txt'})). Checked case-insensitively.
        error_prefix: Optional string prepended to error messages.

    Returns:
        The canonical, resolved Path object strictly inside base_dir.

    Raises:
        PathTraversalError: If the resolved path escapes base_dir, or if symlinks
            are encountered when allow_symlinks is False.
        SubmissionValidationError: If must_exist is True and the file does not exist,
            or if the file extension is not permitted.
    """
    base = Path(base_dir).resolve()
    p = Path(path)

    if not allow_relative and not p.is_absolute():
        msg = f"{error_prefix}: Path must be absolute: {path}" if error_prefix else f"Path must be absolute: {path}"
        raise PathTraversalError(msg)

    # Compute raw target path before resolving symlinks
    raw_target = p if p.is_absolute() else (base / p)

    # Symlink verification before resolve
    if not allow_symlinks:
        if raw_target.is_symlink():
            if error_prefix == "Skill":
                msg = f"Skill path escapes submission directory: {path}"
            elif error_prefix == "Sub-agent":
                msg = f"Sub-agent config_path escapes submission directory: {path}"
            elif error_prefix == "AgentTool":
                msg = f"AgentTool config_path escapes submission directory: {path}"
            elif error_prefix == "!include":
                msg = f"!include target is a symlink: {path}"
            elif error_prefix:
                msg = f"{error_prefix} target is a symlink: {path}"
            else:
                msg = f"Target is a symlink: {path}"
            raise PathTraversalError(msg)
        for parent in raw_target.parents:
            if parent == base:
                break
            if parent.is_symlink():
                if error_prefix == "!include":
                    msg = f"!include path contains a symlink component: {parent.name}"
                elif error_prefix:
                    msg = f"{error_prefix} path contains a symlink component: {parent.name}"
                else:
                    msg = f"Path contains a symlink component: {parent.name}"
                raise PathTraversalError(msg)

    # Strict resolution
    resolved_target = raw_target.resolve()

    # Sandboxing check: target must be inside base directory
    if not resolved_target.is_relative_to(base):
        if error_prefix == "Skill":
            msg = f"Skill path escapes submission directory: {path}"
        elif error_prefix == "Sub-agent":
            msg = f"Sub-agent config_path escapes submission directory: {path}"
        elif error_prefix == "AgentTool":
            msg = f"AgentTool config_path escapes submission directory: {path}"
        elif error_prefix == "!include":
            msg = f"!include path escapes submission directory: {path}"
        elif error_prefix:
            msg = f"{error_prefix} path escapes submission directory: {path}"
        else:
            msg = f"Path escapes submission directory: {path}"
        raise PathTraversalError(msg)

    # Existence check
    if must_exist and not resolved_target.exists():
        if error_prefix == "Skill":
            msg = f"Skill directory not found: {path}"
        elif error_prefix in ("Sub-agent", "AgentTool"):
            msg = f"{error_prefix} config not found: {path}"
        elif error_prefix == "!include":
            msg = f"!include file not found: {path}"
        elif error_prefix:
            msg = f"{error_prefix} not found: {path}"
        else:
            msg = f"File or directory not found: {path}"
        raise SubmissionValidationError(msg)

    # Extension check
    if allowed_extensions is not None and resolved_target.is_file():
        ext = resolved_target.suffix.lower()
        allowed_lower = {e.lower() for e in allowed_extensions}
        if ext not in allowed_lower:
            allowed_str = ", ".join(sorted(allowed_lower))
            if error_prefix == "!include":
                msg = f"!include does not support '{resolved_target.suffix}' files (allowed: {allowed_str}): {path}"
            elif error_prefix:
                msg = f"{error_prefix} disallowed file extension '{resolved_target.suffix}' (allowed: {allowed_str}): {path}"
            else:
                msg = f"Disallowed file extension '{resolved_target.suffix}' (allowed: {allowed_str}): {path}"
            raise SubmissionValidationError(msg)

    return resolved_target


def handle_schema_validation_error(e: Exception, context_desc: str = "") -> NoReturn:
    """Inspect schema validation exceptions and raise appropriate typed domain errors.

    Args:
        e: The caught exception during Pydantic schema validation.
        context_desc: Contextual description of the configuration being parsed.

    Raises:
        PathTraversalError: If the underlying validation failure was caused by path traversal.
        SubmissionSchemaError: For all other schema parsing/validation failures.
    """
    if isinstance(e, PathTraversalError):
        raise e

    if isinstance(e, ValidationError):
        for err in e.errors():
            ctx = err.get("ctx")
            ctx_err = ctx.get("error") if isinstance(ctx, dict) else None
            if isinstance(ctx_err, PathTraversalError):
                desc_str = f" in {context_desc}" if context_desc else ""
                raise PathTraversalError(
                    f"Path traversal attempted{desc_str}: {ctx_err}"
                ) from e

    # Format schema validation error message according to context
    if context_desc and "sub-agent config" in context_desc.lower():
        sub_info = context_desc[context_desc.find("(") :] if "(" in context_desc else ""
        prefix = f"Sub-agent schema validation failed {sub_info}".strip()
        raise SubmissionSchemaError(f"{prefix}: {e}") from e
    elif context_desc and "agenttool config" in context_desc.lower():
        tool_info = context_desc[context_desc.find("(") :] if "(" in context_desc else ""
        prefix = f"AgentTool schema validation failed {tool_info}".strip()
        raise SubmissionSchemaError(f"{prefix}: {e}") from e
    elif context_desc == "config":
        raise SubmissionSchemaError(f"Submission schema validation failed: {e}") from e
    else:
        desc_str = f" ({context_desc})" if context_desc else ""
        raise SubmissionSchemaError(
            f"Submission schema validation failed{desc_str}: {e}"
        ) from e
