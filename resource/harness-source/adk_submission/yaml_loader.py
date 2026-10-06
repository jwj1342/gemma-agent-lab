"""Sandboxed YAML loader with ``!include`` support.

The ``!include`` tag resolves file paths **relative to the including
file's directory**, but strictly within the submission root directory.
Path traversal outside the root (via ``..`` or symlinks) is blocked.

Public entry points:
    - :func:`load_yaml`: Load a YAML file with sandboxed include support.
    - :func:`make_sandboxed_loader`: Create a bound loader class for custom loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .errors import LimitExceededError, SubmissionValidationError
from .limits import SubmissionLimits
from .paths import validate_sandboxed_path

# File extensions that !include may resolve to.
_INCLUDABLE_TEXT_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})
_INCLUDABLE_YAML_EXTENSIONS: frozenset[str] = frozenset({".yaml", ".yml"})
_INCLUDABLE_EXTENSIONS: frozenset[str] = (
    _INCLUDABLE_TEXT_EXTENSIONS | _INCLUDABLE_YAML_EXTENSIONS
)
_MAX_INCLUDE_DEPTH = 10

_ACTIVE_LIMITS_BY_ROOT: dict[Path, SubmissionLimits] = {}


def set_active_limits_for_root(root_dir: Path, limits: SubmissionLimits) -> None:
    """Record active SubmissionLimits for a given root_dir during compilation."""
    resolved = Path(root_dir).resolve()
    if len(_ACTIVE_LIMITS_BY_ROOT) >= 64 and resolved not in _ACTIVE_LIMITS_BY_ROOT:
        _ACTIVE_LIMITS_BY_ROOT.pop(next(iter(_ACTIVE_LIMITS_BY_ROOT)), None)
    _ACTIVE_LIMITS_BY_ROOT[resolved] = limits


def clear_active_limits_for_root(root_dir: Path) -> None:
    """Remove cached SubmissionLimits for a given root_dir."""
    _ACTIVE_LIMITS_BY_ROOT.pop(Path(root_dir).resolve(), None)


def _resolve_include_path(
    rel_path: str,
    current_dir: Path,
    root_dir: Path,
) -> Path:
    """Resolve a relative include path safely within the sandbox boundary.

    The path is resolved relative to ``current_dir`` (the directory of the
    file containing the ``!include`` tag), then sandboxed against ``root_dir``.

    Args:
        rel_path: The relative path specified in the ``!include`` tag.
        current_dir: The directory of the file containing the ``!include`` tag.
        root_dir: The submission root directory (sandbox boundary).

    Returns:
        The resolved absolute Path object.

    Raises:
        PathTraversalError: If the resolved path escapes root_dir, or if the target or any parent component is a symlink.
        SubmissionValidationError: If the file does not exist or has a disallowed extension.
    """
    raw_target = current_dir / rel_path
    return validate_sandboxed_path(
        path=raw_target,
        base_dir=root_dir,
        allow_relative=True,
        must_exist=True,
        allow_symlinks=False,
        allowed_extensions=_INCLUDABLE_EXTENSIONS,
        error_prefix="!include",
    )


def _measure_expanded_size(
    data: Any,
    max_chars: int,
    max_nodes: int = 50_000,
) -> int:
    """Measure expanded character size of parsed YAML data, aborting immediately if limits are exceeded."""
    total_chars = 0
    visited_nodes = 0
    stack: list[Any] = [data]
    while stack:
        curr = stack.pop()
        visited_nodes += 1
        if visited_nodes > max_nodes:
            raise LimitExceededError(
                f"Expanded YAML node count ({visited_nodes:,}) exceeds limit ({max_nodes:,}); "
                "possible YAML anchor bomb"
            )
        if isinstance(curr, dict):
            total_chars += 2
            for k, v in curr.items():
                total_chars += len(str(k)) + 4
                if total_chars > max_chars:
                    raise LimitExceededError(
                        f"Expanded YAML size ({total_chars:,} chars) exceeds limit ({max_chars:,})"
                    )
                stack.append(v)
        elif isinstance(curr, list):
            total_chars += 2 + max(0, len(curr) * 2)
            if total_chars > max_chars:
                raise LimitExceededError(
                    f"Expanded YAML size ({total_chars:,} chars) exceeds limit ({max_chars:,})"
                )
            stack.extend(curr)
        else:
            total_chars += len(str(curr))
            if total_chars > max_chars:
                raise LimitExceededError(
                    f"Expanded YAML size ({total_chars:,} chars) exceeds limit ({max_chars:,})"
                )
    return total_chars


class SandboxedYamlLoader(yaml.SafeLoader):
    """A ``yaml.SafeLoader`` subclass that resolves ``!include`` tags within a sandboxed root directory.

    **Do not instantiate directly** — use :func:`make_sandboxed_loader`
    or :func:`load_yaml` instead.

    Attributes:
        _root_dir: The submission root directory (sandbox boundary), set dynamically by make_sandboxed_loader.
        _current_dir: The directory of the file being loaded, set dynamically by make_sandboxed_loader.
        _depth: Current include nesting depth, used to enforce _MAX_INCLUDE_DEPTH.
    """

    _root_dir: Path  # set by make_sandboxed_loader
    _current_dir: Path  # set by make_sandboxed_loader
    _depth: int = 0
    _include_state: dict[str, int] = {}

    def compose_node(self, parent: yaml.Node | None, index: Any) -> yaml.Node | None:
        """Track composed YAML nodes during parsing."""
        if self._include_state is not None:
            max_bytes = self._include_state.get(
                "max_bytes", SubmissionLimits().max_yaml_size_bytes
            )
            max_nodes = max(max_bytes // 8, 10_000)
            count = self._include_state.get("composed_nodes", 0) + 1
            self._include_state["composed_nodes"] = count
            if count > max_nodes:
                raise LimitExceededError(
                    f"YAML node composition count ({count:,}) exceeds limit ({max_nodes:,})"
                )
        return super().compose_node(parent, index)

    def include(self, node: yaml.Node) -> Any:
        """Handle ``!include path/to/file`` tags during YAML parsing.

        Args:
            node: The YAML scalar node containing the relative file path.

        Returns:
            The parsed YAML content (if a .yaml or .yml file) or the raw text string (if a .md or .txt file).

        Raises:
            SubmissionValidationError: If include nesting depth exceeds ``_MAX_INCLUDE_DEPTH``, if the included file is not found, or if the file extension is disallowed.
            LimitExceededError: If an included file or cumulative included bytes exceed ``max_yaml_size_bytes``.
            PathTraversalError: If the include path attempts to escape the submission root directory.
            yaml.YAMLError: If an included YAML file is malformed.
        """
        rel_path: str = self.construct_scalar(node)  # type: ignore[arg-type]
        abs_path = _resolve_include_path(rel_path, self._current_dir, self._root_dir)

        file_size = abs_path.stat().st_size
        max_bytes = self._include_state.get(
            "max_bytes", SubmissionLimits().max_yaml_size_bytes
        )
        if file_size > max_bytes:
            raise LimitExceededError(
                f"Included file '{rel_path}' size {file_size:,} bytes "
                f"exceeds limit of {max_bytes:,} bytes"
            )
        if self._include_state:
            self._include_state["total_bytes"] = (
                self._include_state.get("total_bytes", 0) + file_size
            )
            if self._include_state["total_bytes"] > max_bytes:
                raise LimitExceededError(
                    f"Cumulative YAML/!include size ({self._include_state['total_bytes']:,} bytes) "
                    f"exceeds limit of {max_bytes:,} bytes"
                )

        content = abs_path.read_text(encoding="utf-8")

        if abs_path.suffix.lower() in _INCLUDABLE_YAML_EXTENSIONS:
            if self._depth >= _MAX_INCLUDE_DEPTH:
                raise SubmissionValidationError(
                    f"!include nesting too deep (max {_MAX_INCLUDE_DEPTH}): {rel_path}"
                )
            # Nested YAML: resolve relative to the *included* file's dir
            if "&" in content or "*" in content:
                self._include_state["has_anchors"] = 1
            loader_cls = make_sandboxed_loader(
                self._root_dir,
                current_dir=abs_path.parent,
                include_state=self._include_state,
            )
            loader_cls._depth = self._depth + 1
            data = yaml.load(content, Loader=loader_cls)
            max_expanded = max_bytes
            if "&" in content and "*" in content:
                max_expanded = min(max_expanded, max(file_size * 20, 65_536))
            _measure_expanded_size(data, max_expanded)
            return data

        # .md, .txt → raw string
        return content


def make_sandboxed_loader(
    root_dir: Path,
    current_dir: Path | None = None,
    include_state: dict[str, int] | None = None,
) -> type[SandboxedYamlLoader]:
    """Create a loader class bound to a root and current directory.

    Args:
        root_dir: The submission root directory (sandbox boundary).
        current_dir: The directory of the file being loaded. ``!include``
            paths are resolved relative to this. Defaults to ``root_dir``.
        include_state: Optional shared dictionary tracking cumulative included bytes.

    Returns:
        A dynamically created subclass of :class:`SandboxedYamlLoader` bound to the specified root and current directories.
        Each call returns a **new subclass** so that different directories don't interfere with each other.
    """
    resolved_root = Path(root_dir).resolve()
    if include_state is None:
        active_limits = _ACTIVE_LIMITS_BY_ROOT.get(resolved_root) or SubmissionLimits()
        include_state = {
            "total_bytes": 0,
            "max_bytes": active_limits.max_yaml_size_bytes,
        }

    class BoundLoader(SandboxedYamlLoader):
        pass

    BoundLoader._root_dir = resolved_root
    BoundLoader._current_dir = (current_dir or root_dir).resolve()
    BoundLoader._include_state = include_state
    BoundLoader.add_constructor("!include", BoundLoader.include)
    return BoundLoader


def load_yaml(
    path: Path,
    root_dir: Path,
    limits: SubmissionLimits | None = None,
) -> dict[str, Any]:
    """Load a YAML file with sandboxed ``!include`` support.

    ``!include`` paths are resolved relative to the directory of ``path``,
    but sandboxed within ``root_dir``.

    Args:
        path: The YAML file to load.
        root_dir: The root directory (sandbox boundary).
        limits: Optional submission limits to enforce on YAML and !include byte sizes.

    Returns:
        Parsed YAML content as a dictionary mapping.

    Raises:
        SubmissionValidationError: If the loaded YAML is not a top-level mapping, if an included file is not found, or if include nesting is too deep.
        LimitExceededError: If the YAML file, cumulative !include files, or anchor expansion exceeds ``max_yaml_size_bytes``.
        PathTraversalError: If ``path`` or any included file path escapes ``root_dir``.
        yaml.YAMLError: If the YAML file or any included YAML file is malformed.
    """
    validated_path = validate_sandboxed_path(
        path=path,
        base_dir=root_dir,
        allow_relative=True,
        must_exist=True,
        allow_symlinks=False,
        allowed_extensions={".yaml", ".yml"},
    )

    resolved_root = Path(root_dir).resolve()
    effective_limits = (
        limits
        or _ACTIVE_LIMITS_BY_ROOT.get(resolved_root)
        or SubmissionLimits()
    )
    max_bytes = effective_limits.max_yaml_size_bytes

    file_size = validated_path.stat().st_size
    if file_size > max_bytes:
        raise LimitExceededError(
            f"YAML file size {file_size:,} bytes exceeds limit of {max_bytes:,} bytes: {path}"
        )

    include_state = {"total_bytes": file_size, "max_bytes": max_bytes}
    loader_cls = make_sandboxed_loader(
        resolved_root,
        current_dir=validated_path.parent,
        include_state=include_state,
    )
    raw_text = validated_path.read_text(encoding="utf-8")
    data = yaml.load(raw_text, Loader=loader_cls)

    max_expanded = max_bytes
    if "&" in raw_text and "*" in raw_text:
        max_expanded = min(max_expanded, max(file_size * 20, 65_536))
    _measure_expanded_size(data, max_expanded)

    if not isinstance(data, dict):
        raise SubmissionValidationError(
            f"Expected YAML mapping at top level, got {type(data).__name__}: "
            f"{path}"
        )
    return data
