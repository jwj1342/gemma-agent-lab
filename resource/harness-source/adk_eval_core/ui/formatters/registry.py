"""Registry for managing and resolving tool formatters."""

from __future__ import annotations

from adk_eval_core.ui.formatters.command import CommandToolFormatter
from adk_eval_core.ui.formatters.file import FileToolFormatter
from adk_eval_core.ui.formatters.generic import GenericToolFormatter
from adk_eval_core.ui.formatters.models import ToolFormatter
from adk_eval_core.ui.formatters.python import PythonToolFormatter


class ToolFormatterRegistry:
    """Registry for managing and resolving tool formatters by exact name or prefix pattern."""

    def __init__(
        self,
        fallback_formatter: ToolFormatter | None = None,
        *,
        fallback: ToolFormatter | None = None,
    ) -> None:
        self._exact_formatters: dict[str, ToolFormatter] = {}
        self._prefix_formatters: list[tuple[str, ToolFormatter]] = []
        resolved_fallback = fallback_formatter or fallback or GenericToolFormatter()
        self._fallback_formatter: ToolFormatter = resolved_fallback

    @property
    def fallback(self) -> ToolFormatter:
        """The fallback tool formatter."""
        return self._fallback_formatter

    @property
    def _fallback(self) -> ToolFormatter:
        """Private alias for fallback tool formatter."""
        return self._fallback_formatter

    def copy(self) -> ToolFormatterRegistry:
        """Return a shallow copy of this registry."""
        cloned = ToolFormatterRegistry(fallback_formatter=self._fallback_formatter)
        cloned._exact_formatters = dict(self._exact_formatters)
        cloned._prefix_formatters = list(self._prefix_formatters)
        return cloned

    def register(
        self,
        tool_name: str,
        formatter: ToolFormatter,
        *,
        prefix: bool = False,
    ) -> None:
        """Register a formatter for a specific tool name or name prefix."""
        if prefix:
            clean_prefix = tool_name.rstrip("*")
            self._prefix_formatters = [
                (p, f)
                for p, f in self._prefix_formatters
                if p != clean_prefix and p != tool_name
            ]
            self._prefix_formatters.append((clean_prefix, formatter))
        else:
            self._exact_formatters[tool_name] = formatter

    def unregister(self, tool_name: str) -> bool:
        """Remove a registered exact or prefix formatter. Returns True if removed."""
        removed = False
        if tool_name in self._exact_formatters:
            del self._exact_formatters[tool_name]
            removed = True
        clean_prefix = tool_name.rstrip("*")
        before_len = len(self._prefix_formatters)
        self._prefix_formatters = [
            (p, f)
            for p, f in self._prefix_formatters
            if p != tool_name and p != clean_prefix
        ]
        if len(self._prefix_formatters) < before_len:
            removed = True
        return removed

    def get(self, tool_name: str) -> ToolFormatter:
        """Resolve the appropriate formatter for the given tool name."""
        if tool_name in self._exact_formatters:
            return self._exact_formatters[tool_name]

        # Check prefixes (longest matching prefix takes precedence)
        matching = [
            (p, f) for p, f in self._prefix_formatters if tool_name.startswith(p)
        ]
        if matching:
            matching.sort(key=lambda item: len(item[0]), reverse=True)
            return matching[0][1]

        return self._fallback_formatter

    def list_registered(self) -> list[str]:
        """Return list of all registered exact tool names and prefixes."""
        names = list(self._exact_formatters.keys())
        names.extend(f"{p}*" for p, _ in self._prefix_formatters)
        return sorted(names)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool name is registered."""
        return tool_name in self._exact_formatters or any(
            tool_name.startswith(p) for p, _ in self._prefix_formatters
        )

    def __len__(self) -> int:
        return len(self._exact_formatters) + len(self._prefix_formatters)


def create_default_tool_formatter_registry() -> ToolFormatterRegistry:
    """Create and populate a ToolFormatterRegistry with standard core formatters."""
    registry = ToolFormatterRegistry()
    cmd_fmt = CommandToolFormatter()
    file_fmt = FileToolFormatter()
    py_fmt = PythonToolFormatter()

    # Register command tools
    for name in (
        "run_command",
        "bash",
        "execute_command",
        "shell",
        "exec",
        "terminal",
    ):
        registry.register(name, cmd_fmt)

    # Register filesystem tools
    for name in (
        "write_file",
        "read_file",
        "view_file",
        "edit_file",
        "replace_file_content",
        "create_file",
    ):
        registry.register(name, file_fmt)

    # Register python tools
    for name in (
        "python",
        "python_repl",
        "execute_python",
        "run_python",
        "py_exec",
    ):
        registry.register(name, py_fmt)

    return registry


# Module-level default registry instance and helpers
_DEFAULT_REGISTRY: ToolFormatterRegistry | None = None


def get_default_formatter_registry() -> ToolFormatterRegistry:
    """Get or initialize the global default ToolFormatterRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = create_default_tool_formatter_registry()
    return _DEFAULT_REGISTRY


def set_default_formatter_registry(registry: ToolFormatterRegistry) -> None:
    """Set the global default ToolFormatterRegistry."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = registry


def register_tool_formatter(
    tool_name: str,
    formatter: ToolFormatter,
    *,
    prefix: bool = False,
) -> None:
    """Convenience function to register a tool formatter in the global default registry."""
    get_default_formatter_registry().register(tool_name, formatter, prefix=prefix)


__all__ = [
    "ToolFormatterRegistry",
    "create_default_tool_formatter_registry",
    "get_default_formatter_registry",
    "register_tool_formatter",
    "set_default_formatter_registry",
]
