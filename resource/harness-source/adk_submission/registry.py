"""Closed registries for tools, models, skills, and callbacks.

Competition organizers populate these registries before calling
``compile_submission()``.  Submissions can only reference names that
have been registered — no ``importlib`` resolution is permitted.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar

from .errors import (
    CallbackNotFoundError,
    ModelNotFoundError,
    SkillNotFoundError,
    SubmissionError,
    ToolNotFoundError,
)

V = TypeVar("V")


# ---------------------------------------------------------------------------
# Generic base
# ---------------------------------------------------------------------------


class Registry(Generic[V]):
    """Base generic registry for storing and retrieving pre-approved objects by name.

    Provides common registration, lookup, and listing capabilities for competition
    assets (tools, models, skills). Subclasses must define the ``_error_cls`` class
    attribute with a :class:`SubmissionError` subclass that accepts ``name`` and
    ``available`` keyword arguments for error reporting.
    """

    _error_cls: ClassVar[type[SubmissionError]]

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._items: dict[str, V] = {}

    def register(self, name: str, value: V) -> None:
        """Register an asset under a short identifier name.

        If the specified name is already registered, its value will be silently overwritten.

        Args:
            name: The short string identifier for the asset.
            value: The asset instance or configuration to store (type ``V``).

        Returns:
            None
        """
        self._items[name] = value

    def get(self, name: str) -> V:
        """Retrieve a registered asset by its identifier name.

        Args:
            name: The string identifier of the asset to retrieve.

        Returns:
            The registered asset instance of type ``V``.

        Raises:
            SubmissionError: Specifically, the exception defined by ``_error_cls``
                (e.g., :class:`ToolNotFoundError`, :class:`ModelNotFoundError`) if the name is not registered.
        """
        try:
            return self._items[name]
        except KeyError:
            raise self._error_cls(name, available=self.list_available()) from None  # type: ignore[call-arg]

    def list_available(self) -> list[str]:
        """Return a lexicographically sorted list of all registered asset names.

        Returns:
            A sorted list of string identifier names currently in the registry.
        """
        return sorted(self._items)

    def get_all(self) -> list[V]:
        """Return a list of all registered asset values in insertion order.

        Returns:
            A list containing all registered values of type ``V``.
        """
        return list(self._items.values())

    def __contains__(self, name: str) -> bool:
        """Check if an identifier name is currently registered.

        Args:
            name: The string identifier to check.

        Returns:
            ``True`` if the name is registered, ``False`` otherwise.
        """
        return name in self._items

    def __len__(self) -> int:
        """Return the total number of registered assets.

        Returns:
            The integer count of items in the registry.
        """
        return len(self._items)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class ToolRegistry(Registry[Any]):
    """Registry of pre-approved tools available to submitted agents.

    Tools are typically registered as Python callables or ADK :class:`AgentTool`
    instances. Submitted agents reference these tools by their registered string names.

    Inherits from :class:`Registry` and raises :class:`ToolNotFoundError` when an
    unregistered tool name is requested.

    Example::

        tools = ToolRegistry()
        tools.register("google_search", google_search_fn)
        tools.register("load_web_page", load_web_page_fn)
    """

    _error_cls = ToolNotFoundError


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class ModelRegistry(Registry[Any]):
    """Registry mapping model aliases to LLM configuration strings or endpoints.

    Competition organizers map short model aliases (e.g., ``'fast'``) to fully qualified
    model identifiers or serving endpoints. Discovered adapters are also registered here
    under the ``adapter:<name>`` prefix.

    Inherits from :class:`Registry` and raises :class:`ModelNotFoundError` when an
    unregistered model alias is requested.

    Example::

        models = ModelRegistry()
        models.register("fast", "gemini-2.5-flash")
        models.register("strong", "gemini-2.5-pro")
        # After adapter discovery:
        models.register("adapter:my_lora", "hosted_vllm/llama3:my_lora")
    """

    _error_cls = ModelNotFoundError


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------


class SkillRegistry(Registry[Any]):
    """Registry mapping skill identifier names to instantiated ADK Skill objects.

    Populated either manually by the competition organizer or automatically via
    :meth:`SkillManifest.register_all` following skill discovery.

    Inherits from :class:`Registry` and raises :class:`SkillNotFoundError` when an
    unregistered skill name is requested.

    Example::

        skills = SkillRegistry()
        # After skill discovery:
        skill_manifest.register_all(skills, lambda info: load_skill_from_dir(info.path))
    """

    _error_cls = SkillNotFoundError

    def list_skills(self) -> list[str]:
        """Return a lexicographically sorted list of all registered skill names."""
        return self.list_available()


# ---------------------------------------------------------------------------
# Callback Registry
# ---------------------------------------------------------------------------


class CallbackType(Enum):
    """The position in the agent lifecycle where a callback can be used.

    Attributes:
        BEFORE_AGENT: Executed immediately before an agent begins its workflow or execution cycle.
        AFTER_AGENT: Executed immediately after an agent completes its workflow or execution cycle.
        BEFORE_MODEL: Executed before the underlying LLM generate_content call is made.
        AFTER_MODEL: Executed after the underlying LLM generate_content call returns.
        BEFORE_TOOL: Executed before a tool call is invoked by the agent.
        AFTER_TOOL: Executed after a tool call completes and returns its output.
    """

    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"


class CallbackRegistry:
    """Registry of pre-approved lifecycle callbacks available to submitted agents.

    Unlike :class:`ToolRegistry` or :class:`ModelRegistry`, callbacks are stored with
    optional lifecycle position restrictions (:class:`CallbackType`). Retrieving a callback
    via ``get()`` requires specifying the intended callback position to ensure safety and
    prevent invalid callback usage. This specialized lookup requirement prevents reuse of the
    generic :class:`Registry` base class.

    Example::

        callbacks = CallbackRegistry()
        callbacks.register(
            "rate_limit",
            rate_limit_fn,
            types={CallbackType.BEFORE_MODEL},
        )
        callbacks.register("log_to_state", log_fn)  # usable anywhere
    """

    def __init__(self) -> None:
        """Initialize an empty callback registry."""
        # name → (callable, allowed_types | None)
        self._callbacks: dict[str, tuple[Callable[..., Any], set[CallbackType] | None]] = {}

    def register(
        self,
        name: str,
        callback: Callable[..., Any],
        types: set[CallbackType] | None = None,
    ) -> None:
        """Register a lifecycle callback callable under a short identifier name.

        If the specified name is already registered, the previous callback and its
        allowed types will be silently overwritten.

        Args:
            name: The short string identifier submissions will use.
            callback: The callback callable function or object.
            types: Optional set of :class:`CallbackType` positions where this callback
                is permitted. If ``None``, the callback may be used in any position.

        Returns:
            None
        """
        self._callbacks[name] = (callback, set(types) if types is not None else None)

    def get(self, name: str, callback_type: CallbackType) -> Callable[..., Any]:
        """Retrieve a registered callback validated for a specific lifecycle position.

        Args:
            name: The string identifier of the callback to retrieve.
            callback_type: The :class:`CallbackType` position where the callback will be executed.

        Returns:
            The registered callback callable.

        Raises:
            CallbackNotFoundError: If the callback name is not registered, or if it is
                registered but not permitted for the requested ``callback_type``.
        """
        entry = self._callbacks.get(name)
        if entry is None:
            raise CallbackNotFoundError(
                name,
                callback_type=callback_type.value,
                available=self.list_available(),
            )

        callback, allowed_types = entry
        if allowed_types is not None and callback_type not in allowed_types:
            allowed_str = ", ".join(t.value for t in sorted(allowed_types, key=lambda t: t.value))
            raise CallbackNotFoundError(
                name,
                callback_type=callback_type.value,
                available=self.list_available(),
                allowed_types=allowed_str,
            )

        return callback

    def list_available(self) -> list[str]:
        """Return a lexicographically sorted list of all registered callback names.

        Returns:
            A sorted list of string identifier names currently in the registry.
        """
        return sorted(self._callbacks)

    def __contains__(self, name: str) -> bool:
        """Check if a callback identifier name is currently registered.

        Args:
            name: The string identifier to check.

        Returns:
            ``True`` if the callback name is registered, ``False`` otherwise.
        """
        return name in self._callbacks

    def __len__(self) -> int:
        """Return the total number of registered callbacks.

        Returns:
            The integer count of callbacks in the registry.
        """
        return len(self._callbacks)
