"""Sandboxed ADK agent compiler for Kaggle competition submissions.

Compile ADK agents from declarative YAML configs without permitting
arbitrary code execution. All tools, models, skills, and callbacks are resolved
through closed registries provided by the competition organizer.

Quick start::

    from adk_submission import (
        compile_submission, discover_adapters, discover_skills,
        ToolRegistry, ModelRegistry, SkillRegistry, CallbackRegistry,
        CallbackType, SubmissionLimits
    )

    # 1. Initialize and populate registries with custom callables
    tools = ToolRegistry()
    # `custom_google_search_fn` represents a user-provided tool callable
    tools.register("google_search", custom_google_search_fn)

    models = ModelRegistry()
    models.register("fast", "gemini-2.5-flash")

    skills = SkillRegistry()
    callbacks = CallbackRegistry()
    limits = SubmissionLimits()

    # 2. Discover adapters and skills from the unpacked submission directory
    submission_path = "/path/to/unpacked/submission"
    adapter_manifest = discover_adapters(submission_path, adapter_extensions={".safetensors"})
    adapter_manifest.register_all(models, lambda info: f"hosted_vllm/llama3:{info.name}")

    skill_manifest = discover_skills(submission_path, limits=limits)
    skill_manifest.register_all(skills, lambda info: load_skill_from_dir(info.path))

    # 3. Compile the submission into a runnable ADK agent
    agent = compile_submission(
        submission_dir=submission_path,
        tool_registry=tools,
        model_registry=models,
        skill_registry=skills,
        callback_registry=callbacks,
        limits=limits,
    )
"""

from typing import TYPE_CHECKING, Any

from .compiler import compile_submission
from .discovery import (
    AdapterInfo,
    AdapterManifest,
    SkillInfo,
    SkillManifest,
    SubmissionDirectory,
    discover_adapters,
    discover_declared_models,
    discover_skills,
    find_root_config,
    validate_directory,
)
from .errors import (
    AdapterNotFoundError,
    CallbackNotFoundError,
    LimitExceededError,
    ModelNotFoundError,
    PathTraversalError,
    ServerStartupError,
    SkillNotFoundError,
    SubmissionError,
    SubmissionSchemaError,
    SubmissionValidationError,
    ToolNotFoundError,
)
from .limits import GenerationConstraints, NumericRange, SubmissionLimits
from .registry import (
    CallbackRegistry,
    CallbackType,
    ModelRegistry,
    SkillRegistry,
    ToolRegistry,
)

if TYPE_CHECKING:
    from .server import (
        BaseInferenceServer,
        TransformersConfig,
        TransformersServer,
        VllmConfig,
        VllmServer,
        spawn_server,
        spawn_transformers_server,
        spawn_vllm_server,
    )

_SERVING_EXPORTS = {
    "BaseInferenceServer",
    "TransformersConfig",
    "TransformersServer",
    "VllmConfig",
    "VllmServer",
    "spawn_server",
    "spawn_transformers_server",
    "spawn_vllm_server",
}


def __getattr__(name: str) -> Any:
    if name in _SERVING_EXPORTS:
        from . import server
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SERVING_EXPORTS))


__version__ = "0.2.12"

__all__ = [
    "AdapterInfo",
    "AdapterManifest",
    "AdapterNotFoundError",
    # Server & Metric Execution
    "BaseInferenceServer",
    "CallbackNotFoundError",
    "CallbackRegistry",
    "CallbackType",
    "GenerationConstraints",
    "LimitExceededError",
    "ModelNotFoundError",
    "ModelRegistry",
    "NumericRange",
    "PathTraversalError",
    "ServerStartupError",
    "SkillInfo",
    "SkillManifest",
    "SkillNotFoundError",
    "SkillRegistry",
    "SubmissionDirectory",
    # Errors
    "SubmissionError",
    # Limits & constraints
    "SubmissionLimits",
    "SubmissionSchemaError",
    "SubmissionValidationError",
    "ToolNotFoundError",
    # Registries
    "ToolRegistry",
    "TransformersConfig",
    "TransformersServer",
    "VllmConfig",
    "VllmServer",
    # Core
    "compile_submission",
    # Discovery
    "discover_adapters",
    "discover_declared_models",
    "discover_skills",
    "find_root_config",
    "spawn_server",
    "spawn_transformers_server",
    "spawn_vllm_server",
    "validate_directory",
]
