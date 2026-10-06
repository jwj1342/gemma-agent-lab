"""Data models and model registry setup for SWE-bench evaluation."""

from swegemma.models.discovery import (
    discover_declared_models,
    normalize_model_name,
    resolve_local_model_path,
    validate_single_declared_model,
)
from swegemma.models.registry import (
    resolve_swegemma_adapter,
    setup_gemma_model_registry,
)
from swegemma.models.task import (
    EvaluationResult,
    Task,
    TaskResult,
    extract_test_files_from_patch,
    load_tasks,
)

__all__ = [
    'EvaluationResult',
    'Task',
    'TaskResult',
    'discover_declared_models',
    'extract_test_files_from_patch',
    'load_tasks',
    'normalize_model_name',
    'resolve_local_model_path',
    'resolve_swegemma_adapter',
    'setup_gemma_model_registry',
    'validate_single_declared_model',
]
