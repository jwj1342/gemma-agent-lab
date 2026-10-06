"""Model discovery, validation, and local path resolution for Gemma 4 submissions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from adk_submission.discovery import (
    discover_declared_models as _adk_discover_declared_models,
)
from adk_submission.discovery import (
    find_root_config,
)
from adk_submission.errors import PathTraversalError, SubmissionValidationError
from adk_submission.paths import validate_sandboxed_path
from adk_submission.yaml_loader import load_yaml

from swegemma.submission import ParticipantVisibleError

logger = logging.getLogger(__name__)

# Known Kaggle model dataset versions for Gemma 4 family
_DEFAULT_KAGGLE_MODEL_VERSIONS: dict[str, str] = {
    'gemma-4-31b-it-qat-w4a16-ct': '1',
    'gemma-4-31b-it': '1',
    'gemma-4-12b-it': '2',
    'gemma-4-26b-a4b-it': '1',
    'gemma-4-27b-it': '1',
    'gemma-4-9b-it': '1',
    'gemma-4-e4b-it': '1',
    'gemma-4-e2b-it': '1',
    'gemma-4-31b': '1',
    'gemma-4-27b': '1',
    'gemma-4-26b-a4b': '1',
    'gemma-4-12b': '1',
    'gemma-4-9b': '1',
    'gemma-4-e4b': '1',
    'gemma-4-e2b': '1',
    'diffusiongemma-26b-a4b-it': '1',
}


def normalize_model_name(name: str) -> str:
    """Normalize model identifier by stripping framework prefixes and whitespace."""
    val = name.strip().lower()
    for prefix in ('openai/', 'google/', 'hosted_vllm/', 'custom/'):
        if val.startswith(prefix):
            val = val[len(prefix):]
    return val


def _safe_load_yaml(path: Path, root_dir: Path) -> Any:
    """Load a YAML file using adk_submission's sandboxed path validator and loader."""
    try:
        resolved = validate_sandboxed_path(
            path=path,
            base_dir=root_dir.resolve(),
            must_exist=True,
            allow_symlinks=False,
        )
        return load_yaml(resolved, root_dir.resolve())
    except PathTraversalError as err:
        raise ValueError(f'Path traversal detected: {err}') from err


def _find_root_config(root: Path) -> Path:
    """Locate the root agent configuration file in the submission directory."""
    try:
        return find_root_config(root)
    except SubmissionValidationError as err:
        raise ParticipantVisibleError(str(err)) from err


def discover_declared_models(agent_dir: str | Path) -> set[str]:
    """Discover all model identifiers explicitly declared across an agent submission.

    Delegates YAML graph traversal and validation to :func:`adk_submission.discovery.discover_declared_models`.
    """
    try:
        return _adk_discover_declared_models(
            agent_dir, normalize_fn=normalize_model_name
        )
    except PathTraversalError as err:
        msg = str(err)
        if 'Path traversal detected' not in msg:
            msg = f'Path traversal detected: {msg}'
        raise ValueError(msg) from err
    except SubmissionValidationError as err:
        raise ParticipantVisibleError(str(err)) from err


def validate_single_declared_model(agent_dir: str | Path) -> str:
    """Validate that an agent submission declares exactly one unambiguous model.

    Args:
        agent_dir: Path to the submission directory.

    Returns:
        The single normalized model name (e.g. 'gemma-4-31b-it').

    Raises:
        ParticipantVisibleError: If no model or multiple distinct models are declared.
    """
    models = discover_declared_models(agent_dir)

    if not models:
        raise ParticipantVisibleError(
            "No model declared in agent configuration. "
            "Expected exactly one model to be declared (e.g. 'gemma-4-31b-it')."
        )

    if len(models) > 1:
        sorted_models = sorted(models)
        raise ParticipantVisibleError(
            f"Ambiguous model declaration: multiple distinct models declared across "
            f"agent configuration ({sorted_models}). "
            "This competition requires all agents in a submission to share exactly one model."
        )

    return next(iter(models))


def resolve_local_model_path(
    model_name: str,
    kaggle_input_root: Path | str = '/kaggle/input',
) -> str:
    """Resolve a model name to its local Kaggle mount path if available.

    Checks standard Kaggle input directory conventions for Gemma 4 models.
    Falls back to a canonical repository identifier if running outside Kaggle.

    Args:
        model_name: Normalized or raw model name (e.g. 'gemma-4-31b-it').
        kaggle_input_root: Root directory for Kaggle inputs (default: '/kaggle/input').

    Returns:
        Absolute filesystem path if mounted, otherwise 'google/{model_name}'.
    """
    clean_name = normalize_model_name(model_name)
    version = _DEFAULT_KAGGLE_MODEL_VERSIONS.get(clean_name, '1')
    input_root = Path(kaggle_input_root)

    candidates = [
        input_root / f'models/google/gemma-4/transformers/{clean_name}/{version}',
        input_root / f'models/google/gemma-4/Transformers/{clean_name}/{version}',
        input_root / clean_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    base_gemma_dir = input_root / 'models' / 'google' / 'gemma-4'
    if base_gemma_dir.exists():
        matched_dirs = [
            p for p in base_gemma_dir.glob(f'*/{clean_name}/*')
            if p.is_dir()
        ]
        if matched_dirs:
            def _ver_sort_key(p: Path) -> tuple[int, int | str]:
                return (1, int(p.name)) if p.name.isdigit() else (0, p.name)

            best_dir = max(matched_dirs, key=_ver_sort_key)
            return str(best_dir)

    if '/' in model_name:
        return model_name
    return f'google/{clean_name}'
