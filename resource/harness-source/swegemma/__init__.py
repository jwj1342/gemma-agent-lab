"""SWE-bench evaluation system for coding agents."""

import logging
import os

from swegemma.budget import EvaluationBudget, HarnessLimits, TokenBudget
from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.deduplication import (
    DeduplicationSummary,
    apply_patch_to_dir,
    clean_circular_symlinks,
    clean_repo_name,
    create_binary_patch,
    create_patch_from_archives,
    deduplicate_snapshots,
    get_base_snapshot_filename,
    get_task_cl,
    identify_base_snapshot,
    identify_base_snapshots,
    purge_tar_gz_archives,
    reconstruct_snapshot,
    resolve_task_snapshot_paths,
    validate_archive_extensions,
    validate_patch_reconstruction,
    verify_byte_identical_trees,
)
from swegemma.difficulty import (
    DifficultyWeights,
    TaskDifficultyMetrics,
    TaskDifficultyResult,
    analyze_task,
    analyze_tasks_file,
    compute_dimension_scores,
    extract_task_metrics,
    generate_difficulty_summary,
    render_summary_markdown,
)
from swegemma.edit import EditResult, apply_replacement
from swegemma.evaluate import Evaluator
from swegemma.harness.sample_verification import (
    ALL_BENCHMARK_REPOS,
    LATENCY_SLA_THRESHOLD_SECONDS,
    SampleVerificationResult,
    SweepSummary,
    load_all_benchmark_tasks,
    resolve_task_snapshot,
    run_sample_verification_sweep,
    select_sample_tasks,
    verify_sample_task_airgapped,
)
from swegemma.models import EvaluationResult, Task, TaskResult, load_tasks
from swegemma.results import append_task_result, save_results
from swegemma.sandbox import (
    BaseSandboxManager,
    ContainerConfig,
    ContainerManager,
    ExecResult,
    SubprocessManager,
)

__version__ = '0.2.7'

# Suppress OpenTelemetry detach errors and verbose third-party client logs
os.environ.setdefault("LITELLM_LOG", "WARNING")
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
for _noisy_logger in (
    "LiteLLM",
    "LiteLLM Router",
    "LiteLLM Proxy",
    "litellm",
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
):
    _lg = logging.getLogger(_noisy_logger)
    _lg.setLevel(logging.WARNING)
    if "litellm" in _noisy_logger.lower():
        _lg.propagate = False

try:
    import litellm

    litellm.drop_params = True
except ImportError:
    pass

__all__ = [
    'ALL_BENCHMARK_REPOS',
    'LATENCY_SLA_THRESHOLD_SECONDS',
    'BaseSandboxManager',
    'ContainerConfig',
    'ContainerManager',
    'DeduplicationSummary',
    'DifficultyWeights',
    'EditResult',
    'EvalConfig',
    'EvaluationBudget',
    'EvaluationResult',
    'Evaluator',
    'ExecResult',
    'HarnessLimits',
    'SampleVerificationResult',
    'SubprocessManager',
    'SweepSummary',
    'SwegemmaContext',
    'Task',
    'TaskDifficultyMetrics',
    'TaskDifficultyResult',
    'TaskResult',
    'TokenBudget',
    '__version__',
    'analyze_task',
    'analyze_tasks_file',
    'append_task_result',
    'apply_patch_to_dir',
    'apply_replacement',
    'clean_circular_symlinks',
    'clean_repo_name',
    'compute_dimension_scores',
    'create_binary_patch',
    'create_patch_from_archives',
    'deduplicate_snapshots',
    'extract_task_metrics',
    'generate_difficulty_summary',
    'get_base_snapshot_filename',
    'get_task_cl',
    'identify_base_snapshot',
    'identify_base_snapshots',
    'load_all_benchmark_tasks',
    'load_tasks',
    'purge_tar_gz_archives',
    'reconstruct_snapshot',
    'render_summary_markdown',
    'resolve_task_snapshot',
    'resolve_task_snapshot_paths',
    'run_sample_verification_sweep',
    'save_results',
    'select_sample_tasks',
    'validate_archive_extensions',
    'validate_patch_reconstruction',
    'verify_byte_identical_trees',
    'verify_sample_task_airgapped',
]

