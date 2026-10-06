"""Configuration models for SWE-bench evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from adk_submission import (
    AdapterManifest,
    GenerationConstraints,
    ModelRegistry,
    NumericRange,
    SubmissionLimits,
)

from swegemma.budget import EvaluationBudget, HarnessLimits

MAX_SUBMISSION_SIZE_BYTES: int = 3 * 1024 * 1024 * 1024  # 3 GiB (3,221,225,472 bytes)
ALLOWED_SUBMISSION_EXTENSIONS: frozenset[str] = frozenset(
    {
        '.yaml',
        '.yml',
        '.md',
        '.txt',
        '.py',
        '.json',
        '.safetensors',
    }
)
ALLOWED_ADAPTER_EXTENSIONS: frozenset[str] = frozenset({'.safetensors'})


def build_submission_limits(
    max_loop_iterations: int = 500,
) -> tuple[SubmissionLimits, GenerationConstraints]:
    """Assemble default competition submission limits and generation constraints.

    Enforces a 3 GiB total unpacked submission size cap (including adapters/),
    restricts file types to strictly required config/prompt/skill/LoRA formats
    (.yaml, .yml, .md, .txt, .py, .json, .safetensors), and bounds token generation
    parameters to the 4x L4 vLLM context length (32,768 tokens).
    """
    limits = SubmissionLimits(
        max_total_size_bytes=MAX_SUBMISSION_SIZE_BYTES,
        max_yaml_size_bytes=50 * 1024 * 1024,
        max_skill_size_bytes=50 * 1024 * 1024,
        max_file_count=10_000,
        max_yaml_files=1_000,
        max_instruction_chars=1_000_000,
        max_total_instruction_chars=10_000_000,
        max_agents=500,
        max_sub_agent_depth=50,
        max_skills=1_000,
        max_loop_iterations=max_loop_iterations,
        allowed_file_extensions=ALLOWED_SUBMISSION_EXTENSIONS,
        adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS,
    )
    gen_constraints = GenerationConstraints(
        allowed_fields=None,
        max_output_tokens=NumericRange(1, 32768),
        thinking_budget=NumericRange(0, 32768),
        defaults={
            'max_output_tokens': 16384,
            'thinking_config': {'thinking_budget': 4096},
        },
    )
    return limits, gen_constraints

if TYPE_CHECKING:
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
else:
    try:
        from google.adk.agents.context_cache_config import ContextCacheConfig
    except ImportError:
        ContextCacheConfig = Any

    try:
        from google.adk.apps._configs import EventsCompactionConfig
    except ImportError:
        try:
            from google.adk.apps.app import EventsCompactionConfig
        except ImportError:
            EventsCompactionConfig = Any


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    tasks_path: Path
    snapshots_dir: Path
    results_dir: Path
    submission_dir: Path
    models: ModelRegistry
    image: str = 'swebench-sandbox:latest'
    sandbox: str = 'docker'  # 'docker' (default) or 'subprocess' (Kaggle/notebooks)
    wheels_dir: Path | None = None
    graph_dir: str = 'data/graphs'
    embeddings_dir: str = 'data/embeddings'
    budget: EvaluationBudget = field(default_factory=EvaluationBudget)
    harness: HarnessLimits = field(default_factory=HarnessLimits)
    # Direct convenience kwargs for flat configuration
    timeout_seconds: int | None = None
    max_time_minutes: int | float | None = None
    max_tool_calls: int | None = None
    max_turns: int | None = None
    max_llm_calls: int | None = None
    task_ids: list[str] | None = None
    verbose: bool = False
    skip_agent_patch: bool = False
    limits: SubmissionLimits | None = None
    generation_constraints: GenerationConstraints | None = None
    adapter_manifest: AdapterManifest | None = None
    context_cache_config: ContextCacheConfig | None = None
    events_compaction_config: EventsCompactionConfig | None = None
    concurrency: int = 1
    shard_index: int | None = None
    num_shards: int | None = None
    display_mode: Literal['auto', 'dashboard', 'single', 'quiet'] | str = 'auto'
    enable_sandbox_testing: bool = True

    def __post_init__(self) -> None:
        """Merge flat kwargs into budget and harness if explicitly provided."""
        valid_modes = {'auto', 'dashboard', 'single', 'quiet'}
        if self.display_mode not in valid_modes:
            raise ValueError(
                f"Invalid display_mode {self.display_mode!r}, must be one of {valid_modes}"
            )

        if self.num_shards is not None and self.num_shards < 1:
            raise ValueError(f'num_shards must be >= 1, got {self.num_shards}')
        if self.shard_index is not None:
            if self.num_shards is None or self.num_shards < 1:
                raise ValueError('shard_index requires num_shards >= 1')
            if self.shard_index < 0 or self.shard_index >= self.num_shards:
                raise ValueError(
                    f'shard_index ({self.shard_index}) must be in range [0, {self.num_shards})'
                )

        default_limits, default_gen_constraints = build_submission_limits()
        if self.limits is None:
            self.limits = default_limits
        if self.generation_constraints is None:
            self.generation_constraints = default_gen_constraints

        t_min = (
            self.max_time_minutes
            if self.max_time_minutes is not None
            else self.budget.time_minutes
        )
        tc = (
            self.max_tool_calls
            if self.max_tool_calls is not None
            else self.budget.tool_calls
        )
        turns = (
            self.max_turns
            if self.max_turns is not None
            else (
                self.max_llm_calls
                if self.max_llm_calls is not None
                else self.budget.turns
            )
        )
        object.__setattr__(
            self,
            'budget',
            EvaluationBudget(
                time_minutes=t_min,
                tool_calls=tc,
                turns=turns,
                cost_usd=self.budget.cost_usd,
                total_tokens=self.budget.total_tokens,
            ),
        )

        cmd_timeout = (
            self.timeout_seconds
            if self.timeout_seconds is not None
            else self.harness.command_timeout_seconds
        )
        object.__setattr__(
            self,
            'harness',
            HarnessLimits(
                command_timeout_seconds=cmd_timeout,
                max_stdout_chars=self.harness.max_stdout_chars,
                max_file_lines=self.harness.max_file_lines,
                max_file_chars=self.harness.max_file_chars,
            ),
        )

        if self.tasks_path is not None:
            tasks_parent = Path(self.tasks_path).parent
            if self.graph_dir == 'data/graphs' and not Path(self.graph_dir).exists():
                cand_graphs = tasks_parent / 'graphs'
                if cand_graphs.is_dir():
                    object.__setattr__(self, 'graph_dir', str(cand_graphs))
            if (
                self.embeddings_dir == 'data/embeddings'
                and not Path(self.embeddings_dir).exists()
            ):
                cand_emb = tasks_parent / 'embeddings'
                if cand_emb.is_dir():
                    object.__setattr__(self, 'embeddings_dir', str(cand_emb))

