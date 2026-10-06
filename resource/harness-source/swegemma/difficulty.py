"""Task difficulty estimation engine for SWE-bench benchmark tasks.

Provides multi-dimensional feature extraction, calibrated non-linear dimension scoring,
composite weighting, discrete tier classification, and aggregate reporting.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regular expression patterns for diff and prompt analysis
DIFF_FILE_PATTERN: re.Pattern[str] = re.compile(r'^diff --git "?a/(.*?)"? "?b/(.*?)"?$', re.MULTILINE)
DIFF_HUNK_PATTERN: re.Pattern[str] = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', re.MULTILINE)
MOCK_PATTERN: re.Pattern[str] = re.compile(
    r'\b(mock|patch|MagicMock|Fake|fake_gfile|unittest\.mock|mocker|monkeypatch)\b',
    re.IGNORECASE,
)
TRACEBACK_OR_CODE_PATTERN: re.Pattern[str] = re.compile(
    r'(Traceback \(most recent call last\):|File ".*?", line \d+|```|`[a-zA-Z0-9_\.]+\(\)`|\bAssertionError\b|\bTypeError\b|\bValueError\b|\bKeyError\b)',
    re.MULTILINE,
)


@dataclass(frozen=True)
class DifficultyWeights:
    """Configurable weights for composite difficulty index calculation."""

    w_dispersion: float = 0.30
    w_churn: float = 0.25
    w_spec: float = 0.25
    w_test: float = 0.20

    def total(self) -> float:
        """Returns the sum of all weights for normalization."""
        return self.w_dispersion + self.w_churn + self.w_spec + self.w_test


@dataclass(frozen=True)
class TaskDifficultyMetrics:
    """Raw structural and textual metrics extracted from a benchmark task."""

    instance_id: str
    repo: str
    split: str
    is_empty_patch: bool
    num_files: int
    num_dirs: int
    lines_added: int
    lines_deleted: int
    total_churn: int
    num_hunks: int
    prompt_words: int
    has_lexical_overlap: bool
    has_traceback_or_code: bool
    test_files: int
    test_lines_added: int
    test_mock_count: int


@dataclass(frozen=True)
class TaskDifficultyResult:
    """Computed difficulty dimensions, composite score, and assigned difficulty tier."""

    metrics: TaskDifficultyMetrics
    score_dispersion: float
    score_churn: float
    score_spec: float
    score_test: float
    composite_score: float
    difficulty_tier: str

    def to_dict(self) -> dict[str, Any]:
        """Flattens metrics and scores into a single serializable dictionary."""
        data = asdict(self.metrics)
        data['score_dispersion'] = self.score_dispersion
        data['score_churn'] = self.score_churn
        data['score_spec'] = self.score_spec
        data['score_test'] = self.score_test
        data['composite_score'] = self.composite_score
        data['difficulty_tier'] = self.difficulty_tier
        return data


def extract_task_metrics(task: Mapping[str, Any], split: str = 'untracked') -> TaskDifficultyMetrics:
    """Extracts raw structural and textual metrics from a task dictionary.

    Args:
        task: Dictionary containing standard SWE-bench task keys:
            'instance_id', 'repo', 'patch', 'test_patch', 'problem_statement'.
        split: Benchmark split assignment ('public_test', 'private_test', 'broken', 'untracked').

    Returns:
        A populated TaskDifficultyMetrics dataclass.
    """
    from swegemma.models.task import extract_test_files_from_patch

    instance_id = str(task.get('instance_id', ''))
    repo = str(task.get('repo', ''))
    patch = str(task.get('patch', '') or '')
    test_patch = str(task.get('test_patch', '') or '')
    problem_statement = str(task.get('problem_statement', '') or '')

    # Developer patch parsing
    matched_patch_files = [m[1] or m[0] for m in DIFF_FILE_PATTERN.findall(patch)]
    if not matched_patch_files and patch.strip():
        matched_patch_files = extract_test_files_from_patch(patch)
    num_files = len(matched_patch_files)
    is_empty_patch = num_files == 0 and not patch.strip()

    dirs = {'/'.join(p.split('/')[:-1]) for p in matched_patch_files if '/' in p}
    num_dirs = len(dirs)

    lines_added = sum(1 for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    lines_deleted = sum(1 for line in patch.splitlines() if line.startswith('-') and not line.startswith('---'))
    total_churn = lines_added + lines_deleted
    num_hunks = len(DIFF_HUNK_PATTERN.findall(patch))

    # Problem statement parsing
    words = len(problem_statement.split())
    has_traceback_or_code = bool(TRACEBACK_OR_CODE_PATTERN.search(problem_statement))

    # Lexical overlap detection: check if file stems or sub-tokens appear in prompt
    has_lexical_overlap = False
    desc_lower = problem_statement.lower()
    for file_path in matched_patch_files:
        bname = file_path.split('/')[-1].lower()
        stem = bname.rsplit('.', 1)[0]
        if len(stem) >= 4 and stem in desc_lower:
            has_lexical_overlap = True
            break
        # Also check components for snake_case tokens
        parts = stem.split('_')
        for part in parts:
            if len(part) >= 5 and part in desc_lower:
                has_lexical_overlap = True
                break
        if has_lexical_overlap:
            break

    # Test patch parsing
    matched_test_files = [m[1] or m[0] for m in DIFF_FILE_PATTERN.findall(test_patch)]
    if not matched_test_files and test_patch.strip():
        matched_test_files = extract_test_files_from_patch(test_patch)
    test_files = len(matched_test_files)
    test_lines_added = sum(
        1 for line in test_patch.splitlines() if line.startswith('+') and not line.startswith('+++')
    )
    test_mock_count = len(MOCK_PATTERN.findall(test_patch))

    return TaskDifficultyMetrics(
        instance_id=instance_id,
        repo=repo,
        split=split,
        is_empty_patch=is_empty_patch,
        num_files=num_files,
        num_dirs=num_dirs,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        total_churn=total_churn,
        num_hunks=num_hunks,
        prompt_words=words,
        has_lexical_overlap=has_lexical_overlap,
        has_traceback_or_code=has_traceback_or_code,
        test_files=test_files,
        test_lines_added=test_lines_added,
        test_mock_count=test_mock_count,
    )


def compute_dimension_scores(
    metrics: TaskDifficultyMetrics,
    weights: DifficultyWeights | None = None,
) -> TaskDifficultyResult:
    """Computes normalized 0-100 dimension scores and the composite tier.

    Args:
        metrics: Raw extracted task metrics.
        weights: Optional custom DifficultyWeights instance.

    Returns:
        A TaskDifficultyResult instance with all dimension and composite scores.
    """
    if weights is None:
        weights = DifficultyWeights()

    if metrics.is_empty_patch:
        return TaskDifficultyResult(
            metrics=metrics,
            score_dispersion=0.0,
            score_churn=0.0,
            score_spec=0.0,
            score_test=0.0,
            composite_score=0.0,
            difficulty_tier='Empty Patch',
        )

    # 1. Edit Scope & Dispersion (0 - 100)
    if metrics.num_files <= 1:
        base_dispersion = 20.0
    else:
        log_scale = math.log2(metrics.num_files) / math.log2(10.0)
        base_dispersion = min(100.0, log_scale * 80.0 + 20.0)

    dir_bonus = min(20.0, max(0.0, (metrics.num_dirs - 1) * 7.0))
    score_dispersion = min(100.0, max(0.0, base_dispersion * 0.8 + dir_bonus))

    # 2. Code Churn & Complexity (0 - 100)
    effective_churn = max(1, metrics.total_churn)
    base_churn = min(100.0, max(10.0, (math.log10(effective_churn) / 3.0) * 90.0 + 10.0))
    hunk_bonus = min(15.0, max(0.0, (metrics.num_hunks - 1) * 1.5))
    score_churn = min(100.0, max(0.0, base_churn * 0.85 + hunk_bonus))

    # 3. Problem Specification & Localization Difficulty (0 - 100)
    if metrics.prompt_words < 25:
        base_spec = 75.0
    elif metrics.prompt_words < 60:
        base_spec = 60.0
    elif metrics.prompt_words < 150:
        base_spec = 45.0
    else:
        base_spec = 35.0

    if metrics.has_lexical_overlap:
        base_spec -= 25.0
    if metrics.has_traceback_or_code:
        base_spec -= 10.0

    score_spec = min(100.0, max(10.0, base_spec))

    # 4. Test & Verification Surface (0 - 100)
    if metrics.test_files == 0:
        score_test = 10.0
    else:
        effective_test_adds = max(1, metrics.test_lines_added)
        base_test = min(80.0, max(15.0, (math.log10(effective_test_adds) / 2.5) * 70.0 + 15.0))
        mock_bonus = min(20.0, metrics.test_mock_count * 3.0)
        score_test = min(100.0, max(0.0, base_test + mock_bonus))

    # Composite score calculation
    total_w = weights.total()
    if total_w <= 0.0:
        composite = 0.0
    else:
        composite = (
            weights.w_dispersion * score_dispersion
            + weights.w_churn * score_churn
            + weights.w_spec * score_spec
            + weights.w_test * score_test
        ) / total_w

    composite = min(100.0, max(0.0, composite))

    # Tier assignment
    if composite < 35.0:
        tier = 'Easy'
    elif composite < 55.0:
        tier = 'Medium'
    elif composite < 72.0:
        tier = 'Hard'
    else:
        tier = 'Extreme'

    return TaskDifficultyResult(
        metrics=metrics,
        score_dispersion=round(score_dispersion, 1),
        score_churn=round(score_churn, 1),
        score_spec=round(score_spec, 1),
        score_test=round(score_test, 1),
        composite_score=round(composite, 1),
        difficulty_tier=tier,
    )


def analyze_task(
    task: Mapping[str, Any],
    split: str = 'untracked',
    weights: DifficultyWeights | None = None,
) -> TaskDifficultyResult:
    """Extracts metrics and computes difficulty for a single task."""
    metrics = extract_task_metrics(task, split=split)
    return compute_dimension_scores(metrics, weights=weights)


def load_split_map(split_csv_path: Path | str | None) -> dict[str, str]:
    """Loads split mappings from CSV mapping 'repo_hash' or 'instance_id' to 'split'."""
    if not split_csv_path:
        return {}

    path = Path(split_csv_path)
    if not path.exists():
        logger.warning('Split file not found: %s', path)
        return {}

    split_map: dict[str, str] = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row.get('repo_hash') or row.get('instance_id')
            split = row.get('split')
            if iid and split:
                split_map[iid.strip()] = split.strip()

    return split_map


def analyze_tasks_file(
    tasks_path: Path | str,
    split_map_path: Path | str | None = None,
    weights: DifficultyWeights | None = None,
) -> list[TaskDifficultyResult]:
    """Analyzes all tasks in a JSON Lines file and assigns difficulty metrics.

    Args:
        tasks_path: Path to the .jsonl file containing task objects.
        split_map_path: Optional path to the CSV file mapping tasks to splits.
        weights: Optional custom DifficultyWeights instance.

    Returns:
        A list of TaskDifficultyResult instances for all tasks in the file.
    """
    path = Path(tasks_path)
    if not path.exists():
        raise FileNotFoundError(f'Tasks file not found: {path}')

    split_map = load_split_map(split_map_path) if split_map_path else {}
    results: list[TaskDifficultyResult] = []

    with path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                task = json.loads(line_str)
            except json.JSONDecodeError as err:
                logger.warning('Skipping invalid JSON on line %d: %s', line_num, err)
                continue

            iid = str(task.get('instance_id', ''))
            split = split_map.get(iid, 'untracked')
            result = analyze_task(task, split=split, weights=weights)
            results.append(result)

    return results


def generate_difficulty_summary(results: Sequence[TaskDifficultyResult]) -> dict[str, Any]:
    """Computes aggregate distribution statistics across repositories, splits, and tiers."""
    total_count = len(results)
    empty_patch_count = sum(1 for r in results if r.metrics.is_empty_patch)
    valid_results = [r for r in results if not r.metrics.is_empty_patch]
    valid_count = len(valid_results)

    tier_counts = Counter(r.difficulty_tier for r in results)
    split_counts = Counter(r.metrics.split for r in results)
    repo_counts = Counter(r.metrics.repo for r in results)

    # Split by tier matrix
    split_tier_matrix: dict[str, dict[str, int]] = {}
    for r in results:
        s = r.metrics.split
        t = r.difficulty_tier
        if s not in split_tier_matrix:
            split_tier_matrix[s] = Counter()
        split_tier_matrix[s][t] += 1

    # Score stats on valid tasks
    if valid_results:
        scores = [r.composite_score for r in valid_results]
        sorted_scores = sorted(scores)
        mean_score = sum(scores) / len(scores)
        median_score = sorted_scores[len(sorted_scores) // 2]
        p25 = sorted_scores[int(len(sorted_scores) * 0.25)]
        p75 = sorted_scores[int(len(sorted_scores) * 0.75)]
        min_score = sorted_scores[0]
        max_score = sorted_scores[-1]
    else:
        mean_score = median_score = p25 = p75 = min_score = max_score = 0.0

    return {
        'total_count': total_count,
        'valid_count': valid_count,
        'empty_patch_count': empty_patch_count,
        'tier_counts': dict(tier_counts),
        'split_counts': dict(split_counts),
        'repo_counts': dict(repo_counts),
        'split_tier_matrix': {k: dict(v) for k, v in split_tier_matrix.items()},
        'score_stats': {
            'mean': round(mean_score, 1),
            'median': round(median_score, 1),
            'p25': round(p25, 1),
            'p75': round(p75, 1),
            'min': round(min_score, 1),
            'max': round(max_score, 1),
        },
    }


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Renders a formatted Markdown report of difficulty metrics and split distributions."""
    total = summary['total_count']
    valid = summary['valid_count']
    empty = summary['empty_patch_count']
    stats = summary['score_stats']
    tier_counts = summary['tier_counts']
    matrix = summary['split_tier_matrix']
    repo_counts = summary['repo_counts']

    tiers_ordered = ['Easy', 'Medium', 'Hard', 'Extreme', 'Empty Patch']

    lines: list[str] = [
        '# Task Difficulty Analysis Summary Report',
        '',
        '## Executive Overview',
        f'- **Total Tasks Evaluated**: {total}',
        f'- **Valid Evaluated Tasks**: {valid}',
        f'- **Empty Patch Tasks**: {empty} (untracked candidate pool only)',
        f"- **Composite Score Range**: {stats['min']} to {stats['max']} (Mean: {stats['mean']}, Median: {stats['median']}, IQR: {stats['p25']}-{stats['p75']})",
        '',
        '## Overall Difficulty Tier Breakdown',
        '| Difficulty Tier | Task Count | Percentage of Total | Percentage of Valid |',
        '| :--- | :---: | :---: | :---: |',
    ]

    for tier in tiers_ordered:
        cnt = tier_counts.get(tier, 0)
        pct_total = (cnt / total * 100.0) if total > 0 else 0.0
        pct_valid = (cnt / valid * 100.0) if valid > 0 and tier != 'Empty Patch' else 0.0
        pct_valid_str = f'{pct_valid:.1f}%' if tier != 'Empty Patch' else 'N/A'
        lines.append(f'| **{tier}** | {cnt} | {pct_total:.1f}% | {pct_valid_str} |')

    lines.extend([
        '',
        '## Difficulty Tier Distribution by Benchmark Split',
        '| Split | Easy | Medium | Hard | Extreme | Empty Patch | Total |',
        '| :--- | :---: | :---: | :---: | :---: | :---: | :---: |',
    ])

    for split_name in sorted(matrix.keys()):
        row = matrix[split_name]
        split_total = sum(row.values())
        lines.append(
            f"| **`{split_name}`** | {row.get('Easy', 0)} | {row.get('Medium', 0)} | {row.get('Hard', 0)} | {row.get('Extreme', 0)} | {row.get('Empty Patch', 0)} | **{split_total}** |"
        )

    lines.extend([
        '',
        '## Repository Task Breakdown',
        '| Repository | Task Count | Share of Dataset |',
        '| :--- | :---: | :---: |',
    ])

    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        pct = (count / total * 100.0) if total > 0 else 0.0
        lines.append(f'| **`{repo}`** | {count} | {pct:.1f}% |')

    lines.append('')
    return '\n'.join(lines)
