"""Results aggregation, saving, and summary reporting."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from swegemma.models import EvaluationResult, TaskResult

logger = logging.getLogger(__name__)


def _serialize_task_result(result: TaskResult) -> dict[str, Any]:
    """Serialize a TaskResult to a dictionary for jsonl export."""
    data: dict[str, Any] = {
        'instance_id': result.instance_id,
        'repo': result.repo,
        'resolved': result.resolved,
        'agent_patch_size': len(result.agent_patch),
        'test_exit_code': result.test_exit_code,
        'duration_seconds': round(result.duration_seconds, 2),
        'error': result.error,
        'tool_calls': (
            getattr(result, 'tool_calls', 0)
            or getattr(result, 'tool_calls_used', 0)
            or 0
        ),
        'total_llm_calls': getattr(result, 'total_llm_calls', 0) or 0,
    }
    if getattr(result, 'task_index', None) is not None:
        data['task_index'] = result.task_index
    return data


def append_task_result(
    result: TaskResult,
    output_dir: Path,
    running_results: list[TaskResult] | None = None,
) -> None:
    """Incrementally append a single task result and update summary.json atomically.

    Creates / updates:
      output_dir/task_results.jsonl (appended)
      output_dir/summary.json (atomically updated with running summary)
      output_dir/patches/<instance_id>.patch
      output_dir/test_outputs/<instance_id>.log
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / 'task_results.jsonl'

    data = _serialize_task_result(result)

    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

    if result.agent_patch:
        patches_dir = output_dir / 'patches'
        patches_dir.mkdir(exist_ok=True)
        patch_file = patches_dir / f'{result.instance_id.replace("/", "__")}.patch'
        patch_file.write_text(result.agent_patch, encoding='utf-8')

    if result.test_output:
        test_outputs_dir = output_dir / 'test_outputs'
        test_outputs_dir.mkdir(exist_ok=True)
        test_output_file = (
            test_outputs_dir / f'{result.instance_id.replace("/", "__")}.log'
        )
        test_output_file.write_text(result.test_output, encoding='utf-8')

    if running_results:
        eval_res = EvaluationResult(task_results=running_results)
        summary_path = output_dir / 'summary.json'
        tmp_summary = (
            output_dir / f'summary.json.tmp.{os.getpid()}_{uuid.uuid4().hex[:8]}'
        )
        summary_data = {
            'total_tasks': eval_res.total,
            'resolved': eval_res.resolved,
            'resolution_rate': round(eval_res.resolution_rate, 4),
            'by_repo': eval_res.by_repo(),
            'errors': sum(1 for r in running_results if r.error is not None),
        }
        tmp_summary.write_text(json.dumps(summary_data, indent=2), encoding='utf-8')
        tmp_summary.replace(summary_path)


def save_results(evaluation_result: EvaluationResult, output_dir: Path) -> None:
    """Save evaluation results to disk atomically.

    Creates:
      output_dir/task_results.jsonl
      output_dir/summary.json
      output_dir/patches/<instance_id>.patch
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / 'patches'
    patches_dir.mkdir(exist_ok=True)

    jsonl_path = output_dir / 'task_results.jsonl'
    summary_path = output_dir / 'summary.json'
    tmp_suffix = f'{os.getpid()}_{uuid.uuid4().hex[:8]}'
    tmp_jsonl = output_dir / f'task_results.jsonl.tmp.{tmp_suffix}'
    tmp_summary = output_dir / f'summary.json.tmp.{tmp_suffix}'

    # Write per-task JSONL and individual patch files
    with open(tmp_jsonl, 'w', encoding='utf-8') as f:
        for result in evaluation_result.task_results:
            data = _serialize_task_result(result)
            f.write(json.dumps(data) + '\n')

            if result.agent_patch:
                patch_file = (
                    patches_dir / f'{result.instance_id.replace("/", "__")}.patch'
                )
                patch_file.write_text(result.agent_patch, encoding='utf-8')

            if result.test_output:
                test_outputs_dir = output_dir / 'test_outputs'
                test_outputs_dir.mkdir(exist_ok=True)
                test_output_file = (
                    test_outputs_dir / f'{result.instance_id.replace("/", "__")}.log'
                )
                test_output_file.write_text(result.test_output, encoding='utf-8')

    tmp_jsonl.replace(jsonl_path)

    summary_data = {
        'total_tasks': evaluation_result.total,
        'resolved': evaluation_result.resolved,
        'resolution_rate': round(evaluation_result.resolution_rate, 4),
        'by_repo': evaluation_result.by_repo(),
        'errors': sum(1 for r in evaluation_result.task_results if r.error is not None),
    }

    tmp_summary.write_text(json.dumps(summary_data, indent=2), encoding='utf-8')
    tmp_summary.replace(summary_path)
    logger.info('Saved evaluation results to %s', output_dir)
