"""Build SFT training records from verified agent traces.

A trace alone is not training data. A trajectory is kept only when three things hold together for
that task: the reference (gold) patch passed, the agent actually resolved the issue, and the trace
contains at least one tool call. Everything else is counted and reported, never silently trained on.

Inputs are the layout this repository really writes: one results directory per run,
`<results>/traces/trace_<instance_id>.json` for traces (ATIF v1.7, as produced by
`SessionTrace.save`; the legacy entry-list format is also accepted), and a per-task
`result.json` with `task_id`, `resolved`, and `test_exit_code`. Pass one or more task directories
with `--task-dir`, each holding the `result.json` files whose controls were verified.

    python scripts/build_trajectory_dataset.py \
        --results-dir runs/<run>/infer/results \
        --task-dir runs/<run>/infer/results \
        --output data/processed/trajectories.jsonl

Not executed against a real trace yet: no trace artifact exists in this checkout, so the ATIF
handling is derived from `resource/harness-source/adk_eval_core/tracing/trace.py` and remains
unverified end to end. Inspect the first real trace before training on its output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    # utf-8-sig: PowerShell redirection and some editors write a BOM, which plain utf-8 rejects.
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise SystemExit(f'{path} is not a JSON object')
    return data


def passed(result: dict) -> bool:
    """A control passes when the patch applied, the tests ran, and they exited clean."""
    return (result.get('resolved') is True
            and result.get('test_exit_code') == 0
            and not result.get('error_message'))


def collect_results(task_dirs: list[Path]) -> dict[str, dict]:
    """Map task_id to its result record, keeping the strongest evidence per task."""
    found: dict[str, dict] = {}
    for directory in task_dirs:
        for path in sorted(directory.glob('**/result.json')):
            try:
                result = load_json(path)
            except (SystemExit, json.JSONDecodeError):
                continue
            task_id = result.get('task_id')
            if not task_id:
                continue
            record = found.setdefault(task_id, {'passes': 0, 'fails': 0, 'paths': []})
            record['passes' if passed(result) else 'fails'] += 1
            record['paths'].append(str(path))
            record.setdefault('resolved', result.get('resolved'))
    return found


def extract_steps(raw: dict) -> list[dict]:
    """Return ATIF steps, or normalize legacy trace entries into step form."""
    if isinstance(raw.get('steps'), list):
        return raw['steps']
    entries = raw.get('entries')
    if not isinstance(entries, list):
        raise SystemExit('trace has neither ATIF steps nor legacy entries')
    steps: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kind = entry.get('type')
        step = {'source': entry.get('author', 'agent'), 'message': entry.get('content') or None}
        if kind == 'tool_call':
            step['tool_calls'] = [{
                'tool_call_id': str(len(steps)),
                'function_name': entry.get('tool', ''),
                'arguments': entry.get('args') or {},
            }]
        elif kind == 'tool_response':
            step['observation'] = {'content': entry.get('result', '')}
        elif kind in {'text', 'thinking', 'final'}:
            step['metrics'] = {'event_type': kind}
        else:
            continue
        steps.append(step)
    return steps


def steps_to_messages(steps: list[dict]) -> list[dict]:
    """Convert ATIF steps into chat messages with tool calls and tool results."""
    messages: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        source = step.get('source') or 'agent'
        role = {'user': 'user', 'system': 'user', 'agent': 'assistant'}.get(source, 'assistant')
        message = step.get('message')
        if isinstance(message, list):
            message = '\n'.join(
                part.get('text', '') for part in message if isinstance(part, dict)
            ).strip() or None
        calls = step.get('tool_calls') or []
        if calls:
            messages.append({
                'role': role,
                'content': message or '',
                'tool_calls': [{
                    'id': str(call.get('tool_call_id') or index),
                    'type': 'function',
                    'function': {
                        'name': call.get('function_name', ''),
                        'arguments': json.dumps(call.get('arguments') or {}, sort_keys=True),
                    },
                } for index, call in enumerate(calls)],
            })
        elif message:
            messages.append({'role': role, 'content': message})
        observation = step.get('observation')
        if isinstance(observation, dict):
            content = observation.get('content', '')
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            messages.append({
                'role': 'tool',
                'content': content,
                'tool_call_id': str(observation.get('tool_call_id') or ''),
            })
    return messages


def build(results_dir: Path, task_dirs: list[Path], output: Path, max_records: int | None) -> dict:
    trace_files = sorted((results_dir / 'traces').glob('trace_*.json'))
    task_results = collect_results(task_dirs)
    if not task_results:
        raise SystemExit(f'no result.json found under {[str(d) for d in task_dirs]}; '
                         'controls must be verified before any trace is trusted')
    counts = {'traces_found': len(trace_files), 'tasks_with_results': len(task_results),
              'kept': 0, 'skipped_task_failed': 0, 'skipped_no_tool_call': 0,
              'skipped_unreadable': 0, 'skipped_no_result': 0}
    records: list[dict] = []
    for path in trace_files:
        instance_id = path.stem.removeprefix('trace_').replace('__', '/')
        evidence = task_results.get(instance_id)
        if evidence is None:
            counts['skipped_no_result'] += 1
            continue
        if evidence['passes'] == 0:
            counts['skipped_task_failed'] += 1
            continue
        try:
            steps = extract_steps(load_json(path))
        except (SystemExit, json.JSONDecodeError):
            counts['skipped_unreadable'] += 1
            continue
        messages = steps_to_messages(steps)
        if not any(message.get('tool_calls') for message in messages):
            counts['skipped_no_tool_call'] += 1
            continue
        records.append({'instance_id': instance_id, 'messages': messages,
                        'source': 'verified_trajectory',
                        'evidence': {'trace': str(path), 'results': evidence['paths']}})
    counts['kept'] = len(records)
    if max_records is not None:
        records = records[:max_records]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    return {
        'output': str(output),
        'records_written': len(records),
        'counts': counts,
        'schema': 'messages JSONL for TRL; one record per verified, tool-using trajectory',
        'note': 'Verified trajectories only. Do not add unresolved traces to raise the record count.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Directory holding traces/trace_<instance_id>.json')
    parser.add_argument('--task-dir', type=Path, action='append', required=True,
                        help='Directory searched for result.json evidence; repeatable')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-records', type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(build(args.results_dir.resolve(), [d.resolve() for d in args.task_dir],
                           args.output.resolve(), args.max_records), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
