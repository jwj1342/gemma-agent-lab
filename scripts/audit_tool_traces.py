"""Summarize recorded tool feedback without changing an evaluation's frozen runtime."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex


GRAPH_TOOLS = {'get_code_neighbors', 'get_code_subgraph', 'search_similar_code'}
SKILL_TOOLS = {'load_skill', 'load_skill_resource', 'run_skill_script', 'list_skills'}


def audit_trace(trace):
    calls, targets, errors = Counter(), Counter(), Counter()
    missing = nonzero_commands = unknown_observations = streak = longest_streak = 0
    previous_error = None
    for step in trace.get('steps', []):
        recorded_calls = step.get('tool_calls') or []
        if not recorded_calls:
            continue
        for call in recorded_calls:
            name = call.get('function_name') or call.get('function', {}).get('name') or call.get('name') or 'unknown'
            calls[name] += 1
            arguments = call.get('arguments', call.get('function', {}).get('arguments', {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {}
            if name in {'edit_file', 'write_file'} and isinstance(arguments, dict):
                path = arguments.get('filepath')
                if isinstance(path, str):
                    targets[path] += 1
        observation = (step.get('observation') or {}).get('content')
        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except ValueError:
                pass
        if not isinstance(observation, dict):
            unknown_observations += 1
            previous_error, streak = None, 0
            continue
        error = observation.get('error')
        if error or observation.get('status') == 'error':
            signature = str(error or observation.get('error_message') or observation.get('message')
                            or 'Unspecified tool error')
            errors[signature] += 1
            missing += 'mandatory input parameters are not present' in signature
            streak = streak + 1 if signature == previous_error else 1
            longest_streak = max(longest_streak, streak)
            previous_error = signature
        else:
            previous_error, streak = None, 0
        returncode = observation.get('returncode', observation.get('exit_code'))
        if returncode is None and isinstance(observation.get('details'), dict):
            returncode = observation['details'].get('exit_code')
        if any((c.get('function_name') or c.get('function', {}).get('name') or c.get('name')) == 'run_command'
               for c in recorded_calls) and isinstance(returncode, int) and returncode != 0:
            nonzero_commands += 1
    return {'tool_calls': dict(calls), 'recorded_tool_calls': sum(calls.values()),
        'graph_calls': sum(calls[n] for n in GRAPH_TOOLS),
        'skill_calls': sum(calls[n] for n in SKILL_TOOLS),
        'submission_calls': calls['submit_patch'], 'edit_targets': dict(targets),
        'tool_error_events': sum(errors.values()), 'missing_argument_events': missing,
        'longest_repeated_error_streak': longest_streak,
        'nonzero_command_results': nonzero_commands, 'unparsed_observations': unknown_observations,
        'error_frequencies': dict(errors)}


def patch_files(patch):
    files = []
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            parts = shlex.split(line)
            if len(parts) == 4:
                files.append(parts[3].removeprefix('b/'))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'run-manifest.json').read_text())
    summary = json.loads((args.run_dir / 'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Finish the frozen evaluation before auditing its complete cohort')
    rows = []
    for task in summary['tasks']:
        if not task['controls_valid']:
            rows.append({'task_id': task['task_id'], 'status': 'invalid_controls'})
            continue
        task_dir = args.run_dir / 'infer' / task['task_id']
        prediction = json.loads((task_dir / 'prediction.json').read_text())
        path = task_dir / 'results/traces' / ('trace_' + task['task_id'] + '.json')
        row = {'task_id': task['task_id'], 'resolved': task['resolved'],
               'completion_clean': task['completion_clean'], 'agent_seconds': task.get('agent_seconds'),
               'budgeted_tool_calls': prediction.get('tool_calls'),
               'patch_files': patch_files(prediction.get('agent_patch', ''))}
        if path.exists():
            content = path.read_bytes()
            row.update(audit_trace(json.loads(content)), trace_sha256=hashlib.sha256(content).hexdigest())
        else:
            row['trace_missing'] = True
        rows.append(row)
    result = {'agent_sha256': manifest['agent_archive_sha256'],
        'public_task_sha256': manifest['public_task_sha256'], 'tasks': rows,
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'summary_sha256': hashlib.sha256((args.run_dir / 'summary.json').read_bytes()).hexdigest(),
        'scope': 'Post-hoc trace audit. Tool errors are observed feedback events, not a complete schema validator. '
                 'A nonzero command can be an intended reproduction. Missing traces are not clean runs. '
                 'Workspace edit attempts and final protected-file-filtered patches can differ.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'tasks': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
