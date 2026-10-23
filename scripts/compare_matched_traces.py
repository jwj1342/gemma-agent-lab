"""Select discordant r5 tasks before semantic reading; export private timelines.

No traced command is executed. Reader files contain exact quotes and must remain
private. Independent AI readings do not constitute teammate approval.
"""
import argparse
import json
from pathlib import Path

from src.failure_mechanisms import changed_old_lines, events, load_run, sha, views

ARMS = ('budgetfit', 'c2', 'c2-seed1', 'c2-recover', 'c3')
SELECTION = ('Lexicographic first N task IDs with valid controls and trace/prediction '
             'artifacts in all five r5 arms and discordant c2/c2-seed1 outcomes; '
             'selection uses outcomes, never trace content.')


def select_tasks(rows, count=3):
    if count < 1:
        raise ValueError('count must be positive')
    grouped = {}
    for row in rows:
        key = row['task_id'], row['run']
        if key[1] in grouped.setdefault(key[0], {}):
            raise ValueError('Duplicate task/arm')
        grouped[key[0]][key[1]] = row
    eligible = []
    for task, arms in sorted(grouped.items()):
        if not all(name in arms for name in ARMS):
            continue
        if not all(r['outcome'].get('controls_valid') is True
                   and type(r['outcome'].get('resolved')) is bool
                   and r['trace_present'] and r['prediction_present'] for r in arms.values()):
            continue
        if arms['c2']['outcome']['resolved'] != arms['c2-seed1']['outcome']['resolved']:
            eligible.append(task)
    return eligible[:count], eligible


def point(event):
    return {'call': event['call'], 'step_id': event['step_id']}


def checked_reading(reading, ev, accepted_call):
    """Check provenance, not semantic correctness or whether an event is earliest."""
    result = {}
    write = reading.get('source_write')
    if write is not None:
        matching = [e for e in ev if e['step_id'] == write['step_id']]
        if len(matching) != 1:
            raise ValueError('Source write must identify one event')
        event = matching[0]
        quote = write.get('quote')
        if (event['tool'] != 'run_command' or event['obs'].get('status') != 'ok'
                or event['obs'].get('exit_code') != 0 or not write.get('rationale')
                or not isinstance(quote, str) or not quote.strip()
                or quote not in event['args'].get('command', '')):
            raise ValueError('Invalid reader-confirmed command write evidence')
        result['source_write'] = {**write, **point(event)}
        accepted_call = min(accepted_call, event['call']) if accepted_call is not None else event['call']
    else:
        result['source_write'] = None
    for field in ('hypothesis', 'verification'):
        item = reading.get(field)
        if item is None:
            result[field] = None
            continue
        matching = [e for e in ev if e['step_id'] == item['step_id']]
        if len(matching) != 1 or not item.get('rationale', '').strip():
            raise ValueError('Reading must identify one tool event and explain its judgment')
        event = matching[0]
        quote = item.get('quote')
        source = event['thinking'] if field == 'hypothesis' else json.dumps(event['obs'], ensure_ascii=False)
        if not isinstance(quote, str) or not quote.strip() or quote not in source:
            raise ValueError('Reading quote absent from trace')
        if field == 'verification' and (accepted_call is None or event['call'] <= accepted_call
                                        or event['tool'] != 'run_command'):
            raise ValueError('Verification must follow an accepted editor or reader-confirmed source write')
        result[field] = {**item, **point(event)}
    result['uncertainty_notes'] = reading.get('hypothesis_unknown_reason', reading.get('uncertainty', reading.get('notes')))
    return result


def compare(release, tasks, output, readers=(), count=3):
    release, tasks, output = Path(release), Path(tasks), Path(output)
    if output.exists():
        raise ValueError('Preserve previous comparison output')
    reference_path = release / 'reference-patch-locations.json'
    reference = json.loads(reference_path.read_text())
    rows = [r for arm in ARMS for r in load_run(release, arm, reference)]
    selected, eligible = select_tasks(rows, count)
    patches = {}
    for line in tasks.read_text().splitlines():
        if line.strip():
            task = json.loads(line)
            if task['instance_id'] in patches:
                raise ValueError('Duplicate task')
            patches[task['instance_id']] = task['patch']
    from src.failure_mechanisms import reference_hunks
    for task in selected:
        if reference_hunks(patches[task]) != reference[task]:
            raise ValueError('Reference map mismatch')
    reviews = [json.loads(Path(p).read_text()) for p in readers]
    identities = [r['reader'] for r in reviews]
    if len(set(identities)) != len(identities):
        raise ValueError('Distinct reader identities required')
    comparisons = []
    for row in rows:
        task, arm = row['task_id'], row['run']
        if task not in selected:
            continue
        path = release / arm / 'tasks' / task / 'trace.json'
        trace = json.loads(path.read_text())
        ev = events(trace)
        edits = [e for e in ev if e['tool'] in ('edit_file', 'write_file')]
        accepted = next((e for e in edits if e['obs'].get('status') == 'ok'), None)
        # Positions after any possible mutation are not mapped back to base source.
        boundary = min([e['call'] for e in edits] +
                       [e['call'] for e in ev if e['tool'] == 'run_command' and any(
                           flag['call'] == e['call'] for flag in row['possible_shell_mutations'])] + [float('inf')])
        changed = changed_old_lines(patches[task])
        exposure = next((e for e in ev if e['call'] < boundary and any(
            any(start <= n <= end for n in changed.get(p, ()))
            for p, start, end, _, _ in views(e))), None)
        readings = {}
        for review in reviews:
            reading = review['readings'][arm + '/' + task]
            if reading['trace_sha256'] != sha(path):
                raise ValueError('Reader trace identity mismatch')
            readings[review['reader']] = checked_reading(reading, ev,
                                                        accepted['call'] if accepted else None)
        # Preserve all readings. Different timestamps or known/unknown are disagreements.
        disagreement = {field: len({r[field]['step_id'] if r[field] else None
                                    for r in readings.values()}) > 1 if len(readings) >= 2 else None
                        for field in ('hypothesis', 'verification')}
        comparisons.append({'task_id': task, 'arm': arm, 'resolved': row['outcome']['resolved'],
                            'trace_sha256': sha(path), 'prediction_sha256': row['prediction_sha256'],
                            'first_positional_old_line_exposure': point(exposure) if exposure else None,
                            'exposure_observation': exposure['obs'] if exposure else None,
                            'first_editor_attempt': point(edits[0]) if edits else None,
                            'first_accepted_editor_call': point(accepted) if accepted else None,
                            'editor_attempts': [{**point(e), 'status': e['obs'].get('status'),
                                                 'path': e['args'].get('filepath')} for e in edits],
                            'final_patch_paths': row['final_patch_paths'],
                            'readers': readings, 'disagreements': disagreement})
    result = {'selection_rule': SELECTION, 'count': count, 'eligible_tasks': eligible,
              'selected_tasks': selected, 'reference_sha256': sha(reference_path),
              'tasks_sha256': sha(tasks),
              'source_sha256': {str(p): sha(p) for p in (Path(__file__), Path('src/failure_mechanisms.py'))},
              'run_metadata_sha256': {str(p.relative_to(release)): sha(p) for arm in ARMS
                                      for p in sorted((release / arm).glob('*.json'))},
              'reader_files': {str(p): sha(Path(p)) for p in readers},
              'attempts': comparisons,
              'limits': ['Outcome-selected illustrative sample; no mechanism prevalence or causal claims.',
                         'Exposure is positional, pre-mutation, numbered-source only; null means unknown.',
                         'Hypothesis timestamps are reader judgments, not executable correctness proofs.',
                         'Accepted editor calls and verification attempts do not certify a correct patch.',
                         'AI reader agreement does not replace independent teammate approval.']}
    output.mkdir(parents=True)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reader', type=Path, action='append', default=[])
    p.add_argument('--count', type=int, default=3)
    a = p.parse_args()
    result = compare(a.release, a.tasks, a.output, a.reader, a.count)
    print(json.dumps({'selected_tasks': result['selected_tasks'], 'attempts': len(result['attempts'])}))


if __name__ == '__main__':
    main()
