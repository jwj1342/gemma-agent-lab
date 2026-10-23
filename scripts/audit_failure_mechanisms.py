"""Audit frozen r4/r5 releases on CPU; never execute traced commands.

Run as python -m scripts.audit_failure_mechanisms. Outputs contain task-derived
evidence and must remain in ignored runs/, not Git or reusable agent prompts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from src.failure_mechanisms import (apply_annotation, load_run, sha,
                                    reference_hunks, statement_exposure, terminal_flags)


def run_audit(release, runs, output, annotations=None, tasks=None):
    release, output = Path(release), Path(output)
    if output.exists():
        raise ValueError('Preserve previous audit outputs')
    if any(not re.fullmatch(r'[a-zA-Z0-9_-]+', name) for name in runs) or len(set(runs)) != len(runs):
        raise ValueError('Use distinct plain run names')
    reference_path = release / 'reference-patch-locations.json'
    reference = json.loads(reference_path.read_text())
    labels = json.loads(Path(annotations).read_text()) if annotations else {}
    patches = {}
    if tasks:
        for line in Path(tasks).read_text().split('\n'):
            if line.strip():
                task = json.loads(line)
                if task['instance_id'] in patches:
                    raise ValueError('Duplicate task identity')
                patches[task['instance_id']] = task['patch']
    rows = [row for name in runs for row in load_run(release, name, reference)]
    if tasks:
        for task_id in {row['task_id'] for row in rows}:
            if task_id not in patches or reference_hunks(patches[task_id]) != reference[task_id]:
                raise ValueError('Task reference patch does not match the released location map')
    failures = [r for r in rows if r['outcome'].get('controls_valid') is True
                and r['outcome'].get('resolved') is False]
    valid_keys = {r['run'] + '/' + r['task_id'] for r in failures}
    if set(labels) - valid_keys:
        raise ValueError('Annotation outside the valid-control failure cohort')
    for row in failures:
        path = release / row['run'] / 'tasks' / row['task_id'] / 'trace.json'
        trace = json.loads(path.read_text()) if row['trace_present'] else None
        row['mechanism'] = apply_annotation(row, labels.get(row['run'] + '/' + row['task_id']), trace)
        row['termination'] = terminal_flags(row)
        row['statement_exposure'] = (statement_exposure(trace, patches[row['task_id']])
                                     if trace is not None and row['task_id'] in patches else None)
    summary = {}
    for name in runs:
        cohort = [r for r in rows if r['run'] == name]
        failed = [r for r in failures if r['run'] == name]
        valid = [r for r in cohort if r['outcome'].get('controls_valid') is True]
        summary[name] = {'attempts': len(cohort), 'valid_controls': len(valid),
                         'resolved': sum(r['outcome'].get('resolved') is True for r in valid),
                         'failed': len(failed), 'unique_failed_tasks': len({r['task_id'] for r in failed}),
                         'missing_trace': sum(not r['trace_present'] for r in failed),
                         'missing_prediction': sum(not r['prediction_present'] for r in failed),
                         'mechanisms': dict(Counter(r['mechanism']['category'] for r in failed))}
        for key in ('read_reference', 'attempted_reference', 'successful_editor_reference',
                    'final_patch_reference', 'read_no_reference_patch'):
            summary[name][key] = sum(r.get(key) is True for r in failed)
        summary[name]['empty_final_patch'] = sum(r.get('final_patch_paths') == [] for r in failed)
        measured = [r['statement_exposure'] for r in failed if r['statement_exposure'] is not None]
        summary[name]['old_line_coverage_measured'] = len(measured)
        summary[name]['with_changed_old_lines'] = sum(r['changed_old_lines'] > 0 for r in measured)
        summary[name]['with_exposed_changed_old_lines'] = sum(r['exposed_changed_old_lines'] > 0 for r in measured)
    result = {'schema_version': 1, 'reference_sha256': sha(reference_path),
              'annotations_sha256': sha(annotations) if annotations else None,
              'tasks_sha256': sha(tasks) if tasks else None,
              'total_failed_attempts': len(failures),
              'unique_failed_tasks': len({r['task_id'] for r in failures}),
              'summary': summary, 'failures': failures,
              'limits': ['Verifier outcomes retained, not rerun.',
                         'Hunk/old-line exposure is not recognition; manual labels are reviewer judgments.',
                         'Unnumbered shell output and changed intermediate source positions need manual review.',
                         'Missing artifacts remain unknown; task attempts are not independent tasks.']}
    # Per-task summaries and frozen runtime manifests are part of the evidence identity.
    result['input_files'] = {str(p.relative_to(release)): sha(p) for name in runs
                             for p in sorted((release / name).glob('*.json'))}
    output.mkdir(parents=True)
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps({k: v for k, v in result.items()
                                                    if k != 'failures'}, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, help='Private run/task keyed reviewer annotations')
    parser.add_argument('--tasks', type=Path, help='Private tasks JSONL for changed old-line coverage')
    args = parser.parse_args()
    result = run_audit(args.release, args.runs, args.output, args.annotations, args.tasks)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
