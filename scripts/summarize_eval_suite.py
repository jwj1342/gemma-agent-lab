"""Report task resolution, coverage, costs, and paired candidate differences."""
import argparse
from collections import Counter
import json
from pathlib import Path

from eval_suite_lib import artifact_valid, read_json, read_tasks, wilson, write_json
from verify_eval_run import verify_inputs


def trace_metrics(path):
    if not path.exists():
        return {'graph_calls': None, 'tool_names': {}, 'tokens': {}}
    trace = read_json(path)
    names = Counter()
    # Traces record each tool once in an assistant step's tool_calls list.
    for step in trace.get('steps', []):
        for call in step.get('tool_calls', []):
            name = call.get('function_name') or call.get('function', {}).get('name') or call.get('name')
            if name:
                names[name] += 1
    return {'graph_calls': sum(names[n] for n in ['get_code_neighbors', 'get_code_subgraph', 'search_similar_code']),
            'tool_names': dict(names), 'tokens': trace.get('final_metrics', {})}


def summarize(run):
    manifest = verify_inputs(run)
    assets = read_json(run / 'runtime-inputs.json')
    controls = read_json(run / 'controls.json')
    tasks = {t['instance_id']: t for t in read_tasks(run / 'inputs/tasks.jsonl')}
    rows = []
    for task_id in manifest['task_ids']:
        row = {'task_id': task_id, 'repo': tasks[task_id]['repo'], 'controls_valid': controls[task_id]['valid'],
               'resolved': None, 'completion_clean': False, 'evaluated': False}
        if not row['controls_valid']:
            row['status'] = 'invalid_controls'
            row['control_errors'] = {k: controls[task_id][k].get('error_message') for k in ['empty', 'gold']}
        else:
            prediction_file = run / 'infer' / task_id / 'prediction.json'
            grade_file = run / 'verify/agent' / task_id / 'result.json'
            if artifact_valid(prediction_file) and artifact_valid(grade_file):
                prediction, grade = read_json(prediction_file), read_json(grade_file)
                if prediction['task_id'] != task_id or grade['task_id'] != task_id:
                    raise ValueError('Result task identity mismatch')
                row.update(resolved=grade.get('resolved') is True, evaluated=True,
                           completion_clean=not prediction.get('error') and bool(prediction['agent_patch']),
                           inference_error=prediction.get('error'), verification_error=grade.get('error_message'),
                           inference_seconds=prediction.get('duration_seconds'), agent_seconds=prediction.get('agent_seconds'),
                           verification_seconds=grade.get('duration_seconds', 0),
                           llm_calls=prediction.get('llm_calls'), budgeted_tool_calls=prediction.get('tool_calls'))
                if row['resolved']:
                    row['status'] = 'resolved'
                elif prediction.get('error'):
                    row['status'] = 'inference_error'
                elif not prediction['agent_patch']:
                    row['status'] = 'empty_patch'
                elif grade.get('error_message'):
                    row['status'] = 'verification_error'
                else:
                    row['status'] = 'tests_failed'
                row.update(trace_metrics(run / 'infer' / task_id / 'results/traces' / ('trace_' + task_id + '.json')))
            else:
                row['status'] = 'pending'
        rows.append(row)
    valid = sum(row['controls_valid'] for row in rows)
    evaluated = sum(row['evaluated'] for row in rows)
    solved = sum(row['resolved'] is True for row in rows)
    complete = evaluated == valid and valid > 0
    sessions = [read_json(p) for p in sorted((run / 'infer/results/sessions').glob('*.json'))]
    timings = [read_json(p) for p in sorted((run / 'timings').glob('*.json'))]
    repo_results = {}
    for repo in sorted({row['repo'] for row in rows}):
        rs = [row for row in rows if row['repo'] == repo]
        repo_results[repo] = {'selected': len(rs), 'valid_controls': sum(r['controls_valid'] for r in rs),
                              'evaluated': sum(r['evaluated'] for r in rs),
                              'resolved': sum(r['resolved'] is True for r in rs)}
    starts = sum(s.get('model_startups', 0) for s in sessions)
    startup_seconds = sum(s.get('model_startup_seconds', 0) for s in sessions)
    return {'cohort': manifest['cohort'], 'benchmark_sha256': manifest['benchmark_sha256'],
            'public_task_sha256': manifest['public_task_sha256'], 'agent_sha256': manifest['agent_archive_sha256'],
            'model_sha256': assets['model_sha256'], 'verification_identity': assets['verification_identity'],
            'protocol': manifest['protocol'], 'budget_sha256': assets['budget_sha256'],
            'complete': complete, 'selected': len(rows), 'valid_controls': valid,
            'evaluated': evaluated, 'resolved': solved,
            'resolution_rate': solved / valid if complete else None,
            'resolution_rate_denominator': 'All fixed tasks with valid pre-inference controls, including agent failures',
            'selected_task_success_fraction': solved / len(rows) if complete else None,
            'wilson_95_interval': wilson(solved, valid) if complete else None,
            'interval_limit': 'Descriptive binomial interval; repository correlation and adaptive development limit inference.',
            'completion_clean': sum(row['completion_clean'] for row in rows), 'by_repository': repo_results,
            'cost': {'model_startups': starts, 'model_startup_seconds': startup_seconds,
                     'inference_seconds': sum(r.get('inference_seconds', 0) or 0 for r in rows),
                     'verification_seconds': sum(r.get('verification_seconds', 0) or 0 for r in rows),
                     'runner_seconds': sum(t.get('total_seconds', 0) for t in timings),
                     'control_cache_hits': sum(t.get('control_hits', 0) for t in timings),
                     'control_cache_misses': sum(t.get('control_misses', 0) for t in timings),
                     'model_startups_avoided_vs_one_per_task': max(evaluated - starts, 0),
                     'queue_time_included': False},
            'hardware': [s['hardware'] for s in sessions if 'hardware' in s], 'tasks': rows,
            'scope': 'Local public-task resolution under the official verifier. H100 timing is not L4 timing. '
                     'Scores with invalid controls have reduced coverage and cannot be compared as full-cohort scores.'}


def compare(candidate, baseline):
    for key in ('benchmark_sha256', 'public_task_sha256', 'model_sha256', 'verification_identity', 'protocol', 'budget_sha256'):
        if candidate[key] != baseline[key]:
            raise ValueError(f'Comparison requires matching {key}')
    if not candidate['complete'] or not baseline['complete']:
        raise ValueError('Both evaluations must finish before comparison')
    left, right = ({r['task_id']: r for r in x['tasks']} for x in [candidate, baseline])
    if set(left) != set(right) or any(left[i]['controls_valid'] != right[i]['controls_valid'] for i in left):
        raise ValueError('Task membership or valid-control masks differ')
    ids = [i for i in left if left[i]['controls_valid']]
    gains = [i for i in ids if left[i]['resolved'] and not right[i]['resolved']]
    losses = [i for i in ids if right[i]['resolved'] and not left[i]['resolved']]
    return {'baseline_agent_sha256': baseline['agent_sha256'], 'candidate_agent_sha256': candidate['agent_sha256'],
            'matched_tasks': len(ids), 'newly_resolved': gains, 'regressions': losses,
            'both_resolved': [i for i in ids if left[i]['resolved'] and right[i]['resolved']],
            'neither_resolved': [i for i in ids if not left[i]['resolved'] and not right[i]['resolved']],
            'resolution_rate_difference': (len(gains) - len(losses)) / len(ids),
            'scope': 'Paired descriptive comparison. One stochastic attempt per task; no claim of statistical significance.'}


def render(report):
    lines = ['# Local evaluation report', '',
             f"Cohort: **{report['cohort']}**. Resolved **{report['resolved']}/{report['valid_controls']}** eligible tasks; "
             f"**{report['valid_controls']}/{report['selected']}** tasks passed environment controls.", '',
             report['scope'], '', '| Task | Repository | Outcome | Inference seconds | Graph calls |',
             '| --- | --- | --- | ---: | ---: |']
    for row in report['tasks']:
        seconds = row.get('inference_seconds')
        lines.append(f"| {row['task_id']} | {row['repo']} | {row['status']} | "
                     f"{round(seconds, 2) if seconds is not None else '—'} | {row.get('graph_calls', '—')} |")
    cost = report['cost']
    lines += ['', f"Model startups: {cost['model_startups']}; startup time: {cost['model_startup_seconds']:.2f} s; "
              f"runner time: {cost['runner_seconds']:.2f} s (queue excluded).",
              f"Control cache: {cost['control_cache_hits']} hits / {cost['control_cache_misses']} misses.", '']
    if 'comparison' in report:
        c = report['comparison']
        lines += [f"Paired change: {len(c['newly_resolved'])} newly resolved, {len(c['regressions'])} regressions.",
                  f"Newly resolved: {', '.join(c['newly_resolved']) or 'none'}.",
                  f"Regressions: {', '.join(c['regressions']) or 'none'}.", '', c['scope'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    report = summarize(args.run_dir)
    if args.baseline:
        report['comparison'] = compare(report, summarize(args.baseline))
    write_json(args.run_dir / 'summary.json', report)
    (args.run_dir / 'report.md').write_text(render(report))
    print(json.dumps({k: report[k] for k in ['complete', 'selected', 'valid_controls', 'resolved', 'resolution_rate', 'cost']}, indent=2))
    if not report['complete']:
        raise SystemExit('Evaluation incomplete; resume the same frozen run')


if __name__ == '__main__':
    main()
