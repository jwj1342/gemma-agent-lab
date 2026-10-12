"""Report a registered development expansion separately from previously exposed tasks."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_eval_suite import compare


IDENTITIES = ('benchmark_sha256', 'model_sha256', 'verification_identity', 'protocol', 'budget_sha256')


def checked_rows(report):
    rows = report['tasks']
    ids = [r['task_id'] for r in rows]
    if len(set(ids)) != len(ids) or report['selected'] != len(rows):
        raise ValueError('Summary contains duplicate tasks or inconsistent selected count')
    eligible = [r for r in rows if r['controls_valid']]
    if (not report['complete'] or not eligible
            or any(not r['evaluated'] or type(r['resolved']) is not bool for r in eligible)
            or report['valid_controls'] != len(eligible)):
        raise ValueError('Summary must contain completed outcomes for every eligible task')
    return {r['task_id']: r for r in rows}


def metrics(rows):
    eligible = [r for r in rows if r['controls_valid']]
    resolved = sum(r['resolved'] for r in eligible)
    return {
        'selected': len(rows), 'eligible': len(eligible), 'resolved': resolved,
        'resolution_rate': resolved / len(eligible) if eligible else None,
        'clean_resolved': sum(r['resolved'] and r['completion_clean'] for r in eligible),
        'completion_clean': sum(r['completion_clean'] for r in eligible),
        'inference_seconds': sum(r.get('inference_seconds') or 0 for r in eligible),
        'llm_calls': sum(r.get('llm_calls') or 0 for r in eligible),
        'reported_model_tokens': sum((r.get('tokens') or {}).get('total_tokens', 0) for r in eligible),
        'tasks_missing_token_totals': sum('total_tokens' not in (r.get('tokens') or {}) for r in eligible),
    }


def expansion(candidate, baseline, fast_baseline, registration):
    left, right, fast = (checked_rows(r) for r in (candidate, baseline, fast_baseline))
    # Check the full development comparison before selecting either reporting group.
    full_comparison = compare(candidate, baseline)
    if candidate['cohort'] != 'dev' or baseline['cohort'] != 'dev' or fast_baseline['cohort'] != 'fast':
        raise ValueError('Expected two development summaries and a fast baseline')
    for key in IDENTITIES:
        if baseline[key] != fast_baseline[key]:
            raise ValueError('Expansion changed fast-baseline ' + key)
    if baseline['agent_sha256'] != fast_baseline['agent_sha256']:
        raise ValueError('Expansion changed the baseline agent')
    additional = registration['additional_task_ids']
    if len(set(additional)) != len(additional):
        raise ValueError('Registration has duplicate additional tasks')
    if (set(fast) & set(additional) or set(fast) | set(additional) != set(right)
            or len(right) != registration['task_count']
            or len(fast) != registration['previously_exposed_fast_task_count']):
        raise ValueError('Registered expansion does not partition the complete development cohort')
    groups = {}
    for label, ids in [('full_dev', sorted(right)), ('additional_dev', sorted(additional)),
                       ('previously_exposed_fast', sorted(fast))]:
        candidate_rows, baseline_rows = ([mapping[i] for i in ids] for mapping in (left, right))
        if not any(r['controls_valid'] for r in baseline_rows):
            paired = None
        elif label == 'full_dev':
            paired = full_comparison
        else:
            paired = compare(dict(candidate, tasks=candidate_rows), dict(baseline, tasks=baseline_rows))
        groups[label] = {'task_ids': ids, 'candidate': metrics(candidate_rows),
                         'baseline': metrics(baseline_rows), 'comparison': paired}
    return {
        'groups': groups,
        'interpretation': 'The full development cohort includes tasks used for candidate selection. '
            'Additional tasks are reported separately under the registered partition. '
            'All valid-control tasks remain in each denominator, including agent failures. '
            'One stochastic attempt per task does not establish statistical significance or leaderboard gains.',
        'cost_scope': 'Per-task inference includes setup and excludes model startup, controls, grading and queue time. '
            'Token totals include repeated context; missing totals are counted explicitly.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True, help='Completed development run directory')
    parser.add_argument('--baseline', type=Path, required=True, help='Completed development baseline directory')
    parser.add_argument('--fast-baseline', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {'candidate': args.candidate / 'summary.json', 'baseline': args.baseline / 'summary.json',
             'fast_baseline': args.fast_baseline / 'summary.json', 'registration': args.registration}
    values, sources = {}, {}
    for name, path in paths.items():
        raw = path.read_bytes()
        values[name] = json.loads(raw)
        sources[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    report = expansion(**values)
    report['sources'] = sources
    report['reporter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'groups': list(report['groups'])}))


if __name__ == '__main__':
    main()
