"""Freeze grouped public cohorts after excluding known training and development exposure."""
import argparse
from collections import defaultdict
from pathlib import Path

from build_submission import sha256
from eval_suite_lib import count_repositories, digest, read_json, read_tasks, snapshot_group, write_json


def define(tasks, exposures, known_ids, seed=42):
    groups = defaultdict(list)
    for task in tasks:
        groups[snapshot_group(task)].append(task)
    known_groups = {snapshot_group(t) for t in tasks if t['instance_id'] in known_ids}
    excluded, eligible = [], []
    for group, members in sorted(groups.items()):
        reasons = sorted({reason for prefix, reason in exposures if group.startswith(prefix.lower())})
        if group in known_groups:
            reasons.append('Previously inspected end-to-end development task')
        if reasons:
            excluded.extend({'task_id': t['instance_id'], 'group': group, 'reasons': reasons} for t in members)
        else:
            eligible.append((group, members))
    by_repo = defaultdict(list)
    for group, members in eligible:
        by_repo[members[0]['repo']].append((group, members))
    # Cohort sizes count complete snapshot groups. This public inventory has unique
    # snapshots after exclusions; future inventories may produce larger cohorts.
    dev_groups = {'fastapi/fastapi': 23, 'Textualize/rich': 12, 'psf/requests': 4, 'encode/httpx': 1}
    fast_groups = {'fastapi/fastapi': 6, 'Textualize/rich': 4, 'psf/requests': 1, 'encode/httpx': 1}
    cohorts = {'fast': [], 'dev': [], 'holdout': [], 'smoke': sorted(known_ids)}
    for repo, items in sorted(by_repo.items()):
        items.sort(key=lambda item: digest([seed, item[0]]))
        ndev = dev_groups.get(repo, 0)
        nfast = fast_groups.get(repo, 0)
        for index, (_, members) in enumerate(items):
            ids = sorted(t['instance_id'] for t in members)
            cohorts['dev' if index < ndev else 'holdout'].extend(ids)
            if index < nfast:
                cohorts['fast'].extend(ids)
    return {'schema_version': 1, 'seed': seed, 'cohorts': cohorts, 'excluded': excluded,
            'eligible_count': sum(len(members) for _, members in eligible),
            'repository_counts': count_repositories(tasks),
            'cohort_repository_counts': {name: count_repositories([t for t in tasks if t['instance_id'] in ids])
                                         for name, ids in cohorts.items()},
            'task_metadata': {t['instance_id']: {'repo': t['repo'], 'base_commit': t['base_commit'],
                                               'public_group': snapshot_group(t)} for t in tasks},
            'scope': 'Exact-snapshot exclusion, not proof of unseen repositories or pretraining decontamination. '
                     'Fast is nested in dev. Holdout must not guide routine tuning. '
                     'Eligibility still requires empty/gold environment controls.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('data/raw/gemma-4-developer-agent/tasks.jsonl'))
    parser.add_argument('--swe-manifest', type=Path, action='append', required=True)
    parser.add_argument('--graph-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Do not overwrite a frozen benchmark; use a new version')
    exposures = []
    for path in args.swe_manifest:
        for split, groups in read_json(path)['groups'].items():
            exposures.extend((g, f'SWE actions: {path.parent.name}/{split}') for g in groups)
    for item in read_json(args.graph_manifest)['executed_components']:
        exposures.append((item['group'], 'Graph components: ' + item['split']))
    result = define(read_tasks(args.tasks), set(exposures), {'rich_3454'})
    result['source_tasks_sha256'] = sha256(args.tasks)
    result['exposure_sources'] = {str(p): sha256(p) for p in [*args.swe_manifest, args.graph_manifest]}
    write_json(args.output, result)
    print({name: len(ids) for name, ids in result['cohorts'].items()})


if __name__ == '__main__':
    main()
