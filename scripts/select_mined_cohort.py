"""Freeze a stratified cohort from the tasks whose gold/empty controls passed.

Usage:
  python scripts/select_mined_cohort.py --tasks <tasks.jsonl> --logs <log1> [<log2> ...] \
      --seed mined-cohort-2026-11-21 --size 105 --out cohort.json

Strict controls come from detailed sweep records with JUnit counts. Old rc-only logs require
--legacy-screen and produce a provisional manifest, never inference approval.
Selection is deterministic: per repository, terse statements first, then ascending SHA-256("<seed>:<id>");
per-repository quotas are proportional with largest-remainder rounding. The draw is frozen: rerunning with
the same inputs and seed produces the same cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from collect_mined_sweep import from_log

def valid_ids(logs: list[Path], tasks: Path, legacy_screen=False) -> set[str]:
    latest = {}
    for path in logs:
        for row in from_log(path, tasks):
            if 'legacy_valid' in row:
                accepted = (legacy_screen and row['legacy_valid']
                            and str(row['gold_rc']) == '0' and str(row['empty_rc']) == '1')
            else:
                accepted = row['valid']
            latest[row['instance_id']] = bool(accepted)
    return {task_id for task_id, accepted in latest.items() if accepted}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--logs', nargs='+', type=Path, required=True)
    parser.add_argument('--seed', default='mined-cohort-2026-11-21')
    parser.add_argument('--size', type=int, default=105)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--legacy-screen', action='store_true',
                        help='Allow historical rc 0/1 rows for a provisional cohort; never inference approval')
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    valid = valid_ids(args.logs, args.tasks, args.legacy_screen)
    pool = [task for task in tasks if task['instance_id'] in valid]
    if not pool:
        raise SystemExit('no valid tasks found in the logs')
    if args.size < 1 or args.size > len(pool):
        raise SystemExit('cohort size must be between one and the accepted pool size')

    by_repo: dict[str, list[dict]] = {}
    for task in pool:
        by_repo.setdefault(task['repo'], []).append(task)
    for repo in by_repo:
        by_repo[repo].sort(key=lambda t: (t.get('statement_type') != 'terse',
                                          hashlib.sha256(f"{args.seed}:{t['instance_id']}".encode()).hexdigest()))

    total = len(pool)
    quotas = {repo: len(items) * args.size // total for repo, items in by_repo.items()}
    remainders = sorted(((len(items) * args.size) % total, repo) for repo, items in by_repo.items())[::-1]
    for _, repo in remainders:
        if sum(quotas.values()) >= args.size:
            break
        quotas[repo] += 1

    chosen = []
    for repo in sorted(by_repo):
        chosen.extend(by_repo[repo][:quotas[repo]])
    chosen.sort(key=lambda t: hashlib.sha256(f"{args.seed}:{t['instance_id']}".encode()).hexdigest())

    record = {
        'seed': args.seed,
        'schema_version': 2,
        'control_policy': 'legacy_return_code_screen' if args.legacy_screen else 'junit_assertion_controls',
        'evaluation_ready': False,
        'source_tasks_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        'source_log_sha256': [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.logs],
        'requested_size': args.size,
        'valid_pool': total,
        'valid_by_repository': {repo: len(items) for repo, items in sorted(by_repo.items())},
        'quotas': quotas,
        'selected': len(chosen),
        'tasks': [{'instance_id': t['instance_id'], 'repo': t['repo'],
                   'statement_type': t.get('statement_type'), 'leak_flag': t.get('leak_flag'),
                   'base_commit': t['base_commit']} for t in chosen],
        'statement_mix': {kind: sum(1 for t in chosen if t.get('statement_type') == kind)
                          for kind in ('terse', 'concrete', 'design')},
        'leak_mix': {str(flag): sum(1 for t in chosen if t.get('leak_flag') is flag) for flag in (True, False)},
        'source_logs': [path.name for path in args.logs],
        'note': 'Recheck strict controls on the frozen runtime before inference. Legacy rows lack JUnit evidence.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps({k: record[k] for k in ('seed', 'valid_pool', 'valid_by_repository', 'quotas',
                                             'selected', 'statement_mix', 'leak_mix')}, indent=2))
    print('written:', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
