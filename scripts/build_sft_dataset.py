"""Build the LoRA SFT dataset from the public tasks already in this repository.

This is the offline path the repository can actually support today: `data/public/tasks.jsonl` holds
129 tasks with their problem statement and reference patch. No verified agent trajectory exists
locally (the run traces live in private Kaggle kernel outputs), so this builds issue-to-reference-patch
records, which is the weaker signal the training plan warns about:

  "Issue-to-diff training alone does not teach the entire agent loop."

Two guardrails are built in, because the failure mode here is silent and expensive:

1. Repository-disjoint splits. The public set is four repositories (fastapi, rich, requests, httpx)
   that are hidden at scoring time, so a random split would leak the same repositories into both
   sides and report a transfer that does not exist. The split is by repository.
2. Size filtering. The largest reference patch is ~308 kB, which cannot be trained on; records are
   capped at a defensible length and the drops are reported, never silently truncated.

    python scripts/build_sft_dataset.py --output data/processed/sft-public-v1.jsonl

The records pair a prompt and a completion so that `completion_only_loss=True` masks the prompt.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / 'data/public/tasks.jsonl'

PROMPT_TEMPLATE = """You are an expert software engineer. Resolve the issue in the {repo} repository.

Issue:
{problem_statement}

Respond with a single unified git diff that fixes the issue. Output only the diff."""


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with path.open(encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def build(tasks_path: Path, output: Path, max_patch_chars: int, holdout_repos: list[str]) -> dict:
    tasks = load_tasks(tasks_path)
    if not tasks:
        raise SystemExit(f'no tasks in {tasks_path}')
    repos = sorted({task['repo'] for task in tasks})
    unknown_holdouts = sorted(set(holdout_repos) - set(repos))
    if unknown_holdouts:
        raise SystemExit(f'holdout repositories not present in {tasks_path}: {unknown_holdouts}')
    if len(holdout_repos) >= len(repos):
        raise SystemExit('every repository is held out; no training data would remain')

    kept, dropped = [], Counter()
    for task in tasks:
        patch = task.get('patch') or ''
        statement = (task.get('problem_statement') or '').strip()
        if not statement:
            dropped['empty_problem_statement'] += 1
            continue
        if not patch.strip():
            dropped['empty_patch'] += 1
            continue
        if len(patch) > max_patch_chars:
            dropped['patch_too_long'] += 1
            continue
        kept.append({
            'instance_id': task['instance_id'],
            'repo': task['repo'],
            'split': 'holdout' if task['repo'] in holdout_repos else 'train',
            'prompt': PROMPT_TEMPLATE.format(repo=task['repo'], problem_statement=statement),
            'completion': patch,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for record in kept:
            handle.write(json.dumps(record) + '\n')

    counts = Counter(record['split'] for record in kept)
    train_repos = sorted({record['repo'] for record in kept if record['split'] == 'train'})
    return {
        'output': str(output),
        'records_written': len(kept),
        'split_counts': dict(counts),
        'train_repositories': train_repos,
        'holdout_repositories': sorted(holdout_repos),
        'tasks_total': len(tasks),
        'dropped': dict(dropped),
        'max_patch_chars': max_patch_chars,
        'schema': 'prompt/completion JSONL; train with completion_only_loss=True',
        'caveat': 'Issue-to-reference-patch supervision. It cannot teach the tool loop, and the four '
                  'public repositories are hidden at scoring time, so transfer is limited to format '
                  'and patch-shape behaviour. Treat any adapter trained on it as an experiment that '
                  'needs a paired control to be interpretable.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tasks', type=Path, default=DEFAULT_TASKS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-patch-chars', type=int, default=8000)
    parser.add_argument('--holdout-repos', nargs='+', default=['encode/httpx'])
    args = parser.parse_args()
    print(json.dumps(build(args.tasks, args.output, args.max_patch_chars, args.holdout_repos),
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
