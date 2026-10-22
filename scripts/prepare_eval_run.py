"""Freeze a submission and a redacted public task into a new evaluation run directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from build_submission import build, collect_files, sha256
from prepare_eval_task import PUBLIC_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--agent-dir', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--task-id')
    selection.add_argument('--task-ids', nargs='+')
    parser.add_argument('--tasks', type=Path, default=Path('data/raw/gemma-4-developer-agent/tasks.jsonl'))
    args = parser.parse_args()
    source, files = collect_files(args.agent_dir)
    rows = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = args.task_ids or [args.task_id]
    rows = [row for row in rows if row['instance_id'] in ids]
    if len(set(ids)) != len(ids) or sorted(row['instance_id'] for row in rows) != sorted(ids):
        raise ValueError('Expected exactly one matching public task per requested id')
    if args.run_dir.exists():
        raise ValueError('Run directory already exists; preserve its evidence and choose a new name')
    inputs = args.run_dir / 'inputs'
    agent_dir = inputs / 'agent'
    agent_dir.mkdir(parents=True)
    for path in files:
        target = agent_dir / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    public_tasks = [{key: value for key, value in row.items() if key in PUBLIC_FIELDS} for row in rows]
    public_task = public_tasks[0]
    task_file = inputs / 'tasks.jsonl'
    # Freeze identical task bytes on Windows and Linux/Kaggle.
    task_file.write_bytes((''.join(json.dumps(task) + '\n' for task in public_tasks)).encode('utf-8'))
    bundle = build(agent_dir, inputs / 'submission.zip')
    revision = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'task_id': args.task_id, 'repo': public_task['repo'], 'base_commit': public_task['base_commit'],
                'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
                'agent_source': str(source), 'agent_archive_sha256': bundle['sha256'],
                'public_task_sha256': sha256(task_file), 'public_fields': sorted(public_task),
                'model_handle': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
                'status': 'prepared; no model or tests executed'}
    if args.task_ids:
        for key in ('task_id', 'repo', 'base_commit'):
            manifest.pop(key)
        manifest['task_ids'] = ids
    (args.run_dir / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
