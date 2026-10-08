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
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--tasks', type=Path, default=Path('data/raw/gemma-4-developer-agent/tasks.jsonl'))
    args = parser.parse_args()
    source, files = collect_files(args.agent_dir)
    rows = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [row for row in rows if row['instance_id'] == args.task_id]
    if len(rows) != 1:
        raise ValueError('Expected one matching public task')
    if args.run_dir.exists():
        raise ValueError('Run directory already exists; preserve its evidence and choose a new name')
    inputs = args.run_dir / 'inputs'
    agent_dir = inputs / 'agent'
    agent_dir.mkdir(parents=True)
    for path in files:
        target = agent_dir / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    public_task = {key: value for key, value in rows[0].items() if key in PUBLIC_FIELDS}
    task_file = inputs / 'tasks.jsonl'
    # Freeze identical task bytes on Windows and Linux/Kaggle.
    task_file.write_bytes((json.dumps(public_task) + '\n').encode('utf-8'))
    bundle = build(agent_dir, inputs / 'submission.zip')
    revision = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'task_id': args.task_id, 'repo': public_task['repo'], 'base_commit': public_task['base_commit'],
                'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
                'agent_source': str(source), 'agent_archive_sha256': bundle['sha256'],
                'public_task_sha256': sha256(task_file), 'public_fields': sorted(public_task),
                'model_handle': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
                'status': 'prepared; no model or tests executed'}
    (args.run_dir / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
