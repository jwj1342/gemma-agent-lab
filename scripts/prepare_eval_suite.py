"""Freeze a candidate, cohort, and runtime for batch evaluation without loading models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from build_submission import build, collect_files, sha256
from eval_suite_lib import read_json, read_tasks, write_json
from prepare_eval_task import PUBLIC_FIELDS

RUNTIME_FILES = ['build_submission.py', 'prepare_eval_task.py', 'verify_eval_run.py',
                 'eval_suite_lib.py', 'evaluate_agent.py', 'validate_agent.py', 'summarize_eval_smoke.py',
                 'run_eval_container.sh', 'container_entry.sh', 'run_eval_suite.py',
                 'summarize_eval_suite.py']


def prepare(run, agent, tasks, benchmark, cohort, allow_holdout=False):
    if run.exists():
        raise ValueError('Run already exists; resume it or choose a new name')
    definition = read_json(benchmark)
    if sha256(tasks) != definition['source_tasks_sha256']:
        raise ValueError('Task inventory differs from the frozen benchmark')
    if cohort == 'holdout' and not allow_holdout:
        raise ValueError('Holdout requires --allow-holdout; reserve it for milestone decisions')
    ids = definition['cohorts'][cohort]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Cohort must contain unique tasks')
    lookup = {t['instance_id']: t for t in read_tasks(tasks)}
    selected = [{k: v for k, v in lookup[i].items() if k in PUBLIC_FIELDS} for i in ids]
    root, files = collect_files(agent)
    frozen = run / 'inputs/agent'
    frozen.mkdir(parents=True)
    for path in files:
        target = frozen / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    public_file = run / 'inputs/tasks.jsonl'
    public_file.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in selected))
    shutil.copy2(benchmark, run / 'inputs/benchmark.json')
    bundle = build(frozen, run / 'inputs/submission.zip')
    runtime = run / 'inputs/runtime'
    runtime.mkdir()
    for name in RUNTIME_FILES:
        shutil.copy2(Path(__file__).parent / name, runtime / name)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    manifest = {'suite_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'task_ids': ids, 'cohort': cohort, 'benchmark_sha256': sha256(benchmark),
                'git_revision': git.stdout.strip() if git.returncode == 0 else None,
                'working_tree_dirty': bool(dirty.stdout.strip()),
                'agent_source': str(root), 'agent_archive_sha256': bundle['sha256'],
                'public_task_sha256': sha256(public_file),
                'runtime_sha256': {name: sha256(runtime / name) for name in RUNTIME_FILES},
                'model_handle': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
                'protocol': {'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.8,
                             'max_model_len': 32768, 'vllm_seed': 42, 'task_concurrency': 1},
                'status': 'Frozen preparation only; no inference or verification executed'}
    write_json(run / 'run-manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--agent-dir', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, default=Path('data/raw/gemma-4-developer-agent/tasks.jsonl'))
    parser.add_argument('--benchmark', type=Path, default=Path('configs/evaluation/public-v1.json'))
    parser.add_argument('--cohort', choices=['smoke', 'fast', 'dev', 'holdout'], default='fast')
    parser.add_argument('--allow-holdout', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.run_dir, args.agent_dir, args.tasks, args.benchmark, args.cohort,
                             args.allow_holdout), indent=2))


if __name__ == '__main__':
    main()
