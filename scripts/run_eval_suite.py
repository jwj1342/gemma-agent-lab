"""Orchestrate a frozen local suite, cached controls, resumable inference, and grading.

Use inside an allocation. Model inference never mounts labels or the control cache.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from build_submission import sha256
from eval_suite_lib import (artifact_valid, cache_record, digest, read_control_cache, read_json,
                            read_tasks, save_artifact, write_json)
from verify_eval_run import verify_inputs


def stamp(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def indexed_hash(path, index, force=False):
    key = str(path.absolute())
    before = stamp(path)
    cached = index.get(key)
    # Some shared filesystems expose coarse timestamps. Never trust an index
    # recorded immediately after a write: another same-tick edit can have the
    # same size and stat tuple. Small source/config files are cheap to reread.
    if not force and cached and cached.get('stable') and cached['stat'] == before and before[2] > 16 * 1024**2:
        return cached['sha256']
    value = sha256(path)
    if stamp(path) != before:
        raise ValueError(f'Asset changed while hashing: {path}')
    index[key] = {'stat': before, 'sha256': value,
                  'stable': max(before[3:]) < time.time_ns() - 2_000_000_000}
    return value


def tree_hash(root, index, force=False):
    records = {}
    # Directory links are not followed; lib64 commonly aliases lib. Individual file
    # symlinks include their target content and stat identity. Runtime Python is also
    # covered by the pinned container image.
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {'__pycache__', '.git', '.cache'})
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix == '.pyc' or not path.is_file():
                continue
            records[str(path.relative_to(root))] = indexed_hash(path, index, force)
    if not records:
        raise ValueError(f'Missing or empty asset tree: {root}')
    return digest(records)


def identify_assets(project, run, manifest, cache, force=False):
    data = project / 'data/raw/gemma-4-developer-agent'
    benchmark = read_json(run / 'inputs/benchmark.json')
    if sha256(data / 'tasks.jsonl') != benchmark['source_tasks_sha256']:
        raise ValueError('Original tasks changed; prepare a new benchmark version')
    rows = {t['instance_id']: t for t in read_tasks(data / 'tasks.jsonl')}
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'asset-index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = cache / 'asset-index.json'
        index = read_json(index_path) if index_path.exists() else {}
        started = time.perf_counter()
        print('Checking model, runtime, wheels, and selected task identities...', flush=True)
        image = indexed_hash(project / 'resource/downloads/images/vllm-0.19.1.sif', index, force)
        environment = tree_hash(Path('/home/jwj/venvs/gemma4-eval-v29'), index, force)
        wheels = tree_hash(data / 'wheels', index, force)
        test_wheels = tree_hash(project / 'resource/downloads/eval-test-wheels-v3', index, force)
        model = tree_hash(project / 'model/base/gemma-4-31b-it-qat-w4a16-ct-v2', index, force)
        runtime = manifest['runtime_sha256']
        # Budget can affect verification subprocess timeouts. Include it even though
        # prompts, adapters, and generation choices do not change control validity.
        budget = sha256(run / 'inputs/agent/eval_config.yaml')
        verification = digest({'image': image, 'environment': environment, 'wheels': wheels,
                               'test_wheels': test_wheels,
                               'runtime': runtime, 'budget': budget, 'backend': 'subprocess', 'fast_path': False})
        task_assets = {}
        for task_id in manifest['task_ids']:
            task = rows[task_id]
            stem = task['repo'].split('/')[-1] + '_' + task['base_commit']
            assets = {'snapshot': indexed_hash(data / 'snapshots' / (task_id + '.tgz'), index, force),
                      'graph': indexed_hash(data / 'graphs' / (stem + '.json'), index, force),
                      'embeddings': indexed_hash(data / 'embeddings' / (stem + '.npz'), index, force),
                      'task': digest(task)}
            assets['control_key'] = digest([verification, assets['snapshot'], assets['task']])
            task_assets[task_id] = assets
        write_json(index_path, index)
    return {'verification_identity': verification, 'image_sha256': image, 'environment_sha256': environment,
            'wheels_sha256': wheels, 'test_wheels_sha256': test_wheels, 'model_sha256': model, 'budget_sha256': budget,
            'tasks': task_assets, 'protocol': manifest['protocol'],
            'asset_check_seconds': time.perf_counter() - started,
            'hash_policy': 'SHA-256 of content; reuse indexed hashes only when device/inode/size/mtime/ctime match. '
                           'Use --rehash-assets for a full reread.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--rehash-assets', action='store_true')
    parser.add_argument('--controls-only', action='store_true')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run this Trillium wrapper inside a Slurm allocation')
    project, run = args.project_root.resolve(), args.run_dir.resolve()
    cache = (args.cache_dir or project / 'runs/eval-cache/v1').resolve()
    manifest = verify_inputs(run)
    if manifest.get('suite_version') != 1:
        raise ValueError('Use prepare_eval_suite.py first')
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        timing = {'started_at': datetime.now(timezone.utc).isoformat(), 'job_id': os.environ['SLURM_JOB_ID'],
                  'steps': [], 'control_hits': 0, 'control_misses': 0}
        attempt = str(time.time_ns())
        environment = dict(os.environ, GEMMA_PROJECT_ROOT=str(project))
        runner = run / 'inputs/runtime/run_eval_container.sh'

        def execute(mode, *extra):
            tick = time.perf_counter()
            result = subprocess.run(['bash', str(runner), mode, str(run), *map(str, extra)], env=environment)
            timing['steps'].append({'mode': mode, 'arguments': list(map(str, extra)),
                                    'seconds': time.perf_counter() - tick, 'exit_code': result.returncode})
            write_json(run / 'timings' / f'{attempt}.json', timing)
            if result.returncode:
                raise RuntimeError(f'{mode} failed with status {result.returncode}; inspect saved logs, then resume')

        try:
            assets = identify_assets(project, run, manifest, cache, args.rehash_assets)
            assets_path = run / 'runtime-inputs.json'
            if assets_path.exists():
                previous = read_json(assets_path)
                for key in ('verification_identity', 'model_sha256', 'tasks', 'protocol'):
                    if previous[key] != assets[key]:
                        raise ValueError(f'Runtime inputs changed on resume: {key}; start a new run')
            else:
                write_json(assets_path, assets)
            timing['asset_check_seconds'] = assets['asset_check_seconds']
            if not (run / 'compile/validation.json').exists():
                execute('compile')
            if read_json(run / 'compile/validation.json').get('status') != 'compiled':
                raise ValueError('Official agent compilation did not pass')
            (run / 'infer').mkdir(exist_ok=True)
            controls, missing = {}, []
            for task_id in manifest['task_ids']:
                key = assets['tasks'][task_id]['control_key']
                cache_path = cache / 'controls' / (key + '.json')
                if cache_path.exists():
                    controls[task_id] = read_control_cache(cache_path, key, task_id)
                    timing['control_hits'] += 1
                else:
                    missing.append(task_id)
                    timing['control_misses'] += 1
            for source in ('empty', 'gold'):
                pending = [i for i in missing if not artifact_valid(run / 'verify' / source / i / 'result.json')]
                if pending:
                    execute('verify', '--task-ids', *pending, '--patch-source', source,
                            '--results', '/work/run/' + source)
            for task_id in missing:
                key = assets['tasks'][task_id]['control_key']
                result = cache_record(key, read_json(run / 'verify/empty' / task_id / 'result.json'),
                                      read_json(run / 'verify/gold' / task_id / 'result.json'))
                write_json(cache / 'controls' / (key + '.json'), result)
                controls[task_id] = result['body']
            # Cache data stays outside the inference mount. Store only eligibility in
            # the orchestration report; the agent sees sanitized tasks and its own phase.
            write_json(run / 'controls.json', controls)
            eligible = [i for i in manifest['task_ids'] if controls[i]['valid']]
            print(json.dumps({'controls_valid': len(eligible), 'selected': len(manifest['task_ids']),
                              'cache_hits': timing['control_hits'], 'cache_misses': timing['control_misses']}), flush=True)
            if args.controls_only:
                timing['controls_only'] = True
                return
            if not eligible:
                raise ValueError('No task passed empty/gold controls; fix the runtime before inference')
            pending = [i for i in eligible if not artifact_valid(run / 'infer' / i / 'prediction.json')]
            if pending:
                execute('infer', '--task-ids', *pending)
            grade = []
            for task_id in eligible:
                output = run / 'verify/agent' / task_id / 'result.json'
                if artifact_valid(output):
                    continue
                prediction = read_json(run / 'infer' / task_id / 'prediction.json')
                if prediction['agent_patch'].strip():
                    # Official evaluation can grade a recovered patch even when the
                    # agent reports a timeout/turn error. Track completion separately.
                    grade.append(task_id)
                else:
                    save_artifact(output, {'task_id': task_id, 'resolved': False, 'test_exit_code': None,
                                           'error_message': 'No generated patch', 'test_output': '',
                                           'patch_source': 'agent', 'verification_skipped': 'empty patch'})
            if grade:
                execute('verify', '--task-ids', *grade, '--patch-source', 'agent', '--results', '/work/run/agent')
            command = ['python3', str(run / 'inputs/runtime/summarize_eval_suite.py'), '--run-dir', str(run)]
            if args.baseline:
                command += ['--baseline', str(args.baseline.resolve())]
            subprocess.run(command, check=True)
            timing['completed'] = True
        except Exception as error:
            timing.update(completed=False, error=str(error))
            raise
        finally:
            timing['total_seconds'] = time.perf_counter() - started
            write_json(run / 'timings' / f'{attempt}.json', timing)
            # Include final timing in the report after the timer has been recorded.
            if timing.get('completed'):
                subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
