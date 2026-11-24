"""Prepare a CPU-only mined-task smoke through the pinned official verifier.

Snapshots and test wheels are rebuilt online, then sandbox dependency installation is offline.
No inference, model weights, credentials, or competition submission are involved.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from build_submission import build

ROOT = Path(__file__).resolve().parents[1]
CELL = r'''
import asyncio, base64, gzip, hashlib, io, json, os, subprocess, sys, time, zipfile
from importlib.metadata import version
from pathlib import Path

P = json.loads(gzip.decompress(base64.b85decode(PAYLOAD)))
WORK = Path('/kaggle/working/mined-verifier')
WORK.mkdir(exist_ok=False)
SCRIPTS = WORK / 'scripts'
SCRIPTS.mkdir()
for name, text in P['scripts'].items():
    assert hashlib.sha256(text.encode()).hexdigest() == P['script_sha256'][name]
    (SCRIPTS / name).write_bytes(text.encode())
sys.path.insert(0, str(SCRIPTS))
data = WORK / 'data'
data.mkdir()
tasks_path = data / 'tasks.jsonl'
tasks_path.write_bytes(''.join(json.dumps(row) + '\n' for row in P['tasks']).encode())
wheelhouse = next(path for path in Path('/kaggle/input').rglob('*')
                  if path.is_dir() and len(list(path.glob('swegemma-*.whl'))) == 1)
actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in wheelhouse.glob('*.whl')}
assert actual == P['wheelhouse'], 'Wheelhouse differs from the pinned snapshot'
with (WORK / 'install.log').open('w') as log:
    for wheel in sorted(wheelhouse.glob('*.whl')):
        if any(name in wheel.name.lower() for name in ('vllm', 'bitsandbytes', 'cutlass')):
            continue
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--force-reinstall', str(wheel)], stdout=log, stderr=log, check=True)
from build_mined_eval_bundle import fetch_snapshot, download_wheels, REQUIREMENTS, SANDBOX_BOOTSTRAP
snapshots = data / 'snapshots'
snapshots.mkdir()
wheels = data / 'wheels'
requirements = set(SANDBOX_BOOTSTRAP)
for task in P['tasks']:
    fetch_snapshot(task['repo'], task['base_commit'], snapshots / (task['instance_id'] + '.tgz'))
    requirements.update(req for req in REQUIREMENTS[task['repo']] if not req.startswith('pytest'))
download_wheels(sorted(requirements), wheels)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(P['agent_zip']))) as archive:
    archive.extractall(WORK / 'agent')
for task in P['tasks']:
    subprocess.run([sys.executable, str(SCRIPTS / 'prepare_eval_run.py'), '--run-dir',
                    str(WORK / task['instance_id']), '--agent-dir', str(WORK / 'agent'),
                    '--task-id', task['instance_id'], '--tasks', str(tasks_path)], check=True)
receipt = {'preparation_revision': P['git_revision'], 'script_sha256': P['script_sha256'],
           'tasks_sha256': hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
           'snapshots': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshots.iterdir()},
           'wheels': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in wheels.iterdir()},
           'wheelhouse': actual, 'model_execution': False, 'python': sys.version}
(WORK / 'inputs.json').write_text(json.dumps(receipt, indent=2))

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.models import load_tasks
from swegemma.sandbox import SubprocessManager
import swegemma.harness.verification as verification
from swegemma.harness.container_setup import ContainerSetupError
from summarize_eval_smoke import control_checks

def offline_dependencies(manager, sandbox_id, repo='', **kwargs):
    paths = manager.sandboxes[sandbox_id]
    python = paths['venv'] / 'bin/python3'
    workspace = paths['workspace']
    command = [sys.executable, '-m', 'pip', '--python', str(python), 'install', '--no-index',
               '--find-links=' + str(wheels), '--no-build-isolation', *SANDBOX_BOOTSTRAP,
               *(req for req in REQUIREMENTS.get(repo, []) if not req.startswith('pytest'))]
    done = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if done.returncode:
        raise ContainerSetupError(done.stdout + done.stderr)
    editable = command[:command.index('--no-build-isolation') + 1] + ['--no-deps', '-e', str(workspace)]
    done = subprocess.run(editable, capture_output=True, text=True, timeout=180)
    if done.returncode:
        raise ContainerSetupError(done.stdout + done.stderr)
    module = repo.split('/')[-1].replace('-', '_')
    probe = subprocess.run([str(python), '-c', 'import importlib.util;print(importlib.util.find_spec('
                            + repr(module) + ').origin)'], cwd=workspace, capture_output=True, text=True)
    if probe.returncode or not Path(probe.stdout.strip()).resolve().is_relative_to(workspace.resolve()):
        raise ContainerSetupError('Workspace import provenance failed: ' + probe.stdout + probe.stderr)

assert version('swegemma') == '0.2.11'
verification.install_test_dependencies = offline_dependencies
async def smoke():
    rows = []
    for task in load_tasks(tasks_path):
        run_dir = WORK / task.instance_id
        subprocess.run([sys.executable, str(SCRIPTS / 'verify_eval_run.py'), '--run-dir', str(run_dir)], check=True)
        results = {}
        for side in ('empty', 'gold'):
            output = run_dir / 'verify' / side
            output.mkdir(parents=True)
            config = EvalConfig(tasks_path=tasks_path, snapshots_dir=snapshots, results_dir=output,
                                submission_dir=run_dir / 'inputs/agent', models=ModelRegistry(),
                                sandbox='subprocess', wheels_dir=wheels, timeout_seconds=120)
            manager = SubprocessManager(timeout_seconds=120, system_site_packages=False)
            result = await verification.verify_task(manager, config, task,
                         snapshots / (task.instance_id + '.tgz'), fast_path=False,
                         agent_patch=task.patch if side == 'gold' else '', start_time=time.perf_counter())
            record = result.model_dump(mode='json', exclude={'trace'})
            record['task_id'] = task.instance_id
            record['error_message'] = record.get('error_message') or record.get('error')
            (output / 'result.json').write_text(json.dumps(record, indent=2))
            (output / 'tests.log').write_text(record.get('test_output', ''))
            results[side] = record
        checks = control_checks(results['empty'], results['gold'], task.instance_id)
        row = {'task_id': task.instance_id, 'checks': checks, 'valid': all(checks.values()),
               'empty_rc': results['empty'].get('test_exit_code'),
               'gold_rc': results['gold'].get('test_exit_code')}
        rows.append(row)
        print('VERIFIER_RESULT ' + json.dumps(row), flush=True)
        for side, record in results.items():
            print('VERIFIER_DETAIL ' + json.dumps({'task_id': task.instance_id, 'side': side, **record}), flush=True)
    (WORK / 'summary.json').write_text(json.dumps(rows, indent=2))
    print('VERIFIER_SUMMARY ' + json.dumps(rows), flush=True)
await smoke()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--snapshot', type=Path,
                        default=ROOT / 'resource/manifests/mined-verifier-runtime-20261124.json')
    parser.add_argument('--agent-dir', type=Path, default=ROOT / 'configs/baseline')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--owner', default='junzhejiao')
    parser.add_argument('--slug', default='mined-verifier-smoke-20261124')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = [row for row in rows if row['instance_id'] in set(args.ids)]
    if len(selected) != len(set(args.ids)) or args.out.exists():
        raise ValueError('Missing/duplicate IDs or output directory already exists')
    args.out.mkdir(parents=True)
    archive = args.out / 'agent.zip'
    build(args.agent_dir, archive)
    names = ['build_mined_eval_bundle.py', 'prepare_eval_task.py', 'prepare_eval_run.py',
             'build_submission.py', 'verify_eval_run.py', 'summarize_eval_smoke.py']
    scripts = {name: (ROOT / 'scripts' / name).read_text(encoding='utf-8') for name in names}
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))['snapshots'][-1]
    if snapshot.get('dataset_version') != 31:
        raise ValueError('This verifier smoke requires the reviewed v31 snapshot')
    payload = {'tasks': selected, 'scripts': scripts,
               'script_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in scripts.items()},
               'wheelhouse': snapshot['wheels'], 'agent_zip': base64.b64encode(archive.read_bytes()).decode(),
               'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    blob = base64.b85encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    code = CELL.replace('PAYLOAD', repr(blob))
    compile(code, '<mined-verifier-notebook>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    notebook = {'cells': [{'cell_type': 'code', 'id': 'official-verifier-smoke', 'metadata': {},
                          'execution_count': None, 'outputs': [], 'source': [code]}],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    raw = (json.dumps(notebook) + '\n').encode()
    if len(raw) > 1_000_000:
        raise ValueError('Notebook exceeds the 1 MB source limit')
    (args.out / 'verifier.ipynb').write_bytes(raw)
    metadata = {'id': args.owner + '/' + args.slug, 'title': args.slug.replace('-', ' '),
                'code_file': 'verifier.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'],
                'competition_sources': [], 'model_sources': [], 'kernel_sources': []}
    metadata['docker_image'] = json.loads((ROOT / 'resource/notebooks/official/kernel-metadata.json').read_text())['dockerImage']
    (args.out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'tasks': args.ids, 'bytes': len(raw), 'kernel': metadata['id']}))


if __name__ == '__main__':
    main()
