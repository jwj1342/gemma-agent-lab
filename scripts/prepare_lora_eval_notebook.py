"""Generate the LoRA improvement-programme evaluation notebook.

One private Kaggle L4 kernel that evaluates a budget-fit agent over several control-valid tasks with the
official harness, in batch mode: one vLLM server, one sandbox per task, every produced patch graded by the
official verifier. It exists because the 245 MB adapter cannot travel into a 1 MB notebook, so the
programme's later runs must *train and evaluate inside one session*; this generator produces the evaluation
half of that kernel and is validated on its own first (milestone M1).

What the notebook does, in order:

1. verify the attached 41-wheel wheelhouse against the v31 snapshot digests, then install the reviewed
   harness stack (vLLM 0.19.1, swegemma 0.2.11, adk_submission 0.2.13, transformers 5.13.1) plus torchao
   0.16.0 from its own dataset;
2. unpack the frozen agent tree and the runner scripts from an embedded payload;
3. run `evaluate_agent.py infer --task-ids <ids>` against the mounted competition data, then
   `evaluate_agent.py verify --patch-source agent`, and write one summary JSON.

The evaluation hook (`local_eval_setup.py`) is generated here rather than read from the repository because
the repository version hardcodes the official container's `/data/wheels` and `/eval-test-wheels` mounts and
asserts `swegemma == 0.2.10`; this kernel needs environment-driven wheel paths and accepts 0.2.11, which is
what wheelhouse v31 ships. Nothing in the harness itself is modified.

When `--train-dataset` is given the same kernel trains the variant adapter in this session, and judges the
trained arm from its loss trajectory before evaluating it (`resource/l4_feasibility/collapse_guard.py`): a
flat or collapsed arm writes `collapse-guard.json` and a `skipped` summary instead of spending the task
budget. This is a conservative heuristic based on two observed arms, not a quality guarantee.

Nothing here submits a competition entry.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE_SNAPSHOT = ROOT / 'resource/manifests/kaggle-wheelhouse-snapshots.json'
OFFICIAL_METADATA = ROOT / 'resource/notebooks/official/kernel-metadata.json'
BASE_MODEL_HANDLE = 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'
WHEELHOUSE_DATASET = 'metric/gemma-4-developer-agent-wheelhouse'
TORCHAO_DATASET = 'kajarnaklokossou/torchao-0-16-0'
COMPETITION_SOURCE = 'gemma-4-developer-agent'
WORKER_FILES = ('prepare_model_stack.py', 'quantization_config_compat.py', 'train_l4_device_map.py',
                'train_l4_feasibility.py', 'verify_l4_adapter.py', 'probe_packed_base.py',
                'serving_l4_adapter.py', 'collapse_guard.py')

SETUP_CELL = r'''import base64, gzip, hashlib, io, json, os, shutil, subprocess, sys, tarfile, time, zipfile
from pathlib import Path

PAYLOAD = json.loads(gzip.decompress(base64.b85decode(PAYLOAD_B85)))
WORK = Path('/kaggle/working/lora-eval')
assert not WORK.exists(), 'Runtime output already exists; start a fresh session'
WORK.mkdir(parents=True)
(WORK / 'scripts').mkdir()
(WORK / 'inputs').mkdir()

DATA_CANDIDATES = ['/kaggle/input/competitions/gemma-4-developer-agent',
                   '/kaggle/input/gemma-4-developer-agent',
                   '/kaggle/input/competition-data']
DATA = next((Path(p) for p in DATA_CANDIDATES if (Path(p) / 'tasks.jsonl').is_file()), None)
assert DATA is not None, 'competition data mount not found: ' + repr(sorted(str(p) for p in Path('/kaggle/input').glob('*')))
WHEELHOUSE = next((p for p in (Path('/kaggle/input/gemma-4-developer-agent-wheelhouse'),
                               Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
                               Path('/kaggle/input/metric/gemma-4-developer-agent-wheelhouse')) if p.is_dir()), None)
assert WHEELHOUSE is not None, 'official wheelhouse dataset not mounted'
TORCHAO = next((p for p in (Path('/kaggle/input/torchao-0-16-0'),
                            Path('/kaggle/input/datasets/kajarnaklokossou/torchao-0-16-0')) if p.is_dir()), None)
assert TORCHAO is not None, 'torchao dataset not mounted'
MODEL = Path('@@@MODEL@@@')
assert MODEL.is_dir(), 'competition model v2 not mounted'
print('Mounts:', {'data': str(DATA), 'wheelhouse': str(WHEELHOUSE), 'torchao': str(TORCHAO), 'model': str(MODEL)}, flush=True)

# 1. Verify the attached wheelhouse against the v31 snapshot, then install the reviewed stack.
actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in WHEELHOUSE.glob('*.whl')}
missing = sorted(set(PAYLOAD['wheelhouse']) - set(actual))
extra = sorted(set(actual) - set(PAYLOAD['wheelhouse']))
changed = sorted(name for name in set(actual) & set(PAYLOAD['wheelhouse'])
                 if actual[name] != PAYLOAD['wheelhouse'][name])
assert not (missing or extra or changed), {'missing': missing, 'extra': extra, 'changed': changed}
print('Wheelhouse verified:', len(actual), 'wheels match the v31 snapshot', flush=True)

torchao_wheel = next(iter(sorted(TORCHAO.glob('torchao-0.16.0-*.whl'))))
with (WORK / 'install.log').open('w') as log:
    for wheel in [torchao_wheel, *sorted(WHEELHOUSE.glob('*.whl'))]:
        if 'cutlass' in wheel.name.lower():
            continue
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--force-reinstall', str(wheel)], stdout=log, stderr=subprocess.STDOUT, check=True)
print('Reviewed stack installed', flush=True)

# 2. Unpack the frozen agent and the runner scripts.
raw = base64.b64decode(PAYLOAD['agent_bundle'])
assert hashlib.sha256(raw).hexdigest() == PAYLOAD['agent_bundle_sha256'], 'agent bundle hash changed'
AGENT = WORK / 'inputs' / 'agent'
AGENT.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
    archive.extractall(AGENT)
for name, source in PAYLOAD['scripts'].items():
    (WORK / 'scripts' / name).write_text(source, encoding='utf-8')
print('Agent files:', sorted(p.relative_to(AGENT).as_posix() for p in AGENT.rglob('*') if p.is_file()), flush=True)

# 3. Optional training phase: train the variant adapter in this same session (no adapter transport).
STAGE = WORK / 'eval'
for name in ('home', 'tmp', 'cache'):
    (STAGE / name).mkdir(parents=True)
ENV = os.environ.copy()
ENV.update(HOME=str(STAGE / 'home'), TMPDIR=str(STAGE / 'tmp'), XDG_CACHE_HOME=str(STAGE / 'cache'),
           HF_HOME=str(STAGE / 'cache/hf'), TORCH_HOME=str(STAGE / 'cache/torch'),
           TORCH_EXTENSIONS_DIR=str(STAGE / 'cache/torch-extensions'), TRITON_CACHE_DIR=str(STAGE / 'cache/triton'),
           VLLM_CACHE_ROOT=str(STAGE / 'cache/vllm'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', LITELLM_LOCAL_MODEL_COST_MAP='True',
           OTEL_SDK_DISABLED='true', TRANSFORMERS_NO_TF='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
           VLLM_NO_USAGE_STATS='1', VLLM_ENGINE_READY_TIMEOUT_S='1800', WANDB_MODE='disabled',
           VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           EVAL_WHEELS=str(WHEELHOUSE), TEST_WHEELS=str(DATA / 'wheels'), TASK_WHEELS=str(DATA / 'wheels'))

GUARD = {'verdict': 'proceed', 'reason': 'no_training'}
TRAIN = PAYLOAD.get('training')
if TRAIN:
    WORKERS = WORK / 'workers'
    WORKERS.mkdir()
    for name, blob in TRAIN['workers'].items():
        payload = base64.b64decode(blob)
        assert hashlib.sha256(payload).hexdigest() == TRAIN['worker_digests'][name], 'worker hash changed'
        (WORKERS / name).write_bytes(payload)
    dataset = WORK / 'train/dataset.jsonl'
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(TRAIN['dataset_text'], encoding='utf-8')
    train_work = WORK / 'train'
    command = [sys.executable, str(WORKERS / 'train_l4_device_map.py'), '--base', str(MODEL),
               '--work', str(train_work), '--updates', '8', '--seq-len', str(TRAIN['seq_len']),
               '--rank', str(TRAIN['rank']), '--alpha', str(TRAIN['alpha']), '--batch-size', '1',
               '--max-updates', str(TRAIN['max_updates']), '--learning-rate', str(TRAIN['learning_rate']),
               '--dataset', str(dataset), '--wall-budget-seconds', str(TRAIN['wall_budget']),
               '--load-budget-seconds', str(TRAIN['load_budget'])]
    train_log = WORK / 'train.log'
    started_training = time.monotonic()
    with train_log.open('w') as stream:
        trained = subprocess.run(command, env=ENV, cwd=str(WORK), stdout=stream, stderr=subprocess.STDOUT)
    print(train_log.read_text(errors='replace')[-8000:], flush=True)
    print('training returncode', trained.returncode, 'seconds', round(time.monotonic() - started_training, 1), flush=True)
    result_path = train_work / 'device-map-result.json'
    assert result_path.is_file(), 'training wrote no result'
    trained_result = json.loads(result_path.read_text(encoding='utf-8'))
    assert trained.returncode == 0, 'training process failed; inspect train.log'
    assert not trained_result.get('training_error'), 'training result contains an error'
    assert trained_result.get('adapter'), 'training produced no adapter: ' + str(trained_result.get('verdict'))
    target = AGENT / 'adapters' / TRAIN['adapter_name']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(train_work / 'adapter', target, dirs_exist_ok=True)
    agent_yaml = AGENT / 'agent.yaml'
    text = agent_yaml.read_text(encoding='utf-8')
    if 'adapter:' not in text:
        lines = text.splitlines()
        anchor = next(i for i, line in enumerate(lines) if line.startswith('model:'))
        lines.insert(anchor + 1, 'adapter: ' + TRAIN['adapter_name'])
        agent_yaml.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (WORK / 'training-summary.json').write_text(json.dumps({'adapter': trained_result.get('adapter'),
        'updates': trained_result.get('updates'), 'verdict': trained_result.get('verdict'),
        'seconds': round(time.monotonic() - started_training, 1)}, indent=2, default=str) + '\n')
    print('Adapter mounted for evaluation at', target, flush=True)

    # 3b. Collapse guard: judge the trained arm from its loss trajectory before spending the task budget.
    # Screen for the two observed failure trajectories (programme document, sections 12 and 14): a flat arm
    # reproduces the baseline for about 22,000 quota-seconds, and a collapsed arm costs the timeout on
    # every task (29,935 quota-seconds for 0 of 24). The verdict and the trajectory are always recorded.
    sys.path.insert(0, str(WORKERS))
    from collapse_guard import collapse_guard
    GUARD = collapse_guard(trained_result, TRAIN['collapse_loss_floor'], TRAIN['collapse_flat_ratio'])
    (WORK / 'collapse-guard.json').write_text(json.dumps(GUARD, indent=2, default=str) + '\n')
    print('Collapse guard:', json.dumps(GUARD, default=str), flush=True)

# 4. Batch evaluation: one server, one sandbox per task, official verification of every patch.
def evaluate_arm():
    IDS = PAYLOAD['task_ids']
    frozen = STAGE / 'frozen'
    subprocess.run([sys.executable, str(WORK / 'scripts/prepare_eval_run.py'),
                    '--agent-dir', str(AGENT), '--task-ids', *IDS,
                    '--tasks', str(DATA / 'tasks.jsonl'), '--run-dir', str(frozen)],
                   env=ENV, check=True)
    frozen_agent = frozen / 'inputs/agent'
    public_tasks = frozen / 'inputs/tasks.jsonl'
    manifest = json.loads((frozen / 'run-manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(public_tasks.read_bytes()).hexdigest() == manifest['public_task_sha256']
    assert hashlib.sha256((frozen / 'inputs/submission.zip').read_bytes()).hexdigest() == manifest['agent_archive_sha256']

    def run(step, arguments, log_name=None):
        started = time.monotonic()
        log = STAGE / (log_name or (step + '.log'))
        with log.open('w') as stream:
            completed = subprocess.run([sys.executable, str(WORK / 'scripts' / 'evaluate_agent.py'), step, *arguments],
                                       env=ENV, cwd=str(WORK / 'scripts'), stdout=stream, stderr=subprocess.STDOUT)
        print(log.read_text(errors='replace')[-6000:], flush=True)
        print(step, 'returncode', completed.returncode, 'seconds', round(time.monotonic() - started, 1), flush=True)
        return completed.returncode

    common = ['--task-ids', *IDS, '--data-dir', str(DATA), '--agent-dir', str(frozen_agent)]
    # Revalidate controls on this exact environment before starting the model.
    sys.path.insert(0, str(WORK / 'scripts'))
    from summarize_eval_smoke import control_checks, ordinary_test_failure
    control_codes = {}
    for source in ('empty', 'gold'):
        control_codes[source] = run('verify', [*common, '--tasks', str(DATA / 'tasks.jsonl'),
            '--patch-source', source, '--results', str(STAGE / 'controls' / source)], source + '.log')
    controls = {}
    for task_id in IDS:
        results = {}
        for source in ('empty', 'gold'):
            path = STAGE / 'controls' / source / task_id / 'result.json'
            results[source] = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        empty, gold = results['empty'], results['gold']
        try:
            controls[task_id] = all(control_checks(empty, gold, task_id).values())
        except ValueError:
            controls[task_id] = False
    controls_valid = all(code == 0 for code in control_codes.values()) and all(controls.values())
    (WORK / 'controls-summary.json').write_text(json.dumps(
        {'controls_valid': controls_valid, 'tasks': controls, 'returncodes': control_codes}, indent=2) + '\n')
    # Default behaviour is all-or-nothing: one invalid control stops the whole arm. With
    # PER_TASK_CONTROLS the run measures the subset whose empty/gold controls both behaved, and reports
    # the excluded tasks, which is how the L4 r5 cohort was read (21 of 29 usable).
    PER_TASK_CONTROLS = @@PER_TASK_CONTROLS@@
    if PER_TASK_CONTROLS and all(code == 0 for code in control_codes.values()):
        SELECTED = [task for task in IDS if controls.get(task)]
        EXCLUDED = [task for task in IDS if not controls.get(task)]
    elif controls_valid:
        SELECTED = list(IDS)
        EXCLUDED = []
    else:
        SELECTED = []
        EXCLUDED = list(IDS)
    if not SELECTED:
        summary = {'task_ids': IDS, 'measured_tasks': 0, 'resolved': None,
                   'skipped': 'invalid_controls', 'controls_valid': False}
        (WORK / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    selected_common = ['--task-ids', *SELECTED, '--data-dir', str(DATA), '--agent-dir', str(frozen_agent)]
    infer_code = run('infer', [*selected_common, '--tasks', str(public_tasks), '--results', str(STAGE / 'results'),
                               '--model', str(MODEL), '--tensor-parallel-size', '4',
                               '--gpu-memory-utilization', '0.85'], 'infer.log')
    verify_code = run('verify', [*selected_common, '--tasks', str(DATA / 'tasks.jsonl'), '--patch-source', 'agent',
                                 '--results', str(STAGE / 'verify'),
                                 '--prediction', str(STAGE / 'prediction.json')], 'verify.log')

    summary = {'task_ids': SELECTED, 'requested_task_ids': IDS, 'controls_valid': controls_valid,
               'excluded_invalid_controls': EXCLUDED,
               'infer_returncode': infer_code, 'verify_returncode': verify_code,
               'wall_seconds': round(time.monotonic() - STARTED, 1), 'tasks': {}}
    for task_id in SELECTED:
        record = {'resolved': None, 'error': None, 'tool_calls': None, 'llm_calls': None,
                  'patch_bytes': None, 'test_exit_code': None, 'duration_seconds': None}
        prediction = STAGE / task_id / 'prediction.json'
        if prediction.is_file():
            payload = json.loads(prediction.read_text(encoding='utf-8'))
            record.update({'error': payload.get('error'), 'tool_calls': payload.get('tool_calls'),
                           'llm_calls': payload.get('llm_calls'),
                           'patch_bytes': len((payload.get('agent_patch') or '').encode()),
                           'duration_seconds': payload.get('duration_seconds')})
        verified = STAGE / 'verify' / task_id / 'result.json'
        if verified.is_file():
            payload = json.loads(verified.read_text(encoding='utf-8'))
            record.update({'resolved': payload.get('resolved'), 'test_exit_code': payload.get('test_exit_code'),
                           'verify_error': payload.get('error_message')})
            record['tests_executed'] = ordinary_test_failure(payload) or (
                payload.get('resolved') is True and payload.get('test_exit_code') == 0
                and not payload.get('error_message'))
        failure = STAGE / 'verify' / task_id / 'failure.json'
        if failure.is_file():
            record['verify_failure'] = json.loads(failure.read_text(encoding='utf-8')).get('message')
        infer_failure = STAGE / task_id / 'results' / 'failure.json'
        if infer_failure.is_file():
            record['infer_failure'] = json.loads(infer_failure.read_text(encoding='utf-8')).get('message')
        summary['tasks'][task_id] = record
    summary['resolved'] = sum(1 for r in summary['tasks'].values() if r.get('resolved') is True)
    summary['measured_tasks'] = len(SELECTED)
    summary['pipeline_executed'] = (infer_code == verify_code == 0 and all(
        not r.get('error') and not r.get('infer_failure') and not r.get('verify_failure')
        and r.get('tests_executed') for r in summary['tasks'].values()))
    (WORK / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


if GUARD['verdict'] == 'proceed':
    evaluate_arm()
else:
    # The arm is outside the measured window; its trajectory is the result and the task budget is kept.
    (WORK / 'summary.json').write_text(json.dumps(
        {'task_ids': PAYLOAD['task_ids'], 'measured_tasks': 0, 'resolved': None, 'skipped': GUARD['verdict'],
         'collapse_guard': GUARD,
         'note': 'judged from the training loss before evaluation; no task budget was spent'},
        indent=2, default=str) + '\n')
    print('Collapse guard stopped the arm before evaluation:', json.dumps(GUARD, default=str), flush=True)
print('LoRA arm kernel finished; no competition entry was submitted.', flush=True)
'''


def pack_agent(agent_dir: Path) -> tuple[str, dict]:
    """Pack the agent tree deterministically into a base85 gzip tar and hash it."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for path in sorted(agent_dir.rglob('*')):
            if not path.is_file():
                continue
            payload = path.read_bytes()
            info = tarfile.TarInfo(name=path.relative_to(agent_dir).as_posix())
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    raw = gzip.compress(buffer.getvalue(), mtime=0)
    return base64.b64encode(raw).decode(), {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def hook_source(system_site_packages: bool = False) -> str:
    """Evaluation hook with environment-driven wheel paths (see the module docstring).

    The M1 run measured a hard constraint here: the mounted competition data ships pytest but not every
    package of the official test wheelhouse (for example ``pytest-timeout==2.1.0``), because the official
    container mounts a separate ``/eval-test-wheels`` dataset that a Kaggle kernel cannot attach. A strict
    install therefore fails the whole task. This hook installs what the mounts provide, reports what is
    missing, and continues, so a missing optional plugin shows up as a test result instead of a setup error.
    """
    return '''"""Offline sandbox dependency setup for the Kaggle evaluation kernel."""
from importlib.metadata import version
from pathlib import Path
import json
import os
import subprocess
import sys
import time

COMMON = ['pytest==8.3.4', 'pytest-timeout==2.1.0', 'inline-snapshot==0.21.1',
          'dirty-equals==0.9.0', 'setuptools==82.0.1', 'wheel==0.45.1',
          'hatchling==1.30.1', 'flit-core==3.12.0', 'pdm-backend==2.4.9', 'poetry-core==2.4.1', 'editables==0.6']
REPO_EXTRAS = {
    'fastapi/fastapi': ['pydantic==2.13.4'],
    'Textualize/rich': ['pygments==2.19.2'],
    'psf/requests': ['pytest-httpbin==2.1.0', 'trustme==1.2.1'],
    'encode/httpx': ['pytest-mock==3.16.0', 'pytest-asyncio==0.26.0', 'trio==0.31.0'],
}
SYSTEM_SITE_PACKAGES = @@SYSTEM_SITE_PACKAGES@@
_installed = set()
_hook_installed = False


def _links():
    paths = [os.environ.get('EVAL_WHEELS'), os.environ.get('TEST_WHEELS'), os.environ.get('TASK_WHEELS')]
    return [f'--find-links={p}' for p in paths if p and Path(p).is_dir()]


def install_local_dependencies(manager, sandbox_id, repo='', **kwargs):
    from swegemma.harness.container_setup import ContainerSetupError
    if sandbox_id in _installed:
        return
    paths = manager.sandboxes[sandbox_id]
    python = paths['venv'] / 'bin/python3'
    workspace = paths['workspace']
    log = paths['root'] / 'local-setup.log'
    base = [sys.executable, '-m', 'pip', '--python', str(python), 'install', '--no-index',
            *_links(), '--disable-pip-version-check', '--no-compile']
    started = time.perf_counter()
    outputs, missing = [], []

    def attempt(packages, extra=()):
        command = base + list(extra) + list(packages)
        result = subprocess.run(command, cwd=workspace, text=True, capture_output=True, timeout=300)
        outputs.append(' '.join(packages) + '\\n' + result.stdout + result.stderr)
        log.write_text('\\n'.join(outputs))
        return result.returncode == 0

    # Install one package at a time so an unavailable plugin does not block the whole setup.
    for package in COMMON + REPO_EXTRAS.get(repo, []):
        if not attempt([package]):
            missing.append(package)
    if repo != 'encode/httpx':
        if not attempt(['httpx==0.28.1']):
            missing.append('httpx==0.28.1')
    editable = attempt([], extra=['--no-build-isolation', '-e', str(workspace)])
    if not editable:
        raise ContainerSetupError('Workspace editable install failed: ' + outputs[-1][-4000:])
    if missing:
        print(f'Local sandbox deps for {repo}: missing from mounted wheels {missing}', flush=True)
    package = repo.split('/')[-1]
    probe = subprocess.run([str(python), '-c',
                            'import importlib.util,json; s=importlib.util.find_spec(' + repr(package) + '); '
                            'print(json.dumps({"origin":s.origin if s else None}))'],
                           cwd=workspace, capture_output=True, text=True)
    try:
        origin = json.loads(probe.stdout or '{}').get('origin')
    except json.JSONDecodeError:
        origin = None
    if not origin or not Path(origin).resolve().is_relative_to(workspace.resolve()):
        raise ContainerSetupError('Task package would be imported outside its workspace: ' + str(origin))
    _installed.add(sandbox_id)
    print(f'Local sandbox dependencies ready for {repo}: {time.perf_counter() - started:.2f}s', flush=True)


def configure(evaluator):
    global _hook_installed
    if version('swegemma') not in {'0.2.10', '0.2.11'}:
        raise RuntimeError('Review the local dependency hook before changing the official harness version')
    evaluator.sandbox.system_site_packages = SYSTEM_SITE_PACKAGES
    if not _hook_installed:
        import swegemma.harness.agent_runner as agent_runner
        import swegemma.harness.verification as verification
        agent_runner.install_test_dependencies = install_local_dependencies
        verification.install_test_dependencies = install_local_dependencies
        _hook_installed = True
'''.replace('@@SYSTEM_SITE_PACKAGES@@', repr(bool(system_site_packages)))


def wheelhouse_digests(snapshot: Path) -> dict:
    data = json.loads(snapshot.read_text(encoding='utf-8'))
    latest = data['snapshots'][-1]
    if not latest.get('dataset_version'):
        raise SystemExit('the snapshot file has no versioned entry')
    return dict(latest['wheels'])


def pack_workers() -> tuple[dict, dict]:
    """Embed the four-L4 training workers so the same kernel can train before it evaluates."""
    blobs, digests = {}, {}
    for name in WORKER_FILES:
        payload = (ROOT / 'resource/l4_feasibility' / name).read_bytes()
        blobs[name] = base64.b64encode(payload).decode()
        digests[name] = hashlib.sha256(payload).hexdigest()
    return blobs, digests


def build_notebook(args) -> tuple[dict, dict]:
    bundle, bundle_meta = pack_agent(args.agent_dir)
    scripts = {
        'evaluate_agent.py': Path(args.evaluate_agent).read_text(encoding='utf-8'),
        'local_eval_setup.py': hook_source(args.site_packages),
    }
    for name in ('prepare_eval_run.py', 'prepare_eval_task.py', 'build_submission.py',
                 'summarize_eval_smoke.py'):
        scripts[name] = (ROOT / 'scripts' / name).read_text(encoding='utf-8')
    payload = {'wheelhouse': wheelhouse_digests(args.snapshot), 'agent_bundle': bundle,
               'agent_bundle_sha256': bundle_meta['sha256'], 'task_ids': args.task_ids,
               'scripts': scripts, 'model': BASE_MODEL_HANDLE, 'training': None}
    if args.train_dataset:
        workers, digests = pack_workers()
        payload['training'] = {
            'dataset_text': Path(args.train_dataset).read_text(encoding='utf-8'),
            'seq_len': args.seq_len, 'rank': args.rank, 'alpha': args.alpha,
            'max_updates': args.max_updates, 'learning_rate': args.learning_rate,
            'adapter_name': args.adapter_name, 'wall_budget': args.train_wall_budget,
            'load_budget': args.train_load_budget, 'workers': workers, 'worker_digests': digests,
            'collapse_loss_floor': args.collapse_loss_floor,
            'collapse_flat_ratio': args.collapse_flat_ratio,
        }
    blob = base64.b85encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    setup = SETUP_CELL.replace('@@@MODEL@@@', '/kaggle/input/models/' + BASE_MODEL_HANDLE.lower())
    setup = setup.replace('@@PER_TASK_CONTROLS@@', repr(bool(args.per_task_controls)))
    cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# LoRA programme evaluation kernel\n\n',
            'Private, offline, no competition submission. Official harness, batch inference, official '
            'verification of every produced patch.\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': [f'PAYLOAD_B85 = {blob!r}\n', 'import time\n', 'STARTED = time.monotonic()\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [setup]},
    ]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                           'name': 'python3'},
                                            'language_info': {'name': 'python', 'version': '3.12'}},
                'nbformat': 4, 'nbformat_minor': 5}
    official = json.loads(OFFICIAL_METADATA.read_text(encoding='utf-8'))
    metadata = {'id': f'{args.owner}/{args.slug}', 'title': args.title, 'code_file': 'evaluation.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
                'enable_tpu': False, 'enable_internet': False, 'machine_shape': 'NvidiaL4',
                'docker_image': official['dockerImage'], 'docker_image_pinning_type': 'original',
                'competition_sources': [COMPETITION_SOURCE],
                'dataset_sources': [WHEELHOUSE_DATASET, TORCHAO_DATASET],
                'kernel_sources': [], 'model_sources': [BASE_MODEL_HANDLE]}
    return notebook, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--agent-dir', type=Path, required=True, help='Budget-fit agent tree to freeze')
    parser.add_argument('--task-ids', nargs='+', required=True)
    parser.add_argument('--evaluate-agent', type=Path,
                        default=ROOT / 'scripts/evaluate_agent.py',
                        help='Batch evaluator source: main-branch evaluate_agent.py (--task-ids support)')
    parser.add_argument('--snapshot', type=Path, default=WHEELHOUSE_SNAPSHOT)
    parser.add_argument('--train-dataset', type=Path, default=None,
                        help='Optional prompt/completion JSONL; when given the kernel trains this adapter '
                             'before evaluating, so no adapter has to cross the notebook size limit')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--max-updates', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--adapter-name', default='main_lora')
    parser.add_argument('--train-wall-budget', type=float, default=900.0)
    parser.add_argument('--train-load-budget', type=float, default=900.0)
    parser.add_argument('--collapse-loss-floor', type=float, default=1.0,
                        help='Stop the arm before evaluation when the minimum per-update training loss '
                             'falls below this value (measured collapsed arm: 0.73, flat arm: 2.10); '
                             '0 disables the check')
    parser.add_argument('--collapse-flat-ratio', type=float, default=0.005,
                        help='Stop the arm before evaluation when the loss does not move by this '
                             'fraction of its first value (measured flat arm: 1.1e-3); 0 disables the '
                             'check')
    parser.add_argument('--site-packages', action='store_true',
                        help='Let each sandbox fall back to the image site-packages. Needed for repositories '
                             'whose dependency set is not fully present in the mounted wheels (measured: '
                             'fastapi needs annotated-doc, which only the image provides)')
    parser.add_argument('--per-task-controls', action='store_true',
                        help='Measure the tasks whose empty/gold controls both behave and report the '
                             'excluded ones, instead of stopping the whole arm when any control is invalid '
                             '(the 2026-11-02 cohort had 5 of 10 stale: the October manifest no longer '
                             'predicts control validity). Default keeps the all-or-nothing gate.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} already exists; use a fresh --output path')
    notebook, metadata = build_notebook(args)
    encoded = (json.dumps(notebook, indent=1) + '\n').encode('utf-8')
    if len(encoded) > 1_000_000:
        raise SystemExit('notebook exceeds the 1 MB source limit')
    args.output.mkdir(parents=True)
    (args.output / 'evaluation.ipynb').write_bytes(encoded)
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'notebook': str(args.output / 'evaluation.ipynb'),
                      'bytes': (args.output / 'evaluation.ipynb').stat().st_size,
                      'tasks': args.task_ids}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
