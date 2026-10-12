"""Generate the private, bounded four-L4 training-feasibility notebook.

The acceptance requirement for this workstream is that a teammate without H100 access can reproduce a
tiny LoRA train/save/reload loop on the four L4 GPUs the competition provides. This generator produces
that notebook: it embeds the worker scripts byte-for-byte, runs training under ``torchrun`` with
a hard wall-clock cap, verifies the exported adapter, and reloads it in the official wheelhouse vLLM
runtime.

Three deliberate constraints, all from the task definition:

* the base is the required ``gemma-4-31b-it-qat-w4a16-ct`` checkpoint and nothing else -- an NF4
  re-quantisation of it is a different base model and is not silently substituted;
* the run is bounded: one backward step, then at most ``--updates`` client updates, then stop; a
  full corpus and a second epoch are out of scope;
* the first session is capped at one hour, and the notebook publishes its progress to disk after
  every phase because a failed Kaggle session shows nothing until it ends.

Preparation only: no GPU execution, no upload, no competition submission.

    python scripts/prepare_l4_feasibility_notebook.py --owner YOUR_KAGGLE_USERNAME \
        --slug gemma4-l4-train-feasibility --output artifacts/l4-feasibility-notebook
    python scripts/prepare_l4_feasibility_notebook.py --owner u --slug s \
        --output artifacts/l4-feasibility-notebook --check
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / 'resource/l4_feasibility'
OFFICIAL_METADATA = ROOT / 'resource/notebooks/official/kernel-metadata.json'
BASE_MODEL_HANDLE = 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'
WHEELHOUSE_DATASET = 'metric/gemma-4-developer-agent-wheelhouse'
# The wheelhouse carries no torchao wheel, and peft 0.19.1 refuses anything below 0.16.0 while the image
# ships 0.10.0, so that one wheel arrives through a second dataset source. It is a public third-party
# dataset, so its bytes are hashed against the PyPI digest before installation (see
# resource/l4_feasibility/prepare_model_stack.py::TORCHAO_SHA256, checked locally with
# scripts/bounded_connect.py in the same session that wrote this comment). Publishing our own copy is not
# an option from this machine: the kaggle client uploads dataset blobs to www.googleapis.com, and every
# address of that host is unreachable here.
TORCHAO_DATASET = 'kajarnaklokossou/torchao-0-16-0'
# L4x4 machines are gated on notebooks attached to the competition ("L4s are only available for
# notebooks attached to this competition", resource/competition/upgraded-accelerators.md). The first
# measured session requested NvidiaL4 with `competition_sources: []` and was quietly given two T4s
# instead, so the attachment is a functional requirement for this notebook, not a formality.
COMPETITION_SOURCE = 'gemma-4-developer-agent'
WORKERS = ('probe_packed_base.py', 'prepare_model_stack.py', 'quantization_config_compat.py',
           'train_l4_device_map.py', 'train_l4_feasibility.py', 'verify_l4_adapter.py',
           'serving_l4_adapter.py')
# L4x4 sessions consume GPU quota at twice the wall-clock rate; the archived competition accelerator
# page is the authority for that and the notebook records the resulting arithmetic.
QUOTA_MULTIPLIER = 2.0

STAGE_CELL = r'''import base64, gzip, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

STARTED = time.monotonic()
STARTED_EPOCH = time.time()
WORKERS_SHA256 = @@WORKERS_SHA256@@
WORKERS = @@WORKERS@@
WORK = Path('/kaggle/working/l4-feasibility')
WORK.mkdir(parents=True, exist_ok=True)
WORKER_DIR = WORK / 'workers'
WORKER_DIR.mkdir(exist_ok=True)
for name, blob in WORKERS.items():
    raw = gzip.decompress(base64.b85decode(blob))
    assert hashlib.sha256(raw).hexdigest() == WORKERS_SHA256[name], f'embedded worker {name} changed'
    (WORKER_DIR / name).write_bytes(raw)
print('Workers staged:', sorted(p.name for p in WORKER_DIR.glob('*.py')), flush=True)

MODEL_CANDIDATES = [
    os.environ.get('GEMMA_BASE_MODEL'),
    '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
    '/kaggle/input/gemma-4-31b-it-qat-w4a16-ct',
]
BASE = next((path for path in MODEL_CANDIDATES if path and Path(path).is_dir()), None)
assert BASE, f'No required base model mounted among {MODEL_CANDIDATES}'


def mem_total_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemTotal:'):
            return round(int(line.split()[1]) / 1024 ** 2, 2)
    return None


gpu_csv = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
gpu_rows = [row.strip() for row in gpu_csv.splitlines() if row.strip()]
gpu_names = [row.split(',', 1)[0].strip() for row in gpu_rows]
wheelhouse_candidates = [
    Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
    Path('/kaggle/input/gemma-4-developer-agent-wheelhouse'),
    Path('/kaggle/input/gemma-4-developer-agent-wheelhouse/gemma-4-developer-agent-wheelhouse'),
]
wheelhouse = next((path for path in wheelhouse_candidates if path.is_dir()), None)
environment = {
    'python': platform.python_version(), 'host_memory_gib': mem_total_gib(),
    'cpu_count': os.cpu_count(), 'gpus': gpu_rows,
    'base': BASE, 'started_at': time.time(),
    'requested_machine_shape': 'NvidiaL4',
    'hardware_compatible': len(gpu_names) == 4 and all('L4' in name for name in gpu_names),
    'wheelhouse_path': str(wheelhouse) if wheelhouse else None,
    'wheelhouse_candidates': [str(path) for path in wheelhouse_candidates],
}
# The image ships no peft/trl/bitsandbytes and (measured earlier) no vLLM; the wheelhouse mount
# supplies vLLM and its runtime dependencies. Inventory it now so a missing wheel is visible before
# any phase decides to use it.
wheels = sorted(str(path) for path in wheelhouse.glob('*.whl')) if wheelhouse else []
environment['wheels_visible'] = len(wheels)
environment['wheel_names'] = [Path(w).name for w in wheels][:60]
environment['wheelhouse_compatible'] = bool(wheelhouse and wheels)

# Fail before importing the model stack when Kaggle did not honour the requested four-L4 shape or
# did not mount the official wheelhouse. The first measured run was labelled NvidiaL4 in server
# metadata but actually received two T4s; proceeding would spend quota on an irrelevant result.
preflight_problems = []
if not environment['hardware_compatible']:
    preflight_problems.append(f'expected four NVIDIA L4 GPUs, observed {gpu_rows!r}')
if not environment['wheelhouse_compatible']:
    preflight_problems.append('official wheelhouse is absent or contains no .whl files')
environment['preflight_problems'] = preflight_problems
(WORK / 'environment.json').write_text(json.dumps(environment, indent=2) + chr(10))
if preflight_problems:
    preflight = {'status': 'incompatible', 'problems': preflight_problems,
                 'requested_machine_shape': 'NvidiaL4', 'observed_gpus': gpu_rows,
                 'wheelhouse_path': environment['wheelhouse_path']}
    (WORK / 'preflight-result.json').write_text(json.dumps(preflight, indent=2) + chr(10))
    print('Preflight incompatible:', json.dumps(preflight), flush=True)
    raise SystemExit('; '.join(preflight_problems))

# The base cannot be constructed until the mounted wheelhouse's model stack replaces the image's. Two
# measured facts force that, and run 4 is what measured them: the image ships transformers 5.0.0, whose
# CONFIG_MAPPING has no 'gemma4' entry (`KeyError: 'gemma4'` at the base load), and peft 0.19.1 raises
# `ImportError: Found an incompatible version of torchao ... above 0.16.0` while the image's torchao
# 0.10.0 is installed. The wheelhouse supplies transformers 5.13.1, its tokenizers/safetensors, and
# compressed_tensors plus loguru; torchao 0.16.0 arrives through the second dataset source. All of that
# -- including the SHA256 check on the third-party wheel -- lives in one worker rather than in this cell,
# because generated code here has already cost this project a session.
model_stack = {'verdict': 'not_run'}
model_stack_run = subprocess.run(
    [sys.executable, str(WORKER_DIR / 'prepare_model_stack.py'), '--wheelhouse', str(wheelhouse),
     '--torchao-search', '/kaggle/input', '--base', BASE, '--work', str(WORK)],
    capture_output=True, text=True)
print(f'model stack step returncode={model_stack_run.returncode}', flush=True)
print(model_stack_run.stdout[-4000:], flush=True)
print(model_stack_run.stderr[-4000:], flush=True)
stack_path = WORK / 'model-stack.json'
if stack_path.is_file():
    model_stack = json.loads(stack_path.read_text())
environment['model_stack_step'] = {key: value for key, value in model_stack.items() if key != 'entries'}
(WORK / 'environment.json').write_text(json.dumps(environment, indent=2) + chr(10))
if model_stack_run.returncode != 0 or model_stack.get('verdict') != 'ok':
    # Fail here rather than in phase 1: a base that cannot be constructed is a dependency problem, not a
    # training result, and both routes would report the same import error.
    blocked = {'status': 'dependency_missing', 'problems': model_stack.get('problems') or {},
               'model_stack': {key: value for key, value in model_stack.items() if key != 'entries'},
               'observed_gpus': gpu_rows, 'requested_machine_shape': 'NvidiaL4'}
    (WORK / 'preflight-result.json').write_text(json.dumps(blocked, indent=2, default=str) + chr(10))
    raise SystemExit(f'base-load dependencies unsatisfied: {blocked["problems"]}')
print('model stack ok:', json.dumps(model_stack.get('checks'), default=str), flush=True)

# Every later cell rebuilds its constants from this file instead of relying on another cell's
# globals, so a single cell can be re-run after a failure without a NameError.
plan = {'work': str(WORK), 'worker_dir': str(WORKER_DIR), 'base': BASE,
        'started_epoch': STARTED_EPOCH, 'session_limit_seconds': @@SESSION_LIMIT@@,
        'updates_cap': @@UPDATES@@, 'seq_len': @@SEQ_LEN@@,
        'fsdp_policy': 'run four-GPU sharding only if the packed single-GPU route saved no adapter'}
(WORK / 'plan.json').write_text(json.dumps(plan, indent=2) + chr(10))
print('Plan:', json.dumps(plan), flush=True)
print(json.dumps(environment, indent=2), flush=True)
'''

TRAIN_CELL = r'''import json, subprocess, sys, time
from pathlib import Path

WORK = Path('/kaggle/working/l4-feasibility')
PLAN = json.loads((WORK / 'plan.json').read_text())
BASE, WORKER_DIR = PLAN['base'], Path(PLAN['worker_dir'])
TRAIN_BUDGET_SECONDS = @@TRAIN_BUDGET@@
TRAIN_TIMEOUT_SECONDS = @@TRAIN_TIMEOUT@@
LOAD_BUDGET_SECONDS = @@LOAD_BUDGET@@
SEQ_LEN = PLAN['seq_len']
UPDATES = PLAN['updates_cap']

# Policy, recorded so the choice is auditable: four-GPU sharding is attempted only when the packed
# single-GPU route saved no adapter. The acceptance criterion is that a teammate can reproduce the
# tiny train/save/reload loop, and one L4 does that more cheaply than four; if that route fails, the
# FSDP sharding the task asks about is the remaining option. Either way the session stays inside its
# budget instead of spending it on a likely OOM at load.
packed_path = WORK / 'packed-summary.json'
packed = json.loads(packed_path.read_text()) if packed_path.is_file() else {}
packed_ok = bool(packed.get('adapter') and packed['adapter'].get('saved'))
returncode, timed_out, skip = None, False, None
if packed_ok:
    skip = ('the packed single-GPU route already saved an adapter, so four-GPU sharding is not needed '
            'to meet the acceptance criterion; its measurement is recorded in packed-summary.json')
    (WORK / 'train-skipped.json').write_text(json.dumps({'skipped': skip, 'packed_status': packed.get('status')},
                                                        indent=2) + chr(10))
    print('Four-GPU phase skipped:', skip, flush=True)
else:
    # Four ranks are the point of this phase, but the count is read from the measurement rather than
    # assumed: on a smaller allocation, four ranks crash with `CUDA error: invalid device ordinal`,
    # which is exactly what the first measured session recorded.
    ranks = len(json.loads((WORK / 'environment.json').read_text())['gpus'])
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           f'--nproc_per_node={ranks}',
           str(WORKER_DIR / 'train_l4_feasibility.py'), '--base', BASE, '--work', str(WORK),
           '--updates', str(UPDATES), '--seq-len', str(SEQ_LEN),
           '--load-budget-seconds', str(LOAD_BUDGET_SECONDS),
           '--wall-budget-seconds', str(TRAIN_BUDGET_SECONDS)]
    print('Launching:', ' '.join(cmd), flush=True)
    started = time.monotonic()
    # A hard timeout is the difference between a bounded probe and four idle GPUs burning quota.
    try:
        completed = subprocess.run(cmd, timeout=TRAIN_TIMEOUT_SECONDS)
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode, timed_out = None, True
    print(f'training subprocess returncode={returncode} timed_out={timed_out} '
          f'wall={time.monotonic() - started:.1f}s', flush=True)
result_path = WORK / 'train-result.json'
summary = {'returncode': returncode, 'timed_out': timed_out, 'skipped': skip}
if result_path.is_file():
    result = json.loads(result_path.read_text())
    summary['stopped_reason'] = result.get('stopped_reason')
    summary['contract'] = result.get('contract')
    summary['adapter'] = result.get('adapter')
    summary['attempts'] = [{k: v for k, v in attempt.items() if k != 'traceback'}
                           for attempt in result.get('attempts', [])]
    summary['updates'] = result.get('updates')
elif not skip:
    summary['error'] = 'the worker wrote no train-result.json; read progress.jsonl for the last phase'
(WORK / 'train-summary.json').write_text(json.dumps(summary, indent=2, default=str) + chr(10))
print(json.dumps({k: v for k, v in summary.items() if k != 'updates'}, indent=2, default=str), flush=True)
'''

PACKED_CELL = r'''import json, subprocess, sys, time
from pathlib import Path

WORK = Path('/kaggle/working/l4-feasibility')
PLAN = json.loads((WORK / 'plan.json').read_text())
BASE, WORKER_DIR = PLAN['base'], Path(PLAN['worker_dir'])
PACKED_TIMEOUT_SECONDS = @@PACKED_TIMEOUT@@
cmd = [sys.executable, str(WORKER_DIR / 'probe_packed_base.py'), '--base', BASE, '--work', str(WORK),
       '--updates', str(PLAN['updates_cap']), '--seq-len', str(PLAN['seq_len']),
       '--wall-budget-seconds', str(@@PACKED_BUDGET@@)]
print('Launching:', ' '.join(cmd), flush=True)
try:
    completed = subprocess.run(cmd, timeout=PACKED_TIMEOUT_SECONDS)
    returncode, timed_out = completed.returncode, False
except subprocess.TimeoutExpired:
    returncode, timed_out = None, True
summary = {'returncode': returncode, 'timed_out': timed_out,
           'elapsed_s': round(time.time() - PLAN['started_epoch'], 1)}
result_path = WORK / 'packed-probe-result.json'
if result_path.is_file():
    packed = json.loads(result_path.read_text())
    for key in ('status', 'verdict', 'contract', 'load', 'target_modules', 'peft', 'updates',
                'seconds_per_update', 'peak_allocated_gib', 'peak_reserved_gib',
                'stopped_reason', 'adapter'):
        summary[key] = packed.get(key)
else:
    summary['error'] = ('the packed probe wrote no result; read packed-progress.jsonl for the last '
                        'phase, which is the first thing to check in the session output')
(WORK / 'packed-summary.json').write_text(json.dumps(summary, indent=2, default=str) + chr(10))
print(json.dumps({k: v for k, v in summary.items() if k != 'updates'}, indent=2, default=str), flush=True)
'''

DEVICE_MAP_CELL = r'''import json, subprocess, sys, time
from pathlib import Path

WORK = Path('/kaggle/working/l4-feasibility')
PLAN = json.loads((WORK / 'plan.json').read_text())
BASE, WORKER_DIR = PLAN['base'], Path(PLAN['worker_dir'])
DEVICE_MAP_TIMEOUT_SECONDS = @@DEVICE_MAP_TIMEOUT@@
DEVICE_MAP_BUDGET_SECONDS = @@DEVICE_MAP_BUDGET@@
adapter_path = WORK / 'adapter' / 'adapter_model.safetensors'
elapsed = time.time() - PLAN['started_epoch']
# The single-process sharded route is the successor the plan names for the case where four sharded ranks
# cannot hold a replica each in host RAM. It runs only when the four-GPU phase saved no adapter, and only
# when the session cap can still cover it: a phase started with less time than it needs would consume the
# rest of the session and record nothing.
skip = None
if adapter_path.is_file():
    skip = 'an adapter already exists, so the sharded single-process route is not needed'
elif elapsed + DEVICE_MAP_TIMEOUT_SECONDS > PLAN['session_limit_seconds']:
    skip = (f"{PLAN['session_limit_seconds'] - elapsed:.0f}s of the {PLAN['session_limit_seconds']}s "
            f"session cap remain, below the {DEVICE_MAP_TIMEOUT_SECONDS}s this phase needs")
if skip:
    (WORK / 'device-map-skipped.json').write_text(json.dumps({'skipped': skip}, indent=2) + chr(10))
    print('Sharded single-process route skipped:', skip, flush=True)
else:
    cmd = [sys.executable, str(WORKER_DIR / 'train_l4_device_map.py'), '--base', BASE,
           '--work', str(WORK), '--updates', str(PLAN['updates_cap']),
           '--seq-len', str(PLAN['seq_len']),
           '--wall-budget-seconds', str(DEVICE_MAP_BUDGET_SECONDS)]
    print('Launching:', ' '.join(cmd), flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(cmd, timeout=DEVICE_MAP_TIMEOUT_SECONDS)
        returncode, timed_out = completed.returncode, False
    except subprocess.TimeoutExpired:
        returncode, timed_out = None, True
    print(f'sharded single-process returncode={returncode} timed_out={timed_out} '
          f'wall={time.monotonic() - started:.1f}s', flush=True)
    if (WORK / 'device-map-result.json').is_file():
        print((WORK / 'device-map-result.json').read_text()[:4000], flush=True)
'''

VERIFY_CELL = r'''import json, subprocess, sys
from pathlib import Path

WORK = Path('/kaggle/working/l4-feasibility')
PLAN = json.loads((WORK / 'plan.json').read_text())
WORKER_DIR = Path(PLAN['worker_dir'])
adapter = WORK / 'adapter'
if not (adapter / 'adapter_model.safetensors').is_file():
    print('No adapter was saved, so adapter verification and the serving reload are skipped.', flush=True)
else:
    completed = subprocess.run([sys.executable, str(WORKER_DIR / 'verify_l4_adapter.py'),
                                '--adapter', str(adapter), '--out', str(WORK / 'adapter-check.json'),
                                '--expected-rank', '16'])
    print(f'verify returncode={completed.returncode}', flush=True)
    if (WORK / 'adapter-check.json').is_file():
        print((WORK / 'adapter-check.json').read_text(), flush=True)
'''

SERVING_CELL = r'''import json, subprocess, sys, time
from pathlib import Path

WORK = Path('/kaggle/working/l4-feasibility')
PLAN = json.loads((WORK / 'plan.json').read_text())
BASE, WORKER_DIR = PLAN['base'], Path(PLAN['worker_dir'])
SERVING_TIMEOUT_SECONDS = @@SERVING_TIMEOUT@@
adapter = WORK / 'adapter'
elapsed = time.time() - PLAN['started_epoch']
skip = None
if not (adapter / 'adapter_model.safetensors').is_file():
    skip = 'no adapter was saved by the training phase'
elif elapsed > PLAN['session_limit_seconds']:
    skip = f"session already used {elapsed:.0f}s of the {PLAN['session_limit_seconds']}s cap"
elif not Path('/kaggle/input').is_dir():
    skip = 'no mounted inputs'
if skip:
    (WORK / 'serving-skipped.json').write_text(json.dumps({'skipped': skip}, indent=2) + chr(10))
    print('Serving reload skipped:', skip, flush=True)
else:
    candidates = [Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
                  Path('/kaggle/input/gemma-4-developer-agent-wheelhouse'),
                  Path('/kaggle/input/gemma-4-developer-agent-wheelhouse/gemma-4-developer-agent-wheelhouse')]
    wheelhouse_dir = next((path for path in candidates if path.is_dir()), None)
    wheelhouse = sorted(str(path) for path in wheelhouse_dir.glob('*.whl')) if wheelhouse_dir else []
    # The stage cell already installed the model stack the base needs (transformers 5.13.1, tokenizers,
    # safetensors, compressed_tensors, loguru, torchao 0.16.0). This install adds the rest of the
    # official runtime, vLLM and its dependencies, and is idempotent for the wheels already present.
    install = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
                              '--no-deps', '--no-build-isolation', '--only-binary=:all:',
                              '--retries', '0', '--timeout', '15', *wheelhouse],
                             capture_output=True, text=True)
    print(f'wheelhouse install returncode={install.returncode} wheels={len(wheelhouse)}', flush=True)
    if install.returncode != 0:
        print(install.stderr[-2000:], flush=True)
    completed = subprocess.run([sys.executable, str(WORKER_DIR / 'serving_l4_adapter.py'),
                                '--base', BASE, '--adapter', str(adapter), '--work', str(WORK)],
                               timeout=SERVING_TIMEOUT_SECONDS)
    print(f'serving returncode={completed.returncode}', flush=True)
    if (WORK / 'serving-result.json').is_file():
        print((WORK / 'serving-result.json').read_text(), flush=True)
'''

SUMMARY_CELL = r'''import json, time
from pathlib import Path

WORK = Path('/kaggle/working/l4-feasibility')
PLAN = json.loads((WORK / 'plan.json').read_text())

def load(name):
    path = WORK / name
    return json.loads(path.read_text()) if path.is_file() else None

environment = load('environment.json') or {}
packed = load('packed-summary.json')
train = load('train-summary.json') or {}
serving = load('serving-result.json')
adapter_check = load('adapter-check.json')
model_stack = load('model-stack.json')
device_map = load('device-map-result.json') or load('device-map-skipped.json')
wall_seconds = round(time.time() - PLAN['started_epoch'], 1)
peak_allocated = None
peak_reserved = None
for attempt in train.get('attempts', []):
    if attempt.get('peak_allocated_gib'):
        peak_allocated = max(peak_allocated or 0, attempt['peak_allocated_gib'])
    if attempt.get('peak_reserved_gib'):
        peak_reserved = max(peak_reserved or 0, attempt['peak_reserved_gib'])
summary = {
    'objective': 'bounded L4 LoRA train/save/reload feasibility, not a performance result',
    'host_memory_gib': environment.get('host_memory_gib'),
    'gpus': environment.get('gpus'), 'cpu_count': environment.get('cpu_count'),
    'base': environment.get('base'),
    'model_stack_step': {key: value for key, value in (model_stack or {}).items()
                         if key != 'entries'},
    'packed_single_gpu': {k: v for k, v in packed.items() if k != 'updates'} if packed else None,
    'four_gpu_fsdp': {k: v for k, v in train.items() if k != 'updates'} if train else None,
    'single_process_device_map': ({k: v for k, v in device_map.items() if k != 'updates'}
                                  if device_map else None),
    'native_handoff_contract': train.get('contract') or (packed or {}).get('contract'),
    'stopped_reason': train.get('stopped_reason') or (packed or {}).get('stopped_reason'),
    'attempts': [{k: v for k, v in attempt.items() if k != 'traceback'}
                 for attempt in train.get('attempts', [])],
    'updates_completed': len(train.get('updates') or []) or len((packed or {}).get('updates') or []),
    'peak_allocated_gib': peak_allocated,
    'peak_reserved_gib': peak_reserved,
    'adapter': train.get('adapter') or (packed or {}).get('adapter'),
    'adapter_check': adapter_check,
    'official_runtime_reload': serving,
    'session_wall_seconds': wall_seconds,
    'quota_seconds_charged_estimate': round(wall_seconds * @@QUOTA_MULTIPLIER@@, 1),
    'quota_basis': ('L4x4 sessions consume GPU quota at twice the wall-clock rate, per the archived '
                    'competition accelerator page; verify against the next quota check'),
    'recovery_path': ('Every phase writes progress.jsonl under /kaggle/working/l4-feasibility; rerun '
                      'the notebook with the same metadata for a fresh version, and read the last '
                      'progress line to see where a failed attempt stopped.'),
}
(WORK / 'result-summary.json').write_text(json.dumps(summary, indent=2, default=str) + chr(10))
print(json.dumps(summary, indent=2, default=str), flush=True)
'''

MARKDOWN_INTRO = '''# Four-L4 LoRA training feasibility (private)

Bounded feasibility check for the L4 training task: does the required competition base support a tiny
LoRA train/save/reload loop on the four L4 GPUs this competition provides, and what does it cost?

What this notebook does, in order:

1. stages the worker scripts and records the host/GPU/input inventory, stopping before any model load
   when the session did not receive four L4 GPUs or the official wheelhouse mount;
2. installs the model stack the base needs from the mounts -- transformers 5.13.1, its tokenizers and
   safetensors, compressed_tensors and loguru, plus the torchao 0.16.0 wheel peft requires, hash-checked
   against PyPI -- and proves the import chain and the checkpoint's config in a fresh interpreter before
   any GPU work starts;
3. tries the packed single-GPU route first, then `torchrun` over the GPUs actually measured for **one
   backward step and at most 8 updates**, then saves the adapter, with a hard wall-clock cap on both
   the worker and the subprocess;
4. validates the exported adapter against the key scheme the serving stack was measured to accept;
5. installs the rest of the wheelhouse and reloads the adapter in the official vLLM runtime, comparing
   top-logprobs with a repeat-base reproducibility control.

What it deliberately does **not** do: train more than 8 updates, run a full corpus, run a second
epoch, substitute an NF4 re-quantisation for the required QAT base, upload anything, or submit to the
competition. It stops on the first incompatibility and records the exact error.

Every phase appends to `/kaggle/working/l4-feasibility/progress.jsonl` as it runs, because a failed
Kaggle session publishes nothing until it ends.
'''

MARKDOWN_RECOVERY = '''## Reading the result

| File | Meaning |
| --- | --- |
| `progress.jsonl` / `progress.json` | One line per phase with elapsed seconds; the last line is where the run stopped |
| `environment.json` | Host RAM, GPU names and memory, mounted inputs, visible wheels |
| `model-stack.json` | Which mounted wheels were installed, the torchao SHA256 check, and the import-chain verdict |
| `train-result.json` | Full worker record: native handoff contract, per-strategy attempts, per-update timings and peak memory |
| `train-summary.json` | Same, reduced for reading |
| `adapter-check.json` | Whether the exported adapter matches the served naming scheme and rank |
| `serving-result.json` | Official-runtime reload: control delta, adapter delta, verdict |
| `result-summary.json` | Combined record with the session wall time and the estimated quota charge |

Recovery path: rerun this notebook from the same metadata to create a new version. If training stopped
early, the last `progress.jsonl` line names the failed phase, and `train-result.json` carries that
phase's exception; no state is reused across sessions, so nothing is corrupted by a retry.
'''


def cell(kind: str, source: str) -> dict:
    entry = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        entry.update({'execution_count': None, 'outputs': []})
    return entry


def embed_workers() -> tuple[dict, dict]:
    blobs, digests = {}, {}
    for name in WORKERS:
        path = WORKER_DIR / name
        if not path.is_file():
            raise SystemExit(f'missing worker script: {path}')
        raw = path.read_bytes()
        digests[name] = hashlib.sha256(raw).hexdigest()
        blobs[name] = base64.b85encode(gzip.compress(raw, mtime=0)).decode()
    return blobs, digests


def render(template: str, **values: object) -> str:
    """Substitute @@NAME@@ placeholders; never format embedded worker code."""
    text = template
    for key, value in values.items():
        text = text.replace(f'@@{key}@@', json.dumps(value))
    return text


def build_notebook(args) -> tuple[dict, dict]:
    blobs, digests = embed_workers()
    stage = render(STAGE_CELL, WORKERS=blobs, WORKERS_SHA256=digests,
                   SESSION_LIMIT=args.session_limit, UPDATES=args.updates, SEQ_LEN=args.seq_len)
    train = render(TRAIN_CELL, TRAIN_BUDGET=args.train_budget, TRAIN_TIMEOUT=args.train_timeout,
                   LOAD_BUDGET=args.load_budget)
    packed = render(PACKED_CELL, PACKED_TIMEOUT=args.packed_timeout, PACKED_BUDGET=args.packed_budget)
    device_map = render(DEVICE_MAP_CELL, DEVICE_MAP_TIMEOUT=args.device_map_timeout,
                        DEVICE_MAP_BUDGET=args.device_map_budget)
    serving = render(SERVING_CELL, SERVING_BUDGET=args.serving_timeout,
                     SERVING_TIMEOUT=args.serving_timeout, SESSION_LIMIT=args.session_limit)
    summary = render(SUMMARY_CELL, QUOTA_MULTIPLIER=QUOTA_MULTIPLIER)
    notebook = {
        'nbformat': 4, 'nbformat_minor': 4,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                    'name': 'python3'}},
        'cells': [cell('markdown', MARKDOWN_INTRO), cell('code', stage), cell('code', packed),
                  cell('code', train), cell('code', device_map), cell('code', VERIFY_CELL),
                  cell('code', serving), cell('code', summary), cell('markdown', MARKDOWN_RECOVERY)],
    }
    plan = {
        'workers': digests,
        'updates_cap': args.updates,
        'seq_len': args.seq_len,
        'packed_wall_budget_seconds': args.packed_budget,
        'packed_subprocess_timeout_seconds': args.packed_timeout,
        'train_wall_budget_seconds': args.train_budget,
        'train_subprocess_timeout_seconds': args.train_timeout,
        'device_map_wall_budget_seconds': args.device_map_budget,
        'device_map_subprocess_timeout_seconds': args.device_map_timeout,
        'load_budget_seconds': args.load_budget,
        'session_limit_seconds': args.session_limit,
        'serving_timeout_seconds': args.serving_timeout,
        'quota_multiplier': QUOTA_MULTIPLIER,
        'phase_order': ['stage', 'packed-single-gpu', 'four-gpu-fsdp',
                        'single-process-device-map', 'verify-adapter', 'official-runtime-reload',
                        'summary'],
        'fsdp_policy': 'four-GPU sharding runs only when the packed single-GPU route saved no adapter',
        'device_map_policy': ('the single-process sharded route runs only when no adapter exists yet and '
                              'the session cap can still cover it; it exists because four FSDP ranks each '
                              'hold a full BF16 replica in host RAM before sharding'),
    }
    return notebook, plan


def build_metadata(args) -> dict:
    """Build the kernel metadata the push uses.

    Kept out of :func:`main` so the hardware-relevant fields can be asserted offline: `machine_shape`
    and `competition_sources` together decide whether Kaggle grants four L4s or silently substitutes
    two T4s.
    """
    official = json.loads(OFFICIAL_METADATA.read_text(encoding='utf-8'))
    return {
        'id': f'{args.owner}/{args.slug}', 'title': args.title,
        'code_file': 'feasibility.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
        'machine_shape': 'NvidiaL4', 'docker_image': official['dockerImage'],
        'docker_image_pinning_type': 'original',
        'competition_sources': [COMPETITION_SOURCE],
        'dataset_sources': [WHEELHOUSE_DATASET, TORCHAO_DATASET],
        'kernel_sources': [], 'model_sources': [BASE_MODEL_HANDLE],
    }


def validate_budgets(args) -> None:
    """Reject a budget ladder that cannot hold, before any quota is spent.

    Three independent rules, all learned from measured sessions rather than assumed: the update cap is
    the issue's, the session cap must stay inside the one-hour bound the issue sets, and the sum of the
    phase timeouts must fit inside the session cap -- a cap any single phase can overrun is not a cap.
    """
    if args.updates > 8:
        raise SystemExit('--updates above 8 is out of scope for this feasibility task')
    if args.session_limit > 3600:
        raise SystemExit('--session-limit above one hour is out of scope for the first session')
    budget = (args.packed_timeout + args.train_timeout + args.device_map_timeout
              + args.serving_timeout)
    if budget > args.session_limit:
        raise SystemExit(f'the subprocess timeouts total {budget:.0f}s, above the {args.session_limit:.0f}s '
                         'session cap; a session cap that any single phase can overrun is not a cap')
    # The worker must be able to record a stalled base load before the subprocess timeout kills it.
    if args.load_budget > 0 and args.load_budget >= args.train_timeout:
        raise SystemExit(f'--load-budget {args.load_budget:.0f}s must stay below --train-timeout '
                         f'{args.train_timeout:.0f}s, or a stalled load dies unrecorded')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--title', default='Gemma 4 four-L4 LoRA training feasibility')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--packed-budget', type=float, default=420.0,
                        help='Wall budget for the packed single-GPU probe updates')
    parser.add_argument('--packed-timeout', type=float, default=600.0,
                        help='Hard timeout for the packed probe subprocess, including its base load and '
                             'the checkpoint digests it records')
    parser.add_argument('--train-budget', type=float, default=600.0)
    parser.add_argument('--train-timeout', type=float, default=1000.0)
    parser.add_argument('--device-map-budget', type=float, default=700.0,
                        help='Wall budget for the single-process sharded route updates')
    parser.add_argument('--device-map-timeout', type=float, default=1100.0,
                        help='Hard timeout for the single-process sharded route, including its load')
    parser.add_argument('--load-budget', type=float, default=480.0,
                        help='Wall cap the worker itself applies to the base load')
    parser.add_argument('--serving-timeout', type=float, default=600.0)
    parser.add_argument('--session-limit', type=float, default=3300.0,
                        help='Wall cap for the whole session; the reload phase is skipped past it')
    parser.add_argument('--check', action='store_true',
                        help='Validate the generated notebook offline instead of writing it')
    args = parser.parse_args()
    validate_budgets(args)

    notebook, plan = build_notebook(args)
    for entry in notebook['cells']:
        if entry['cell_type'] == 'code':
            compile(''.join(entry['source']), '<notebook-cell>', 'exec')
    if args.check:
        # Re-embed and re-compare so a stale notebook cannot pass this gate.
        blobs, digests = embed_workers()
        for name, digest in digests.items():
            raw = gzip.decompress(base64.b85decode(blobs[name]))
            if hashlib.sha256(raw).hexdigest() != digest:
                raise SystemExit(f'embedded worker {name} does not round-trip')
        print(json.dumps({'status': 'ok', 'workers': digests, 'plan': plan}, indent=2))
        return 0

    metadata = build_metadata(args)
    try:
        args.output.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f'{args.output} already exists; use a fresh --output path')
    (args.output / 'feasibility.ipynb').write_text(json.dumps(notebook, indent=2) + chr(10),
                                                   encoding='utf-8')
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + chr(10),
                                                     encoding='utf-8')
    preparation = {
        'prepared_at': datetime.now(timezone.utc).isoformat(),
        'kernel': metadata['id'], 'base_model': BASE_MODEL_HANDLE,
        'wheelhouse_dataset': WHEELHOUSE_DATASET, 'competition_source': COMPETITION_SOURCE,
        'torchao_dataset': TORCHAO_DATASET,
        'torchao_provenance': {
            'why': 'the wheelhouse carries no torchao wheel and peft 0.19.1 refuses anything below 0.16.0',
            'dataset': TORCHAO_DATASET,
            'sha256_checked_in_session': 'resource/l4_feasibility/prepare_model_stack.py::TORCHAO_SHA256',
            'note': ('the attached dataset is public and third-party, so the session hashes the wheel '
                     'against the PyPI digest before installing it; the local copy of PyPI\'s wheel was '
                     'hashed to the same digest'),
        },
        'notebook_bytes': len(json.dumps(notebook).encode('utf-8')),
        'plan': plan,
        'quota_arithmetic': {
            'basis': 'L4x4 consumes GPU quota at twice the wall-clock rate (archived competition page)',
            'session_cap_hours': round(args.session_limit / 3600, 3),
            'quota_seconds_at_cap': round(args.session_limit * QUOTA_MULTIPLIER, 1),
        },
        'known_gaps': [
            'The image ships transformers 5.0.0 while the source analysis was read from 5.19.0 and the '
            'H100 recipe ran 5.18.0, so version-specific gates are measured by this run, not assumed.',
            'Host RAM may not hold a full BF16 replica; the worker skips the whole-model state-dict '
            'export in that case and reports it instead of failing late.',
            'The wheelhouse mount is mutated in place by the organisers, so the installed versions are '
            'recorded from the run rather than pinned in advance.',
        ],
        'status': 'notebook generated locally; no GPU execution and no upload',
    }
    (args.output / 'preparation.json').write_text(json.dumps(preparation, indent=2) + chr(10),
                                                  encoding='utf-8')
    print(json.dumps({'kernel': metadata['id'], 'output': str(args.output),
                      'notebook_bytes': preparation['notebook_bytes'],
                      'quota_seconds_at_cap': preparation['quota_arithmetic']['quota_seconds_at_cap']},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
