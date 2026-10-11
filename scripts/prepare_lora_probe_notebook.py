"""Generate a private, offline notebook that decides whether LoRA is applied at serving time.

Why this exists: the competition serves exactly one base, ``gemma-4-31b-it-qat-w4a16-ct``
(compressed-tensors W4A16), and LoRA on that stack has a documented history of failing **quietly** --
the adapter loads, the server reports success, and the output is byte-identical to the base model
(discussion 743508, upstream issue #50059). A trained adapter that is silently dropped scores exactly
like the base model while consuming the entire training and evaluation budget.

``scripts/make_lora_probe.py`` builds a deliberately loud adapter whose random ``lora_A`` *and*
``lora_B`` must move the logits. This generator embeds that adapter in a notebook that loads the
competition base directly with vLLM and compares greedy logprobs between the base model name and the
adapter name on one fixed prompt:

* logprobs differ -> the serving stack applies LoRA, so training is worth funding;
* logprobs identical -> LoRA is inert here, and the entire adapter line should be retired first.

It needs no training run, no competition submission, and no local model weights: the adapter is
embedded as gzip+base85 (it is small), and the base model arrives from the Kaggle model mount.

    python scripts/prepare_lora_probe_notebook.py --adapter-dir artifacts/lora-probe \\
        --owner <KAGGLE_USER> --slug gemma4-lora-serving-probe --output artifacts/lora-probe-notebook
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_METADATA = ROOT / 'resource/notebooks/official/kernel-metadata.json'
WHEELHOUSE_MANIFEST = ROOT / 'resource/manifests/eval-wheelhouse-2026-10-09.json'
RUNTIME_ADAPTER_TEMPLATE = ROOT / 'resource/lora_probe/runtime_adapter.py'
BASE_MODEL_HANDLE = 'google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'
MODEL_SOURCE = 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'
MODEL_PATH = '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'
WHEELHOUSE_SOURCE = 'metric/gemma-4-developer-agent-wheelhouse'
PROMPT = 'Count from 2 to 29, listing only the prime numbers, separated by commas.'
# Mirrors the patched build the scorer uses; see resource/manifests/eval-wheelhouse-2026-10-09.json.
VLLM_VERSION = '0.19.1'


def cell(kind: str, source: str) -> dict:
    return {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)} if kind == 'markdown' \
        else {'cell_type': kind, 'metadata': {}, 'execution_count': None, 'outputs': [],
              'source': source.splitlines(keepends=True)}


def safetensors_summary(path: Path) -> dict:
    """Read a safetensors header without torch or safetensors: names, dtypes, shapes."""
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length).decode('utf-8'))
    if '__metadata__' in header:
        raise SystemExit(f'{path} carries __metadata__; this tool expects a plain adapter export')
    tensors = {}
    for name, entry in header.items():
        if not isinstance(entry, dict) or 'shape' not in entry or 'data_offsets' not in entry:
            raise SystemExit(f'{path} has an unexpected header entry for {name!r}')
        tensors[name] = {'dtype': entry['dtype'], 'shape': entry['shape']}
    if not tensors:
        raise SystemExit(f'{path} contains no tensors')
    return tensors


def read_adapter(adapter_dir: Path) -> dict:
    config_path = adapter_dir / 'adapter_config.json'
    weights_path = adapter_dir / 'adapter_model.safetensors'
    for required in (config_path, weights_path):
        if not required.is_file():
            raise SystemExit(f'adapter directory is missing {required.name}: {adapter_dir}')
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(config.get('r'), int) or config['r'] <= 0:
        raise SystemExit('adapter_config.json must carry a positive integer r')
    targets = config.get('target_modules')
    if not isinstance(targets, list) or not targets:
        raise SystemExit('adapter_config.json must carry a non-empty target_modules list')
    tensors = safetensors_summary(weights_path)
    config_raw = config_path.read_bytes()
    weights_raw = weights_path.read_bytes()
    return {
        'config': config,
        'targets': targets,
        'tensors': tensors,
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'weights_sha256': hashlib.sha256(weights_raw).hexdigest(),
        'weights_bytes': len(weights_raw),
        'blob_b85': base64.b85encode(gzip.compress(
            json.dumps({'adapter_config.json': base64.b64encode(config_raw).decode(),
                        'adapter_model.safetensors': base64.b64encode(weights_raw).decode()},
                       sort_keys=True).encode(), mtime=0)).decode(),
    }


def build_setup(adapter: dict, model_path: str) -> str:
    return f'''import base64, gzip, hashlib, json, sys
from pathlib import Path

ADAPTER_B85 = {adapter['blob_b85']!r}
ADAPTER_CONFIG_SHA256 = {adapter['config_sha256']!r}
ADAPTER_WEIGHTS_SHA256 = {adapter['weights_sha256']!r}
ADAPTER_WEIGHTS_BYTES = {adapter['weights_bytes']!r}
ADAPTER_NAME = 'main_lora'
MODEL_PATH = Path({model_path!r})
PROBE = {{'prompt': {PROMPT!r}, 'temperature': 0, 'max_tokens': 40, 'logprobs': 3,
          'tensor_parallel_size': 4, 'gpu_memory_utilization': 0.90, 'max_model_len': 4096,
          'enable_lora': True, 'max_loras': 1, 'max_lora_rank': 128, 'enforce_eager': True}}

WORK = Path('/kaggle/working/lora-serving-probe') if Path('/kaggle/working').is_dir() \\
    else Path.cwd() / 'lora-serving-probe'
assert not (WORK / 'probe-result.json').exists(), \\
    'probe-result.json already exists; export it and start a fresh session before rerunning'
ADAPTER = WORK / 'adapter' / ADAPTER_NAME
ADAPTER.mkdir(parents=True, exist_ok=True)

# The adapter is embedded rather than mounted so this probe needs no dataset publication, and its
# identity is proven by hash before vLLM ever sees it.
decoded = json.loads(gzip.decompress(base64.b85decode(ADAPTER_B85)).decode('utf-8'))
for name, expected_sha, expected_bytes in (
        ('adapter_config.json', ADAPTER_CONFIG_SHA256, None),
        ('adapter_model.safetensors', ADAPTER_WEIGHTS_SHA256, ADAPTER_WEIGHTS_BYTES)):
    raw = base64.b64decode(decoded[name])
    assert hashlib.sha256(raw).hexdigest() == expected_sha, f'embedded {{name}} hash mismatch'
    if expected_bytes is not None:
        assert len(raw) == expected_bytes, f'embedded {{name}} size mismatch'
    (ADAPTER / name).write_bytes(raw)
config = json.loads((ADAPTER / 'adapter_config.json').read_text(encoding='utf-8'))
assert config['r'] > 0 and config['target_modules'], 'embedded adapter config is not a usable PEFT LoRA'
print('Adapter staged:', ADAPTER, 'rank', config['r'], 'targets', config['target_modules'],
      'bytes', ADAPTER_WEIGHTS_BYTES, flush=True)

# A missing mount is reported here and asserted in the probe cell, so the staging step can be
# verified offline without the competition inputs present.
inputs = {{str(path): path.is_dir() for path in (MODEL_PATH, Path('/kaggle/input'))}}
print('Input mounts:', inputs, flush=True)
(WORK / 'probe-config.json').write_text(json.dumps(
    dict(PROBE, adapter_name=ADAPTER_NAME, adapter_dir=str(ADAPTER),
         adapter_config=config, adapter_weights_sha256=ADAPTER_WEIGHTS_SHA256,
         adapter_weights_bytes=ADAPTER_WEIGHTS_BYTES,
         model_path=str(MODEL_PATH), model_handle={MODEL_SOURCE!r},
         vllm_version={VLLM_VERSION!r}), indent=2) + '\\n', encoding='utf-8')
'''


def build_install(manifest: dict) -> str:
    hashes = {item['path']: item['sha256'] for item in manifest['wheelhouse_files']}
    # vLLM and its runtime dependencies are the only wheels this probe cannot run without. The
    # organizers republished the wheelhouse in place on 2026-10-09 (swegemma 0.2.10 -> 0.2.11), so a
    # missing non-critical wheel is recorded as drift instead of aborting the run.
    required = ['vllm', 'transformers', 'compressed_tensors', 'xgrammar', 'flashinfer_python',
                'flashinfer_cubin', 'bitsandbytes', 'tokenizers', 'safetensors']
    return f'''import glob, json, subprocess, sys, time, zipfile
from pathlib import Path

# The starter and official images do not ship vLLM: a first attempt to run this probe failed at
# `ModuleNotFoundError: No module named 'vllm'`. The patched build is not on PyPI either, so the only
# offline source is the wheelhouse mount. Every wheel's identity is checked against the reviewed
# manifest before installation.
WHEELHOUSE_HASHES = {hashes!r}
REQUIRED_PACKAGES = {required!r}
WORK = Path('/kaggle/working/lora-serving-probe') if Path('/kaggle/working').is_dir() \\
    else Path.cwd() / 'lora-serving-probe'
PROGRESS = WORK / 'progress.json'
STARTED = time.time()


def stage(name, **fields):
    """Record each phase on disk: Kaggle only publishes a cell's output when it finishes, so a stalled
    cell leaves no log at all. This file is the only way to see where a long run actually stopped."""
    record = json.loads(PROGRESS.read_text(encoding='utf-8')) if PROGRESS.is_file() else {{}}
    record[name] = dict(fields, at_seconds=round(time.time() - STARTED, 1))
    PROGRESS.write_text(json.dumps(record, indent=2) + '\\n', encoding='utf-8')
    print('[' + name + ']', json.dumps(fields), flush=True)


candidates = [Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
              Path('/kaggle/input/gemma-4-developer-agent-wheelhouse'),
              Path('/kaggle/input/gemma-4-developer-agent-wheelhouse/gemma-4-developer-agent-wheelhouse')]
WHEELS = next((path for path in candidates if path.is_dir()), None)
assert WHEELS is not None, 'Wheelhouse mount missing; attach ' + {WHEELHOUSE_SOURCE!r} + ' to this kernel.'

import hashlib
mounted = {{path.name: path for path in sorted(WHEELS.glob('*.whl'))}}
missing = sorted(set(WHEELHOUSE_HASHES) - set(mounted))
mismatched = [name for name, expected in WHEELHOUSE_HASHES.items()
              if name in mounted and hashlib.sha256(mounted[name].read_bytes()).hexdigest() != expected]
absent_required = [package for package in REQUIRED_PACKAGES
                   if not any(name.startswith(package + '-') for name in mounted)]
assert not absent_required, f'Wheelhouse is missing required packages: {{absent_required}}'
stage('inventory', wheelhouse=str(WHEELS), mounted=len(mounted), absent=missing,
      hash_mismatches=mismatched, absent_required=absent_required)

# Mirror the reviewed runtime path: drop the cutlass .pth shims, then install every mounted wheel
# offline with --no-deps so nothing reaches for the network. --no-build-isolation and
# --only-binary=:all: keep pip from resolving or building anything, which is what an offline session
# cannot afford to wait on.
for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for shim in glob.glob(pattern):
        Path(shim).unlink(missing_ok=True)
normalized = Path('/tmp/gemma4-probe-wheels')
normalized.mkdir(exist_ok=True)
packages = []
for name in sorted(mounted):
    source = mounted[name]
    if 'cutlass' in source.name.lower():
        continue
    with zipfile.ZipFile(source) as archive:
        metadata = next(entry for entry in archive.namelist() if entry.endswith('.dist-info/METADATA'))
        version = next(line[9:] for line in archive.read(metadata).decode().splitlines()
                       if line.startswith('Version: '))
    parts = source.name.split('-')
    parts[1] = version
    target = normalized / '-'.join(parts)
    if not target.exists():
        target.symlink_to(source)
    packages.append(str(target))
stage('install-start', packages=len(packages),
      total_bytes=sum(mounted[name].stat().st_size for name in mounted if 'cutlass' not in name))
with (WORK / 'installation.log').open('w') as log:
    completed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                                '--no-build-isolation', '--only-binary=:all:', '--retries', '0',
                                '--timeout', '15', '--force-reinstall', *packages],
                               stdout=log, stderr=subprocess.STDOUT)
stage('install-done', returncode=completed.returncode,
      log_tail=(WORK / 'installation.log').read_text(errors='replace')[-600:])
assert completed.returncode == 0, 'Wheelhouse installation failed; see progress.json and installation.log'

# Prove the import in a fresh interpreter so an installed-but-unimportable vLLM cannot pass silently.
probe = subprocess.run(
    [sys.executable, '-c', 'import json, vllm; print(json.dumps({{"vllm": vllm.__version__}}))'],
    capture_output=True, text=True)
stage('import-check', returncode=probe.returncode, stdout=probe.stdout.strip()[-300:],
      stderr=probe.stderr.strip()[-800:])
(WORK / 'install-result.json').write_text(json.dumps({{
    'wheelhouse': str(WHEELS), 'mounted_wheels': len(mounted),
    'reviewed_wheels': len(WHEELHOUSE_HASHES),
    'absent_reviewed_wheels': missing, 'hash_mismatches': mismatched,
    'absent_required_packages': absent_required,
    'mounted_names': sorted(mounted), 'installed': len(packages),
    'pip_returncode': completed.returncode,
    'pip_log_tail': (WORK / 'installation.log').read_text(errors='replace')[-1500:],
    'import_returncode': probe.returncode,
    'import_stdout': probe.stdout.strip()[-500:], 'import_stderr': probe.stderr.strip()[-2000:],
}}, indent=2) + '\\n', encoding='utf-8')
print('Wheelhouse:', WHEELS, '| mounted', len(mounted), '| absent', missing,
      '| hash mismatches', mismatched, flush=True)
print('vLLM import:', probe.returncode, probe.stdout.strip()[-200:], flush=True)
assert probe.returncode == 0, 'vLLM is not importable after installing the wheelhouse; see install-result.json'
'''


def build_runtime_adapter(rank: int, targets: list[str]) -> str:
    """Render the runtime wide-probe module from its template.

    An all-layer rank-32 adapter is hundreds of megabytes and cannot be embedded in a 1 MB notebook, so
    it is generated inside the session with PEFT -- the same library that will export a trained adapter
    -- and its tensor keys and shapes are recorded. The template lives in
    `resource/lora_probe/runtime_adapter.py` so it can be compiled and tested on its own; the seed and
    the A/B scales match the verified embedded probe, because a weaker runtime adapter would look like
    an inert serving path.
    """
    template = RUNTIME_ADAPTER_TEMPLATE.read_text(encoding='utf-8')
    substitutions = {
        '__RANK__': repr(rank), '__TARGETS__': repr(list(targets)), '__SEED__': '1234',
        '__A_STD__': '0.02', '__B_STD__': '1.0',
        # Recorded in resource/manifests/sample-adapters.json: the only verified shapes for this base.
        '__EXPECTED_Q__': '(5376, 8192)', '__EXPECTED_O__': '(8192, 5376)',
    }
    for token, value in substitutions.items():
        if token not in template:
            raise SystemExit('runtime adapter template is missing ' + token)
        template = template.replace(token, value)
    compile(template, 'runtime_adapter_template', 'exec')
    return template


def build_probe(runtime_adapter_source: str) -> str:
    return '''import json, logging, os, time
from pathlib import Path

RUNTIME_ADAPTER_SOURCE = ''' + repr(runtime_adapter_source) + '''

# This cell re-derives everything it needs from the files the staging cell wrote, so the two cells
# stay independent rather than relying on shared notebook globals.
WORK = Path('/kaggle/working/lora-serving-probe') if Path('/kaggle/working').is_dir() \\
    else Path.cwd() / 'lora-serving-probe'
config_path = WORK / 'probe-config.json'
if not config_path.is_file():
    raise SystemExit(f'Run the staging cell first: {config_path} is missing.')
PROBE = json.loads(config_path.read_text(encoding='utf-8'))
ADAPTER_NAME = PROBE['adapter_name']
ADAPTER = Path(PROBE['adapter_dir'])
ADAPTER_WEIGHTS_SHA256 = PROBE['adapter_weights_sha256']
ADAPTER_CONFIG = PROBE['adapter_config']
MODEL_PATH = Path(PROBE['model_path'])
RESULT = WORK / 'probe-result.json'
PROGRESS = WORK / 'probe-progress.json'
STARTED = time.time()


def stage(name, **fields):
    """Phases go to disk because Kaggle publishes a cell's output only when it finishes."""
    # This cell is not an f-string, so no braces need escaping here.
    record = json.loads(PROGRESS.read_text(encoding='utf-8')) if PROGRESS.is_file() else {}
    record[name] = dict(fields, at_seconds=round(time.time() - STARTED, 1))
    PROGRESS.write_text(json.dumps(record, indent=2) + '\\n', encoding='utf-8')
    print('[' + name + ']', json.dumps(fields), flush=True)


def emit(status, **fields):
    payload = dict(status=status, **fields)
    RESULT.write_text(json.dumps(payload, indent=2) + '\\n', encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if not MODEL_PATH.is_dir():
    raise SystemExit(f'Base model mount is missing: {MODEL_PATH}; attach the competition model input.')

try:
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except Exception as error:
    raise SystemExit(f'vLLM is not importable in this session: {error!r}') from error

hardware = os.popen(
    'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader').read().strip().splitlines()
visible = torch.cuda.device_count()
print('vLLM', vllm.__version__, '| GPUs', visible, '|', hardware, flush=True)
assert visible >= PROBE['tensor_parallel_size'], \\
    f"probe needs {PROBE['tensor_parallel_size']} GPUs but {visible} are visible"
stage('environment', vllm=vllm.__version__, torch=torch.__version__, visible_gpus=visible,
      hardware=hardware)

# Capture the engine's own LoRA loader messages: the aliasing defect appears as successful loads
# followed by silent skips, and that distinction matters if the logprobs come back unchanged.
log_path = WORK / 'vllm-lora.log'
root_logger = logging.getLogger()
handlers_before = list(root_logger.handlers)
stream = log_path.open('w', encoding='utf-8')
handler = logging.FileHandler(log_path, encoding='utf-8')
handler.setLevel(logging.DEBUG)
root_logger.addHandler(handler)

# Construction mirrors the harness defaults (adk_submission/server.py) rather than inventing flags.
# `enforce_eager` skips CUDA-graph capture: the probe measures whether the adapter changes the logits,
# not throughput, and graph capture on a 31B model is the slowest part of a vLLM start-up.
stage('engine-start', config={key: PROBE[key] for key in
                              ('tensor_parallel_size', 'max_model_len', 'gpu_memory_utilization',
                               'enable_lora', 'max_loras', 'max_lora_rank', 'enforce_eager')})
llm = LLM(model=str(MODEL_PATH), tensor_parallel_size=PROBE['tensor_parallel_size'],
          max_model_len=PROBE['max_model_len'], gpu_memory_utilization=PROBE['gpu_memory_utilization'],
          enable_lora=PROBE['enable_lora'], max_loras=PROBE['max_loras'],
          max_lora_rank=PROBE['max_lora_rank'], enforce_eager=PROBE['enforce_eager'])
stage('engine-ready')
sampling = SamplingParams(temperature=PROBE['temperature'], max_tokens=PROBE['max_tokens'],
                          logprobs=PROBE['logprobs'])
request = LoRARequest(ADAPTER_NAME, 1, str(ADAPTER))


def run(lora_request=None):
    outputs = llm.chat([{'role': 'user', 'content': PROBE['prompt']}], sampling,
                       lora_request=lora_request)
    completion = outputs[0].outputs[0]
    entries = []
    for position, entry in enumerate(completion.logprobs or []):
        ranked = sorted(entry.items(), key=lambda item: item[1].logprob, reverse=True)[:PROBE['logprobs']]
        entries.append({'position': position, 'token_id': completion.token_ids[position],
                        'top': [[token_id, round(candidate.logprob, 6)] for token_id, candidate in ranked]})
    return {'text': completion.text, 'token_ids': list(completion.token_ids), 'top_logprobs': entries}


base = run()
stage('base-generated', tokens=len(base['token_ids']), text=base['text'][:80])
# A second base run is the control that makes the comparison readable: if the engine reproduces its
# own logprobs exactly, any base-versus-adapter difference is attributable to the adapter rather than
# to run-to-run nondeterminism. Without this, "logprobs differ" is not evidence of anything.
base_repeat = run()
stage('base-repeat-generated', tokens=len(base_repeat['token_ids']), text=base_repeat['text'][:80])
adapter = run(request)
stage('adapter-generated', tokens=len(adapter['token_ids']), text=adapter['text'][:80])
# The wide probe: rank 32 on every layer, which is the configuration we actually intend to train and
# the corner upstream issue #50059 warns about. It cannot be embedded (hundreds of megabytes), so it is
# generated here with PEFT -- the same library that will export the trained adapter.
runtime_request = None
runtime = None
try:
    with (WORK / 'runtime_adapter.py').open('w', encoding='utf-8') as stream:
        stream.write(RUNTIME_ADAPTER_SOURCE)
    namespace = {'__name__': 'runtime_adapter'}
    exec(compile(RUNTIME_ADAPTER_SOURCE, 'runtime_adapter', 'exec'), namespace)
    runtime_path = WORK / 'runtime-adapter'
    runtime_info = (namespace['build'](runtime_path, MODEL_PATH, stage)
                    if namespace.get('build') else {})
    if runtime_info:
        stage('runtime-adapter', **runtime_info)
    runtime_request = LoRARequest('runtime_lora', 2, str(runtime_path))
    runtime = run(runtime_request)
    stage('runtime-adapter-generated', tokens=len(runtime['token_ids']), text=runtime['text'][:80])
except Exception as error:
    stage('runtime-adapter-failed', error=repr(error)[:400])
root_logger.removeHandler(handler)
handler.close()
for previous in handlers_before:
    root_logger.addHandler(previous)

log_lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
loaded = [line for line in log_lines if 'Successfully loaded LoRA weights for module' in line]
skipped = [line for line in log_lines if 'No LoRA weights found for module' in line]

shared = min(len(base['top_logprobs']), len(adapter['top_logprobs']))


def top_map(entries):
    # dict(...) rather than a comprehension so the generated cell carries no set-literal ambiguity.
    return dict((entry['position'], dict(entry['top'])) for entry in entries)


def divergence(left, right):
    """Per-position logprob deltas between two generations, restricted to shared positions."""
    left_map, right_map = top_map(left['top_logprobs']), top_map(right['top_logprobs'])
    deltas = []
    for position in sorted(set(left_map) & set(right_map)):
        for token_id, value in left_map[position].items():
            if token_id in right_map[position]:
                deltas.append(abs(value - right_map[position][token_id]))
    return deltas


def compare(candidate):
    """Score one adapter against the base, with the repeat-base run as the noise floor."""
    signal = divergence(base, candidate)
    changed = [position for position in range(shared)
               if base['top_logprobs'][position]['top'] != candidate['top_logprobs'][position]['top']]
    return dict(
        delta_max=round(max(signal) if signal else 0.0, 8),
        changed_positions=len(changed),
        text_changed=base['text'] != candidate['text'],
        token_ids_changed=base['token_ids'] != candidate['token_ids'])


noise = divergence(base, base_repeat)
noise_max = max(noise) if noise else 0.0
# An adapter is only credited when its effect exceeds what the engine produces on its own.
reproducible = bool(noise) and noise_max <= 1e-6
embedded = compare(adapter)
wide = compare(runtime) if runtime is not None else None


def verdict_for(result, label, loaded_lines, skipped_lines):
    if result is None:
        return label + ': not measured (the runtime adapter was not generated, or its generation failed)'
    if not reproducible:
        return (label + ': inconclusive -- two base runs already differ, so a difference cannot be '
                'attributed to the adapter')
    if result['delta_max'] > max(noise_max, 1e-6):
        return (label + ': applied -- the engine reproduces its own logprobs exactly and the adapter '
                'moves them' + ('' if not (result['text_changed'] or result['token_ids_changed'])
                                else ', changing the generated tokens'))
    if loaded_lines and skipped_lines:
        return (label + ': inert -- weights loaded under the layer names and the YOLO decoder aliases '
                'were then skipped (the aliasing wipe of discussion 743508)')
    return label + ': inert -- identical logprobs, so LoRA is not reaching the served model'


verdict = verdict_for(embedded, 'embedded rank-8 adapter', loaded, skipped)
if wide is not None:
    verdict += ' | ' + verdict_for(wide, 'runtime rank-32 all-layer adapter', loaded, skipped)

if not base['top_logprobs'] or not adapter['top_logprobs']:
    raise SystemExit('Base or adapter generation returned no logprobs; the probe would not measure anything.')

emit(
    'measured',
    applied=embedded['delta_max'] > max(noise_max, 1e-6),
    wide_applied=(wide['delta_max'] > max(noise_max, 1e-6)) if wide else None,
    verdict=verdict,
    engine_reproducible=reproducible,
    base_repeat_noise_max_abs_logprob_delta=round(noise_max, 8),
    embedded=embedded, wide_configuration=wide,
    compared_positions=shared,
    text_changed=embedded['text_changed'], token_ids_changed=embedded['token_ids_changed'],
    base_text=base['text'], adapter_text=adapter['text'],
    base_token_ids=base['token_ids'][:16], adapter_token_ids=adapter['token_ids'][:16],
    base_top_logprobs=base['top_logprobs'][:8], adapter_top_logprobs=adapter['top_logprobs'][:8],
    base_repeat_top_logprobs=base_repeat['top_logprobs'][:8],
    wide_top_logprobs=(runtime['top_logprobs'][:8] if runtime else None),
    runtime_adapter_info=(namespace.get('RUNTIME') if runtime is not None else None),
    lora_loaded_modules=len(loaded), lora_skipped_modules=len(skipped),
    lora_skip_example=skipped[0] if skipped else None,
    lora_load_example=loaded[0] if loaded else None,
    log_lines=len(log_lines),
    adapter_config=ADAPTER_CONFIG, adapter_weights_sha256=ADAPTER_WEIGHTS_SHA256,
    model_path=str(MODEL_PATH), model_handle='google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
    vllm_version=vllm.__version__, torch_version=torch.__version__, visible_gpus=visible,
    probe=PROBE)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--adapter-dir', type=Path, required=True,
                        help='PEFT adapter directory to embed (probe or trained)')
    parser.add_argument('--owner', required=True, help='Kaggle account that will own the private kernel')
    parser.add_argument('--slug', default='gemma4-lora-serving-probe')
    parser.add_argument('--title', default=None)
    parser.add_argument('--model-path', default=MODEL_PATH,
                        help='Base model path inside the Kaggle session')
    parser.add_argument('--wide-rank', type=int, default=32,
                        help='Rank of the runtime-generated all-layer probe (the training configuration)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true',
                        help='Validate an already generated output directory instead of writing one')
    args = parser.parse_args()

    adapter = read_adapter(args.adapter_dir.resolve())
    if args.check:
        return check_output(args.output.resolve(), adapter)

    official = json.loads(OFFICIAL_METADATA.read_text(encoding='utf-8'))
    wheelhouse = json.loads(WHEELHOUSE_MANIFEST.read_text(encoding='utf-8'))
    notebook = {
        'nbformat': 4, 'nbformat_minor': 4,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                   'name': 'python3'}},
        'cells': [
            cell('markdown',
                 '# LoRA serving-path probe (private)\n'
                 'Embeds a deliberately loud random adapter (rank '
                 f"{adapter['config']['r']}, targets {adapter['targets']}) and compares greedy vLLM\n"
                 'logprobs between the base model name and the adapter name on one fixed prompt, on the\n'
                 'competition base. It trains nothing, needs no competition submission, and spends one\n'
                 'short L4x4 session. Identical logprobs mean LoRA is inert on this stack.\n\n'
                 'Cells: (1) stage and hash-verify the embedded adapter, (2) install the reviewed\n'
                 'wheelhouse offline and prove `import vllm`, (3) run the paired logprob probe.\n'),
            cell('code', build_setup(adapter, args.model_path)),
            cell('code', build_install(wheelhouse)),
            cell('code', build_probe(build_runtime_adapter(args.wide_rank, adapter['targets']))),
        ],
    }
    if len(json.dumps(notebook).encode('utf-8')) >= 1_000_000:
        raise SystemExit('Notebook source exceeds the Kaggle 1 MB limit; embed a smaller adapter')
    for entry in notebook['cells']:
        if entry['cell_type'] == 'code':
            compile(''.join(entry['source']), '<notebook-cell>', 'exec')

    try:
        args.output.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f'{args.output} already exists; use a fresh --output path for a new run')
    (args.output / 'probe.ipynb').write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    metadata = {
        'id': f'{args.owner}/{args.slug}', 'title': args.title or args.slug.replace('-', ' '),
        'code_file': 'probe.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
        'machine_shape': 'NvidiaL4', 'docker_image': official['dockerImage'],
        'docker_image_pinning_type': 'original',
        'competition_sources': ['gemma-4-developer-agent'],
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'],
        'kernel_sources': [], 'model_sources': [MODEL_SOURCE],
    }
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                                      encoding='utf-8')
    preparation = {
        'prepared_at': datetime.now(timezone.utc).isoformat(),
        'kernel': metadata['id'],
        'adapter_dir': str(args.adapter_dir.resolve()),
        'adapter_config': adapter['config'],
        'adapter_config_sha256': adapter['config_sha256'],
        'adapter_weights_sha256': adapter['weights_sha256'],
        'adapter_weights_bytes': adapter['weights_bytes'],
        'adapter_tensors': adapter['tensors'],
        'model_path': args.model_path, 'model_handle': MODEL_SOURCE,
        'prompt': PROMPT, 'vllm_version': VLLM_VERSION,
        'status': 'notebook generated locally; no GPU execution or upload',
    }
    (args.output / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n',
                                                  encoding='utf-8')
    print(json.dumps({'kernel': metadata['id'], 'output': str(args.output),
                      'tensors': len(adapter['tensors']), 'assets': len(adapter['targets']),
                      'weights_bytes': adapter['weights_bytes'],
                      'reviewed_wheels': len(wheelhouse['wheelhouse_files']),
                      'notebook_bytes': (args.output / 'probe.ipynb').stat().st_size}, indent=2))
    return 0


def check_output(output: Path, adapter: dict) -> int:
    notebook_path = output / 'probe.ipynb'
    metadata_path = output / 'kernel-metadata.json'
    preparation_path = output / 'preparation.json'
    for required in (notebook_path, metadata_path, preparation_path):
        if not required.is_file():
            raise SystemExit(f'{required} is missing')
    notebook = json.loads(notebook_path.read_text(encoding='utf-8-sig'))
    code_cells = [entry for entry in notebook['cells'] if entry['cell_type'] == 'code']
    for entry in code_cells:
        compile(''.join(entry['source']), '<notebook-cell>', 'exec')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    preparation = json.loads(preparation_path.read_text(encoding='utf-8-sig'))
    identity_ok = all([
        metadata['model_sources'] == [MODEL_SOURCE],
        metadata['dataset_sources'] == ['metric/gemma-4-developer-agent-wheelhouse'],
        metadata['competition_sources'] == ['gemma-4-developer-agent'],
        preparation['adapter_weights_sha256'] == adapter['weights_sha256'],
        preparation['adapter_config_sha256'] == adapter['config_sha256'],
        preparation['adapter_weights_bytes'] == adapter['weights_bytes'],
        f"ADAPTER_WEIGHTS_SHA256 = {adapter['weights_sha256']!r}" in ''.join(code_cells[0]['source']),
        f'vllm-{VLLM_VERSION}-' in ''.join(code_cells[1]['source']),
    ])
    return finish_check({
        'status': 'ok' if identity_ok else 'mismatch',
        'output': str(output), 'code_cells': len(code_cells),
        'tensors': len(adapter['tensors']),
        'adapter_weights_sha256': adapter['weights_sha256'],
        'vllm_wheel_pinned': f'vllm-{VLLM_VERSION}-' in json.dumps(notebook),
        'identity_fields_match': identity_ok,
    })


def finish_check(result: dict) -> int:
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
