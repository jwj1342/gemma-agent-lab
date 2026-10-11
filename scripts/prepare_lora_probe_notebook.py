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
BASE_MODEL_HANDLE = 'google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'
MODEL_SOURCE = 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'
MODEL_PATH = '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'
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
          'enable_lora': True, 'max_loras': 1, 'max_lora_rank': 128}}

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


def build_probe() -> str:
    return '''import json, logging, os
from pathlib import Path

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


def finish(status, **fields):
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
llm = LLM(model=str(MODEL_PATH), tensor_parallel_size=PROBE['tensor_parallel_size'],
          max_model_len=PROBE['max_model_len'], gpu_memory_utilization=PROBE['gpu_memory_utilization'],
          enable_lora=PROBE['enable_lora'], max_loras=PROBE['max_loras'],
          max_lora_rank=PROBE['max_lora_rank'])
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
adapter = run(request)
root_logger.removeHandler(handler)
handler.close()
for previous in handlers_before:
    root_logger.addHandler(previous)

log_lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
loaded = [line for line in log_lines if 'Successfully loaded LoRA weights for module' in line]
skipped = [line for line in log_lines if 'No LoRA weights found for module' in line]

shared = min(len(base['top_logprobs']), len(adapter['top_logprobs']))
changed_positions = [position for position in range(shared)
                     if base['top_logprobs'][position]['top'] != adapter['top_logprobs'][position]['top']]
token_ids_changed = base['token_ids'] != adapter['token_ids']
text_changed = base['text'] != adapter['text']
where = next((position for position in range(shared) if base['token_ids'][position]
              != adapter['token_ids'][position]), None)

if changed_positions or token_ids_changed or text_changed:
    verdict = ('applied: the serving stack honours the adapter, so a trained adapter is worth funding'
               if text_changed or token_ids_changed else
               'applied: the serving stack honours the adapter, and as expected a loud probe mostly '
               'shifts the distribution rather than rewriting the text')
elif loaded and skipped:
    verdict = ('inert: the engine loaded the adapter under the layer names and then skipped the YOLO '
               'decoder aliases, which is the exact aliasing wipe of discussion 743508')
elif loaded:
    verdict = ('inert: the engine reports the adapter as loaded and the logprobs are still identical, '
               'so LoRA is dropped downstream of loading (upstream issue #50059 pattern)')
else:
    verdict = ('inert: identical logprobs and no LoRA load messages, so the adapter never reached the '
               'engine at all')

if not base['top_logprobs'] or not adapter['top_logprobs']:
    raise SystemExit('Base or adapter generation returned no logprobs; the probe would not measure anything.')

finish(
    'measured',
    applied=bool(changed_positions or token_ids_changed or text_changed),
    verdict=verdict,
    compared_positions=shared,
    changed_logprob_positions=len(changed_positions),
    first_changed_position=where if where is not None else (changed_positions[0] if changed_positions else None),
    text_changed=text_changed,
    token_ids_changed=token_ids_changed,
    base_text=base['text'], adapter_text=adapter['text'],
    base_token_ids=base['token_ids'][:16], adapter_token_ids=adapter['token_ids'][:16],
    base_top_logprobs=base['top_logprobs'][:8], adapter_top_logprobs=adapter['top_logprobs'][:8],
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true',
                        help='Validate an already generated output directory instead of writing one')
    args = parser.parse_args()

    adapter = read_adapter(args.adapter_dir.resolve())
    if args.check:
        return check_output(args.output.resolve(), adapter)

    official = json.loads(OFFICIAL_METADATA.read_text(encoding='utf-8'))
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
                 'short L4x4 session. Identical logprobs mean LoRA is inert on this stack.\n'),
            cell('code', build_setup(adapter, args.model_path)),
            cell('code', build_probe()),
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
    ])
    return finish_check({
        'status': 'ok' if identity_ok else 'mismatch',
        'output': str(output), 'code_cells': len(code_cells),
        'tensors': len(adapter['tensors']),
        'adapter_weights_sha256': adapter['weights_sha256'],
        'identity_fields_match': identity_ok,
    })


def finish_check(result: dict) -> int:
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
