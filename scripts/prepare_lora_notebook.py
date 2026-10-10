"""Generate a self-contained LoRA SFT notebook from the dataset and adapter config in this repository.

The notebook embeds its training records (gzip+base85), so it needs no separate dataset mount, and it
installs any training wheels supplied with `--wheels-dir` before importing them. That check is the
point: the competition wheelhouse carries `transformers` and `bitsandbytes` but **no `peft`, `trl`,
`accelerate`, or `datasets`**, so an offline Kaggle session cannot fine-tune until those wheels are
attached. The notebook fails with an explicit list of what is missing instead of failing later inside
a training loop, after the GPU session has already been spent.

    python scripts/prepare_lora_notebook.py \
        --dataset artifacts/sft-public-v1.jsonl \
        --agent-dir configs/experiments/lora-trajectory-v1 \
        --owner YOUR_KAGGLE_USERNAME --slug gemma4-lora-sft-v1 \
        --output artifacts/lora-sft-notebook

Preparation only: no GPU execution, no upload, no competition submission.
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
OFFICIAL_METADATA = ROOT / 'resource/notebooks/official/kernel-metadata.json'
TRAINING_MODULES = ['torch', 'transformers', 'peft', 'trl', 'datasets', 'accelerate', 'bitsandbytes']
BASE_MODEL_HANDLE = 'google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'


def cell(kind: str, source: str) -> dict:
    return {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)} if kind == 'markdown' \
        else {'cell_type': kind, 'metadata': {}, 'execution_count': None, 'outputs': [],
              'source': source.splitlines(keepends=True)}


def build_setup(dataset_b85: str, dataset_sha: str, config: dict, wheel_count: int) -> str:
    return f'''import base64, gzip, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path

DATASET_B85 = {dataset_b85!r}
DATASET_SHA256 = {dataset_sha!r}
ADAPTER_CONFIG = json.loads({json.dumps(json.dumps(config))!r})
REQUIRED = {TRAINING_MODULES!r}
WHEEL_COUNT = {wheel_count}

WORK = Path('/kaggle/working/lora-sft') if Path('/kaggle/working').is_dir() else Path.cwd() / 'lora-sft'
WORK.mkdir(parents=True, exist_ok=True)
print('Working directory:', WORK, flush=True)

# 1. Training wheels, installed offline when supplied. Kaggle mounts datasets under /kaggle/input.
wheel_dirs = [p for p in sorted(Path('/kaggle/input').glob('*')) if p.is_dir()]
wheels = sorted(w for d in wheel_dirs for w in d.glob('*.whl'))
if wheels:
    print(f'Installing {{len(wheels)}} training wheels offline', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index',
                    '--no-deps', *[str(w) for w in wheels]], check=True)
elif WHEEL_COUNT:
    raise SystemExit('Expected training wheels in the mounted inputs but found none.')

# 2. Training records: the embedded blob is gzip-compressed source JSONL, verified by hash and
#    written out. This runs before the dependency check so the data identity is always proven.
raw_dataset = gzip.decompress(base64.b85decode(DATASET_B85))
assert hashlib.sha256(raw_dataset).hexdigest() == DATASET_SHA256, 'embedded dataset hash mismatch'
records = [json.loads(line) for line in raw_dataset.decode('utf-8').splitlines() if line.strip()]
assert records, 'no training records were embedded'
(WORK / 'dataset.jsonl').write_bytes(raw_dataset)
print(f'Training records: {{len(records)}} (sha256 {{DATASET_SHA256[:16]}}...)', flush=True)

# 3. Dependency gate. The competition wheelhouse carries transformers and bitsandbytes but not
#    peft/trl/accelerate/datasets, so an offline session needs those wheels attached.
missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        'Offline environment is missing training dependencies: ' + ', '.join(missing) +
        '. Attach a wheel dataset with peft/trl/accelerate/datasets and rerun; the competition '
        'wheelhouse does not carry them, and Internet is disabled in this notebook.')

# 4. Base model. Kaggle mounts the served W4A16 checkpoint; a training host may pass its own path
#    through the GEMMA_BASE_MODEL environment variable (the unquantized QAT checkpoint is preferred).
MODEL_CANDIDATES = [
    os.environ.get('GEMMA_BASE_MODEL'),
    '/kaggle/input/gemma-4-31b-it-qat-w4a16-ct',
    '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
]
BASE_MODEL = next((m for m in MODEL_CANDIDATES if m and Path(m).is_dir()), None)
assert BASE_MODEL, f'No base model found among {{MODEL_CANDIDATES}}'
print('Base model:', BASE_MODEL, flush=True)
print('Adapter config:', json.dumps(ADAPTER_CONFIG), flush=True)
'''


def build_train(max_steps: int, learning_rate: float, max_length: int) -> str:
    return f'''import json, time
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MAX_STEPS = {max_steps}
LEARNING_RATE = {learning_rate}
MAX_LENGTH = {max_length}
RANK = ADAPTER_CONFIG['r']
TARGETS = ADAPTER_CONFIG['target_modules']

assert torch.cuda.is_available(), 'No CUDA device; this notebook must run with a GPU accelerator enabled.'
print('CUDA devices:', torch.cuda.device_count(), flush=True)

quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quantization,
                                             torch_dtype=torch.bfloat16, device_map={{'': 0}})
model = prepare_model_for_kbit_training(model)

present = {{name.rsplit('.', 1)[-1] for name, module in model.named_modules()
           if isinstance(module, torch.nn.Linear)}}
absent = sorted(set(TARGETS) - present)
assert not absent, (
    f'Configured target modules are absent from the base model: {{absent}}. '
    'Record the real module names before widening target_modules; several public notebooks assumed '
    'names from other Gemma variants and trained nothing.')

model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ADAPTER_CONFIG['lora_alpha'],
                                         lora_dropout=ADAPTER_CONFIG['lora_dropout'],
                                         bias=ADAPTER_CONFIG['bias'], task_type='CAUSAL_LM',
                                         target_modules=TARGETS))
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(f'Trainable tensors: {{len(trainable)}}', flush=True)
assert trainable, 'LoRA wrapped nothing; training would be a no-op.'

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
dataset = load_dataset('json', data_files=str(WORK / 'dataset.jsonl'), split='train')
trainer = SFTTrainer(
    model=model, processing_class=tokenizer, train_dataset=dataset,
    args=SFTConfig(output_dir=str(WORK / 'checkpoints'), max_steps=MAX_STEPS,
                   per_device_train_batch_size=1, gradient_accumulation_steps=8,
                   learning_rate=LEARNING_RATE, max_length=MAX_LENGTH, logging_steps=1,
                   save_steps=max(1, MAX_STEPS // 2), save_total_limit=2, bf16=True,
                   report_to=[], seed=42, completion_only_loss=True))
started = time.time()
trainer.train()
print(f'Training finished in {{time.time() - started:.1f}} s', flush=True)

EXPORT = WORK / 'adapter'
trainer.save_model(str(EXPORT))
tokenizer.save_pretrained(str(EXPORT))
# Keep the reviewed adapter config: it is the rank the serving path reads.
(EXPORT / 'adapter_config.json').write_text(json.dumps(ADAPTER_CONFIG, indent=2) + '\\n', encoding='utf-8')

weights = EXPORT / 'adapter_model.safetensors'
assert weights.is_file(), 'Export did not produce adapter_model.safetensors'
from safetensors import safe_open
with safe_open(str(weights), framework='pt') as handle:
    keys = sorted(handle.keys())
print(json.dumps({{
    'status': 'trained',
    'records': len(dataset),
    'trainable_tensors': len(trainable),
    'export': str(EXPORT),
    'adapter_model_bytes': weights.stat().st_size,
    'adapter_model_sha256': __import__('hashlib').sha256(weights.read_bytes()).hexdigest(),
    'lora_tensor_keys': len(keys),
    'example_key': keys[0] if keys else None,
}}, indent=2))
print('Pair this adapter with a no-adapter control on the same frozen tasks before claiming a gain.')
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='JSONL with prompt/completion records from scripts/build_sft_dataset.py')
    parser.add_argument('--agent-dir', type=Path,
                        default=ROOT / 'configs/experiments/lora-trajectory-v1')
    parser.add_argument('--adapter-name', default='main_lora')
    parser.add_argument('--owner', required=True, help='Kaggle username that will own the notebook')
    parser.add_argument('--slug', required=True)
    parser.add_argument('--title', default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheels-dir', type=Path, default=None,
                        help='Directory of offline training wheels to mount as a Kaggle dataset source')
    parser.add_argument('--wheels-source', default=None,
                        help='Kaggle dataset handle for those wheels, e.g. owner/gemma4-training-wheels')
    parser.add_argument('--max-steps', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-length', type=int, default=4096)
    args = parser.parse_args()

    config_path = args.agent_dir / 'adapters' / args.adapter_name / 'adapter_config.json'
    if not config_path.is_file():
        raise SystemExit(f'missing adapter config: {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not args.dataset.is_file():
        raise SystemExit(f'missing dataset: {args.dataset} (run scripts/build_sft_dataset.py first)')

    raw = args.dataset.read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not records:
        raise SystemExit(f'no records in {args.dataset}')
    for field in ('prompt', 'completion'):
        if not all(field in record for record in records):
            raise SystemExit(f'every record must carry a {field!r} field for completion_only_loss')
    blob = gzip.compress(raw, mtime=0)
    dataset_sha = hashlib.sha256(raw).hexdigest()
    if args.wheels_source and not args.wheels_dir:
        raise SystemExit('--wheels-source needs --wheels-dir so the wheel inventory can be recorded')

    official = json.loads(OFFICIAL_METADATA.read_text(encoding='utf-8'))
    wheel_count = len(sorted(args.wheels_dir.glob('*.whl'))) if args.wheels_dir else 0
    setup = build_setup(base64.b85encode(blob).decode(), dataset_sha, config, wheel_count)
    notebook = {
        'nbformat': 4, 'nbformat_minor': 4,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                    'name': 'python3'}},
        'cells': [
            cell('markdown',
                 '# Gemma 4 LoRA SFT (private)\n'
                 'Trains the configured PEFT adapter on the embedded public-task records, offline.\n'
                 'This notebook is preparation output; it does not submit to the competition and its\n'
                 'result is not a benchmark until a paired control run on the same frozen tasks exists.'),
            cell('code', setup),
            cell('code', build_train(args.max_steps, args.learning_rate, args.max_length)),
        ],
    }
    if len(json.dumps(notebook).encode('utf-8')) >= 1_000_000:
        raise SystemExit('Notebook source exceeds the Kaggle 1 MB limit; reduce the dataset or records')
    for entry in notebook['cells']:
        if entry['cell_type'] == 'code':
            compile(''.join(entry['source']), '<notebook-cell>', 'exec')

    try:
        args.output.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f'{args.output} already exists; use a fresh --output path for a new run')
    (args.output / 'train.ipynb').write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    metadata = {
        'id': f'{args.owner}/{args.slug}', 'title': args.title or args.slug.replace('-', ' '),
        'code_file': 'train.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
        'machine_shape': 'NvidiaL4', 'docker_image': official['dockerImage'],
        'docker_image_pinning_type': 'original',
        'competition_sources': [], 'dataset_sources': [args.wheels_source] if args.wheels_source else [],
        'kernel_sources': [], 'model_sources': [BASE_MODEL_HANDLE],
    }
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                                      encoding='utf-8')
    preparation = {
        'prepared_at': datetime.now(timezone.utc).isoformat(),
        'kernel': metadata['id'],
        'dataset': str(args.dataset),
        'records': len(records),
        'dataset_blob_sha256': dataset_sha,
        'adapter_name': args.adapter_name,
        'adapter_config': json.loads(config_path.read_text(encoding='utf-8-sig')),
        'max_steps': args.max_steps, 'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'wheels_dir': str(args.wheels_dir) if args.wheels_dir else None,
        'wheels_count': wheel_count,
        'training_modules_required': TRAINING_MODULES,
        'known_gap': ('The competition wheelhouse provides transformers and bitsandbytes but not '
                      'peft/trl/accelerate/datasets, so --wheels-dir plus --wheels-source must supply '
                      'them for an offline run; otherwise the notebook stops with the missing list.'),
        'status': 'notebook generated locally; no GPU execution and no upload',
    }
    (args.output / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n',
                                                  encoding='utf-8')
    print(json.dumps({'kernel': metadata['id'], 'records': len(records),
                      'output': str(args.output), 'wheels': wheel_count}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
