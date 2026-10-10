"""Verify the offline SFT stack end to end on CPU, with no base weights and no GPU.

`--self-check` in `train_lora_adapter.py` proves PEFT wiring. This script proves more: that
`datasets` can load the generated JSONL, that `SFTTrainer` with `completion_only_loss=True` runs a
real optimisation step on a real `transformers` causal LM, and that the export is a PEFT directory
the competition discovery code recognises.

It exists because the offline training set is assembled from several distributions and only an
end-to-end run catches the version constraints and missing transitive dependencies. Two real
examples already found this way: `transformers==5.13.1` requires `huggingface-hub>=1.5.0,<2.0`
(2.2.0 breaks the import), and `datasets` needs `pandas` and `multiprocess`, which the competition
wheelhouse does not carry.

    python scripts/check_training_stack.py --dataset artifacts/sft-public-v1.jsonl \
        --agent-dir configs/experiments/lora-trajectory-v1 --output artifacts/stack-check
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TINY_MODEL = {
    'model_type': 'gemma3_text',
    'hidden_size': 64,
    'intermediate_size': 128,
    'num_hidden_layers': 2,
    'num_attention_heads': 4,
    'num_key_value_heads': 2,
    'head_dim': 16,
    'vocab_size': 256,
    'max_position_embeddings': 512,
    'rms_norm_eps': 1e-6,
    'sliding_window': 128,
    'sliding_window_pattern': 2,
    'rope_theta': 10000.0,
    'torch_dtype': 'float32',
}


def _local_tokenizer():
    """A byte-level stand-in tokenizer, so the check runs with no download and no cache."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {f'tok{i}': i for i in range(250)}
    vocabulary.update({'[UNK]': 250, '[PAD]': 251, '[BOS]': 252, '[EOS]': 253})
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
        bos_token='[BOS]', eos_token='[EOS]', model_max_length=256)


def run(dataset: Path, agent_dir: Path, adapter_name: str, output: Path, max_steps: int) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3TextConfig
    from trl import SFTConfig, SFTTrainer

    config_path = agent_dir / 'adapters' / adapter_name / 'adapter_config.json'
    adapter_config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    records = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    if not records:
        raise SystemExit(f'no records in {dataset}')
    if not all({'prompt', 'completion'} <= set(record) for record in records):
        raise SystemExit('records must carry prompt and completion for completion_only_loss')

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # Prefer the real Gemma tokenizer when it is already cached or reachable; otherwise build one
    # locally, because this check is about the training loop, not about tokenizer fidelity.
    try:
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
        tokenizer_source = 'google/gemma-3-1b-it'
    except Exception as error:
        tokenizer = _local_tokenizer()
        tokenizer_source = f'local stand-in (pretrained unavailable: {type(error).__name__})'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_config(Gemma3TextConfig(**TINY_MODEL))
    model = get_peft_model(model, LoraConfig(
        r=adapter_config['r'], lora_alpha=adapter_config['lora_alpha'],
        lora_dropout=adapter_config['lora_dropout'], bias=adapter_config['bias'],
        task_type='CAUSAL_LM', target_modules=adapter_config['target_modules']))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise SystemExit('LoRA wrapped nothing; the configured target modules do not exist here')

    # A tiny subset: this checks the loop works, never that training improves anything.
    subset = output / 'subset.jsonl'
    subset.write_text('\n'.join(json.dumps(record) for record in records[:8]) + '\n',
                      encoding='utf-8')
    dataset_obj = load_dataset('json', data_files=str(subset), split='train')
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset_obj,
        args=SFTConfig(output_dir=str(output / 'checkpoints'), max_steps=max_steps,
                       per_device_train_batch_size=1, gradient_accumulation_steps=1,
                       learning_rate=1e-4, max_length=256, logging_steps=1, save_strategy='no',
                       report_to=[], seed=0, completion_only_loss=True, use_cpu=True))
    result = trainer.train()
    losses = [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
    if not losses:
        raise SystemExit('SFTTrainer produced no loss entries; the loss mask may have removed all tokens')

    export = output / 'adapter'
    trainer.save_model(str(export))
    (export / 'adapter_config.json').write_text(json.dumps(adapter_config, indent=2) + '\n',
                                                encoding='utf-8')
    weights = export / 'adapter_model.safetensors'
    if not weights.is_file():
        raise SystemExit('export did not produce adapter_model.safetensors')
    from safetensors import safe_open
    with safe_open(str(weights), framework='pt') as handle:
        keys = sorted(handle.keys())
    if not any('lora_A' in key for key in keys) or not any('lora_B' in key for key in keys):
        raise SystemExit(f'exported tensors do not look like LoRA weights: {keys[:4]}')

    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_config(Gemma3TextConfig(**TINY_MODEL)), str(export))
    reload_ok = all(torch.equal(reloaded.state_dict()[key], model.state_dict()[key])
                    for key in model.state_dict() if 'lora_' in key)
    return {
        'status': 'ok',
        'resolved_versions': {name: _version(name) for name in
                              ['torch', 'transformers', 'huggingface-hub', 'peft', 'trl',
                               'datasets', 'pandas', 'tokenizers']},
        'tokenizer_source': tokenizer_source,
        'records_total': len(records),
        'records_used': min(len(records), 8),
        'trainable_tensors': len(trainable),
        'loss_entries': len(losses),
        'first_loss': losses[0],
        'last_loss': losses[-1],
        'train_runtime_s': result.metrics.get('train_runtime'),
        'export': str(export),
        'adapter_model_bytes': weights.stat().st_size,
        'lora_tensor_keys': len(keys),
        'reload_matches': reload_ok,
        'scope': ('Offline SFT stack and export layout on a tiny random model. It proves the '
                  'toolchain runs; it says nothing about accuracy or about the real base model.'),
    }


def _version(name: str) -> str:
    import importlib.metadata as metadata
    try:
        return metadata.version(name)
    except Exception:
        return 'missing'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/sft-public-v1.jsonl')
    parser.add_argument('--agent-dir', type=Path,
                        default=ROOT / 'configs/experiments/lora-trajectory-v1')
    parser.add_argument('--adapter-name', default='main_lora')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/stack-check')
    parser.add_argument('--max-steps', type=int, default=3)
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()
    result = run(args.dataset.resolve(), args.agent_dir.resolve(), args.adapter_name,
                 args.output.resolve(), args.max_steps)
    text = json.dumps(result, indent=2) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
