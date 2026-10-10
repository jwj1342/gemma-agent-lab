"""Build or check a loud probe adapter that tests whether LoRA is applied at serving time.

The competition requirement is a compressed-tensors W4A16 base, and LoRA on such a base has a history
of failing quietly: adapters load and the server reports success, yet the output is byte-identical to
the base model. Waiting until after a training run to discover that wastes the whole run.

This tool produces a deliberately loud adapter -- random ``lora_A`` and ``lora_B`` at a scale that
must change the logits -- or checks an exported one. Point the harness at the *competition model*
(it only needs a valid tokenizer, so any checkpoint with the same vocabulary works) and compare
greedy logprobs with and without the adapter:

* logprobs change  -> the serving stack applies LoRA; training is worth doing.
* logprobs identical -> LoRA is inert on this stack; do not spend GPU time training until that is
  resolved, because every adapter would score exactly like the base model.

    python scripts/make_lora_probe.py --check --output artifacts/lora-probe
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_TARGETS = ['q_proj', 'o_proj']
# Small enough to stay a valid LoRA, large enough that an applied adapter cannot leave logprobs
# unchanged. The public example adapters shipped with all-zero B and are therefore useless for this.
A_STD = 0.02
B_STD = 1.0
# Recorded from the shipped sample adapter (resource/manifests/sample-adapters.json), the only
# verified shapes for this base: gemma-4-31b-it-qat-w4a16-ct.
KNOWN_SHAPES = {
    'q_proj': (5376, 8192),
    'o_proj': (8192, 5376),
}


def build(rank: int, targets: list[str], output: Path, seed: int) -> dict:
    import torch
    from safetensors.torch import save_file

    unknown = sorted(set(targets) - set(KNOWN_SHAPES))
    if unknown:
        raise SystemExit(f'no verified shapes for {unknown}; add them from a real probe first')
    if output.exists():
        raise SystemExit(f'{output} already exists; use a fresh --output path')
    output.mkdir(parents=True)

    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        in_features, out_features = KNOWN_SHAPES[target]
        stem = f'base_model.model.model.language_model.layers.0.self_attn.{target}'
        tensors[f'{stem}.lora_A.weight'] = (
            torch.randn(rank, in_features, generator=generator, dtype=torch.float32) * A_STD
        ).to(torch.bfloat16)
        tensors[f'{stem}.lora_B.weight'] = (
            torch.randn(out_features, rank, generator=generator, dtype=torch.float32) * B_STD
        ).to(torch.bfloat16)
    weights = output / 'adapter_model.safetensors'
    save_file(tensors, str(weights))
    (output / 'adapter_config.json').write_text(json.dumps({
        'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': rank, 'lora_alpha': rank,
        'lora_dropout': 0.0, 'bias': 'none', 'inference_mode': True,
        'base_model_name_or_path': 'gemma-4-31b-it-qat-w4a16-ct',
        'target_modules': targets, 'modules_to_save': [],
    }, indent=2) + '\n', encoding='utf-8')
    return {
        'status': 'built',
        'kind': 'loud probe adapter -- random A and B, not trained, not for submission',
        'rank': rank, 'lora_alpha': rank, 'targets': targets, 'seed': seed,
        'tensors': sorted(tensors), 'bytes': weights.stat().st_size, 'output': str(output),
    }


def check(output: Path) -> dict:
    from safetensors import safe_open

    weights = output / 'adapter_model.safetensors'
    config_path = output / 'adapter_config.json'
    if not weights.is_file() or not config_path.is_file():
        raise SystemExit(f'{output} does not hold a PEFT adapter directory')
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    with safe_open(str(weights), framework='pt') as handle:
        keys = sorted(handle.keys())
        nonzero = {}
        for key in keys:
            tensor = handle.get_tensor(key)
            nonzero[key] = float(tensor.abs().max())
    zero_sided = [key for key, value in nonzero.items() if value == 0.0]
    a_keys = [key for key in keys if 'lora_A' in key]
    b_keys = [key for key in keys if 'lora_B' in key]
    return {
        'status': 'ok' if not zero_sided else 'inert',
        'rank': config.get('r'),
        'targets': config.get('target_modules'),
        'tensor_count': len(keys),
        'lora_A_tensors': len(a_keys),
        'lora_B_tensors': len(b_keys),
        'zero_tensors': zero_sided,
        'max_abs_per_tensor': nonzero,
        'verdict': ('A or B is all zero, so this adapter cannot change any output; it is the trap '
                    'the shipped example adapters fall into'
                    if zero_sided else
                    'both sides carry signal, so an unchanged output with this adapter mounted proves '
                    'the serving path is dropping LoRA rather than the adapter being trivial'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Validate an existing probe instead of building one')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--targets', nargs='+', default=DEFAULT_TARGETS)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    result = check(args.output.resolve()) if args.check else build(
        args.rank, args.targets, args.output.resolve(), args.seed)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('ok', 'built') else 1


if __name__ == '__main__':
    raise SystemExit(main())
