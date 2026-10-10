"""Record the real projection module names and shapes of the required base model.

Run this once in a session that has the base weights (Trillium or Kaggle) before widening
`target_modules` in any adapter config. It answers exactly one question with evidence instead of
names copied from unrelated Gemma variants: which `nn.Linear` projections exist, and with what
in_features/out_features.

    python scripts/probe_model_modules.py \
        --base-model "$GEMMA4_QAT_BASE" \
        --output resource/manifests/model-modules.json

Loading needs the model files; the shipped model is a compressed-tensors W4A16 checkpoint, so
`transformers` plus the quantization support it requires must be present. For module names and shapes
alone, `--device-map cpu` (the default) avoids touching a GPU: the probe never runs a forward pass.
Nothing is trained, generated, or downloaded here.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECTION_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def probe(base_model: str, device_map: str) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map,
                                                torch_dtype=torch.bfloat16)
    projections: dict[str, dict] = {}
    layer_indices: set[int] = set()
    other_linear: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit('.', 1)[-1]
        if leaf not in PROJECTION_SUFFIXES:
            other_linear.add(leaf)
            continue
        entry = projections.setdefault(leaf, {
            'in_features': module.in_features,
            'out_features': module.out_features,
            'example_module_path': name,
            'count': 0,
        })
        if (entry['in_features'], entry['out_features']) != (module.in_features, module.out_features):
            raise SystemExit(
                f'{leaf} has inconsistent shapes: {entry} versus '
                f'{(module.in_features, module.out_features)} at {name}'
            )
        entry['count'] += 1
        for part in name.split('.'):
            if part.isdigit():
                layer_indices.add(int(part))
                break
    layer_count = max(layer_indices) + 1 if layer_indices else 0
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'base_model': base_model,
        'model_type': getattr(config, 'model_type', None),
        'architectures': list(getattr(config, 'architectures', []) or []),
        'projections': dict(sorted(projections.items())),
        'projection_layer_count': layer_count,
        'other_linear_leaves': sorted(other_linear),
        'total_parameters': total,
        'dtype': 'bfloat16',
        'scope': 'Module names and shapes only; no forward pass, no generation, no training',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-model', required=True,
                        help='Local path or mounted handle of the required base model')
    parser.add_argument('--output', type=Path,
                        default=Path('resource/manifests/model-modules.json'))
    parser.add_argument('--device-map', default='cpu',
                        help='"cpu" (default) avoids allocating GPU memory for this probe')
    args = parser.parse_args()
    result = probe(args.base_model, args.device_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('base_model', 'projection_layer_count', 'total_parameters', 'scope')}, indent=2))
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
