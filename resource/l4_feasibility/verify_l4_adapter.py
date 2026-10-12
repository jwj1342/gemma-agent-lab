"""Validate an exported LoRA adapter directory without a GPU, a base model, or a network.

This is the cheap gate that runs before any official-runtime reload: if the export is malformed, no
serving test can succeed, and finding that out costs nothing here instead of a GPU session.

The expected key scheme is the one ``scripts/make_lora_probe.py`` established for this base and the
one the competition serving stack was measured to accept:

    base_model.model.model.language_model.layers.<N>.self_attn.<q_proj|o_proj|...>.lora_A.weight
    base_model.model.model.language_model.layers.<N>.self_attn.<q_proj|o_proj|...>.lora_B.weight

    python verify_l4_adapter.py --adapter /kaggle/working/l4-feasibility/adapter --out adapter-check.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

KEY = re.compile(r'^base_model\.model\.model\.language_model\.layers\.(\d+)\.'
                 r'(?:self_attn|mlp)\.(?P<projection>[a-z_]+)\.lora_(?P<side>[AB])\.weight$')


def analyse(keys, shapes, config, expected_rank: int) -> dict:
    """Judge an adapter from its tensor names, shapes and config.

    Kept free of file and framework access so the decision rule can be tested without writing an
    adapter to disk.
    """
    unmatched = [key for key in keys if not KEY.match(key)]
    matched = [key for key in keys if KEY.match(key)]
    layers = sorted({int(KEY.match(key).group(1)) for key in matched})
    sides = Counter(KEY.match(key).group('side') for key in matched)
    projections = Counter(KEY.match(key).group('projection') for key in matched)
    rank_from_shapes = sorted({shapes[key][0] for key in keys
                               if key.endswith('lora_A.weight') and len(shapes[key]) == 2})
    in_features = sorted({shapes[key][1] for key in keys
                          if key.endswith('lora_A.weight') and len(shapes[key]) == 2})
    result = {
        'tensor_count': len(keys),
        'served_scheme_keys': len(matched),
        'unmatched_keys': unmatched[:5],
        'unmatched_count': len(unmatched),
        'unmatched_prefixes': sorted({key.split('.lora_')[0] for key in unmatched if '.lora_' in key})[:5],
        'layers_covered': len(layers),
        'layer_min': layers[0] if layers else None,
        'layer_max': layers[-1] if layers else None,
        'a_tensors': sides.get('A', 0), 'b_tensors': sides.get('B', 0),
        'projections': dict(projections),
        'rank_from_shapes': rank_from_shapes,
        'in_features_from_shapes': in_features,
        'example_key': keys[0] if keys else None,
    }
    problems = []
    if not matched:
        problems.append('no key follows the naming scheme the serving stack is measured to accept')
    if sides.get('A') != sides.get('B'):
        problems.append('lora_A and lora_B counts differ, so some layer has only one side')
    if rank_from_shapes != [expected_rank]:
        problems.append(f'rank from shapes is {rank_from_shapes}, expected [{expected_rank}]')
    if config.get('r') != expected_rank:
        problems.append(f"adapter_config r={config.get('r')} but expected {expected_rank}")
    # A differing prefix is reported, not failed: a PEFT export may use a different module path while
    # still being loadable. The served-scheme count above is what the serving check has to confirm.
    result['naming_note'] = (f'{len(unmatched)} keys carry a prefix outside the measured scheme; the '
                             'official-runtime reload has to confirm them') if unmatched else 'all keys match'
    result['problems'] = problems
    result['status'] = 'ok' if not problems else 'invalid'
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--expected-rank', type=int, default=16)
    args = parser.parse_args()

    from safetensors import safe_open

    weights = args.adapter / 'adapter_model.safetensors'
    config_path = args.adapter / 'adapter_config.json'
    result = {'adapter': str(args.adapter), 'weights_present': weights.is_file(),
              'config_present': config_path.is_file()}
    if not result['weights_present'] or not result['config_present']:
        result['status'] = 'incomplete'
        print(json.dumps(result, indent=2))
        return 1
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    result['config'] = config

    with safe_open(str(weights), framework='pt') as handle:
        keys = sorted(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    result.update(analyse(keys, shapes, config, args.expected_rank))
    result['bytes'] = weights.stat().st_size
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
