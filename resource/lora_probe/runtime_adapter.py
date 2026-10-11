"""Template for the runtime-generated wide LoRA probe used by the serving check.

`scripts/prepare_lora_probe_notebook.py` reads this file, substitutes the __PLACEHOLDER__ tokens, and
embeds the result in a notebook cell. Keeping it as a real file rather than an f-string inside the
generator means it can be compiled and tested on its own, and that no brace escaping is needed.

Why the adapter is written with safetensors instead of PEFT: on the competition image, importing
`peft` (or `transformers`) raises
``ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above
0.16.0 are supported`` once the wheelhouse has upgraded transformers to 5.13.1, which vLLM 0.19.1
requires. The probe only needs tensors and their shapes, so it writes them directly using the key names
that the verified embedded probe already proved vLLM accepts, and reads the real shapes from the base
checkpoint's safetensors headers.
"""
import json
import struct

import torch
from pathlib import Path
from safetensors.torch import save_file

RUNTIME = {
    'rank': __RANK__,
    'targets': __TARGETS__,
    'seed': __SEED__,
    'a_std': __A_STD__,
    'b_std': __B_STD__,
    'expected_q_proj': __EXPECTED_Q__,
    'expected_o_proj': __EXPECTED_O__,
}
# The embedded probe that vLLM 0.19.1 demonstrably honours uses exactly this scheme:
# base_model.model.model.language_model.layers.<N>.self_attn.<projection>.lora_{A,B}.weight
KEY_TEMPLATE = 'base_model.model.model.language_model.layers.%d.self_attn.%s.%s.weight'
PROJECTION_SUFFIX = '.self_attn.%s.weight'


def _read_header(path):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        return json.loads(stream.read(length).decode('utf-8'))


def _scan_checkpoint(model_path):
    """Read every projection shape from the base checkpoint's own safetensors headers.

    Reading a header costs no GPU memory and needs no index file, which matters because the base is a
    single large shard set with no ``*.index.json``.
    """
    shapes = {}
    shards = []
    for shard in sorted(model_path.rglob('*.safetensors')):
        header = _read_header(shard)
        header.pop('__metadata__', None)
        hit = False
        for name, entry in header.items():
            if not isinstance(entry, dict) or 'shape' not in entry:
                continue
            for projection in RUNTIME['targets']:
                if name.endswith(PROJECTION_SUFFIX % projection):
                    stem = name[: -len(PROJECTION_SUFFIX % projection)]
                    layer = None
                    if '.layers.' in stem:
                        tail = stem.split('.layers.')[-1]
                        if tail.isdigit():
                            layer = int(tail)
                    if layer is not None:
                        shapes[(layer, projection)] = tuple(entry['shape'])
                        hit = True
        if hit:
            shards.append(shard.name)
    return shapes, shards


def _config_layer_count(model_path):
    """Fallback layer count from config.json, without importing transformers."""
    config_path = model_path / 'config.json'
    if not config_path.is_file():
        return None, None
    config = json.loads(config_path.read_text(encoding='utf-8'))
    holders = [config]
    for key in ('text_config', 'language_config', 'llm_config'):
        nested = config.get(key)
        if isinstance(nested, dict):
            holders.insert(0, nested)
    for holder in holders:
        for name in ('num_hidden_layers', 'num_layers', 'n_layer', 'num_decoder_layers'):
            value = holder.get(name)
            if isinstance(value, int) and value > 0:
                return value, config.get('model_type')
    return None, config.get('model_type')


def build(path, model_path, stage=None):
    """Write the wide probe adapter at `path` and return what was measured about it."""
    model_path = Path(model_path)
    shapes, shards = _scan_checkpoint(model_path)
    layers = sorted(set(layer for layer, _ in shapes))
    layer_count, model_type = (len(layers), None) if layers else _config_layer_count(model_path)
    if not layer_count:
        raise SystemExit('cannot determine the layer count for ' + str(model_path))

    supported = list(RUNTIME['targets'])
    if shapes:
        missing = [projection for projection in supported if (layers[0], projection) not in shapes]
        if missing:
            raise SystemExit('checkpoint does not expose ' + str(missing) + ' for layer '
                             + str(layers[0]) + '; found ' + str(sorted(shapes))[:200])
        target_shapes = dict((projection, shapes[(layers[0], projection)]) for projection in supported)
        layer_source = 'checkpoint headers'
    else:
        target_shapes = dict(zip(('q_proj', 'o_proj'),
                                 (tuple(RUNTIME['expected_q_proj']), tuple(RUNTIME['expected_o_proj']))))
        layer_source = 'recorded evidence'
    diagnostics = {
        'layer_source': layer_source,
        'shards_with_targets': shards,
        'measured_layers': len(layers),
        'target_shapes': dict((name, list(shape)) for name, shape in target_shapes.items()),
        'recorded_evidence_match': bool(
            tuple(target_shapes['q_proj']) == tuple(RUNTIME['expected_q_proj'])
            and tuple(target_shapes['o_proj']) == tuple(RUNTIME['expected_o_proj'])),
        'model_type': model_type,
    }
    if stage is not None:
        stage('runtime-adapter-plan', **diagnostics)
    if layer_source == 'recorded evidence':
        # An assumed shape would silently produce an adapter vLLM skips, which the probe would read as
        # "LoRA is inert" -- the one conclusion this tool must never reach by accident.
        print('WARNING: no checkpoint shapes found; assuming recorded evidence ' + json.dumps(diagnostics),
              flush=True)

    generator = torch.Generator().manual_seed(RUNTIME['seed'])
    tensors = {}
    for layer in range(layer_count):
        for projection in supported:
            in_features, out_features = target_shapes[projection]
            tensors[KEY_TEMPLATE % (layer, projection, 'lora_A')] = (
                torch.randn(RUNTIME['rank'], in_features, generator=generator) * RUNTIME['a_std']
            ).to(torch.bfloat16)
            tensors[KEY_TEMPLATE % (layer, projection, 'lora_B')] = (
                torch.randn(out_features, RUNTIME['rank'], generator=generator) * RUNTIME['b_std']
            ).to(torch.bfloat16)
    path.mkdir(parents=True, exist_ok=True)
    weights = path / 'adapter_model.safetensors'
    save_file(tensors, str(weights))
    (path / 'adapter_config.json').write_text(json.dumps({
        'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': RUNTIME['rank'],
        'lora_alpha': RUNTIME['rank'], 'lora_dropout': 0.0, 'bias': 'none', 'inference_mode': True,
        'base_model_name_or_path': 'gemma-4-31b-it-qat-w4a16-ct',
        'target_modules': supported, 'modules_to_save': [],
    }, indent=2) + '\n', encoding='utf-8')

    header = _read_header(weights)
    header.pop('__metadata__', None)
    tensor_names = sorted(header)
    a_keys = [name for name in tensor_names if 'lora_A' in name]
    covered = sorted(set(int(name.split('.layers.')[1].split('.')[0])
                         for name in tensor_names if '.layers.' in name))
    expected = len(supported) * layer_count
    if len(a_keys) != expected or len(covered) != layer_count:
        raise SystemExit('generated adapter covers ' + str(len(a_keys)) + ' A tensors across '
                         + str(len(covered)) + ' layers, expected ' + str(expected) + ' across '
                         + str(layer_count))
    return {
        'path': str(path),
        'tensors': len(tensor_names),
        'layers': len(covered),
        'rank': RUNTIME['rank'],
        'targets': supported,
        'example_key': a_keys[0],
        'bytes': weights.stat().st_size,
        'torch': torch.__version__,
        'diagnostics': diagnostics,
    }
