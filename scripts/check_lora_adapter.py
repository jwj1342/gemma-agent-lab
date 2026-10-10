"""Validate a LoRA adapter experiment without a GPU, base weights, or network access.

Checks the PEFT adapter config, the agent wiring, and the serving policy that the official harness
will apply, and re-derives the trainable-parameter and byte estimates from recorded evidence
instead of trusting values typed into documentation.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'resource/manifests/sample-adapters.json'
MODEL_MODULES = ROOT / 'resource/manifests/model-modules.json'
SERVER_SOURCE = ROOT / 'resource/harness-source/adk_submission/server.py'
BASE_MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
TENSOR_LAYER = 'base_model.model.model.language_model.layers.0.'
LORA_SUFFIXES = ('lora_A.weight', 'lora_B.weight')
PROJECTION_SUFFIXES = tuple('self_attn.%s_proj' % n for n in ('q', 'k', 'v', 'o')) + tuple(
    'mlp.%s_proj' % n for n in ('gate', 'up', 'down'))


class Include(str):
    """Marker for an unresolved ``!include`` value in an agent YAML file."""


def load_agent_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit('PyYAML is required: pip install -r configs/ci.requirements.txt') from error
    loader = yaml.SafeLoader

    class AgentLoader(loader):  # type: ignore[misc, valid-type]
        pass

    def construct_include(loader_, node):
        return Include(loader_.construct_scalar(node))

    AgentLoader.add_constructor('!include', construct_include)
    # utf-8-sig: tolerate a BOM, which plain utf-8 rejects.
    return yaml.load(path.read_text(encoding='utf-8-sig'), Loader=AgentLoader) or {}


def sample_module_shapes(sample_manifest: Path) -> tuple[dict[str, tuple[int, int]], int, str]:
    """Evidence from the shipped sample adapter: module paths, shapes, layer count, and sha256."""
    document = json.loads(sample_manifest.read_text(encoding='utf-8-sig'))
    if not document:
        raise SystemExit(f'no sample adapter recorded in {sample_manifest}')
    entry = document[0]
    in_features: dict[str, int] = {}
    out_features: dict[str, int] = {}
    for name, meta in entry['tensors'].items():
        if not name.startswith(TENSOR_LAYER):
            raise SystemExit(f'unexpected sample adapter tensor name: {name}')
        relative = name[len(TENSOR_LAYER):]
        for suffix in LORA_SUFFIXES:
            if relative.endswith('.' + suffix):
                module = relative[: -len(suffix) - 1]
                is_a = suffix == 'lora_A.weight'
                break
        else:
            raise SystemExit(f'unexpected sample adapter tensor name: {name}')
        if not module.endswith(PROJECTION_SUFFIXES):
            raise SystemExit(f'sample adapter uses an unrecognized projection: {module}')
        rows, columns = meta['shape']
        target = in_features if is_a else out_features
        value = columns if is_a else rows
        if target.setdefault(module, value) != value:
            raise SystemExit(f'inconsistent {suffix} shapes for {module}')
    if set(in_features) != set(out_features):
        raise SystemExit('the sample adapter does not pair lora_A with lora_B for every projection')
    shapes = {module: (in_features[module], out_features[module]) for module in in_features}
    layers = sorted({int(match.group(1)) for name in entry['tensors'] if (match := re.search(r'layers\.(\d+)\.', name))})
    if layers != [0]:
        raise SystemExit(f'expected only layer 0 in the sample adapter, found {layers}')
    return shapes, 60, entry['sha256']


def probe_module_shapes(model_modules: Path) -> tuple[dict[str, tuple[int, int]], int, str]:
    """Evidence from a real probe of the loaded base model (scripts/probe_model_modules.py)."""
    document = json.loads(model_modules.read_text(encoding='utf-8-sig'))
    projections = document.get('projections')
    if not isinstance(projections, dict) or not projections:
        raise SystemExit(f'no projections recorded in {model_modules}')
    shapes = {name: (int(entry['in_features']), int(entry['out_features']))
              for name, entry in projections.items()}
    layer_count = document.get('projection_layer_count')
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise SystemExit(f'no usable projection_layer_count in {model_modules}')
    return shapes, layer_count, document.get('base_model', 'unknown')


def module_evidence() -> tuple[dict[str, tuple[int, int]], int, str, str]:
    """Prefer a real probe of the loaded model over the shipped sample adapter."""
    if MODEL_MODULES.is_file():
        shapes, layer_count, base = probe_module_shapes(MODEL_MODULES)
        return shapes, layer_count, f'probe:{MODEL_MODULES.relative_to(ROOT)}', str(base)
    shapes, layer_count, digest = sample_module_shapes(MANIFEST)
    return shapes, layer_count, f'sample:{MANIFEST.relative_to(ROOT)}', digest


def vllm_rank_buckets(server_source: Path) -> tuple[tuple[int, ...], int]:
    text = server_source.read_text(encoding='utf-8')
    bucket = re.search(r'_VALID_VLLM_LORA_RANKS[^=]*=\s*\(([^)]*)\)', text)
    fallback = re.search(r'max_lora_rank:\s*int\s*=\s*(\d+)', text)
    if not bucket or not fallback:
        raise SystemExit(f'cannot read the rank policy from {server_source}')
    return tuple(int(value) for value in re.findall(r'\d+', bucket.group(1))), int(fallback.group(1))


def adapter_config_paths(experiment_dir: Path) -> list[Path]:
    found = sorted(experiment_dir.rglob('adapter_config.json'))
    if not found:
        raise SystemExit(f'no PEFT adapter_config.json under {experiment_dir}')
    return found


def check(experiment_dir: Path, adapter_name: str) -> dict:
    adapter_dir = experiment_dir / 'adapters' / adapter_name
    config_path = adapter_dir / 'adapter_config.json'
    if not config_path.is_file():
        raise SystemExit(f'missing {config_path}')
    unexpected = [p for p in adapter_config_paths(experiment_dir) if p != config_path]
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    weights = adapter_dir / 'adapter_model.safetensors'
    errors: list[str] = []
    if config.get('peft_type') != 'LORA':
        errors.append(f"peft_type must be LORA, got {config.get('peft_type')!r}")
    rank = config.get('r')
    if not isinstance(rank, int) or rank <= 0:
        errors.append(f'r must be a positive integer, got {rank!r}')
        rank = None
    alpha = config.get('lora_alpha')
    targets = config.get('target_modules')
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list) or not targets:
        errors.append(f'target_modules must be a nonempty list, got {targets!r}')
        targets = []
    if config.get('base_model_name_or_path') != BASE_MODEL:
        errors.append(f'base_model_name_or_path must be {BASE_MODEL}, got '
                      f'{config.get("base_model_name_or_path")!r}')
    if config.get('modules_to_save'):
        errors.append('modules_to_save is not supported by the competition adapter path')

    shapes, layer_count, evidence_source, evidence_base = module_evidence()
    # PEFT writes short module names; the safetensors/probe evidence carries full paths.
    short_names = {target: module for module, target in ((m, m.rsplit('.', 1)[-1]) for m in shapes)}
    unknown = sorted(set(targets) - set(short_names))
    if unknown:
        errors.append(f'target modules without recorded shape evidence: {unknown}')
    rank_pattern = config.get('rank_pattern') or {}
    if rank_pattern:
        errors.append('rank_pattern is not supported; the serving path reads a single r')

    missing_coverage = sorted(set(short_names) - set(targets))
    trainable = None
    if rank and not unknown:
        trainable = rank * sum(sum(shapes[short_names[module]]) for module in targets) * layer_count

    buckets, fallback = vllm_rank_buckets(SERVER_SOURCE)
    served_rank = None
    if rank:
        served_rank = next((b for b in buckets if b >= rank), fallback)

    declared: set[str] = set()
    for path in sorted(experiment_dir.rglob('*.yaml')):
        parsed = load_agent_yaml(path)
        if isinstance(parsed, dict) and parsed.get('adapter') is not None:
            declared.add(str(parsed['adapter']))
    if declared != {adapter_name}:
        errors.append(f'agent YAML must declare exactly adapter: {adapter_name}; found {sorted(declared) or "none"}')

    result = {
        'status': 'ok' if not errors else 'failed',
        'experiment_dir': str(experiment_dir),
        'adapter_name': adapter_name,
        'adapter_config': str(config_path.relative_to(ROOT)),
        'weights_present': weights.is_file(),
        'weights_bytes': weights.stat().st_size if weights.is_file() else 0,
        'rank': rank,
        'lora_alpha': alpha,
        'scaling': (alpha / rank) if (isinstance(alpha, (int, float)) and rank) else None,
        'target_modules': targets,
        'coverage': {
            'recorded_modules': sorted(short_names),
            'trained_modules': sorted(targets),
            'not_trained_but_recorded': sorted(set(short_names) - set(targets)),
            'unrecorded_modules': unknown,
        },
        'layer_count': layer_count,
        'trainable_parameters': trainable,
        'trainable_fraction_of_31b': (trainable / 31e9) if trainable else None,
        'adapter_bytes_bf16': (2 * trainable) if trainable else None,
        'serving_policy': {
            'buckets': list(buckets),
            'buckets_source': str(SERVER_SOURCE.relative_to(ROOT)),
            'harness_rank_fallback': fallback,
            'served_max_lora_rank': served_rank,
            'max_loras': len([p for p in adapter_config_paths(experiment_dir)]) if declared else 0,
        },
        'evidence': {'source': evidence_source, 'base_model_identity': evidence_base,
                     'probe_replaces_sample': MODEL_MODULES.is_file()},
        'errors': errors,
        'scope': 'Config, wiring, and serving policy only; no model load, no inference, no GPU',
    }
    if not weights.is_file():
        result['note'] = ('adapter_model.safetensors is absent: wiring only. Do not certify or submit '
                          'this experiment until a trained adapter is exported here.')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment-dir', type=Path,
                        default=ROOT / 'configs/experiments/lora-trajectory-v1')
    parser.add_argument('--adapter-name', default='main_lora')
    args = parser.parse_args()
    result = check(args.experiment_dir.resolve(), args.adapter_name)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
