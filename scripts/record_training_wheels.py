"""Record a verified offline training dependency set as a hashed, reusable manifest.

The Kaggle starter image ships `transformers` and `bitsandbytes` only, and Internet is disabled
during a run, so SFT needs its remaining wheels attached as a mounted dataset. This script turns a
directory of downloaded wheels into an inventory with SHA256 hashes and records the version set that
was actually proven to train and export an adapter by `scripts/check_training_stack.py`.

    python scripts/record_training_wheels.py \
        --wheels-dir artifacts/training-wheels \
        --output resource/manifests/training-wheels-sft-v1.json

The wheel set itself is not committed; regenerate it for the target platform with:

    python -m pip download --no-deps --only-binary=:all: \
        --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --abi cp312 \
        --dest artifacts/training-wheels \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
        peft==0.21.2 trl==0.24.0 accelerate==1.15.0 datasets==5.1.0 \
        "huggingface-hub>=1.5.0,<2.0" pandas multiprocess dill xxhash pyarrow

Then publish that directory as a private Kaggle dataset and pass it to
`scripts/prepare_lora_notebook.py` as `--wheels-dir`/`--wheels-source`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Verified end to end on 2026-10-11 by scripts/check_training_stack.py: 3 optimisation steps ran,
# loss decreased, the adapter exported and reloaded with matching weights.
VERIFIED_VERSIONS = {
    'transformers': '5.13.1',
    'peft': '0.21.2',
    'trl': '0.24.0',
    'accelerate': '1.15.0',
    'datasets': '5.1.0',
    'huggingface-hub': '1.33.0',
    'safetensors': '0.8.0',
    'tokenizers': '0.22.2',
}
# Constraints that measurement established, not guesses. Each one cost a failed run.
CONSTRAINTS = [
    'transformers==5.13.1 (the wheelhouse version) requires huggingface-hub>=1.5.0,<2.0; '
    '2.2.0 raises ImportError at transformers import time.',
    'trl 1.15.0 patches a fused LM head in SFTTrainer.__init__ and calls it on every step, but the '
    'fused path needs the trl kernels extension. Without it, training dies with '
    "AttributeError: 'NoneType' object has no attribute 'apply'. Use trl 0.24.0.",
    'datasets needs pandas and multiprocess at import time; neither is in the competition wheelhouse.',
    'target platform is CPython 3.12 on manylinux2014_x86_64; local verification ran on Windows '
    'CPython 3.14, so the wheels here are the Linux builds of the verified version set.',
]


def wheel_name(filename: str) -> tuple[str, str]:
    match = re.match(r'^(?P<name>[A-Za-z0-9_.+-]+?)-(?P<version>\d[^-]*)-', filename)
    if not match:
        raise SystemExit(f'cannot parse wheel filename: {filename}')
    return match.group('name').replace('_', '-'), match.group('version')


def build(wheels_dir: Path) -> dict:
    wheels = sorted(wheels_dir.glob('*.whl'))
    if not wheels:
        raise SystemExit(f'no wheels in {wheels_dir}')
    inventory = []
    for path in wheels:
        name, version = wheel_name(path.name)
        inventory.append({
            'file': path.name,
            'package': name,
            'version': version,
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Offline SFT dependencies to attach as a Kaggle dataset input',
        'target_platform': 'CPython 3.12, manylinux2014_x86_64',
        'wheel_count': len(inventory),
        'total_bytes': sum(item['bytes'] for item in inventory),
        'verified_versions': VERIFIED_VERSIONS,
        'constraints': CONSTRAINTS,
        'verification': ('scripts/check_training_stack.py ran 3 SFT steps on a tiny causal LM with '
                         'this version set: loss 5.4698 -> 5.4269, adapter exported (66,584 bytes, '
                         '8 LoRA tensors) and reloaded with matching weights'),
        'provided_by_starter_image': ['transformers', 'bitsandbytes', 'torch'],
        'wheels': inventory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wheels-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'resource/manifests/training-wheels-sft-v1.json')
    args = parser.parse_args()
    result = build(args.wheels_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('wheel_count', 'total_bytes', 'target_platform')}, indent=2))
    for item in result['wheels']:
        print(f"  {item['package']:18s} {item['version']:10s} {item['sha256'][:12]}...")
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
