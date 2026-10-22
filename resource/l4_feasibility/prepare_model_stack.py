"""Make the mounted wheelhouse's model stack effective, then prove it imports.

Why this step exists (measured, not assumed -- see ``WORKLOG.md`` and
``docs/experiments/2026-10-12-l4-training-feasibility.md``):

* Run 4 reached the base load and failed there with
  ``ValueError: The checkpoint you are trying to load has model type 'gemma4' but Transformers does not
  recognize this architecture``, raised as ``KeyError: 'gemma4'`` from ``CONFIG_MAPPING``: the
  competition image ships ``transformers 5.0.0``, which predates Gemma 4. The mounted wheelhouse
  carries ``transformers 5.13.1``, whose wheel contains ``transformers/models/gemma4/``.
* ``peft 0.19.1`` raises ``ImportError: Found an incompatible version of torchao. Found version 0.10.0,
  but only versions above 0.16.0 are supported`` while the image's ``torchao 0.10.0`` is installed:
  ``peft/import_utils.py::is_torchao_available()`` returns ``False`` only when torchao is *absent* and
  raises for anything below 0.16.0. So transformers 5.13.1 cannot be installed on its own -- torchao has
  to move to >= 0.16.0 in the same step.
* ``compressed_tensors`` and ``loguru`` must be importable before ``CompressedTensorsConfig(...)`` can
  be constructed at all: its ``__init__`` raises without the package, and
  ``compressed_tensors/logger.py`` imports ``loguru`` at module scope.

Four of those five wheels are on the wheelhouse mount. The torchao wheel is not, so it arrives through a
second dataset source; that dataset is third-party, so its bytes are checked against the PyPI SHA256
before installation rather than trusted because of its name.

Installation is offline by construction (``--no-index --no-deps`` from mounted files) and happens only
when the installed version is below the mounted one, so a re-run records "already satisfied" instead of
reinstalling. The chain is then proved in a *fresh* interpreter -- transformers, the gemma4 config
mapping, peft, torchao, the compressed-tensors quantiser config, and ``AutoConfig.from_pretrained`` on
the real checkpoint -- because ``importlib.metadata`` caches directory listings inside a process and the
check must not be able to read a stale one.

    python prepare_model_stack.py --wheelhouse /kaggle/input/<wheelhouse> \
        --torchao-search /kaggle/input --base /kaggle/input/<base>/2 --work /kaggle/working/l4-feasibility
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

# The model stack the training worker needs, as (distribution, wheel glob on the wheelhouse mount).
# `peft` and `accelerate` deliberately stay at the image's versions: the wheelhouse carries neither, and
# the recorded serving measurement was taken with peft 0.19.1.
MODEL_STACK = (
    ('transformers', 'transformers-*.whl'),
    ('tokenizers', 'tokenizers-*.whl'),
    ('safetensors', 'safetensors-*.whl'),
    ('compressed_tensors', 'compressed_tensors-*.whl'),
    ('loguru', 'loguru-*.whl'),
)

# The one wheel that is not on the wheelhouse mount, with the digest PyPI publishes for it. The copy in
# the attached public dataset was hashed to this value locally before this step existed.
TORCHAO_PATTERN = 'torchao-0.16.0-*.whl'
TORCHAO_SHA256 = {
    'torchao-0.16.0-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl':
        '2d6293a0c57c9dd505efb025a7189459d154965fbed000efd638cf299f9362dd',
}
TORCHAO_MINIMUM = '0.16.0'

# Run in a fresh interpreter: it must observe the tree as it is *after* the installs above.
VERIFY_SOURCE = r'''
import importlib.metadata, json, sys

base = sys.argv[1]
report = {'base': base}


def version(distribution):
    for spelling in (distribution, distribution.replace('_', '-')):
        try:
            return importlib.metadata.version(spelling)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


import transformers
report['transformers_version'] = transformers.__version__
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
report['gemma4_in_config_mapping'] = 'gemma4' in CONFIG_MAPPING
# Membership is not enough: resolution is what run 4 failed at, and it also imports
# transformers.models.gemma4 through the lazy mapping. `_LazyConfigMapping.__contains__` delegates to
# its mapping dict, so the membership test above is meaningful too.
try:
    report['gemma4_config_class'] = CONFIG_MAPPING['gemma4'].__name__
except Exception as error:  # noqa: BLE001 - the exact failure is the measurement
    report['gemma4_config_class'] = f'{type(error).__name__}: {error}'

import peft
report['peft_version'] = version('peft')
from peft.import_utils import is_torchao_available
report['torchao_version'] = version('torchao')
report['is_torchao_available'] = is_torchao_available()
report['compressed_tensors_version'] = version('compressed_tensors')
report['loguru_version'] = version('loguru')

# The quantiser config the recipe builds for the BF16 dense base. The keyword differs by transformers
# version -- 5.13.1 takes `run_compressed` and has no `dequantize` -- and the wrong keyword is dropped
# silently rather than raising, so the shared helper builds it and refuses an effective mode that
# disagrees with the request. Constructing it at all is also the gate that failed in an earlier round with
# `compressed-tensors>=0.15.0 is required`.
sys.path.insert(0, sys.argv[2])
from quantization_config_compat import build_quantization_config
from transformers import CompressedTensorsConfig
quantization, quantization_record = build_quantization_config(CompressedTensorsConfig, True)
report['quantization'] = quantization_record

# The real checkpoint, config only: no weight is read, and this is the call that failed in run 4.
from transformers import AutoConfig
config = AutoConfig.from_pretrained(base)
report['config_class'] = type(config).__name__
report['config_model_type'] = getattr(config, 'model_type', None)

print(json.dumps(report))
'''


def normalise(name: str) -> str:
    """Normalise a distribution name the way packaging does, for comparison only."""
    return re.sub(r'[-_.]+', '-', name).lower()


def version_key(text: str) -> tuple:
    """Order distribution versions numerically.

    Only numeric components are compared: every version this step deals with is numeric-dotted
    (`5.0.0`, `5.13.1`, `0.15.0.1`), and a suffix such as `post1` keeps its numeric part.
    """
    parts = []
    for chunk in str(text).replace('-', '.').split('.'):
        digits = ''.join(character for character in chunk if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def wheel_dist_version(filename: str) -> tuple:
    """Read ``(distribution, version)`` from a wheel filename, or ``(None, None)``."""
    stem = filename[:-4] if filename.endswith('.whl') else filename
    bits = stem.split('-')
    if len(bits) < 2 or not bits[0] or not bits[1]:
        return None, None
    return bits[0], bits[1]


def installed_version(distribution: str) -> str | None:
    """The installed version of a distribution, without importing it."""
    for spelling in (distribution, distribution.replace('_', '-')):
        try:
            return importlib_metadata.version(spelling)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def resolve_wheel(search_root: Path, pattern: str) -> list[Path]:
    """Find a wheel under a Kaggle input tree without walking a whole mounted model.

    Kaggle has produced three layouts for the same dataset: `/kaggle/input/<slug>/<wheel>`,
    `/kaggle/input/datasets/<owner>/<slug>/<wheel>`, and the wheelhouse's extra nested copy. A bounded
    depth walk covers all of them and stops instead of descending into a 62 GB model mount.
    """
    found = []
    if not search_root.is_dir():
        return found
    for depth in range(0, 4):
        found.extend(sorted(search_root.glob('/'.join(['*'] * depth + [pattern]))))
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def install_plan(wheelhouse: Path, torchao_wheel: Path | None, installed=installed_version) -> dict:
    """Decide which wheels to install. No file is written and no process is started.

    Kept free of installation so the decision rule can be tested with fabricated wheels: a wheel is
    installed when its distribution is missing or older than the mounted copy, and skipped otherwise.
    """
    entries, problems = [], {}
    for distribution, pattern in MODEL_STACK:
        candidates = sorted(wheelhouse.glob(pattern)) if wheelhouse.is_dir() else []
        current = installed(distribution)
        entry = {'distribution': distribution, 'pattern': pattern, 'installed_before': current,
                 'candidates': [path.name for path in candidates], 'wheel': None,
                 'target_version': None, 'sha256': None, 'expected_sha256': None, 'provenance': None}
        if not candidates:
            problems[distribution] = f'no wheel matching {pattern} is mounted'
            entry['action'] = 'blocked'
            entries.append(entry)
            continue
        wheel = candidates[0]
        entry['wheel'] = wheel.name
        entry['target_version'] = wheel_dist_version(wheel.name)[1]
        needed = current is None or version_key(current) < version_key(entry['target_version'] or '0')
        entry['action'] = 'install' if needed else 'already_satisfied'
        entry['reason'] = ((f'not installed, mounting {entry["target_version"]}' if current is None else
                            f'installed {current} is below the mounted {entry["target_version"]}') if needed
                           else f'installed {current} is at or above the mounted {entry["target_version"]}')
        entries.append(entry)

    current = installed('torchao')
    torchao = {'distribution': 'torchao', 'pattern': TORCHAO_PATTERN, 'installed_before': current,
               'candidates': [torchao_wheel.name] if torchao_wheel else [], 'wheel': None,
               'target_version': None, 'sha256': None, 'expected_sha256': None, 'provenance': None}
    if torchao_wheel is None:
        problems['torchao'] = f'no wheel matching {TORCHAO_PATTERN} is mounted'
        torchao['action'] = 'blocked'
    else:
        torchao['wheel'] = torchao_wheel.name
        torchao['target_version'] = wheel_dist_version(torchao_wheel.name)[1]
        expected = TORCHAO_SHA256.get(torchao_wheel.name)
        torchao['expected_sha256'] = expected
        if expected is None:
            problems['torchao'] = (f'{torchao_wheel.name} has no recorded PyPI digest, so its '
                                   'provenance cannot be established')
            torchao['action'] = 'blocked'
        else:
            torchao['sha256'] = sha256_file(torchao_wheel)
            if torchao['sha256'] != expected:
                problems['torchao'] = (f'sha256 {torchao["sha256"]} does not match the recorded PyPI '
                                       f'digest {expected}')
                torchao['action'] = 'blocked'
            else:
                needed = current is None or version_key(current) < version_key(TORCHAO_MINIMUM)
                torchao['action'] = 'install' if needed else 'already_satisfied'
                torchao['reason'] = (f'installed {current or "nothing"} is below the {TORCHAO_MINIMUM} '
                                     'floor peft enforces') if needed else \
                                    f'installed {current} already satisfies the floor'
                torchao['provenance'] = ('public Kaggle dataset kajarnaklokossou/torchao-0-16-0; bytes '
                                         'match the PyPI digest recorded here')
    entries.append(torchao)
    return {'entries': entries, 'problems': problems}


def run_install(wheel: Path) -> dict:
    """Install one mounted wheel offline, recording the return code and the tail of any failure."""
    completed = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
         '--no-build-isolation', '--only-binary=:all:', '--retries', '0', '--timeout', '15', str(wheel)],
        capture_output=True, text=True)
    return {'returncode': completed.returncode,
            'tail': (completed.stderr or completed.stdout)[-600:] if completed.returncode else ''}


def apply_installs(plan: dict, wheelhouse: Path, torchao_wheel: Path | None) -> None:
    """Install every entry the plan marked for installation, in plan order."""
    for entry in plan['entries']:
        if entry['action'] != 'install':
            continue
        wheel = torchao_wheel if entry['distribution'] == 'torchao' else wheelhouse / entry['wheel']
        entry['install'] = run_install(wheel)


def run_verification(base: str) -> dict:
    """Prove the import chain in a fresh interpreter and return its report."""
    completed = subprocess.run(
        [sys.executable, '-c', VERIFY_SOURCE, str(base), str(Path(__file__).resolve().parent)],
        capture_output=True, text=True)
    report: dict = {'returncode': completed.returncode}
    if completed.returncode == 0:
        try:
            report['observed'] = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            report['returncode'] = 9
            report['error'] = 'the verification printed no JSON report'
            report['stdout_tail'] = completed.stdout[-600:]
    else:
        report['stderr_tail'] = completed.stderr[-1500:]
    return report


def collect_problems(plan: dict) -> dict:
    """The real blocking problems: an unmounted wheel, a failed hash, or a failed install.

    Kept separate from :func:`judge` because the caller must ask "is anything actually wrong?" without a
    missing verification being reported as a problem. Folding that into one return value is what made the
    first session of this step skip its own verification: `judge(plan, None)['problems']` was never empty,
    so the guard never let the verification run.
    """
    problems = dict(plan['problems'])
    for entry in plan['entries']:
        failed = entry.get('install', {}).get('returncode')
        if failed:
            problems[entry['distribution']] = f'pip install returned {failed}: {entry["install"]["tail"]}'
    return problems


def judge(plan: dict, verification: dict | None) -> dict:
    """Turn the plan and the verification into a verdict, separating dependency from import failure."""
    problems = collect_problems(plan)
    if verification is None:
        return {'verdict': 'blocked', 'problems': problems}
    observed = verification.get('observed') or {}
    checks = {
        'transformers_import': verification['returncode'] == 0,
        'gemma4_in_config_mapping': bool(observed.get('gemma4_in_config_mapping')),
        'gemma4_config_resolves': observed.get('gemma4_config_class') == 'Gemma4Config',
        'torchao_satisfies_peft_floor': bool(observed.get('is_torchao_available')),
        'dequantize_mode_is_effective': (observed.get('quantization') or {}).get(
            'effective_dequantize') is True,
        'config_model_type_is_gemma4': observed.get('config_model_type') == 'gemma4',
    }
    for name, passed in checks.items():
        if not passed:
            problems[name] = verification.get('stderr_tail') or f'{name} is false'
    return {'verdict': 'ok' if not problems else 'blocked', 'problems': problems, 'checks': checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--torchao-search', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--torchao-wheel', type=Path, default=None,
                        help='Explicit torchao wheel; overrides --torchao-search')
    parser.add_argument('--base', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true',
                        help='Plan and verify hashes only; install nothing and import nothing')
    args = parser.parse_args()

    torchao_wheel = args.torchao_wheel
    candidates = [] if torchao_wheel else resolve_wheel(args.torchao_search, TORCHAO_PATTERN)
    if torchao_wheel is None and candidates:
        torchao_wheel = candidates[0]
    result: dict = {'wheelhouse': str(args.wheelhouse), 'base': args.base,
                    'wheelhouse_exists': args.wheelhouse.is_dir(),
                    'wheelhouse_wheel_count': len(list(args.wheelhouse.glob('*.whl')))
                    if args.wheelhouse.is_dir() else 0,
                    'torchao_candidates': [str(path) for path in candidates],
                    'torchao_wheel': str(torchao_wheel) if torchao_wheel else None}

    plan = install_plan(args.wheelhouse, torchao_wheel)
    if not args.dry_run:
        apply_installs(plan, args.wheelhouse, torchao_wheel)
        blocking = collect_problems(plan)
        verification = None if blocking else run_verification(args.base)
        verdict = judge(plan, verification)
        result['verification'] = verification
        result['checks'] = verdict.get('checks')
        result['problems'] = verdict['problems']
        result['verdict'] = verdict['verdict']
    else:
        result['problems'] = plan['problems']
        result['verdict'] = 'planned' if not plan['problems'] else 'blocked'
    result['entries'] = plan['entries']

    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / 'model-stack.json').write_text(json.dumps(result, indent=2, default=str) + '\n',
                                                encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'entries'}, indent=2,
                     default=str), flush=True)
    return 0 if result['verdict'] in ('ok', 'planned') else 3


if __name__ == '__main__':
    raise SystemExit(main())
