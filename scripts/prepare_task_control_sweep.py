"""Generate a private, CPU-only Kaggle notebook that validates public-task controls.

Why this exists
---------------
`swegemma` runs with `sandbox='subprocess'` on Kaggle (Docker is unavailable). In that mode
`install_test_dependencies()` returns immediately and the subprocess sandbox only uses its
`wheels_dir` for `/wheels` path translation, so it never installs anything: the sandbox venv is
created with `system_site_packages=True` plus a `_host_env.pth` that points at the notebook
interpreter's site-packages. Repository test dependencies therefore have to exist in the
notebook interpreter, and the public wheelhouse does not contain them.

This generator produces a self-contained notebook that
  1. installs a reviewed, hash-verified set of test-only wheels offline (`--no-index --no-deps`),
  2. records interpreter/package provenance, and
  3. runs the official empty and gold controls for each task with no model call.

A task is usable for the agent comparison only when its empty patch fails and its gold patch
passes. Everything is written under the notebook working directory for download.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from test_dependency_payload import (  # noqa: E402
    build_test_dependency_blob,
    install_cell_source,
    official_wheelhouse_cell,
)

DATA_PRIMARY = '/kaggle/input/competitions/gemma-4-developer-agent'
DATA_FALLBACK = '/kaggle/input/gemma-4-developer-agent'
DATA_LEGACY = '/kaggle/input/competition-data'
HARNESS_WHEELS_PRIMARY = '/kaggle/input/gemma-4-developer-agent-wheelhouse'
HARNESS_WHEELS_FALLBACK = '/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'
WORK = '/kaggle/working/task-control-sweep'
# The reviewed harness wheels the organisers have replaced in place at least once. Their reviewed
# bytes come from the private publisher kernel `junzhejiao/gemma4-pinned-wheels`; without restoring
# them the harness cannot import `swegemma.models.registry` and every control arm returns no result.
PINNED_HARNESS_WHEELS = (
    'swegemma-0.2.10-py3-none-any.whl',
    'adk_eval_core-0.1.0-py3-none-any.whl',
    'adk_submission-0.2.13-py3-none-any.whl',
)

PROBE_MODULES = [
    'pytest', 'pluggy', 'pydantic', 'starlette', 'fastapi', 'requests', 'httpx', 'rich',
    'dirty_equals', 'inline_snapshot', 'typing_inspection', 'flask', 'werkzeug',
    'ujson', 'orjson', 'multipart',
]


def cell(kind: str, source: str) -> dict:
    body = {'cell_type': kind, 'metadata': {}, 'source': source}
    if kind == 'code':
        body.update({'execution_count': None, 'outputs': []})
    return body


SETUP = f'''import base64, gzip, json
from pathlib import Path


def _resolve_harness_wheels():
    """Find the attached official wheelhouse dataset mount, or explain what is mounted."""
    candidates = [
        Path('{HARNESS_WHEELS_PRIMARY}'),
        Path('{HARNESS_WHEELS_FALLBACK}'),
        Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
        Path('/kaggle/input/gemma-4-developer-agent-wheelhouse'),
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob('*.whl')):
            return candidate
    available = sorted(str(path) for path in Path('/kaggle/input').glob('*'))
    nested = sorted(str(path) for path in Path('/kaggle/input').glob('datasets/*'))
    raise AssertionError('official wheelhouse dataset not mounted; /kaggle/input: '
                         + repr(available) + ' datasets: ' + repr(nested))


WORK = Path('{WORK}')
DATA = Path('{DATA_PRIMARY}')
if not DATA.is_dir():
    DATA = Path('{DATA_FALLBACK}')
if not DATA.is_dir():
    DATA = Path('{DATA_LEGACY}')
HARNESS_WHEELS = _resolve_harness_wheels()
WHEELS = DATA / 'wheels'
mounts = {{'data': DATA.is_dir(), 'task_wheels': WHEELS.is_dir(),
          'harness_wheels': HARNESS_WHEELS.is_dir()}}
assert all(mounts.values()), 'Required Kaggle inputs missing: ' + repr(mounts)
print('Input mounts:', {{'data': str(DATA), 'task_wheels': str(WHEELS),
                        'harness_wheels': str(HARNESS_WHEELS)}}, flush=True)
assert not WORK.exists(), 'Runtime output already exists; export it before rerunning.'
(WORK / 'inputs').mkdir(parents=True)
(WORK / 'scripts').mkdir(exist_ok=True)
for name, source in PAYLOAD['scripts'].items():
    (WORK / 'scripts' / name).write_text(source, encoding='utf-8')
AGENT = WORK / 'inputs' / 'agent'
(AGENT / 'configs').mkdir(parents=True, exist_ok=True)
(AGENT / 'eval_config.yaml').write_text(PAYLOAD['eval_config'], encoding='utf-8')
print('Prepared runner scripts; the reviewed test-only wheels install in the next cell.', flush=True)
print('Task ids:', PAYLOAD['task_ids'], flush=True)
'''

PROBE = '''# Record interpreter/package versions so the sweep stays comparable to the image baseline.
import importlib.metadata as md, json, platform, sys
from pathlib import Path
versions = {}
for name in PROBE_MODULES:
    try:
        versions[name] = md.version(name)
    except Exception:
        versions[name] = None
provenance = {
    'recorded_at': __import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(),
    'python': sys.version,
    'platform': platform.platform(),
    'packages': versions,
    'test_deps_manifest_sha256': PAYLOAD['test_deps']['manifest_sha256'],
    'installed': [item['name'] for item in PAYLOAD['test_deps']['install']],
}
Path(WORK / 'environment-provenance.json').write_text(json.dumps(provenance, indent=2) + '\\n')
print(json.dumps(versions, indent=2))
'''

OFFICIAL_WHEELS_NOTE = '''# Record the mounted wheelhouse and the platform setup script before running controls.
import hashlib, json, subprocess, sys
from pathlib import Path
mounted = sorted(path.name for path in WHEELS.glob('*.whl'))
inventory = {'mounted_count': len(mounted), 'mounted': mounted,
             'reviewed_harness_manifest_sha256': PAYLOAD['wheel_manifest_sha256']}
(WORK / 'mounted-wheelhouse.json').write_text(json.dumps(inventory, indent=2) + '\\n')
print('mounted wheelhouse files:', len(mounted), flush=True)
missing_reviewed = sorted(set(PAYLOAD['wheels']) - set(mounted))
report = {'reviewed_harness_wheels_present': len(PAYLOAD['wheels']) - len(missing_reviewed),
          'reviewed_harness_wheels_missing': missing_reviewed}
(WORK / 'wheelhouse-check.json').write_text(json.dumps(report, indent=2) + '\\n')
print(json.dumps(report, indent=2)[:2000], flush=True)
setup_script = DATA / 'sandbox' / 'setup.py'
if setup_script.is_file():
    (WORK / 'setup-script.py').write_bytes(setup_script.read_bytes())
    print('sandbox/setup.py bytes:', setup_script.stat().st_size,
          'sha256:', hashlib.sha256(setup_script.read_bytes()).hexdigest(), flush=True)
    probe = subprocess.run([sys.executable, str(setup_script), '--fast-path'],
                           capture_output=True, text=True, cwd=str(WORK))
    (WORK / 'setup-fast-path.log').write_text(
        'returncode: %d\\n--- stdout ---\\n%s\\n--- stderr ---\\n%s\\n'
        % (probe.returncode, probe.stdout[-4000:], probe.stderr[-4000:]))
    print('setup.py --fast-path returncode:', probe.returncode, flush=True)
    print(probe.stdout[-2000:], probe.stderr[-2000:], flush=True)
else:
    print('no sandbox/setup.py in the mounted competition data', flush=True)
'''

VERIFY_INSTALL = '''# Import check before spending time on the whole sweep.
import importlib, json
modules = ['swegemma', 'adk_submission', 'adk_eval_core', 'google.adk', 'vllm', 'torch',
           'litellm', 'yaml']
report = {}
for name in modules:
    try:
        importlib.import_module(name)
        report[name] = 'ok'
    except Exception as error:
        report[name] = f'{type(error).__name__}: {error}'
print(json.dumps(report, indent=2))
(WORK / 'import-check.json').write_text(json.dumps(report, indent=2) + '\\n')
missing = [name for name, state in report.items() if state != 'ok']
assert not missing, 'required modules failed to import: ' + repr(missing)
'''

HIDE_IPYTHON = '''# IPython registers extra Pygments lexers, which changes `Syntax.guess_lexer` results and breaks
# reference patches that expect the plain `python` lexer. A public end-to-end comparison records
# the same workaround. Only the installed distributions are removed; nothing else is changed.
import json, shutil, site, sys
from pathlib import Path
hidden = []
for distribution in sorted(Path(site.getsitepackages()[0]).glob('IPython*')):
    shutil.rmtree(distribution, ignore_errors=True)
    hidden.append(distribution.name)
for path in sorted(Path(site.getsitepackages()[0]).glob('ipython*')):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        hidden.append(path.name)
for name in ('IPython', 'ipython'):
    for base in (Path('/usr/local/lib'), Path('/usr/lib')):
        for candidate in base.glob(f'python*/dist-packages/{name}*'):
            shutil.rmtree(candidate, ignore_errors=True)
            hidden.append(str(candidate))
for name in ('IPython', 'ipython', 'IPython.lib', 'IPython.core'):
    sys.modules.pop(name, None)
report = {'hidden': hidden}
(WORK / 'hidden-ipython.json').write_text(json.dumps(report, indent=2) + '\\n')
print('hidden IPython distributions:', hidden, flush=True)
'''

PIN_TASK_ENVIRONMENT = '''# The harness installs test dependencies only in its Docker backend. On the subprocess backend
# the notebook interpreter keeps the highest Starlette in the task wheelhouse (1.6.0), which makes
# FastAPI snapshots fail at collection with "Router.__init__() got an unexpected keyword argument
# 'on_startup'". Pin the framework version each task declares, from the same wheelhouse.
import io, json, re, shutil, subprocess, sys, tarfile
from pathlib import Path

REQUIREMENT_RE = re.compile(r'(?m)^\\s*["\\']?(?P<name>[A-Za-z0-9_.\\-]+)\\s*'
                            r'(?P<operator>>=|<=|==|~=|!=|>|<)\\s*(?P<version>[0-9][0-9A-Za-z.\\-]*)\\s*["\\']?\\s*,?\\s*$')
# Only Starlette is repinned. Repinning pydantic would install a version whose pydantic-core
# binding is not in the task wheelhouse (pydantic 2.13.5 against pydantic-core 2.46.4), which
# breaks litellm for every task and makes all controls fail before any test runs.
TARGET_PACKAGES = {'starlette': 'starlette'}


def _task_metadata(task_id, names):
    snapshot = DATA / 'snapshots' / (task_id + '.tgz')
    if not snapshot.is_file():
        return {}
    found = {}
    with tarfile.open(snapshot, 'r:gz') as archive:
        for member in archive.getmembers():
            base = member.name.rsplit('/', 1)[-1]
            if member.isfile() and base in names and len(member.name.split('/')) <= 3:
                try:
                    found[base] = archive.extractfile(member).read().decode('utf-8', 'ignore')
                except Exception:
                    continue
    return found


def _parse_version(text):
    return tuple(int(part) for part in re.findall(r'\\d+', text)[:3]) or (0,)


def _available(package):
    out = []
    for path in WHEELS.glob(package + '-*.whl'):
        version = path.name.split('-')[1]
        out.append((_parse_version(version), version, path))
    return sorted(out, key=lambda item: item[0])


def _choose(package, constraint, operator):
    candidates = _available(package)
    # Starlette 1.x removed Router(on_startup=...); never select it for these snapshots.
    if package == 'starlette':
        candidates = [item for item in candidates if item[0] < (1,)]
    if not candidates:
        return None
    if operator in ('>=', '>', '~='):
        wanted = _parse_version(constraint)
        eligible = [item for item in candidates if item[0] >= wanted]
        return eligible[-1] if eligible else candidates[0]
    wanted = _parse_version(constraint)
    eligible = [item for item in candidates if item[0] <= wanted]
    return eligible[-1] if eligible else candidates[0]


def _metadata_text(task_id):
    return _task_metadata(task_id, {'pyproject.toml', 'setup.py', 'requirements.txt',
                                    'requirements-tests.txt'})


records = []
for task_id in PAYLOAD['task_ids']:
    entry = {'task_id': task_id, 'files': [], 'pinned': {}, 'error': None}
    try:
        texts = _metadata_text(task_id)
        entry['files'] = sorted(texts)
        installed = {}
        for line in '\\n'.join(texts.values()).splitlines():
            match = REQUIREMENT_RE.search(line)
            if not match:
                continue
            name = match.group('name').lower().replace('_', '-')
            if name not in TARGET_PACKAGES or name in installed:
                continue
            chosen = _choose(TARGET_PACKAGES[name], match.group('version'), match.group('operator'))
            if chosen is None:
                continue
            target = Path('/tmp/pinned-wheels'); target.mkdir(exist_ok=True)
            link = target / chosen[2].name
            if not link.exists():
                link.symlink_to(chosen[2])
            completed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                                        '--no-deps', '--force-reinstall', str(link)],
                                       capture_output=True, text=True)
            installed[name] = {'requested': match.group('operator') + match.group('version'),
                               'installed': chosen[1], 'returncode': completed.returncode}
            if completed.returncode:
                installed[name]['detail'] = (completed.stdout or '')[-200:]
        entry['pinned'] = installed
    except Exception as error:
        entry['error'] = f'{type(error).__name__}: {error}'
    records.append(entry)
    print(task_id, json.dumps(entry['pinned'] or entry['error']), flush=True)
    (WORK / 'task-environment-pins.json').write_text(json.dumps(records, indent=2) + '\\n')
(WORK / 'task-environment-pins.json').write_text(json.dumps(records, indent=2) + '\\n')
'''

SWEEP = '''# Empty and gold controls per task. No model, no inference, no competition submission.
import json, os, subprocess, sys
from pathlib import Path
WORK = Path(WORK); DATA = Path(DATA)
AGENT = WORK / 'inputs' / 'agent'
TASKS = WORK / 'inputs' / 'tasks.jsonl'
TASKS.write_bytes(DATA.joinpath('tasks.jsonl').read_bytes())


def run_control(task_id, source):
    results = WORK / 'controls' / task_id / source
    stage = WORK / 'controls' / task_id
    for name in ('home', 'tmp', 'cache'):
        (stage / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(HOME=str(stage / 'home'), TMPDIR=str(stage / 'tmp'),
               XDG_CACHE_HOME=str(stage / 'cache'), HF_HUB_OFFLINE='1', PIP_NO_INDEX='1',
               OTEL_SDK_DISABLED='true', LITELLM_LOCAL_MODEL_COST_MAP='True')
    command = [sys.executable, str(WORK / 'scripts' / 'evaluate_agent.py'), 'verify',
               '--task-id', task_id, '--tasks', str(TASKS), '--data-dir', str(DATA),
               '--agent-dir', str(AGENT), '--patch-source', source, '--results', str(results)]
    log = results.parent / (source + '.log')
    with log.open('w') as stream:
        completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = {'task_id': task_id, 'patch_source': source, 'returncode': completed.returncode}
    result_path = results / 'result.json'
    if result_path.is_file():
        payload = json.loads(result_path.read_text())
        text = payload.get('test_output') or ''
        record.update({
            'status': payload.get('status'), 'resolved': payload.get('resolved'),
            'test_exit_code': payload.get('test_exit_code'),
            'nonempty_patch': bool(payload.get('agent_patch')),
            'duration_seconds': payload.get('duration_seconds'),
            'collection_error': 'ERROR collecting' in text or 'errors during collection' in text,
            'module_error': 'ModuleNotFoundError' in text,
            'error_message': payload.get('error_message'),
        })
    else:
        failure = results.parent / source / 'failure.json'
        record['status'] = 'NO_RESULT_FILE'
        if failure.is_file():
            detail = json.loads(failure.read_text())
            record['error_type'] = detail.get('error_type')
            record['error_message'] = detail.get('message')
            record['phase'] = detail.get('phase')
    return record


records = []
for index, task_id in enumerate(PAYLOAD['task_ids'], start=1):
    for source in ('empty', 'gold'):
        print(f'[{index}/{len(PAYLOAD["task_ids"])}] {task_id} {source}', flush=True)
        try:
            record = run_control(task_id, source)
        except Exception as error:
            record = {'task_id': task_id, 'patch_source': source,
                      'status': 'RUN_EXCEPTION', 'returncode': None,
                      'error_message': f'{type(error).__name__}: {error}'}
        records.append(record)
        print('   ->', json.dumps(record), flush=True)
        (WORK / 'control-records.json').write_text(json.dumps(records, indent=2) + '\\n')

summary = {}
for record in records:
    summary.setdefault(record['task_id'], {})[record['patch_source']] = record
usable, gold_failed, empty_passed = [], [], []
for task_id, entry in sorted(summary.items()):
    empty = entry.get('empty', {})
    gold = entry.get('gold', {})
    valid = (empty.get('returncode') == 0 and empty.get('resolved') is False
             and gold.get('returncode') == 0 and gold.get('resolved') is True)
    entry['controls_valid'] = valid
    if valid:
        usable.append(task_id)
    elif gold.get('resolved') is not True:
        gold_failed.append(task_id)
    if empty.get('resolved') is True:
        empty_passed.append(task_id)
report = {
    'recorded_at': __import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(),
    'task_count': len(PAYLOAD['task_ids']),
    'usable_task_ids': usable, 'gold_failed_task_ids': gold_failed,
    'empty_passed_task_ids': empty_passed,
    'usable_count': len(usable), 'tasks': summary,
}
(WORK / 'control-sweep-report.json').write_text(json.dumps(report, indent=2) + '\\n')
print('USABLE', len(usable), 'of', len(PAYLOAD['task_ids']))
print(json.dumps({'usable': usable, 'gold_failed': gold_failed, 'empty_passed': empty_passed}))
'''


def build_pinned_blob(pinned_dir: Path, names) -> tuple:
    """Pack the reviewed harness wheels the organisers replaced, hash-checked against their files."""
    pinned_dir = Path(pinned_dir)
    install, buffer = [], io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name in names:
            path = pinned_dir / name
            if not path.is_file():
                raise SystemExit(f'reviewed harness wheel is missing: {path}')
            payload = path.read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
            install.append({'name': name, 'bytes': len(payload),
                            'sha256': hashlib.sha256(payload).hexdigest()})
    blob = base64.b85encode(gzip.compress(buffer.getvalue(), mtime=0)).decode()
    return blob, install


def pinned_wheel_cell(blob: str, pinned: dict) -> str:
    """Reinstall the reviewed harness wheels the organisers replaced in place.

    Measured on 2026-10-13: the control sweep reported
    `harness wheelhouse version drift: [('swegemma-0.2.10-py3-none-any.whl',
    ['swegemma-0.2.11-py3-none-any.whl'])]` plus changed bytes for `adk_eval_core` and
    `adk_submission`, and then returned **0 usable tasks out of 8** -- including the four that had passed
    on 2026-10-09 before the in-place republication. Every arm produced no test result at all, so the
    drifted harness is what broke a working instrument, not the tasks.

    The reviewed bytes travel **inside the notebook** rather than through a `kernel_sources` mount. The
    mount was tried first and came back empty (`pinned wheel mount is incomplete`), and the project has
    seen this before: identical metadata worked for one kernel and failed for another, so a mount cannot
    carry a measurement instrument. Each extracted wheel is verified against the frozen manifest before
    it is installed, exactly as the paired A/B runs did.
    """
    return (
        "# Restore the reviewed harness wheels that the organisers replaced in place.\n"
        "import io, tarfile\n"
        "PINNED_WHEELS = " + repr(pinned) + "\n"
        "PINNED_DIR = Path('/kaggle/working/pinned-wheels')\n"
        "PINNED_DIR.mkdir(parents=True, exist_ok=True)\n"
        "_blob = " + repr(blob) + "\n"
        "with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b85decode(_blob)))) as _archive:\n"
        "    _archive.extractall(PINNED_DIR)\n"
        "_pinned = {'source': 'embedded payload, hash-verified against the reviewed manifest',\n"
        "           'installed': [], 'missing': []}\n"
        "for _name, _expected in sorted(PINNED_WHEELS.items()):\n"
        "    _source = PINNED_DIR / _name\n"
        "    if not _source.is_file():\n"
        "        _pinned['missing'].append(_name)\n"
        "        continue\n"
        "    _digest = hashlib.sha256(_source.read_bytes()).hexdigest()\n"
        "    assert _digest == _expected, 'pinned wheel differs from the frozen manifest: ' + _name\n"
        "    _restore = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',\n"
        "                               '--no-deps', '--force-reinstall', str(_source)],\n"
        "                              capture_output=True, text=True)\n"
        "    print('===', _name, 'rc=', _restore.returncode, flush=True)\n"
        "    assert _restore.returncode == 0, 'pinned wheel install failed: ' + _name\n"
        "    _pinned['installed'].append(_name)\n"
        "(WORK / 'pinned-wheels.json').write_text(json.dumps(_pinned, indent=2) + '\\n')\n"
        "print(json.dumps(_pinned, indent=2), flush=True)\n"
        "assert not _pinned['missing'], 'pinned wheel payload is incomplete: ' + repr(_pinned['missing'])\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-dir', type=Path, required=True,
                        help='Frozen agent directory; its eval_config.yaml sets the budgets')
    parser.add_argument('--task-ids', nargs='+', required=True)
    parser.add_argument('--test-deps-dir', type=Path, required=True)
    parser.add_argument('--scripts-dir', type=Path, default=Path('scripts'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--title', default=None)
    parser.add_argument('--wheel-manifest', type=Path,
                        help='Reviewed wheelhouse inventory; defaults to the October 9 snapshot')
    parser.add_argument('--verify-install', action='store_true',
                        help='Add an import check after installation, before the control sweep')
    parser.add_argument('--dataset-source', action='append', default=[],
                        help='Kaggle dataset source (owner/slug) to attach; repeatable')
    parser.add_argument('--kernel-source', action='append', default=[],
                        help='Kaggle kernel source (owner/slug) to attach; repeatable. Needed for the '
                             'reviewed harness wheels, whose publisher kernel is a private kernel '
                             'rather than a dataset.')
    parser.add_argument('--pinned-wheels-dir', type=Path, default=None,
                        help='Directory holding the reviewed copies of the harness wheels the '
                             'organisers replaced in place (swegemma, adk_eval_core, adk_submission). '
                             'They are embedded in the notebook and hash-checked against the reviewed '
                             'manifest before installation. Prefer this over a kernel-source mount, '
                             'which has silently come back empty for this project before.')
    parser.add_argument('--hide-ipython', action='store_true',
                        help='Remove IPython so its extra Pygments lexers do not change '
                             'Syntax.guess_lexer results for the rich tasks')
    parser.add_argument('--pin-task-environment', action='store_true',
                        help='Pin starlette/pydantic/anyio per task from the task wheelhouse, '
                             'matching the version each snapshot declares')
    args = parser.parse_args()

    scripts = {}
    for name in ('evaluate_agent.py', 'validate_agent.py'):
        candidate = args.scripts_dir / name
        if not candidate.is_file():
            raise SystemExit(f'{name} not found under {args.scripts_dir}')
        scripts[name] = candidate.read_text(encoding='utf-8')

    eval_config = (args.agent_dir / 'eval_config.yaml').read_text(encoding='utf-8')
    encoded, deps_blob = build_test_dependency_blob(args.test_deps_dir)
    root = Path(__file__).resolve().parents[1]
    wheel_manifest = args.wheel_manifest or root / 'resource/manifests/eval-wheelhouse-2026-10-09.json'
    assets = json.loads(wheel_manifest.read_text(encoding='utf-8'))

    payload = base64.b85encode(gzip.compress(json.dumps({
        'scripts': scripts, 'task_ids': args.task_ids, 'eval_config': eval_config,
        'test_deps': deps_blob,
        'harness_wheels': {item['path']: item['sha256'] for item in assets['wheelhouse_files']},
        'wheels': {item['path']: item['sha256'] for item in assets['wheelhouse_files']},
        'wheel_manifest_sha256': hashlib.sha256(wheel_manifest.read_bytes()).hexdigest(),
    }).encode('utf-8'), compresslevel=9)).decode('ascii')

    probe = PROBE.replace('PROBE_MODULES', json.dumps(PROBE_MODULES))
    cells = [
        cell('markdown',
             '# Public task control sweep (CPU, no model)\n\n'
             'Installs the official wheelhouse and the reviewed test-only dependencies offline, '
             'then validates the empty and gold controls for the predeclared task set. No agent '
             'inference and no competition submission. A task is usable only when the empty patch '
             'fails and the gold patch passes.'),
        cell('code', 'PAYLOAD_B85 = ' + repr(payload) + '\n'
             'import base64, gzip, json\n'
             'PAYLOAD = json.loads(gzip.decompress(base64.b85decode(PAYLOAD_B85)))\n'
             + SETUP),
        cell('code', OFFICIAL_WHEELS_NOTE),
        cell('code', official_wheelhouse_cell('harness_wheels', 'HARNESS_WHEELS')),
    ]
    if args.pinned_wheels_dir:
        reviewed = {item['path']: item['sha256'] for item in assets['wheelhouse_files']
                    if item['path'] in PINNED_HARNESS_WHEELS}
        missing_pinned = sorted(set(PINNED_HARNESS_WHEELS) - set(reviewed))
        if missing_pinned:
            raise SystemExit(f'the reviewed wheel manifest does not list {missing_pinned}')
        pinned_blob, pinned_install = build_pinned_blob(args.pinned_wheels_dir, PINNED_HARNESS_WHEELS)
        for item in pinned_install:
            if item['sha256'] != reviewed[item['name']]:
                raise SystemExit(f'{item["name"]} differs from the reviewed manifest on disk')
        cells.append(cell('code', pinned_wheel_cell(pinned_blob, reviewed)))
    cells.extend([
        cell('code', install_cell_source(encoded, deps_blob)),
        cell('code', probe),
    ])
    if args.verify_install:
        cells.append(cell('code', VERIFY_INSTALL))
    if args.hide_ipython:
        cells.append(cell('code', HIDE_IPYTHON))
    if args.pin_task_environment:
        cells.append(cell('code', PIN_TASK_ENVIRONMENT))
    cells.append(cell('code', SWEEP))
    notebook = {
        'nbformat': 4,
        'nbformat_minor': 4,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                    'name': 'python3'}},
        'cells': cells,
    }
    encoded_notebook = json.dumps(notebook, indent=1) + '\n'
    size = len(encoded_notebook.encode('utf-8'))
    if size >= 1_000_000:
        raise SystemExit(f'notebook source would exceed the 1 MB API limit ({size} bytes)')

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'notebook.ipynb').write_text(encoded_notebook, encoding='utf-8')
    metadata = {
        'id': args.slug,
        'title': args.title or args.slug.split('/')[-1],
        'code_file': 'notebook.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': False,
        'enable_internet': False,
        'competition_sources': ['gemma-4-developer-agent'],
        'dataset_sources': list(args.dataset_source),
        'model_sources': [],
        'kernel_sources': list(args.kernel_source),
    }
    (out_dir / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                                  encoding='utf-8')
    record = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'slug': args.slug,
        'task_ids': args.task_ids,
        'agent_dir': str(args.agent_dir),
        'kernel_sources': list(args.kernel_source),
        'pinned_wheels': sorted(PINNED_HARNESS_WHEELS) if args.pinned_wheels_dir else [],
        'eval_config': eval_config,
        'notebook_sha256': hashlib.sha256((out_dir / 'notebook.ipynb').read_bytes()).hexdigest(),
        'metadata_sha256': hashlib.sha256(
            (out_dir / 'kernel-metadata.json').read_bytes()).hexdigest(),
        'test_deps_manifest_sha256': deps_blob['manifest_sha256'],
        'installed_wheels': [item['name'] for item in deps_blob['install']],
        'notebook_source_bytes': size,
    }
    (out_dir / 'generation-record.json').write_text(json.dumps(record, indent=2) + '\n',
                                                    encoding='utf-8')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
