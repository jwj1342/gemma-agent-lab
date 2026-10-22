"""Shared helper for embedding a reviewed, hash-verified offline test-dependency wheel set.

The Kaggle subprocess sandbox never installs repository test dependencies: in that backend
`swegemma.harness.container_setup.install_test_dependencies` returns immediately, and
`SubprocessManager` only uses its `wheels_dir` to translate `/wheels` in commands. The sandbox
venv is created with `system_site_packages=True` plus a `_host_env.pth` pointing at the notebook
interpreter, so those dependencies must already exist there. The public wheelhouse does not
contain them, which is why gold patches fail at pytest collection for many FastAPI/Requests
tasks (see resource/discussions/743973-full.json).

Keep the install list limited to test-only packages. Upgrading anything the image already
provides at a task-relevant version (pytest, rich, pydantic, starlette, flask, ...) would move
the measurement away from the environment the tasks were authored against.
"""
import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

# Package name -> wheel resolved by distribution name, not by a hard-coded filename.
# `httpbin`/`pytest_httpbin` are deliberately excluded: putting them in site-packages registers a
# `pytest11` entry point that imports `httpbin`, which fails against the image's Werkzeug 3 and
# aborts collection for every task, including ones that never use httpbin.
TEST_DEPENDENCY_PACKAGES = (
    'asttokens',
    'executing',
    'dirty_equals',
    'inline_snapshot',
    'ujson',
    'orjson',
)

# Packages deliberately not installed because the image already provides them, or because
# installing them breaks collection (see TEST_DEPENDENCY_PACKAGES).
EXCLUDED_IMAGE_PACKAGES = (
    'pytest', 'pluggy', 'rich', 'pydantic', 'starlette', 'flask', 'werkzeug', 'gevent',
    'typing_inspection', 'requests', 'httpx', 'httpbin', 'pytest_httpbin',
    'python_multipart',
)


def normalize(name: str) -> str:
    return name.lower().replace('-', '_').replace('.', '_')


def wheel_distribution(filename: str) -> str:
    """Distribution name of a wheel file, normalized.

    Wheel filenames are `{distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl`, and
    every distribution in the official harness wheelhouse is already normalized with underscores,
    so the first dash-separated field is the distribution name.
    """
    return normalize(filename.split('-')[0])


def find_wheel(manifest: dict, package: str) -> dict:
    """Return the manifest entry whose distribution name matches `package`."""
    target = normalize(package)
    for item in manifest['wheels']:
        filename = item['path']
        if not filename.endswith('.whl'):
            continue
        distribution = normalize(filename.split('-')[0])
        if distribution == target:
            return item
    raise ValueError(f'no wheel for {package!r} in the reviewed dependency set')


def build_test_dependency_blob(deps_dir: Path) -> tuple[str, dict]:
    """Pack the reviewed test-only wheels; return (base85 tar.gz payload, manifest record)."""
    deps_dir = Path(deps_dir)
    manifest_path = deps_dir / 'test-deps-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    install = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for package in TEST_DEPENDENCY_PACKAGES:
            entry = find_wheel(manifest, package)
            payload = (deps_dir / entry['path']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise ValueError(f'wheel bytes changed on disk: {entry["path"]}')
            info = tarfile.TarInfo(name=entry['path'])
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
            install.append({'name': entry['path'], 'sha256': entry['sha256'],
                            'bytes': entry['bytes']})
    blob = base64.b85encode(gzip.compress(buffer.getvalue(), mtime=0)).decode()
    record = {
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'manifest_path': str(manifest_path),
        'target': manifest.get('target'),
        'install': install,
        'excluded_image_packages': list(EXCLUDED_IMAGE_PACKAGES),
    }
    return blob, record


INSTALL_CELL_SOURCE = '''# Install the reviewed test-only dependencies offline. No image package is upgraded.
import base64 as _b64, gzip as _gzip, hashlib as _hash, io as _io, json as _json
import shutil as _shutil, subprocess as _sp, sys as _sys, tarfile as _tar
from pathlib import Path as _Path
TEST_DEPS = _json.loads(TEST_DEPS_RECORD)
_DEPOT = _Path('/tmp/reviewed-test-deps'); _DEPOT.mkdir(parents=True, exist_ok=True)
with _tar.open(fileobj=_io.BytesIO(_gzip.decompress(_b64.b85decode(TEST_DEPS_B85)))) as _archive:
    _archive.extractall(_DEPOT)
_wheels = []
for _item in TEST_DEPS['install']:
    _wheel = _DEPOT / _item['name']
    assert _hash.sha256(_wheel.read_bytes()).hexdigest() == _item['sha256'], 'test dependency changed'
    _wheels.append(str(_wheel))
print('installing', len(_wheels), 'reviewed test-only wheels', flush=True)
_install = _sp.run([_sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--force-reinstall', *_wheels], capture_output=True, text=True)
print(_install.stdout[-3000:]); print(_install.stderr[-2000:], flush=True)
_install.check_returncode()
# Publish the verified wheels as kernel output so a dependent GPU notebook can mount them
# through kernel_sources; the agent bundle already fills most of the 1 MB source budget.
_publish = _Path('/kaggle/working/reviewed-test-deps')
_publish.mkdir(parents=True, exist_ok=True)
for _wheel in _DEPOT.iterdir():
    if _wheel.is_file():
        _shutil.copyfile(_wheel, _publish / _wheel.name)
(_publish / 'test-deps-manifest.json').write_text(
    _json.dumps({'manifest_sha256': TEST_DEPS['manifest_sha256'],
                 'install': TEST_DEPS['install']}, indent=2) + '\\n')
(_publish / 'publish-sha256.json').write_text(_json.dumps(
    {_w.name: _hash.sha256(_w.read_bytes()).hexdigest() for _w in _publish.iterdir()
     if _w.is_file()}, indent=2) + '\\n')
print('published', len(_wheels), 'wheel files for dependent kernels', flush=True)
print('reviewed test-only dependencies installed; image packages untouched.')
'''


def install_cell_source(blob: str, record: dict) -> str:
    """Install the reviewed wheels from an inline payload (used when the budget allows)."""
    return ('TEST_DEPS_B85 = ' + repr(blob) + '\n'
            + 'TEST_DEPS_RECORD = ' + repr(json.dumps(record)) + '\n'
            + INSTALL_CELL_SOURCE)


def official_wheelhouse_cell(payload_key: str = 'wheels', wheels_var: str = 'WHEELS',
                             attempts: int = 8, pause: int = 20) -> str:
    """Install the mounted official harness wheels, recording exactly what the platform served.

    Neither filename nor byte equality with a reviewed snapshot can be required: the organisers
    republish this dataset in place. On 2026-10-09T14:43Z, six minutes before a pushed run, the
    mount moved `swegemma` from 0.2.10 to 0.2.11 and changed the bytes of `adk_eval_core` and
    `adk_submission` under unchanged filenames. Refusing to run would not change what the
    platform serves, so the cell installs whatever is mounted and records the observed
    filenames, versions and hashes beside the reviewed ones for provenance.

    Absence is therefore judged per *distribution*: a version bump is a republish, not a missing
    dependency, while a distribution the mount no longer carries is fatal. A large dataset can
    also mount before all files are materialised, so absence is retried. The notebook is offline,
    so a genuinely absent distribution raises a clear error instead of attempting a download.
    """
    return (
        "# Install the mounted official harness wheels; record observed filenames and hashes.\n"
        "import glob, hashlib, json, subprocess, sys, time, zipfile\n"
        "from pathlib import Path\n"
        "def _distribution(filename):\n"
        "    return filename.split('-')[0].lower().replace('-', '_').replace('.', '_')\n"
        "reviewed = PAYLOAD['" + payload_key + "']\n"
        "reviewed_distributions = {_distribution(name) for name in reviewed}\n"
        "present, missing_distributions = {}, []\n"
        "for attempt in range(1, " + str(attempts + 1) + "):\n"
        "    present = {path.name: path for path in " + wheels_var + ".glob('*.whl')}\n"
        "    mounted_distributions = {_distribution(name) for name in present}\n"
        "    missing_distributions = sorted(reviewed_distributions - mounted_distributions)\n"
        "    if not missing_distributions:\n"
        "        break\n"
        "    print('harness wheelhouse attempt', attempt, 'missing distributions',\n"
        "          missing_distributions, flush=True)\n"
        "    time.sleep(" + str(pause) + ")\n"
        "observed, changed, version_drift = {}, [], []\n"
        "mounted_by_distribution = {}\n"
        "for name, path in sorted(present.items()):\n"
        "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "    observed[name] = digest\n"
        "    mounted_by_distribution.setdefault(_distribution(name), []).append(name)\n"
        "    if name in reviewed and reviewed[name] != digest:\n"
        "        changed.append({'name': name, 'reviewed': reviewed[name], 'observed': digest})\n"
        "for name in sorted(reviewed):\n"
        "    if name not in present:\n"
        "        version_drift.append({'reviewed': name, 'mounted':\n"
        "                              mounted_by_distribution.get(_distribution(name), [])})\n"
        "report = {'reviewed': len(reviewed), 'mounted_total': len(present),\n"
        "          'missing_distributions': missing_distributions, 'version_drift': version_drift,\n"
        "          'changed_bytes': changed, 'observed': observed}\n"
        "(WORK / 'harness-wheelhouse-check.json').write_text(json.dumps(report, indent=2) + '\\n')\n"
        "print('harness wheelhouse missing distributions:', missing_distributions, flush=True)\n"
        "print('harness wheelhouse version drift:',\n"
        "      [(item['reviewed'], item['mounted']) for item in version_drift], flush=True)\n"
        "print('harness wheelhouse changed bytes:', [item['name'] for item in changed], flush=True)\n"
        "assert present, 'harness wheelhouse dataset is not mounted'\n"
        "assert not missing_distributions, ('harness wheelhouse is missing required distributions: '\n"
        "                                   + repr(missing_distributions))\n"
        "for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',\n"
        "                '/usr/local/lib/python*/site-packages/*cutlass*.pth'):\n"
        "    for pth in glob.glob(pattern):\n"
        "        Path(pth).unlink(missing_ok=True)\n"
        "normalized = Path('/tmp/gemma4-official-wheels')\n"
        "normalized.mkdir(exist_ok=True)\n"
        "staged, skipped = [], []\n"
        "for name in sorted(present):\n"
        "    source = present[name]\n"
        "    if 'cutlass' in source.name.lower():\n"
        "        continue\n"
        "    with zipfile.ZipFile(source) as archive:\n"
        "        meta = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))\n"
        "        version = next(x[9:] for x in archive.read(meta).decode().splitlines()\n"
        "                       if x.startswith('Version: '))\n"
        "    parts = source.name.split('-'); parts[1] = version\n"
        "    target = normalized / '-'.join(parts)\n"
        "    if not target.exists():\n"
        "        target.symlink_to(source)\n"
        "    staged.append(target)\n"
        "log_path = WORK / 'installation.log'\n"
        "with log_path.open('w') as log:\n"
        "    for target in staged:\n"
        "        completed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',\n"
        "                                    '--no-deps', '--force-reinstall', str(target)],\n"
        "                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
        "        log.write(f'=== {target.name} rc={completed.returncode} ===\\n')\n"
        "        log.write(completed.stdout or '')\n"
        "        if completed.returncode:\n"
        "            skipped.append({'name': target.name, 'returncode': completed.returncode,\n"
        "                            'detail': (completed.stdout or '')[-300:]})\n"
        "install_report = {'installed': len(staged) - len(skipped), 'staged': len(staged),\n"
        "                  'skipped': skipped}\n"
        "(WORK / 'harness-install.json').write_text(json.dumps(install_report, indent=2) + '\\n')\n"
        "print(json.dumps(install_report, indent=2), flush=True)\n"
    )


def load_test_dependency_record(deps_dir: Path) -> dict:
    """Read the manifest record for a reviewed dependency set without packing the wheels."""
    deps_dir = Path(deps_dir)
    manifest_path = deps_dir / 'test-deps-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    install = []
    for package in TEST_DEPENDENCY_PACKAGES:
        entry = find_wheel(manifest, package)
        payload = (deps_dir / entry['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError(f'wheel bytes changed on disk: {entry["path"]}')
        install.append({'name': entry['path'], 'sha256': entry['sha256'],
                        'bytes': entry['bytes']})
    return {
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'manifest_path': str(manifest_path),
        'target': manifest.get('target'),
        'install': install,
        'excluded_image_packages': list(EXCLUDED_IMAGE_PACKAGES),
    }


def mounted_install_cell_source(mount: str, record: dict) -> str:
    """Install the reviewed wheels from a read-only dataset mount, verifying every hash."""
    return (
        # `repr(record)` embeds the record as a Python literal. `repr(json.dumps(record))` embeds
        # a JSON *string*, which made `TEST_DEPS['install']` raise
        # `TypeError: string indices must be integers` in every notebook that used this helper.
        'TEST_DEPS = ' + repr(record) + '\n'
        'import hashlib as _hash, glob as _glob, json as _json, subprocess as _sp, sys as _sys\n'
        'from pathlib import Path as _Path\n'
        'TEST_DEPS_MOUNT = _Path(' + repr(mount) + ')\n'
        'assert TEST_DEPS_MOUNT.is_dir(), "test dependency mount missing: " + str(TEST_DEPS_MOUNT)\n'
        '_available = {_Path(p).name: _Path(p) for p in _glob.glob(str(TEST_DEPS_MOUNT / "**" / "*.whl"),\n'
        '                                                       recursive=True)}\n'
        '_wheels = []\n'
        "for _item in TEST_DEPS['install']:\n"
        '    _wheel = _available.get(_item["name"])\n'
        '    assert _wheel is not None, "missing reviewed wheel: " + _item["name"]\n'
        '    assert _hash.sha256(_wheel.read_bytes()).hexdigest() == _item["sha256"], \\\n'
        '        "test dependency changed: " + _item["name"]\n'
        '    _wheels.append(str(_wheel))\n'
        'print("installing", len(_wheels), "reviewed test-only wheels", flush=True)\n'
        '# Install each wheel separately. The review set is built for one interpreter, but the\n'
        '# pinned official starter image runs CPython 3.12 while a CPU probe kernel runs 3.13, so a\n'
        '# rejected binary wheel (ujson/orjson are cp313) must not abort the whole install. Every\n'
        '# rejection is recorded, and the empty/gold controls inside the same run decide whether the\n'
        '# case stays usable; a missing import shows up there as an invalid control, not as a\n'
        '# silently accepted result.\n'
        '_installed, _skipped = [], []\n'
        'for _wheel in _wheels:\n'
        '    _install = _sp.run([_sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",\n'
        '                        "--force-reinstall", _wheel], capture_output=True, text=True)\n'
        '    _name = _Path(_wheel).name\n'
        '    _output = (_install.stdout or "") + (_install.stderr or "")\n'
        '    print("===", _name, "rc=", _install.returncode, flush=True)\n'
        '    if _install.returncode:\n'
        '        _skipped.append({"name": _name, "returncode": _install.returncode,\n'
        '                         "detail": _output[-400:]})\n'
        '    else:\n'
        '        _installed.append(_name)\n'
        '_report = {"interpreter": _sys.version, "installed": _installed, "skipped": _skipped}\n'
        '(WORK / "test-deps-install.json").write_text(_json.dumps(_report, indent=2) + "\\n")\n'
        'print(_json.dumps(_report, indent=2), flush=True)\n'
        'assert _installed, "no reviewed test dependency could be installed"\n'
        'print("reviewed test-only dependencies installed; image packages untouched.")\n'
    )
