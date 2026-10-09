"""Select a recorded wheelhouse snapshot, or fingerprint an explicit exploratory run.

This module uses only the standard library and is embedded in generated notebooks.
A captured SHA256 identifies bytes; it does not establish upstream authenticity or
runtime compatibility. Every installation rechecks the snapshot captured at setup.
"""
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import warnings
import zipfile


DATASET = 'metric/gemma-4-developer-agent-wheelhouse'
CORE_PACKAGES = {'adk-submission', 'adk-eval-core', 'swegemma', 'google-adk', 'vllm'}


class WheelhouseMismatch(ValueError):
    """The mounted inputs differ from the requested or captured snapshot."""


def _validate_files(files):
    if not isinstance(files, dict) or not files:
        raise ValueError('A wheelhouse snapshot must contain wheel files')
    for name, digest in files.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*\.whl', name)):
            raise ValueError('Invalid wheel filename in snapshot: ' + str(name))
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid SHA256 for ' + name)


def load_registry(path):
    registry = json.loads(Path(path).read_text())
    if registry.get('schema_version') != 1 or not registry.get('snapshots'):
        raise ValueError('Unsupported or empty wheelhouse registry')
    ids = set()
    for snapshot in registry['snapshots']:
        identity = snapshot.get('id')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('Wheelhouse snapshot IDs must be unique nonempty strings')
        if snapshot.get('dataset') != DATASET:
            raise ValueError('Unexpected wheelhouse dataset')
        _validate_files(snapshot.get('wheels'))
        ids.add(identity)
    return registry['snapshots']


def fingerprint(directory):
    files = {}
    for path in sorted(Path(directory).glob('*.whl')):
        if not path.is_file():
            raise ValueError('Wheel input is not a regular file: ' + path.name)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(chunk)
        files[path.name] = h.hexdigest()
    _validate_files(files)
    return files


def difference(expected, actual):
    return {'missing': sorted(expected.keys() - actual.keys()),
            'added': sorted(actual.keys() - expected.keys()),
            'changed_sha256': sorted(n for n in expected.keys() & actual.keys()
                                     if expected[n] != actual[n])}


def _versions(directory, files):
    versions = {}
    for name in files:
        with zipfile.ZipFile(Path(directory) / name) as wheel:
            metadata = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:
                raise ValueError('Expected one package METADATA in ' + name)
            info = BytesParser().parsebytes(wheel.read(metadata[0]))
        if not info.get('Name') or not info.get('Version'):
            raise ValueError('Missing package name/version in ' + name)
        package = re.sub(r'[-_.]+', '-', info['Name']).lower()
        if package in versions:
            raise ValueError('Multiple wheels for package ' + package)
        versions[package] = info['Version']
    missing = CORE_PACKAGES - versions.keys()
    if missing:
        raise ValueError('Required runtime packages missing: ' + ', '.join(sorted(missing)))
    return versions


def capture(directory, snapshots, mode='verified', requested='auto'):
    if mode not in {'verified', 'record'}:
        raise ValueError('Unknown wheelhouse mode: ' + mode)
    if requested != 'auto' and mode == 'record':
        raise ValueError('An explicit snapshot requires verified mode')
    selected = [s for s in snapshots if requested == 'auto' or s['id'] == requested]
    if not selected:
        raise ValueError('Unknown wheelhouse snapshot: ' + requested)
    actual = fingerprint(directory)
    matches = [s for s in selected if s['wheels'] == actual]
    differences = {s['id']: difference(s['wheels'], actual) for s in selected}
    if not matches and mode == 'verified':
        raise WheelhouseMismatch(
            'Official wheelhouse inventory or SHA256 changed; no registered snapshot matches. '
            + json.dumps(differences, sort_keys=True)
            + '\nUpdate the checkout and regenerate the notebook, attach the requested '
              'dataset version, or explicitly use --wheelhouse-mode record for exploration. '
              'Record mode captures new hashes; compatibility still needs validation.')
    matched = matches[0] if matches else None
    if matched is None:
        warnings.warn('Using an unreviewed wheelhouse in record mode. Hashes and versions '
                      'will be recorded; runtime compatibility has not been established.', RuntimeWarning)
    versions = matched.get('versions', {}) if matched else _versions(directory, actual)
    return {'schema_version': 1, 'dataset': DATASET, 'mode': mode,
            'requested_snapshot': requested,
            'matched_snapshot': matched['id'] if matched else None,
            'status': 'registered_snapshot' if matched else 'unreviewed_snapshot',
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'wheels': actual, 'versions': versions,
            'compatibility': matched.get('compatibility') if matched else 'not validated',
            'differences': {} if matched else differences}


def recheck(directory, captured):
    """Reject any file replacement, addition, or removal after initial capture."""
    actual = fingerprint(directory)
    if actual != captured:
        raise WheelhouseMismatch('Wheelhouse changed after setup: '
                                 + json.dumps(difference(captured, actual), sort_keys=True))
