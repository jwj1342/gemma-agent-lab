"""Standard-library contracts for frozen, resumable local evaluations."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

from build_submission import sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.partial-' + uuid.uuid4().hex)
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    pending.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_tasks(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row['instance_id'] for row in rows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('Task IDs must be unique, safe path components')
    return rows


def snapshot_group(task):
    return task['repo'].lower() + '.' + task['base_commit'].lower()


def expected_missing_api(empty):
    """Two released feature tests import a symbol introduced by the reference patch.

    Keep the exception narrow: other collection failures remain ineligible.
    The complete positive control is still required by check_controls.
    """
    symbols = {'fastapi_15745': ('_IncludedRouter', 'fastapi.routing'),
               'fastapi_14609': ('PydanticV1NotSupportedError', 'fastapi.exceptions')}
    symbol = symbols.get(empty.get('task_id'))
    output = empty.get('test_output', '')
    if (symbol is None or empty.get('resolved') is not False or empty.get('test_exit_code') != 2
            or empty.get('error_message') or '1 error during collection' not in output):
        return False
    errors = re.findall(r'^E\s+(.+Error:.*)$', output, re.MULTILINE)
    expected = "ImportError: cannot import name '%s' from '%s'" % symbol
    return len(errors) == 1 and errors[0].startswith(expected)


def check_controls(empty, gold):
    """Require a reproduced failure and complete passing reference verification."""
    from summarize_eval_smoke import ordinary_test_failure
    return (empty.get('task_id') == gold.get('task_id')
            and (ordinary_test_failure(empty) or expected_missing_api(empty))
            and gold.get('resolved') is True and gold.get('test_exit_code') == 0
            and not gold.get('error_message'))


def cache_record(key, empty, gold):
    body = {'key': key, 'empty': empty, 'gold': gold, 'valid': check_controls(empty, gold)}
    return {'body': body, 'sha256': digest(body)}


def read_control_cache(path, key, task_id):
    record = read_json(path)
    body = record['body']
    if (record['sha256'] != digest(body) or body['key'] != key
            or body['empty'].get('task_id') != task_id or body['gold'].get('task_id') != task_id
            or body['valid'] != check_controls(body['empty'], body['gold'])):
        raise ValueError('Control cache identity or integrity mismatch')
    return body


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [center - half, center + half]


def artifact_valid(path):
    """A finished task is reusable only if its saved bytes match its receipt."""
    path = Path(path)
    receipt = path.with_suffix('.sha256')
    if not path.exists():
        return False
    if not receipt.exists() or receipt.read_text().strip() != sha256(path):
        raise ValueError(f'Incomplete or changed result: {path}; preserve it and start a new run')
    return True


def save_artifact(path, value):
    write_json(path, value)
    Path(path).with_suffix('.sha256').write_text(sha256(path) + '\n')


def count_repositories(tasks):
    return dict(sorted(Counter(t['repo'] for t in tasks).items()))
