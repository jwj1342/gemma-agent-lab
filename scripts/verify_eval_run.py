"""Validate a frozen run's task, agent files and exact ZIP without runtime dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from build_submission import collect_files, sha256
from prepare_eval_task import PUBLIC_FIELDS


def verify_inputs(run):
    run = Path(run)
    manifest = json.loads((run / 'run-manifest.json').read_text())
    tasks = run / 'inputs/tasks.jsonl'
    if sha256(tasks) != manifest['public_task_sha256']:
        raise ValueError('Frozen public task hash changed')
    with tasks.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected_ids = manifest['task_ids'] if manifest.get('suite_version') else [manifest['task_id']]
    if ([row.get('instance_id') for row in rows] != expected_ids
            or len(set(expected_ids)) != len(expected_ids) or not rows
            or any(not set(row) <= PUBLIC_FIELDS for row in rows)):
        raise ValueError('Frozen tasks must match the manifest without validation labels')
    if manifest.get('suite_version'):
        if sha256(run / 'inputs/benchmark.json') != manifest['benchmark_sha256']:
            raise ValueError('Frozen benchmark changed')
        runtime = run / 'inputs/runtime'
        actual = {p.name: sha256(p) for p in runtime.iterdir() if p.is_file()}
        if actual != manifest['runtime_sha256']:
            raise ValueError('Frozen runtime changed')
    bundle = run / 'inputs/submission.zip'
    archive_manifest = json.loads(bundle.with_suffix('.manifest.json').read_text())
    digest = sha256(bundle)
    if digest != manifest['agent_archive_sha256'] or digest != archive_manifest['sha256']:
        raise ValueError('Submission ZIP differs from the prepared run')
    root, files = collect_files(run / 'inputs/agent')
    current = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    records = archive_manifest['files']
    expected = {item['path']: item['sha256'] for item in records}
    if len(expected) != len(records) or current != expected:
        raise ValueError('Agent source changed after the run was prepared')
    with zipfile.ZipFile(bundle) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError('ZIP member validation failed')
        for name, expected_hash in expected.items():
            member_hash = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    member_hash.update(chunk)
            if member_hash.hexdigest() != expected_hash:
                raise ValueError(f'ZIP member hash differs from frozen agent: {name}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = verify_inputs(args.run_dir)
    print(json.dumps({'task_ids': result.get('task_ids', [result.get('task_id')]), 'status': 'Frozen inputs verified'}))
