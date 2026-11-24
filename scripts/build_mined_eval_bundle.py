"""Build the snapshot + wheel bundle our evaluator needs for mined SWE tasks.

Usage:
  python scripts/build_mined_eval_bundle.py --tasks <mined tasks.jsonl> --ids id1 id2 ... --out <dir>
  python scripts/build_mined_eval_bundle.py --tasks ... --split mined-197 --limit 5 --out <dir>

For every selected task it:
  1. fetches the repository at `base_commit` from codeload.github.com (git protocol is blocked here),
  2. repacks it as `<out>/snapshots/<instance_id>.tgz` with the repository content at the archive root,
  3. downloads an offline wheel set for that repository's test dependencies into `<out>/wheels`
     (Linux cp312 manylinux, which is what the Kaggle kernel runs),
  4. writes `<out>/bundle-manifest.json` with source commits, digests, sizes and the layout used.

The `.tgz` layout matches what swegemma 0.2.11 expects: `evaluate` extracts the archive directly into the
sandbox directory (`tar -xzf <archive> -C <dest>`) and treats that directory as the repository, so the
repository content must sit at the archive root. Nothing is uploaded and no GPU is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Per-repository test dependencies for the sampled holdout repositories. Kept deliberately small: the
# kernel wheelhouse already carries the test harness, this list only covers what those repos import.
REQUIREMENTS = {
    'pallets/click': ['pytest==8.3.4', 'pytest-mock'],
    'google/ml_collections': ['pytest', 'absl-py', 'numpy', 'pyyaml'],
    'lark-parser/lark': ['pytest', 'regex'],
    'google/mobly': ['pytest==8.3.4', 'mock', 'pytz', 'pyyaml', 'portpicker'],
    'google/openhtf': ['pytest==8.3.4', 'mock', 'pyyaml', 'protobuf'],
    'andialbrecht/sqlparse': ['pytest'],
    'python-poetry/tomlkit': ['pytest'],
    'arrow-py/arrow': ['pytest', 'dateparser', 'pytz', 'simplejson', 'pytest-mock', 'pytest-cov'],
}
BUILD_REQUIREMENTS = ['setuptools', 'wheel', 'flit-core', 'hatchling', 'poetry-core', 'editables']
SANDBOX_BOOTSTRAP = ['pytest==8.3.4', 'wrapt', *BUILD_REQUIREMENTS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_snapshot(repo: str, commit: str, destination: Path) -> int:
    url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f'empty archive for {repo}@{commit}')
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / 'repo.tgz'
        archive.write_bytes(payload)
        extracted = Path(temp) / 'src'
        extracted.mkdir()
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.name.count('/') >= 1]
            tar.extractall(extracted, members=members)
        roots = [p for p in extracted.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f'unexpected archive layout for {repo}: {[p.name for p in roots]}')
        root = roots[0]
        with tarfile.open(destination, 'w:gz') as out:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    out.add(path, arcname=path.relative_to(root).as_posix())
    return len(payload)


def download_wheels(requirements: list[str], wheels: Path) -> list[str]:
    wheels.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
               '--python-version', '312', '--implementation', 'cp', '--abi', 'cp312',
               '--platform', 'manylinux2014_x86_64', '--index-url', 'https://pypi.org/simple',
               '--dest', str(wheels), *requirements]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        print(completed.stdout[-800:])
        print(completed.stderr[-800:], file=sys.stderr)
        raise RuntimeError('pip download failed')
    return sorted(path.name for path in wheels.glob('*.whl'))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--split')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row['instance_id'] in wanted]
        if {row['instance_id'] for row in rows} != wanted:
            raise SystemExit('Requested task IDs are missing')
    if args.split:
        rows = [row for row in rows if row.get('paper_split') == args.split]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit('no tasks selected')

    out = args.out
    if out.exists():
        raise SystemExit('Output exists; preserve the previous bundle and choose a new directory')
    snapshots = out / 'snapshots'
    wheels = out / 'wheels'
    snapshots.mkdir(parents=True, exist_ok=True)
    (out / 'tasks.jsonl').write_bytes(''.join(json.dumps(row) + '\n' for row in rows).encode())
    from prepare_eval_task import PUBLIC_FIELDS
    (out / 'tasks.public.jsonl').write_bytes(''.join(
        json.dumps({key: value for key, value in row.items() if key in PUBLIC_FIELDS}) + '\n'
        for row in rows).encode())
    record = {'layout': 'tgz root = repository content at base_commit (no .git, no top-level directory); '
                        'matches swegemma 0.2.11, which untars the archive into the sandbox directory',
              'source': 'codeload.github.com tar.gz', 'python': 'cp312 manylinux2014_x86_64',
              'tasks': [], 'wheels': {}, 'errors': [],
              'tasks_sha256': sha256(out / 'tasks.jsonl'),
              'public_tasks_sha256': sha256(out / 'tasks.public.jsonl')}
    repos: dict[str, list[str]] = {}

    for row in rows:
        task_id, repo, commit = row['instance_id'], row['repo'], row['base_commit']
        target = snapshots / f'{task_id}.tgz'
        try:
            raw = fetch_snapshot(repo, commit, target)
            record['tasks'].append({'instance_id': task_id, 'repo': repo, 'base_commit': commit,
                                    'snapshot_bytes': target.stat().st_size, 'download_bytes': raw,
                                    'snapshot_sha256': sha256(target)})
            repos.setdefault(repo, [])
            if task_id not in repos[repo]:
                repos[repo].append(task_id)
            print(f'ok {task_id} {repo}@{commit[:8]} -> {target.stat().st_size} bytes')
        except Exception as error:  # noqa: BLE001 - report and continue with the next task
            record['errors'].append({'instance_id': task_id, 'error': f'{type(error).__name__}: {error}'})
            print(f'fail {task_id}: {error}', file=sys.stderr)

    for repo in sorted(repos):
        requirements = [*SANDBOX_BOOTSTRAP,
                        *(req for req in REQUIREMENTS.get(repo, []) if req != 'pytest' and not req.startswith('pytest=='))]
        try:
            names = download_wheels(requirements, wheels)
            record['wheels'][repo] = {'requirements': requirements, 'files': names}
            print(f'wheels {repo}: {len(names)} files')
        except Exception as error:  # noqa: BLE001
            record['errors'].append({'repo': repo, 'error': f'wheels: {type(error).__name__}: {error}'})
            print(f'wheels fail {repo}: {error}', file=sys.stderr)

    record['wheel_sha256'] = {path.name: sha256(path) for path in sorted(wheels.glob('*.whl'))}
    (out / 'bundle-manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    total = sum(item['snapshot_bytes'] for item in record['tasks'])
    print(json.dumps({'tasks': len(record['tasks']), 'errors': len(record['errors']),
                      'snapshot_bytes': total, 'repos': sorted(repos),
                      'wheel_files': sum(len(v['files']) for v in record['wheels'].values())}, indent=2))
    return 1 if record['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
