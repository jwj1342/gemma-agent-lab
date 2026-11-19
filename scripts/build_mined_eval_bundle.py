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

The `.tgz` layout is the one assumption this script makes; confirm it against a public snapshot inside the
Kaggle kernel before running controls. Nothing is uploaded and no GPU is used.
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
    'pallets/click': ['pytest'],
    'google/ml_collections': ['pytest', 'absl-py', 'numpy', 'pyyaml'],
    'lark-parser/lark': ['pytest'],
    'google/mobly': ['pytest', 'pyyaml', 'portpicker'],
    'google/openhtf': ['pytest', 'pyyaml', 'protobuf'],
    'andialbrecht/sqlparse': ['pytest'],
}


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
    command = [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
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
    if args.split:
        rows = [row for row in rows if row.get('paper_split') == args.split]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit('no tasks selected')

    out = args.out
    snapshots = out / 'snapshots'
    wheels = out / 'wheels'
    snapshots.mkdir(parents=True, exist_ok=True)
    record = {'layout': 'tgz root = repository content at base_commit (no .git, no top-level directory)',
              'source': 'codeload.github.com tar.gz', 'python': 'cp312 manylinux2014_x86_64',
              'tasks': [], 'wheels': {}, 'errors': []}
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
        requirements = REQUIREMENTS.get(repo, ['pytest'])
        try:
            names = download_wheels(requirements, wheels)
            record['wheels'][repo] = {'requirements': requirements, 'files': names}
            print(f'wheels {repo}: {len(names)} files')
        except Exception as error:  # noqa: BLE001
            record['errors'].append({'repo': repo, 'error': f'wheels: {type(error).__name__}: {error}'})
            print(f'wheels fail {repo}: {error}', file=sys.stderr)

    (out / 'bundle-manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    total = sum(item['snapshot_bytes'] for item in record['tasks'])
    print(json.dumps({'tasks': len(record['tasks']), 'errors': len(record['errors']),
                      'snapshot_bytes': total, 'repos': sorted(repos),
                      'wheel_files': sum(len(v['files']) for v in record['wheels'].values())}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
