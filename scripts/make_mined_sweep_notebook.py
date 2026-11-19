"""Generate the CPU-only control sweep notebook for every holdout mined task.

Embeds the holdout tasks as a gzip+base85 payload (0.5 MB for 297 tasks, under the 1 MB notebook limit),
downloads each repository once per base commit, then per task runs the gold+test control and the
test-only control. Results are appended to `/kaggle/working/results.jsonl` after every task so a partial
run is still readable, and a per-repository summary is printed at the end.

No GPU, no competition data, no credentials. Usage:
  python scripts/make_mined_sweep_notebook.py --tasks <tasks.jsonl> --splits mined-197 google-100 --out <dir>
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

CELL = r'''
import base64, gzip, json, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path

PAYLOAD = json.loads(gzip.decompress(base64.b85decode(PAYLOAD_B85)))
WORK = Path('/kaggle/working/mined-sweep')
CACHE = WORK / 'commits'
RESULTS = WORK / 'results.jsonl'
WORK.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

DEPS = {
    'pallets/click': ['pytest'], 'google/ml_collections': ['pytest', 'absl-py', 'numpy', 'pyyaml'],
    'lark-parser/lark': ['pytest'], 'google/mobly': ['pytest', 'pyyaml', 'portpicker'],
    'google/openhtf': ['pytest', 'pyyaml', 'protobuf'], 'andialbrecht/sqlparse': ['pytest'],
    'python-poetry/tomlkit': ['pytest'], 'python-humanize/humanize': ['pytest'], 'arrow-py/arrow': ['pytest'],
}

def sh(command, cwd=None, timeout=600):
    return subprocess.run(command, cwd=cwd, shell=isinstance(command, str),
                          capture_output=True, text=True, timeout=timeout)

def boot():
    for repo in sorted({t['repo'] for t in PAYLOAD}):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *DEPS.get(repo, ['pytest'])],
                       capture_output=True, text=True)
    print('dependencies installed', flush=True)

def checkout(task):
    key = task['repo'].replace('/', '__') + '@' + task['base_commit']
    cached = CACHE / key
    if cached.exists():
        return cached
    temp = CACHE / (key + '.tmp')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    url = f"https://codeload.github.com/{task['repo']}/tar.gz/{task['base_commit']}"
    archive = temp / 'repo.tgz'
    with urllib.request.urlopen(url, timeout=300) as response:
        archive.write_bytes(response.read())
    with tarfile.open(archive) as tar:
        tar.extractall(temp / 'src')
    roots = [p for p in (temp / 'src').iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError('unexpected archive layout')
    roots[0].rename(cached)
    shutil.rmtree(temp, ignore_errors=True)
    return cached

def copy_tree(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination

def apply_patch(repo_dir, text, label):
    path = repo_dir.parent / f'{label}.diff'
    path.write_text(text if text.endswith('\n') else text + '\n')
    done = sh(['patch', '-p1', '--batch', '--forward', '-i', str(path)], cwd=repo_dir)
    if done.returncode:
        done = sh(['git', 'apply', '--unsafe-paths', '-p1', str(path)], cwd=repo_dir)
    return None if done.returncode == 0 else (done.stdout + done.stderr)[-400:]

def run_tests(repo_dir, node_ids, timeout=420):
    try:
        done = subprocess.run([sys.executable, '-m', 'pytest', '-x', '-q', *node_ids],
                              cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return 124, 'pytest timeout'

boot()
for index, task in enumerate(PAYLOAD, 1):
    task_id = task['instance_id']
    record = {'instance_id': task_id, 'repo': task['repo']}
    try:
        base = checkout(task)
        gold_dir = copy_tree(base, WORK / 'work' / (task_id + '-gold'))
        error = apply_patch(gold_dir, task['test_patch'], 'test')
        record['test_patch_ok'] = error is None
        if error:
            record['error'] = 'test_patch: ' + error
        else:
            error = apply_patch(gold_dir, task['patch'], 'gold')
            record['gold_patch_ok'] = error is None
            if error:
                record['error'] = 'gold_patch: ' + error
            else:
                record['gold_rc'], record['gold_tail'] = run_tests(gold_dir, task['FAIL_TO_PASS'])
        shutil.rmtree(gold_dir, ignore_errors=True)
        empty_dir = copy_tree(base, WORK / 'work' / (task_id + '-empty'))
        error = apply_patch(empty_dir, task['test_patch'], 'test')
        record['empty_patch_ok'] = error is None
        if error:
            record['error'] = record.get('error') or ('empty patch: ' + error)
        else:
            record['empty_rc'], record['empty_tail'] = run_tests(empty_dir, task['FAIL_TO_PASS'])
        shutil.rmtree(empty_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['valid'] = record.get('gold_rc') == 0 and record.get('empty_rc') not in (0, None)
    with RESULTS.open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    print(f"[{index}/{len(PAYLOAD)}] {task_id} gold={record.get('gold_rc')} empty={record.get('empty_rc')} "
          f"valid={record['valid']}", flush=True)

rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
per_repo = {}
for row in rows:
    bucket = per_repo.setdefault(row['repo'], {'tasks': 0, 'valid': 0})
    bucket['tasks'] += 1
    bucket['valid'] += int(row['valid'])
summary = {'tasks': len(rows), 'valid': sum(r['valid'] for r in rows), 'per_repo': per_repo,
           'errors': sum(1 for r in rows if r.get('error'))}
(WORK / 'sweep-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['mined-197', 'google-100'])
    parser.add_argument('--owner', default='junzhejiao')
    parser.add_argument('--slug', default='mined-control-sweep-2026-11-20')
    parser.add_argument('--title', default='mined control sweep 2026-11-20')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    hold = [row for row in rows if row.get('paper_split') in set(args.splits)]
    payload = [{'instance_id': r['instance_id'], 'repo': r['repo'], 'base_commit': r['base_commit'],
                'patch': r['patch'], 'test_patch': r['test_patch'], 'FAIL_TO_PASS': r['FAIL_TO_PASS']}
               for r in hold]
    blob = base64.b85encode(gzip.compress(json.dumps(payload).encode(), 9)).decode()
    code = CELL.replace('PAYLOAD_B85', repr(blob))
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'metadata': {}, 'source': [
                '# Mined holdout control sweep\n\n', f'CPU only, {len(payload)} tasks. '
                'Per task: gold+test patch must pass the FAIL_TO_PASS tests; test-only patch must fail them.']},
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [code]},
        ],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python', 'version': '3.12'}},
        'nbformat': 4, 'nbformat_minor': 5,
    }
    metadata = {'id': f'{args.owner}/{args.slug}', 'title': args.title, 'code_file': 'sweep.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
                'enable_tpu': False, 'enable_internet': True, 'kernel_sources': [],
                'competition_sources': [], 'dataset_sources': [], 'model_sources': []}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sweep.ipynb').write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    size = (out / 'sweep.ipynb').stat().st_size
    print(json.dumps({'notebook': str(out / 'sweep.ipynb'), 'tasks': len(payload), 'bytes': size}, indent=2))
    if size > 1_000_000:
        raise SystemExit('notebook exceeds the 1 MB source limit')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
