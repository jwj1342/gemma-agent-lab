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

CHECKS_SOURCE = Path(__file__).with_name('mined_control_checks.py').read_text(encoding='utf-8')

CELL = r'''
import base64, gzip, hashlib, json, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path

exec(CONTROL_CHECKS_SOURCE, globals())
PAYLOAD = json.loads(gzip.decompress(base64.b85decode(PAYLOAD_B85)))
WORK = Path('/kaggle/working/mined-sweep')
CACHE = WORK / 'commits'
RESULTS = WORK / 'results.jsonl'
WORK.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

DEPS = {
    'pallets/click': ['pytest', 'pytest-mock'],
    'google/ml_collections': ['pytest', 'absl-py', 'numpy', 'pyyaml'],
    'lark-parser/lark': ['pytest', 'regex'],
    'google/mobly': ['pytest', 'mock', 'pytz', 'pyyaml', 'portpicker'],
    'google/openhtf': ['pytest', 'mock', 'protobuf', 'pyyaml'],
    'andialbrecht/sqlparse': ['pytest'],
    'python-poetry/tomlkit': ['pytest'],
    'python-humanize/humanize': ['pytest'],
    'arrow-py/arrow': ['pytest', 'dateparser', 'pytz', 'simplejson', 'pytest-mock', 'pytest-cov'],
}

def sh(command, cwd=None, timeout=600):
    return subprocess.run(command, cwd=cwd, shell=isinstance(command, str),
                          capture_output=True, text=True, timeout=timeout)

def boot():
    for repo in sorted({t['repo'] for t in PAYLOAD}):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *DEPS.get(repo, ['pytest'])],
                       capture_output=True, text=True, check=True)
    print('dependencies installed', flush=True)

def install_repo(repo_dir):
    """Make src-layout checkouts importable; plain layouts already import from the repository root."""
    if not (repo_dir / 'src').is_dir():
        return None
    done = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', '.'],
                          cwd=repo_dir, capture_output=True, text=True)
    return done.returncode

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
    """Run the FAIL_TO_PASS node IDs.

    `-o addopts=` drops repository-level coverage gates. Recorded node IDs are kept exact;
    collection errors are not retried with a broader, potentially unrelated set of tests.
    """
    report = repo_dir.parent / (repo_dir.name + '-junit.xml')
    report.unlink(missing_ok=True)
    command = [sys.executable, '-m', 'pytest', '-x', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
               '--junitxml=' + str(report)]
    try:
        done = subprocess.run(command + node_ids, cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr)[-6000:], junit_counts(report)
    except subprocess.TimeoutExpired:
        return 124, 'pytest timeout', junit_counts(report)

boot()
(WORK / 'runtime.json').write_text(json.dumps({
    'python': sys.version, 'model_execution': False,
    'payload_sha256': hashlib.sha256(json.dumps(PAYLOAD, sort_keys=True).encode()).hexdigest(),
    'control_checks_sha256': hashlib.sha256(CONTROL_CHECKS_SOURCE.encode()).hexdigest(),
    'installed': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
}, indent=2))
for index, task in enumerate(PAYLOAD, 1):
    task_id = task['instance_id']
    record = {'instance_id': task_id, 'repo': task['repo'], 'base_commit': task['base_commit']}
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
                record['gold_install_rc'] = install_repo(gold_dir)
                record['gold_rc'], record['gold_tail'], record['gold_tests'] = run_tests(gold_dir, task['FAIL_TO_PASS'])
        shutil.rmtree(gold_dir, ignore_errors=True)
        empty_dir = copy_tree(base, WORK / 'work' / (task_id + '-empty'))
        error = apply_patch(empty_dir, task['test_patch'], 'test')
        record['empty_patch_ok'] = error is None
        if error:
            record['error'] = record.get('error') or ('empty patch: ' + error)
        else:
            record['empty_install_rc'] = install_repo(empty_dir)
            record['empty_rc'], record['empty_tail'], record['empty_tests'] = run_tests(empty_dir, task['FAIL_TO_PASS'])
        shutil.rmtree(empty_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        record['error'] = f'{type(exc).__name__}: {exc}'
    record.update(control_verdict(record))
    with RESULTS.open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    print(f"[{index}/{len(PAYLOAD)}] {task_id} gold={record.get('gold_rc')} empty={record.get('empty_rc')} "
          f"valid={record['valid']}", flush=True)
    print('CONTROL_RESULT ' + json.dumps(record), flush=True)

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
    parser.add_argument('--ids', nargs='+', help='Run only these IDs, irrespective of split')
    parser.add_argument('--repos', nargs='*', help='only these repositories (exact owner/name)')
    parser.add_argument('--owner', default='junzhejiao')
    parser.add_argument('--slug', default='mined-control-sweep-2026-11-20')
    parser.add_argument('--title', default='mined control sweep 2026-11-20')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Output exists; preserve it and choose a new directory')

    rows = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    hold = [row for row in rows if (args.ids and row['instance_id'] in set(args.ids)
            or not args.ids and row.get('paper_split') in set(args.splits))
            and (not args.repos or row['repo'] in set(args.repos))]
    if args.ids and {row['instance_id'] for row in hold} != set(args.ids):
        raise SystemExit('Requested task IDs are missing')
    if not hold:
        raise SystemExit('No tasks selected')
    payload = [{'instance_id': r['instance_id'], 'repo': r['repo'], 'base_commit': r['base_commit'],
                'patch': r['patch'], 'test_patch': r['test_patch'], 'FAIL_TO_PASS': r['FAIL_TO_PASS']}
               for r in hold]
    blob = base64.b85encode(gzip.compress(json.dumps(payload).encode(), 9)).decode()
    code = CELL.replace('CONTROL_CHECKS_SOURCE', repr(CHECKS_SOURCE)).replace('PAYLOAD_B85', repr(blob))
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
    metadata['docker_image'] = json.loads((Path(__file__).resolve().parents[1] /
                                         'resource/notebooks/official/kernel-metadata.json').read_text())['dockerImage']
    if len((json.dumps(notebook, indent=1) + '\n').encode()) > 1_000_000:
        raise SystemExit('notebook exceeds the 1 MB source limit')
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
