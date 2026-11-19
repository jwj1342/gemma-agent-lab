"""Generate the CPU-only gold/empty control notebook for a sample of mined SWE tasks.

The notebook downloads each repository at its base commit, applies the gold patch plus the test patch, runs
the listed FAIL_TO_PASS tests (expected to pass), then re-downloads, applies only the test patch and runs the
same tests (expected to fail). That is the control semantics our evaluator requires before any inference.

No GPU, no competition data, no credentials. Usage:
  python scripts/make_mined_control_notebook.py --tasks <mined tasks.jsonl> --ids ... --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CELL = r'''
import json, os, subprocess, sys, tarfile, urllib.request, shutil
from pathlib import Path

PAYLOAD = json.loads(PAYLOAD_JSON)
WORK = Path('/kaggle/working/mined-controls')
REPO_ROOT = WORK / 'repos'
REPO_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS = []

DEPS = {
    'pallets/click': ['pytest'],
    'google/ml_collections': ['pytest', 'absl-py', 'numpy', 'pyyaml'],
    'lark-parser/lark': ['pytest'],
}

def sh(command, cwd=None, timeout=900):
    return subprocess.run(command, cwd=cwd, shell=isinstance(command, str),
                          capture_output=True, text=True, timeout=timeout)

def fetch(repo, commit, destination):
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
    archive = destination / 'repo.tgz'
    with urllib.request.urlopen(url, timeout=180) as response:
        archive.write_bytes(response.read())
    with tarfile.open(archive) as tar:
        tar.extractall(destination / 'src')
    roots = [p for p in (destination / 'src').iterdir() if p.is_dir()]
    return roots[0]

def apply_patch(repo_dir, text, label):
    path = repo_dir.parent / f'{label}.diff'
    path.write_text(text if text.endswith('\n') else text + '\n')
    for command in ([['patch', '-p1', '--batch', '--forward', '-i', str(path)]],
                    [['git', 'apply', '--unsafe-paths', '-p1', str(path)]]):
        done = sh(command[0], cwd=repo_dir)
        if done.returncode == 0:
            return None
        last = done
    return (last.stdout + last.stderr)[-800:]

def run_tests(repo_dir, node_ids, timeout=900):
    command = [sys.executable, '-m', 'pytest', '-x', '-q', *node_ids]
    done = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
    return done.returncode, (done.stdout + done.stderr)[-1200:]

for task in PAYLOAD:
    task_id = task['instance_id']
    deps = DEPS.get(task['repo'], ['pytest'])
    pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *deps],
                         capture_output=True, text=True)
    record = {'instance_id': task_id, 'repo': task['repo'], 'deps_rc': pip.returncode}
    try:
        # 1. gold + test patch: FAIL_TO_PASS must pass
        repo_dir = fetch(task['repo'], task['base_commit'], WORK / task_id / 'gold')
        error = apply_patch(repo_dir, task['test_patch'], 'test_patch')
        record['test_patch_applied'] = error is None
        if error:
            record['error'] = 'test_patch: ' + error
        else:
            error = apply_patch(repo_dir, task['patch'], 'gold_patch')
            record['gold_patch_applied'] = error is None
            if error:
                record['error'] = 'gold_patch: ' + error
            else:
                code, tail = run_tests(repo_dir, task['FAIL_TO_PASS'])
                record['gold_rc'] = code
                record['gold_tail'] = tail
        # 2. test patch only: the same tests must fail
        repo_dir = fetch(task['repo'], task['base_commit'], WORK / task_id / 'empty')
        error = apply_patch(repo_dir, task['test_patch'], 'test_patch')
        record['empty_test_patch_applied'] = error is None
        if not error:
            code, tail = run_tests(repo_dir, task['FAIL_TO_PASS'])
            record['empty_rc'] = code
            record['empty_tail'] = tail
    except Exception as exc:  # noqa: BLE001
        record['error'] = f'{type(exc).__name__}: {exc}'
    RESULTS.append(record)
    print(json.dumps({k: record[k] for k in record if not k.endswith('_tail')}), flush=True)

summary = {
    'tasks': len(RESULTS),
    'controls_valid': sum(1 for r in RESULTS if r.get('gold_rc') == 0 and r.get('empty_rc') not in (0, None)),
    'gold_ok': sum(1 for r in RESULTS if r.get('gold_rc') == 0),
    'empty_failed': sum(1 for r in RESULTS if r.get('empty_rc') not in (0, None)),
    'details': RESULTS,
}
(WORK.parent / 'controls-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ('tasks', 'controls_valid', 'gold_ok', 'empty_failed')}, indent=2))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--owner', default='junzhejiao')
    parser.add_argument('--slug', default='mined-task-controls-2026-11-20')
    parser.add_argument('--title', default='mined task controls 2026-11-20')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    wanted = set(args.ids)
    selected = []
    for row in rows:
        if row['instance_id'] in wanted:
            selected.append({
                'instance_id': row['instance_id'],
                'repo': row['repo'],
                'base_commit': row['base_commit'],
                'patch': row['patch'],
                'test_patch': row['test_patch'],
                'FAIL_TO_PASS': row['FAIL_TO_PASS'],
            })
    if len(selected) != len(wanted):
        missing = wanted - {row['instance_id'] for row in selected}
        raise SystemExit(f'missing tasks: {sorted(missing)}')

    code = CELL.replace('PAYLOAD_JSON', json.dumps(json.dumps(selected)))
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'metadata': {}, 'source': [
                '# Mined task gold/empty controls\n\n',
                'Private, CPU only. Downloads each repository at its base commit, applies the gold patch plus '
                'the test patch (FAIL_TO_PASS must pass), then applies only the test patch (the same tests '
                'must fail). No GPU, no competition data, no submission.']},
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [code]},
        ],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python', 'version': '3.12'}},
        'nbformat': 4, 'nbformat_minor': 5,
    }
    metadata = {'id': f'{args.owner}/{args.slug}', 'title': args.title, 'code_file': 'control.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
                'enable_tpu': False, 'enable_internet': True, 'kernel_sources': [],
                'competition_sources': [], 'dataset_sources': [], 'model_sources': []}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'control.ipynb').write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'notebook': str(out / 'control.ipynb'), 'tasks': [t['instance_id'] for t in selected],
                      'bytes': (out / 'control.ipynb').stat().st_size}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
