"""Collect the mined control sweep results and summarise validity per repository.

Usage:
  python scripts/collect_mined_sweep.py --kernel junzhejiao/mined-control-sweep-2026-11-20 --out runs/...
  python scripts/collect_mined_sweep.py ... --post      # also comment the table on #34

Reads `results.jsonl` and `sweep-summary.json` from the finished kernel output, recomputes the per-repository
validity rate with a Wilson 95% interval, and writes `sweep-report.md` next to the raw files. Nothing is
uploaded unless `--post` is given, and that only posts a comment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

GITHUB = 'https://api.github.com/repos/jwj1342/gemma-agent-lab'
DEFAULT_TOKEN_FILE = Path.home() / '.github-token'


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def fetch(kernel: str, out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve().parent / 'fetch_kernel_output.py'), kernel, str(out)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for path in out.rglob('results.jsonl'):
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    return rows


def from_log(path: Path, tasks: Path) -> list[dict]:
    """Parse a kernel log instead of the output tree.

    The sweep's working directory also holds the cached repository checkouts, so downloading the whole
    output tree can take longer than the sweep itself. The notebook prints one line per task, and the log
    is a few hundred kilobytes, so this path is the fast fallback.
    """
    units = json.loads(path.read_text(encoding='utf-8'))
    text = ''.join(unit.get('data', '') for unit in units)
    repo_of = {}
    if tasks.is_file():
        for line in tasks.read_text(encoding='utf-8').splitlines():
            if line.strip():
                task = json.loads(line)
                repo_of[task['instance_id']] = task['repo']
    rows = []
    for match in re.finditer(r'\[(\d+)/(\d+)\] (\S+) gold=(\S+) empty=(\S+) valid=(True|False)', text):
        rows.append({'instance_id': match.group(3), 'repo': repo_of.get(match.group(3), 'unknown'),
                     'gold_rc': match.group(4), 'empty_rc': match.group(5),
                     'valid': match.group(6) == 'True'})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default='junzhejiao/mined-control-sweep-2026-11-20')
    parser.add_argument('--out', type=Path, default=Path('runs/2026-11-20-mined-control-sweep'))
    parser.add_argument('--from-log', type=Path, help='parse this kernel log instead of downloading the output')
    parser.add_argument('--tasks', type=Path, default=Path('artifacts/tmp/swe-eval-validity/tasks.jsonl'),
                        help='mined tasks file, used to map instance IDs to repositories with --from-log')
    parser.add_argument('--post', action='store_true')
    parser.add_argument('--token-file', type=Path, default=DEFAULT_TOKEN_FILE,
                        help='file holding a GitHub token for --post (GITHUB_TOKEN wins if set)')
    args = parser.parse_args()

    rows = from_log(args.from_log, args.tasks) if args.from_log else fetch(args.kernel, args.out)
    if not rows:
        raise SystemExit('no control rows found')

    per_repo: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_repo.setdefault(row['repo'], {'tasks': 0, 'valid': 0, 'errors': 0})
        bucket['tasks'] += 1
        bucket['valid'] += int(bool(row.get('valid')))
        bucket['errors'] += int(bool(row.get('error')))

    total = len(rows)
    valid = sum(1 for row in rows if row.get('valid'))
    lines = ['| Repository | Tasks | Valid controls | Rate (Wilson 95%) | Errors |',
             '| --- | ---: | ---: | --- | ---: |']
    for repo in sorted(per_repo):
        bucket = per_repo[repo]
        low, high = wilson(bucket['valid'], bucket['tasks'])
        lines.append(f"| {repo} | {bucket['tasks']} | {bucket['valid']} | "
                     f"{bucket['valid'] / bucket['tasks']:.0%} ({low:.0%}-{high:.0%}) | {bucket['errors']} |")
    low, high = wilson(valid, total)
    lines.append(f'| **all** | **{total}** | **{valid}** | **{valid / total:.0%} ({low:.0%}-{high:.0%})** | '
                 f"{sum(b['errors'] for b in per_repo.values())} |")

    report = ['# Mined holdout control sweep', '',
              f'Kernel: `{args.kernel}`. CPU only, no GPU quota.', '',
              f'Valid controls: **{valid}/{total}** ({valid / total:.1%}).', '']
    report.extend(lines)
    failures = [row for row in rows if not row.get('valid')]
    report.append('')
    report.append(f'Invalid or failed tasks: {len(failures)}. First ten:')
    report.append('')
    for row in failures[:10]:
        report.append(f"- `{row['instance_id']}` ({row['repo']}): gold_rc={row.get('gold_rc')} "
                      f"empty_rc={row.get('empty_rc')} {row.get('error') or ''}".rstrip())
    (args.out / 'sweep-report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))

    if args.post:
        token = os.environ.get('GITHUB_TOKEN', '').strip()
        if not token and args.token_file.is_file():
            token = args.token_file.read_text(encoding='utf-8').strip()
        if not token:
            raise SystemExit('set GITHUB_TOKEN or pass --token-file to post the comment')
        body = ('**Mined holdout control sweep (CPU only).**\n\n'
                + '\n'.join(lines)
                + '\n\nFull run: `' + args.kernel + '`; report and per-task rows are in our working copy.\n'
                'Every task needs this gate: the dataset labels did not transfer (see the 5-task sample).')
        request = urllib.request.Request(f'{GITHUB}/issues/34/comments',
                                         data=json.dumps({'body': body}).encode(),
                                         headers={'Authorization': f'Bearer {token}',
                                                  'Accept': 'application/vnd.github+json',
                                                  'Content-Type': 'application/json',
                                                  'User-Agent': 'dsh-agent'},
                                         method='POST')
        with urllib.request.urlopen(request, timeout=60) as response:
            print(f'comment posted: HTTP {response.status}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
