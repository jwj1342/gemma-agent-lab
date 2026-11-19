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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default='junzhejiao/mined-control-sweep-2026-11-20')
    parser.add_argument('--out', type=Path, default=Path('runs/2026-11-20-mined-control-sweep'))
    parser.add_argument('--post', action='store_true')
    parser.add_argument('--token-file', type=Path, default=DEFAULT_TOKEN_FILE,
                        help='file holding a GitHub token for --post (GITHUB_TOKEN wins if set)')
    args = parser.parse_args()

    rows = fetch(args.kernel, args.out)
    if not rows:
        raise SystemExit('no results.jsonl found in the kernel output')

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
