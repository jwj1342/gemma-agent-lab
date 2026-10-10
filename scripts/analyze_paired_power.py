"""Assess whether a paired LoRA-versus-baseline comparison can actually be interpreted.

A paired A/B result is only readable when three things hold: the tasks have valid controls (the
unpatched code fails and the reference patch passes), both arms were measured on exactly those tasks,
and there are enough of them to separate an effect from noise. This script checks the first two by
reading the control records that already exist, and computes the third instead of guessing at it.

    python scripts/analyze_paired_power.py --runs-dir runs --json-out artifacts/paired-power.json

Statistics used, all exact and standard-library only:

* Discordant pairs are the only pairs that carry information. A task both arms resolve, or neither
  resolves, cannot distinguish them.
* McNemar's exact test is the two-sided binomial test on the discordant split.
* Power is the exact binomial probability of rejecting at alpha, given an assumed baseline rate and
  an assumed improvement on tasks the baseline fails.

Power here is a statement about the *task set*, not about the model. It answers "if the adapter were
genuinely better by this much, would this task set show it?" -- which is why the required task count
is a precondition for training budget, not an afterthought.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Scores are pass/fail per task, so the smallest effect worth detecting is one task. Ten percentage
# points is the working target the project has used when judging candidate changes.
TARGET_EFFECTS = (0.05, 0.10, 0.20, 0.30)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def power_for(baseline_rate: float, expected_gain: float, tasks: int, alpha: float = 0.05) -> float:
    """Exact power: probability that McNemar rejects, when the adapter converts `expected_gain` of
    the tasks the baseline fails and never breaks a task the baseline resolves."""
    if tasks <= 0:
        return 0.0
    fail = 1.0 - baseline_rate
    p_discordant = fail * expected_gain
    total = 0.0
    for observed in range(tasks + 1):
        probability = math.comb(tasks, observed) * p_discordant ** observed * (1 - p_discordant) ** (tasks - observed)
        if probability and exact_mcnemar_p(observed, 0) < alpha:
            total += probability
    return total


def min_detectable_gain(baseline_rate: float, tasks: int, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Smallest gain whose exact power reaches `power`, or None when even 1.0 cannot."""
    for step in range(5, 105, 5):
        gain = step / 100
        if power_for(baseline_rate, gain, tasks, alpha) >= power:
            return gain
    return None


def tasks_required(baseline_rate: float, expected_gain: float, alpha: float = 0.05,
                   power: float = 0.8, cap: int = 2000) -> int | None:
    """Fewest tasks whose exact power reaches `power`, or None under the cap."""
    for tasks in range(2, cap + 1, 2):
        if power_for(baseline_rate, expected_gain, tasks, alpha) >= power:
            return tasks
    return None


def collect_controls(runs_dir: Path) -> dict[str, dict]:
    """Find per-task empty/gold controls anywhere under runs/ and judge each task."""
    found: dict[str, dict] = {}
    for result in sorted(runs_dir.rglob('result.json')):
        try:
            data = json.loads(result.read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError, OSError):
            continue
        task_id = data.get('task_id')
        parent = result.parent.name
        if not task_id or parent not in ('empty', 'gold'):
            continue
        record = found.setdefault(task_id, {})
        record[parent] = {
            'resolved': data.get('resolved') is True,
            'test_exit_code': data.get('test_exit_code'),
            'error': bool(data.get('error_message')),
        }
    for record in found.values():
        empty, gold = record.get('empty'), record.get('gold')
        # The empty arm must fail by running tests and failing them, which pytest reports as exit 1.
        # Exit 0 would mean the tests passed without the fix, and exit 2 or higher is a collection or
        # infrastructure error, not evidence that the task is measurable.
        record['valid'] = bool(
            empty and gold
            and empty['resolved'] is False and empty['test_exit_code'] == 1 and not empty['error']
            and gold['resolved'] is True and gold['test_exit_code'] == 0 and not gold['error'])
    return found


def collect_arm_results(runs_dir: Path) -> dict[str, dict[str, bool]]:
    """Map task_id -> {arm: resolved} from arm results outside the control directories."""
    arms: dict[str, dict[str, bool]] = {}
    for result in sorted(runs_dir.rglob('result.json')):
        if result.parent.name in ('empty', 'gold', 'controls'):
            continue
        try:
            data = json.loads(result.read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError, OSError):
            continue
        task_id = data.get('task_id')
        if not task_id or 'resolved' not in data:
            continue
        arm = result.parent.name
        arms.setdefault(task_id, {})[arm] = data.get('resolved') is True
    return arms


def summarize(runs_dir: Path) -> dict:
    controls = collect_controls(runs_dir)
    arms = collect_arm_results(runs_dir)
    valid = sorted(task for task, record in controls.items() if record['valid'])
    invalid = sorted(task for task, record in controls.items() if not record['valid'])
    repositories = Counter(re.split(r'[_/]', task)[0] for task in valid)

    paired = {task: arms[task] for task in valid if task in arms and len(arms[task]) >= 2}
    paired_counts = None
    p_value = None
    if paired:
        both = sum(1 for arms_map in paired.values() if all(arms_map.values()))
        neither = sum(1 for arms_map in paired.values() if not any(arms_map.values()))
        left = sum(1 for arms_map in paired.values() if list(arms_map.values())[0] and not list(arms_map.values())[-1])
        right = sum(1 for arms_map in paired.values() if not list(arms_map.values())[0] and list(arms_map.values())[-1])
        paired_counts = {'both_resolved': both, 'neither': neither, 'first_only': left, 'second_only': right}
        p_value = exact_mcnemar_p(right, left)

    # Baseline rate: use the measured rate when paired arms exist, otherwise assume the conservative
    # 0.0 because a candidate evaluated so far resolved nothing on control-valid tasks.
    baseline_rate = 0.0
    if paired:
        first_arm = [list(arms_map.values())[0] for arms_map in paired.values()]
        baseline_rate = sum(1 for value in first_arm if value) / len(first_arm)

    n = len(valid)
    return {
        'runs_dir': str(runs_dir),
        'control_tasks_total': len(controls),
        'control_valid_tasks': valid,
        'control_invalid_tasks': invalid,
        'control_valid_count': n,
        'valid_by_repository': dict(repositories),
        'single_repository_only': len(repositories) == 1,
        'arms_found': {task: arms[task] for task in sorted(arms)},
        'paired_tasks': sorted(paired),
        'paired_counts': paired_counts,
        'mcnemar_exact_p': p_value,
        'assumed_baseline_resolve_rate': baseline_rate,
        'power_at_current_size': {f'gain_{int(gain * 100)}pct': round(power_for(baseline_rate, gain, n), 4)
                                  for gain in TARGET_EFFECTS},
        'min_detectable_gain_at_80pct_power': min_detectable_gain(baseline_rate, n),
        'tasks_required_for_80pct_power': {f'gain_{int(gain * 100)}pct':
                                           tasks_required(baseline_rate, gain)
                                           for gain in TARGET_EFFECTS},
        'caveat': ('Controls only tell us a task is measurable. The valid tasks here come from a '
                   'single repository, so any paired result cannot separate an adapter effect from '
                   'a repository effect, and this script cannot measure training quality at all.'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()
    result = summarize(args.runs_dir.resolve())
    text = json.dumps(result, indent=2) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
