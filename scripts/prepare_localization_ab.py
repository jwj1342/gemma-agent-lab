"""Freeze three localization arms and paired tasks for manual remote evaluation.

This CPU-only preparation never launches a GPU or uploads a notebook. Each case is
frozen by prepare_eval_run.py, and Kaggle notebooks use the existing official runner.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import benchmark_localization_v1 as benchmark
from build_submission import sha256
from verify_eval_run import verify_inputs

ROOT = Path(__file__).resolve().parents[1]
SKILL_TEMPLATE = '''---
name: localization-v1
description: Rank candidate source files for an issue when no explicit target path is given.
---

# Localization v1

Run the read-only localizer once with the complete issue. Use `run_skill_script`:

- `skill_name`: `localization-v1`
- `file_path`: `scripts/localize.py`
- `args`: {{"issue": "<complete problem statement>", "k": 10, "mode": "{mode}"}}

Inspect the top three paths and confirm the source before editing. If they are
irrelevant, use exact search and the existing analyzer. Rankings are hypotheses.
The script discovers the mounted workspace and does not modify source files.
'''


def choose_pilot(tasks, count, seed):
    """Round-robin repositories; selection never reads ranks or reference paths."""
    groups = {}
    for task in sorted(tasks, key=lambda item: item.instance_id):
        groups.setdefault(task.repo, []).append(task)
    for repo, group in groups.items():
        random.Random(seed + benchmark.stable_hash(repo)).shuffle(group)
    selected = []
    while len(selected) < min(count, len(tasks)):
        for repo in sorted(groups):
            if groups[repo] and len(selected) < count:
                selected.append(groups[repo].pop())
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/public/tasks.jsonl")
    parser.add_argument("--agent-dir", type=Path, default=ROOT / "configs/experiments/localization-v1")
    parser.add_argument("--owner", help="Also generate private Kaggle notebooks for pilot cases")
    parser.add_argument("--pilot-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior runs; output directory already exists")
    if args.pilot_size < 1:
        parser.error("pilot size must be positive")
    split = benchmark.split_tasks(benchmark.load_tasks(args.tasks), args.seed)
    tasks = sorted(split["dev"] + split["test"], key=lambda task: task.instance_id)
    pilot = {task.instance_id for task in choose_pilot(tasks, args.pilot_size, args.seed)}
    agents = args.output / "agents"
    shutil.copytree(ROOT / "configs/baseline", agents / "baseline")
    for arm, mode in (("bm25", "bm25"), ("portable", "portable")):
        shutil.copytree(args.agent_dir, agents / arm)
        skill = agents / arm / "skills/localization-v1/SKILL.md"
        skill.write_text(SKILL_TEMPLATE.format(mode=mode), encoding="utf-8", newline="\n")
    cases = []
    for index, task in enumerate(tasks):
        # Rotate execution order to reduce systematic warm-cache/order effects.
        arms = ["baseline", "bm25", "portable"]
        arms = arms[index % 3:] + arms[:index % 3]
        for arm in arms:
            case = args.output / "cases" / task.instance_id / arm
            command = [sys.executable, str(ROOT / "scripts/prepare_eval_run.py"),
                       "--run-dir", str(case), "--agent-dir", str(agents / arm),
                       "--task-id", task.instance_id, "--tasks", str(args.tasks)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            manifest = verify_inputs(case)
            record = {"task_id": task.instance_id, "repo": task.repo, "arm": arm,
                      "pilot": task.instance_id in pilot,
                      "path": case.relative_to(args.output).as_posix(),
                      "bundle_sha256": manifest["agent_archive_sha256"],
                      "public_task_sha256": manifest["public_task_sha256"]}
            if args.owner and task.instance_id in pilot:
                notebook = args.output / "notebooks" / task.instance_id / arm
                slug = f"localization-ab-{task.instance_id.replace('_', '-')}-{arm}"
                subprocess.run([sys.executable, str(ROOT / "scripts/prepare_kaggle_runtime.py"),
                                "--owner", args.owner, "--slug", slug, "--run-dir", str(case),
                                "--output", str(notebook)], check=True, capture_output=True, text=True)
                record["notebook"] = notebook.relative_to(args.output).as_posix()
                record["kernel"] = f"{args.owner}/{slug}"
            cases.append(record)
        print(f"froze {task.instance_id} ({index + 1}/{len(tasks)})", flush=True)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "tasks_sha256": sha256(args.tasks), "cases": cases,
        "pilot_task_ids": sorted(pilot), "confirmation_task_ids": [t.instance_id for t in tasks],
        "policy": "Pilot validates integration only; primary comparison uses every predeclared valid-control task.",
        "primary_comparison": "portable versus baseline issue resolution",
        "secondary_comparison": "portable versus BM25 skill issue resolution",
        "acceptance": {"minimum_absolute_resolution_gain": 0.05,
                       "one_sided_exact_mcnemar_p_max": 0.05,
                       "max_mean_agent_time_ratio": 1.10,
                       "require_all_cases_present": True,
                       "require_valid_empty_gold_controls": True},
        "status": "Prepared only; no GPU run, upload, or repair-resolution result.",
        "localization_evidence": "Retrospective public data; not independent hidden-repository confirmation.",
    }
    (args.output / "ab-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "pilot_tasks": len(pilot),
                      "protocol": str(args.output / "ab-protocol.json")}, indent=2))


if __name__ == "__main__":
    main()
