"""Prepare a reproducible, grouped SWE-smith action-level SFT pilot from a pinned shard.

Run as python -m scripts.prepare_sft_data from the repository root.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.sft_data import UnsupportedAction, action_examples, convert_trajectory, group_key, split_for_group

DATASET = "SWE-bench/SWE-smith-trajectories"
REVISION = "08e109b4a59eaeebf80e4675cd125d42e7ac99a4"
SHARD_SHA256 = "e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reject-setup-actions", action="store_true",
                        help="Reject whole trajectories containing explicit dependency/network setup commands")
    parser.add_argument("--preserve-reasoning", action="store_true",
                        help="Copy explicit text reasoning/reasoning_content into assistant reasoning; never infer it")
    parser.add_argument("--tasks", type=Path, default=Path("data/raw/gemma-4-developer-agent/tasks.jsonl"))
    parser.add_argument("--eval-benchmark", type=Path, default=Path("configs/evaluation/public-v1.json"),
                        help="Exclude all frozen development/holdout snapshots from future training")
    args = parser.parse_args()
    import pyarrow.parquet as pq
    if args.output.exists():
        raise ValueError("Output already exists; preserve the prior dataset")
    shard_digest = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    if shard_digest != SHARD_SHA256:
        raise ValueError("Pilot shard differs from the pinned upstream artifact")
    rows = pq.read_table(args.parquet).to_pylist()
    args.output.mkdir(parents=True)
    examples = {"train": [], "validation": []}
    rejected = Counter()
    groups = {"train": set(), "validation": set()}
    selected = []
    public_tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    reserved = {t["repo"] + "." + t["base_commit"][:8] for t in public_tasks if t["instance_id"] == "rich_3454"}
    benchmark_hash = None
    if not args.eval_benchmark.exists():
        raise ValueError("Frozen evaluation benchmark is missing; provide --eval-benchmark explicitly")
    if args.eval_benchmark.exists():
        benchmark = json.loads(args.eval_benchmark.read_text())
        protected = set(benchmark["cohorts"]["dev"] + benchmark["cohorts"]["holdout"])
        reserved.update(t["repo"] + "." + t["base_commit"][:8] for t in public_tasks if t["instance_id"] in protected)
        benchmark_hash = hashlib.sha256(args.eval_benchmark.read_bytes()).hexdigest()
    rows.sort(key=lambda r: hashlib.sha256(f"{args.seed}:{r['traj_id']}".encode()).hexdigest())
    seen = set()
    for row in rows:
        if len(selected) >= args.max_trajectories:
            break
        if row["instance_id"] in seen:
            rejected["duplicate issue attempt"] += 1
            continue
        try:
            messages = convert_trajectory(row, reject_setup_actions=args.reject_setup_actions,
                                          preserve_reasoning=args.preserve_reasoning)
            group = group_key(row["instance_id"])
            if group.lower() in {g.lower() for g in reserved}:
                raise UnsupportedAction("Reserved evaluation snapshot")
        except (UnsupportedAction, KeyError, ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
            continue
        seen.add(row["instance_id"])
        split = split_for_group(group, args.seed)
        metadata = {"source": DATASET, "revision": REVISION, "license": "MIT", "instance_id": row["instance_id"],
                    "traj_id": row["traj_id"], "group": group, "split": split, "teacher_model": row["model"],
                    "verification": "upstream resolved=True; target-tool replay not performed",
                    "reasoning_policy": "explicit-text" if args.preserve_reasoning else "actions-only",
                    "assistant_messages_with_reasoning": sum(bool(m.get("reasoning")) for m in messages)}
        selected.append(metadata)
        groups[split].add(group)
        examples[split].extend(action_examples(messages, metadata))
    assert groups["train"].isdisjoint(groups["validation"])
    for split, records in examples.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    manifest = {"dataset": DATASET, "revision": REVISION, "license": "MIT", "shard": args.parquet.name,
                "sha256": shard_digest, "seed": args.seed,
                "rows_in_shard": len(rows), "selected_trajectories": len(selected),
                "actions": {k: len(v) for k, v in examples.items()}, "groups": {k: sorted(v) for k, v in groups.items()},
                "selected": selected, "rejected": dict(rejected), "reserved_evaluation_groups": sorted(reserved),
                "reject_setup_actions": args.reject_setup_actions,
                "preserve_reasoning": args.preserve_reasoning,
                "reasoning_policy": "explicit-text" if args.preserve_reasoning else "actions-only",
                "eval_benchmark_sha256": benchmark_hash,
                "scope": "Action-level warm start. Upstream success labels retained; observations have not been replayed with competition tools."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("selected", "groups", "rejected")}, indent=2))


if __name__ == "__main__":
    main()
