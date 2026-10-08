"""Fit portable full-tree ranking weights and a gate with nested cross-fitting.

The archived checkpoint is never used. Every held-out prediction refits both the
ranker and gate without that group. The historical public test remains retrospective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import benchmark_localization_v1 as benchmark
from train_localization_gate import stratified_bootstrap_deltas


RUNTIME = benchmark.LOCALIZER
ROOT = Path(__file__).resolve().parents[1]
BASE_WEIGHTS = [1.0] + [0.0] * (len(RUNTIME.PORTABLE_FEATURE_NAMES) - 1)
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranked(row, weights):
    scores = RUNTIME.portable_scores(row["features"], weights)
    order = RUNTIME.descending_order(scores)
    gold = set(row["gold_indices"])
    rank = next(i for i, index in enumerate(order, 1) if index in gold)
    return rank, order, scores


def measures(ranks):
    n = len(ranks)
    return {"tasks": n, **{
        f"hit_at_{k}": sum(rank <= k for rank in ranks) / n if n else 0.0
        for k in (1, 3, 5, 10)
    }, "mrr": sum(1 / rank for rank in ranks) / n if n else 0.0}


def evaluate(rows, weights):
    return measures([ranked(row, weights)[0] for row in rows])


def objective(rows, ranks):
    # Equal repository weight keeps the largest public repository from dominating.
    repos = sorted({row["repo"] for row in rows})
    scores = [measures([rank for row, rank in zip(rows, ranks) if row["repo"] == repo])
              for repo in repos]
    if not scores:
        return (0.0, 0.0)
    return (sum(0.5 * (s["hit_at_1"] + s["hit_at_5"]) for s in scores) / len(scores),
            sum(s["mrr"] for s in scores) / len(scores))


def fit_ranker(rows):
    """Two deterministic coordinate passes over nonnegative evidence weights."""
    weights = BASE_WEIGHTS[:]
    def key(candidate):
        ranks = [ranked(row, candidate)[0] for row in rows]
        primary, mrr = objective(rows, ranks)
        return (primary, mrr - 0.001 * sum(candidate[1:]), -sum(candidate))
    best = key(weights)
    for _ in range(2):
        changed = False
        for feature in range(1, len(weights)):
            for value in GRID:
                if value == weights[feature]:
                    continue
                proposal = weights[:]
                proposal[feature] = value
                score = key(proposal)
                if score > best:
                    weights, best, changed = proposal, score, True
        if not changed:
            break
    return weights


def folds_for(rows, kind="repository", seed=52, folds=3):
    if kind == "repository":
        return [[row["instance_id"] for row in rows if row["repo"] == repo]
                for repo in sorted({row["repo"] for row in rows})]
    # Group shared base commits, so a snapshot cannot occur on both sides of a fold.
    result = [[] for _ in range(folds)]
    for repo in sorted({row["repo"] for row in rows}):
        commits = sorted({row["commit"] for row in rows if row["repo"] == repo})
        random.Random(seed + benchmark.stable_hash(repo)).shuffle(commits)
        for i, commit in enumerate(commits):
            result[i % folds].extend(row["instance_id"] for row in rows
                                    if row["repo"] == repo and row["commit"] == commit)
    return [group for group in result if group]


def prediction(row, weights):
    rank, order, scores = ranked(row, weights)
    base_rank, base_order, base_scores = ranked(row, BASE_WEIGHTS)
    return {
        "instance_id": row["instance_id"], "repo": row["repo"],
        "commit": row["commit"], "bm25_rank": base_rank, "candidate_rank": rank,
        "bm25_margin": RUNTIME.normalized_margin(base_scores, base_order),
        "candidate_margin": RUNTIME.normalized_margin(scores, order),
        "top1_agree": base_order[0] == order[0],
    }


def cross_fit_ranker(rows, seed=52):
    predictions = []
    for ids in folds_for(rows, "snapshot", seed):
        held = set(ids)
        training = [row for row in rows if row["instance_id"] not in held]
        weights = fit_ranker(training)
        predictions.extend(prediction(row, weights) for row in rows if row["instance_id"] in held)
    return predictions


def matches(row, gate):
    return RUNTIME.portable_gate_matches(row["bm25_margin"], row["candidate_margin"], gate)


def no_regression(candidate, baseline):
    return all(candidate[name] + 1e-12 >= baseline[name]
               for name in ("hit_at_1", "hit_at_5", "mrr"))


def fit_gate(rows, min_support=6):
    best_gate = {"kind": "off"}
    baseline = measures([row["bm25_rank"] for row in rows])
    best = (*objective(rows, [row["bm25_rank"] for row in rows]), 0)
    for max_margin in (0.1, 0.3, 0.6, 1.0):
        for min_margin in (0.0, 0.025, 0.075, 0.15):
            gate = {"kind": "confidence", "max_bm25_margin": max_margin,
                    "min_candidate_margin": min_margin}
            enabled = [matches(row, gate) for row in rows]
            changed = sum(use and row["bm25_rank"] != row["candidate_rank"]
                          for row, use in zip(rows, enabled))
            ranks = [row["candidate_rank"] if use else row["bm25_rank"]
                     for row, use in zip(rows, enabled)]
            value = measures(ranks)
            key = (*objective(rows, ranks), -sum(enabled))
            if changed >= min_support and no_regression(value, baseline) and key > best:
                best_gate, best = gate, key
    return best_gate


def policy_summary(predictions):
    return {"bm25": measures([p["bm25_rank"] for p in predictions]),
            "candidate": measures([p["candidate_rank"] for p in predictions]),
            "gated": measures([p["candidate_rank"] if p["enabled"] else p["bm25_rank"]
                               for p in predictions]),
            "enabled": sum(p["enabled"] for p in predictions)}


def nested_validation(rows, kind, seed):
    predictions, audit = [], []
    for number, ids in enumerate(folds_for(rows, kind, seed, folds=5)):
        held = set(ids)
        training = [row for row in rows if row["instance_id"] not in held]
        calibration = cross_fit_ranker(training, seed)
        gate = fit_gate(calibration)
        weights = fit_ranker(training)
        for row in rows:
            if row["instance_id"] in held:
                item = prediction(row, weights)
                item["enabled"] = matches(item, gate)
                predictions.append(item)
        audit.append({"fold": number, "training_ids": [r["instance_id"] for r in training],
                      "validation_ids": ids, "weights": weights, "gate": gate})
        print(f"nested {kind}: {number + 1} folds completed", flush=True)
    return {**policy_summary(predictions), "predictions": predictions, "folds": audit}


def build_rows(tasks, repositories):
    snapshots, inventories, stats = benchmark.load_snapshots(tasks, repositories)
    rows, excluded = [], []
    for i, task in enumerate(tasks):
        docs = snapshots[task.repo, task.commit]
        # Feature extraction receives issue/source only. Labels are joined afterwards.
        features = RUNTIME.portable_features(docs, task.issue)
        gold = [j for j, doc in enumerate(docs) if doc.path in task.gold_paths]
        if not gold:
            excluded.append(task.instance_id)
            continue
        rows.append({"instance_id": task.instance_id, "repo": task.repo,
                     "commit": task.commit, "features": features, "gold_indices": gold})
        if (i + 1) % 10 == 0:
            print(f"portable features: {i + 1}/{len(tasks)}", flush=True)
    return rows, excluded, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/public/tasks.jsonl")
    parser.add_argument("--repositories", type=Path, default=ROOT / "data/raw/source-repos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output already exists; preserve previous experiment evidence")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap samples must be positive")
    started = time.perf_counter()
    identity = {"tasks_sha256": digest(args.tasks),
                "runtime_sha256": digest(benchmark.LOCALIZER_PATH),
                "benchmark_sha256": digest(Path(benchmark.__file__))}
    tasks = benchmark.load_tasks(args.tasks)
    if args.cache and args.cache.exists():
        cached = json.loads(args.cache.read_text(encoding="utf-8"))
        if cached["identity"] != identity:
            raise ValueError("Feature cache does not match frozen input/source hashes")
        rows, excluded, stats = cached["rows"], cached["excluded"], cached["stats"]
    else:
        rows, excluded, stats = build_rows(tasks, args.repositories)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps({"identity": identity, "rows": rows,
                                             "excluded": excluded, "stats": stats}), encoding="utf-8")
    splits = benchmark.split_tasks(tasks, args.seed)
    ids = {name: {task.instance_id for task in subset} for name, subset in splits.items()}
    confirmation_commits = {(task.repo, task.commit)
                            for task in splits["dev"] + splits["test"]}
    training = [row for row in rows if row["instance_id"] in ids["train"]
                and (row["repo"], row["commit"]) not in confirmation_commits]
    print(f"training: {len(training)}; archived checkpoint excluded", flush=True)
    weights = fit_ranker(training)
    calibration = cross_fit_ranker(training, args.seed)
    gate = fit_gate(calibration)
    print(json.dumps({"frozen_weights": weights, "frozen_gate": gate}), flush=True)
    nested = {kind: nested_validation(training, kind, args.seed)
              for kind in ("snapshot", "repository")}
    results = {}
    for name in ("dev", "test"):
        predictions = [prediction(row, weights) for row in rows if row["instance_id"] in ids[name]]
        for item in predictions:
            item["enabled"] = matches(item, gate)
        intervals = stratified_bootstrap_deltas(
            predictions, [p["enabled"] for p in predictions], args.bootstrap_samples, args.seed
        )
        results[name] = {**policy_summary(predictions), "bootstrap": intervals, "predictions": predictions}
    checks = {
        "gate_nontrivial": gate["kind"] != "off",
        **{f"nested_{kind}_no_regression": no_regression(value["gated"], value["bm25"])
           for kind, value in nested.items()},
        **{f"{name}_no_regression": no_regression(value["gated"], value["bm25"])
           for name, value in results.items()},
        "fresh_confirmation_passed": False,
        "remote_paired_repair_improvement": False,
    }
    policy = {"schema_version": 1, "model_type": "portable_localization_v1",
              "features": list(RUNTIME.PORTABLE_FEATURE_NAMES), "weights": weights,
              "gate": gate, "enabled": False, "promotion_checks": checks,
              "decision": "Experimental only; fresh confirmation and remote paired repair evidence required."}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    payload = {"policy": policy, "nested_validation": nested, "evaluation": results,
               "training_metrics": {"bm25": evaluate(training, BASE_WEIGHTS),
                                    "candidate": evaluate(training, weights)},
               "calibration_predictions": calibration, "identity": identity,
               "trainer_sha256": digest(__file__), "git_revision": revision.stdout.strip(),
               "python": platform.python_version(), "platform": platform.platform(),
               "dependencies": "Python standard library; CPU only", "seed": args.seed,
               "weight_grid": GRID, "coordinate_passes": 2, "repository_stats": stats,
               "training_ids": [row["instance_id"] for row in training], "excluded_new_file_only": excluded,
               "holdout_status": "Retrospective; previously inspected public test labels are not fresh confirmation.",
               "elapsed_seconds": round(time.perf_counter() - started, 3)}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "study.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"policy": policy, "training": payload["training_metrics"],
                      "nested": {k: {a: b for a, b in v.items() if a not in ("predictions", "folds")}
                                 for k, v in nested.items()},
                      "evaluation": {k: {a: b for a, b in v.items() if a != "predictions"}
                                     for k, v in results.items()},
                      "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
