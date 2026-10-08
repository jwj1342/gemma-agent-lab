"""Learn a conservative BM25-versus-v1 localization gate.

The gate uses only label-free signals available while an issue is being ranked.
Reference-patch paths are consulted after ranking to fit on the training split,
confirm the frozen rule on development, and report the designated holdout.  The
holdout never selects a rule; it can only veto deployment.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_localization_v1.py"
DEFAULT_CANDIDATE_WEIGHTS = {"bm25": 1.0, "exact": 0.4, "legacy": 0.5}
FEATURES = (
    "log_candidates",
    "log_issue_tokens",
    "bm25_margin",
    "bm25_positive_fraction",
    "legacy_margin",
    "legacy_positive_fraction",
    "exact_top",
    "exact_positive_fraction",
    "top1_agree",
    "top3_jaccard",
    "candidate_top_in_bm25",
    "bm25_top_in_candidate",
)


def load_benchmark():
    spec = importlib.util.spec_from_file_location("localization_gate_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


BENCHMARK = load_benchmark()
LOCALIZER = BENCHMARK.LOCALIZER
if FEATURES != LOCALIZER.GATE_FEATURE_NAMES:
    raise RuntimeError("Training and runtime gate feature schemas differ")


def gate_features(
    issue: str,
    components: dict[str, list[float]],
    candidate_weights: dict[str, float],
) -> tuple[dict[str, float], list[int], list[int]]:
    """Return runtime-safe features and the two component rankings."""
    candidate = LOCALIZER.reciprocal_rank_fusion(components, candidate_weights)
    return (
        LOCALIZER.gate_features(issue, components, candidate_weights),
        LOCALIZER.descending_order(components["bm25"]),
        LOCALIZER.descending_order(candidate),
    )


def rank_of_gold(order: list[int], paths: list[str], gold: set[str]) -> int:
    return next(position for position, index in enumerate(order, 1) if paths[index] in gold)


def build_rows(tasks, snapshots, inventories, model, candidate_weights):
    rows = []
    for index, task in enumerate(tasks, 1):
        documents = snapshots[(task.repo, task.commit)]
        inventory = inventories[(task.repo, task.commit)]
        gold = set(task.gold_paths) & inventory
        if not gold:
            continue
        components = {
            "bm25": LOCALIZER.bm25_scores(documents, LOCALIZER.tokenize(task.issue)),
            "exact": [LOCALIZER.exact_score(document, task.issue) for document in documents],
            "legacy": [model.score(task.issue, document.path) for document in documents],
        }
        features, bm25_order, candidate_order = gate_features(
            task.issue, components, candidate_weights
        )
        paths = [document.path for document in documents]
        bm25_rank = rank_of_gold(bm25_order, paths, gold)
        candidate_rank = rank_of_gold(candidate_order, paths, gold)
        rows.append(
            {
                "instance_id": task.instance_id,
                "repo": task.repo,
                "features": features,
                "bm25_rank": bm25_rank,
                "candidate_rank": candidate_rank,
                "v1_outcome": (
                    "win"
                    if candidate_rank < bm25_rank
                    else "loss"
                    if candidate_rank > bm25_rank
                    else "tie"
                ),
            }
        )
        if index % 10 == 0:
            print(f"features: {index}/{len(tasks)}", flush=True)
    return rows


def policy_metrics(rows: list[dict], enabled: list[bool]) -> dict[str, float | int]:
    ranks = [
        row["candidate_rank"] if use_v1 else row["bm25_rank"]
        for row, use_v1 in zip(rows, enabled)
    ]
    count = len(ranks)
    return {
        "tasks": count,
        "v1_enabled": sum(enabled),
        "hit_at_1": sum(rank <= 1 for rank in ranks) / count if count else 0.0,
        "hit_at_3": sum(rank <= 3 for rank in ranks) / count if count else 0.0,
        "hit_at_5": sum(rank <= 5 for rank in ranks) / count if count else 0.0,
        "hit_at_10": sum(rank <= 10 for rank in ranks) / count if count else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks) / count if count else 0.0,
        "mean_rank": sum(ranks) / count if count else 0.0,
    }


def metric_key(value: dict[str, float | int]) -> tuple[float, float, float, int]:
    return (
        0.5 * float(value["hit_at_1"]) + 0.5 * float(value["hit_at_5"]),
        float(value["mrr"]),
        float(value["hit_at_10"]),
        -int(value["v1_enabled"]),
    )


def rule_matches(row: dict, rule: dict) -> bool:
    if rule["kind"] == "off":
        return False
    conditions = rule.get("conditions", [rule])
    for condition in conditions:
        value = row["features"][condition["feature"]]
        if condition["operator"] == "<=":
            matched = value <= condition["threshold"]
        else:
            matched = value > condition["threshold"]
        if not matched:
            return False
    return True


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    return [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]


def atomic_conditions(rows: list[dict], min_enabled: int) -> list[dict]:
    conditions = []
    for feature in FEATURES:
        thresholds = candidate_thresholds([row["features"][feature] for row in rows])
        for operator in ("<=", ">"):
            for threshold in thresholds:
                condition = {
                    "kind": "stump",
                    "feature": feature,
                    "operator": operator,
                    "threshold": threshold,
                }
                support = sum(rule_matches(row, condition) for row in rows)
                if min_enabled <= support < len(rows):
                    conditions.append(condition)
    return conditions


def fit_gate(rows: list[dict], min_enabled: int = 3, max_depth: int = 2) -> tuple[dict, dict]:
    off = {"kind": "off"}
    best_rule = off
    best_metrics = policy_metrics(rows, [False] * len(rows))
    best_key = metric_key(best_metrics)
    conditions = atomic_conditions(rows, min_enabled)
    rules = list(conditions)
    if max_depth >= 2:
        for left, right in itertools.combinations(conditions, 2):
            if left["feature"] == right["feature"]:
                continue
            rules.append({"kind": "all", "conditions": [left, right]})
    for rule in rules:
        enabled = [rule_matches(row, rule) for row in rows]
        if sum(enabled) < min_enabled or sum(enabled) == len(rows):
            continue
        value = policy_metrics(rows, enabled)
        key = metric_key(value)
        rule_key = json.dumps(rule, sort_keys=True)
        best_rule_key = json.dumps(best_rule, sort_keys=True)
        if key > best_key or (key == best_key and rule_key < best_rule_key):
            best_rule, best_metrics, best_key = rule, value, key
    return best_rule, best_metrics


def leave_one_out(rows: list[dict], min_enabled: int, max_depth: int) -> tuple[dict, list[dict]]:
    enabled = []
    decisions = []
    for index, row in enumerate(rows):
        training = rows[:index] + rows[index + 1 :]
        rule, _ = fit_gate(
            training,
            min(min_enabled, max(1, len(training) - 1)),
            max_depth=max_depth,
        )
        decision = rule_matches(row, rule)
        enabled.append(decision)
        decisions.append({"instance_id": row["instance_id"], "enabled": decision, "rule": rule})
    return policy_metrics(rows, enabled), decisions


def leave_one_repo_out(rows: list[dict], min_enabled: int, max_depth: int) -> tuple[dict, list[dict]]:
    predictions: dict[str, bool] = {}
    decisions = []
    for repo in sorted({row["repo"] for row in rows}):
        training = [row for row in rows if row["repo"] != repo]
        validation = [row for row in rows if row["repo"] == repo]
        if not training or not validation:
            continue
        rule, _ = fit_gate(
            training,
            min(min_enabled, max(1, len(training) - 1)),
            max_depth=max_depth,
        )
        for row in validation:
            predictions[row["instance_id"]] = rule_matches(row, rule)
        decisions.append(
            {
                "held_out_repo": repo,
                "training_tasks": len(training),
                "validation_tasks": len(validation),
                "rule": rule,
            }
        )
    enabled = [predictions.get(row["instance_id"], False) for row in rows]
    return policy_metrics(rows, enabled), decisions


def stratified_k_fold(
    rows: list[dict],
    min_enabled: int,
    max_depth: int,
    folds: int = 5,
    seed: int = 52,
) -> tuple[dict, list[dict]]:
    """Return out-of-fold decisions from deterministic repository-stratified folds."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    fold_indices: list[list[int]] = [[] for _ in range(folds)]
    for repo in sorted({row["repo"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["repo"] == repo]
        random.Random(seed + BENCHMARK.stable_hash(repo)).shuffle(indices)
        for position, index in enumerate(indices):
            fold_indices[position % folds].append(index)

    predictions = [False] * len(rows)
    decisions = []
    for fold, validation_indices in enumerate(fold_indices):
        if not validation_indices:
            continue
        validation_set = set(validation_indices)
        training = [row for index, row in enumerate(rows) if index not in validation_set]
        rule, _ = fit_gate(
            training,
            min(min_enabled, max(1, len(training) - 1)),
            max_depth=max_depth,
        )
        for index in validation_indices:
            predictions[index] = rule_matches(rows[index], rule)
        decisions.append(
            {
                "fold": fold,
                "training_tasks": len(training),
                "validation_tasks": len(validation_indices),
                "validation_repositories": sorted(
                    {rows[index]["repo"] for index in validation_indices}
                ),
                "rule": rule,
            }
        )
    return policy_metrics(rows, predictions), decisions


def metric_deltas(rows: list[dict], enabled: list[bool]) -> dict[str, float]:
    baseline = policy_metrics(rows, [False] * len(rows))
    candidate = policy_metrics(rows, enabled)
    return {
        metric: float(candidate[metric]) - float(baseline[metric])
        for metric in ("hit_at_1", "hit_at_5", "mrr")
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def stratified_bootstrap_deltas(
    rows: list[dict],
    enabled: list[bool],
    samples: int = 5_000,
    seed: int = 52,
) -> dict[str, dict[str, float]]:
    """Estimate paired metric deltas while preserving repository mixture."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    by_repo: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_repo.setdefault(row["repo"], []).append(index)
    rng = random.Random(seed)
    draws = {metric: [] for metric in ("hit_at_1", "hit_at_5", "mrr")}
    for _ in range(samples):
        sampled_indices = []
        for repo in sorted(by_repo):
            indices = by_repo[repo]
            sampled_indices.extend(rng.choice(indices) for _ in indices)
        sampled_rows = [rows[index] for index in sampled_indices]
        sampled_enabled = [enabled[index] for index in sampled_indices]
        for metric, value in metric_deltas(sampled_rows, sampled_enabled).items():
            draws[metric].append(value)
    point = metric_deltas(rows, enabled)
    return {
        metric: {
            "point_delta": point[metric],
            "lower_95": percentile(sorted(values), 0.025),
            "upper_95": percentile(sorted(values), 0.975),
        }
        for metric, values in draws.items()
    }


def confidence_no_regression(intervals: dict[str, dict[str, float]]) -> bool:
    return all(value["lower_95"] >= 0.0 for value in intervals.values())


def summarize_feature(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def distribution_shift(dev_rows: list[dict], test_rows: list[dict]) -> dict[str, dict]:
    result = {}
    for feature in FEATURES:
        dev_values = [row["features"][feature] for row in dev_rows]
        test_values = [row["features"][feature] for row in test_rows]
        dev = summarize_feature(dev_values)
        test = summarize_feature(test_values)
        pooled = math.sqrt(
            (dev["standard_deviation"] ** 2 + test["standard_deviation"] ** 2) / 2.0
        )
        result[feature] = {
            "dev": dev,
            "holdout": test,
            "standardized_mean_difference": (
                (test["mean"] - dev["mean"]) / pooled if pooled > 1e-12 else 0.0
            ),
        }
    return result


def outcome_counts(rows: list[dict]) -> dict[str, int]:
    return {
        outcome: sum(row["v1_outcome"] == outcome for row in rows)
        for outcome in ("win", "tie", "loss")
    }


def no_primary_regression(candidate: dict, baseline: dict) -> bool:
    return (
        float(candidate["hit_at_1"]) >= float(baseline["hit_at_1"])
        and float(candidate["hit_at_5"]) >= float(baseline["hit_at_5"])
        and float(candidate["mrr"]) >= float(baseline["mrr"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/public/tasks.jsonl")
    parser.add_argument("--repositories", type=Path, default=ROOT / "data/raw/source-repos")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/2026-10-08-localization-v1-gate/gate-study.json",
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        help="Optional generated gate JSON; review it before copying into the agent bundle.",
    )
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--min-enabled", type=int, default=8)
    parser.add_argument("--max-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()

    started = time.perf_counter()
    tasks = BENCHMARK.load_tasks(args.tasks)
    splits = BENCHMARK.split_tasks(tasks, args.seed)
    study_tasks = [*splits["train"], *splits["dev"], *splits["test"]]
    snapshots, inventories, repository_stats = BENCHMARK.load_snapshots(
        study_tasks, args.repositories
    )
    model = LOCALIZER.LegacyModel.load(BENCHMARK.MODEL_PATH)
    train_rows = build_rows(
        splits["train"], snapshots, inventories, model, DEFAULT_CANDIDATE_WEIGHTS
    )
    dev_rows = build_rows(
        splits["dev"], snapshots, inventories, model, DEFAULT_CANDIDATE_WEIGHTS
    )
    test_rows = build_rows(
        splits["test"], snapshots, inventories, model, DEFAULT_CANDIDATE_WEIGHTS
    )

    selected_rule, train_fit = fit_gate(train_rows, args.min_enabled, args.max_depth)
    train_baseline = policy_metrics(train_rows, [False] * len(train_rows))
    train_always_v1 = policy_metrics(train_rows, [True] * len(train_rows))
    train_cv, cv_decisions = stratified_k_fold(
        train_rows, args.min_enabled, args.max_depth, args.folds, args.seed
    )
    train_logo, logo_decisions = leave_one_repo_out(
        train_rows, args.min_enabled, args.max_depth
    )
    dev_baseline = policy_metrics(dev_rows, [False] * len(dev_rows))
    dev_always_v1 = policy_metrics(dev_rows, [True] * len(dev_rows))
    dev_enabled = [rule_matches(row, selected_rule) for row in dev_rows]
    dev_gated = policy_metrics(dev_rows, dev_enabled)
    dev_confidence = stratified_bootstrap_deltas(
        dev_rows, dev_enabled, args.bootstrap_samples, args.seed + 1
    )
    test_baseline = policy_metrics(test_rows, [False] * len(test_rows))
    test_always_v1 = policy_metrics(test_rows, [True] * len(test_rows))
    test_enabled = [rule_matches(row, selected_rule) for row in test_rows]
    test_gated = policy_metrics(test_rows, test_enabled)
    test_confidence = stratified_bootstrap_deltas(
        test_rows, test_enabled, args.bootstrap_samples, args.seed + 2
    )

    checks = {
        # This checkpoint was fitted on the very training tasks used by gate CV.
        # Cross-fitting only the gate cannot establish whole-system generalization.
        "ranker_cross_fitted_for_calibration": False,
        "fresh_confirmation_passed": False,
        "remote_paired_repair_improvement": False,
        "nontrivial_rule": selected_rule["kind"] != "off",
        "training_fit_no_primary_regression": no_primary_regression(
            train_fit, train_baseline
        ),
        "training_cross_validation_no_primary_regression": no_primary_regression(
            train_cv, train_baseline
        ),
        "training_leave_one_repo_out_no_primary_regression": no_primary_regression(
            train_logo, train_baseline
        ),
        "development_no_primary_regression": no_primary_regression(
            dev_gated, dev_baseline
        ),
        "development_bootstrap_95_no_regression": confidence_no_regression(
            dev_confidence
        ),
        "holdout_no_primary_regression": no_primary_regression(test_gated, test_baseline),
        "holdout_bootstrap_95_no_regression": confidence_no_regression(test_confidence),
        "holdout_has_v1_activation": int(test_gated["v1_enabled"]) > 0,
    }
    deploy = all(checks.values())
    gate = {
        "schema_version": 1,
        "enabled": deploy,
        "fallback": "bm25",
        "candidate": "v1_hybrid",
        "candidate_weights": DEFAULT_CANDIDATE_WEIGHTS,
        "features": list(FEATURES),
        "rule": selected_rule,
        "safety_checks": checks,
        "decision": (
            "Enable the learned gate; all development, cross-validation, and holdout checks passed."
            if deploy
            else "Keep v1 disabled by default; at least one promotion safety check failed."
        ),
    }
    payload = {
        "study": "label-safe BM25-versus-v1 runtime gating",
        "label_policy": "gold paths fit the training rule and score frozen development/holdout evaluations only; runtime features contain no gold data",
        "holdout_status": "retrospective: this repository had already reported aggregate holdout results before the gate study",
        "calibration_limitation": "Legacy checkpoint saw training labels; gate-only CV is not out-of-fold ranker evidence. Use train_portable_localization.py for whole-system nested validation.",
        "seed": args.seed,
        "learner": {
            "kind": "conservative_conjunction",
            "max_depth": args.max_depth,
            "min_enabled": args.min_enabled,
            "cross_validation_folds": args.folds,
            "stratified_bootstrap_samples": args.bootstrap_samples,
        },
        "candidate_weights": DEFAULT_CANDIDATE_WEIGHTS,
        "task_counts": {name: len(rows) for name, rows in splits.items()},
        "scorable_counts": {
            "training": len(train_rows),
            "development": len(dev_rows),
            "holdout": len(test_rows),
        },
        "repository_stats": repository_stats,
        "selected_rule": selected_rule,
        "promotion": gate,
        "training": {
            "outcomes": outcome_counts(train_rows),
            "bm25": train_baseline,
            "always_v1": train_always_v1,
            "fitted_gate": train_fit,
            "cross_validation_gate": train_cv,
            "leave_one_repo_out_gate": train_logo,
        },
        "development": {
            "outcomes": outcome_counts(dev_rows),
            "bm25": dev_baseline,
            "always_v1": dev_always_v1,
            "frozen_gate": dev_gated,
            "bootstrap_deltas": dev_confidence,
        },
        "holdout": {
            "outcomes": outcome_counts(test_rows),
            "bm25": test_baseline,
            "always_v1": test_always_v1,
            "frozen_gate": test_gated,
            "bootstrap_deltas": test_confidence,
        },
        "distribution_shift": {
            "training_to_development": distribution_shift(train_rows, dev_rows),
            "training_to_holdout": distribution_shift(train_rows, test_rows),
            "development_to_holdout": distribution_shift(dev_rows, test_rows),
        },
        "cross_validation_decisions": cv_decisions,
        "leave_one_repo_out_decisions": logo_decisions,
        "task_results": {
            "training": train_rows,
            "development": dev_rows,
            "holdout": test_rows,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.gate_output:
        args.gate_output.parent.mkdir(parents=True, exist_ok=True)
        args.gate_output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_rule": selected_rule,
                "promotion": gate,
                "training": payload["training"],
                "development": payload["development"],
                "holdout": payload["holdout"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")
    if args.gate_output:
        print(f"wrote {args.gate_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
