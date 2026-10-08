"""Unit tests for the dependency-free Localization v1 gate learner."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts/train_localization_gate.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("localization_gate_test_runtime", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


TRAINER = load_trainer()


def row(name, signal, bm25_rank, candidate_rank, repo="fixture/repo"):
    features = {feature: 0.25 for feature in TRAINER.FEATURES}
    features["bm25_margin"] = signal
    return {
        "instance_id": name,
        "repo": repo,
        "features": features,
        "bm25_rank": bm25_rank,
        "candidate_rank": candidate_rank,
        "v1_outcome": "win" if candidate_rank < bm25_rank else "loss",
    }


class LocalizationGateTests(unittest.TestCase):
    def test_gate_feature_schema_is_shared_with_runtime(self):
        self.assertEqual(TRAINER.FEATURES, TRAINER.LOCALIZER.GATE_FEATURE_NAMES)

    def test_depth_one_gate_learns_when_v1_has_a_separable_safe_region(self):
        rows = [
            row("low-1", 0.1, 5, 1),
            row("low-2", 0.2, 4, 1),
            row("low-3", 0.3, 6, 2),
            row("high-1", 0.7, 1, 8),
            row("high-2", 0.8, 1, 6),
            row("high-3", 0.9, 2, 5),
        ]
        rule, gated = TRAINER.fit_gate(rows, min_enabled=3, max_depth=1)
        baseline = TRAINER.policy_metrics(rows, [False] * len(rows))
        self.assertEqual(rule["feature"], "bm25_margin")
        self.assertEqual(rule["operator"], "<=")
        self.assertGreater(gated["hit_at_1"], baseline["hit_at_1"])
        self.assertTrue(TRAINER.rule_matches(rows[0], rule))
        self.assertFalse(TRAINER.rule_matches(rows[-1], rule))

    def test_distribution_report_detects_a_shift(self):
        dev = [row("dev-1", 0.1, 1, 1), row("dev-2", 0.2, 1, 1)]
        holdout = [row("test-1", 0.8, 1, 1), row("test-2", 0.9, 1, 1)]
        shift = TRAINER.distribution_shift(dev, holdout)
        self.assertGreater(shift["bm25_margin"]["standardized_mean_difference"], 1.0)

    def test_leave_one_repo_out_never_trains_on_the_validation_repo(self):
        rows = [
            row("a-1", 0.1, 4, 1, "repo/a"),
            row("a-2", 0.2, 5, 1, "repo/a"),
            row("b-1", 0.8, 1, 5, "repo/b"),
            row("b-2", 0.9, 1, 6, "repo/b"),
        ]
        metrics, decisions = TRAINER.leave_one_repo_out(rows, min_enabled=1, max_depth=1)
        self.assertEqual(metrics["tasks"], 4)
        self.assertEqual({item["held_out_repo"] for item in decisions}, {"repo/a", "repo/b"})
        self.assertTrue(all(item["training_tasks"] == 2 for item in decisions))

    def test_stratified_folds_cover_every_task_once(self):
        rows = [
            row(f"a-{index}", index / 10, 4, 1, "repo/a")
            for index in range(6)
        ] + [
            row(f"b-{index}", 0.5 + index / 10, 1, 4, "repo/b")
            for index in range(6)
        ]
        metrics, decisions = TRAINER.stratified_k_fold(
            rows, min_enabled=2, max_depth=1, folds=3, seed=7
        )
        self.assertEqual(metrics["tasks"], len(rows))
        self.assertEqual(sum(item["validation_tasks"] for item in decisions), len(rows))
        self.assertTrue(
            all(item["validation_repositories"] == ["repo/a", "repo/b"] for item in decisions)
        )

    def test_stratified_bootstrap_reports_paired_improvement(self):
        rows = [
            row("a-1", 0.1, 5, 1, "repo/a"),
            row("a-2", 0.2, 4, 1, "repo/a"),
            row("b-1", 0.3, 6, 1, "repo/b"),
            row("b-2", 0.4, 3, 1, "repo/b"),
        ]
        intervals = TRAINER.stratified_bootstrap_deltas(
            rows, [True] * len(rows), samples=100, seed=3
        )
        self.assertGreater(intervals["hit_at_1"]["lower_95"], 0.0)
        self.assertTrue(TRAINER.confidence_no_regression(intervals))


if __name__ == "__main__":
    unittest.main()
