"""Behavior and isolation checks for portable ranking and nested calibration."""
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
previous = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import train_portable_localization as training
finally:
    sys.dont_write_bytecode = previous
    sys.path.pop(0)
RUNTIME = training.RUNTIME


def fixture(identifier, repo, commit):
    return {"instance_id": identifier, "repo": repo, "commit": commit,
            "features": [[1.0, 0, 0, 0, 0, 0, 0], [0.5, 1, 0, 1, 0, 0, 0]],
            "gold_indices": [1]}


class PortableLocalizationTests(unittest.TestCase):
    def test_baseline_coordinate_preserves_bm25_order(self):
        docs = [RUNTIME.make_document("pkg/cache.py", "def flush_cache(): pass"),
                RUNTIME.make_document("pkg/screen.py", "def resize_screen(): pass")]
        features = RUNTIME.portable_features(docs, "resize_screen returns invalid width")
        raw = RUNTIME.bm25_scores(docs, RUNTIME.tokenize("resize_screen returns invalid width"))
        self.assertEqual(RUNTIME.descending_order(raw), RUNTIME.descending_order(
            RUNTIME.portable_scores(features, training.BASE_WEIGHTS)))

    def test_training_can_learn_transferable_path_evidence(self):
        rows = [fixture(str(i), "repo/a", str(i)) for i in range(8)]
        weights = training.fit_ranker(rows)
        self.assertGreater(training.evaluate(rows, weights)["mrr"],
                           training.evaluate(rows, training.BASE_WEIGHTS)["mrr"])

    def test_shared_snapshot_never_crosses_fold_boundaries(self):
        rows = [fixture("a", "repo/a", "same"), fixture("b", "repo/a", "same"),
                fixture("c", "repo/a", "other"), fixture("d", "repo/b", "other")]
        folds = training.folds_for(rows, "snapshot", 52, 3)
        self.assertEqual(sum(len(fold) for fold in folds), len(rows))
        self.assertTrue(any({"a", "b"} <= set(fold) for fold in folds))

    def test_outer_validation_excluded_from_ranker_and_gate_training(self):
        rows = [fixture(f"{repo}-{i}", repo, str(i))
                for repo in ("a", "b", "c") for i in range(3)]
        seen = []
        def fit(subset):
            seen.append({row["instance_id"] for row in subset})
            return training.BASE_WEIGHTS[:]
        with mock.patch.object(training, "fit_ranker", side_effect=fit):
            report = training.nested_validation(rows, "repository", 52)
        self.assertEqual(report["gated"]["tasks"], len(rows))
        self.assertTrue(all(len(ids) < len(rows) for ids in seen))
        for fold in report["folds"]:
            self.assertFalse(set(fold["training_ids"]) & set(fold["validation_ids"]))
        # Three inner refits and a final fit per outer fold; every one excludes that repo.
        for index, fold in enumerate(report["folds"]):
            for ids in seen[index * 4:(index + 1) * 4]:
                self.assertFalse(ids & set(fold["validation_ids"]))

    def test_gate_abstains_when_candidate_only_harms(self):
        predictions = [{"repo": "a", "bm25_rank": 1, "candidate_rank": 4,
                        "bm25_margin": 0.1, "candidate_margin": 0.5} for _ in range(10)]
        self.assertEqual(training.fit_gate(predictions), {"kind": "off"})

    def test_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            RUNTIME.portable_scores([[1] * 7], [float("nan")] * 7)

    def test_disabled_portable_policy_does_not_compute_candidate(self):
        docs = [RUNTIME.make_document("pkg/cache.py", "def flush_cache(): pass")]
        policy = RUNTIME.PortablePolicy([1, 2, 0, 2, 2, 0, 1], {"kind": "off"})
        with mock.patch.object(RUNTIME, "portable_features", side_effect=AssertionError("expensive path")):
            results, active = RUNTIME.rank_documents_portable(docs, "flush_cache", policy)
        self.assertFalse(active)
        self.assertEqual(results[0].path, "pkg/cache.py")

    def test_runtime_and_training_choose_identical_ranking_and_gate(self):
        docs = [RUNTIME.make_document("pkg/z.py", "def flush_cache(): pass"),
                RUNTIME.make_document("pkg/a.py", "def resize_screen(): pass")]
        weights = [1, 2, 0, 2, 2, 0, 1]
        gate = {"kind": "confidence", "max_bm25_margin": 1.0, "min_candidate_margin": 0.0}
        ordered = sorted(docs, key=lambda doc: doc.path)
        features = RUNTIME.portable_features(ordered, "flush_cache error")
        row = {"instance_id": "x", "repo": "x", "commit": "x", "features": features, "gold_indices": [1]}
        predicted = training.prediction(row, weights)
        results, active = RUNTIME.rank_documents_portable(
            docs, "flush_cache error", RUNTIME.PortablePolicy(weights, gate), experimental=True)
        self.assertEqual(active, training.matches(predicted, gate))
        self.assertEqual([r.path for r in results], [ordered[i].path for i in training.ranked(row, weights)[1]])

    def test_policy_loader_rejects_truthy_string_activation(self):
        payload = {"schema_version": 1, "model_type": "portable_localization_v1",
                   "features": list(RUNTIME.PORTABLE_FEATURE_NAMES), "weights": [1] * 7,
                   "gate": {"kind": "off"}, "enabled": "false"}
        with mock.patch.object(Path, "read_text", return_value=json.dumps(payload)):
            with self.assertRaises(ValueError):
                RUNTIME.PortablePolicy.load(Path("fixture.json"))


if __name__ == "__main__":
    unittest.main()
