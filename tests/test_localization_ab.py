"""Tests for paired repair evidence and failure-denominator handling."""
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import summarize_localization_ab as paired
finally:
    sys.path.pop(0)


class LocalizationABTests(unittest.TestCase):
    def test_exact_paired_test_distinguishes_wins_from_ties(self):
        result = paired.paired_comparison([(False, True)] * 6 + [(True, True)] * 10)
        self.assertEqual(result["one_sided_exact_mcnemar_p"], 1 / 64)
        self.assertEqual(result["absolute_resolution_gain"], 6 / 16)
        self.assertEqual(paired.paired_comparison([(True, True)] * 20)["one_sided_exact_mcnemar_p"], 1.0)

    def test_missing_candidate_result_cannot_disappear_from_denominator(self):
        protocol = {"confirmation_task_ids": ["a"], "cases": [
            {"task_id": "a", "arm": arm, "path": arm, "bundle_sha256": "b", "public_task_sha256": "t"}
            for arm in ("baseline", "bm25", "portable")],
            "acceptance": {"minimum_absolute_resolution_gain": 0.05,
                           "one_sided_exact_mcnemar_p_max": 0.05, "max_mean_agent_time_ratio": 1.1}}
        def outcome(path):
            if path.name == "portable":
                raise FileNotFoundError("missing prediction")
            return {"agent_resolved": True, "controls_valid": True,
                    "pipeline_executed": True, "inference_seconds": 20}
        with mock.patch.object(Path, "read_text", return_value=json.dumps(protocol)), \
             mock.patch.object(paired, "verify_inputs", return_value={
                 "agent_archive_sha256": "b", "public_task_sha256": "t"}), \
             mock.patch.object(paired, "summarize", side_effect=outcome):
            result = paired.report(Path("fixture"))
        self.assertFalse(result["remote_evidence_passed"])
        self.assertEqual(result["comparisons"]["portable_vs_baseline"]["tasks"], 1)
        self.assertEqual(result["comparisons"]["portable_vs_baseline"]["losses"], 1)


if __name__ == "__main__":
    unittest.main()
