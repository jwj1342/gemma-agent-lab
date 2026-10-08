"""Focused CPU tests for the packaged Localization v1 skill."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "configs/experiments/localization-v1/skills/localization-v1"
LOCALIZER_PATH = SKILL / "scripts/localize.py"


def load_localizer():
    spec = importlib.util.spec_from_file_location("localization_v1_test_runtime", LOCALIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LOCALIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


LOCALIZER = load_localizer()


class LocalizationV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = ROOT / ".tmp-tests"
        cls.temp_root.mkdir(exist_ok=True)

    def temporary(self):
        return tempfile.TemporaryDirectory(prefix="localization-v1-", dir=self.temp_root)

    def test_hybrid_ranks_distinctive_symbol_and_source_terms(self):
        documents = [
            LOCALIZER.make_document("src/cache.py", "def clear_cache():\n    return None\n"),
            LOCALIZER.make_document(
                "src/widget_handler.py",
                "def process_widget():\n    # enforce the retry budget\n    return None\n",
            ),
        ]
        results = LOCALIZER.rank_documents(
            documents,
            "process_widget ignores the retry budget",
            mode="hybrid",
            k=2,
        )
        self.assertEqual(results[0].path, "src/widget_handler.py")
        self.assertEqual(results[0].symbol, "process_widget")

    def test_signed_component_keeps_relative_ranking(self):
        fused = LOCALIZER.reciprocal_rank_fusion(
            {"legacy": [-2.0, -1.0, 0.0]}, {"legacy": 1.0}
        )
        self.assertGreater(fused[2], fused[1])
        self.assertGreater(fused[1], fused[0])

    def test_repository_scan_excludes_tests_and_symlinks(self):
        with self.temporary() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "tests").mkdir()
            (root / "pkg/main.py").write_text("def main():\n    pass\n", encoding="utf-8")
            (root / "tests/test_main.py").write_text("def test_main():\n    pass\n", encoding="utf-8")
            try:
                (root / "pkg/link.py").symlink_to(root / "pkg/main.py")
            except OSError:
                pass
            paths = [document.path for document in LOCALIZER.read_repository(root)]
        self.assertEqual(paths, ["pkg/main.py"])

    def test_model_loader_rejects_missing_and_incompatible_models(self):
        with self.temporary() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                LOCALIZER.LegacyModel.load(root / "missing.json")
            incompatible = root / "model.json"
            incompatible.write_text(
                json.dumps(
                    {
                        "model_type": "hashed_pairwise_file_localizer_v1",
                        "feature_set": "old_features",
                        "dimension": 8,
                        "weights": [[1, 0.5]],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "feature set"):
                LOCALIZER.LegacyModel.load(incompatible)

    def test_cli_emits_bounded_ranked_paths(self):
        with self.temporary() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "pkg/widget.py").write_text(
                "def process_widget():\n    return 'retry budget'\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(LOCALIZER_PATH),
                    "--root",
                    str(root),
                    "--k",
                    "1",
                    "--issue",
                    "process_widget retry budget",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("adaptive(v1=off)", result.stdout)
        self.assertIn("top 1 of 1 files", result.stdout)
        self.assertIn("pkg/widget.py:1 process_widget", result.stdout)

    def test_disabled_gate_is_a_fail_safe_bm25_alias(self):
        gate = LOCALIZER.AdaptiveGate.load(SKILL / "resources/gate.json")
        self.assertFalse(gate.enabled)
        documents = [
            LOCALIZER.make_document("pkg/cache.py", "def clear_cache():\n    pass\n"),
            LOCALIZER.make_document("pkg/widget.py", "def process_widget():\n    pass\n"),
        ]
        adaptive, use_v1, features = LOCALIZER.rank_documents_adaptive(
            documents, "process_widget fails", gate=gate, model=None, k=2
        )
        bm25 = LOCALIZER.rank_documents(documents, "process_widget fails", mode="bm25", k=2)
        self.assertFalse(use_v1)
        self.assertEqual(features, {})
        self.assertEqual([row.path for row in adaptive], [row.path for row in bm25])

    def test_enabled_gate_can_select_v1_per_issue(self):
        class ZeroModel:
            @staticmethod
            def score(issue, path):
                return 0.0

        gate = LOCALIZER.AdaptiveGate(
            enabled=True,
            candidate_weights={"bm25": 1.0, "exact": 0.4, "legacy": 0.5},
            rule={
                "kind": "stump",
                "feature": "log_candidates",
                "operator": ">",
                "threshold": 0.0,
            },
            decision="test fixture",
        )
        documents = [
            LOCALIZER.make_document("pkg/cache.py", "def clear_cache():\n    pass\n"),
            LOCALIZER.make_document("pkg/widget.py", "def process_widget():\n    pass\n"),
        ]
        _results, use_v1, features = LOCALIZER.rank_documents_adaptive(
            documents, "process_widget fails", gate=gate, model=ZeroModel(), k=2
        )
        self.assertTrue(use_v1)
        self.assertEqual(set(features), set(LOCALIZER.GATE_FEATURE_NAMES))

    def test_skill_is_attached_directly_with_a_root_relative_path(self):
        agent = (ROOT / "configs/experiments/localization-v1/agent.yaml").read_text()
        analyzer = (
            ROOT / "configs/experiments/localization-v1/sub_agents/code_analyzer.yaml"
        ).read_text()
        self.assertIn("skills:\n  - skills/localization-v1\n", agent)
        self.assertNotIn("skills:", analyzer)
        self.assertNotIn("../skills", agent + analyzer)


if __name__ == "__main__":
    unittest.main()
