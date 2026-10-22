"""Focused CPU tests for the packaged repair checkpoint skill."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/repair-checkpoint-v1"
CHECKPOINT_PATH = EXPERIMENT / "skills/repair-checkpoint/scripts/checkpoint.py"


def load_checkpoint():
    spec = importlib.util.spec_from_file_location("repair_checkpoint_test_runtime", CHECKPOINT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CHECKPOINT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


CHECKPOINT = load_checkpoint()


class RepairCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = ROOT / ".tmp-tests"
        cls.temp_root.mkdir(exist_ok=True)

    def repository(self):
        temporary = tempfile.TemporaryDirectory(prefix="repair-checkpoint-", dir=self.temp_root)
        root = Path(temporary.name)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "Checkpoint Test"], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "checkpoint@example.invalid"],
            check=True,
        )
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src/widget.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "tests/test_widget.py").write_text("def test_widget():\n    pass\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
        return temporary, root

    def test_valid_source_change_is_ready(self):
        temporary, root = self.repository()
        with temporary:
            (root / "src/widget.py").write_text("VALUE = 2\n", encoding="utf-8")
            result = CHECKPOINT.analyze(root)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["candidate_files"], ["src/widget.py"])
        self.assertEqual(result["changed_files"][0]["added"], 1)
        self.assertEqual(result["changed_files"][0]["deleted"], 1)

    def test_empty_patch_requires_repair(self):
        temporary, root = self.repository()
        with temporary:
            result = CHECKPOINT.analyze(root)
        self.assertEqual(result["status"], "repair")
        self.assertEqual(result["reasons"], ["empty_patch"])

    def test_protected_edit_requires_repair(self):
        temporary, root = self.repository()
        with temporary:
            (root / "tests/test_widget.py").write_text("def test_widget():\n    assert False\n", encoding="utf-8")
            result = CHECKPOINT.analyze(root)
        self.assertEqual(result["status"], "repair")
        self.assertIn("protected_edits", result["reasons"])
        self.assertIn("no_candidate_source_change", result["reasons"])
        self.assertEqual(result["protected_edits"], ["tests/test_widget.py"])

    def test_syntax_conflict_and_debugger_residue_are_reported(self):
        temporary, root = self.repository()
        with temporary:
            (root / "src/widget.py").write_text(
                "<<<<<<< ours\ndef broken(:\n    breakpoint()\n=======\n>>>>>>> theirs\n",
                encoding="utf-8",
            )
            result = CHECKPOINT.analyze(root)
        self.assertEqual(result["status"], "repair")
        self.assertIn("python_syntax_errors", result["reasons"])
        self.assertIn("conflict_markers", result["reasons"])
        self.assertIn("debugger_residue", result["reasons"])

    def test_untracked_source_is_checked_without_staging(self):
        temporary, root = self.repository()
        with temporary:
            path = root / "src/new_module.py"
            path.write_text("def created():\n    return True\n", encoding="utf-8")
            before = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1"])
            result = CHECKPOINT.analyze(root)
            after = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["candidate_files"], ["src/new_module.py"])
        self.assertEqual(before, after)

    def test_cli_and_agent_contract(self):
        temporary, root = self.repository()
        with temporary:
            (root / "src/widget.py").write_text("VALUE = 3\n", encoding="utf-8")
            process = subprocess.run(
                [sys.executable, str(CHECKPOINT_PATH), "--root", str(root)],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "ready")
        agent = (EXPERIMENT / "agent.yaml").read_text(encoding="utf-8")
        prompt = (EXPERIMENT / "prompts/system.md").read_text(encoding="utf-8")
        self.assertIn("skills:\n  - skills/repair-checkpoint\n", agent)
        self.assertIn("`file_path`: `scripts/checkpoint.py`", prompt)
        self.assertNotIn("../skills", agent)


if __name__ == "__main__":
    unittest.main()
