"""Check that the issue #12 candidate changes only the registered prompt policy."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_submission import build


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SelectiveGraphCandidateTests(unittest.TestCase):
    def test_non_prompt_configuration_matches_baseline(self):
        baseline = ROOT / 'configs/baseline'
        candidate = ROOT / 'configs/prompt-selective-graph'
        baseline_files = {
            path.relative_to(baseline)
            for path in baseline.rglob('*')
            if path.is_file()
        }
        candidate_files = {
            path.relative_to(candidate)
            for path in candidate.rglob('*')
            if path.is_file()
        }
        self.assertEqual(baseline_files, candidate_files)
        for relative in (
            'agent.yaml',
            'configs/sampling.yaml',
            'eval_config.yaml',
            'prompts/analyzer.md',
            'sub_agents/code_analyzer.yaml',
        ):
            self.assertEqual(
                (baseline / relative).read_bytes(),
                (candidate / relative).read_bytes(),
                relative,
            )

    def test_prompt_bounds_graph_retrieval(self):
        prompt = (ROOT / 'configs/prompt-selective-graph/prompts/system.md').read_text()
        self.assertIn('After identifying the first concrete source symbol', prompt)
        self.assertIn('call `get_code_neighbors` once', prompt)
        self.assertIn('`max_neighbors=8`', prompt)
        self.assertIn('at most two relevant neighbors', prompt)
        self.assertIn('Do not use issue prose', prompt)
        self.assertIn('at least two exact graph node names', prompt)
        self.assertNotIn(
            'adapter:',
            (ROOT / 'configs/prompt-selective-graph/agent.yaml').read_text(),
        )

    def test_preregistration_matches_frozen_inputs(self):
        registration = json.loads(
            (ROOT / 'configs/evaluation/selective-graph-v1.json').read_text()
        )
        self.assertEqual(
            registration['benchmark']['sha256'],
            sha256(ROOT / registration['benchmark']['path']),
        )
        for name in ('baseline', 'candidate'):
            agent = registration['agents'][name]
            root = ROOT / agent['path']
            self.assertEqual(
                agent['system_prompt_sha256'],
                sha256(root / 'prompts/system.md'),
            )
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / f'{name}.zip'
                self.assertEqual(agent['archive_sha256'], build(root, archive)['sha256'])

        fixed = registration['agents']['fixed_configuration']
        baseline = ROOT / registration['agents']['baseline']['path']
        self.assertEqual(fixed['agent_yaml_sha256'], sha256(baseline / 'agent.yaml'))
        self.assertEqual(
            fixed['sampling_yaml_sha256'],
            sha256(baseline / 'configs/sampling.yaml'),
        )
        self.assertEqual(
            fixed['eval_config_sha256'],
            sha256(baseline / 'eval_config.yaml'),
        )


if __name__ == '__main__':
    unittest.main()
