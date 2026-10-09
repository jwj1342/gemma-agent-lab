"""Check prompt ablation boundaries and a self-contained competition archive."""
import tempfile
from pathlib import Path
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_prompt_candidate import build_candidate
from build_submission import build


class PromptCandidateTests(unittest.TestCase):
    def test_ablations_and_archive(self):
        source = Path(__file__).resolve().parents[1] / 'configs/prompt-icl'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('workflow', 'icl', 'reflect'):
                output = root / variant
                receipt = build_candidate(source, output, variant)
                text = (output / 'prompts/system.md').read_text()
                self.assertEqual(receipt['weight_updates'], 0)
                self.assertEqual('Small examples' in text, variant != 'workflow')
                self.assertEqual('Learn from execution' in text, variant == 'reflect')
                self.assertEqual((output / 'skills').exists(), variant == 'reflect')
                self.assertNotIn('adapter:', (output / 'agent.yaml').read_text())
                self.assertTrue(build(output, root / (variant + '.zip'))['sha256'])
                with self.assertRaises(ValueError):
                    build_candidate(source, output, variant)


if __name__ == '__main__':
    unittest.main()
