"""Check prompt ablation boundaries and a self-contained competition archive."""
import tempfile
from pathlib import Path
import unittest
import sys
import re
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_prompt_candidate import build_candidate
from build_submission import build


class PromptCandidateTests(unittest.TestCase):
    def test_ablations_and_archive(self):
        source = Path(__file__).resolve().parents[1] / 'configs/prompt-icl'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('workflow', 'protocol', 'recover', 'icl', 'reflect'):
                output = root / variant
                receipt = build_candidate(source, output, variant)
                text = (output / 'prompts/system.md').read_text()
                self.assertEqual(receipt['weight_updates'], 0)
                self.assertEqual('Small examples' in text, variant in ('icl', 'reflect'))
                self.assertEqual('Tool arguments and recovery' in text,
                                 variant in ('protocol', 'recover'))
                self.assertEqual('inline Python heredoc' in text, variant == 'recover')
                self.assertEqual('PYTHONDONTWRITEBYTECODE=1' in text, variant == 'recover')
                for step in ('1. Validate the replacement', '2. Write the source',
                             '3. Verify the actual implementation'):
                    self.assertEqual(step in text, variant == 'recover')
                self.assertNotIn('rm fix.py', text)
                self.assertEqual('Learn from execution' in text, variant == 'reflect')
                self.assertEqual((output / 'skills').exists(), variant == 'reflect')
                if variant == 'reflect':
                    skill = output / 'skills/repair-playbook'
                    instructions = (skill / 'SKILL.md').read_text()
                    metadata = yaml.safe_load(instructions.split('---', 2)[1])
                    self.assertEqual(metadata['name'], skill.name)
                    self.assertRegex(skill.name, r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
                    references = re.findall(r'`(references/[^`]+)`', instructions)
                    self.assertEqual(len(references), 4)
                    self.assertTrue(all((skill / name).is_file() for name in references))
                    self.assertFalse((skill / 'resources').exists())
                self.assertNotIn('adapter:', (output / 'agent.yaml').read_text())
                self.assertTrue(build(output, root / (variant + '.zip'))['sha256'])
                with self.assertRaises(ValueError):
                    build_candidate(source, output, variant)


if __name__ == '__main__':
    unittest.main()
