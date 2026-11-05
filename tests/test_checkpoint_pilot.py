"""Verify paired seed isolation, source identity and reproducible packaging."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_checkpoint_pilot import build_pilot, inventory


class CheckpointPilotTests(unittest.TestCase):
    def test_four_paired_seeds_and_no_control_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'control'
            (source / 'configs').mkdir(parents=True)
            (source / 'prompts').mkdir()
            (source / 'agent.yaml').write_bytes(b'name: fixture\n')
            sampling = b'temperature: 0.2\n'
            (source / 'configs/sampling.yaml').write_bytes(sampling)
            (source / 'prompts/system.md').write_bytes(b'prefix\nOLD\nsuffix\n')
            before = inventory(source)
            spec = {'name': 'fixture', 'control_members': before,
                    'prompt_path': 'prompts/system.md', 'old_paragraph': 'OLD',
                    'new_paragraph': 'NEW', 'evaluation': {'seeds': [42, 1, 2, 3]}}
            spec_path = root / 'spec.json'
            spec_path.write_text(json.dumps(spec))
            with patch('scripts.build_checkpoint_pilot.SPEC', spec_path):
                first = build_pilot(source, root / 'first')
                second = build_pilot(source, root / 'second')
                self.assertEqual(first, second)
                self.assertEqual([r['sampling_seed'] for r in first['pairs']], [None, 1, 2, 3])
                for seed in [42, 1, 2, 3]:
                    expected = sampling + (b'' if seed == 42 else f'seed: {seed}\n'.encode())
                    for arm in ['control', 'candidate']:
                        agent = root / 'first' / f'seed-{seed}' / arm / 'agent'
                        self.assertEqual((agent / 'configs/sampling.yaml').read_bytes(), expected)
                    control = first['pairs'][[42, 1, 2, 3].index(seed)]
                    self.assertEqual(control['changed_members'], ['prompts/system.md'])
                self.assertEqual(inventory(source), before)
                for output in [root / 'first', source / 'nested', root]:
                    with self.assertRaises(ValueError):
                        build_pilot(source, output)
                (source / 'prompts/system.md').write_bytes(b'tampered\n')
                with self.assertRaises(ValueError):
                    build_pilot(source, root / 'tampered')
                self.assertFalse((root / 'tampered').exists())
