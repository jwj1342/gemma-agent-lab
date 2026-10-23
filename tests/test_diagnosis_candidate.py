"""Check one-factor candidate boundaries and deterministic private packaging."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.build_diagnosis_candidate import build_candidate


class DiagnosisCandidateTests(unittest.TestCase):
    def test_identity_one_member_diff_determinism_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'control'
            source.mkdir()
            (source / 'agent.yaml').write_text('name: test\n')
            (source / 'prompt.md').write_bytes(b'before\nOLD\nafter\n')
            spec = {'name': 'fixture', 'control_members': {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
                    'prompt_path': 'prompt.md', 'old_paragraph': 'OLD', 'new_paragraph': 'NEW'}
            spec_path = root / 'spec.json'
            spec_path.write_text(json.dumps(spec))
            first = build_candidate(source, root / 'first', spec_path)
            second = build_candidate(source, root / 'second', spec_path)
            self.assertEqual(first['archive_sha256'], second['archive_sha256'])
            self.assertEqual(first['changed_members'], ['prompt.md'])
            self.assertEqual((root / 'first/agent/agent.yaml').read_bytes(), (source / 'agent.yaml').read_bytes())
            with zipfile.ZipFile(root / 'first/submission.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.read('prompt.md'), b'before\nNEW\nafter\n')
            for output in (root / 'first', source / 'nested', root):
                with self.assertRaises(ValueError):
                    build_candidate(source, output, spec_path)
            (source / 'prompt.md').write_text('changed control')
            with self.assertRaises(ValueError):
                build_candidate(source, root / 'changed', spec_path)
            (source / 'prompt.md').write_bytes(b'before\nOLD\nafter\n')
            (source / 'extra.md').write_text('unexpected helper')
            with self.assertRaises(ValueError):
                build_candidate(source, root / 'extra', spec_path)


if __name__ == '__main__':
    unittest.main()
