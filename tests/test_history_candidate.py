"""Protect the ablation's frozen control, configuration and package boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_history_candidate import build_candidate
from build_submission import sha256


class HistoryCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'control'
        (self.source / 'configs').mkdir(parents=True)
        (self.source / 'agent.yaml').write_text('name: test\ntools: [read_file, edit_file]\n')
        (self.source / 'configs/sampling.yaml').write_text(
            'max_output_tokens: 8192\nthinking_config:\n'
            '  thinking_budget: 4096\n  include_thoughts: true\n')
        self.expected = self.hashes()

    def hashes(self):
        return {p.relative_to(self.source).as_posix(): sha256(p)
                for p in self.source.rglob('*') if p.is_file()}

    def test_only_history_setting_changes_and_archives_are_repeatable(self):
        before = {p: (self.source / p).read_bytes() for p in self.expected}
        first = build_candidate(self.source, self.root / 'first', self.expected)
        second = build_candidate(self.source, self.root / 'second', self.expected)
        self.assertEqual(first['archive_sha256'], second['archive_sha256'])
        self.assertEqual(self.hashes(), self.expected)
        for path, original in before.items():
            actual = (self.root / 'first/agent' / path).read_bytes()
            self.assertEqual(actual, original.replace(b'include_thoughts: true',
                                                      b'include_thoughts: false'))
        with zipfile.ZipFile(self.root / 'first/submission.zip') as archive:
            self.assertEqual(set(archive.namelist()), set(self.expected))
            self.assertIsNone(archive.testzip())
        receipt = json.loads((self.root / 'first/ablation.json').read_text())
        self.assertEqual(receipt['control_files'], self.expected)
        with self.assertRaises(ValueError):
            build_candidate(self.source, self.root / 'first', self.expected)

    def test_rejects_changed_control_and_extra_helper_without_creating_run(self):
        for path in ('agent.yaml', 'fix.py'):
            target = self.source / path
            old = target.read_bytes() if target.exists() else None
            target.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'frozen'):
                build_candidate(self.source, self.root / 'bad', self.expected)
            self.assertFalse((self.root / 'bad').exists())
            if old is None:
                target.unlink()
            else:
                target.write_bytes(old)

    def test_rejects_nested_output_and_disabled_current_reasoning(self):
        with self.assertRaises(ValueError):
            build_candidate(self.source, self.source / 'nested', self.expected)
        sampling = self.source / 'configs/sampling.yaml'
        sampling.write_text(sampling.read_text().replace('4096', '0'))
        with self.assertRaisesRegex(ValueError, '4096'):
            build_candidate(self.source, self.root / 'bad', self.hashes())
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
