"""Version migration, deliberate exploration, and immutable install inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from wheelhouse_runtime import (CORE_PACKAGES, DATASET, WheelhouseMismatch, capture,
                               fingerprint, load_registry, recheck)


class WheelhouseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.wheels = Path(temporary.name)
        self.first = self.make_wheel('swegemma', '0.2.7')
        self.old = self.snapshot('old')

    def make_wheel(self, package, version):
        filename = f'{package.replace("-", "_")}-{version}-py3-none-any.whl'
        path = self.wheels / filename
        with zipfile.ZipFile(path, 'w') as wheel:
            wheel.writestr(f'{package}-{version}.dist-info/METADATA',
                           f'Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n')
        return path

    def snapshot(self, identity):
        return {'id': identity, 'dataset': DATASET, 'wheels': fingerprint(self.wheels),
                'versions': {}, 'compatibility': 'fixture only'}

    def test_auto_accepts_old_and_new_exact_snapshots(self):
        new_dir = self.wheels / 'other'
        new_dir.mkdir()
        self.first.rename(new_dir / self.first.name)
        self.make_wheel('swegemma', '0.2.10')
        new = self.snapshot('new')
        registry = [self.old, new]
        self.assertEqual(capture(self.wheels, registry)['matched_snapshot'], 'new')
        self.assertEqual(capture(new_dir, registry)['matched_snapshot'], 'old')
        with self.assertRaisesRegex(WheelhouseMismatch, 'missing.*swegemma-0.2.7'):
            capture(self.wheels, registry, requested='old')

    def test_identical_filename_with_changed_bytes_is_rejected(self):
        self.first.write_bytes(self.first.read_bytes() + b'changed')
        with self.assertRaisesRegex(WheelhouseMismatch, 'changed_sha256.*swegemma'):
            capture(self.wheels, [self.old])

    def test_unknown_inventory_reports_added_and_missing_files(self):
        self.first.unlink()
        self.make_wheel('swegemma', '0.2.10')
        with self.assertRaises(WheelhouseMismatch) as caught:
            capture(self.wheels, [self.old])
        self.assertIn('swegemma-0.2.10', str(caught.exception))
        self.assertIn('swegemma-0.2.7', str(caught.exception))
        self.assertIn('--wheelhouse-mode record', str(caught.exception))

    def test_record_accepts_unreviewed_inputs_and_captures_real_versions(self):
        for package in CORE_PACKAGES - {'swegemma'}:
            self.make_wheel(package, '1.0')
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter('always')
            record = capture(self.wheels, [self.old], mode='record')
        self.assertTrue(warning)
        self.assertEqual(record['status'], 'unreviewed_snapshot')
        self.assertIsNone(record['matched_snapshot'])
        self.assertEqual(record['versions']['swegemma'], '0.2.7')
        self.assertEqual(record['wheels'][self.first.name],
                         hashlib.sha256(self.first.read_bytes()).hexdigest())
        recheck(self.wheels, record['wheels'])
        self.first.write_bytes(self.first.read_bytes() + b'changed')
        with self.assertRaisesRegex(WheelhouseMismatch, 'changed_sha256'):
            recheck(self.wheels, record['wheels'])

    def test_recheck_rejects_added_or_missing_wheels(self):
        self.make_wheel('extra', '1.0')
        with self.assertRaisesRegex(WheelhouseMismatch, 'added'):
            recheck(self.wheels, self.old['wheels'])
        (self.wheels / 'extra-1.0-py3-none-any.whl').unlink()
        other = self.make_wheel('other', '1.0')
        both = fingerprint(self.wheels)
        other.unlink()
        with self.assertRaisesRegex(WheelhouseMismatch, 'missing'):
            recheck(self.wheels, both)

    def test_record_rejects_incomplete_or_invalid_wheels(self):
        self.make_wheel('extra', '1.0')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaisesRegex(ValueError, 'Required runtime packages missing'):
                capture(self.wheels, [self.old], mode='record')
            self.first.write_bytes(b'not a wheel')
            with self.assertRaises(zipfile.BadZipFile):
                capture(self.wheels, [self.old], mode='record')

    def test_record_cannot_masquerade_as_an_explicit_snapshot(self):
        with self.assertRaisesRegex(ValueError, 'explicit snapshot'):
            capture(self.wheels, [self.old], mode='record', requested='old')

    def test_registry_rejects_invalid_paths_hashes_and_duplicate_ids(self):
        path = self.wheels / 'registry.json'
        for files in ({'../outside.whl': 'a' * 64}, {'ok.whl': 'wrong'}):
            path.write_text(json.dumps({'schema_version': 1,
                                       'snapshots': [dict(self.old, wheels=files)]}))
            with self.assertRaises(ValueError):
                load_registry(path)
        path.write_text(json.dumps({'schema_version': 1, 'snapshots': [self.old, self.old]}))
        with self.assertRaisesRegex(ValueError, 'unique'):
            load_registry(path)


if __name__ == '__main__':
    unittest.main()
