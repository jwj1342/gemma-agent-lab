"""Check the base-weight identity used to accept or refuse checkpoint recovery."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.train_swe_lora import base_weights_digest, digest


class BaseWeightsDigestTests(unittest.TestCase):
    def test_single_file_keeps_the_existing_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'model.safetensors').write_bytes(b'weights')
            self.assertEqual(base_weights_digest(base), digest(base / 'model.safetensors'))

    def test_shard_and_index_changes_change_the_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            index = base / 'model.safetensors.index.json'
            first = base / 'model-00001-of-00002.safetensors'
            second = base / 'model-00002-of-00002.safetensors'
            first.write_bytes(b'first')
            second.write_bytes(b'second')
            index.write_text(json.dumps({'weight_map': {'b': second.name, 'a': first.name}}), encoding='utf-8')
            original = base_weights_digest(base)
            self.assertEqual(len(original), hashlib.sha256().digest_size * 2)
            second.write_bytes(b'changed')
            self.assertNotEqual(base_weights_digest(base), original)
            changed_shard = base_weights_digest(base)
            index.write_text(json.dumps({'weight_map': {'a': first.name, 'b': second.name},
                                         'metadata': {'total_size': 13}}), encoding='utf-8')
            self.assertNotEqual(base_weights_digest(base), changed_shard)

    def test_missing_or_ambiguous_layout_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaisesRegex(ValueError, 'model.safetensors.index.json'):
                base_weights_digest(base)
            index = base / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'a': 'model-00001-of-00002.safetensors'}}),
                             encoding='utf-8')
            with self.assertRaisesRegex(FileNotFoundError, 'model-00001-of-00002.safetensors'):
                base_weights_digest(base)
            (base / 'model.safetensors').write_bytes(b'weights')
            with self.assertRaisesRegex(ValueError, 'Ambiguous base weights'):
                base_weights_digest(base)

    def test_invalid_shard_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            index = base / 'model.safetensors.index.json'
            for name in ('../outside.safetensors', {'unexpected': 'mapping'}):
                index.write_text(json.dumps({'weight_map': {'a': name}}), encoding='utf-8')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'invalid filename'):
                    base_weights_digest(base)


if __name__ == '__main__':
    unittest.main()
