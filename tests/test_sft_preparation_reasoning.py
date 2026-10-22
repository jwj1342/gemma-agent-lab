"""Check CLI selection policy and export receipts without downloading a shard."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_sft_data


class SFTPreparationReasoningTests(unittest.TestCase):
    def test_cli_records_opt_in_and_historical_policy(self):
        row = {'traj_id': 'fixture', 'instance_id': 'owner__repo.abcdef.mutation',
               'model': 'fixture-teacher', 'resolved': True,
               'messages': json.dumps([{'role': 'user', 'content': 'Repair'},
                   {'role': 'assistant', 'reasoning': 'Observed a source conflict.', 'tool_calls': [
                       {'id': 'c1', 'function': {'name': 'bash', 'arguments': {'command': 'git diff'}}}]},
                   {'role': 'tool', 'tool_call_id': 'c1', 'content': 'real observation'}])}
        parquet = ModuleType('pyarrow.parquet')
        parquet.read_table = lambda path: SimpleNamespace(to_pylist=lambda: [row])
        package = ModuleType('pyarrow')
        package.parquet = parquet
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard, tasks, benchmark = root / 'shard', root / 'tasks.jsonl', root / 'benchmark.json'
            shard.write_bytes(b'fixture')
            tasks.write_text('')
            benchmark.write_text(json.dumps({'cohorts': {'dev': [], 'holdout': []}}))
            for preserve in (False, True):
                output = root / str(preserve)
                args = ['prepare_sft_data', '--parquet', str(shard), '--output', str(output),
                        '--tasks', str(tasks), '--eval-benchmark', str(benchmark)]
                if preserve:
                    args.append('--preserve-reasoning')
                with patch.dict(sys.modules, {'pyarrow': package, 'pyarrow.parquet': parquet}), \
                     patch.object(prepare_sft_data, 'SHARD_SHA256', hashlib.sha256(b'fixture').hexdigest()), \
                     patch.object(sys, 'argv', args), patch('builtins.print'):
                    prepare_sft_data.main()
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['preserve_reasoning'], preserve)
                self.assertEqual(manifest['selected'][0]['assistant_messages_with_reasoning'], int(preserve))
                expected = 'explicit-text' if preserve else 'actions-only'
                self.assertEqual(manifest['reasoning_policy'], expected)
                records = [json.loads(line) for split in ('train', 'validation')
                           for line in (output / (split + '.jsonl')).read_text().splitlines()]
                self.assertEqual(records[0]['metadata']['reasoning_policy'], expected)
                self.assertEqual('reasoning' in records[0]['messages'][-1], preserve)


if __name__ == '__main__':
    unittest.main()
