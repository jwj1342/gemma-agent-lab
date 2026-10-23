"""Test outcome-based selection and evidence identity boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_matched_traces import ARMS, checked_reading, compare, select_tasks


class MatchedTraceTests(unittest.TestCase):
    def rows(self):
        return [{'task_id': task, 'run': arm, 'trace_present': True,
                 'prediction_present': True,
                 'outcome': {'controls_valid': True, 'resolved': arm != 'c2'}}
                for task in ('z', 'a', 'b') for arm in ARMS]

    def test_selection_is_sorted_and_requires_complete_valid_discordance(self):
        rows = self.rows()
        self.assertEqual(select_tasks(rows, 2), (['a', 'b'], ['a', 'b', 'z']))
        rows[0]['trace_present'] = False
        rows[6]['outcome']['controls_valid'] = False
        rows[12]['outcome']['resolved'] = False  # c2-seed1 matches c2 on b
        self.assertEqual(select_tasks(rows), ([], []))
        with self.assertRaises(ValueError):
            select_tasks(self.rows() + [self.rows()[0]])

    def test_readings_require_real_quotes_and_post_edit_verification(self):
        ev = [{'call': 1, 'step_id': 3, 'tool': 'run_command',
               'thinking': 'Copy the list.', 'obs': {'exit_code': 0}},
              {'call': 4, 'step_id': 6, 'tool': 'run_command',
               'thinking': 'Check.', 'obs': {'exit_code': 1}}]
        reading = {'hypothesis': {'step_id': 3, 'quote': 'Copy the list.', 'rationale': 'Observed aliasing'},
                   'verification': {'step_id': 6, 'quote': '"exit_code": 1', 'rationale': 'Failed check'}}
        result = checked_reading(reading, ev, 2)
        self.assertEqual(result['hypothesis']['call'], 1)
        self.assertEqual(result['verification']['call'], 4)
        with self.assertRaises(ValueError):
            checked_reading(reading, ev, None)
        bad = copy.deepcopy(reading)
        bad['hypothesis']['quote'] = 'Unobserved claim'
        with self.assertRaises(ValueError):
            checked_reading(bad, ev, 2)
        self.assertEqual(checked_reading({}, ev, None), {'hypothesis': None, 'verification': None, 'source_write': None, 'uncertainty_notes': None})
        command_event = {'call': 2, 'step_id': 4, 'tool': 'run_command',
                         'args': {'command': "Path('pkg/a.py').write_text('new')"},
                         'thinking': '', 'obs': {'status': 'ok', 'exit_code': 0}}
        shell_reading = copy.deepcopy(reading)
        shell_reading['source_write'] = {'step_id': 4, 'quote': "write_text('new')",
                                         'rationale': 'Manually confirmed source write'}
        self.assertEqual(checked_reading(shell_reading, ev + [command_event], None)
                         ['source_write']['call'], 2)
        command_event['obs']['exit_code'] = 1
        with self.assertRaises(ValueError):
            checked_reading(shell_reading, ev + [command_event], None)


    def test_comparison_exports_successes_failures_and_unknown_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {'task': {'pkg/a.py': [{'old_start': 1, 'old_len': 1}]}}
            (root / 'reference-patch-locations.json').write_text(json.dumps(reference))
            tasks = root / 'tasks.jsonl'
            tasks.write_text(json.dumps({'instance_id': 'task', 'patch':
                '--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-old\n+new\n'}) + '\n')
            for arm in ARMS:
                task = root / arm / 'tasks/task'
                task.mkdir(parents=True)
                (root / arm / 'cohort-summary.json').write_text(json.dumps({'tasks': [
                    {'task_id': 'task', 'controls_valid': True, 'resolved': arm != 'c2'}]}))
                (task / 'prediction.json').write_text(json.dumps({'agent_patch': ''}))
                (task / 'trace.json').write_text(json.dumps({'steps': [
                    {'step_id': 3, 'tool_calls': [{'function_name': 'read_file',
                     'arguments': {'filepath': 'pkg/a.py'}}], 'observation': {'content': json.dumps(
                     {'status': 'ok', 'filepath': 'pkg/a.py', 'start_line': 1, 'content': 'old'})}}]}))
            result = compare(root, tasks, root / 'output')
            self.assertEqual(result['selected_tasks'], ['task'])
            self.assertEqual(len(result['attempts']), 5)
            self.assertEqual(sum(r['resolved'] for r in result['attempts']), 4)
            for row in result['attempts']:
                self.assertEqual(row['first_positional_old_line_exposure'], {'call': 1, 'step_id': 3})
                self.assertIsNone(row['first_accepted_editor_call'])
                self.assertIsNone(row['disagreements']['hypothesis'])


if __name__ == '__main__':
    unittest.main()
