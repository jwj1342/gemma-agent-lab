"""Keep invocation errors distinct from intentional failing tests and patch content."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_tool_traces import audit_trace, patch_files


def step(name, result, **arguments):
    return {'tool_calls': [{'function_name': name, 'arguments': arguments}],
            'observation': {'content': result}}


class TraceAuditTests(unittest.TestCase):
    def test_failures_recovery_and_legitimate_nonzero_test(self):
        missing = {'error': 'mandatory input parameters are not present: old_string'}
        report = audit_trace({'steps': [step('edit_file', missing), {},
            step('edit_file', missing), step('run_command', {'status': 'ok', 'exit_code': 1}),
            step('edit_file', missing), step('edit_file', {'status': 'ok'}, filepath='tests/test_x.py'),
            step('get_code_neighbors', {'status': 'ok', 'neighbors': []}),
            step('load_skill_resource', {'status': 'ok'}), step('submit_patch', {'status': 'ok'})]})
        self.assertEqual(report['missing_argument_events'], 3)
        self.assertEqual(report['tool_error_events'], 3)
        self.assertEqual(report['longest_repeated_error_streak'], 2)
        self.assertEqual(report['nonzero_command_results'], 1)
        self.assertEqual(report['graph_calls'], 1)
        self.assertEqual(report['skill_calls'], 1)
        self.assertEqual(report['submission_calls'], 1)
        self.assertEqual(report['edit_targets'], {'tests/test_x.py': 1})

    def test_unknown_feedback_is_not_reported_as_clean(self):
        report = audit_trace({'steps': [step('edit_file', 'not structured JSON')]})
        self.assertEqual(report['unparsed_observations'], 1)
        self.assertEqual(report['tool_error_events'], 0)

    def test_official_command_error_keeps_nested_exit_code_and_message(self):
        result = {'status': 'error', 'error_type': 'CommandError',
                  'error_message': '1 failed, 82 passed', 'details': {'exit_code': 1}}
        report = audit_trace({'steps': [step('run_command', result)]})
        self.assertEqual(report['nonzero_command_results'], 1)
        self.assertEqual(report['missing_argument_events'], 0)
        self.assertEqual(report['error_frequencies'], {'1 failed, 82 passed': 1})

    def test_final_patch_paths_can_differ_from_workspace_edits(self):
        self.assertEqual(patch_files('diff --git a/src/x.py b/src/x.py\n'
                                    'diff --git "a/a b.py" "b/a b.py"\n'), ['src/x.py', 'a b.py'])


if __name__ == '__main__':
    unittest.main()
