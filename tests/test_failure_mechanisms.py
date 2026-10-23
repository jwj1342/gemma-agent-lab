"""Exercise evidence boundaries with synthetic traces, never benchmark solutions."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_failure_mechanisms import run_audit
from src.failure_mechanisms import (apply_annotation, audit, changed_old_lines,
                                    statement_exposure, terminal_flags)


REF = {'pkg/a.py': [{'old_start': 10, 'old_len': 3}]}
PATCH = '--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -10,3 +10,3 @@\n context\n-old\n+new\n context\n'


def trace(*calls):
    return {'steps': [{'step_id': i + 1, 'message': 'Observed source; diagnosis remains uncertain.',
                       'tool_calls': [{'function_name': name, 'arguments': args}],
                       'observation': {'content': json.dumps(obs)}}
                      for i, (name, args, obs) in enumerate(calls)]}


def read(start=10, content='context\nold\ncontext\n'):
    return ('read_file', {'filepath': 'pkg/a.py', 'start_line': 10, 'end_line': 12},
            {'status': 'ok', 'filepath': 'pkg/a.py', 'start_line': start, 'end_line': 12,
             'content': content})


class FailureMechanismTests(unittest.TestCase):
    def test_header_like_docstring_lines_are_hunk_content(self):
        import difflib
        before = '"""\n-- old separator\n"""\n'
        after = '"""\n++ b/phantom.py\n"""\n'
        compile(before, 'before.py', 'exec')
        compile(after, 'after.py', 'exec')
        patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                            fromfile='a/pkg/a.py', tofile='b/pkg/a.py'))
        from src.failure_mechanisms import patch_paths, reference_hunks
        self.assertEqual(patch_paths(patch), ['pkg/a.py'])
        self.assertEqual(changed_old_lines(patch), {'pkg/a.py': {2}})
        self.assertEqual(reference_hunks(patch), {'pkg/a.py': [{'old_start': 1, 'old_len': 3}]})

    def test_invalid_patch_fields_remain_unknown_with_and_without_trace(self):
        from src.failure_mechanisms import load_run
        for prediction in ({}, {'agent_patch': None}, {'agent_patch': 12}, {'agent_patch': ''}):
            expected = [] if prediction.get('agent_patch') == '' else None
            self.assertEqual(audit(trace(read()), prediction, REF)['final_patch_paths'], expected)
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                task = root / 'run/tasks/task'
                task.mkdir(parents=True)
                (task / 'prediction.json').write_text(json.dumps(prediction))
                (root / 'run/summary.json').write_text(json.dumps({'tasks': [
                    {'task_id': 'task', 'controls_valid': True, 'resolved': False}]}))
                row = load_run(root, 'run', {'task': REF})[0]
                self.assertEqual(row['final_patch_paths'], expected)
                self.assertEqual(row['final_patch_reference'], False if expected == [] else None)

    def test_context_overlap_is_not_changed_statement_exposure(self):
        t = trace(read(content='context\n'))
        self.assertTrue(audit(t, {'agent_patch': ''}, REF)['read_reference'])
        self.assertEqual(statement_exposure(t, PATCH)['exposed_changed_old_lines'], 0)
        self.assertEqual(statement_exposure(trace(read()), PATCH)['exposed_changed_old_lines'], 1)

    def test_returned_range_truncation_and_numbered_shell(self):
        self.assertFalse(audit(trace(read(start=1, content='header\n')), {}, REF)['read_reference'])
        t = trace(('run_command', {'command': 'rg -n old pkg/a.py'},
                   {'status': 'ok', 'stdout': '11:old\n'}))
        self.assertEqual(statement_exposure(t, PATCH)['exposed_changed_old_lines'], 1)
        self.assertFalse(audit(t, {}, REF)['reference_exposures'][0]['full_hunk'])

    def test_rejected_edits_missing_predictions_and_shared_observations(self):
        t = trace(read(), ('edit_file', {'filepath': 'pkg/a.py'},
                          {'status': 'error', 'error_type': 'BudgetExceeded'}))
        result = audit(t, None, REF)
        self.assertTrue(result['attempted_reference'])
        self.assertFalse(result['successful_editor_reference'])
        self.assertIsNone(result['final_patch_reference'])
        self.assertIsNone(result['read_no_reference_patch'])
        result['outcome'] = {'error': 'Agent exceeded session timeout (8 min)'}
        flags = terminal_flags(result)
        self.assertTrue(flags['reported_timeout'])
        self.assertTrue(flags['budget_rejection'])
        t['steps'][1]['tool_calls'].append({'function_name': 'write_file', 'arguments': {'filepath': 'pkg/b.py'}})
        t['steps'][1]['observation']['content'] = '{"status":"ok"}'
        self.assertFalse(audit(t, {}, REF)['successful_editor_reference'])

    def test_shell_patch_partial_implementation_and_deleted_file(self):
        refs = {**REF, 'pkg/b.py': REF['pkg/a.py']}
        b = read()
        b[1]['filepath'] = b[2]['filepath'] = 'pkg/b.py'
        result = audit(trace(read(), b), {'agent_patch': PATCH}, refs)
        self.assertTrue(result['final_patch_reference'])
        self.assertFalse(result['successful_editor_reference'])
        self.assertEqual(result['exposed_paths_not_in_final_patch'], ['pkg/b.py'])
        deleted = audit(trace(read()), {'agent_patch': '--- a/pkg/a.py\n+++ /dev/null\n'}, REF)
        self.assertEqual(deleted['final_patch_paths'], ['pkg/a.py'])

    def test_new_code_has_no_old_statement_and_quoted_paths_are_rejected(self):
        self.assertEqual(changed_old_lines('--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n'), {})
        self.assertEqual(changed_old_lines('--- a/x.py\n+++ b/x.py\n@@ -1,0 +2 @@\n+x\n'), {})
        with self.assertRaises(ValueError):
            changed_old_lines('--- "a/quoted path.py"\n')

    def test_manual_labels_require_identity_exact_evidence_and_source_review(self):
        t = trace(read())
        row = {'trace_sha256': 'abc', 'prediction_sha256': 'def', 'final_patch_paths': []}
        annotation = {'category': 'recognized_not_applied', 'trace_sha256': 'abc',
                      'prediction_sha256': 'def', 'reviewer': 'human', 'rationale': 'Manual interpretation',
                      'evidence': [{'step_id': 1, 'field': 'message', 'quote': 'Observed source'}]}
        self.assertFalse(apply_annotation(row, None, t)['reviewed'])
        with self.assertRaises(ValueError):
            apply_annotation(row, annotation, t)
        annotation['source_supported'] = True
        annotation['evidence'].append({'step_id': 1, 'field': 'observation', 'quote': 'context'})
        self.assertTrue(apply_annotation(row, annotation, t)['reviewed'])
        for field, value in [('trace_sha256', 'stale'), ('prediction_sha256', 'stale'), ('reviewer', ''), ('evidence', []),
                             ('category', 'wrong_or_incomplete_patch')]:
            with self.assertRaises(ValueError):
                apply_annotation(row, {**annotation, field: value}, t)
        with self.assertRaises(ValueError):
            apply_annotation(row, {**annotation, 'evidence': [{'step_id': 1, 'field': 'message', 'quote': 'invented'}]}, t)

    def test_release_cohort_unknowns_denominators_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'reference-patch-locations.json').write_text(json.dumps({'task': REF, 'excluded': REF}))
            for name in ('seed0', 'seed1'):
                task_dir = root / name / 'tasks/task'
                task_dir.mkdir(parents=True)
                (root / name / 'cohort-summary.json').write_text(json.dumps({'tasks': [
                    {'task_id': 'task', 'controls_valid': True, 'resolved': False},
                    {'task_id': 'excluded', 'controls_valid': False, 'resolved': False}]}))
                if name == 'seed0':
                    (task_dir / 'trace.json').write_text(json.dumps(trace(read())))
            out = root / 'report'
            result = run_audit(root, ['seed0', 'seed1'], out)
            self.assertEqual(result['total_failed_attempts'], 2)
            self.assertEqual(result['unique_failed_tasks'], 1)
            self.assertEqual(result['summary']['seed1']['missing_trace'], 1)
            self.assertEqual(result['summary']['seed0']['empty_final_patch'], 0)
            self.assertEqual(result['summary']['seed0']['mechanisms'], {'insufficient_evidence': 1})
            with self.assertRaises(ValueError):
                run_audit(root, ['seed0'], out)
            for names in (['../escape'], ['seed0', 'seed0']):
                with self.assertRaises(ValueError):
                    run_audit(root, names, root / 'unused')

    def test_duplicate_tasks_and_mismatched_patch_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'reference-patch-locations.json').write_text(json.dumps({'task': REF}))
            run = root / 'seed0'
            run.mkdir()
            task = {'task_id': 'task', 'controls_valid': True, 'resolved': False}
            summary = run / 'cohort-summary.json'
            summary.write_text(json.dumps({'tasks': [task, task]}))
            with self.assertRaises(ValueError):
                run_audit(root, ['seed0'], root / 'duplicate')
            summary.write_text(json.dumps({'tasks': [task]}))
            tasks = root / 'tasks.jsonl'
            tasks.write_text(json.dumps({'instance_id': 'task', 'patch': PATCH.replace('-10,3', '-20,3')}) + '\n')
            with self.assertRaises(ValueError):
                run_audit(root, ['seed0'], root / 'mismatched', tasks=tasks)
            tasks.write_text(json.dumps({'instance_id': 'task', 'patch': PATCH}) + '\n')
            result = run_audit(root, ['seed0'], root / 'matching', tasks=tasks)
            self.assertEqual(result['summary']['seed0']['missing_trace'], 1)

    def test_prediction_survives_missing_trace_without_inventing_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'reference-patch-locations.json').write_text(json.dumps({'task': REF}))
            task_dir = root / 'seed0/tasks/task'
            task_dir.mkdir(parents=True)
            (root / 'seed0/cohort-summary.json').write_text(json.dumps({'tasks': [
                {'task_id': 'task', 'controls_valid': True, 'resolved': False}]}))
            (task_dir / 'prediction.json').write_text(json.dumps({'agent_patch': PATCH}))
            result = run_audit(root, ['seed0'], root / 'output')
            row = result['failures'][0]
            self.assertTrue(row['final_patch_reference'])
            self.assertNotIn('read_reference', row)
            self.assertEqual(row['mechanism']['category'], 'insufficient_evidence')


if __name__ == '__main__':
    unittest.main()
