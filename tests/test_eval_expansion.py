"""Keep exposed tasks, missing controls and agent failures distinct in expansion reports."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_eval_expansion import expansion


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        def row(name, resolved=False, valid=True):
            return {'task_id': name, 'controls_valid': valid, 'evaluated': valid,
                    'resolved': resolved if valid else None, 'completion_clean': valid}
        self.baseline = {'cohort': 'dev', 'complete': True, 'selected': 4, 'valid_controls': 3,
            'agent_sha256': 'baseline', 'benchmark_sha256': 'benchmark', 'model_sha256': 'model',
            'public_task_sha256': 'dev-tasks', 'verification_identity': 'verifier',
            'protocol': {'seed': 42}, 'budget_sha256': 'budget',
            'tasks': [row('exposed', True), row('gain'), row('loss', True), row('invalid', valid=False)]}
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate['agent_sha256'] = 'candidate'
        self.candidate['tasks'][1].update(resolved=True, completion_clean=False)
        self.candidate['tasks'][2].update(resolved=False, completion_clean=False, inference_error='timeout')
        self.fast = dict(self.baseline, cohort='fast', selected=1, valid_controls=1,
                         public_task_sha256='fast-tasks', tasks=[self.baseline['tasks'][0]])
        self.registration = {'task_count': 4, 'previously_exposed_fast_task_count': 1,
                             'additional_task_ids': ['gain', 'loss', 'invalid']}

    def report(self):
        return expansion(self.candidate, self.baseline, self.fast, self.registration)

    def test_separates_exposed_tasks_and_retains_failed_attempts(self):
        group = self.report()['groups']['additional_dev']
        self.assertEqual(group['candidate']['selected'], 3)
        self.assertEqual(group['candidate']['eligible'], 2)
        self.assertEqual(group['candidate']['resolution_rate'], 0.5)
        self.assertEqual(group['candidate']['clean_resolved'], 0)
        self.assertEqual(group['comparison']['newly_resolved'], ['gain'])
        self.assertEqual(group['comparison']['regressions'], ['loss'])
        self.assertNotIn('exposed', group['task_ids'])
        self.assertEqual(group['candidate']['tasks_missing_token_totals'], 2)

    def test_rejects_posthoc_partition_changes(self):
        for ids in [['gain', 'invalid'], ['gain', 'loss', 'exposed'], ['gain', 'gain', 'loss', 'invalid']]:
            with self.subTest(ids=ids):
                self.registration['additional_task_ids'] = ids
                with self.assertRaises(ValueError):
                    self.report()

    def test_rejects_changed_runtime_or_eligibility(self):
        for key, value in [('verification_identity', 'other'), ('budget_sha256', 'other')]:
            with self.subTest(key=key):
                saved = self.candidate[key]
                self.candidate[key] = value
                with self.assertRaises(ValueError):
                    self.report()
                self.candidate[key] = saved
        self.candidate['tasks'][1]['controls_valid'] = False
        self.candidate['valid_controls'] = 2
        with self.assertRaisesRegex(ValueError, 'masks differ'):
            self.report()

    def test_rejects_incomplete_or_duplicate_outcomes(self):
        self.candidate['tasks'][1]['evaluated'] = False
        with self.assertRaisesRegex(ValueError, 'completed outcomes'):
            self.report()
        self.candidate['tasks'][1]['evaluated'] = True
        self.candidate['tasks'][1]['task_id'] = 'exposed'
        with self.assertRaisesRegex(ValueError, 'duplicate tasks'):
            self.report()

    def test_reports_zero_eligible_additional_tasks_without_division(self):
        for report in (self.candidate, self.baseline):
            report['valid_controls'] = 1
            for row in report['tasks'][1:]:
                row.update(controls_valid=False, evaluated=False, resolved=None, completion_clean=False)
        additional = self.report()['groups']['additional_dev']
        self.assertIsNone(additional['comparison'])
        self.assertIsNone(additional['candidate']['resolution_rate'])


if __name__ == '__main__':
    unittest.main()
