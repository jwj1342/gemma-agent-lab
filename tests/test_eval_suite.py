"""Check cohort isolation, frozen inputs, cache invalidation, and paired score accounting."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_submission import sha256
from define_eval_splits import define
from eval_suite_lib import cache_record, digest, read_control_cache, save_artifact, write_json
from prepare_eval_suite import prepare
from prepare_eval_run import main as prepare_single
from run_eval_suite import indexed_hash
from summarize_eval_suite import compare, summarize, trace_metrics
from verify_eval_run import verify_inputs


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = [{'instance_id': f'fastapi_{i}', 'repo': 'fastapi/fastapi',
                       'base_commit': f'{i:040x}', 'problem_statement': 'Fix a bug.\u2028Keep text intact.',
                       'patch': 'SECRET_FIX', 'test_patch': 'SECRET_TEST'} for i in range(26)]
        self.path = self.root / 'tasks.jsonl'
        self.path.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in self.tasks))
        self.definition = define(self.tasks, [], set())
        self.definition['source_tasks_sha256'] = sha256(self.path)
        self.benchmark = self.root / 'benchmark.json'
        write_json(self.benchmark, self.definition)

    def frozen(self, name='run'):
        run = self.root / name
        prepare(run, ROOT / 'configs/baseline', self.path, self.benchmark, 'fast')
        return run

    def controls(self, task_id):
        empty = {'task_id': task_id, 'resolved': False, 'test_exit_code': 1,
                 'error_message': None, 'test_output': '1 failed, 2 passed'}
        gold = {'task_id': task_id, 'resolved': True, 'test_exit_code': 0, 'error_message': None}
        return empty, gold

    def test_group_siblings_and_exposure_never_cross_cohorts(self):
        sibling = dict(self.tasks[0], instance_id='fastapi_sibling')
        result = define([*self.tasks, sibling], [(self.tasks[1]['repo'] + '.' + self.tasks[1]['base_commit'][:35], 'train')], set())
        # A short prefix intentionally excludes a whole commit prefix family.
        excluded = {x['task_id'] for x in result['excluded']}
        self.assertIn('fastapi_1', excluded)
        clean = define([*self.tasks, sibling], [], set())
        for cohort in ['fast', 'dev', 'holdout']:
            self.assertEqual('fastapi_0' in clean['cohorts'][cohort], 'fastapi_sibling' in clean['cohorts'][cohort])
        self.assertTrue(set(clean['cohorts']['fast']) <= set(clean['cohorts']['dev']))
        self.assertFalse(set(clean['cohorts']['dev']) & set(clean['cohorts']['holdout']))
        self.assertEqual(clean, define(list(reversed([*self.tasks, sibling])), [], set()))

    def test_freeze_redacts_labels_preserves_unicode_and_detects_changes(self):
        run = self.frozen()
        manifest = verify_inputs(run)
        public = (run / 'inputs/tasks.jsonl').read_text()
        self.assertNotIn('SECRET_', public)
        self.assertIn('\u2028', public)
        self.assertEqual(len(manifest['task_ids']), 6)
        runtime = run / 'inputs/runtime/evaluate_agent.py'
        runtime.write_text(runtime.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'runtime changed'):
            verify_inputs(run)

    def test_holdout_requires_explicit_milestone_switch(self):
        with self.assertRaisesRegex(ValueError, 'allow-holdout'):
            prepare(self.root / 'sealed', ROOT / 'configs/baseline', self.path, self.benchmark, 'holdout')

    def test_single_task_freezes_offline_runtime_and_rejects_mutation(self):
        run = self.root / 'single'
        with patch.object(sys, 'argv', ['prepare_eval_run', '--run-dir', str(run),
             '--agent-dir', str(ROOT / 'configs/baseline'), '--tasks', str(self.path),
             '--task-id', 'fastapi_0']):
            prepare_single()
        manifest = verify_inputs(run)
        self.assertEqual(manifest['task_id'], 'fastapi_0')
        runtime = run / 'inputs/runtime/local_eval_setup.py'
        self.assertTrue(runtime.is_file())
        self.assertNotIn('SECRET_', (run / 'inputs/tasks.jsonl').read_text())
        runtime.write_text(runtime.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'runtime changed'):
            verify_inputs(run)

    def test_control_cache_rejects_changed_identity_and_collection_errors(self):
        empty, gold = self.controls('fastapi_1')
        key = digest({'task': 1, 'environment': 'v1'})
        path = self.root / 'control.json'
        write_json(path, cache_record(key, empty, gold))
        self.assertTrue(read_control_cache(path, key, 'fastapi_1')['valid'])
        with self.assertRaises(ValueError):
            read_control_cache(path, digest({'task': 1, 'environment': 'v2'}), 'fastapi_1')
        with self.assertRaises(ValueError):
            read_control_cache(path, key, 'fastapi_2')
        empty['test_output'] = '1 failed, 2 passed; ERROR during collection'
        self.assertFalse(cache_record(key, empty, gold)['body']['valid'])
        stored = json.loads(path.read_text())
        stored['body']['gold']['resolved'] = False
        write_json(path, stored)
        with self.assertRaises(ValueError):
            read_control_cache(path, key, 'fastapi_1')

    def test_content_index_detects_same_size_edit_even_with_restored_mtime(self):
        path = self.root / 'asset'
        path.write_bytes(b'first')
        index = {}
        old = indexed_hash(path, index)
        stat = path.stat()
        path.write_bytes(b'other')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(indexed_hash(path, index), old)

    def test_documented_new_api_requires_exact_failure_and_passing_gold(self):
        empty, gold = self.controls('fastapi_15745')
        empty.update(test_exit_code=2, test_output=(
            "E   ImportError: cannot import name '_IncludedRouter' from 'fastapi.routing' (/workspace/fastapi/routing.py)\n"
            "1 error during collection"))
        self.assertTrue(cache_record('key', empty, gold)['body']['valid'])
        gold['resolved'] = False
        self.assertFalse(cache_record('key', empty, gold)['body']['valid'])
        gold['resolved'] = True
        empty['test_output'] += '\nE   ModuleNotFoundError: missing_dependency'
        self.assertFalse(cache_record('key', empty, gold)['body']['valid'])
        empty['test_output'] = empty['test_output'].split('\nE   ModuleNotFoundError')[0].replace('_IncludedRouter', 'UnexpectedAPI')
        self.assertFalse(cache_record('key', empty, gold)['body']['valid'])

    def populate(self, run):
        m = verify_inputs(run)
        controls = {}
        write_json(run / 'runtime-inputs.json', {'model_sha256': 'model', 'verification_identity': 'env', 'budget_sha256': 'budget'})
        for j, task_id in enumerate(m['task_ids']):
            empty, gold = self.controls(task_id)
            if j == 0:
                gold['resolved'] = False
            controls[task_id] = cache_record('key', empty, gold)['body']
            if j > 0:
                save_artifact(run / 'infer' / task_id / 'prediction.json',
                              {'task_id': task_id, 'agent_patch': 'patch' if j < 5 else '',
                               'error': 'timeout' if j == 1 else None, 'duration_seconds': 2})
                save_artifact(run / 'verify/agent' / task_id / 'result.json',
                              {'task_id': task_id, 'resolved': j == 1, 'test_exit_code': 0 if j == 1 else 1})
        write_json(run / 'controls.json', controls)

    def test_fixed_denominator_keeps_agent_failures_and_reports_invalid_coverage(self):
        run = self.frozen()
        self.populate(run)
        report = summarize(run)
        self.assertTrue(report['complete'])
        self.assertEqual((report['selected'], report['valid_controls'], report['resolved']), (6, 5, 1))
        self.assertEqual(report['resolution_rate'], 0.2)
        self.assertFalse(report['tasks'][1]['completion_clean'])
        candidate = copy.deepcopy(report)
        candidate['tasks'][1]['resolved'] = False
        candidate['tasks'][2]['resolved'] = True
        difference = compare(candidate, report)
        self.assertEqual(len(difference['newly_resolved']), 1)
        self.assertEqual(len(difference['regressions']), 1)
        candidate['verification_identity'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'verification_identity'):
            compare(candidate, report)

    def test_missing_grade_is_incomplete_not_a_smaller_successful_benchmark(self):
        run = self.frozen()
        self.populate(run)
        task_id = verify_inputs(run)['task_ids'][-1]
        (run / 'verify/agent' / task_id / 'result.json').unlink()
        report = summarize(run)
        self.assertFalse(report['complete'])
        self.assertIsNone(report['resolution_rate'])

    def test_trace_counts_official_function_name(self):
        path = self.root / 'trace.json'
        write_json(path, {'steps': [{'tool_calls': [{'function_name': 'get_code_neighbors'}, {'function_name': 'read_file'}]}]})
        self.assertEqual(trace_metrics(path)['graph_calls'], 1)


if __name__ == '__main__':
    unittest.main()
