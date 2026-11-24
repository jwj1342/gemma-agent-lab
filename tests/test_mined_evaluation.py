"""Exercise control acceptance, chronological evidence, and portable handoff preparation."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mined_control_checks import control_verdict, junit_counts
from collect_mined_sweep import from_log
from select_mined_cohort import valid_ids
import build_mined_eval_bundle as builder
from make_mined_verifier_notebook import CELL as VERIFIER_CELL


def passing_control():
    return {'instance_id': 'fixture', 'repo': 'example/repo', 'gold_rc': 0, 'empty_rc': 1,
            'gold_tests': {'tests': 2, 'failures': 0, 'errors': 0, 'skipped': 0},
            'empty_tests': {'tests': 1, 'failures': 1, 'errors': 0, 'skipped': 0}}


class MinedControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def log(self, name, text):
        path = self.work / name
        path.write_text(json.dumps([{'data': text}]), encoding='utf-8')
        return path

    def test_rejects_collection_timeout_no_tests_and_skips(self):
        self.assertTrue(control_verdict(passing_control())['valid'])
        for code in (2, 4, 5, 124, -1, None, 0):
            row = passing_control()
            row['empty_rc'] = code
            self.assertFalse(control_verdict(row)['valid'], code)
        for side in ('gold', 'empty'):
            for counts in (None, {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0},
                           {'tests': 1, 'failures': 0, 'errors': 1, 'skipped': 0},
                           {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 1}):
                row = passing_control()
                row[side + '_tests'] = counts
                self.assertFalse(control_verdict(row)['valid'])
        row = passing_control()
        row['gold_install_rc'] = 1
        self.assertFalse(control_verdict(row)['valid'])

    def test_junit_distinguishes_assertion_from_collection_error(self):
        path = self.work / 'junit.xml'
        self.assertIsNone(junit_counts(path))
        path.write_text('<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1"/></testsuites>')
        self.assertEqual(junit_counts(path), {'tests': 3, 'failures': 1, 'errors': 0, 'skipped': 1})
        path.write_text('<testsuite tests="1" failures="0" errors="1" skipped="0"/>')
        self.assertEqual(junit_counts(path)['errors'], 1)

    def test_latest_control_failure_replaces_older_success(self):
        good = passing_control()
        bad = copy.deepcopy(good)
        bad.update(gold_rc=2, valid=True)
        logs = [self.log('first.json', 'CONTROL_RESULT ' + json.dumps(good) + '\n'),
                self.log('second.json', 'CONTROL_RESULT ' + json.dumps(bad) + '\n')]
        self.assertEqual(valid_ids(logs, self.work / 'missing-tasks'), set())
        self.assertFalse(from_log(logs[1], self.work / 'missing-tasks')[0]['valid'])

    def test_legacy_screen_is_explicit_and_rejects_non_assertion_codes(self):
        path = self.log('legacy.json', '[1/2] good gold=0 empty=1 valid=True\n'
                        '[2/2] bad gold=0 empty=4 valid=True\n')
        self.assertEqual(valid_ids([path], self.work / 'missing'), set())
        self.assertEqual(valid_ids([path], self.work / 'missing', legacy_screen=True), {'good'})

    def test_sweep_is_self_contained_cpu_only_and_preserves_output(self):
        task = {'instance_id': 'fixture', 'repo': 'example/repo', 'base_commit': 'a' * 40,
                'patch': '', 'test_patch': '', 'FAIL_TO_PASS': [], 'paper_split': 'holdout'}
        tasks = self.work / 'tasks.jsonl'
        tasks.write_text(json.dumps(task) + '\n')
        out = self.work / 'notebook'
        command = [sys.executable, str(ROOT / 'scripts/make_mined_sweep_notebook.py'),
                   '--tasks', str(tasks), '--ids', 'fixture', '--out', str(out)]
        subprocess.run(command, capture_output=True, check=True)
        notebook = json.loads((out / 'sweep.ipynb').read_text())
        code = ''.join(notebook['cells'][1]['source'])
        compile(code, '<sweep>', 'exec')
        self.assertIn('def control_verdict', code)
        self.assertIn('CONTROL_RESULT', code)
        self.assertFalse(json.loads((out / 'kernel-metadata.json').read_text())['enable_gpu'])
        before = (out / 'sweep.ipynb').read_bytes()
        result = subprocess.run(command, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((out / 'sweep.ipynb').read_bytes(), before)

    def test_bundle_exports_redacted_inputs_and_reports_partial_failure(self):
        task = {'instance_id': 'fixture', 'repo': 'lark-parser/lark', 'base_commit': 'a' * 40,
                'problem_statement': 'A public issue.', 'patch': 'private gold',
                'test_patch': 'private tests', 'FAIL_TO_PASS': ['tests/test_one.py::test_one']}
        tasks = self.work / 'tasks.jsonl'
        tasks.write_text(json.dumps(task) + '\n')
        out = self.work / 'bundle'
        args = ['builder', '--tasks', str(tasks), '--ids', 'fixture', '--out', str(out)]
        with patch.object(sys, 'argv', args), patch.object(builder, 'fetch_snapshot', side_effect=OSError('offline')):
            self.assertEqual(builder.main(), 1)
        public = json.loads((out / 'tasks.public.jsonl').read_text())
        self.assertNotIn('patch', public)
        self.assertNotIn('test_patch', public)
        self.assertNotIn('FAIL_TO_PASS', public)
        self.assertEqual(json.loads((out / 'tasks.jsonl').read_text()), task)
        self.assertEqual(len(json.loads((out / 'bundle-manifest.json').read_text())['errors']), 1)

    def test_official_smoke_has_kernel_and_frozen_cpu_inputs(self):
        tasks = self.work / 'tasks.jsonl'
        tasks.write_text(json.dumps({'instance_id': 'fixture', 'repo': 'lark-parser/lark',
                                    'base_commit': 'a' * 40}) + '\n')
        snapshot = self.work / 'snapshot.json'
        snapshot.write_text(json.dumps({'snapshots': [{'dataset_version': 31, 'wheels': {'fixture.whl': 'b' * 64}}]}))
        out = self.work / 'verifier'
        subprocess.run([sys.executable, str(ROOT / 'scripts/make_mined_verifier_notebook.py'),
                        '--tasks', str(tasks), '--ids', 'fixture', '--snapshot', str(snapshot),
                        '--out', str(out)], check=True, capture_output=True)
        notebook = json.loads((out / 'verifier.ipynb').read_text())
        self.assertEqual(notebook['metadata']['kernelspec']['name'], 'python3')
        compile(''.join(notebook['cells'][0]['source']), '<verifier>', 'exec',
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        metadata = json.loads((out / 'kernel-metadata.json').read_text())
        self.assertFalse(metadata['enable_gpu'])
        self.assertEqual(metadata['model_sources'], [])
        self.assertIn('@sha256:', metadata['docker_image'])

    def test_offline_backend_is_installed_before_editable_metadata_is_built(self):
        workspace = self.work / 'workspace'
        workspace.mkdir()
        state = {'bootstrapped': False, 'editable': False}

        def run(command, **kwargs):
            if '-c' in command:
                self.assertTrue(state['editable'])
                return SimpleNamespace(returncode=0, stdout=str(workspace / '__init__.py'), stderr='')
            if '-e' in command:
                self.assertTrue(state['bootstrapped'], 'pip resolves editable metadata before installing other arguments')
                state['editable'] = True
            else:
                self.assertIn('hatchling', command)
                state['bootstrapped'] = True
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        source = VERIFIER_CELL.split('def offline_dependencies', 1)[1].split("assert version('swegemma')", 1)[0]
        namespace = {'Path': Path, 'sys': sys, 'subprocess': SimpleNamespace(run=run),
                     'wheels': self.work / 'wheels', 'ContainerSetupError': RuntimeError,
                     'SANDBOX_BOOTSTRAP': builder.SANDBOX_BOOTSTRAP, 'REQUIREMENTS': {}}
        exec('def offline_dependencies' + source, namespace)
        manager = SimpleNamespace(sandboxes={'sandbox': {'workspace': workspace, 'venv': self.work / 'venv'}})
        namespace['offline_dependencies'](manager, 'sandbox', repo='example/fixture')
        self.assertTrue(state['editable'])


if __name__ == '__main__':
    unittest.main()
