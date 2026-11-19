"""Offline checks for the LoRA evaluation notebook and its collapse guard.

These tests must run on CPU with no GPU, no competition data, no base weights and no network. The
generator only reads the frozen agent tree, the training dataset, the evaluator source, a wheelhouse
snapshot and the official kernel metadata, so every branch that the 2026-10-24 round depends on is
exercised here before a kernel is pushed.

The guard's decision rule is a pure function (`resource/l4_feasibility/collapse_guard.py`) so it can be
exercised with in-memory loss trajectories; the generator is loaded by path for the same reason the
four-L4 tests do it, because ``scripts`` is not a package.
"""
import ast
import base64
import contextlib
import gzip
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD = load_module(ROOT / 'resource/l4_feasibility/collapse_guard.py', 'collapse_guard')
GENERATOR = load_module(ROOT / 'scripts/prepare_lora_eval_notebook.py', 'prepare_lora_eval_notebook')


def training_result(*losses: float, gradient: float = 1.0) -> dict:
    """A ``device-map-result.json`` shaped payload carrying one loss per optimizer update."""
    return {'updates': [{'update': number, 'loss': loss, 'gradient_norm': gradient}
                        for number, loss in enumerate(losses)]}


class CollapseGuardRuleTests(unittest.TestCase):
    """The two measured failure modes must be recognised, and a moving arm must not be stopped."""

    def test_measured_flat_arm_is_flat(self):
        record = GUARD.collapse_guard(training_result(2.0966, 2.0977, 2.0989))
        self.assertEqual(record['verdict'], GUARD.FLAT)
        self.assertEqual(record['reason'], 'loss_did_not_move')
        self.assertAlmostEqual(record['relative_movement'], 0.0011, places=4)

    def test_measured_collapsed_arm_is_collapsed(self):
        record = GUARD.collapse_guard(training_result(2.0966, 1.5204, 0.7323))
        self.assertEqual(record['verdict'], GUARD.COLLAPSED)
        self.assertEqual(record['reason'], 'min_loss_below_floor')
        self.assertAlmostEqual(record['min_loss'], 0.7323)

    def test_arm_inside_the_window_proceeds(self):
        # Between the measured points: the loss moved far more than the flat ratio and stayed above
        # the collapse floor, which is the arm the next refresh is looking for.
        record = GUARD.collapse_guard(training_result(2.0966, 1.8210, 1.6402))
        self.assertEqual(record['verdict'], GUARD.PROCEED)
        self.assertAlmostEqual(record['min_loss'], 1.6402)

    def test_one_update_cannot_be_judged(self):
        record = GUARD.collapse_guard(training_result(2.0966))
        self.assertEqual(record['verdict'], GUARD.INVALID)
        self.assertEqual(record['reason'], 'fewer_than_two_updates')

    def test_no_updates_cannot_be_judged(self):
        record = GUARD.collapse_guard({'updates': []})
        self.assertEqual(record['verdict'], GUARD.INVALID)
        self.assertEqual(record['updates'], 0)
        self.assertIsNone(record['last_loss'])

    def test_thresholds_can_be_disabled(self):
        flat = GUARD.collapse_guard(training_result(2.0966, 2.0989), flat_ratio=0)
        self.assertEqual(flat['verdict'], GUARD.PROCEED)
        collapsed = GUARD.collapse_guard(training_result(2.0966, 0.7323), loss_floor=0)
        self.assertEqual(collapsed['verdict'], GUARD.PROCEED)

    def test_record_keeps_the_whole_trajectory(self):
        record = GUARD.collapse_guard(training_result(2.0966, 1.5, 1.2, gradient=28.7))
        self.assertEqual(record['losses'], [2.0966, 1.5, 1.2])
        self.assertEqual(record['first_loss'], 2.0966)
        self.assertEqual(record['last_loss'], 1.2)
        self.assertEqual(record['updates'], 3)
        self.assertEqual(record['max_gradient_norm'], 28.7)
        self.assertEqual(record['loss_floor'], 1.0)
        self.assertEqual(record['flat_ratio'], 0.005)

    def test_unreadable_rows_stop_the_arm(self):
        result = {'updates': [{'loss': 2.0, 'gradient_norm': 1.0}, {'loss': None}, 'not-a-row',
                              {'loss': 1.4, 'gradient_norm': None}]}
        record = GUARD.collapse_guard(result)
        self.assertEqual(record['updates'], 2)
        self.assertEqual(record['verdict'], GUARD.INVALID)

    def test_nonfinite_metrics_and_training_errors_stop_the_arm(self):
        for value in (float('nan'), float('inf'), -1.0, 'broken'):
            with self.subTest(loss=value):
                self.assertEqual(GUARD.collapse_guard(training_result(2.1, value))['verdict'],
                                 GUARD.INVALID)
        record = training_result(2.1, 1.4)
        record['training_error'] = 'optimizer failed'
        self.assertEqual(GUARD.collapse_guard(record)['verdict'], GUARD.INVALID)
        self.assertEqual(GUARD.collapse_guard(training_result(
            2.1, 1.4, gradient=float('nan')))['verdict'], GUARD.INVALID)

    def test_intermediate_upward_excursion_is_not_flat(self):
        self.assertEqual(GUARD.collapse_guard(training_result(2.1, 3.0, 2.1))['verdict'],
                         GUARD.PROCEED)

    def test_invalid_thresholds_are_refused(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GUARD.collapse_guard(training_result(2.1, 1.4), loss_floor=value)


class NotebookBuildTests(unittest.TestCase):
    """Build a notebook from fixtures and read back what the kernel will execute."""

    def setUp(self):
        # Not tempfile.TemporaryDirectory: ``mkdir(mode=0o700)`` leaves a directory that a restricted
        # Windows sandbox denies to the process running these checks, while ``Path.mkdir`` inherits the
        # parent's permissions on both platforms.
        self.work = Path(tempfile.gettempdir()) / f'gemma-lora-eval-{uuid.uuid4().hex}'
        self.work.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.work, True)
        self.agent = self.work / 'agent'
        (self.agent / 'prompts').mkdir(parents=True)
        (self.agent / 'agent.yaml').write_text('model: main\n', encoding='utf-8')
        (self.agent / 'prompts/system.md').write_text('You are a fixture agent.\n', encoding='utf-8')
        self.dataset = self.work / 'dataset.jsonl'
        self.dataset.write_text('{"prompt": "p", "completion": "c"}\n', encoding='utf-8')
        self.evaluator = self.work / 'evaluate_agent.py'
        self.evaluator.write_text('print("fixture")\n', encoding='utf-8')
        self.snapshot = self.work / 'snapshots.json'
        self.snapshot.write_text(json.dumps({'snapshots': [{'dataset_version': 'fixture-v1',
                                                            'wheels': {'a.whl': 'deadbeef'}}]}),
                                 encoding='utf-8')

    def build(self, name: str = 'notebook', flags: tuple = (), **overrides) -> Path:
        output = self.work / name
        arguments = {
            '--owner': 'fixture', '--slug': name, '--title': 'fixture build',
            '--agent-dir': str(self.agent), '--task-ids': ('fastapi_1', 'rich_2'),
            '--evaluate-agent': str(self.evaluator), '--snapshot': str(self.snapshot),
            '--output': str(output),
        }
        arguments.update(overrides)
        argv = ['prepare_lora_eval_notebook.py']
        for key, value in arguments.items():
            if isinstance(value, tuple):
                argv.extend([key, *value])
            else:
                argv.extend([key, str(value)])
        argv.extend(flags)
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sys, 'argv', argv):
                GENERATOR.main()
        return output

    @staticmethod
    def payload(directory: Path) -> dict:
        notebook = json.loads((directory / 'evaluation.ipynb').read_text(encoding='utf-8'))
        source = ''.join(notebook['cells'][1]['source'])
        literal = source.split('PAYLOAD_B85 = ', 1)[1]
        literal = literal[:literal.index('\n')]
        return json.loads(gzip.decompress(base64.b85decode(eval(literal))))

    @staticmethod
    def setup_cell(directory: Path) -> str:
        notebook = json.loads((directory / 'evaluation.ipynb').read_text(encoding='utf-8'))
        return ''.join(notebook['cells'][2]['source'])

    def test_trained_notebook_carries_the_guard(self):
        directory = self.build('trained', **{'--train-dataset': str(self.dataset),
                                             '--learning-rate': '5e-5', '--max-updates': '8'})
        payload = self.payload(directory)
        self.assertEqual(payload['training']['learning_rate'], 5e-05)
        self.assertEqual(payload['training']['max_updates'], 8)
        self.assertEqual(payload['training']['collapse_loss_floor'], 1.0)
        self.assertEqual(payload['training']['collapse_flat_ratio'], 0.005)
        self.assertIn('collapse_guard.py', payload['training']['workers'])
        setup = self.setup_cell(directory)
        self.assertIn('from collapse_guard import collapse_guard', setup)
        self.assertIn("if GUARD['verdict'] == 'proceed':", setup)
        self.assertIn("'skipped': GUARD['verdict']", setup)
        self.assertIn('GUARD = collapse_guard(trained_result', setup)
        for index, cell in enumerate(json.loads((directory / 'evaluation.ipynb').read_text(encoding='utf-8'))['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))

    def test_baseline_notebook_has_no_guard_parameters(self):
        directory = self.build('baseline')
        payload = self.payload(directory)
        self.assertIsNone(payload['training'])
        setup = self.setup_cell(directory)
        self.assertIn("GUARD = {'verdict': 'proceed', 'reason': 'no_training'}", setup)
        ast.parse(setup)

    def test_guard_flags_reach_the_payload(self):
        directory = self.build('custom', **{'--train-dataset': str(self.dataset),
                                            '--collapse-loss-floor': '0.9',
                                            '--collapse-flat-ratio': '0.01'})
        payload = self.payload(directory)
        self.assertEqual(payload['training']['collapse_loss_floor'], 0.9)
        self.assertEqual(payload['training']['collapse_flat_ratio'], 0.01)

    def test_site_packages_reaches_the_hook(self):
        directory = self.build('site', flags=('--site-packages',))
        hook = self.payload(directory)['scripts']['local_eval_setup.py']
        self.assertIn('SYSTEM_SITE_PACKAGES = True', hook)

    def test_existing_output_directory_is_refused(self):
        self.build('once')
        with self.assertRaises(SystemExit):
            self.build('once')

    def test_oversize_notebook_is_rejected_before_output_is_created(self):
        with patch.object(GENERATOR, 'build_notebook', return_value=({'text': 'x' * 1_000_001}, {})):
            with self.assertRaises(SystemExit):
                self.build('oversize')
        self.assertFalse((self.work / 'oversize').exists())

    def test_skip_branch_executes_without_evaluation(self):
        source = self.setup_cell(self.build('skip'))
        branch = ast.parse(source).body
        branch = next(node for node in branch if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "GUARD['verdict'] == 'proceed'")
        for verdict in (GUARD.FLAT, GUARD.COLLAPSED, GUARD.INVALID):
            with self.subTest(verdict=verdict):
                def unexpected():
                    self.fail('a stopped arm ran inference')
                namespace = {'GUARD': {'verdict': verdict}, 'WORK': self.work,
                             'PAYLOAD': {'task_ids': ['rich_1']}, 'json': json,
                             'evaluate_arm': unexpected}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.Module(body=[branch], type_ignores=[]), '<guard>', 'exec'), namespace)
                summary = json.loads((self.work / 'summary.json').read_text())
                self.assertEqual(summary['measured_tasks'], 0)
                self.assertIsNone(summary['resolved'])
                self.assertEqual(summary['skipped'], verdict)

    def test_invalid_controls_stop_before_inference(self):
        source = self.setup_cell(self.build('controls'))
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_arm')
        stage = self.work / 'stage'
        stage.mkdir()
        frozen = stage / 'frozen'
        (frozen / 'inputs').mkdir(parents=True)
        (frozen / 'inputs/tasks.jsonl').write_text('{}\n')
        (frozen / 'inputs/submission.zip').write_bytes(b'zip fixture')
        import hashlib
        import subprocess
        import time
        (frozen / 'run-manifest.json').write_text(json.dumps({
            'public_task_sha256': hashlib.sha256((frozen / 'inputs/tasks.jsonl').read_bytes()).hexdigest(),
            'agent_archive_sha256': hashlib.sha256(b'zip fixture').hexdigest()}))
        (self.work / 'scripts').mkdir()
        shutil.copyfile(ROOT / 'scripts/summarize_eval_smoke.py',
                        self.work / 'scripts/summarize_eval_smoke.py')
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            self.assertNotIn('infer', command)
            return subprocess.CompletedProcess(command, 0)
        namespace = {'PAYLOAD': {'task_ids': ['rich_1']}, 'STAGE': stage,
                     'WORK': self.work, 'AGENT': self.agent, 'DATA': self.work, 'ENV': {},
                     'sys': sys, 'json': json, 'hashlib': hashlib, 'time': time,
                     'subprocess': subprocess}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<evaluation>', 'exec'), namespace)
        with patch.object(subprocess, 'run', side_effect=run), patch.object(sys, 'path', sys.path[:]):
            with contextlib.redirect_stdout(io.StringIO()):
                result = namespace['evaluate_arm']()
        self.assertEqual(result['skipped'], 'invalid_controls')
        self.assertEqual(result['measured_tasks'], 0)
        self.assertEqual(len(calls), 3)


    def test_per_task_controls_measure_only_the_valid_subset(self):
        directory = self.build('per-task', flags=('--per-task-controls',))
        source = self.setup_cell(directory)
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_arm')
        stage = self.work / 'stage'
        stage.mkdir()
        frozen = stage / 'frozen'
        (frozen / 'inputs').mkdir(parents=True)
        (frozen / 'inputs/tasks.jsonl').write_text('{}\n')
        (frozen / 'inputs/submission.zip').write_bytes(b'zip fixture')
        import hashlib
        import subprocess
        import time
        (frozen / 'run-manifest.json').write_text(json.dumps({
            'public_task_sha256': hashlib.sha256((frozen / 'inputs/tasks.jsonl').read_bytes()).hexdigest(),
            'agent_archive_sha256': hashlib.sha256(b'zip fixture').hexdigest()}))
        (self.work / 'scripts').mkdir()
        shutil.copyfile(ROOT / 'scripts/summarize_eval_smoke.py',
                        self.work / 'scripts/summarize_eval_smoke.py')
        empty_ok = {'task_id': 'fastapi_1', 'resolved': False, 'test_exit_code': 1,
                    'error_message': None, 'test_output': '1 failed'}
        gold_ok = {'task_id': 'fastapi_1', 'resolved': True, 'test_exit_code': 0, 'error_message': None}
        empty_bad = {'task_id': 'rich_2', 'resolved': True, 'test_exit_code': 0, 'error_message': None}
        gold_bad = {'task_id': 'rich_2', 'resolved': False, 'test_exit_code': 1, 'error_message': None}
        for source_name, payload in (('empty', empty_ok), ('gold', gold_ok)):
            target = stage / 'controls' / source_name / 'fastapi_1'
            target.mkdir(parents=True)
            (target / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
        for source_name, payload in (('empty', empty_bad), ('gold', gold_bad)):
            target = stage / 'controls' / source_name / 'rich_2'
            target.mkdir(parents=True)
            (target / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)
        namespace = {'PAYLOAD': {'task_ids': ['fastapi_1', 'rich_2']}, 'STAGE': stage,
                     'WORK': self.work, 'AGENT': self.agent, 'DATA': self.work, 'ENV': {},
                     'sys': sys, 'json': json, 'hashlib': hashlib, 'time': time,
                     'subprocess': subprocess, 'MODEL': self.work / 'model',
                     'STARTED': time.monotonic()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<evaluation>', 'exec'), namespace)
        with patch.object(subprocess, 'run', side_effect=run), patch.object(sys, 'path', sys.path[:]):
            with contextlib.redirect_stdout(io.StringIO()):
                result = namespace['evaluate_arm']()
        infer = next(command for command in calls if 'infer' in command)
        task_ids = infer[infer.index('--task-ids') + 1:infer.index('--data-dir')]
        self.assertEqual(task_ids, ['fastapi_1'])
        self.assertEqual(result['task_ids'], ['fastapi_1'])
        self.assertEqual(result['excluded_invalid_controls'], ['rich_2'])
        self.assertEqual(result['measured_tasks'], 1)
        self.assertFalse(result['controls_valid'])


if __name__ == '__main__':
    unittest.main()
