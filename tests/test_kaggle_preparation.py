"""Exercise fresh-clone and frozen-run notebook preparation without GPUs or network."""
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = '02bb939095ded0c970088fc9d19634e203e2d3119a36a5feff20bad55c9343fc'


class NotebookPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gemma-preparation-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.output = self.work / 'notebook'
        self.task = {'instance_id': 'fixture_task', 'repo': 'example/repo', 'base_commit': 'abc123',
                     'problem_statement': 'Fix the public issue.', 'hints_text': '',
                     'patch': 'PRIVATE_REFERENCE_PATCH', 'test_patch': 'PRIVATE_TEST_PATCH',
                     'FAIL_TO_PASS': ['PRIVATE_TEST'], 'PASS_TO_PASS': ['PRIVATE_PASS_TEST']}

    def command(self, script, *args, succeeds=True, checkout=ROOT):
        # A working directory outside the clone ensures paths are not accidentally
        # resolved through the maintainer's existing data or model installation.
        result = subprocess.run([sys.executable, str(checkout / 'scripts' / script), *map(str, args)],
                                cwd=self.work, capture_output=True, text=True)
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def generate(self, *args, succeeds=True):
        return self.command('prepare_kaggle_runtime.py', '--owner', 'example-user',
                            '--slug', 'team-first-smoke', '--output', self.output,
                            *args, succeeds=succeeds)

    def fresh(self):
        self.generate('--agent-dir', ROOT / 'configs/baseline', '--task-id', 'fixture_task')
        return self.notebook()

    def notebook(self):
        notebook = json.loads((self.output / 'runtime.ipynb').read_text())
        source = notebook['cells'][1]['source']
        encoded = ast.literal_eval(ast.parse(source).body[0].value)
        return notebook, json.loads(base64.b64decode(encoded))

    def frozen(self):
        tasks = self.work / 'tasks.jsonl'
        tasks.write_text(json.dumps(self.task) + '\n')
        run = self.work / 'run'
        self.command('prepare_eval_run.py', '--agent-dir', ROOT / 'configs/baseline',
                     '--task-id', 'fixture_task', '--tasks', tasks, '--run-dir', run)
        return run

    def setup_cell(self, notebook, payload, task=None):
        mounts = self.work / 'mounts'
        for name in ('data', 'wheels', 'model'):
            (mounts / name).mkdir(parents=True, exist_ok=True)
        (mounts / 'data/tasks.jsonl').write_text(json.dumps(task or self.task) + '\n')
        (mounts / 'wheels/fixture.txt').write_bytes(b'tiny fixture')
        payload = dict(payload, wheels={'fixture.txt': hashlib.sha256(b'tiny fixture').hexdigest()})
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        source = notebook['cells'][1]['source'].split('\n', 1)[1]
        substitutions = {
            '/kaggle/working/runtime-smoke': self.work / 'runtime',
            '/kaggle/input/competitions/gemma-4-developer-agent': mounts / 'data',
            '/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse': mounts / 'wheels',
            '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2': mounts / 'model',
        }
        for original, fixture in substitutions.items():
            source = source.replace(repr(original), repr(str(fixture)))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, '<setup-cell-fixture>', 'exec'), {'PAYLOAD_B64': encoded})
        return self.work / 'runtime'

    def test_fresh_clone_generates_pinned_private_notebook_and_same_agent(self):
        checkout = self.work / 'clean-checkout'
        # Copy only versioned preparation inputs: no data/, model/, runs/,
        # environments, tokens, or downloaded wheels from the maintainer's tree.
        paths = [*Path(ROOT / 'scripts').glob('*.py'), *Path(ROOT / 'configs/baseline').rglob('*'),
                 ROOT / 'resource/notebooks/official/kernel-metadata.json',
                 ROOT / 'resource/manifests/eval-assets-verification.json']
        for path in paths:
            if path.is_file():
                target = checkout / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        self.command('prepare_kaggle_runtime.py', '--owner', 'example-user', '--slug', 'team-first-smoke',
                     '--output', self.output, '--agent-dir', checkout / 'configs/baseline',
                     '--task-id', 'fixture_task', checkout=checkout)
        notebook, payload = self.notebook()
        self.assertIsNone(payload['public_task'])
        self.assertEqual(hashlib.sha256(base64.b64decode(payload['bundle'])).hexdigest(), BASELINE_SHA)
        for name, source in payload['scripts'].items():
            self.assertEqual(source, (ROOT / 'scripts' / name).read_text())
        self.assertEqual(len(payload['wheels']), 41)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(cell['source'])
        metadata = json.loads((self.output / 'kernel-metadata.json').read_text())
        self.assertEqual(metadata['id'], 'example-user/team-first-smoke')
        self.assertTrue(metadata['is_private'])
        self.assertTrue(metadata['enable_gpu'])
        self.assertFalse(metadata['enable_internet'])
        self.assertEqual(metadata['machine_shape'], 'NvidiaL4')
        official = json.loads((ROOT / 'resource/notebooks/official/kernel-metadata.json').read_text())
        self.assertEqual(metadata['docker_image'], official['dockerImage'])
        self.assertEqual(metadata['model_sources'], ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'])

    def test_mounted_task_is_sanitized_and_runtime_outputs_are_preserved(self):
        notebook, payload = self.fresh()
        runtime = self.setup_cell(notebook, payload)
        public = (runtime / 'inputs/tasks.jsonl').read_text()
        task = json.loads(public)
        self.assertEqual(task['problem_statement'], self.task['problem_statement'])
        self.assertTrue(set(task) <= set(payload['public_fields']))
        self.assertNotIn('PRIVATE_', public)
        manifest = json.loads((runtime / 'run-manifest.json').read_text())
        self.assertEqual(manifest['public_task_sha256'], hashlib.sha256(public.encode()).hexdigest())
        self.assertEqual(manifest['agent_archive_sha256'], BASELINE_SHA)
        self.assertTrue((runtime / 'inputs/agent/agent.yaml').is_file())
        with self.assertRaisesRegex(AssertionError, 'Runtime output already exists'):
            self.setup_cell(notebook, payload)
        self.assertEqual((runtime / 'inputs/tasks.jsonl').read_text(), public)

    def test_missing_task_fails_before_install_or_output_creation(self):
        notebook, payload = self.fresh()
        with self.assertRaisesRegex(AssertionError, 'exactly one'):
            self.setup_cell(notebook, payload, dict(self.task, instance_id='different_task'))
        self.assertFalse((self.work / 'runtime').exists())

    def test_frozen_run_still_works_and_mounted_task_must_match(self):
        run = self.frozen()
        self.generate('--run-dir', run)
        notebook, payload = self.notebook()
        self.assertNotIn('PRIVATE_', payload['public_task'])
        with self.assertRaisesRegex(AssertionError, 'Frozen task differs'):
            self.setup_cell(notebook, payload, dict(self.task, problem_statement='Changed input'))
        self.assertFalse((self.work / 'runtime').exists())
        self.setup_cell(notebook, payload)

    def test_frozen_task_hash_and_label_injection_are_rejected(self):
        run = self.frozen()
        task_path = run / 'inputs/tasks.jsonl'
        task_path.write_text(task_path.read_text() + ' ')
        failed = self.generate('--run-dir', run, succeeds=False)
        self.assertIn('Frozen public task hash changed', failed.stderr)
        self.assertFalse(self.output.exists())
        task_path.write_text(json.dumps(self.task) + '\n')
        manifest_path = run / 'run-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['public_task_sha256'] = hashlib.sha256(task_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        failed = self.generate('--run-dir', run, succeeds=False)
        self.assertIn('without validation labels', failed.stderr)
        self.assertFalse(self.output.exists())

    def test_existing_notebook_is_not_overwritten(self):
        self.fresh()
        original = {path.name: path.read_bytes() for path in self.output.iterdir()}
        failed = self.generate('--agent-dir', ROOT / 'configs/baseline', '--task-id', 'other', succeeds=False)
        self.assertIn('Output directory exists', failed.stderr)
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_source_archive_can_generate_without_git_installed(self):
        with patch.dict(os.environ, {'PATH': ''}):
            _, payload = self.fresh()
        self.assertIsNone(payload['git_revision'])
        self.assertEqual(payload['bundle_sha256'], BASELINE_SHA)


if __name__ == '__main__':
    unittest.main()
