"""Exercise fresh-clone and frozen-run notebook preparation without GPUs or network."""
import ast
import base64
import contextlib
import gzip
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
import zipfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


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
        # Compare against a direct build on this checkout: ZIP compression and
        # checkout line endings can differ between Python versions and platforms.
        archive = self.work / 'baseline.zip'
        self.command('build_submission.py', '--agent-dir', ROOT / 'configs/baseline',
                     '--output', archive)
        self.baseline_bytes = archive.read_bytes()
        self.baseline_sha = hashlib.sha256(self.baseline_bytes).hexdigest()

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
        payload = json.loads(gzip.decompress(base64.b85decode(encoded)))
        bundle = base64.b64decode(payload['bundle'])
        self.assertEqual(bundle, self.baseline_bytes)
        self.assertEqual(payload['bundle_sha256'], self.baseline_sha)
        # Independently verify every packaged member against its source bytes.
        root = ROOT / 'configs/baseline'
        expected = {p.relative_to(root).as_posix(): p.read_bytes()
                    for p in root.rglob('*') if p.is_file()}
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertEqual(len(archive.namelist()), len(expected))
            self.assertEqual({name: archive.read(name) for name in archive.namelist()}, expected)
        return notebook, payload

    def frozen(self):
        tasks = self.work / 'tasks.jsonl'
        tasks.write_text(json.dumps(self.task) + '\n')
        run = self.work / 'run'
        self.command('prepare_eval_run.py', '--agent-dir', ROOT / 'configs/baseline',
                     '--task-id', 'fixture_task', '--tasks', tasks, '--run-dir', run)
        return run

    def setup_cell(self, notebook, payload, task=None, flat_mounts=False):
        mounts = self.work / 'mounts'
        for name in ('data', 'wheels', 'model'):
            (mounts / name).mkdir(parents=True, exist_ok=True)
        (mounts / 'data/tasks.jsonl').write_text(json.dumps(task or self.task) + '\n')
        (mounts / 'wheels/fixture.whl').write_bytes(b'tiny fixture')
        payload = dict(payload, wheels={'fixture.whl': hashlib.sha256(b'tiny fixture').hexdigest()})
        encoded = base64.b85encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
        source = notebook['cells'][1]['source'].split('\n', 1)[1]
        substitutions = {
            '/kaggle/working/runtime-smoke': self.work / 'runtime',
            '/kaggle/input/competitions/gemma-4-developer-agent': mounts / 'data',
            '/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse': mounts / 'wheels',
            '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2': mounts / 'model',
        }
        if flat_mounts:
            substitutions['/kaggle/input/competitions/gemma-4-developer-agent'] = mounts / 'absent-data'
            substitutions['/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'] = mounts / 'absent-wheels'
            substitutions['/kaggle/input/gemma-4-developer-agent'] = mounts / 'data'
            substitutions['/kaggle/input/gemma-4-developer-agent-wheelhouse'] = mounts / 'wheels'
        for original, fixture in substitutions.items():
            source = source.replace(repr(original), repr(str(fixture)))
        self.setup_namespace = {'PAYLOAD_B85': encoded}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, '<setup-cell-fixture>', 'exec'), self.setup_namespace)
        return self.work / 'runtime'

    def test_fresh_clone_generates_pinned_private_notebook_and_same_agent(self):
        checkout = self.work / 'clean-checkout'
        # Copy only versioned preparation inputs: no data/, model/, runs/,
        # environments, tokens, or downloaded wheels from the maintainer's tree.
        paths = [*Path(ROOT / 'scripts').glob('*.py'), *Path(ROOT / 'configs/baseline').rglob('*'),
                 ROOT / 'resource/notebooks/official/kernel-metadata.json',
                 ROOT / 'resource/manifests/eval-wheelhouse-2026-10-09.json']
        for path in paths:
            if path.is_file():
                target = checkout / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        self.command('prepare_kaggle_runtime.py', '--owner', 'example-user', '--slug', 'team-first-smoke',
                     '--output', self.output, '--agent-dir', checkout / 'configs/baseline',
                     '--task-id', 'fixture_task', checkout=checkout)
        notebook, payload = self.notebook()
        self.assertLess(len(json.dumps(notebook).encode('utf-8')), 1_000_000)
        self.assertIsNone(payload['public_task'])
        self.assertEqual(hashlib.sha256(base64.b64decode(payload['bundle'])).hexdigest(), self.baseline_sha)
        for name, source in payload['scripts'].items():
            self.assertEqual(source, (ROOT / 'scripts' / name).read_text())
        self.assertEqual(len(payload['wheels']), 41)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(cell['source'])
        metadata = json.loads((self.output / 'kernel-metadata.json').read_text())
        self.assertEqual(metadata['id'], 'example-user/team-first-smoke')
        self.assertEqual(metadata['title'], 'team first smoke')
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
        self.assertEqual(manifest['agent_archive_sha256'], self.baseline_sha)
        self.assertTrue((runtime / 'inputs/agent/agent.yaml').is_file())
        with self.assertRaisesRegex(AssertionError, 'Runtime output already exists'):
            self.setup_cell(notebook, payload)
        self.assertEqual((runtime / 'inputs/tasks.jsonl').read_text(), public)

    def test_missing_task_fails_before_install_or_output_creation(self):
        notebook, payload = self.fresh()
        with self.assertRaisesRegex(AssertionError, 'exactly one'):
            self.setup_cell(notebook, payload, dict(self.task, instance_id='different_task'))
        self.assertFalse((self.work / 'runtime').exists())

    def test_flat_kaggle_mounts_preserve_frozen_inputs(self):
        run = self.frozen()
        self.generate('--run-dir', run)
        notebook, payload = self.notebook()
        runtime = self.setup_cell(notebook, payload, flat_mounts=True)
        self.command('verify_eval_run.py', '--run-dir', runtime)
        manifest = json.loads((runtime / 'run-manifest.json').read_text())
        self.assertEqual(manifest['input_paths']['data'], str(self.work / 'mounts/data'))
        self.assertEqual(manifest['input_paths']['wheels'], str(self.work / 'mounts/wheels'))

    def test_unknown_wheel_fails_before_installation(self):
        notebook, payload = self.fresh()
        wheels = self.work / 'mounts/wheels'
        wheels.mkdir(parents=True)
        extra = wheels / 'unchecked_extra.whl'
        extra.write_bytes(b'Not in the reviewed inventory')
        with self.assertRaisesRegex(AssertionError, 'wheelhouse inventory changed'):
            self.setup_cell(notebook, payload)
        shutil.rmtree(self.work / 'runtime')
        extra.unlink()
        self.setup_cell(notebook, payload)
        # Catch input changes between setup and installation, before pip or any deletion.
        extra.write_bytes(b'Added after setup')
        with patch('subprocess.run') as pip:
            with self.assertRaisesRegex(AssertionError, 'wheelhouse inventory changed'):
                exec(notebook['cells'][2]['source'], self.setup_namespace)
            pip.assert_not_called()

    def test_downloaded_kaggle_layout_passes_submission_dry_run(self):
        notebook, payload = self.fresh()
        runtime = self.setup_cell(notebook, payload)
        self.command('verify_eval_run.py', '--run-dir', runtime)
        # Tiny synthetic result records test the handoff, not model performance.
        records = {
            'compile/validation.json': {'status': 'compiled'},
            'verify/empty/result.json': {'task_id': 'fixture_task', 'resolved': False,
                                         'test_exit_code': 1, 'test_output': '1 failed'},
            'verify/gold/result.json': {'task_id': 'fixture_task', 'resolved': True, 'test_exit_code': 0},
            'verify/agent/result.json': {'task_id': 'fixture_task', 'resolved': True, 'test_exit_code': 0},
            'infer/prediction.json': {'task_id': 'fixture_task', 'agent_patch': 'fixture patch', 'llm_calls': 1},
        }
        for name, record in records.items():
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))
        result = self.command('submit_competition.py', '--run-dir', runtime)
        self.assertEqual(json.loads(result.stdout)['sha256'], self.baseline_sha)
        self.assertFalse((runtime / 'competition-submission.json').exists())
        task_path = runtime / 'inputs/tasks.jsonl'
        original = task_path.read_bytes()
        task_path.write_bytes(original.replace(b'Fix the public issue.', b'Changed task'))
        failed = self.command('submit_competition.py', '--run-dir', runtime, succeeds=False)
        self.assertIn('Frozen public task hash changed', failed.stderr)
        task_path.write_bytes(original)
        for name, record in records.items():
            if 'task_id' in record:
                (runtime / name).write_text(json.dumps(dict(record, task_id='other_task')))
        failed = self.command('submit_competition.py', '--run-dir', runtime, succeeds=False)
        self.assertIn('Smoke results differ from the prepared task', failed.stderr)

    def test_frozen_agent_and_zip_changes_are_rejected(self):
        run = self.frozen()
        agent = run / 'inputs/agent/agent.yaml'
        original = agent.read_bytes()
        agent.write_bytes(original + b'\n# Changed after preparation\n')
        failed = self.command('verify_eval_run.py', '--run-dir', run, succeeds=False)
        self.assertIn('Agent source changed', failed.stderr)
        agent.write_bytes(original)
        archive = run / 'inputs/submission.zip'
        archive.write_bytes(archive.read_bytes() + b'changed')
        failed = self.command('verify_eval_run.py', '--run-dir', run, succeeds=False)
        self.assertIn('Submission ZIP differs', failed.stderr)

    def test_preparation_revision_and_embedded_scripts_are_separate_from_source_run(self):
        run = self.frozen()
        path = run / 'run-manifest.json'
        manifest = json.loads(path.read_text())
        manifest['git_revision'] = 'older-source-run-revision'
        path.write_text(json.dumps(manifest))
        self.generate('--run-dir', run)
        _, payload = self.notebook()
        self.assertEqual(payload['source_run_git_revision'], 'older-source-run-revision')
        self.assertNotEqual(payload['git_revision'], payload['source_run_git_revision'])
        self.assertEqual(payload['script_sha256'],
                         {name: hashlib.sha256(source.encode()).hexdigest()
                          for name, source in payload['scripts'].items()})

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

    def test_large_inline_payload_fails_before_output_creation(self):
        agent = self.work / 'large-agent'
        shutil.copytree(ROOT / 'configs/baseline', agent)
        (agent / 'large.safetensors').write_bytes(os.urandom(850_000))
        failed = self.generate('--agent-dir', agent, '--task-id', 'fixture_task', succeeds=False)
        self.assertIn('Notebook source exceeds Kaggle 1 MB limit', failed.stderr)
        self.assertFalse(self.output.exists())

    def test_explicit_wheel_manifest_is_frozen_in_notebook(self):
        manifest = self.work / 'reviewed-wheels.json'
        digest = hashlib.sha256(b'tiny fixture').hexdigest()
        manifest.write_text(json.dumps({'wheelhouse_files': [{'path': 'fixture.whl', 'sha256': digest}]}))
        expected_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.generate('--agent-dir', ROOT / 'configs/baseline', '--task-id', 'fixture_task',
                      '--wheel-manifest', manifest)
        notebook, payload = self.notebook()
        self.assertEqual(payload['wheels'], {'fixture.whl': digest})
        self.assertEqual(payload['wheel_manifest_sha256'], expected_hash)
        runtime = self.setup_cell(notebook, payload)
        record = json.loads((runtime / 'run-manifest.json').read_text())
        self.assertEqual(record['wheel_manifest_sha256'], expected_hash)

    def test_source_archive_can_generate_without_git_installed(self):
        with patch.dict(os.environ, {'PATH': ''}):
            _, payload = self.fresh()
        self.assertIsNone(payload['git_revision'])
        self.assertEqual(payload['bundle_sha256'], self.baseline_sha)


if __name__ == '__main__':
    unittest.main()
