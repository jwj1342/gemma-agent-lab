"""Offline checks for the public-task control-sweep notebook generator.

The generator is the instrument the measurement set is grown with, so two properties are pinned here
without a GPU or a network: the reviewed test-only wheels are embedded from a hash-verified manifest,
and the reviewed harness wheels the organisers replaced in place can be restored from the private
publisher kernel mount.

The inputs are synthesized rather than taken from `runs/`, which is gitignored and absent in CI: a
minimal agent directory, a test-dependency directory whose dummy wheels match their own manifest, and
the tracked reviewed wheel manifest the generator already defaults to.
"""
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / 'scripts/prepare_task_control_sweep.py'
PAYLOAD = ROOT / 'scripts/test_dependency_payload.py'
REVIEWED_MANIFEST = ROOT / 'resource/manifests/eval-wheelhouse-2026-10-09.json'
# Fetched once with `kernels_output` from the private publisher kernel; gitignored, so the test that
# needs the real reviewed bytes skips where they are absent.
PINNED_WHEELS_FETCHED = ROOT / 'runs/2026-10-13-pinned-wheels/harness-wheels-pinned'


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ControlSweepNotebookTests(unittest.TestCase):
    def setUp(self):
        # The generator imports `test_dependency_payload` as a sibling, which is how it runs as a
        # script; a test that loads it by path has to provide the same sys.path entry.
        scripts = str(ROOT / 'scripts')
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
            self.addCleanup(sys.path.remove, scripts)
        self.module = load_module(GENERATOR, 'control_sweep_generator')
        self.payload = load_module(PAYLOAD, 'control_sweep_payload')
        self.root = Path(tempfile.mkdtemp(prefix='control-sweep-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.agent = self.build_agent()
        self.deps = self.build_deps()

    def build_agent(self) -> Path:
        agent = self.root / 'agent'
        agent.mkdir()
        (agent / 'eval_config.yaml').write_text(
            'evaluation:\n  timeout_seconds: 60\n  max_tool_calls: 40\n  max_time_minutes: 4\n'
            '  max_turns: 80\n', encoding='utf-8')
        return agent

    def build_deps(self) -> Path:
        """A dependency directory whose dummy wheels match the hashes in its own manifest."""
        deps = self.root / 'deps'
        deps.mkdir()
        wheels = []
        for package in self.payload.TEST_DEPENDENCY_PACKAGES:
            name = f'{package}-1.0-py3-none-any.whl'
            data = f'dummy wheel for {package}'.encode('utf-8')
            (deps / name).write_bytes(data)
            wheels.append({'path': name, 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()})
        (deps / 'test-deps-manifest.json').write_text(
            json.dumps({'target': {'python_version': '3.13'}, 'wheels': wheels}), encoding='utf-8')
        return deps

    def generate(self, **overrides) -> tuple:
        output = Path(tempfile.mkdtemp(prefix='out-', dir=self.root))
        arguments = {'agent_dir': self.agent, 'task_ids': ['rich_3454'], 'test_deps_dir': self.deps,
                     'slug': 'someone/control-sweep-slug', **overrides}
        argv = ['prepare_task_control_sweep.py',
                '--agent-dir', str(arguments.pop('agent_dir')),
                '--task-ids', *arguments.pop('task_ids'),
                '--test-deps-dir', str(arguments.pop('test_deps_dir')),
                '--slug', arguments.pop('slug'),
                '--output', str(output)]
        for key, value in arguments.items():
            option = f'--{key.replace("_", "-")}'
            if value is True:
                argv.append(option)
            elif value is False or value is None:
                continue
            else:
                argv.extend([option, str(value)])
        with patch.object(sys, 'argv', argv):
            self.assertEqual(self.module.main(), 0, 'generator reported failure')
        notebook = json.loads((output / 'notebook.ipynb').read_text(encoding='utf-8'))
        metadata = json.loads((output / 'kernel-metadata.json').read_text(encoding='utf-8'))
        return output, notebook, metadata

    @staticmethod
    def sources(notebook: dict) -> list:
        return [''.join(cell.get('source', [])) for cell in notebook['cells']]

    def test_every_code_cell_compiles_and_the_source_stays_under_the_api_limit(self):
        output, notebook, _ = self.generate(hide_ipython=True, pin_task_environment=True)
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            compile(''.join(cell.get('source', [])), f'<cell {index}>', 'exec')
        self.assertLess((output / 'notebook.ipynb').stat().st_size, 1_000_000)

    def test_pinned_restore_is_opt_in(self):
        _, plain, plain_metadata = self.generate()
        self.assertEqual(plain_metadata['kernel_sources'], [])
        self.assertNotIn('pinned-wheels.json', '\n'.join(self.sources(plain)))

    def test_a_pinned_wheel_that_differs_from_the_manifest_is_rejected(self):
        """The restore step must refuse bytes the review never saw, before any notebook is written."""
        pinned = self.root / 'pinned'
        pinned.mkdir()
        for name in self.module.PINNED_HARNESS_WHEELS:
            (pinned / name).write_bytes(b'not the reviewed wheel')
        with self.assertRaises(SystemExit):
            self.generate(pinned_wheels_dir=pinned)

    def test_the_embedded_pinned_payload_installs_after_hash_verification(self):
        reviewed_dir = PINNED_WHEELS_FETCHED
        if not reviewed_dir.is_dir():
            self.skipTest(f'the reviewed harness wheels are not staged at {reviewed_dir} '
                          '(they live under gitignored runs/)')
        output, notebook, metadata = self.generate(pinned_wheels_dir=reviewed_dir)
        self.assertEqual(metadata['kernel_sources'], [])
        restore = [source for source in self.sources(notebook)
                   if 'pinned-wheels.json' in source]
        self.assertEqual(len(restore), 1, 'exactly one restore cell is expected')
        self.assertIn('--force-reinstall', restore[0])
        self.assertIn('reviewed harness wheels', restore[0])
        reviewed = json.loads(REVIEWED_MANIFEST.read_text(encoding='utf-8'))
        hashes = {item['path']: item['sha256'] for item in reviewed['wheelhouse_files']}
        for name in self.module.PINNED_HARNESS_WHEELS:
            self.assertIn(hashes[name], restore[0], name)
        record = json.loads((output / 'generation-record.json').read_text(encoding='utf-8'))
        self.assertEqual(record['pinned_wheels'], sorted(self.module.PINNED_HARNESS_WHEELS))

    def test_starlette_pin_is_measured_rather_than_assumed(self):
        """Versions from 0.47 reject `Router(on_startup=...)`, and the FastAPI snapshots declare none.

        Measured on 2026-10-13: both `fastapi_13713` and `fastapi_14786` failed collection in **both**
        arms with `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`, while
        `task-environment-pins.json` recorded `{}` for them.
        """
        _, notebook, _ = self.generate(pin_task_environment=True)
        joined = '\n'.join(self.sources(notebook))
        self.assertIn('_measured_starlette', joined)
        self.assertIn('Router(on_startup=[])', joined)
        self.assertIn('starlette_probe', joined)

    def test_the_reviewed_manifest_lists_every_pinned_wheel(self):
        """A restore step can only be honest if the frozen manifest carries each reviewed digest."""
        reviewed = json.loads(REVIEWED_MANIFEST.read_text(encoding='utf-8'))
        paths = {item['path'] for item in reviewed['wheelhouse_files']}
        missing = sorted(set(self.module.PINNED_HARNESS_WHEELS) - paths)
        self.assertEqual(missing, [], f'{missing} are not in the reviewed manifest')

    def test_the_reviewed_manifest_does_not_list_the_drifted_wheels(self):
        """The point of the restore step: the live mount serves bytes the review never saw."""
        reviewed = json.loads(REVIEWED_MANIFEST.read_text(encoding='utf-8'))
        paths = {item['path'] for item in reviewed['wheelhouse_files']}
        self.assertNotIn('swegemma-0.2.11-py3-none-any.whl', paths)


if __name__ == '__main__':
    unittest.main()
