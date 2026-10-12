"""Regression checks for workspace provenance in disposable evaluation environments."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_eval_setup


class LocalSetupTests(unittest.TestCase):
    def test_shadowed_package_fails_and_does_not_mark_setup_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir()
            manager = SimpleNamespace(sandboxes={'test-sandbox': {
                'venv': root / 'venv', 'workspace': workspace, 'root': root}})
            origin = '/host/site-packages/httpx/__init__.py'
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                stdout = json.dumps({'origin': origin}) if '-c' in command else 'installed'
                return subprocess.CompletedProcess(command, 0, stdout, '')

            vendor = SimpleNamespace(ContainerSetupError=RuntimeError)
            with patch.dict(sys.modules, {'swegemma.harness.container_setup': vendor}), \
                    patch.object(local_eval_setup, '_installed', set()), \
                    patch.object(local_eval_setup.subprocess, 'run', side_effect=run), \
                    patch.object(local_eval_setup.subprocess, 'check_output', return_value='httpx==0.1\n'):
                with self.assertRaisesRegex(RuntimeError, 'outside its workspace'):
                    local_eval_setup.install_local_dependencies(manager, 'test-sandbox', repo='encode/httpx')
                self.assertNotIn('test-sandbox', local_eval_setup._installed)
                origin = str(workspace / 'src/httpx/__init__.py')
                local_eval_setup.install_local_dependencies(manager, 'test-sandbox', repo='encode/httpx',
                                                            config=SimpleNamespace(results_dir=root / 'results'))
                self.assertIn('test-sandbox', local_eval_setup._installed)
                report = json.loads((root / 'results/dependencies-test-sandbox.json').read_text())
                self.assertEqual(report['workspace_package_origin'], origin)
                for command in commands:
                    if 'install' in command:
                        self.assertEqual(command[command.index('--python') + 1], str(root / 'venv/bin/python3'))
                        self.assertIn('--no-index', command)
                before = len(commands)
                local_eval_setup.install_local_dependencies(manager, 'test-sandbox', repo='encode/httpx')
                self.assertEqual(len(commands), before)

    def test_unknown_harness_version_requires_review(self):
        evaluator = SimpleNamespace(sandbox=SimpleNamespace(system_site_packages=True))
        with patch.object(local_eval_setup, 'version', return_value='0.2.8'):
            with self.assertRaisesRegex(RuntimeError, 'harness version'):
                local_eval_setup.configure(evaluator)
        self.assertTrue(evaluator.sandbox.system_site_packages)


if __name__ == '__main__':
    unittest.main()
