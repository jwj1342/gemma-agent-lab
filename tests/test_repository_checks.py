"""Exercise CI gates against small, isolated Git repositories."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gemma-ci-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git_environment = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        self.git('init', '--quiet')
        for name in ['check_repository.py', 'build_submission.py']:
            destination = self.root / 'scripts' / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT / 'scripts' / name, destination)
        self.write('configs/baseline/agent.yaml', 'name: fixture\ninstruction: !include prompt.md\n')
        self.write('configs/baseline/prompt.md', 'A tiny test agent.\n')
        self.write('README.md', '[Agent](configs/baseline/agent.yaml)\n')
        self.git('add', '.')

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], env=self.git_environment,
                              capture_output=True, text=True, check=True)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def check(self):
        return subprocess.run([sys.executable, str(self.root / 'scripts/check_repository.py')],
                              cwd=self.root, capture_output=True, text=True, env=self.git_environment)

    def test_valid_agent_and_documentation_pass(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Repository checks passed', result.stdout)

    def test_existing_but_untracked_link_target_fails(self):
        self.write('README.md', '[Local file](local-only.md)\n')
        self.write('local-only.md', 'This will not be in the clone.\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('untracked or missing local link: local-only.md', result.stderr)

    def test_reference_outside_agent_bundle_fails(self):
        self.write('configs/shared.md', 'Tracked, but absent from the agent ZIP.\n')
        self.git('add', 'configs/shared.md')
        self.write('configs/baseline/agent.yaml', 'instruction: !include ../shared.md\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('escapes agent bundle', result.stderr)
        self.write('configs/baseline/agent.yaml', 'config_path: ../shared.md\n')
        self.assertIn('Non-portable config_path', self.check().stderr)

    def test_nested_reference_resolution_matches_bundle_rules(self):
        self.write('configs/baseline/agent.yaml', 'config_path: nested/child.yaml\n')
        self.write('configs/baseline/nested/child.yaml',
                   'instruction: !include ../prompt.md\nconfig_path: sibling.yaml\n')
        self.write('configs/baseline/sibling.yaml', 'name: root_fallback\n')
        self.git('add', 'configs')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.write('configs/baseline/nested/sibling.yaml', 'name: directory_preferred\n')
        self.git('add', 'configs')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_force_added_ignored_artifact_fails(self):
        self.write('.gitignore', 'access_token\n')
        self.write('access_token', 'DUMMY_TEST_VALUE\n')
        self.git('add', '.gitignore')
        self.git('add', '--force', 'access_token')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('access_token: tracked file matches an ignore rule', result.stderr)

    def test_broken_python_inside_agent_bundle_fails(self):
        self.write('configs/baseline/callback.py', 'def broken(:\n')
        self.git('add', 'configs/baseline/callback.py')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('configs/baseline/callback.py', result.stderr)

    def test_missing_include_and_invalid_yaml_fail(self):
        self.write('configs/baseline/agent.yaml', 'instruction: !include missing.md\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Untracked or missing configuration reference: missing.md', result.stderr)
        self.write('configs/baseline/agent.yaml', 'tools: [unterminated\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('configs/baseline/agent.yaml', result.stderr)

    def test_reference_links_are_checked_but_example_code_is_skipped(self):
        self.write('README.md', '[Agent][agent]\n\n[agent]: configs/baseline/agent.yaml\n'
                   '\n```markdown\n[Example](missing-example.md)\n```\n'
                   '\n`[Inline example](missing-inline.md)`\n')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.write('README.md', '[Missing][missing]\n\n[missing]: absent.md\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('absent.md', result.stderr)

    def test_upstream_source_is_not_executed_or_linted(self):
        self.write('resource/notebooks/public/example/source.py', 'deliberately invalid upstream Python !!!\n')
        self.write('resource/competition/original.md', '[Original source link](unavailable.md)\n')
        self.git('add', 'resource')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_team_python_syntax_error_fails(self):
        self.write('src/broken.py', 'def broken(:\n')
        self.git('add', 'src/broken.py')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('src/broken.py', result.stderr)


if __name__ == '__main__':
    unittest.main()
