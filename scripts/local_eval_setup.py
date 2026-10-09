"""Versioned test-environment setup for the local subprocess backend.

The official subprocess path skips the Docker dependency installer. Give each
sandbox its own packages so installed host libraries cannot shadow a src-layout
repository. Official agent execution and patch verification remain unchanged.
"""
from importlib.metadata import version
from pathlib import Path
import json
import subprocess
import sys
import time

COMMON = ['pytest==8.3.4', 'pytest-timeout==2.1.0', 'inline-snapshot==0.21.1',
          'dirty-equals==0.9.0', 'setuptools==82.0.1', 'wheel==0.45.1',
          'hatchling==1.30.1', 'flit-core==3.12.0', 'pdm-backend==2.4.9', 'poetry-core==2.4.1', 'editables==0.6']
_installed = set()
_hook_installed = False


def install_local_dependencies(manager, sandbox_id, repo='', **kwargs):
    """Install into a disposable sandbox venv; fail visibly on missing offline inputs."""
    from swegemma.harness.container_setup import ContainerSetupError
    if sandbox_id in _installed:
        return
    paths = manager.sandboxes[sandbox_id]
    python = paths['venv'] / 'bin/python3'
    workspace = paths['workspace']
    log = paths['root'] / 'local-setup.log'
    base = [sys.executable, '-m', 'pip', '--python', str(python), 'install', '--no-index',
            '--find-links=/data/wheels', '--find-links=/eval-test-wheels', '--disable-pip-version-check',
            '--no-compile']
    commands = [base + COMMON]
    if repo == 'fastapi/fastapi':
        commands.append(base + ['pydantic==2.13.4'])
    commands.append(base + ['--no-build-isolation', '-e', str(workspace)])
    if repo == 'Textualize/rich':
        # Match the repository's poetry.lock; newer lexer colors break snapshots.
        commands.append(base + ['pygments==2.19.2'])
    # The modern FastAPI test client and existing Requests tests require httpx,
    # while the HTTPX task must import its own workspace distribution.
    if repo != 'encode/httpx':
        commands.append(base + ['httpx==0.28.1'])
    if repo == 'psf/requests':
        commands.append(base + ['pytest-httpbin==2.1.0', 'trustme==1.2.1'])
    if repo == 'encode/httpx':
        commands.append(base + ['pytest-mock==3.16.0', 'pytest-asyncio==0.26.0', 'trio==0.31.0'])
    started = time.perf_counter()
    outputs = []
    for command in commands:
        result = subprocess.run(command, cwd=workspace, text=True, capture_output=True, timeout=180)
        outputs.append(result.stdout + result.stderr)
        log.write_text('\n'.join(outputs))
        if result.returncode:
            raise ContainerSetupError('Local offline workspace setup failed: ' + outputs[-1][-5000:])
    # Keep package provenance after the disposable sandbox is removed, and ensure
    # the package being tested is actually the task's source tree.
    package = repo.split('/')[-1]
    probe = subprocess.run([str(python), '-c',
                            'import importlib.util,json; s=importlib.util.find_spec(' + repr(package) + '); '
                            'print(json.dumps({"origin":s.origin if s else None}))'],
                           cwd=workspace, capture_output=True, text=True, check=True)
    origin = json.loads(probe.stdout)['origin']
    if not origin or not Path(origin).resolve().is_relative_to(workspace.resolve()):
        raise ContainerSetupError('Task package would be imported outside its workspace: ' + str(origin))
    _installed.add(sandbox_id)
    installed = subprocess.check_output([sys.executable, '-m', 'pip', '--python', str(python), 'freeze'], text=True)
    config = kwargs.get('config')
    if config is not None:
        output = Path(config.results_dir) / ('dependencies-' + sandbox_id + '.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'repo': repo, 'workspace_package_origin': origin,
                                      'installed': installed.splitlines(), 'setup_seconds': time.perf_counter() - started}, indent=2) + '\n')
    print(f'Local sandbox dependencies ready for {repo}: {time.perf_counter() - started:.2f}s', flush=True)


def configure(evaluator):
    global _hook_installed
    if version('swegemma') != '0.2.10':
        raise RuntimeError('Review the local dependency hook before changing the official harness version')
    evaluator.sandbox.system_site_packages = False
    if not _hook_installed:
        # These are the two explicit preparation call sites in pinned swegemma
        # 0.2.10. The vendor files and verification/test-selection code are untouched.
        import swegemma.harness.agent_runner as agent_runner
        import swegemma.harness.verification as verification
        agent_runner.install_test_dependencies = install_local_dependencies
        verification.install_test_dependencies = install_local_dependencies
        _hook_installed = True
