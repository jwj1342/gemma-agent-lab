"""Single-task controls must use the same isolated setup as batch controls."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_agent


class SingleTaskEnvironmentTests(unittest.TestCase):
    def test_single_and_batch_controls_configure_isolation_before_verification(self):
        for batch in (False, True):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'eval_config.yaml').write_text(
                    'evaluation:\n  timeout_seconds: 60\n  max_time_minutes: 4\n')
                configured = []

                class Evaluator:
                    def __init__(self, config):
                        self.sandbox = SimpleNamespace(system_site_packages=True, close=lambda: None)
                        self.docker = self

                def configure(evaluator):
                    evaluator.sandbox.system_site_packages = False
                    configured.append(evaluator)

                async def verify(manager, config, task, snapshot, **kwargs):
                    # Offline grading has the same command allowance in both
                    # modes; it must not inherit the agent's 60-second limit.
                    self.assertEqual(config.timeout_seconds, 300)
                    self.assertEqual(config.max_time_minutes, 4)
                    # Model a collection failure when setup was skipped. This
                    # exercises the real single/batch orchestration, not imports.
                    isolated = not manager.sandbox.system_site_packages
                    result = {'task_id': task.instance_id, 'resolved': isolated,
                              'test_exit_code': 0 if isolated else 2,
                              'test_output': '1 passed' if isolated else 'ImportError: missing dependency'}
                    return SimpleNamespace(model_dump=lambda **kw: result)

                task = SimpleNamespace(instance_id='fixture', repo='owner/repo', patch='gold patch')
                definitions = {
                    'litellm': {},
                    'adk_submission': {'VllmConfig': object, 'VllmServer': object,
                        'discover_adapters': lambda *a, **k: SimpleNamespace(adapters={})},
                    'swegemma.config': {'EvalConfig': SimpleNamespace, 'ALLOWED_ADAPTER_EXTENSIONS': []},
                    'swegemma.context': {'SwegemmaContext': object},
                    'swegemma.evaluate': {'Evaluator': Evaluator},
                    'swegemma.harness.agent_runner': {'run_agent_sandbox': object},
                    'swegemma.harness.verification': {'verify_task': verify, 'save_trace_artifact': object},
                    'swegemma.models': {'load_tasks': lambda path: [task]},
                    'swegemma.models.registry': {'setup_gemma_model_registry': lambda **kw: {}},
                    'google.adk.agents.context_cache_config': {'ContextCacheConfig': SimpleNamespace},
                    'google.adk.apps._configs': {'EventsCompactionConfig': SimpleNamespace},
                    'local_eval_setup': {'configure': configure},
                }
                modules = {}
                for name, attributes in definitions.items():
                    parts = name.split('.')
                    for end in range(1, len(parts) + 1):
                        parent = '.'.join(parts[:end])
                        modules.setdefault(parent, ModuleType(parent))
                    modules[name].__dict__.update(attributes)
                args = SimpleNamespace(mode='verify', task_ids=['fixture'] if batch else None,
                    task_id=None if batch else 'fixture', tasks=root / 'tasks.jsonl',
                    data_dir=root, agent_dir=root, results=root / 'results',
                    prediction=root / 'prediction.json', patch_source='gold')
                with patch.dict(sys.modules, modules), patch.object(evaluate_agent, 'version', return_value='test'), \
                        contextlib.redirect_stdout(io.StringIO()):
                    asyncio.run(evaluate_agent.run(args))
                result_path = args.results / 'fixture/result.json' if batch else args.results / 'result.json'
                result = json.loads(result_path.read_text())
                self.assertEqual(len(configured), 1)
                self.assertEqual(result['test_exit_code'], 0)
                self.assertTrue(result['resolved'])


if __name__ == '__main__':
    unittest.main()
