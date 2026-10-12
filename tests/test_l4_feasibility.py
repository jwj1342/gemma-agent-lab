"""Offline checks for the four-L4 training-feasibility tooling.

These tests must run on CPU with no GPU, no base weights and no network. They cover the parts whose
correctness can be established without a model: the adapter key normalisation that turns an FSDP/PEFT
state dict into the naming scheme the serving stack was measured to accept, the adapter verifier's
decision rule, and the notebook generator's embedding and scope guards.

The workers are loaded by path rather than imported as ``resource.l4_feasibility``: on Linux
``resource`` is a standard-library module, so an import under that name would resolve to the wrong
package in CI.

The verifier's decision rule is a pure function so it can be exercised with in-memory tensor names.
The file-level CLI checks need a writable nested directory, which some sandboxes deny a confined
process; they skip there with the reason recorded instead of failing.
"""
import base64
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / 'resource/l4_feasibility'
SERVED_KEY = 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight'
TUNABLE_KEY = 'base_model.model.model.language_model.layers.0.self_attn.q_proj.base_layer.weight'
TORCHAO_WHEEL_NAME = 'torchao-0.16.0-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl'


def environment_allows_scratch_directories() -> bool:
    """Whether a confined process here can create a directory and write a file inside it."""
    raw = None
    try:
        raw = Path(tempfile.mkdtemp(prefix='l4-probe-'))
        nested = raw / 'adapter'
        nested.mkdir()
        (nested / 'probe.txt').write_text('x', encoding='utf-8')
        return True
    except OSError:
        return False
    finally:
        if raw is not None:
            shutil.rmtree(raw, ignore_errors=True)


SCRATCH_ALLOWED = environment_allows_scratch_directories()
SCRATCH_SKIP = 'this sandbox denies a confined process nested directory creation'


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapter_keys(rank: int = 16, layers=(0, 1), projections=('q_proj', 'o_proj')):
    """Tensor names and shapes shaped like a real export of this base."""
    keys, shapes = [], {}
    for layer in layers:
        for projection in projections:
            stem = f'base_model.model.model.language_model.layers.{layer}.self_attn.{projection}'
            keys.append(f'{stem}.lora_A.weight')
            shapes[f'{stem}.lora_A.weight'] = [rank, 8]
            keys.append(f'{stem}.lora_B.weight')
            shapes[f'{stem}.lora_B.weight'] = [4, rank]
    return keys, shapes


@contextmanager
def scratch_directory():
    """A scratch directory, cleaned up best-effort so a denied delete cannot fail a test."""
    raw = Path(tempfile.mkdtemp(prefix='l4-feasibility-'))
    try:
        yield raw / 'adapter'
    finally:
        shutil.rmtree(raw, ignore_errors=True)


class KeyNormalisationTests(unittest.TestCase):
    """The exported adapter must land on the naming scheme the serving stack already accepted."""

    def setUp(self):
        self.module = load_module(WORKERS / 'train_l4_feasibility.py', 'l4_train')

    def test_maps_peft_and_fsdp_decoration_onto_the_served_scheme(self):
        decorated = ('_fsdp_wrapped_module.base_model.model.model.language_model.layers.0'
                     '.self_attn.q_proj.lora_A.default.weight')
        self.assertEqual(self.module.normalise_peft_key(decorated), SERVED_KEY)

    def test_leaves_an_already_normalised_key_unchanged(self):
        self.assertEqual(self.module.normalise_peft_key(SERVED_KEY), SERVED_KEY)

    def test_does_not_strip_unrelated_keys(self):
        self.assertEqual(self.module.normalise_peft_key(TUNABLE_KEY), TUNABLE_KEY)

    def test_host_memory_helper_returns_a_number(self):
        self.assertIsInstance(self.module.host_memory_gib(), float)


class NativeBatchTests(unittest.TestCase):
    """The handoff check must not depend on a module-global torch binding."""

    class Tensor:
        def __init__(self, rows):
            self.shape = (len(rows), len(rows[0]))

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return 'prompt' if kwargs.get('add_generation_prompt') else 'prompt-target'

        def convert_tokens_to_ids(self, token):
            return 50

        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text == 'prompt' else [1, 2, 3, 50]

    def setUp(self):
        self.module = load_module(WORKERS / 'train_l4_feasibility.py', 'l4_native_batch')

    def test_imports_torch_locally_and_builds_the_supervised_boundary(self):
        fake_torch = types.SimpleNamespace(long='long', tensor=lambda rows, dtype: self.Tensor(rows))
        with patch.dict(sys.modules, {'torch': fake_torch}):
            batch = self.module.native_batch(self.Tokenizer(), 8)
        self.assertEqual(batch['target_start'], 2)
        self.assertEqual(batch['checks']['handoff_token_id'], 50)
        self.assertEqual(batch['checks']['sequence_tokens'], 4)


class AdapterVerifierRuleTests(unittest.TestCase):
    """The accept/reject decision, exercised without touching the filesystem."""

    def setUp(self):
        self.module = load_module(WORKERS / 'verify_l4_adapter.py', 'l4_verify')

    def test_accepts_a_well_formed_adapter(self):
        keys, shapes = adapter_keys()
        result = self.module.analyse(keys, shapes, {'r': 16}, 16)
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['a_tensors'], result['b_tensors'])
        self.assertEqual(result['rank_from_shapes'], [16])
        self.assertEqual(result['layers_covered'], 2)
        self.assertEqual(result['naming_note'], 'all keys match')

    def test_rejects_a_rank_that_disagrees_with_the_config(self):
        keys, shapes = adapter_keys(rank=16)
        result = self.module.analyse(keys, shapes, {'r': 16}, 8)
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('rank from shapes' in problem for problem in result['problems']))

    def test_rejects_a_config_rank_that_disagrees_with_the_tensors(self):
        keys, shapes = adapter_keys(rank=16)
        result = self.module.analyse(keys, shapes, {'r': 16}, 16)
        self.assertEqual(result['problems'], [])
        broken = self.module.analyse(keys, shapes, {'r': 32}, 16)
        self.assertTrue(any('adapter_config r=32' in problem for problem in broken['problems']))

    def test_rejects_an_adapter_with_one_missing_side(self):
        keys, shapes = adapter_keys(layers=(0,))
        dropped = 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight'
        keys = [key for key in keys if key != dropped]
        result = self.module.analyse(keys, shapes, {'r': 16}, 16)
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('one side' in problem for problem in result['problems']))

    def test_rejects_an_adapter_with_no_recognised_key(self):
        result = self.module.analyse([TUNABLE_KEY], {TUNABLE_KEY: [4, 16]}, {'r': 16}, 16)
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('naming scheme' in problem for problem in result['problems']))

    def test_an_unexpected_prefix_is_reported_not_failed(self):
        """A PEFT export may use another module path; the serving check has to confirm it."""
        keys, shapes = adapter_keys()
        foreign = 'base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight'
        keys.append(foreign)
        shapes[foreign] = [16, 8]
        result = self.module.analyse(keys, shapes, {'r': 16}, 16)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['unmatched_count'], 1)
        self.assertIn('outside the measured scheme', result['naming_note'])


@unittest.skipUnless(SCRATCH_ALLOWED, SCRATCH_SKIP)
class AdapterVerifierCliTests(unittest.TestCase):
    """The CLI over a real adapter directory."""

    def write_adapter(self, directory: Path, rank: int, config_rank=None) -> None:
        import numpy
        from safetensors.numpy import save_file

        keys, shapes = adapter_keys(rank=rank)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({key: numpy.zeros(shapes[key], dtype=numpy.float32) for key in keys},
                  str(directory / 'adapter_model.safetensors'))
        (directory / 'adapter_config.json').write_text(json.dumps({
            'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': config_rank or rank,
            'lora_alpha': (config_rank or rank) * 2, 'lora_dropout': 0.0, 'bias': 'none',
            'inference_mode': True, 'base_model_name_or_path': 'gemma-4-31b-it-qat-w4a16-ct',
            'target_modules': ['o_proj', 'q_proj'], 'modules_to_save': [],
        }, indent=2) + '\n', encoding='utf-8')

    def run_verifier(self, adapter: Path, rank: int):
        completed = subprocess.run(
            [sys.executable, str(WORKERS / 'verify_l4_adapter.py'), '--adapter', str(adapter),
             '--expected-rank', str(rank)],
            capture_output=True, text=True, cwd=str(ROOT))
        self.assertTrue(completed.stdout.strip(), completed.stderr)
        return completed.returncode, json.loads(completed.stdout)

    def test_accepts_a_well_formed_adapter_on_disk(self):
        with scratch_directory() as adapter:
            self.write_adapter(adapter, 16)
            code, result = self.run_verifier(adapter, 16)
            self.assertEqual(code, 0, result)
            self.assertEqual(result['problems'], [])
            self.assertGreater(result['bytes'], 0)

    def test_rejects_a_rank_mismatch_on_disk(self):
        with scratch_directory() as adapter:
            self.write_adapter(adapter, 16)
            code, result = self.run_verifier(adapter, 8)
            self.assertEqual(code, 1)
            self.assertEqual(result['status'], 'invalid')

    def test_reports_a_missing_adapter_instead_of_succeeding(self):
        with scratch_directory() as adapter:
            code, result = self.run_verifier(adapter, 16)
            self.assertEqual(code, 1)
            self.assertEqual(result['status'], 'incomplete')


class TrainableParameterSplitTests(unittest.TestCase):
    """Which parameters the optimizer sees, and which need an explicit all-reduce.

    Exercised with plain stubs so the test needs no torch and no GPU, matching the rest of the suite.
    """

    class Parameter:
        def __init__(self, requires_grad: bool):
            self.requires_grad = requires_grad

    class Network:
        def __init__(self, named):
            self._named = named

        def parameters(self):
            return [parameter for _, parameter in self._named]

        def named_parameters(self):
            return list(self._named)

    def setUp(self):
        self.module = load_module(WORKERS / 'train_l4_feasibility.py', 'l4_train')

    def network(self, strategy_pairs):
        named = [(name, self.Parameter(flag)) for name, flag in strategy_pairs]
        return self.Network(named), {name: parameter for name, parameter in named}

    def test_ignored_strategy_collects_only_lora_parameters_for_the_all_reduce(self):
        network, by_name = self.network([
            ('base_model.model.layers.0.self_attn.q_proj.base_layer.weight', False),
            ('base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight', True),
            ('base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight', True),
        ])
        trainable, replicated = self.module.trainable_parameters(network, 'fsdp-ignored')
        self.assertEqual(trainable, [by_name['base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight'],
                                     by_name['base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight']])
        self.assertEqual(replicated, trainable)

    def test_fsdp_all_needs_no_manual_all_reduce(self):
        network, by_name = self.network([
            ('layers.0.self_attn.q_proj.base_layer.weight', False),
            ('layers.0.self_attn.q_proj.lora_A.default.weight', True),
        ])
        trainable, replicated = self.module.trainable_parameters(network, 'fsdp-all')
        self.assertEqual(trainable, [by_name['layers.0.self_attn.q_proj.lora_A.default.weight']])
        self.assertEqual(replicated, [])

    def test_a_frozen_adapter_yields_nothing_to_train(self):
        network, _ = self.network([('layers.0.self_attn.q_proj.base_layer.weight', False)])
        trainable, replicated = self.module.trainable_parameters(network, 'fsdp-ignored')
        self.assertEqual((trainable, replicated), ([], []))


class NotebookGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(ROOT / 'scripts/prepare_l4_feasibility_notebook.py', 'l4_prepare')

    def arguments(self, **overrides):
        class Args:
            owner, slug = 'someone', 'gemma4-l4-train-feasibility'
            title = 't'
            output = Path('unused')
            updates, seq_len = 8, 2048
            packed_budget, packed_timeout = 420.0, 600.0
            train_budget, train_timeout = 600.0, 1000.0
            device_map_budget, device_map_timeout = 700.0, 1100.0
            serving_timeout, session_limit = 600.0, 3300.0
            load_budget = 480.0
            check = True

        args = Args()
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_every_code_cell_compiles_and_workers_round_trip(self):
        notebook, plan = self.module.build_notebook(self.arguments())
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(code_cells), 7)
        for cell in code_cells:
            compile(''.join(cell['source']), '<cell>', 'exec')
        self.assertEqual(set(plan['workers']), set(self.module.WORKERS))
        for name, digest in plan['workers'].items():
            self.assertEqual(len(digest), 64)
            self.assertEqual(hashlib.sha256((WORKERS / name).read_bytes()).hexdigest(), digest)

    def test_stage_cell_embeds_compressed_workers_with_matching_digests(self):
        notebook, plan = self.module.build_notebook(self.arguments())
        stage = ''.join(notebook['cells'][1]['source'])
        blobs, digests = self.module.embed_workers()
        for name, blob in blobs.items():
            self.assertIn(blob, stage)
            self.assertEqual(hashlib.sha256(gzip.decompress(base64.b85decode(blob))).hexdigest(),
                             digests[name])
        self.assertIn('WORKERS_SHA256', stage)
        for name, digest in plan['workers'].items():
            self.assertIn(f'"{name}": "{digest}"', stage)

    def test_stage_cell_rejects_wrong_hardware_and_resolves_both_mount_layouts(self):
        notebook, _ = self.module.build_notebook(self.arguments())
        stage = ''.join(notebook['cells'][1]['source'])
        self.assertIn("len(gpu_names) == 4 and all('L4' in name", stage)
        self.assertIn("Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')", stage)
        self.assertIn("Path('/kaggle/input/gemma-4-developer-agent-wheelhouse')", stage)
        self.assertIn('preflight-result.json', stage)
        self.assertIn('raise SystemExit', stage)

    def test_plan_stays_inside_the_declared_bounds(self):
        _, plan = self.module.build_notebook(self.arguments())
        self.assertLessEqual(plan['updates_cap'], 8)
        self.assertLessEqual(plan['session_limit_seconds'], 3600)
        self.assertLess(plan['packed_wall_budget_seconds'], plan['packed_subprocess_timeout_seconds'])
        self.assertLess(plan['train_wall_budget_seconds'], plan['train_subprocess_timeout_seconds'])
        self.assertLess(plan['device_map_wall_budget_seconds'],
                        plan['device_map_subprocess_timeout_seconds'])
        self.assertEqual(plan['quota_multiplier'], 2.0)
        # A session cap that the phases themselves can overrun is not a cap.
        timeouts = (plan['packed_subprocess_timeout_seconds'] + plan['train_subprocess_timeout_seconds']
                    + plan['device_map_subprocess_timeout_seconds'] + plan['serving_timeout_seconds'])
        self.assertLessEqual(timeouts, plan['session_limit_seconds'])
        self.assertIn('packed-single-gpu', plan['phase_order'])
        self.assertIn('single-process-device-map', plan['phase_order'])
        self.assertIn('only when the packed', plan['fsdp_policy'])
        self.assertIn('no adapter exists yet', plan['device_map_policy'])

    def test_later_cells_rebuild_constants_from_the_plan_file(self):
        notebook, _ = self.module.build_notebook(self.arguments())
        for index in (2, 3, 4, 5, 6, 7):
            source = ''.join(notebook['cells'][index]['source'])
            self.assertIn('plan.json', source)
            self.assertIn("Path('/kaggle/working/l4-feasibility')", source)
            self.assertNotIn('@@', source)

    def test_no_cell_imports_the_training_stack(self):
        """The notebook kernel must stay stdlib-only.

        Importing transformers after the wheelhouse install raises the torchao conflict measured on
        this image, and the kernel must survive that install to reach the reload phase.
        """
        notebook, _ = self.module.build_notebook(self.arguments())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            for forbidden in ('import torch', 'import transformers', 'import peft', 'import vllm'):
                self.assertNotIn(forbidden, source)

    def test_generated_kernel_metadata_attaches_the_competition(self):
        """L4x4 is gated on competition attachment, so a detached kernel silently gets T4x2.

        The first measured session requested `NvidiaL4` with `competition_sources: []` and was given
        two T4s, which the notebook's own stage cell recorded. The metadata is therefore a hardware
        precondition, not boilerplate.
        """
        metadata = self.module.build_metadata(self.arguments())
        self.assertEqual(metadata['machine_shape'], 'NvidiaL4')
        self.assertEqual(metadata['competition_sources'], ['gemma-4-developer-agent'])
        self.assertTrue(metadata['enable_gpu'])
        self.assertFalse(metadata['enable_internet'], 'L4 sessions must have internet disabled')

    def test_budget_ladder_is_rejected_when_it_cannot_hold(self):
        """A phase timeout above the session cap means the session cannot bound itself."""
        arguments = self.arguments()
        self.module.validate_budgets(arguments)
        for field, value in (('session_limit', 4000.0), ('updates', 9),
                             ('load_budget', arguments.train_timeout)):
            with self.subTest(field=field):
                broken = self.arguments(**{field: value})
                with self.assertRaises(SystemExit):
                    self.module.validate_budgets(broken)

    def test_train_cell_passes_the_load_budget_to_the_worker(self):
        """The worker's own load cap must reach it, or a stalled load dies unrecorded."""
        notebook, plan = self.module.build_notebook(self.arguments())
        train = ''.join(notebook['cells'][3]['source'])
        self.assertIn('--load-budget-seconds', train)
        self.assertNotIn('@@', train)
        self.assertLess(plan['load_budget_seconds'], plan['train_subprocess_timeout_seconds'])
        worker = (WORKERS / 'train_l4_feasibility.py').read_text(encoding='utf-8')
        self.assertIn("'load_timeout'", worker)
        self.assertIn('--load-budget-seconds', worker)

    def test_train_cell_launches_the_measured_gpu_count(self):
        """A hard-coded four ranks on a smaller allocation raises `invalid device ordinal`."""
        notebook, _ = self.module.build_notebook(self.arguments())
        train = ''.join(notebook['cells'][3]['source'])
        self.assertIn("environment.json", train)
        self.assertIn("--nproc_per_node={ranks}", train)
        self.assertNotIn("--nproc_per_node=4", train)

    def test_device_map_cell_is_bounded_by_the_session_cap(self):
        """Path D must not start a load it cannot finish inside the session cap."""
        notebook, _ = self.module.build_notebook(self.arguments())
        cell = ''.join(notebook['cells'][4]['source'])
        self.assertIn('train_l4_device_map.py', cell)
        self.assertIn("PLAN['session_limit_seconds']", cell)
        self.assertIn('device-map-skipped.json', cell)
        self.assertIn('adapter_model.safetensors', cell)

    def test_embedded_workers_keep_their_escapes(self):
        """Generated newlines must not depend on how many backslashes survive the template.

        Three layers meet here: the file, the template literal, and the code that ends up in the
        notebook. Both wrong variants have already been produced by hand -- `'\\\\n'` writes the
        literal text ``\\n`` (which corrupts ``plan.json`` and stopped a session at 77 s) and a
        real newline inside the literal stops the cell compiling at all. Generated writes therefore
        use `chr(10)`, which means the same thing at every layer, and no template may spell a
        newline with a backslash.
        """
        source = (ROOT / 'scripts/prepare_l4_feasibility_notebook.py').read_text(encoding='utf-8')
        for cell_name in ('STAGE_CELL', 'TRAIN_CELL', 'PACKED_CELL', 'VERIFY_CELL', 'SERVING_CELL',
                          'SUMMARY_CELL'):
            self.assertIn(f"{cell_name} = r'''", source, cell_name)
        self.assertNotIn("+ '\\n'", source)
        self.assertNotIn("+ '\\\\n'", source)

        notebook, _ = self.module.build_notebook(self.arguments())
        for index, entry in enumerate(notebook['cells']):
            if entry['cell_type'] != 'code':
                continue
            cell_source = ''.join(entry['source'])
            # `+ '<newline>)` is the broken form: a string literal spanning a real newline.
            self.assertNotIn("+ '" + "\n", cell_source, f'cell {index} has a split string literal')
            compile(cell_source, f'<cell {index}>', 'exec')

    def test_generated_writes_append_a_real_newline(self):
        """A JSON file written by a generated cell has to parse, not merely look right."""
        notebook, _ = self.module.build_notebook(self.arguments())
        stage = ''.join(notebook['cells'][1]['source'])
        self.assertIn('+ chr(10))', stage)
        self.assertNotIn("+ '\\", stage)
        # Execute the convention the cells use, rather than trusting the rendering.
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'written.json'
            payload = {'seq_len': 2048, 'fsdp_policy': 'x'}
            target.write_text(json.dumps(payload, indent=2) + chr(10), encoding='utf-8')
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), payload)
            self.assertTrue(target.read_text(encoding='utf-8').endswith('\n'))

    def test_summary_cell_reads_the_peak_fields_the_worker_writes(self):
        """The worker-to-notebook field names are an integration seam with no type system."""
        notebook, _ = self.module.build_notebook(self.arguments())
        summary = ''.join(notebook['cells'][7]['source'])
        worker = (WORKERS / 'train_l4_feasibility.py').read_text(encoding='utf-8')
        for field in ('peak_allocated_gib', 'peak_reserved_gib', 'seconds_per_update',
                      'stopped_reason', 'contract'):
            self.assertIn(field, worker, field)
        for field in ('peak_allocated_gib', 'peak_reserved_gib', 'stopped_reason'):
            self.assertIn(field, summary, field)
        # The old single-peak name must not survive on either side, or the summary would report None.
        self.assertNotIn("'peak_gib'", worker)
        self.assertNotIn('peak_gib', summary)

    def test_stage_cell_runs_the_model_stack_worker_and_attaches_its_dataset(self):
        """The base is unloadable without this step: transformers 5.0.0 has no 'gemma4' config."""
        notebook, plan = self.module.build_notebook(self.arguments())
        stage = ''.join(notebook['cells'][1]['source'])
        self.assertIn('prepare_model_stack.py', stage)
        self.assertIn('model-stack.json', stage)
        self.assertIn("'--torchao-search', '/kaggle/input'", stage)
        self.assertIn('prepare_model_stack.py', plan['workers'])
        metadata = self.module.build_metadata(self.arguments())
        self.assertEqual(metadata['dataset_sources'],
                         ['metric/gemma-4-developer-agent-wheelhouse',
                          'kajarnaklokossou/torchao-0-16-0'])

    def test_summary_cell_loads_every_file_it_reads(self):
        """An unloaded name is a NameError in the last cell of a session that already spent quota.

        `packed` was read by the summary but never assigned, and `compressed_tensors_step` was read from
        `environment.json`, which was written before that key existed.
        """
        notebook, _ = self.module.build_notebook(self.arguments())
        summary = ''.join(notebook['cells'][7]['source'])
        for name in ('environment.json', 'model-stack.json', 'packed-summary.json',
                     'train-summary.json', 'device-map-result.json', 'serving-result.json',
                     'adapter-check.json'):
            self.assertIn(f"load('{name}')", summary, name)
        self.assertIn('model_stack_step', summary)
        self.assertIn('single_process_device_map', summary)


class QuantizationKeywordTests(unittest.TestCase):
    """The loading-mode keyword changed name, and the wrong one is dropped silently.

    transformers 5.13.1 (the wheelhouse's version) takes ``run_compressed`` and 5.18/5.19 take
    ``dequantize``. Passing ``dequantize`` to the older class puts it in ``**kwargs``, which that version
    only forwards when ``config_groups`` or ``kv_cache_scheme`` is given, so it is dropped and the base
    loads compressed while the caller believes it dequantised. These tests pin the mapping and the
    refusal to return a config whose effective mode disagrees with the request.
    """

    def setUp(self):
        self.module = load_module(WORKERS / 'quantization_config_compat.py', 'l4_quant_compat')

    def test_maps_onto_the_older_keyword_and_confirms_the_effective_value(self):
        class Old:
            def __init__(self, run_compressed: bool = True, **kwargs):
                self.run_compressed = run_compressed
                self.swallowed = kwargs

        config, record = self.module.build_quantization_config(Old, True)
        self.assertEqual(record['keyword'], 'run_compressed')
        self.assertTrue(record['effective_dequantize'])
        self.assertFalse(config.run_compressed)

    def test_maps_onto_the_newer_keyword(self):
        class New:
            def __init__(self, dequantize: bool = False, **kwargs):
                self.dequantize = dequantize

        config, record = self.module.build_quantization_config(New, True)
        self.assertEqual(record['keyword'], 'dequantize')
        self.assertTrue(config.dequantize)
        self.assertTrue(record['effective_dequantize'])

    def test_refuses_a_class_with_neither_keyword(self):
        class Neither:
            def __init__(self, config_groups=None):
                self.config_groups = config_groups

        with self.assertRaises(TypeError):
            self.module.build_quantization_config(Neither, True)

    def test_refuses_a_config_that_reports_a_different_mode(self):
        class Wrong:
            """Accepts the request and does the opposite: the shape of the silent-drop failure."""

            def __init__(self, dequantize: bool = False):
                self.dequantize = not dequantize

        with self.assertRaises(ValueError):
            self.module.build_quantization_config(Wrong, True)

    def test_the_workers_build_the_config_through_the_helper(self):
        for name in ('train_l4_feasibility.py', 'probe_packed_base.py'):
            source = (WORKERS / name).read_text(encoding='utf-8')
            self.assertIn('build_quantization_config', source, name)
            self.assertNotIn('CompressedTensorsConfig(dequantize=', source, name)


class ModelStackStepTests(unittest.TestCase):
    """The install decision rule and the torchao provenance check, with fabricated wheels.

    These are the rules that decide whether a session installs, records "already satisfied", or stops
    before touching a GPU, so they are exercised here rather than only inside generated code.
    """

    def setUp(self):
        self.module = load_module(WORKERS / 'prepare_model_stack.py', 'l4_model_stack')
        self.root = Path(tempfile.mkdtemp(prefix='l4-model-stack-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.wheelhouse = self.root / 'wheelhouse'
        self.wheelhouse.mkdir()

    def wheel(self, folder: Path, name: str, payload: bytes = b'x') -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(payload)
        return path

    def mounted_stack(self) -> None:
        for name in ('transformers-5.13.1-py3-none-any.whl', 'tokenizers-0.22.2-cp39-abi3-manylinux.whl',
                     'safetensors-0.8.0-cp310-abi3-manylinux.whl',
                     'compressed_tensors-0.15.0.1-py3-none-any.whl', 'loguru-0.7.3-py3-none-any.whl'):
            self.wheel(self.wheelhouse, name)

    def test_installs_only_below_the_mounted_version(self):
        self.mounted_stack()
        payload = b'torchao-wheel'
        torchao = self.wheel(self.root / 'torchao', TORCHAO_WHEEL_NAME, payload)
        installed = {'transformers': '5.0.0', 'tokenizers': '0.22.2', 'safetensors': '0.7.0',
                     'compressed_tensors': None, 'loguru': None, 'torchao': '0.10.0'}
        with patch.dict(self.module.TORCHAO_SHA256,
                        {TORCHAO_WHEEL_NAME: hashlib.sha256(payload).hexdigest()}):
            plan = self.module.install_plan(self.wheelhouse, torchao, installed=installed.get)
        self.assertEqual(plan['problems'], {})
        self.assertEqual({entry['distribution']: entry['action'] for entry in plan['entries']},
                         {'transformers': 'install', 'tokenizers': 'already_satisfied',
                          'safetensors': 'install', 'compressed_tensors': 'install',
                          'loguru': 'install', 'torchao': 'install'})

    def test_a_wheel_that_does_not_match_the_pypi_digest_is_blocked(self):
        self.mounted_stack()
        payload = b'not-the-pypi-wheel'
        torchao = self.wheel(self.root / 'torchao', TORCHAO_WHEEL_NAME, payload)
        with patch.dict(self.module.TORCHAO_SHA256, {TORCHAO_WHEEL_NAME: 'deadbeef'}):
            plan = self.module.install_plan(self.wheelhouse, torchao, installed=lambda _: None)
        entry = next(item for item in plan['entries'] if item['distribution'] == 'torchao')
        self.assertEqual(entry['action'], 'blocked')
        self.assertEqual(entry['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertIn('sha256', plan['problems']['torchao'])

    def test_a_missing_mounted_wheel_is_blocked_rather_than_skipped(self):
        plan = self.module.install_plan(self.wheelhouse, None, installed=lambda _: None)
        self.assertEqual(set(plan['problems']),
                         {'transformers', 'tokenizers', 'safetensors', 'compressed_tensors', 'loguru',
                          'torchao'})
        for entry in plan['entries']:
            if entry['distribution'] != 'torchao':
                self.assertEqual(entry['action'], 'blocked')

    def test_resolves_every_kaggle_mount_layout(self):
        """`/kaggle/input/<slug>`, `/kaggle/input/<owner>/<slug>` and `/kaggle/input/datasets/<owner>/<slug>`."""
        root = self.root / 'input'
        shallow = self.wheel(root / 'gemma-4-developer-agent-wheelhouse', TORCHAO_WHEEL_NAME)
        nested = self.wheel(root / 'kajarnaklokossou/torchao-0-16-0', TORCHAO_WHEEL_NAME)
        deep = self.wheel(root / 'datasets/kajarnaklokossou/torchao-0-16-0', TORCHAO_WHEEL_NAME)
        found = self.module.resolve_wheel(root, self.module.TORCHAO_PATTERN)
        self.assertEqual(sorted(found), sorted([shallow, nested, deep]))
        # A caller that passes one mount itself rather than the tree above it must resolve too.
        self.assertEqual(self.module.resolve_wheel(root / 'datasets/kajarnaklokossou',
                                                 self.module.TORCHAO_PATTERN), [deep])

    def test_the_recorded_torchao_digest_is_the_pypi_digest(self):
        self.assertEqual(self.module.TORCHAO_SHA256[TORCHAO_WHEEL_NAME],
                         '2d6293a0c57c9dd505efb025a7189459d154965fbed000efd638cf299f9362dd')
        self.assertEqual(self.module.wheel_dist_version(TORCHAO_WHEEL_NAME), ('torchao', '0.16.0'))

    def test_a_failed_check_blocks_the_verdict(self):
        plan = {'entries': [], 'problems': {}}
        passing = {'returncode': 0, 'observed': {'gemma4_in_config_mapping': True,
                                                 'gemma4_config_class': 'Gemma4Config',
                                                 'is_torchao_available': True,
                                                 'quantization': {'effective_dequantize': True},
                                                 'config_model_type': 'gemma4'}}
        self.assertEqual(self.module.judge(plan, passing)['verdict'], 'ok')
        failing = json.loads(json.dumps(passing))
        failing['observed']['is_torchao_available'] = False
        verdict = self.module.judge(plan, failing)
        self.assertEqual(verdict['verdict'], 'blocked')
        self.assertIn('torchao_satisfies_peft_floor', verdict['problems'])

    def test_a_clean_plan_reports_no_problems_before_verification_runs(self):
        """The guard that decides whether to run the verification must see an empty problem set.

        The first session of this step skipped its own verification because a missing verification was
        reported as a problem, which made that guard permanently true.
        """
        clean = {'entries': [], 'problems': {}}
        self.assertEqual(self.module.collect_problems(clean), {})
        self.assertEqual(self.module.judge(clean, None)['problems'], {})
        self.assertEqual(self.module.judge(clean, None)['verdict'], 'blocked')
        failed = {'entries': [{'distribution': 'transformers',
                               'install': {'returncode': 1, 'tail': 'boom'}}], 'problems': {}}
        self.assertIn('transformers', self.module.collect_problems(failed))


class DeviceMapRouteTests(unittest.TestCase):
    """Path D refuses the two load results that both look like success in a log."""

    def setUp(self):
        self.module = load_module(WORKERS / 'train_l4_device_map.py', 'l4_device_map')

    def test_accepts_a_base_spread_over_the_gpus(self):
        verdict = self.module.placement_verdict(
            {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.40': 1, 'lm_head': 3},
            [14.1, 14.4, 14.2, 14.6])
        self.assertEqual(verdict['status'], 'ok')
        self.assertEqual(verdict['problems'], {})
        self.assertEqual(verdict['placement'], {'0': 2, '1': 1, '3': 1})

    def test_refuses_a_base_that_stayed_in_host_ram(self):
        verdict = self.module.placement_verdict({'a': 'cpu', 'b': 'cpu', 'c': 0},
                                               [0.5, 0.0, 0.0, 0.0])
        self.assertEqual(verdict['status'], 'placement_incompatible')
        self.assertIn('host_resident', verdict['problems'])

    def test_refuses_an_unbalanced_placement(self):
        """One L4 holding the base answers a forward pass and then OOMs on the first training step."""
        verdict = self.module.placement_verdict({'a': 0, 'b': 0}, [22.4, 0.2, 0.1, 0.1])
        self.assertEqual(verdict['status'], 'placement_incompatible')
        self.assertIn('single_gpu', verdict['problems'])

    def test_the_route_reuses_the_shared_contract_and_export(self):
        source = (WORKERS / 'train_l4_device_map.py').read_text(encoding='utf-8')
        self.assertIn('from train_l4_feasibility import', source)
        self.assertIn('build_quantization_config', source)
        self.assertIn("device_map='auto'", source)
        self.assertIn('save_adapter', source)


if __name__ == '__main__':
    unittest.main()
