"""Bounded four-L4 LoRA feasibility worker.

This is the training entry point for the L4 training-feasibility task. It is deliberately small and
self-limiting: it loads the required competition base, proves the native chat-template boundary on the
real tokenizer, completes one backward step, then at most ``--updates`` optimizer updates, saves the
adapter and writes machine-readable measurements.

It is *not* a training recipe. It never runs a full corpus or a second epoch, and it stops on the
first incompatibility instead of trying to work around it.

Run under ``torchrun`` from the generated Kaggle notebook; it needs a distributed launch because the
BF16 dequantized base (about 58 GiB) does not fit one L4 (about 22.5 GiB usable):

    torchrun --standalone --nproc_per_node=4 train_l4_feasibility.py \
        --base /kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2 \
        --work /kaggle/working/l4-feasibility --updates 8 --seq-len 2048

Two sharding strategies are attempted in order, and the first that builds is the one that trains:

``fsdp-ignored``
    ``get_peft_model`` first, then FSDP the frozen base with the LoRA modules passed as
    ``ignored_modules``. LoRA parameters stay replicated on every rank, so the adapter export needs no
    parameter gather -- the one memory-safe route on a host whose RAM cannot hold a full BF16 replica.
    It requires an explicit all-reduce of the LoRA gradients because FSDP does not synchronise ignored
    parameters.
``fsdp-all``
    FSDP the whole PEFT model. Export then needs a full state-dict gather, which materialises a
    complete replica in CPU RAM on rank 0 and is only attempted when the host reports enough memory.

Every strategy failure is recorded with its exact exception; the run then moves on rather than
retrying the same thing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantization_config_compat import build_quantization_config  # noqa: E402

SYSTEM_PROMPT = (
    'You repair Python repositories using the available tools. Inspect relevant code, make a focused '
    'change, run a regression test, and submit the patch. Use exact argument names. After a successful '
    'submission, return a brief final completion message.'
)
ISSUE_TEXT = (
    'The report renderer drops trailing spaces when a cell spans multiple lines. Fix the renderer and '
    'run its regression test.'
)
TOOL_SCHEMA = [{
    'type': 'function',
    'function': {
        'name': 'read_file',
        'description': 'Read a workspace file with optional inclusive line bounds.',
        'parameters': {'type': 'object',
                       'properties': {'filepath': {'type': 'string'},
                                      'start_line': {'type': 'integer'},
                                      'end_line': {'type': 'integer'}},
                       'required': ['filepath']},
    },
}, {
    'type': 'function',
    'function': {
        'name': 'submit_patch',
        'description': 'Capture the current Git diff for evaluation.',
        'parameters': {'type': 'object', 'properties': {}, 'required': []},
    },
}]
TARGET_SUFFIXES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
HANDOFF_TOKEN = '<|tool_response>'


def host_memory_gib() -> float:
    """MemTotal from /proc/meminfo, the only dependable figure inside a Kaggle session."""
    try:
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


class Recorder:
    """Append-only measurement log plus an atomically replaced progress file.

    A failed Kaggle session publishes nothing until the notebook ends, so every phase must be readable
    from disk while the session is still running.
    """

    def __init__(self, work: Path, rank: int):
        self.work = work
        self.rank = rank
        self.started = time.monotonic()
        if rank == 0:
            work.mkdir(parents=True, exist_ok=True)

    def phase(self, name: str, **fields) -> None:
        if self.rank != 0:
            return
        record = {'phase': name, 'elapsed_s': round(time.monotonic() - self.started, 3), **fields}
        with (self.work / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
        temporary = self.work / 'progress.json.tmp'
        done = []
        path = self.work / 'progress.jsonl'
        if path.is_file():
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    done.append(json.loads(line))
        temporary.write_text(json.dumps(done, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.work / 'progress.json')
        print(f'[l4] {name} {json.dumps({k: v for k, v in fields.items() if k != "traceback"}, default=str)}',
              flush=True)


def native_batch(tokenizer, seq_len: int) -> dict:
    """Build one native-protocol example and check the supervised handoff boundary on this tokenizer.

    The contract under review: a pending tool call ends at token 50 (``<|tool_response>``), it is part
    of the supervised target, and the template must not move the prompt/assistant loss boundary.
    """
    import torch

    call = {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': 'call_0', 'type': 'function',
        'function': {'name': 'read_file', 'arguments': {'filepath': 'src/report.py'}}}]}
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': ISSUE_TEXT}, call]
    prefix = tokenizer.apply_chat_template(messages[:-1], tools=TOOL_SCHEMA, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
    whole = tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMA, tokenize=False,
                                          add_generation_prompt=False, enable_thinking=True)
    handoff_id = tokenizer.convert_tokens_to_ids(HANDOFF_TOKEN)
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    whole_ids = tokenizer.encode(whole, add_special_tokens=False)
    checks = {
        'handoff_token_id': handoff_id,
        'handoff_token_is_50': handoff_id == 50,
        'whole_starts_with_prefix': whole.startswith(prefix),
        'token_boundary_matches': whole_ids[:len(prompt_ids)] == prompt_ids,
        'target_ends_with_handoff': bool(whole_ids) and whole_ids[-1] == handoff_id,
    }
    if not all(checks[key] for key in ('handoff_token_is_50', 'whole_starts_with_prefix',
                                       'token_boundary_matches', 'target_ends_with_handoff')):
        raise ValueError(f'Native chat-template handoff contract failed: {checks}')
    target = whole_ids[len(prompt_ids):]
    input_ids = torch.tensor([prompt_ids + target], dtype=torch.long)
    checks['prompt_tokens'] = len(prompt_ids)
    checks['target_tokens'] = len(target)
    checks['sequence_tokens'] = int(input_ids.shape[1])
    if input_ids.shape[1] > seq_len:
        raise ValueError(f'Native example is {input_ids.shape[1]} tokens, above --seq-len {seq_len}; '
                         'raise --seq-len rather than truncating the target')
    return {'input_ids': input_ids, 'target_start': len(prompt_ids), 'checks': checks}


def normalise_peft_key(key: str) -> str:
    """Map an FSDP/PEFT state-dict key onto the scheme this base is served with.

    The serving stack was measured to accept
    ``base_model.model.model.language_model.layers.N.self_attn.<proj>.lora_A.weight``; PEFT state
    dicts carry a ``.default`` adapter-name infix and FSDP prefixes wrapped modules.
    """
    return key.replace('_fsdp_wrapped_module.', '').replace('.default.weight', '.weight')


def target_loss(network, input_ids, positions):
    """Cross-entropy over the supervised positions, with logits aligned to the next target token.

    Shared by the four-GPU trainer, the single-GPU packed probe and the device-mapped route so all of them
    measure the same objective.

    The positions go in as an integer count, not as an index tensor. ``transformers`` turns a non-integer
    ``logits_to_keep`` into ``hidden_states[:, indices, :]``
    (``models/gemma4/modeling_gemma4.py``), and on a model dispatched across several GPUs those hidden
    states live on the last GPU, so an index tensor built on the first GPU raises
    ``RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cuda:3)``
    -- measured on run 9. The supervised positions are the trailing ones by construction
    (``positions = arange(target_start, seq_len)``), so keeping ``len(positions) + 1`` trailing hidden
    states and dropping the last row yields exactly the logits ``positions - 1`` used to select: the
    hidden state before each supervised token.
    """
    import torch
    import torch.nn.functional as F

    count = int(positions.numel())
    trailing = torch.arange(input_ids.shape[1] - count, input_ids.shape[1], device=input_ids.device)
    if not torch.equal(positions.to(input_ids.device), trailing):
        raise ValueError('target_loss expects the supervised positions to be the trailing ones, '
                         f'got {positions.tolist()[:4]}... against a sequence of '
                         f'{input_ids.shape[1]} tokens')

    # PEFT keeps adapter master weights in FP32. Autocast is also needed for their projections; a
    # BF16 frozen base alone does not enable AMP.
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = network(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                         logits_to_keep=count + 1, use_cache=False)
    logits = output.logits[0].float()[:count]
    # The targets follow the logits' device: a model dispatched across several GPUs can put the language
    # head on a different device from the embeddings the example was built on, and cross-entropy across
    # two devices raises. On a single-device model this is a no-op.
    return F.cross_entropy(logits, input_ids[0, positions].to(logits.device)), count


def lora_modules(model):
    """The LoRA wrapper modules, identified by the adapter parameters they own."""
    found = []
    for name, module in model.named_modules():
        if any(key.endswith('lora_A') or key.endswith('lora_B') for key in module._parameters):
            found.append((name, module))
    return found


def build_strategy(model, strategy, local_rank):
    """Wrap the model and return ``(network, notes)``.

    Trainable and replicated parameters are read back from the wrapped network by the caller, because
    with ``use_orig_params=True`` FSDP keeps the original parameter objects live and only those are
    the ones the optimizer must see.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

    notes = {}
    if strategy == 'fsdp-ignored':
        ignored = [module for _, module in lora_modules(model)]
        if not ignored:
            raise ValueError('no LoRA modules found to ignore; the adapter wrapped nothing')
        notes['ignored_modules'] = len(ignored)
        network = FSDP(model, device_id=local_rank, sharding_strategy=ShardingStrategy.FULL_SHARD,
                       use_orig_params=True, ignored_modules=ignored, sync_module_states=True)
        return network, notes
    network = FSDP(model, device_id=local_rank, sharding_strategy=ShardingStrategy.FULL_SHARD,
                   use_orig_params=True, sync_module_states=True)
    return network, notes


def trainable_parameters(network, strategy):
    """``(all trainable, replicated LoRA)`` parameters as seen after FSDP wrapping.

    For ``fsdp-ignored`` the LoRA parameters are not sharded, so FSDP never synchronises their
    gradients and the caller must all-reduce them; the second list is exactly that set.
    """
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    if strategy != 'fsdp-ignored':
        return trainable, []
    replicated = [parameter for name, parameter in network.named_parameters()
                  if parameter.requires_grad and 'lora_' in name]
    return trainable, replicated


def save_adapter(network, raw_model, work: Path, strategy: str, config: dict, rank: int,
                 recorder) -> dict:
    """Write a PEFT-format adapter directory, normalising FSDP's key decoration away.

    ``make_lora_probe.py`` established the key scheme this base is served with
    (``base_model.model.model.language_model.layers.N.self_attn.q_proj.lora_A.weight``), so the
    normalisation below is checked against that shape instead of assumed.

    For ``fsdp-ignored`` the adapter is read straight from ``named_parameters``: the LoRA parameters
    were excluded from sharding and are therefore whole on every rank, and reading them directly
    avoids ``state_dict()`` entirely. That matters, because FSDP's default state-dict type gathers a
    complete replica onto rank 0 -- about 58 GiB here -- which is the failure this strategy exists to
    avoid.
    """
    import torch
    import safetensors.torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

    if strategy == 'fsdp-all':
        # The state-dict context is what makes this gather; named_parameters() would return sharded
        # views even inside it.
        context = FSDP.state_dict_type(
            network, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True))
        with context:
            gathered = network.state_dict()
    else:
        gathered = {name: parameter.detach() for name, parameter in raw_model.named_parameters()}

    if rank != 0:
        return {'saved': False, 'reason': 'non-zero rank'}

    def normalise(key: str) -> str:
        return normalise_peft_key(key)

    tensors = {normalise(key): value.detach().to('cpu').to(torch.bfloat16)
               for key, value in gathered.items() if 'lora_' in key}
    lora_keys = [key for key in tensors if key.endswith('.lora_A.weight') or key.endswith('.lora_B.weight')]
    if not lora_keys:
        raise ValueError('no LoRA tensors in the exported state dict; nothing would be reloadable')
    export = work / 'adapter'
    export.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({key: tensors[key] for key in lora_keys},
                                str(export / 'adapter_model.safetensors'))
    (export / 'adapter_config.json').write_text(json.dumps({
        'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': config['rank'],
        'lora_alpha': config['lora_alpha'], 'lora_dropout': 0.0, 'bias': 'none',
        'inference_mode': True, 'base_model_name_or_path': config['base_handle'],
        'target_modules': sorted({key.split('.')[-3] for key in lora_keys}),
        'modules_to_save': [],
    }, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256((export / 'adapter_model.safetensors').read_bytes()).hexdigest()
    return {'saved': True, 'output': str(export), 'tensors': len(lora_keys),
            'bytes': (export / 'adapter_model.safetensors').stat().st_size,
            'sha256': digest, 'example_key': sorted(lora_keys)[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--base-handle', default='google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--wall-budget-seconds', type=float, default=1500.0)
    parser.add_argument('--load-budget-seconds', type=float, default=600.0,
                        help='Wall cap on the base load; the update budget never covers this phase')
    parser.add_argument('--strategies', nargs='+', default=['fsdp-ignored', 'fsdp-all'])
    parser.add_argument('--dequantize', choices=['true', 'false'], default='true',
                        help='true loads the base dequantised to BF16, as the H100 recipe does; '
                             'false keeps the checkpoint as shipped. The keyword that expresses this '
                             'differs by transformers version and is chosen by '
                             'quantization_config_compat.build_quantization_config')
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import importlib.metadata as metadata
    from transformers import AutoModelForImageTextToText, AutoTokenizer, CompressedTensorsConfig
    from transformers.utils.logging import disable_progress_bar
    from peft import LoraConfig, get_peft_model

    disable_progress_bar()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    recorder = Recorder(args.work, rank)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if world > 1:
        dist.init_process_group('nccl')

    environment = {
        'python': platform.python_version(),
        'host_memory_gib': round(host_memory_gib(), 2),
        'cpu_count': os.cpu_count(),
        'world_size': world,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        'gpu_total_mib': [torch.cuda.get_device_properties(index).total_memory // 2 ** 20
                          for index in range(torch.cuda.device_count())],
        'transformers': metadata.version('transformers'),
        'peft': metadata.version('peft'),
        'accelerate': metadata.version('accelerate'),
        'safetensors': metadata.version('safetensors'),
        'dequantize': args.dequantize == 'true',
        'base': args.base,
    }
    for package in ('compressed-tensors', 'torchao', 'bitsandbytes'):
        try:
            environment[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            environment[package] = None
    # The competition image ships transformers 5.0.0 while the H100 recipe ran 5.18.0, and
    # ``CompressedTensorsConfig`` needs the ``compressed_tensors`` package at construction and load
    # time. Recording both facts makes a version-specific failure legible instead of mysterious.
    environment['compressed_tensors_importable'] = importlib.util.find_spec('compressed_tensors') is not None
    try:
        environment['compressed_tensors_config_accepts_dequantize'] = (
            'dequantize' in inspect.signature(CompressedTensorsConfig.__init__).parameters)
        environment['compressed_tensors_config_accepts_kwargs'] = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(CompressedTensorsConfig.__init__).parameters.values())
    except (TypeError, ValueError) as error:
        environment['compressed_tensors_config_signature_error'] = f'{type(error).__name__}: {error}'
    recorder.phase('environment', **environment)

    result = {'environment': environment, 'attempts': [], 'updates': [], 'contract': None,
              'adapter': None, 'stopped_reason': None}
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    batch = native_batch(tokenizer, args.seq_len)
    result['contract'] = batch['checks']
    recorder.phase('native_contract', **batch['checks'])
    input_ids = batch['input_ids'].to(device)
    positions = torch.arange(batch['target_start'], input_ids.shape[1], device=device)

    def loss_for(network):
        return target_loss(network, input_ids, positions)[0]

    loaded = None
    try:
        # The loading-mode keyword differs by transformers version (5.13.1 `run_compressed` versus
        # 5.18/5.19 `dequantize`), and the wrong one is dropped silently. The helper builds the right
        # one and refuses a config whose effective mode disagrees with --dequantize.
        quantization, quantization_record = build_quantization_config(
            CompressedTensorsConfig, args.dequantize == 'true')
        environment['quantization'] = quantization_record
        recorder.phase('quantization_config', **quantization_record)
        # The base load is the longest unbounded step: reading a compressed-tensors checkpoint off the
        # mounted dataset and dequantising it to BF16 on every rank. ``--load-budget-seconds`` bounds
        # it, and when it expires the run records ``load_timeout`` instead of letting the notebook's
        # subprocess timeout kill the process with no measurement at all.
        box: dict = {}

        def load_base() -> None:
            try:
                box['model'] = AutoModelForImageTextToText.from_pretrained(
                    args.base, local_files_only=True, dtype=torch.bfloat16,
                    quantization_config=quantization, attn_implementation='sdpa')
            except BaseException as error:  # re-raised on the main thread below
                box['error'] = error

        loader = threading.Thread(target=load_base, name='base-load', daemon=True)
        loader.start()
        loader.join(args.load_budget_seconds)
        if loader.is_alive():
            recorder.phase('load_timeout', budget_seconds=args.load_budget_seconds,
                           traceback='the base load did not finish within --load-budget-seconds')
            result['stopped_reason'] = 'load_timeout'
            if rank == 0:
                (args.work / 'train-result.json').write_text(
                    json.dumps(result, indent=2, default=str) + '\n', encoding='utf-8')
            if world > 1:
                dist.destroy_process_group()
            return 1
        if 'error' in box:
            raise box['error']
        loaded = box['model']
        targets = [name for name, _ in loaded.named_modules()
                   if name.endswith(tuple(TARGET_SUFFIXES)) and 'vision' not in name]
        if not targets:
            raise ValueError('no text projections matched the configured target suffixes')
        recorder.phase('base_loaded', targets=len(targets),
                       rank0_allocated_gib=round(torch.cuda.memory_allocated(device) / 2 ** 30, 3),
                       rank0_peak_gib=round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3))
    except Exception as error:  # the load itself is a measurement, not a precondition
        recorder.phase('base_load_failed', error=f'{type(error).__name__}: {error}',
                       traceback=traceback.format_exc()[-2000:])
        result['stopped_reason'] = 'base_load_failed'
        if rank == 0:
            (args.work / 'train-result.json').write_text(json.dumps(result, indent=2, default=str) + '\n',
                                                         encoding='utf-8')
        if world > 1:
            dist.destroy_process_group()
        return 1

    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias='none',
                             task_type='CAUSAL_LM', target_modules=targets)
    trained = False
    for strategy in args.strategies:
        if strategy == 'fsdp-all' and host_memory_gib() < 70:
            result['attempts'].append({
                'strategy': strategy, 'built': False,
                'error': f'skipped: the full state-dict gather needs a full replica in host RAM and '
                         f'this host reports {host_memory_gib():.1f} GiB'})
            recorder.phase('strategy_skipped', strategy=strategy,
                           reason='host_memory_below_full_replica')
            continue
        attempt = {'strategy': strategy}
        # The adapter wrap is inside the recorded attempt on purpose: on a checkpoint whose modules are
        # compressed-tensors wrappers, `get_peft_model` is what fails, and while it sat outside this
        # try/except the failure killed the process before any attempt, traceback or result file was
        # written -- the run's own record then said only "the worker wrote no train-result.json".
        try:
            model = get_peft_model(loaded, peft_config) if strategy == 'fsdp-ignored' else loaded
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
            model.enable_input_require_grads()
            if strategy == 'fsdp-all':
                model = get_peft_model(model, peft_config)
            network, notes = build_strategy(model, strategy, local_rank)
            trainable, replicated = trainable_parameters(network, strategy)
            if not trainable:
                raise ValueError('FSDP wrapping left no trainable parameters')
            # The ignored LoRA parameters are replicated, so their random initialisation must be
            # identical on every rank before the first all-reduce; otherwise the ranks would start
            # from different adapters and average unrelated gradients.
            if world > 1 and replicated:
                with torch.no_grad():
                    for parameter in replicated:
                        dist.broadcast(parameter.data, src=0)
            attempt.update({'built': True, 'notes': notes,
                            'trainable_parameters': sum(p.numel() for p in trainable),
                            'replicated_parameters': sum(p.numel() for p in replicated)})
        except Exception as error:
            attempt.update({'built': False, 'error': f'{type(error).__name__}: {error}',
                            'traceback': traceback.format_exc()[-2000:]})
            result['attempts'].append(attempt)
            recorder.phase('strategy_failed', strategy=strategy, error=attempt['error'])
            continue
        recorder.phase('strategy_built', strategy=strategy, notes=notes,
                       allocated_gib=round(torch.cuda.memory_allocated(device) / 2 ** 30, 3),
                       peak_gib=round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3))
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

        def step(number: int) -> dict:
            network.train()
            optimizer.zero_grad(set_to_none=True)
            # Reset before the forward so the peak covers a whole update -- forward, backward and the
            # optimizer's lazily allocated state. A peak measured after the forward would understate
            # exactly the number this run exists to establish.
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            loss = loss_for(network)
            if not torch.isfinite(loss):
                raise ValueError(f'non-finite loss at update {number}')
            loss.backward()
            # FSDP does not synchronise parameters it was told to ignore, so their gradients are
            # all-reduced here; without this every rank would train a different adapter.
            for parameter in replicated:
                if parameter.grad is not None:
                    if world > 1:
                        dist.all_reduce(parameter.grad)
                        parameter.grad /= world
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            return {'update': number, 'loss': float(loss.detach()),
                    'gradient_norm': float(norm), 'seconds': round(time.monotonic() - started, 3),
                    'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3),
                    'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 3),
                    'elapsed_s': round(time.monotonic() - recorder.started, 3)}

        updates = []
        try:
            first = step(0)
            first['kind'] = 'first_update_including_forward'
            updates.append(first)
            recorder.phase('first_update', **first)
            for number in range(1, args.updates):
                if time.monotonic() - recorder.started > args.wall_budget_seconds:
                    result['stopped_reason'] = 'wall_budget'
                    recorder.phase('stopped', reason='wall_budget', completed=len(updates))
                    break
                update = step(number)
                updates.append(update)
                recorder.phase('update', **update)
            if result['stopped_reason'] is None:
                result['stopped_reason'] = 'completed_updates'
            trained = True
        except Exception as error:
            attempt['error'] = f'{type(error).__name__}: {error}'
            attempt['traceback'] = traceback.format_exc()[-2000:]
            recorder.phase('training_failed', strategy=strategy, error=attempt['error'])
        result['updates'] = updates
        attempt['updates'] = len(updates)
        if updates:
            seconds = [row['seconds'] for row in updates]
            attempt['seconds_per_update'] = {'min': min(seconds), 'max': max(seconds),
                                             'mean': round(sum(seconds) / len(seconds), 3)}
            attempt['peak_allocated_gib'] = max(row['peak_allocated_gib'] for row in updates)
            attempt['peak_reserved_gib'] = max(row['peak_reserved_gib'] for row in updates)
        if trained:
            if world > 1:
                dist.barrier()
            try:
                result['adapter'] = save_adapter(network, model, args.work, strategy,
                                                 {'rank': args.rank, 'lora_alpha': args.alpha,
                                                  'base_handle': args.base_handle}, rank, recorder)
                recorder.phase('adapter_saved', **{k: v for k, v in result['adapter'].items()
                                                   if k != 'output'})
            except Exception as error:
                result['adapter'] = {'saved': False, 'error': f'{type(error).__name__}: {error}',
                                     'traceback': traceback.format_exc()[-2000:]}
                recorder.phase('adapter_save_failed', error=result['adapter']['error'])
        attempt['success'] = bool(trained and result['adapter'] and result['adapter'].get('saved'))
        result['attempts'].append(attempt)
        if attempt['success']:
            break

    if rank == 0:
        (args.work / 'train-result.json').write_text(json.dumps(result, indent=2, default=str) + '\n',
                                                     encoding='utf-8')
    recorder.phase('finished', stopped_reason=result['stopped_reason'],
                   adapter_saved=bool(result['adapter'] and result['adapter'].get('saved')))
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
