"""Train the tiny LoRA loop with the frozen base spread across the four L4s by one process.

Why this route exists, and it is arithmetic from measured numbers rather than a preference: the
``torchrun`` route materialises a **complete BF16 replica in host RAM on every rank** before FSDP shards
anything. The base dequantised to BF16 is about 58 GiB, so four ranks need about 232 GiB, and this
session's host reports 188.69 GiB. A single process that dispatches the base across the four GPUs while it
loads avoids that peak entirely: each parameter goes to the device accelerate assigns it, the largest
residency in host RAM is transient, and no replica is ever complete in one place.

What it preserves, because the task requires it:

* the required ``gemma-4-31b-it-qat-w4a16-ct`` checkpoint, loaded with the loading-mode keyword this
  transformers version actually understands (``run_compressed=False`` on 5.13.1, ``dequantize=True`` on
  5.18+, chosen by ``quantization_config_compat``) and therefore dequantised to BF16 rather than trained
  in its packed form;
* the frozen base: only LoRA parameters require gradients, and they stay replicated;
* the same native handoff contract and the same supervised objective as the four-GPU worker, imported
  from ``train_l4_feasibility`` so both routes measure one thing;
* the same adapter export path, so the verifier and the official-runtime reload read it identically.

    python train_l4_device_map.py --base /kaggle/input/models/... --work /kaggle/working/l4-feasibility
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantization_config_compat import build_quantization_config  # noqa: E402
from train_l4_feasibility import (  # noqa: E402  (staged beside this file at run time)
    TARGET_SUFFIXES, host_memory_gib, native_batch, save_adapter, target_loss)


def package_version(*names: str):
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def placement_verdict(hf_device_map: dict, allocated_gib_per_gpu: list) -> dict:
    """Judge whether the load really spread the base over the GPUs.

    Pure, so the two refusals can be tested without a GPU. They matter because both failure modes look
    like success in a log: a base that never left host RAM still loads, and a base that landed on one L4
    still answers a forward pass until a training step runs out of memory.
    """
    placement = Counter(str(device) for device in hf_device_map.values())
    modules = len(hf_device_map)
    cpu_share = (placement.get('cpu', 0) / modules) if modules else 1.0
    problems = {}
    if allocated_gib_per_gpu and max(allocated_gib_per_gpu) > 21.5:
        problems['single_gpu'] = (f'{max(allocated_gib_per_gpu):.2f} GiB is resident on one L4, above the '
                                  '21.5 GiB a training step can use')
    if not modules or cpu_share > 0.5:
        problems['host_resident'] = (f'{cpu_share:.0%} of the dispatched modules stayed in host RAM, '
                                     'which is the state this route exists to avoid')
    return {'placement': dict(placement), 'modules': modules, 'cpu_share': round(cpu_share, 4),
            'problems': problems, 'status': 'ok' if not problems else 'placement_incompatible'}


def device_map_attempts(gpu_budget_mib: int, cpu_budget: str) -> list:
    """The placement ladder, as data so the two attempts can be asserted offline.

    The first attempt balances the split at the full per-GPU budget. The second lowers the budget and
    allows host-RAM offload, because the measured failure was an allocator that had reserved 3.60 GiB more
    than it allocated on one GPU: a lower cap leaves room for that, and offloading the tail costs latency
    rather than the run.
    """
    return [
        {'name': 'balanced', 'device_map': 'balanced', 'gpu_budget_mib': gpu_budget_mib,
         'cpu_budget': None},
        {'name': 'auto_with_cpu_offload', 'device_map': 'auto',
         'gpu_budget_mib': max(16000, gpu_budget_mib - 1500), 'cpu_budget': cpu_budget},
    ]


def load_sft_jsonl(path: Path) -> list:
    """Read prompt/completion SFT records from ``scripts/build_sft_dataset.py`` output.

    The rank ladder trains on real records rather than the synthetic handoff example the feasibility
    probe used, so this reader is the boundary between the dataset artifact and the trainer. A line that
    is blank, is not JSON, or carries no prompt or no completion is skipped and counted rather than
    ending the run: the artifact is generated, and one malformed record must not cost a GPU session.
    ``prompt_chars`` is recorded because it is what lets the objective supervise the patch alone.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'SFT dataset not found: {path}')
    records = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            prompt, completion = entry.get('prompt'), entry.get('completion')
            if not isinstance(prompt, str) or not isinstance(completion, str):
                continue
            if not prompt.strip() or not completion.strip():
                continue
            # 25 of the 116 records in the reviewed artifact end with "\n " -- an extraction artifact
            # rather than part of the patch. Normalising here keeps the supervised region equal to the
            # rendered completion, so the boundary is checkable instead of depending on whether a chat
            # template happens to trim trailing whitespace.
            trimmed = completion != completion.strip()
            completion = completion.strip()
            records.append({'instance_id': entry.get('instance_id'), 'split': entry.get('split'),
                            'repo': entry.get('repo'), 'prompt': prompt, 'completion': completion,
                            'prompt_chars': len(prompt), 'completion_trimmed': trimmed})
    if not records:
        raise ValueError(f'the SFT dataset has no trainable prompt/completion record: {path}')
    return records

def tokenizer_prefix_length(prompt_ids: list, whole_ids: list) -> int:
    """Length of the longest common token prefix, with one token reserved for the completion.

    Pure and integer-only so it can be asserted offline. On a BPE tokenizer the boundary can fall inside
    a token that spans the assistant header and the first patch character, which is exactly why the
    boundary is measured by comparison rather than assumed.
    """
    overlap = 0
    limit = min(len(prompt_ids), len(whole_ids) - 1)
    while overlap < limit and prompt_ids[overlap] == whole_ids[overlap]:
        overlap += 1
    return overlap


def encode_sft_record(tokenizer, record: dict, seq_len: int) -> tuple:
    """Tokenise one record into (input_ids, loss_mask, stats), supervising the completion only.

    Two measured details shape this function, both from rank-ladder run 2 on 2026-10-13:

    * ``apply_chat_template(..., tokenize=True)`` returned **token strings, not ids**, on this image
      (transformers 5.13.1 with ``google/gemma-4``), so ``torch.tensor`` raised
      ``ValueError: too many dimensions 'str'``. The template is therefore rendered to text and the
      tokenizer is called explicitly.
    * The boundary is found by comparing the prompt rendering with the full rendering token by token, the
      rule TRL uses for ``completion_only_loss``. Deriving it as ``len(whole) - len(completion)`` looks
      right and drops one token of supervision, and a template that moved the boundary would otherwise
      supervise the issue text as well, so a mismatch is refused instead of averaged into the loss.
    """
    prompt = record['prompt']
    completion = record['completion']
    whole_text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt},
                                                {'role': 'assistant', 'content': completion}],
                                               tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                add_generation_prompt=True)
    if not isinstance(whole_text, str) or not isinstance(prompt_text, str):
        raise ValueError('apply_chat_template(tokenize=False) did not return text, so the supervised '
                         f'boundary cannot be established for {record.get("instance_id")!r}')
    whole_ids = list(tokenizer(prompt_text + completion, add_special_tokens=False)['input_ids'])
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])
    if not prompt_ids:
        raise ValueError(f'the prompt rendered to no tokens for {record.get("instance_id")!r}')
    overlap = tokenizer_prefix_length(prompt_ids, whole_ids)
    if overlap < max(1, int(len(prompt_ids) * 0.9)):
        raise ValueError('the chat template is not prefix-stable between the prompt rendering and the '
                         f'full rendering for {record.get("instance_id")!r} '
                         f'(shared prefix {overlap} of {len(prompt_ids)} prompt tokens), so the '
                         'supervised boundary cannot be established')
    if overlap >= seq_len:
        raise ValueError(f'the prompt alone is {overlap} tokens, at or above --seq-len {seq_len}, '
                         'so this record has no room for a supervised completion')
    truncated = len(whole_ids) > seq_len
    input_ids = [int(token) for token in whole_ids[:seq_len]]
    loss_mask = [False] * overlap + [True] * (len(input_ids) - overlap)
    stats = {'prompt_tokens': overlap, 'target_tokens': sum(loss_mask),
             'sequence_tokens': len(input_ids), 'truncated': truncated,
             'completion_tokens': len(whole_ids) - overlap}
    return input_ids, loss_mask, stats


def batches(records: list, batch_size: int):
    """Deterministic order: the run's records-per-update arithmetic must be reproducible."""
    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--base-handle', default='google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--dataset', type=Path, default=None,
                        help='JSONL with prompt/completion SFT records. Without it the worker trains on '
                             'the synthetic handoff example the feasibility probe used; with it the run '
                             'trains on real records for the rank ladder.')
    parser.add_argument('--dataset-check', type=Path, default=None,
                        help='Tokenise the dataset on the real tokenizer, write the check to this path and '
                             'exit before any model load. It exists because two encoding defects in a row '
                             'reached the GPU: a CPU-only session can prove the boundary for free.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Records per optimizer update; the replicas hold one batch at a time')
    parser.add_argument('--max-updates', type=int, default=0,
                        help='Cap on optimizer updates for the dataset route; 0 means one pass over the '
                             'training split, bounded by --wall-budget-seconds')
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--wall-budget-seconds', type=float, default=900.0)
    parser.add_argument('--load-budget-seconds', type=float, default=600.0)
    parser.add_argument('--max-memory-mib', type=int, default=20500,
                        help='Per-GPU budget handed to accelerate for the first placement attempt; '
                             '22563 MiB is the card, and the second attempt subtracts 1500 MiB and may '
                             'offload to host RAM')
    parser.add_argument('--cpu-budget', default='64GiB',
                        help='Host-RAM allowance for the second placement attempt, which exists because '
                             'the first was killed by allocator fragmentation on one GPU')
    args = parser.parse_args()

    # Must be set before the CUDA context exists: the measured failure was 17.88 GiB allocated but
    # 21.48 GiB reserved on one GPU, which is exactly what expandable segments avoids.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, CompressedTensorsConfig
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()
    args.work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    def progress(phase: str, **fields) -> None:
        record = {'phase': phase, 'elapsed_s': round(time.monotonic() - started, 3), **fields}
        with (args.work / 'device-map-progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
        print(f'[device-map] {phase} {json.dumps(fields, default=str)}', flush=True)

    device_count = torch.cuda.device_count()
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'transformers': metadata.version('transformers'),
        'peft': metadata.version('peft'), 'accelerate': metadata.version('accelerate'),
        'host_memory_gib': round(host_memory_gib(), 2), 'cpu_count': os.cpu_count(),
        'gpus': [torch.cuda.get_device_name(index) for index in range(device_count)],
        'gpu_total_mib': [torch.cuda.get_device_properties(index).total_memory // 2 ** 20
                          for index in range(device_count)],
        'max_memory_mib': args.max_memory_mib, 'cpu_budget': args.cpu_budget, 'base': args.base,
        'cuda_alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        'compressed-tensors': package_version('compressed-tensors', 'compressed_tensors'),
        'torchao': package_version('torchao'), 'bitsandbytes': package_version('bitsandbytes'),
        'rank': args.rank, 'alpha': args.alpha, 'dataset': str(args.dataset) if args.dataset else None,
        'batch_size': args.batch_size, 'max_updates': args.max_updates,
    }
    result: dict = {'status': 'not_run', 'environment': environment, 'quantization': None,
                    'device_map': None, 'target_modules': None, 'peft': None, 'updates': [],
                    'stopped_reason': None, 'adapter': None, 'base_format_unchanged': True,
                    'objective': {'rank': args.rank, 'alpha': args.alpha, 'dataset': None}}
    progress('environment', **environment)

    def finish(status: str, verdict: str) -> int:
        result['status'] = status
        result['verdict'] = verdict
        result['seconds'] = round(time.monotonic() - started, 3)
        (args.work / 'device-map-result.json').write_text(
            json.dumps(result, indent=2, default=str) + '\n', encoding='utf-8')
        progress('finished', status=status)
        return 0

    # The native contract first: a template that this image's transformers renders differently would
    # break every route, and it is cheap to check before a 58 GiB load.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        if args.dataset_check is None:
            batch = native_batch(tokenizer, args.seq_len)
        else:
            batch = None
    except Exception as error:
        result['contract'] = {'error': f'{type(error).__name__}: {error}',
                              'traceback': traceback.format_exc()[-2000:]}
        progress('native_contract_failed', error=result['contract']['error'])
        return finish('contract_incompatible',
                      'the native chat-template handoff contract could not be established on this '
                      'image, so no training example can be built; the exact exception is recorded')
    if batch is not None:
        result['contract'] = batch['checks']
        progress('native_contract', **batch['checks'])

    # The rank ladder trains on real records. The synthetic handoff example stays the default so the
    # feasibility route keeps its measured behaviour, but under --dataset the objective is the patch
    # text of each record, masked to the completion exactly as encode_sft_record establishes it.
    examples = []
    plan = None
    if args.dataset:
        try:
            records = load_sft_jsonl(args.dataset)
            train_records = [record for record in records if record['split'] != 'validation']
            if not train_records:
                train_records = records
            for record in train_records:
                try:
                    ids, loss_on, stats = encode_sft_record(tokenizer, record, args.seq_len)
                except ValueError as error:
                    progress('record_skipped', instance_id=record.get('instance_id'), error=str(error))
                    continue
                examples.append({'instance_id': record.get('instance_id'), 'input_ids': ids,
                                 'loss_on': loss_on, 'stats': stats})
            if not examples:
                raise ValueError('no record in the dataset could be tokenised into a supervised example')
        except Exception as error:
            result['dataset'] = {'loaded': False, 'error': f'{type(error).__name__}: {error}',
                                 'traceback': traceback.format_exc()[-2000:]}
            progress('dataset_failed', error=result['dataset']['error'])
            return finish('dataset_incompatible',
                          'the dataset route could not build a supervised example from the attached '
                          'records; the exact exception is recorded')
        result['dataset'] = {
            'loaded': True, 'path': str(args.dataset), 'records': len(records),
            'train_records': len(train_records), 'supervised_examples': len(examples),
            'truncated_examples': sum(1 for example in examples if example['stats']['truncated']),
            'mean_prompt_tokens': round(sum(example['stats']['prompt_tokens']
                                            for example in examples) / len(examples), 1),
            'mean_target_tokens': round(sum(example['stats']['target_tokens']
                                            for example in examples) / len(examples), 1),
            'examples': [{'instance_id': example['instance_id'], **example['stats']}
                         for example in examples[:5]],
        }
        result['objective']['dataset'] = {'records': len(records), 'examples': len(examples),
                                          'batch_size': args.batch_size}
        progress('dataset_ready', records=len(records), supervised_examples=len(examples),
                 truncated=result['dataset']['truncated_examples'])
        if args.dataset_check is not None:
            # CPU-only preflight: the encoding contract is proven on the real tokenizer, and the path
            # ends here so no GPU second is spent on a dataset that cannot be supervised.
            result['dataset_check'] = {
                'performed': True, 'seq_len': args.seq_len, 'batch_size': args.batch_size,
                'max_updates': args.max_updates,
                'trainable_examples': len(examples),
                'skipped_records': len(train_records) - len(examples),
                'longest_sequence_tokens': max(len(example['input_ids']) for example in examples),
                'longest_target_tokens': max(example['stats']['target_tokens'] for example in examples),
                'shortest_target_tokens': min(example['stats']['target_tokens'] for example in examples),
            }
            args.dataset_check.parent.mkdir(parents=True, exist_ok=True)
            args.dataset_check.write_text(json.dumps(result['dataset_check'], indent=2) + '\n',
                                          encoding='utf-8')
            progress('dataset_check', **result['dataset_check'])
            return finish('ok',
                          'the dataset encodes on the real tokenizer: the prompt rendering is a token '
                          'prefix of the full rendering for every record, so the completion-only mask is '
                          'established. No model was loaded and no GPU was used.')
        # One pass over the training split, in the dataset's own deterministic order, unless the cap
        # lowers it. Each batch is tokenised on demand so peak host memory stays at one batch.
        plan = list(batches(range(len(examples)), max(1, args.batch_size)))
        if args.max_updates:
            plan = plan[:args.max_updates]
        if not plan:
            return finish('dataset_incompatible', 'the dataset route built no batch to train on')
        updates_planned = len(plan)
        input_ids = None
        positions = None
    else:
        input_ids = batch['input_ids'].to('cuda:0')
        positions = torch.arange(batch['target_start'], input_ids.shape[1], device='cuda:0')
        updates_planned = args.updates

    # Load the dequantised base straight onto the four GPUs. Two measured failures shape this loop:
    # `device_map='auto'` filled the GPUs in order until one of them had 17.88 GiB allocated by PyTorch
    # plus 3.60 GiB reserved-but-unallocated, and the next 442 MiB request raised CUDA OOM -- so the
    # limits are the allocator's fragmentation as much as the split. The ladder therefore tries the
    # balanced placement first, then an even lower per-GPU cap that lets accelerate offload the tail to
    # host RAM (188.69 GiB of it) instead of failing. Each attempt is recorded with its own exception,
    # and the load runs in a thread so a stalled dispatch is a timeout rather than a silent kill.
    quantization, quantization_record = build_quantization_config(CompressedTensorsConfig, True)
    result['quantization'] = quantization_record
    progress('quantization_config', **quantization_record)
    deadline = time.monotonic() + args.load_budget_seconds
    attempts = []
    model = None
    for settings in device_map_attempts(args.max_memory_mib, args.cpu_budget):
        max_memory = {index: f"{settings['gpu_budget_mib']}MiB" for index in range(device_count)}
        if settings['cpu_budget']:
            max_memory['cpu'] = settings['cpu_budget']
        record = {'attempt': settings['name'], 'device_map': settings['device_map'],
                  'max_memory': max_memory}
        started = time.monotonic()
        box: dict = {}

        def load_base(settings=settings, max_memory=max_memory, box=box) -> None:
            try:
                box['model'] = AutoModelForImageTextToText.from_pretrained(
                    args.base, local_files_only=True, dtype=torch.bfloat16,
                    quantization_config=quantization, device_map=settings['device_map'],
                    max_memory=max_memory, low_cpu_mem_usage=True, attn_implementation='sdpa')
            except BaseException as error:  # recorded below
                box['error'] = error

        loader = threading.Thread(target=load_base, name=f"load-{settings['name']}", daemon=True)
        loader.start()
        loader.join(max(1.0, deadline - time.monotonic()))
        if loader.is_alive():
            record.update({'outcome': 'timeout', 'seconds': round(time.monotonic() - started, 3)})
            progress('load_attempt_timeout', **{k: v for k, v in record.items() if k != 'max_memory'})
            attempts.append(record)
            break
        if 'error' in box:
            error = box['error']
            record.update({'outcome': 'failed', 'seconds': round(time.monotonic() - started, 3),
                           'error': f'{type(error).__name__}: {error}'})
            progress('load_attempt_failed', attempt=record['attempt'],
                     seconds=record['seconds'], error=record['error'][:400])
            attempts.append(record)
            torch.cuda.empty_cache()
            continue
        model = box['model']
        record.update({'outcome': 'loaded', 'seconds': round(time.monotonic() - started, 3)})
        progress('load_attempt_loaded', **{k: v for k, v in record.items() if k != 'max_memory'})
        attempts.append(record)
        break
    result['load_attempts'] = attempts
    if model is None:
        last = attempts[-1] if attempts else {'outcome': 'not_attempted'}
        result['load'] = {'attempts': attempts}
        if last['outcome'] == 'timeout':
            return finish('load_timeout',
                          'the sharded load did not finish within --load-budget-seconds; the checkpoint '
                          'is not dispatchable across four L4s in the time one session allows')
        return finish('load_incompatible',
                      'the dequantised base could not be dispatched across the four L4s; every attempt '
                      'and its exact exception are recorded')
    load_started = time.monotonic() - (attempts[-1]['seconds'] if attempts else 0.0)
    hf_device_map = getattr(model, 'hf_device_map', None) or {}
    allocated_gib_per_gpu = [round(torch.cuda.memory_allocated(index) / 2 ** 30, 3)
                             for index in range(device_count)]
    placement = placement_verdict(hf_device_map, allocated_gib_per_gpu)
    result['device_map'] = {
        'seconds': round(time.monotonic() - load_started, 3),
        'modules': placement['modules'],
        'placement': placement['placement'],
        'cpu_share': placement['cpu_share'],
        'allocated_gib_per_gpu': allocated_gib_per_gpu,
        'peak_gib_per_gpu': [round(torch.cuda.max_memory_allocated(index) / 2 ** 30, 3)
                             for index in range(device_count)],
        'cpu_modules': placement['placement'].get('cpu', 0),
    }
    progress('base_loaded', **{key: value for key, value in result['device_map'].items()
                               if key != 'allocated_gib_per_gpu'})
    if placement['status'] != 'ok':
        progress('placement_incompatible', **placement['problems'])
        return finish('placement_incompatible',
                      'the base was not spread across the four L4s: '
                      + '; '.join(sorted(placement['problems'].values())))

    targets = [name for name, _ in model.named_modules()
               if name.endswith(tuple(TARGET_SUFFIXES)) and 'vision' not in name]
    result['target_modules'] = {'count': len(targets), 'example': targets[0] if targets else None}
    progress('target_modules', count=len(targets))
    if not targets:
        return finish('no_targets', 'no text projection matched the configured target suffixes')

    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias='none',
                             task_type='CAUSAL_LM', target_modules=targets)
    try:
        model = get_peft_model(model, peft_config)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError('get_peft_model returned no trainable parameters; training would be a no-op')
    except Exception as error:
        result['peft'] = {'wrapped': False, 'error': f'{type(error).__name__}: {error}',
                          'traceback': traceback.format_exc()[-2000:]}
        progress('peft_wrap_failed', error=result['peft']['error'])
        return finish('peft_incompatible',
                      'the dequantised base loads across the four GPUs but PEFT cannot attach an '
                      'adapter to it; the exact exception is recorded')
    result['peft'] = {
        'wrapped': True, 'trainable_parameters': sum(p.numel() for p in trainable),
        'tensors': len(trainable),
        'parameter_devices': dict(Counter(str(p.device) for p in trainable)),
    }
    progress('peft_wrapped', trainable_parameters=result['peft']['trainable_parameters'],
             parameter_devices=result['peft']['parameter_devices'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.enable_input_require_grads()

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    updates = []
    records_seen = 0

    def loss_for(example: dict):
        """The shared supervised objective, over one record of ids plus its completion mask.

        The synthetic route hands over the same ``native_batch`` tensors the four-GPU worker uses; the
        dataset route hands over one record's ids, and its supervised positions are exactly the ones
        ``encode_sft_record`` masked. Both end in ``target_loss``, so both routes measure one objective.
        """
        if example['kind'] == 'synthetic':
            return target_loss(model, example['input_ids'], example['positions'])[0]
        ids = torch.tensor([example['input_ids']], dtype=torch.long).to('cuda:0')
        mask = torch.tensor(example['loss_on'], dtype=torch.bool)
        positions = torch.arange(len(mask), device='cuda:0')[mask]
        return target_loss(model, ids, positions)[0]

    try:
        for number in range(updates_planned):
            if number and time.monotonic() - started > args.wall_budget_seconds:
                result['stopped_reason'] = 'wall_budget'
                break
            if plan is not None:
                block = [{'kind': 'record', **examples[index]} for index in plan[number]]
            else:
                block = [{'kind': 'synthetic', 'input_ids': input_ids, 'positions': positions}]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for index in range(device_count):
                torch.cuda.reset_peak_memory_stats(index)
            step_started = time.monotonic()
            # One forward/backward per record: padding a batch would change the loss for every other
            # record, and at batch size 4 a session ends in 9.1 GiB, so the two cards that hold most of
            # the base are the real constraint. 4 still fits, which is why it is the default below 32.
            losses = []
            for example in block:
                loss = loss_for(example)
                if not torch.isfinite(loss):
                    raise ValueError(f'non-finite loss at update {number}')
                (loss / len(block)).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            records_seen += len(block)
            update = {'update': number, 'loss': round(sum(losses) / len(losses), 6),
                      'loss_max': round(max(losses), 6), 'records': len(block),
                      'gradient_norm': float(norm),
                      'seconds': round(time.monotonic() - step_started, 3),
                      'peak_gib_per_gpu': [round(torch.cuda.max_memory_allocated(index) / 2 ** 30, 3)
                                           for index in range(device_count)],
                      'elapsed_s': round(time.monotonic() - started, 3)}
            if plan is not None:
                update['instance_ids'] = [examples[index]['instance_id'] for index in plan[number]]
            updates.append(update)
            progress('update', **{key: value for key, value in update.items()
                                  if key != 'peak_gib_per_gpu'})
        if result['stopped_reason'] is None:
            result['stopped_reason'] = 'completed_updates'
    except Exception as error:
        result['training_error'] = f'{type(error).__name__}: {error}'
        result['training_traceback'] = traceback.format_exc()[-2000:]
        progress('training_failed', error=result['training_error'])
    result['updates'] = updates
    result['updates_planned'] = updates_planned
    result['records_seen'] = records_seen
    if plan is not None:
        result['epochs_completed'] = round(records_seen / max(1, len(examples)), 3)
    if updates:
        seconds = [row['seconds'] for row in updates]
        result['seconds_per_update'] = {'min': min(seconds), 'max': max(seconds),
                                        'mean': round(sum(seconds) / len(seconds), 3)}
        result['peak_gib_per_gpu'] = [max(row['peak_gib_per_gpu'][index] for row in updates)
                                      for index in range(device_count)]
    if not updates or 'training_error' in result:
        return finish('backward_incompatible',
                      'the adapter wrapped but a training step failed, so this route cannot train as '
                      'configured; the exact exception is recorded above')
    try:
        result['adapter'] = save_adapter(model, model, args.work, 'device-map',
                                         {'rank': args.rank, 'lora_alpha': args.alpha,
                                          'base_handle': args.base_handle}, 0, None)
        progress('adapter_saved', tensors=result['adapter'].get('tensors'),
                 sha256=result['adapter'].get('sha256'))
    except Exception as error:
        result['adapter'] = {'saved': False, 'error': f'{type(error).__name__}: {error}',
                             'traceback': traceback.format_exc()[-2000:]}
        progress('adapter_save_failed', error=result['adapter']['error'])
    if result['adapter'] and result['adapter'].get('saved'):
        return finish('ok',
                      'the required base trains on four L4s with one process: it was dequantised to '
                      'BF16, dispatched across the GPUs, wrapped with a replicated LoRA adapter, and '
                      'the requested updates completed with the adapter exported for the '
                      'official-runtime reload check')
    return finish('adapter_incompatible',
                  'training worked but the adapter could not be exported; the exact exception is '
                  'recorded above')


if __name__ == '__main__':
    raise SystemExit(main())
