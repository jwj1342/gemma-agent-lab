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
    parser.add_argument('--wall-budget-seconds', type=float, default=900.0)
    parser.add_argument('--load-budget-seconds', type=float, default=600.0)
    parser.add_argument('--max-memory-mib', type=int, default=21000,
                        help='Per-GPU budget handed to accelerate; 21000 MiB stays inside 22563 MiB '
                             'with room for activations')
    args = parser.parse_args()

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
        'max_memory_mib': args.max_memory_mib, 'base': args.base,
        'compressed-tensors': package_version('compressed-tensors', 'compressed_tensors'),
        'torchao': package_version('torchao'), 'bitsandbytes': package_version('bitsandbytes'),
    }
    result: dict = {'status': 'not_run', 'environment': environment, 'quantization': None,
                    'device_map': None, 'target_modules': None, 'peft': None, 'updates': [],
                    'stopped_reason': None, 'adapter': None, 'base_format_unchanged': True}
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
        batch = native_batch(tokenizer, args.seq_len)
    except Exception as error:
        result['contract'] = {'error': f'{type(error).__name__}: {error}',
                              'traceback': traceback.format_exc()[-2000:]}
        progress('native_contract_failed', error=result['contract']['error'])
        return finish('contract_incompatible',
                      'the native chat-template handoff contract could not be established on this '
                      'image, so no training example can be built; the exact exception is recorded')
    result['contract'] = batch['checks']
    progress('native_contract', **batch['checks'])
    input_ids = batch['input_ids'].to('cuda:0')
    positions = torch.arange(batch['target_start'], input_ids.shape[1], device='cuda:0')

    # Load the dequantised base straight onto the four GPUs. The load runs in a thread so a stalled
    # dispatch is recorded as a timeout instead of letting the notebook's subprocess timeout kill the
    # process with no measurement at all.
    quantization, quantization_record = build_quantization_config(CompressedTensorsConfig, True)
    result['quantization'] = quantization_record
    progress('quantization_config', **quantization_record)
    max_memory = {index: f'{args.max_memory_mib}MiB' for index in range(device_count)}
    load_started = time.monotonic()
    box: dict = {}

    def load_base() -> None:
        try:
            box['model'] = AutoModelForImageTextToText.from_pretrained(
                args.base, local_files_only=True, dtype=torch.bfloat16,
                quantization_config=quantization, device_map='auto', max_memory=max_memory,
                low_cpu_mem_usage=True, attn_implementation='sdpa')
        except BaseException as error:  # re-raised on the main thread below
            box['error'] = error

    loader = threading.Thread(target=load_base, name='device-map-load', daemon=True)
    loader.start()
    loader.join(args.load_budget_seconds)
    if loader.is_alive():
        progress('load_timeout', budget_seconds=args.load_budget_seconds)
        return finish('load_timeout',
                      'the sharded load did not finish within --load-budget-seconds; the checkpoint is '
                      'not dispatchable across four L4s in the time one session allows')
    if 'error' in box:
        error = box['error']
        result['load'] = {'error': f'{type(error).__name__}: {error}',
                          'traceback': traceback.format_exc()[-2000:],
                          'seconds': round(time.monotonic() - load_started, 3)}
        progress('load_failed', error=result['load']['error'])
        return finish('load_incompatible',
                      'the dequantised base could not be dispatched across the four L4s; the exact '
                      'exception is recorded')
    model = box['model']
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
    try:
        for number in range(args.updates):
            if number and time.monotonic() - started > args.wall_budget_seconds:
                result['stopped_reason'] = 'wall_budget'
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for index in range(device_count):
                torch.cuda.reset_peak_memory_stats(index)
            step_started = time.monotonic()
            loss, _ = target_loss(model, input_ids, positions)
            if not torch.isfinite(loss):
                raise ValueError(f'non-finite loss at update {number}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            update = {'update': number, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                      'seconds': round(time.monotonic() - step_started, 3),
                      'peak_gib_per_gpu': [round(torch.cuda.max_memory_allocated(index) / 2 ** 30, 3)
                                           for index in range(device_count)],
                      'elapsed_s': round(time.monotonic() - started, 3)}
            updates.append(update)
            progress('update', **update)
        if result['stopped_reason'] is None:
            result['stopped_reason'] = 'completed_updates'
    except Exception as error:
        result['training_error'] = f'{type(error).__name__}: {error}'
        result['training_traceback'] = traceback.format_exc()[-2000:]
        progress('training_failed', error=result['training_error'])
    result['updates'] = updates
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
