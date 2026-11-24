"""Probe whether the required base can be trained in its shipped packed format on one L4.

This runs **first**, before any four-GPU attempt, because it is the cheap decisive measurement:

* it loads the checkpoint as shipped, i.e. with dequantisation explicitly not requested
  (``build_quantization_config(..., False)``, which picks the keyword this transformers version
  understands), which is about 16-17 GiB instead of the ~58 GiB a BF16 replica needs, so it fits one L4;
* it uses one process, so it needs roughly one replica of host RAM instead of four;
* the fact that decides whether it is trainable at all -- the class of the projection modules PEFT
  would have to wrap -- is available seconds after the load.

The base is **not** replaced or re-quantised. Loading the required checkpoint in the format the
competition serves it in is a stricter reading of "preserve the required base" than dequantising it to
BF16 for training and serving the packed weights, and the notebook's last phase still reloads any
adapter it produces in the official runtime. An NF4 re-quantisation is a different base model and is
not what this probe does.

Every incompatibility is recorded as a verdict with the exact exception rather than raised, so one
session yields an answer whichever way it goes:

    python probe_packed_base.py --base /kaggle/input/models/... --work /kaggle/working/l4-feasibility

It imports the shared contract and export helpers from ``train_l4_feasibility``, which the notebook
stages beside it in the same directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_l4_feasibility import (  # noqa: E402  (staged beside this file at run time)
    TARGET_SUFFIXES, host_memory_gib, native_batch, save_adapter, target_loss)
from quantization_config_compat import build_quantization_config  # noqa: E402

CHECKPOINT_FILES = ('config.json', 'model.safetensors', 'model.safetensors.index.json',
                    'quantization_config.json', 'generation_config.json')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package_version(*names: str):
    """The wheel is ``compressed_tensors`` but the import name is ``compressed_tensors`` too, while
    the requirement text and PyPI both spell it ``compressed-tensors``; try both rather than record a
    misleading None."""
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


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
    parser.add_argument('--wall-budget-seconds', type=float, default=420.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, CompressedTensorsConfig
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()
    args.work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device('cuda', 0)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    def progress(phase: str, **fields) -> None:
        record = {'phase': phase, 'elapsed_s': round(time.monotonic() - started, 3), **fields}
        with (args.work / 'packed-progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
        print(f'[packed] {phase} {json.dumps(fields, default=str)}', flush=True)

    environment = {
        'host_memory_gib': round(host_memory_gib(), 2), 'cpu_count': os.cpu_count(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'transformers': metadata.version('transformers'),
        'peft': metadata.version('peft'),
        'gpu': torch.cuda.get_device_name(0),
        'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory // 2 ** 20,
        'dequantize': False,
        'base': args.base,
    }
    for package in (('compressed-tensors', 'compressed_tensors'), 'torchao', 'bitsandbytes'):
        if isinstance(package, tuple):
            environment[package[0]] = package_version(*package)
        else:
            environment[package] = package_version(package)
    base_path = Path(args.base)
    environment['checkpoint_files'] = {
        name: digest(base_path / name) for name in CHECKPOINT_FILES if (base_path / name).is_file()}
    result = {'status': 'not_run', 'environment': environment,
              'base_format_unchanged': True, 'updates': [], 'adapter': None,
              'stopped_reason': None}
    progress('environment', **{k: v for k, v in environment.items() if k != 'checkpoint_files'})

    def finish(status: str, verdict: str) -> int:
        result['status'] = status
        result['verdict'] = verdict
        result['seconds'] = round(time.monotonic() - started, 3)
        (args.work / 'packed-probe-result.json').write_text(
            json.dumps(result, indent=2, default=str) + '\n', encoding='utf-8')
        progress('finished', status=status)
        return 0

    # The native contract is validated here too, but a template that this image's transformers build
    # renders differently must not abort the session with a bare traceback: it is recorded as this
    # probe's verdict, because it is the first thing that would break in any case.
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
    input_ids = batch['input_ids'].to(device)
    positions = torch.arange(batch['target_start'], input_ids.shape[1], device=device)

    # Stage 1: load the checkpoint in its shipped, packed format.
    load_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        # As shipped means "do not dequantise"; the keyword that expresses that differs by transformers
        # version (`run_compressed` in 5.13.1, `dequantize` in 5.18/5.19) and the wrong one is dropped
        # silently, so it is built through the shared helper and recorded.
        quantization, quantization_record = build_quantization_config(CompressedTensorsConfig, False)
        result['quantization'] = quantization_record
        progress('quantization_config', **quantization_record)
        model = AutoModelForImageTextToText.from_pretrained(
            args.base, local_files_only=True, dtype=torch.bfloat16, quantization_config=quantization,
            attn_implementation='sdpa')
    except Exception as error:
        result['load'] = {'error': f'{type(error).__name__}: {error}',
                          'traceback': traceback.format_exc()[-2000:],
                          'seconds': round(time.monotonic() - load_started, 3)}
        progress('packed_load_failed', error=result['load']['error'])
        return finish('load_incompatible',
                      'the packed checkpoint could not be loaded in its shipped format, which rules '
                      'this path out before any training is attempted')
    result['load'] = {
        'seconds': round(time.monotonic() - load_started, 3),
        'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3),
        'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 3),
    }
    progress('packed_loaded', **result['load'])

    # Stage 2: the decisive fact -- what class are the modules PEFT would have to wrap?
    targets = [name for name, _ in model.named_modules()
               if name.endswith(tuple(TARGET_SUFFIXES)) and 'vision' not in name]
    classes = Counter(type(model.get_submodule(name)).__name__ for name in targets)
    linear = [name for name in targets if isinstance(model.get_submodule(name), torch.nn.Linear)]
    result['target_modules'] = {
        'count': len(targets), 'classes': dict(classes),
        'nn_linear_count': len(linear), 'nn_linear_fraction': round(len(linear) / len(targets), 4)
        if targets else None,
        'example': targets[0] if targets else None,
    }
    progress('target_modules', **{k: v for k, v in result['target_modules'].items() if k != 'example'})
    if not targets:
        return finish('no_targets', 'no text projection matched the configured target suffixes')

    # Stage 3: can PEFT attach an adapter to those modules?
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
                      'the packed format loads but PEFT cannot attach an adapter to its modules, so '
                      'this path cannot train without a different adapter mechanism')
    result['peft'] = {'wrapped': True, 'trainable_parameters': sum(p.numel() for p in trainable),
                      'tensors': len(trainable)}
    progress('peft_wrapped', trainable_parameters=result['peft']['trainable_parameters'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.enable_input_require_grads()

    # Stage 4: one backward step, then at most --updates optimizer updates.
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    updates = []
    try:
        for number in range(args.updates):
            if number and time.monotonic() - started > args.wall_budget_seconds:
                result['stopped_reason'] = 'wall_budget'
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.monotonic()
            loss, _ = target_loss(model, input_ids, positions)
            if not torch.isfinite(loss):
                raise ValueError(f'non-finite loss at update {number}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            update = {'update': number, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                      'seconds': round(time.monotonic() - step_started, 3),
                      'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3),
                      'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 3),
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
        result['peak_allocated_gib'] = max(row['peak_allocated_gib'] for row in updates)
        result['peak_reserved_gib'] = max(row['peak_reserved_gib'] for row in updates)
    if not updates or 'training_error' in result:
        return finish('backward_incompatible',
                      'the adapter wrapped but a training step failed, so this path cannot train as '
                      'configured; the exact exception is recorded above')
    try:
        result['adapter'] = save_adapter(model, model, args.work, 'packed',
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
                      'the packed checkpoint trains on one L4: the adapter wrapped, a backward step '
                      'and the requested updates completed, and the adapter was exported for the '
                      'official-runtime reload check')
    return finish('adapter_incompatible',
                  'training worked but the adapter could not be exported; the exact exception is '
                  'recorded above')


if __name__ == '__main__':
    raise SystemExit(main())
