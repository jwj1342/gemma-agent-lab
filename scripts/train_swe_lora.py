"""Train complete shuffled epochs with token-normalized DDP and resumable checkpoints.

Use torchrun in a Slurm allocation. Inputs are trusted local encoded tensors; never
load arbitrary third-party pickle training states. Final exports contain safetensors only.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from src.sft_schedule import epoch_batches, lr_factor


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--probe', action='store_true', help='Exercise the longest training example before training')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training requires a Slurm allocation')
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoModelForImageTextToText, CompressedTensorsConfig
    from transformers.utils.logging import disable_progress_bar
    from peft import LoraConfig, PeftModel, get_peft_model
    disable_progress_bar()
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    random.seed(config['seed'] + rank)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    if rank == 0:
        if args.output.exists() and not args.resume:
            raise ValueError('Use a fresh run directory or explicit resume')
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    started = time.monotonic()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    train = torch.load(args.data / 'train.pt', weights_only=True)
    validation = torch.load(args.data / 'validation.pt', weights_only=True)
    for split in ('train', 'validation'):
        if digest(args.data / (split + '.pt')) != manifest['encoded_files_sha256'][split]:
            raise ValueError('Encoded corpus hash changed')
    fixed_val = {kind: sorted([s for s in validation if s['kind'] == kind],
                             key=lambda s: s['sample_id'])[:config['validation_actions']]
                 for kind in ('swe', 'graph', 'terminal')}
    steps_per_epoch = math.ceil(len(train) / (world * config['gradient_accumulation']))
    total_steps = config['epochs'] * steps_per_epoch
    contract = {'config': config, 'data_manifest_sha256': digest(args.data / 'manifest.json'),
                'world_size': world, 'base_config_sha256': digest(args.base / 'config.json'),
                'base_weights_sha256': digest(args.base / 'model.safetensors'),
                'source_sha256': {str(x): digest(x) for x in
                                 (Path('scripts/train_swe_lora.py'), Path('src/sft_schedule.py'))}}
    if args.resume:
        old = json.loads((args.output / 'run-manifest.json').read_text())
        if old['contract'] != contract:
            raise ValueError('Resume contract changed')
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, local_files_only=True, dtype=torch.bfloat16, device_map={'': str(device)},
        quantization_config=CompressedTensorsConfig(dequantize=True), attn_implementation='sdpa')
    targets = [name for name, _ in model.named_modules()
               if name.endswith(tuple(config['target_suffixes'])) and 'vision' not in name]
    if not targets:
        raise ValueError('No matching text projections')
    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(r=config['rank'], lora_alpha=config['lora_alpha'],
            target_modules=targets, lora_dropout=0, bias='none', task_type='CAUSAL_LM',
            use_rslora=False, use_dora=False))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    parameters = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    step, best, completed_samples, completed_tokens = 0, float('inf'), 0, 0
    if args.resume:
        state = torch.load(args.resume / 'training_state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        rng = state['rng'][rank]
        random.setstate(rng['python'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'], device=device)
        step, best = state['step'], state['best_swe_nll']
        completed_samples, completed_tokens = state['samples'], state['tokens']
        if rank == 0:
            (args.output / 'resume-check.json').write_text(json.dumps({'restored_step': step,
                'optimizer_and_rng_restored': True, 'contract_matches': True}, indent=2) + '\n')
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                     find_unused_parameters=True) if world > 1 else model
    if rank == 0 and not args.resume:
        record = {'contract': contract, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'modules': os.environ.get('LOADEDMODULES', ''),
            'dependencies': {n: importlib.metadata.version(n) for n in
                             ('torch', 'transformers', 'peft', 'compressed-tensors', 'accelerate')},
            'trainable_parameters': sum(x.numel() for x in parameters),
            'target_modules': targets, 'total_steps': total_steps, 'steps_per_epoch': steps_per_epoch,
            'train_actions': len(train), 'validation_actions': {k: len(v) for k, v in fixed_val.items()},
            'validation_sample_ids': {k: [s['sample_id'] for s in v] for k, v in fixed_val.items()}}
        (args.output / 'run-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k not in ('contract', 'target_modules', 'validation_sample_ids')}), flush=True)

    def loss_for(sample, network):
        ids = sample['input_ids'].to(device=device, dtype=torch.long).unsqueeze(0)
        positions = torch.arange(sample['target_start'], ids.shape[-1], device=device)
        output = network(input_ids=ids, attention_mask=torch.ones_like(ids),
                         logits_to_keep=positions - 1, use_cache=False)
        loss = F.cross_entropy(output.logits[0].float(), ids[0, positions])
        return loss, len(positions)

    def evaluate():
        model.eval()
        scores = {}
        with torch.no_grad():
            for kind, values in fixed_val.items():
                sums = torch.zeros(3, device=device, dtype=torch.float64)
                for sample in values[rank::world]:
                    loss, count = loss_for(sample, model)
                    sums += torch.tensor([float(loss) * count, count, 1], device=device, dtype=torch.float64)
                if world > 1:
                    dist.all_reduce(sums)
                scores[kind] = {'assistant_nll': float(sums[0] / sums[1]),
                                'tokens': int(sums[1]), 'actions': int(sums[2])}
        if rank == 0:
            with (args.output / 'validation.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, 'epoch': step / steps_per_epoch, 'scores': scores}) + '\n')
            print('validation', step, json.dumps(scores), flush=True)
        return scores

    def save_checkpoint(scores):
        nonlocal best
        better = scores['swe']['assistant_nll'] < best
        if better:
            best = scores['swe']['assistant_nll']
        rng = {'python': random.getstate(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        if rank == 0:
            dest = args.output / f'checkpoint-{step:05d}'
            temporary = args.output / f'.checkpoint-{step:05d}'
            temporary.mkdir(exist_ok=True)
            model.save_pretrained(temporary, safe_serialization=True)
            torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'rng': states,
                        'best_swe_nll': best, 'samples': completed_samples, 'tokens': completed_tokens},
                       temporary / 'training_state.pt')
            (temporary / 'validation.json').write_text(json.dumps(scores, indent=2) + '\n')
            temporary.rename(dest)
            (args.output / 'latest.json').write_text(json.dumps({'checkpoint': dest.name, 'step': step}) + '\n')
            if better:
                (args.output / 'best.json').write_text(json.dumps({'checkpoint': dest.name, 'step': step, 'swe_nll': best}) + '\n')
            # Retain the current resumable state and the selected checkpoint; discard only this run's older states.
            keep = {dest.name, json.loads((args.output / 'best.json').read_text())['checkpoint']}
            for previous in args.output.glob('checkpoint-*'):
                if previous.is_dir() and previous.name not in keep:
                    shutil.rmtree(previous)
        if world > 1:
            dist.barrier()

    if args.probe and step == 0:
        longest = max(train, key=lambda s: len(s['input_ids']))
        model.train()
        loss, _ = loss_for(longest, wrapped)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print('Longest-example backward passed', len(longest['input_ids']),
                  'tokens;', torch.cuda.max_memory_allocated() / 2**30, 'GiB', flush=True)
    if step == 0:
        evaluate()
    stop = min(total_steps, args.stop_after or total_steps)
    for epoch in range(config['epochs']):
        for batch_no, batch in enumerate(epoch_batches(len(train), world, config['gradient_accumulation'], config['seed'], epoch)):
            scheduled_step = epoch * steps_per_epoch + batch_no
            if scheduled_step < step:
                continue
            if step >= stop:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = config['learning_rate'] * lr_factor(step, total_steps, config['warmup_fraction'], config['minimum_lr_ratio'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            batch_tokens = sum((len(train[i]['input_ids']) - train[i]['target_start']) * weight for i, weight in batch)
            losses = []
            for micro, (index, weight) in enumerate(batch[rank::world]):
                sample = train[index]
                sync = wrapped.no_sync() if world > 1 and micro + 1 < config['gradient_accumulation'] else nullcontext()
                with sync:
                    loss, count = loss_for(sample, wrapped)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    (loss * count * weight * world / batch_tokens).backward()
                losses.append({'sample_id': sample['sample_id'], 'kind': sample['kind'],
                    'group': sample['metadata']['group'], 'loss': float(loss.detach()),
                    'tokens': count * weight, 'weight': weight})
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError('Non-finite or zero gradient')
            optimizer.step()
            step += 1
            completed_samples += sum(w for _, w in batch)
            completed_tokens += batch_tokens
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, losses)
            else:
                gathered[0] = losses
            if rank == 0:
                rows = [x for rank_rows in gathered for x in rank_rows]
                metric = {'step': step, 'epoch': epoch + (batch_no + 1) / steps_per_epoch,
                    'loss': sum(x['loss'] * x['tokens'] for x in rows) / batch_tokens,
                    'learning_rate': lr, 'gradient_norm': float(norm), 'samples_seen': completed_samples,
                    'assistant_tokens_seen': completed_tokens, 'microsteps': rows,
                    'peak_gib': torch.cuda.max_memory_allocated() / 2**30, 'elapsed_s': time.monotonic() - started}
                with (args.output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(metric) + '\n')
                print(json.dumps({k: v for k, v in metric.items() if k != 'microsteps'}), flush=True)
            if step % config['checkpoint_every'] == 0 or step % steps_per_epoch == 0 or step == stop:
                save_checkpoint(evaluate())
        if step >= stop:
            break
    if rank == 0 and step == total_steps:
        selected = json.loads((args.output / 'best.json').read_text())
        export = args.output / 'adapter'
        export.mkdir()
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            shutil.copy2(args.output / selected['checkpoint'] / name, export / name)
        (args.output / 'completed.json').write_text(json.dumps({'steps': step, 'epochs': config['epochs'],
            'samples_seen': completed_samples, 'assistant_tokens_seen': completed_tokens,
            'expected_samples': len(train) * config['epochs'], 'selected': selected,
            'export_sha256': {p.name: digest(p) for p in export.iterdir()},
            'claim': 'Completed SFT; issue-resolution gain remains unproven until evaluated.'}, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
