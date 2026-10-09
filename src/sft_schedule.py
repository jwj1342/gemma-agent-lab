"""Deterministic complete-epoch schedules shared by single-GPU and DDP training."""
import math
import random


def epoch_batches(size, world_size, accumulation, seed, epoch):
    if min(size, world_size, accumulation) < 1:
        raise ValueError('Positive dataset and batch dimensions are required')
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    batch_size = world_size * accumulation
    for start in range(0, size, batch_size):
        batch = indices[start:start + batch_size]
        # Zero-weight padding keeps collective counts equal without duplicating supervision.
        yield [(i, 1) for i in batch] + [(batch[-1], 0)] * (batch_size - len(batch))


def lr_factor(step, total_steps, warmup_fraction, minimum):
    warmup = max(1, math.ceil(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
