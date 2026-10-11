"""Deterministic complete-epoch schedules shared by single-GPU and DDP training."""
import math
import random


def loss_weights(token_counts, example_weights, normalization='token'):
    """Return global-batch coefficients before DDP averages rank gradients."""
    if normalization not in ('token', 'example'):
        raise ValueError('Unknown loss normalization')
    if (not token_counts or len(token_counts) != len(example_weights)
            or any(n <= 0 for n in token_counts) or any(w < 0 for w in example_weights)):
        raise ValueError('Invalid token counts or example weights')
    masses = [w * (n if normalization == 'token' else 1)
              for n, w in zip(token_counts, example_weights)]
    denominator = sum(masses)
    if denominator <= 0:
        raise ValueError('At least one real example is required')
    return [mass / denominator for mass in masses]


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
