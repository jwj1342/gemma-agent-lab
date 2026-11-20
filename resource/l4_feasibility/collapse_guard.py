"""Judge a trained LoRA arm from its own loss trajectory before the evaluation budget is spent.

The screening round measured both failure modes of the training recipe and their prices
(``docs/experiments/2026-10-18-lora-improvement-program.md`` sections 12 and 14):

* **Flat** - at ``lr 2e-5`` the eight AdamW updates moved the loss 2.0966 -> 2.0989 and the exported
  adapter's weight displacement is RMS(dW) = 4.6e-6 against base scales of 1e-2. A byte-identical re-run
  of the baseline resolved exactly the same tasks as that arm (0 discordant), so a flat arm costs a full
  24-task evaluation (about 22,000 quota-seconds) to reproduce the baseline.
* **Collapsed** - at ``lr 2e-4`` the loss fits the targets (2.10 -> 0.73) and serving the adapter sends the
  agent into a decode loop: 0 of 24 tasks resolved, every task with no tool call and the session timeout,
  which cost 29,935 quota-seconds - the most expensive arm of the programme.

These two observed trajectories motivate a conservative budget heuristic, not a proof that training loss
predicts repair quality. This module turns the loss trajectory into a screening verdict.
It is a pure function: no torch, no GPU, no filesystem, so the repository test
suite can exercise the decision rule offline.

The caller decides what a non-``proceed`` verdict costs; in the evaluation kernel it stops the arm and
records the trajectory instead of spending the remaining budget.
"""
from __future__ import annotations
import math

PROCEED = 'proceed'
FLAT = 'flat'
COLLAPSED = 'collapsed'
INVALID = 'invalid'


def collapse_guard(result: dict, loss_floor: float = 1.0, flat_ratio: float = 0.005) -> dict:
    """Return the guard record for one training result.

    ``result`` is ``device-map-result.json`` as written by ``train_l4_device_map.py``; only ``updates``
    and ``training_error`` are read. Parsed metrics and thresholds stay in the returned dict.
    Missing, malformed, or non-finite metrics stop evaluation with an ``invalid`` verdict.

    ``loss_floor`` > 0 marks an arm whose minimum per-update loss falls below it as ``collapsed``; the
    measured separation is wide (collapsed 0.73, flat 2.10), so 1.0 is not a tuned constant.
    ``flat_ratio`` > 0 marks an arm as ``flat`` when the loss does not move by this fraction of its first
    value over the whole run; the measured flat arm moved 1.1e-3.
    """
    if not all(math.isfinite(value) and value >= 0 for value in (loss_floor, flat_ratio)):
        raise ValueError('guard thresholds must be finite and nonnegative')
    rows = result.get('updates') or []
    losses, gradients = [], []
    invalid = None
    for row in rows:
        try:
            loss = float(row['loss'])
            gradient = row.get('gradient_norm')
            gradient = float(gradient) if gradient is not None else None
            if not math.isfinite(loss) or loss < 0 or (
                    gradient is not None and not math.isfinite(gradient)):
                raise ValueError('non-finite or negative training metric')
            losses.append(loss)
            if gradient is not None:
                gradients.append(gradient)
        except (TypeError, KeyError, ValueError, AttributeError):
            invalid = 'invalid_training_metrics'
    record = {
        'verdict': PROCEED, 'reason': 'two_or_more_updates', 'updates': len(losses),
        'losses': losses, 'first_loss': losses[0] if losses else None,
        'last_loss': losses[-1] if losses else None,
        'min_loss': min(losses) if losses else None,
        'movement': None, 'relative_movement': None,
        'max_gradient_norm': max(gradients) if gradients else None,
        'loss_floor': loss_floor, 'flat_ratio': flat_ratio,
    }
    if invalid or result.get('training_error'):
        record['verdict'] = INVALID
        record['reason'] = invalid or 'training_error'
        return record
    if len(losses) < 2:
        record['verdict'] = INVALID
        record['reason'] = 'fewer_than_two_updates'
        return record
    first = losses[0]
    record['movement'] = round(losses[-1] - first, 6)
    span = max(abs(loss - first) for loss in losses)
    record['relative_movement'] = round(span / first, 8) if first > 0 else None
    if loss_floor > 0 and record['min_loss'] is not None and record['min_loss'] < loss_floor:
        record['verdict'] = COLLAPSED
        record['reason'] = 'min_loss_below_floor'
    elif flat_ratio > 0 and record['relative_movement'] is not None \
            and record['relative_movement'] < flat_ratio:
        record['verdict'] = FLAT
        record['reason'] = 'loss_did_not_move'
    return record
