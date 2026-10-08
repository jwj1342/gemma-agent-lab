# Localization v1 large-sample gate study

> The [portable v1 study](2026-10-08-localization-v1-portable.md) corrects an important limitation:
> the archived ranker had already seen these 91 training tasks, so gate-only cross-validation below
> is not whole-system generalization evidence. The new study cross-fits both ranker and gate and
> obtains positive retrospective results with a different, content-based ranking model.

## Question and protocol

This study asks whether substantially more gate training can make the frozen Localization v1 hybrid
safe to activate over the full-tree BM25 fallback. It expands gate fitting from 18 development tasks
to 91 training tasks while preserving separate development and holdout checks.

- Baseline: full-tree BM25.
- Candidate: frozen v1 hybrid with BM25 `1.0`, exact evidence `0.4`, and legacy ranker `0.5`.
- Fit set: 91 scorable training tasks.
- Internal checks: five repository-stratified task folds and leave-one-repository-out validation.
- Frozen confirmation: 18 scorable development tasks, followed by 19 holdout tasks.
- Uncertainty: 5,000 paired repository-stratified bootstrap samples on development and holdout.
- Leakage boundary: reference paths are used only after ranking. Runtime gate features contain no
  reference path, patch, test patch, or other answer-derived value.

The holdout aggregate was inspected by earlier experiments, so it remains a retrospective veto set,
not a new independent confirmation. No rule or weight was selected from holdout results.

## Reproduction

```powershell
python scripts/train_localization_gate.py `
  --output runs/2026-10-08-localization-v1-large-learning/gate-study.json `
  --gate-output runs/2026-10-08-localization-v1-large-learning/gate-generated.json `
  --bootstrap-samples 5000
python -m unittest tests.test_localization_v1 tests.test_localization_gate -v
python scripts/build_submission.py `
  --agent-dir configs/experiments/localization-v1 `
  --output runs/2026-10-08-localization-v1-large-learning/submission-localization-v1-adaptive.zip
```

This is a standard-library CPU study. It does not run Gemma, use a GPU, upload to Kaggle, or create a
competition submission.

The resulting adaptive agent ZIP is 725,248 bytes (4,107,258 unpacked), with SHA256
`b9750171abfa09139b9f5ba80fcebd6d5dc5746c30bd0f3af4243e0696efaf16`.

## Learned rule

The training-selected two-condition rule enables v1 when:

1. BM25's normalized top-score margin is at most `0.36349847748766084`.
2. BM25 assigns a positive score to more than `0.12810649472405317` of candidate files.

The rule activates v1 on most observed tasks. Training cross-validation and repository holdouts still
show large point improvements, but the later holdout reverses the result.

| Evaluation | v1 enabled | BM25 Hit@1 | Gate Hit@1 | BM25 Hit@5 | Gate Hit@5 | BM25 MRR | Gate MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training fit | 86/91 | 23.1% | 52.7% | 54.9% | 78.0% | 0.384 | 0.640 |
| Five-fold out-of-fold | 84/91 | 23.1% | 47.3% | 54.9% | 73.6% | 0.384 | 0.595 |
| Leave-one-repository-out | 84/91 | 23.1% | 48.4% | 54.9% | 74.7% | 0.384 | 0.603 |
| Frozen development | 18/18 | 38.9% | 50.0% | 55.6% | 77.8% | 0.466 | 0.612 |
| Frozen holdout | 18/19 | 31.6% | 15.8% | 68.4% | 68.4% | 0.473 | 0.350 |

Development contains 10 v1 wins, 6 ties, and 2 losses. Holdout contains 6 wins, 3 ties, and 10 losses.
The additional fitting data therefore strengthens the apparent calibration result without fixing the
task-distribution reversal.

## Bootstrap evidence and deployment decision

| Split | Metric | Point delta | 95% bootstrap interval |
| --- | --- | ---: | ---: |
| Development | Hit@1 | +11.1 pp | [-11.1, +33.3] pp |
| Development | Hit@5 | +22.2 pp | [+5.6, +44.4] pp |
| Development | MRR | +0.147 | [+0.012, +0.294] |
| Holdout | Hit@1 | -15.8 pp | [-36.8, +5.3] pp |
| Holdout | Hit@5 | 0.0 pp | [-26.3, +26.3] pp |
| Holdout | MRR | -0.123 | [-0.285, +0.036] |

The development Hit@1 interval crosses zero, and the holdout point estimates regress Hit@1 and MRR.
The versioned gate therefore remains `enabled: false`. Default adaptive execution is exactly BM25 and
does not load the legacy checkpoint. This guarantees that the rejected learned rule cannot reduce the
runtime ranking relative to the current fallback.

## Conclusion

More fitting alone does not establish a safe v1 activation region. The current path-only legacy model
was originally trained on a closed-world changed-file inventory; its strong training behavior does not
transfer reliably across the fixed public split. Further iterations should add genuinely new repositories
or untouched tasks and train against full source trees. Reusing these 19 holdout labels for selection
would invalidate them as confirmation evidence.
