# Localization v1 adaptive gate study

> Superseded for deployment evidence by the
> [large-sample gate study](2026-10-08-localization-v1-large-learning.md), which fits on 91 training
> tasks and reserves development for frozen confirmation. This report remains the initial
> development-versus-holdout diagnosis.

## Question and comparison

- Hypothesis: label-free confidence and ranking-agreement features can identify issues where the
  frozen v1 hybrid improves over the promoted BM25 localizer, while retaining BM25 elsewhere.
- Baseline: full-tree BM25. Candidate: the already-frozen v1 hybrid with weights BM25 `1.0`, exact
  `0.4`, and legacy `0.5`.
- Learner: a dependency-free decision rule with at most two conjunctive thresholds. It sees only
  runtime-available features such as candidate count, issue length, score margins, positive-score
  fractions, exact evidence, and BM25/v1 top-k agreement.
- Data policy: gold paths select the rule on the 19-task development split and score the 19-task
  holdout only after the rule is frozen. One development task changes only a new file and is excluded,
  leaving 18 scorable development and 19 scorable holdout tasks.

The aggregate holdout results had already been inspected during the preceding full-tree experiment.
This is therefore a retrospective distribution-shift study, not a new independent confirmation.

## Reproduction

```powershell
python scripts/train_localization_gate.py `
  --output runs/2026-10-08-localization-v1-gate/gate-study-final.json `
  --gate-output runs/2026-10-08-localization-v1-gate/gate-generated.json
python -m unittest tests.test_localization_v1 tests.test_localization_gate -v
python scripts/build_submission.py `
  --agent-dir configs/experiments/localization-v1 `
  --output runs/2026-10-08-localization-v1-gate/submission-localization-v1-adaptive.zip
```

- Split seed: 52.
- Source trees: each task's exact base commit, excluding test Python files.
- Runtime implementation and gate artifact:
  `configs/experiments/localization-v1/skills/localization-v1/`.
- Adaptive agent ZIP: 725,149 compressed bytes, 4,106,980 unpacked bytes, SHA256
  `fbf74428030892b416fe23b2f56b6fa1dcbc892c4494c1ab4ac3728cef5b75c0`.
- No third-party ML package, GPU, model inference, reference-patch input, Kaggle upload, or
  competition submission is required.

## Development versus holdout

The candidate's task-level behavior reverses between the two splits.

| Split | v1 wins | Ties | v1 losses |
| --- | ---: | ---: | ---: |
| Development | 10 | 6 | 2 |
| Holdout | 6 | 3 | 10 |

The largest label-free feature shifts are shown below. Standardized mean difference is holdout minus
development in pooled-standard-deviation units.

| Feature | Dev mean | Holdout mean | Standardized difference |
| --- | ---: | ---: | ---: |
| BM25 margin | 0.110 | 0.162 | +0.501 |
| Legacy positive fraction | 0.511 | 0.668 | +0.488 |
| BM25 top result's reciprocal v1 rank | 0.691 | 0.509 | -0.477 |
| v1 top result's reciprocal BM25 rank | 0.675 | 0.497 | -0.465 |
| Top-1 agreement | 0.556 | 0.368 | -0.382 |

Repository grouping does not explain the reversal by itself. FastAPI changes from 6 wins / 0 losses
on development to 2 wins / 6 losses on holdout; Requests changes from 0 losses to 2 losses, while
Rich remains 4 wins / 2 losses on both splits. This is within-repository task shift, not merely a
different repository mixture.

## Learned gate and safety decision

The development-selected rule enables v1 when both conditions hold:

1. BM25 gives a positive score to at most `0.9879032258064516` of candidate files.
2. The strongest exact-evidence score is at most `11.75`.

| Policy | Split | v1 enabled | Hit@1 | Hit@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 | Development | 0/18 | 38.9% | 55.6% | 0.466 |
| Fitted gate | Development | 12/18 | 55.6% | 77.8% | 0.641 |
| Leave-one-task-out gate | Development | 15/18 | 50.0% | 77.8% | 0.612 |
| Leave-one-repository-out gate | Development | 11/18 | 50.0% | 72.2% | 0.589 |
| BM25 | Holdout | 0/19 | 31.6% | 68.4% | 0.473 |
| Frozen gate | Holdout | 10/19 | 26.3% | 63.2% | 0.440 |

The frozen gate fails all three primary holdout metrics. `resources/gate.json` therefore records the
learned rule and evidence but sets `enabled: false`. The skill's default `adaptive` mode reads this
artifact and falls back to BM25 without loading the 4 MB checkpoint. If a future independently tested
gate passes development fit, leave-one-task-out, leave-one-repository-out, and holdout checks, the same
runtime can selectively enable v1 without code changes.

## Conclusion and next step

The program successfully distinguishes “a rule can be learned” from “the rule is safe to deploy.”
Current evidence does not support enabling v1 on any automatically predicted subset. The useful result
is the fail-safe adaptive infrastructure plus a quantified diagnosis of the split shift.

Collect a larger grouped calibration set from additional repositories or untouched tasks before
retraining. Keep this holdout retrospective; do not repeatedly optimize against its 19 labels and call
the result independent. End-to-end agent resolution still requires a matched Kaggle or H100 A/B.
