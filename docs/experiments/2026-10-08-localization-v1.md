# Localization v1 full-tree integration

## Question and comparison

- Hypothesis: a bounded, repository-local skill can make the archived file localizer usable by the
  current agent framework without exposing reference patches or depending on platform packages.
- Baseline and proposed change: compare the archived hashed pairwise checkpoint (`legacy`) with
  full-tree BM25, then test exact-score and legacy-score fusion. Attach the promoted policy directly
  to the root agent and retain the existing analyzer as a fallback.
- Tasks and split: 129 public Python repair tasks from FastAPI, Rich, Requests, and HTTPX, using the
  original deterministic repository-stratified 91/19/19 train/development/test split (seed 52).
  Every task is ranked against the complete non-test Python tree at its own base commit, from 21--789
  files depending on repository and revision.
- Success metric: held-out file Hit@1/3/5/10 and MRR. Reference patches are read only after ranking.

## Reproduction

- Source revision: `f1a5a9dec7bbe6cb1c89deff674d65bffa156288` plus this report's working-tree
  changes on `feature/localization-v1-skill`.
- Configuration: `configs/experiments/localization-v1/`.
- Public task SHA256: `e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6`.
- Archived checkpoint SHA256: `56bec1d839b79ecaf9374a7f6da1e5cb496870e87c51750d2284d10f569cfdbc`;
  byte-identical to the protected `artifacts/localization_v3_3k/model.json` backup.
- Commands:

  ```powershell
  python scripts/benchmark_localization_v1.py `
    --output runs/2026-10-08-localization-v1/full-tree-benchmark-final.json
  python -m unittest tests.test_localization_v1 -v
  python scripts/build_submission.py `
    --agent-dir configs/experiments/localization-v1 `
    --output runs/2026-10-08-localization-v1/submission-localization-v1.zip
  ```

- Environment: Windows laptop, Python 3.14.6, standard library for ranking; no GPU or model
  inference. Benchmark seed 52. Agent sampling and evaluation budgets remain byte-for-byte equal to
  `configs/baseline/`.
- Agent ZIP at the time of this experiment: SHA256
  `ed8a6df2da228a4d0b946027c555c45d6379d663e0c299a4a675f903b1989ed7`, 722,599 compressed
  bytes and 4,097,771 unpacked bytes. The later adaptive-gate study supersedes this package while
  retaining BM25 as its safety fallback.

## Results

The development-selected hybrid looked strong on development data but did not generalize. The
exact-only safe fusion also regressed the primary held-out metrics, so neither is the runtime default.

| Full-tree test policy | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived legacy checkpoint | 21.1% | 36.8% | 42.1% | 52.6% | 0.306 | 83.16 |
| Development-selected hybrid | 15.8% | 42.1% | 68.4% | 78.9% | 0.350 | 6.53 |
| BM25 + exact, legacy disabled | 26.3% | 36.8% | 57.9% | 84.2% | 0.402 | 6.42 |
| **Promoted default: full-tree BM25** | **31.6%** | **47.4%** | **68.4%** | **78.9%** | **0.473** | 7.68 |

Relative to the archived legacy-only runtime, the promoted policy improves held-out Hit@1 by 10.5
percentage points, Hit@5 by 26.3 points, and MRR by 0.167. It is also model-free at runtime. The
checkpoint remains packaged only for explicit `legacy` and `hybrid` diagnostics and provenance.

The root agent receives the skill directly through `skills/localization-v1`. Its prompt calls the
skill once only when the issue lacks an explicit target file, verifies the top three paths from source,
and falls back to exact search, graph tools, or the existing read-only analyzer. This avoids the invalid
`../skills` schema path and a two-stage root-to-analyzer-to-skill decision.

Six focused CPU tests pass. Deterministic ZIP construction and CRC validation pass. The local official
compiler command could not start because this development environment intentionally lacks `litellm`;
official compilation and inference remain separate runtime gates.

## Artifacts and next step

- Local ignored artifacts: `runs/2026-10-08-localization-v1/full-tree-benchmark-final.json`,
  `submission-localization-v1.zip`, and its manifest.
- No reference patch, test patch, credential, model inference, Kaggle upload, or competition submission
  was used by the skill or performed in this experiment.
- Conclusion: promote BM25 as Localization v1's safe full-tree runtime policy, retain the learned model
  as a negative/diagnostic ablation, and do not claim end-to-end issue-resolution improvement yet.
- Next experiment: in the prepared Kaggle or H100 evaluation environment, run a matched baseline versus
  Localization v1 repair comparison on valid public tasks. Require both official compilation and a
  resolution-rate gain before replacing `configs/baseline/`.
