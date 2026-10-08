# Portable Localization v1 and nested gate learning

## Outcome and scope

A new content-based refinement of Localization v1 improves full-tree localization against BM25 in
both nested validation protocols and the frozen historical public development/test comparisons.
The archived checkpoint remains preserved. The refinement learns small, portable evidence weights
without using the archived checkpoint or memorized repository/path/token identities.

This is measured localization improvement, not yet a measured Gemma repair improvement. The public
test had already been inspected in earlier studies, so its results remain retrospective. No new
independent confirmation, official compilation, GPU inference, or remote repair A/B was executed.

## Algorithm and validation correction

The previous gate-only cross-validation reused a legacy checkpoint trained on those same 91 tasks.
Its large cross-validation gains therefore did not measure an unseen-task ranker. The legacy trainer
now explicitly blocks promotion on that evidence. Historical reports retain their numerical results,
with this qualification linked from the large-sample report.

The new learner uses seven per-file signals, normalized within the current repository and issue:

1. Original BM25, with its coefficient anchored at 1.
2. BM25 with duplicate query tokens removed.
3. BM25 with weaker length normalization.
4. Path-field retrieval.
5. Symbol-field retrieval.
6. Exact symbol definitions, discounted by definition frequency across files.
7. A generic implementation-versus-documentation/example path indicator; no repository-specific prior.

Six nonnegative coefficients are learned through two coordinate passes over the predeclared grid
`[0, 0.25, 0.5, 1, 2, 4]`. The primary objective equally weights repositories and balances Hit@1 and
Hit@5; regularized MRR breaks ties. The final weights, in the feature order above, are
`[1, 2, 0, 2, 2, 0, 1]`.

The gate is learned from three-fold out-of-fold ranker predictions, not fitted-ranker predictions.
It searches 16 predeclared confidence regions and requires at least six changed-rank observations
plus no aggregate primary-metric regression. The selected gate uses the candidate when BM25's margin
is at most `0.6` and the candidate margin is at least `0.025`.

Both ranker and gate are refit within each outer fold: five snapshot-grouped folds and four
leave-one-repository-out folds. Shared base commits stay in one fold. Training tasks sharing a
confirmation commit would be excluded; none needed exclusion in this run. There are 40 ranker fits
across the final fit, calibration, and two nested protocols. Reference paths are joined after feature
extraction; runtime receives only source and issue text. Development/test labels never choose weights,
thresholds, or checkpoints in this experiment.

## Results

| Evaluation | Tasks | BM25 Hit@1 | Gated v1 Hit@1 | BM25 Hit@5 | Gated v1 Hit@5 | BM25 MRR | Gated v1 MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nested snapshot folds | 91 | 23.1% | 41.8% | 54.9% | 74.7% | 0.384 | 0.555 |
| Nested repository holdout | 91 | 23.1% | 39.6% | 54.9% | 72.5% | 0.384 | 0.526 |
| Frozen development | 18 | 38.9% | 38.9% | 55.6% | 83.3% | 0.466 | 0.549 |
| Retrospective test | 19 | 31.6% | 47.4% | 68.4% | 78.9% | 0.473 | 0.624 |

One development task adds only a new source file and is not scorable for existing-file retrieval.
It is retained in the planned remote repair A/B, giving 38 rather than 37 tasks there.

The gate enables the learned ranking on 17/18 development and 14/19 test tasks. Without the gate,
development Hit@1 falls to 33.3%; the gate restores 38.9%. On the historical test, the ungated candidate
is stronger (Hit@1 57.9%, Hit@5 84.2%, MRR 0.705); the gate sacrifices some gains there. This is a
measured tradeoff, not evidence that the gate improves every dataset or every task.

Five thousand paired, repository-stratified bootstrap draws give test delta intervals of
`[+5.3, +31.6]` percentage points for Hit@1, `[-10.5, +31.6]` points for Hit@5, and
`[+0.028, +0.282]` for MRR. Development Hit@1 and MRR intervals still cross zero. These descriptive
intervals do not erase prior exposure to the public data or establish hidden-repository performance.

Two full runs produced identical weights, gate, and metrics. The final run took 74.120 seconds on
the Windows laptop, Python 3.14.6, standard library only, CPU execution, seed 52. A separate audit
compared the runtime and learner decisions/ranks on all 37 scorable confirmation tasks: all matched.
Median ranking time excluding repository scan/AST construction was 0.248 seconds for portable v1 and
0.271 seconds for the current BM25 wrapper in that audit. These are local measurements, not L4 timing.

## Runtime and recovery

The versioned policy is `configs/experiments/localization-v1/skills/localization-v1/resources/portable-policy.json`.
Explicit `--mode portable` applies the candidate and its gate for experiments without loading the
archived 4 MB model. It marks the output experimental. Default adaptive execution retains BM25 until
the portable policy is deliberately promoted. A disabled policy does not compute candidate features.

```powershell
python scripts/train_portable_localization.py `
  --output-dir runs/2026-10-08-localization-v1-portable/reproduction `
  --cache runs/2026-10-08-localization-v1-portable/features-final.json
python configs/experiments/localization-v1/skills/localization-v1/scripts/localize.py `
  --mode portable --root PATH_TO_REPOSITORY --issue "ISSUE_TEXT"
python -m unittest tests.test_localization_v1 tests.test_localization_gate tests.test_portable_localization tests.test_localization_ab -v
```

The script refuses an existing output directory. Resume by rerunning into a new directory with the
hash-validated feature cache; optimization is deterministic and small enough to recompute. The cache
contains labels and is an ignored training artifact, never an agent input. If source/input hashes
change, use a new cache. The study records the Git revision, exact runtime/trainer/task hashes, Python
version, platform, seed, grid, task IDs, exclusions, source commit statistics, and complete fold membership.

The final study is `runs/2026-10-08-localization-v1-portable/study-final/study.json`.
The earlier `study-01` run is preserved. The Git parent was
`f1a5a9dec7bbe6cb1c89deff674d65bffa156288`; exact hashes identify the uncommitted implementation.

## Frozen remote comparison

`scripts/prepare_localization_ab.py` prepares three arms: the existing remote baseline, the BM25
skill, and portable v1 with its gate. The latter two share all agent files except the explicit mode
in skill instructions. Model, sampling seed 42, analyzer, context settings, and four-minute/40-tool
budgets are unchanged. Each case goes through `prepare_eval_run.py` and frozen-input verification.
Task bytes are explicitly UTF-8/LF on Windows and Kaggle to prevent cross-platform hash mismatch.

```powershell
python scripts/prepare_localization_ab.py `
  --output runs/2026-10-08-localization-v1-portable/new-remote-ab `
  --owner YOUR_KAGGLE_USERNAME
```

The prepared final run is `runs/2026-10-08-localization-v1-portable/remote-ab-final/`.
The portable experimental ZIP is `runs/2026-10-08-localization-v1-portable/submission-localization-v1-portable.zip`
(726,506 compressed bytes, 4,112,735 unpacked). Frozen arm SHA256 values are:

| Arm | SHA256 |
| --- | --- |
| Baseline | `aa597ea39b66412eb53fcd3cd2fcc470c0650853a4ebe69f73d777f0ba7e822c` |
| BM25 skill | `cb1aa5427f2ab42044bb1029371860da4476872774b9176aec3a11c3d3e746b7` |
| Portable v1 with gate | `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48` |

Its `ab-protocol.json` freezes 38 tasks and 114 cases, rotates arm order, and declares acceptance
before inference: at least 5 percentage points resolution gain over baseline, one-sided exact paired
McNemar p-value at most 0.05, positive incremental resolution over the BM25 skill, mean agent-time
ratio at most 1.10, and valid controls/completed pipelines for every predeclared case.

Six tasks are selected by seeded repository round-robin for integration checks, without inspecting
candidate ranks: `fastapi_13713`, `fastapi_14459`, `requests_7433`, `requests_7502`, `rich_3296`,
`rich_3675`. Eighteen private offline notebooks are generated under `notebooks/<task>/<arm>/`.
The pilot alone cannot satisfy the 38-task promotion protocol. Earlier `remote-ab` and `remote-ab-lf`
directories are preserved preparation attempts; use only `remote-ab-final`.

GPU launch stays manual under the repository collaboration guide. Check current Kaggle quota before
starting, then use the [Kaggle runtime procedure](../kaggle-runtime.md) for each predeclared case.
For example, after quota review, the first integration notebook can be launched with:

```powershell
kaggle kernels push `
  -p runs/2026-10-08-localization-v1-portable/remote-ab-final/notebooks/fastapi_13713/baseline `
  --accelerator NvidiaL4 --timeout 2700
```

Each notebook requires empty/gold controls before inference and independent patch verification after.
For non-pilot tasks, generate a notebook with `prepare_kaggle_runtime.py --run-dir <frozen-case>`.
The current runner starts the server per notebook; total time and quota must be checked before running
all cases. Preserve each downloaded `runtime-smoke/` directory, and copy its contents into the matching
`cases/<task>/<arm>/` directory after comparing the frozen hashes in the protocol. Never substitute an
agent bundle or task. To inspect the complete evidence:

```powershell
python scripts/summarize_localization_ab.py `
  --run-dir runs/2026-10-08-localization-v1-portable/remote-ab-final
```

Missing or invalid cases remain in the denominator and block promotion. The summarizer verifies each
bundle/task against the protocol; it cannot turn absent remote results into a success. There has been
no notebook upload, GPU launch, competition submission, PR, or merge in this experiment.

## Verification

All 25 focused localization/paired-evidence tests passed. The full suite passed 39/46; remaining failures
are four platform-dependent baseline ZIP hash expectations and three POSIX-path assertions on Windows.
The previous frozen-task newline mismatch is fixed; its test now reaches the later ZIP-hash assertion.
Official compiler/runtime execution is pending because the laptop environment lacks `litellm`,
`swegemma`, and the required model. Repository checks and package hashes are recorded in the work log.
