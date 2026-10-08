# Fast local evaluation

Use a fixed public cohort to compare a proposed agent with a saved baseline before spending time on
another competition submission. The batch runner loads the official model once, retains independent
per-task sandboxes, caches versioned environment controls, and resumes completed task results.
It uses the official `run_agent_sandbox` and `verify_task` functions and their required-test checks.

The workflow is tracked in [Issue #4](https://github.com/jwj1342/gemma-agent-lab/issues/4) and
[PR #3](https://github.com/jwj1342/gemma-agent-lab/pull/3). The initial LoRA evidence remains in
[first-lora.md](first-lora.md); a lower SFT loss is separate from task resolution.

## Frozen tasks and training boundaries

[public-v1.json](../configs/evaluation/public-v1.json) pins the original task-file hash, membership,
repository/base-commit identities, exposure sources, and exclusion reasons. Membership was chosen
from metadata, before measuring candidate success. The fast cohort is nested in the development cohort.

| Cohort | Tasks | Purpose |
| --- | ---: | --- |
| `smoke` | 1 | Historically inspected `rich_3454`; check compatibility and recovery |
| `fast` | 12 | Frequent comparisons: six FastAPI, four Rich, one Requests, one HTTPX |
| `dev` | 40 | Broader development comparisons, including the same 12 fast tasks |
| `holdout` | 70 | Infrequent milestone assessment; requires `--allow-holdout` during preparation |

Of 129 released tasks, 18 share the 16 snapshots used for graph training or action validation.
Those 18 and the historically inspected smoke task are excluded from the 110 benchmark candidates.
The SWE-smith pilot's selected training repositories do not intersect these four public repositories.
The holdout is a development convention, not an access-control boundary: its source data are public.

Tasks sharing a snapshot stay together. Different commits in the same repository may still contain
similar code or later fixes. This split does not establish unseen-repository generalization or rule out
base-model pretraining exposure. HTTPX contributes only one example and cannot support a stable
repository-level estimate. The hidden competition tasks come from private repositories and cannot be
exported; public resolution rates are useful local evidence, not predictions of a leaderboard score.

Future `prepare_sft_data.py` and `prepare_graph_sft.py` runs read this benchmark by default and exclude
all development and holdout snapshots. Keep the denylist when adding other data producers. Check new
sources for duplicate issues, patches, near-duplicate code, and later commits before expanding training.
A development task used to guide changes stays a development task; moving it into holdout does not
restore independence. Never regenerate membership merely because a candidate performed poorly.

## Prepare a candidate

Preparation needs Python's standard library and the public task file; it does not load a model or
require Kaggle credentials. Keep the baseline agent unchanged. Edit a separate configuration directory
for your method, and put any exported PEFT adapter inside that agent's `adapters/` directory.

```bash
python3 scripts/prepare_eval_suite.py \
  --agent-dir configs/baseline \
  --cohort fast \
  --run-dir runs/20261008-my-baseline
```

A prepared run includes an immutable agent copy, exact submission ZIP, sanitized task records, the
benchmark definition, and copies/hashes of the execution scripts. It records the Git revision and
whether preparation used an uncommitted working tree. Reference patches and test labels are absent
from the inference task file. Existing run directories cannot be overwritten.

The whole cohort is retained even when some environments fail validation. Empty-patch and gold-patch
controls must pass before a task reaches inference. The report shows all excluded tasks and coverage;
invalid environments are not silently replaced with easier tasks.

## Run on the prepared Trillium environment

The following launcher is specific to the maintainer's H100 installation. Read the
[cluster runtime guide](cluster-runtime.md) before adapting paths or accounts. It requests one H100,
24 CPU cores, and two hours; ordinary GPU jobs let Trillium select the partition. All run artifacts and
caches are on SCRATCH. It does not submit a Kaggle entry.

```bash
sbatch --account=def-zhouyang \
  --output=runs/20261008-my-baseline/slurm-%j.log \
  scripts/slurm_eval_suite.sh "$PWD" "$PWD/runs/20261008-my-baseline"
```

For a proposed method, prepare a new run using its own agent directory. Compare with the existing
baseline by adding `--baseline` to the launcher arguments:

```bash
python3 scripts/prepare_eval_suite.py \
  --agent-dir configs/experiments/my-method \
  --cohort fast \
  --run-dir runs/20261008-my-method

sbatch --account=def-zhouyang \
  --output=runs/20261008-my-method/slurm-%j.log \
  scripts/slurm_eval_suite.sh "$PWD" "$PWD/runs/20261008-my-method" \
  --baseline "$PWD/runs/20261008-my-baseline"
```

The baseline's predictions are reused for comparison; the candidate always generates its own patches.
An existing completed run resumes without launching a model or rerunning tests. An interrupted run
skips completed, hash-checked task artifacts and starts a server for any missing predictions. A recorded
model failure is a completed attempt and is not silently retried. Preserve damaged/partial artifacts;
if a result receipt is absent or changed, use a new run rather than selecting the best retry.

Use `--controls-only` to diagnose environments before inference. Use `--rehash-assets` to reread all
asset content instead of accepting unchanged indexed hashes. `--cache-dir` selects a separate cache
when independent revalidation is desired. These options follow the two required launcher arguments.
The broader development set may need a larger `sbatch --time` value; setup and grading add to the
agent's configured four-minute per-task budget.

## What is reused

| Work | Reuse rule |
| --- | --- |
| Model startup | One vLLM startup for all pending tasks in a candidate run |
| Empty/gold controls | Cache keyed by full task record, snapshot, wheels, image, installed environment, execution source, and evaluation-budget hashes |
| Large-asset hashing | SHA-256 is indexed with device, inode, size, modification time, and change time; recently written entries and small files are rehashed |
| Completed predictions/tests | Same frozen run only, with artifact hash receipts |
| Baseline results | Read existing completed evidence for a matched comparison |

Each generation and verification still gets a fresh repository sandbox. Model weights persist across
tasks, while contexts, traces, edits, and budgets are separate. Inference mounts only sanitized task
records and public code/graph inputs; the control cache, reference patches, verification workspace, and
Kaggle credentials stay outside its mounts. The existing Apptainer/subprocess runtime is not equivalent
to the official scoring service's Docker/gVisor security boundary.

Content changes to the environment, grader, data, or budget invalidate control reuse. The hash index
assumes trusted local files and normal filesystem metadata behavior; `--rehash-assets` provides an
independent full-content check. Invalid controls are cached too, with their logs, so a second method does
not spend GPU time rediscovering the same environment problem. Fixing runtime inputs creates a new key.
Concurrent readers see atomically written cache entries; each run has an exclusive process lock.

## Read the result

The run's `report.md` is a short table; `summary.json` contains the structured result. Full predictions,
traces, independent test output, input identities, model startup sessions, and phase timings remain in
that run. Raw data, weights, caches, and run outputs are excluded from Git.

- **Resolution rate:** resolved tasks divided by all fixed tasks with valid environment controls.
  Agent timeouts, empty patches, tool failures, and failed repairs remain in this denominator.
- **Coverage:** valid-control tasks divided by the complete selected cohort. A reduced-coverage score
  must be labeled with both counts. The report also gives successes divided by all selected tasks.
- **Clean completion:** reported separately from repair. A recovered patch can pass official verification
  after an agent timeout; a clean final answer can still contain an incorrect patch.
- **Paired changes:** newly resolved tasks, regressions, tasks solved by both methods, and tasks solved
  by neither. Comparison rejects different cohorts, models, grading environments, budgets, and execution protocols.
- **Costs:** model startups, startup time, per-task inference and grading time, runner time, cache hits,
  model/tool calls, graph calls, and trace token counts. Queue time is excluded.

A 12-task result changes by 8.3 percentage points per task. Use it to catch regressions and prioritize
ideas, then confirm promising methods on development and milestone sets. The descriptive Wilson
interval assumes independent Bernoulli outcomes; shared repositories and adaptive tuning limit that
interpretation. One stochastic attempt per task does not establish statistical significance.

The batch engine uses a fixed vLLM seed of 42 and records the submitted sampling configuration; this
does not promise bitwise deterministic GPU generation. Compare graph-enabled and graph-disabled
agents on identical tasks and budgets before claiming graph utility. More graph calls alone are not a gain.

## Timing and scoring scope

Batching saves repeated startup and setup checks while retaining the official required-test criterion.
H100 memory, tensor parallelism, CPU allocation, and command execution differ from Kaggle L4×4.
Even the same per-task wall-clock limit permits different amounts of computation. Use H100 for frequent
quality comparisons and periodically check a few unchanged tasks on L4×4 to assess deployment costs.
Retain the competition's total inference budget, which includes sandbox setup and excludes final grading.
Do not convert H100 seconds to L4 seconds using a single-task multiplier.

The maintained single-task Kaggle notebook remains available through
[the Kaggle walkthrough](kaggle-runtime.md). The new batch launcher targets local Trillium runs;
no Kaggle submission, retraining, or hidden-task access is part of this workflow.
