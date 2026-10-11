# Fast local evaluation: baseline and pilot adapter

## Question and comparison

Can a new agent be evaluated repeatedly on fixed public tasks without another competition submission,
repeated model startup for every task, or repeated environment controls?

The comparison uses the unchanged baseline and the corrected first SWE/graph LoRA adapter. Their
prompts and evaluation budgets are identical. The candidate adds the `first_lora` adapter.
The [frozen fast cohort](../../configs/evaluation/public-v1.json) contains 12 tasks from four repositories,
within a 40-task development set; a separate 70-task holdout remains unused. Known training/validation
snapshot overlaps and the previously inspected smoke task are excluded from these cohorts.

The metric is official task resolution, with environment coverage and clean agent completion reported
separately. Timeouts and failed patches remain in the eligible-task denominator.
The complete receipts, timing records, per-task outcomes, and hashes are in the
[machine-readable evidence](../../resource/manifests/local-evaluation.json).

## Reproduction

Follow the [local evaluation guide](../local-evaluation.md) to stage dependencies, freeze inputs, launch,
resume, and compare runs. The exact frozen execution files in both runs match commit
`78d2ac80624434c06f83b5495d1e6e4447a025a2`. They were prepared before that commit with recorded working-tree
changes; their individual hashes, rather than the earlier preparation commit alone, identify execution.

| Input | Identity |
| --- | --- |
| Baseline configuration | `configs/baseline` |
| Baseline ZIP SHA-256 | `02bb939095ded0c970088fc9d19634e203e2d3119a36a5feff20bad55c9343fc` |
| Candidate ZIP SHA-256 | `c32b143c6fbce2b5b9c614f183afdd47060bcc2cfe9550ae4b857e7cdac7012f` |
| Adapter SHA-256 | `7a5f5f4a90487565e56f776fd4317263179d12bd8449cc8e62a2cdedf1f37732` |
| Benchmark SHA-256 | `b2d6ed557261dc2d11f6ada0fdbe7ca62f66ccf9ca90273f77fbc56ff1293d37` |
| Sanitized task SHA-256 | `ada1c9c5765053b617210654d98ceccbb2457bac1228ceafd7a265f34d90005d` |
| Required model | `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2` |
| Serving and harness | Patched vLLM 0.19.1, swegemma 0.2.7, adk-submission 0.2.12, google-adk 1.36.1 |
| Hardware | One NVIDIA H100 80 GB, 24 allocated CPU cores per run |
| Serving protocol | TP=1, context=32768, GPU memory fraction=0.8, vLLM seed=42, sequential tasks |

Model, container, environment, supplemental wheels, and individual task asset hashes are recorded in
each run's `runtime-inputs.json`. Each disposable sandbox records package versions and confirms that
the repository package is imported from its own workspace. Inference has no reference labels or
control-cache mount. The preparation records preserve the complete submitted time/tool budgets.

## Controls and coverage

Eleven of twelve tasks passed both controls. Most empty patches fail ordinary assertions;
`fastapi_14609` instead imports the new `PydanticV1NotSupportedError` class before implementation.
The precise expected import failure is accepted only together with a fully passing reference control.

`fastapi_15745` remains excluded. Its reference patch produces **33 passed / 3 skipped**, but the
official required-test check rejects a skipped required test,
`test_included_api_route_without_app_scope_returns_405_response`. In the pinned package,
`swegemma/harness/verification.py` invokes pytest with `-p no:anyio`; the affected tests use
`@pytest.mark.anyio`. We retain the verifier unchanged. A zero pytest exit status is insufficient.
The excluded task stays in the 12-task coverage report; it is not replaced with another task.

Earlier preflights exposed host-package shadowing for HTTPX, missing offline FastAPI dependencies,
and a Rich syntax-highlighting snapshot mismatch. The versioned local setup fixes these preparation
issues without changing official patch application, target-test selection, or required-test checks.

## Measured results

The baseline resolved **4/11 eligible tasks**, with **11/12 coverage**. It solved `rich_3278`,
`httpx_3672`, `fastapi_9753`, and `requests_7502`. Eight tasks completed cleanly; three hit the
four-minute agent timeout. The baseline made two graph-tool calls, both on an unresolved task.
These observations do not establish whether graph access helps.

The LoRA resolved **2/11**, solving `httpx_3672` and `requests_7502`. It gained no task and regressed on
`rich_3278` and `fastapi_9753`, a descriptive difference of **−18.18 percentage points** on the same
eligible mask. Three tasks produced nonempty patches and received independent verification; the other
eight were recorded as failures without repeating known empty-patch tests.

| Task | Baseline | LoRA |
| --- | --- | --- |
| `rich_4079` | Agent timeout | Turn limit; empty patch |
| `rich_3506` | Agent timeout | Turn limit; empty patch |
| `rich_3468` | Tests failed | Turn limit; empty patch |
| `rich_3278` | Resolved | Context limit; empty patch |
| `httpx_3672` | Resolved | Resolved |
| `fastapi_14455` | Tests failed | Normal return; empty patch |
| `fastapi_14266` | Tests failed | Turn limit; empty patch |
| `fastapi_15745` | Excluded: invalid reference control | Same exclusion |
| `fastapi_15800` | Tests failed | Turn limit; empty patch |
| `fastapi_14609` | Agent timeout | Generated patch failed test collection |
| `fastapi_9753` | Resolved | Turn limit; empty patch |
| `requests_7502` | Resolved | Resolved |

The candidate made **273 `submit_patch` calls**, versus nine for the baseline, and six tasks exhausted
the 80-turn limit. Its three clean completions compare with eight for the baseline. This broader cohort
exposes a stopping-policy problem that the earlier one-task smoke did not establish as fixed.
Graph-tool attempts increased from two to eleven, including unmatched node identifiers in the logs.
More attempts did not yield more resolved tasks. An adapter comparison alone cannot isolate the
causal effect of graph access; a matched graph-disabled ablation would address that question.

| Measurement | Observed result |
| --- | ---: |
| First baseline run, including uncached controls | 68.93 minutes |
| Candidate run, with cached controls | 33.90 minutes |
| Empty and reference control phases | 17.99 minutes |
| Model startups per method across 11 eligible tasks | 1 |
| Baseline / candidate model startup | 187.15 / 225.18 seconds |
| Baseline / candidate per-task inference, summed | 39.17 / 26.31 minutes |
| Completed-run resume | 58.81 seconds |
| New model/inference/test phases during resume | 0 |
| Prediction and grade artifacts unchanged after resume | 22/22 |

Baseline job `1060056` completed successfully. Resume job `1061885` hit all 12 control-cache entries
and performed only identity checks and reporting. Candidate job `1061884` also hit all 12 entries,
skipping the previously measured 17.99 minutes of controls, and completed successfully at the runner
level. Its failed model attempts remain in the score. The candidate's lower total time also reflects
shorter failed trajectories and fewer nonempty patches to grade; it is not evidence of a better agent.

The batch avoided ten additional model startups compared with one startup per eligible task.
At the observed baseline startup duration, that is approximately **31.19 minutes of startup work**.
This is an estimate from one measured startup, not a timed execution of the old 11-process workflow.
The 58.81-second resume reuses completed answers; it is not the cost of evaluating a new method.
All timings exclude queue time. Candidate-versus-baseline wall-time differences can also reflect
different generated trajectories, so they cannot all be attributed to caching.

## Artifacts and next step

Local artifact root: `/scratch/jwj/kaggle_gemma4_agent`. The baseline is in
`runs/20261008-eval-fast-baseline-r6/`; the candidate is in `runs/20261008-eval-fast-lora-r6/`.
Each contains frozen inputs, predictions, traces, verification logs, per-phase timings, and a summary.
The pre-resume baseline artifact hashes are in
`runs/20261008-local-evaluation-validation/baseline-before-resume.json`.
These files are excluded from Git; teammates need an owner-granted artifact transfer or an independent
run using the documented preparation commands. The first adapter's provenance is in the
[LoRA report](../first-lora.md).

Keep the submitted baseline. Before scaling this adapter's training, improve and validate stopping
after submission, retaining a real patch, staying within context limits, and using valid graph-node
identifiers. Use this same fast cohort for development regressions, followed by the wider development
set and milestone holdout. The result is one stochastic attempt per method/task and does not establish
statistical significance. Public tasks, repository overlap, a small cohort, and H100 timing limit
generalization to the hidden leaderboard. This workflow submits no competition entry and retrains no model.
