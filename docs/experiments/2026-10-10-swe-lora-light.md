# Bounded SWE LoRA training and evaluation

The compact loop completed data construction, one training epoch, checkpoint
recovery, official quantized serving, real tool interaction and a matched local
evaluation. **The adapter is not a release candidate:** patch resolution was
3/11 versus the baseline's 2/11, but clean nonempty completion fell from 8/11 to
4/11. Both resolved and clean was 2/11 for each. The registered three-task
acceptance gate also failed. No competition upload or reliable score gain is
claimed. The implementation and evidence are ready for teammate review in
[PR #8](https://github.com/jwj1342/gemma-agent-lab/pull/8).

The recipe uses 128 actions, one epoch and 32 updates. It replaces earlier
bounded attempts described below; all original failures and artifacts remain
available. The earlier four-GPU replacement jobs (`1071755`, `1071772`,
`1071773`, `1071774`) were cancelled before starting and used no compute time.

## Completed training: 128 actions and 32 updates

Run `runs/20261010-swe-lora-protocol-r2/`, frozen at
`1fcc891bd7a44adce6b7cc2b08f488b3dd933a7e`, completed one epoch and 32 updates.
Job `1072976` used one H100 for 15m47s (0.263 physical GPU hours), including
fixture generation, encoding, memory checks, training and checkpoint recovery.
Repository checks and 76 tests passed in the allocation and in
[CI run 38093147750](https://github.com/jwj1342/gemma-agent-lab/actions/runs/38093147750).
Official serving passed and the first real task completed and resolved. All
three registered probes and the fixed fast comparison now have recorded
outcomes. FastAPI exceeded its inference time limit, so the original three-task
release gate remains failed. Passing CI and completing evaluation do not change
that candidate-level decision.

The recipe is standard text-attention/MLP LoRA with rank 16, alpha 32, LR 2e-5,
4K context, a one-action microbatch and accumulation four. Each action's mean
target-token loss receives equal weight. Its 64 terminal exercises emphasize
ending after submission; the 32 SWE actions are a small coverage sample, not a
large repair-training corpus. The configuration is
[swe-lora-light-protocol.json](../../configs/training/swe-lora-light-protocol.json).

| Training source | Actions | Snapshot/fixture groups | Input tokens | Target tokens |
| --- | ---: | ---: | ---: | ---: |
| SWE | 32 | 30 | 105,087 | 3,903 |
| Graph | 32 | 10 | 26,148 | 1,196 |
| Termination | 64 | 64 | 104,786 | 1,096 |
| Total | 128 | — | 236,021 | 6,195 |

Each selected action was presented exactly once. SWE selection covers all five
tool types but only 32 of 484 available snapshot/tool strata; graph selection
covers all four tool types and 32 of 40 strata. The 64 selected terminal contexts
contain 29 single-file, 16 two-file and 19 three-file source patches, with actual
official test/submission observations. The entire encoded validation population
remains byte-identical to preceding runs.

The selected export is checkpoint 32, so its coverage matches the complete
epoch. It has 122,429,440 trainable parameters. Peak recorded training allocation
was 67.38 GiB. A fresh process restored checkpoint eight's optimizer/RNG state
and continued with a finite update nine; exact numerical equivalence to an
uninterrupted run was not tested.

| Fixed validation family | Initial NLL | Step 8 NLL | Step 32 NLL |
| --- | ---: | ---: | ---: |
| SWE | 0.506931 | 0.482471 | 0.424963 |
| Graph | 0.858223 | 0.719940 | 0.380186 |
| Termination | 3.520047 | 2.961483 | 1.622376 |

These likelihood changes do not establish convergence or repair gains.
The fixed short serving probes also separate valid syntax from useful choice:

| Short serving probe | Base | Adapter |
| --- | ---: | ---: |
| SWE schema-valid calls | 4/4 | 4/4 |
| SWE expected tool | 4/4 | 3/4 |
| Graph schema-valid calls | 3/4 | 3/4 |
| Graph expected tool | 3/4 | 2/4 |
| Terminal final answer | 4/4 | 4/4 |

Both models used thinking disabled for these short probes; autonomous evaluation
retained the baseline thinking configuration. Loading and measurable weight
effect passed, while the short tool-choice results do not show an improvement.

After the environment corrections below, acceptance recovery job `1073459`
passed rich but failed FastAPI's completion gate; release job `1073460` was
cancelled without starting. Diagnostic jobs `1073600` and `1073601` completed the
third probe and fixed comparison without promoting or retraining the adapter.
The training evidence,
sample coverage, source hashes and recovery receipt are under this run's
`training-evidence.json`, `encoded-data/manifest.json` and `recovery-gate.json`.

### First acceptance attempt and environment correction

The official serving checks passed, including a trained-versus-zero-adapter
effect and four clean terminal answers. The autonomous `rich_3454` run then
completed cleanly and passed official patch verification. It took 108.68 seconds
excluding model startup, made 27 LLM calls and 17 budget-counted calls, and
submitted four times. The trace contains 26 actual tool calls, with no graph
queries. This is one exposed development task and leaves efficiency work open.

Acceptance job `1072977` stopped before FastAPI inference: both its empty and
gold controls returned test-collection errors because `inline_snapshot` was
missing. The single-task runner had skipped the isolated offline dependency
setup already used by batch evaluation. This is an environment failure, not a
failed model prediction. The release dependency was cancelled without running.

The runner now applies the existing setup to single-task and batch execution.
A regression test exercises both orchestration paths and requires isolation
before verification. Validation was retried in a separate output directory with
the exact completed adapter and the same three tasks; all controls were rerun.
The trained model and its data/configuration contract stay unchanged. The final
fast comparison reused the baseline's frozen batch runtime, where
this setup was already active.

Validation source is `84d191dad10501d4ff89a558070903fc0f7722bf`;
[CI run 38095159667](https://github.com/jwj1342/gemma-agent-lab/actions/runs/38095159667)
passed all 77 tests. `validation-r2-registration.json` records the correction,
unchanged adapter hashes, task IDs and output paths before execution.

### Grading recovery preserves completed inference

The environment-corrected `rich_3454` inference completed with 15 LLM calls,
14 actual tool calls and one submission, followed by a final answer. It produced
a 1,570-character two-file patch. Independent grading then stopped during
snapshot extraction, before tests ran. Single-task verification still inherited
the agent's 60-second command limit, while batch grading used 300 seconds.
Job `1073256` failed after 16m01s; its release dependency did not start.

Commit `afa0397be57554642e6be6a305a901d067dd625c` gives both grading paths the
existing 300-second allowance. Inference budgets remain unchanged. The regression
test checks this separation; [CI run 38096242975](https://github.com/jwj1342/gemma-agent-lab/actions/runs/38096242975)
passed all 77 tests. Recovery registers the completed prediction's hash and
reuses it byte-for-byte after confirming identical agent/task identities. Empty
and gold controls run again before grading; no new rich inference is sampled.
The other two predeclared tasks run normally. `validation-r3-registration.json`
and `inference-reuse.json` preserve both runtime identities and this recovery.

Recovered grading of that exact prediction passed all **83 official tests** on
`rich_3454`. Its empty/gold controls were valid and the execution gate passed.
The run used 13 budget-counted calls and 14 actual tool invocations, including
one submission and no graph calls.

### FastAPI acceptance and broader diagnostic comparison

`fastapi_14455` passed its environment controls. The model then made 40 LLM/tool
calls (21 reads, 16 commands, two edits and one write) and reached the four-minute
session limit while still locating and revising the implementation. It never
called `submit_patch`; the harness recovered a 3,343-character working-tree
patch. This differs from the earlier post-submission repetition failure, but it
still fails the registered clean-completion gate. Its result is retained.
Trace timestamps attribute about 3.14 seconds to observed tool execution,
including 2.82 seconds across the 16 commands. The remaining time includes
model processing and harness overhead. The trace is not a GPU profiler, but it
does rule out long shell commands as the principal cause in this attempt.
`fastapi-trace-diagnosis.json` records the extraction and trace hash.

`diagnostic-registration.json` records the next measurements before execution:
independently grade the already captured patch, finish the originally selected
Requests task, and run the unchanged fixed fast cohort against the matching
baseline. These measurements characterize the completed weights. The failed
release gate remains in force, with no upload or promotion. No task is replaced,
no timeout is relaxed for inference, and no development trace enters training.

### Three-task probe outcomes

| Task | Valid controls | Clean nonempty completion | Official patch result |
| --- | --- | --- | --- |
| `rich_3454` | Yes | Yes; 15 LLM calls, one submission | Resolved; 83 tests passed |
| `fastapi_14455` | Yes | No; 40 calls, four-minute timeout, no submission | Unresolved; 1 failed, 3 passed, 1 error |
| `requests_7502` | Yes | Yes; 35 LLM calls, two submissions | Unresolved in this attempt; 1 failed, 337 passed, 1 skipped, 1 xfailed |

Requests inference took 199.49 seconds excluding model startup, with 163.63
seconds recorded by the agent timer. Its only failing test reported a refused
connection to the local HTTPS fixture. The patch changes file-like-object
detection in `models.py`, while the failing test exercises TLS connection pools.
This raises an environment-stability question, but does not prove flakiness or
justify replacing the recorded failure with a pass. The original result remains
unresolved. Job `1073600` completed independent FastAPI grading and the Requests
probe in 14m04s on one H100. None of the three traces used a graph tool.

### Fixed fast comparison: completed

Job `1073601` completed in 58m41s on one H100. It used matching runtime, task,
model, budget and verification hashes, the same valid-control mask as
`runs/20261009-prompt-icl/baseline/fast`, and 12 control-cache hits. All 11 eligible
tasks were evaluated; `fastapi_15745` remained excluded because its reference
control failed in both runs. The baseline prompt, 4-minute agent budget,
40 budgeted calls, 80 LLM turns, seed 42, 32K inference context and concurrency
one were unchanged.

| Metric | Matched baseline | Compact LoRA |
| --- | ---: | ---: |
| Environment coverage | 11/12 | 11/12 |
| Patch passed official verification | 2/11 | 3/11 |
| Clean completion with a nonempty patch | 8/11 | 4/11 |
| Both resolved and clean | 2/11 | 2/11 |
| Session timeouts | 2 | 5 |
| Turn-limit failures | 0 | 1 |
| Graph-tool calls across all 11 traces | 0 | 0 |
| Inference time, excluding shared model startup | 2,393.67 s | 2,824.89 s |

| Task | Baseline repair | LoRA repair | LoRA execution |
| --- | --- | --- | --- |
| `rich_4079` | Failed | Failed | Clean |
| `rich_3506` | Failed | Failed | Timeout; empty patch |
| `rich_3468` | Failed | Failed | Timeout |
| `rich_3278` | Failed | Passed | Clean |
| `httpx_3672` | Passed | Passed | Clean |
| `fastapi_14455` | Failed | Failed | Timeout |
| `fastapi_14266` | Failed | Failed | Empty patch |
| `fastapi_15800` | Failed | Failed | Clean |
| `fastapi_14609` | Failed | Failed | Timeout |
| `fastapi_9753` | Failed | Failed | Timeout |
| `requests_7502` | Passed | Passed | Turn limit; 39 submission calls |

The paired patch comparison has one new success (`rich_3278`) and no lost repair
successes. `requests_7502` passed independent grading of the recovered patch even
though the agent exhausted its turns. Its repair success therefore does not count
as clean completion. This fixed-cohort result is a separate registered attempt
from the earlier single-task Requests probe; neither replaces the other.

One additional solved task on an adaptively exposed 11-task development set is
weak evidence, especially with worse completion reliability and longer inference.
A separate 40-task baseline run (`runs/20261010-prompt-dev/baseline`) previously
solved 3/11 on these overlapping eligible tasks. The present matched reference
remains 2/11; that historical variability further limits interpretation.
The 70-task protected holdout remains sealed. No graph query occurred in the
candidate or reference traces, so this run offers no end-to-end evidence of graph
utility despite the lower graph NLL.

Short terminal checks passed in all four contexts, but the long Requests run
still repeated submission. The next bounded investigation should check long
interaction contexts and repeated observations, audit converted SWE actions,
and evaluate any correction on the same frozen setup. The current training mix
changes sampling, loss weighting and prompt context together; it cannot identify
which factor caused the observed differences. Increasing rank, data or epochs
without those checks is not supported by this result.

### Review and follow-up ownership

PR #8 hands the completed implementation and measured evidence to
**KaltistEsperanta** and **liaodid** for review. This is a research-code handoff;
the adapter has not passed its release gate. KaltistEsperanta owns adapter/runtime review and
the separate four-L4 training feasibility task (#13); liaodid owns data and
tool-boundary auditing (#14). Four-L4 training remains unverified. The present
BF16 training replica peaks above one L4's memory capacity; four GPUs do not
automatically pool their memory. No full training sweep is part of this handoff.

### Reproduction and artifact access

The compact [evidence manifest](../../resource/manifests/swe-lora-light.json)
records the training contract, encoded-file and adapter hashes, recovery receipt,
individual probe results, fixed-cohort comparison and allocated GPU costs. A Git
clone includes this evidence and the implementation. Full traces, prepared data,
weights and HOME environments remain local; coordinate access with the maintainer.

The measured run reused prepared SWE data from
`runs/20261009-swe-lora-v1-full/swe-data` and graph data from
`runs/20261009-swe-lora-v1-complete/graph-data`. It generated new terminal training
fixtures while preserving `retained-terminal-validation.jsonl`. These inputs are
identified by content hashes in `encoded-data/manifest.json`. Starting the launcher
without those prepared inputs invokes its source-preparation defaults, including
one SWE-smith shard, and does **not** reproduce this exact corpus automatically.
Verify source hashes before treating another run as a replication.

For a prepared Trillium workspace, use a new SCRATCH run directory, a frozen code
checkout, the recorded prepared sources and retained terminal validation file.
The lightweight launcher now defaults to the 128-action recipe. Passing the
configuration explicitly also makes the choice visible when using an older
checkout:

```bash
sbatch --time=00:30:00 --output="$RUN/train-%j.out" \
  "$CODE/scripts/slurm_swe_lora_light.sh" "$PROJECT_ROOT" "$RUN" "$CODE" \
  32 configs/training/swe-lora-light-protocol.json
```

`RUN`, `CODE` and `PROJECT_ROOT` are operator-provided paths; create `RUN` before
submission. Load the recorded environments, modules and model version. Keep the
registered three-task acceptance separate from training and stop release on its
failure. The historical fixed comparison uses the runtime frozen with
`runs/20261009-prompt-icl/baseline/fast`; a new evaluator version needs a matching
baseline. A replay must preserve predictions, controls and failures instead of
sampling until a task passes.

## Initial 1,024-action protocol

- Required base: `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`.
- Use the corrected native tool-handoff supervision, rank 16 / alpha 32,
  text attention and MLP projections, LR 5e-5, and a 4,096-token maximum.
- Reuse the pinned raw sources and snapshot exclusions. After encoding and
  deduplication, select 896 SWE, 112 graph, and 16 terminal actions. Round-robin
  over snapshot/tool strata with deterministic hashed ordering and seed 42.
  Preserve every selected action's recent context and complete target.
- Keep the validation population unchanged; use a fixed panel of up to 32
  examples per source for checkpoint selection. Keep the protected holdout sealed.
- One H100, microbatch one, accumulation four, one epoch: 256 optimizer updates.
  Save/restart at step eight, stop at step 32, and test the partial adapter before
  continuing. This is a new adapter and schedule; no old optimizer is reused.
- Native-tokenizer checks cover every retained sample. Record representative
  target prefixes/suffixes in `encoded-data/protocol-examples.json` for inspection.
  The official template ends pending calls at token 50 and final answers with
  `<turn|>` followed by a newline. Prompt tokens remain masked.

The partial adapter must pass the official loading/effect checks and autonomous
agent runs on the predeclared exposed tasks `rich_3454`, `fastapi_14455`, and
`requests_7502`, each with empty/reference controls. The gate requires valid,
nonempty, clean execution artifacts; issue resolution is reported separately.
A failed task is retained and investigated rather than replaced. Passing these
three probes only permits the remaining short epoch, not a performance claim.

After the epoch, the selected adapter must pass the existing release checks and
the unchanged fast cohort before a competition upload. Preserve the submitted
ZIP identity and actual upload receipt. Use the baseline prompt; the separate ICL
candidate is not promoted by its measured development result.

Configuration: [swe-lora-light.json](../../configs/training/swe-lora-light.json).
Launchers: [training](../../scripts/slurm_swe_lora_light.sh) and
[partial-agent validation](../../scripts/slurm_preflight_swe_lora_light.sh).
These launchers are Trillium-specific; dataset selection and the trainer are
separate Python modules.

## Preparation audit

The first allocation (`1072210`) stopped during the native-token audit, before
loading the model or starting training. Graph fallback fixtures lacked
`target_kind` metadata. The encoder now derives it from the actual assistant
target for every source and rejects conflicting metadata. Regression coverage
includes fixtures without that field. Its three dependent jobs were cancelled
without running. The preparation attempt used 7m39s on one H100.

The corrected retry uses a separate run directory, preserving the failed
preparation artifacts. An existing encoded-data directory without a completed
manifest is rejected rather than treated as a usable corpus. Data budgets,
training length, validation tasks, and release requirements remain unchanged.

## Corrected run: observed preparation and recovery

The corrected run is `runs/20261010-swe-lora-light-r2/`, frozen at training source
`55fb9b9b9e73c36021e46e3aeaadfb2c216e495c`. Repository checks and 72 tests passed
in its allocation and in
[CI run 38087195294](https://github.com/jwj1342/gemma-agent-lab/actions/runs/38087195294).

| Selected source | Actions | Snapshot groups | Input tokens | Target tokens |
| --- | ---: | ---: | ---: | ---: |
| SWE repair | 896 | 97 | 2,627,353 | 152,382 |
| Graph exercises | 112 | 10 | 91,028 | 4,355 |
| Terminal exercises | 16 | 16 | 12,567 | 261 |
| Total | 1,024 | — | 2,730,948 | 156,998 |

SWE actions cover 412 upstream trajectories and all 484 available snapshot/tool
strata; graph actions cover all 40 available strata. Every one of the 1,008 tool
targets ends at the supervised handoff token 50. The terminal targets retain the
native `<turn|>` and following newline. All 4,290 encoded validation actions are
preserved; the checkpoint-selection panel contains 32 SWE, 32 graph, and 19
terminal actions. These counts describe actions with context, not 1,024 complete
repair trajectories.

Both backward memory probes passed: the longest input (4,096 tokens) and longest
target (1,959 tokens). The highest Torch allocation recorded in the probes was
70.42 GiB. One allocator retry was logged; the backward operation completed.
Checkpoint eight was saved, loaded in a fresh process, and followed by a successful
update nine. Configuration/data/dependency contracts matched, and optimizer/RNG
state was restored. `recovery-gate.json` retains this evidence; exact equivalence
to an uninterrupted numerical run was not tested.

The initial allocation (`1072257`) completed successfully in 17m08s on one H100,
including full-corpus encoding, the memory probes, save/restart and 32 updates.
Those updates presented 128 selected actions and 20,008 supervised tokens.

| Fixed validation family | Actions | Initial NLL | Step 32 NLL |
| --- | ---: | ---: | ---: |
| SWE | 32 | 0.506931 | 0.243458 |
| Graph | 32 | 0.858223 | 0.099037 |
| Termination | 19 | 3.520047 | 1.596865 |

These teacher-forced losses establish progress on the training objective.
Autonomous interaction and repair outcomes are separate checks. The partial
adapter and its hashes are retained under `preflight-validation/`; this checkpoint
does not count as completing the selected-data epoch.

## Termination diagnosis and 512-action retry

Partial validation job `1072258` loaded the adapter in the official W4A16 runtime
and measured a weight effect against a repeated zero-update control. All eight
short tool-context requests produced schema-valid calls. However, all four
held-out submission contexts generated another tool call instead of a final
answer; the base ended correctly in all four. These short probes disable thinking
for both models; the autonomous agent retains the baseline thinking configuration.

On `rich_3454`, the autonomous adapter inspected and edited source, ran a small
test, and captured a 656-character source patch after about 24 seconds. It then
repeatedly ran invented test names after submission, exhausted 40 budgeted tool
calls, and timed out at four minutes. A separate diagnostic (`1072519`) subsequently
graded the captured patch with the unchanged official verifier: the tests passed
(`resolved=true`, exit zero). The original agent still failed the clean-completion
gate. This separates a successful code change from failed control flow. The native
tool handoff worked across separate calls; post-submission termination remained
broken. The 14m48s validation allocation failed, and both dependent jobs were
cancelled without running. No adapter was uploaded.

Only 261 of the selected corpus's 156,998 supervised tokens were terminal answers
(0.17%). The first 32 updates saw three terminal actions. This motivates a bounded
corrective hypothesis, rather than proving a unique cause:

- Select 384 SWE, 64 graph and 64 terminal actions: 512 total, one epoch and 128
  optimizer updates. Keep the same raw sources, snapshot exclusions and seed.
- Average token loss within each action, then weight real actions equally within
  a global batch. Thus short completion answers receive the same example weight
  as long tool calls. Keep token-weighted validation NLL and also log token-weighted
  training loss; the two training objectives should not be compared as one curve.
- Reduce LR from 5e-5 to 2e-5. Keep rank 16, alpha 32 and the 4K context limit.
- Start a new adapter from the required frozen base. Preserve the stopped run;
  do not reuse its optimizer or alter its source contract.
- Save/restart at eight, stop at 32, and require all four terminal probes to end
  before running the same three real-task probes. Keep the empty/gold controls,
  nonempty-clean-artifact requirement, fast cohort and protected holdout unchanged.

Configuration: [swe-lora-light-balanced.json](../../configs/training/swe-lora-light-balanced.json).
The combined corrective experiment changes sampling, loss weighting and learning
rate, so its outcome cannot isolate their individual effects. The stopped task's
trace is used for diagnosis only; its patch or answer is not added to training.

## Balanced partial checkpoint: observed behavior

The 512-action run encoded 384 SWE actions, 64 graph actions and 64 terminal
actions, totaling 1,233,068 input tokens and 70,771 supervised tokens. Its full
validation file is byte-identical to the preceding run. The memory probes passed
with a 70.42 GiB peak; checkpoint eight restored optimizer/RNG state and proceeded
to update nine. Job `1072507` completed preparation and 32 updates in 17m11s on
one H100. Those updates presented 103 SWE, 14 graph and 11 terminal actions.

| Fixed validation family | Initial NLL | Step 32 NLL |
| --- | ---: | ---: |
| SWE | 0.506931 | 0.325754 |
| Graph | 0.858223 | 0.237467 |
| Termination | 3.520047 | 1.340907 |

Official W4A16 loading and the trained-versus-zero-adapter effect check passed.
All four short terminal contexts ended cleanly, compared with zero of four for
the preceding recipe. Graph calls were schema-valid in four of four contexts,
but matched the expected tool in only two of four, versus three of four for the
base. These observations do not establish a graph reasoning improvement.

The first autonomous task still failed the clean-completion gate. With valid
empty/gold controls, the model edited source, tested, and first submitted after
about 30 seconds. It then repeated `submit_patch`, reaching 68 submission calls
and exhausting 80 LLM turns. The captured patch contained 657 characters; this
run's patch has not yet been independently graded. The harness counted 12
budgeted calls because submission calls are accounted differently; the trace
contains 80 actual tool invocations. Job `1072508` failed after 11m03s, and jobs
`1072509` and `1072510` were cancelled without starting. No adapter was uploaded.

The short probe uses thinking disabled, whereas the autonomous baseline uses a
2,048-token thinking budget. Diagnostic job `1072797` tests the same step-32
weights with thinking disabled on the same three registered tasks, preserving
the prompt, tools, budgets, controls and clean-output gate. This is a generation
configuration diagnostic; it neither completes training nor establishes that
LoRA alone improves repair performance. No continuation was scheduled during
this diagnostic.

## Complete 128-action protocol pilot

The generation diagnostic also failed: thinking disabled still produced a patch
and then repeated tests, status queries and submissions until the 80-turn limit.
It made 35 command calls, 21 submission calls and 18 status calls, plus six file
actions. No training occurred in that 9m50s allocation. This rejects disabling
thinking as a sufficient fix for this checkpoint on the exposed task.

The next bounded pilot completes a smaller corpus: 32 SWE, 32 graph and 64
terminal actions, with one epoch and 32 total optimizer updates. All 64 terminal
actions will be presented, compared with 11 seen before the balanced run's
partial gate. Training terminal contexts use the existing baseline system prompt,
whose content hash becomes part of dataset provenance. This addresses the gap
between the short training instruction and the full deployment policy; it does
not add the failed task, its patch or trace to training. Validation contexts stay
unchanged. The small SWE sample has lower snapshot/tool coverage, which must be
reported rather than extrapolated to broad repair learning.

Use a fresh base/optimizer, the same rank 16 / alpha 32 projections, LR 2e-5,
example-normalized objective, 4K limit, single H100 and accumulation four.
Save/restart at eight, finish at 32, then run the same short official-runtime
checks and three real-task probes. A full epoch and lower NLL still require clean
autonomous artifacts and the fixed fast cohort before release. Use the original
baseline thinking configuration. This combined sampling/prompt correction does
not isolate causal effects or guarantee an increase in repair rate.

Configuration: [swe-lora-light-protocol.json](../../configs/training/swe-lora-light-protocol.json).

Before model training started, a further fixture audit found that every old
terminal observation reported two changed files and a patch size of 658–660
characters. The generator ran Python without excluding bytecode artifacts.
The first preparation attempt (`1072906`) and its unstarted successors were
cancelled during encoding, before a model checkpoint or optimizer update existed.

The corrected generator executes the same official tools against real fixtures
with one, two or three edited source files. It commits a bytecode ignore rule,
runs checks with `python3 -B`, and asserts that the captured changed-file count
equals the intended source-file count. Four task phrasings replace the old
uniform request that explicitly told the model to provide a completion message.
Final text remains synthetic, while test/submission/status observations are
actually executed. The existing 19 held-out records are retained byte for byte;
their reference and hash are recorded separately from newly generated fixture
evidence. These changes broaden the completion contexts without importing a
development task or inventing tool observations.

## Cost evidence and budget

| Scope | Physical H100 hours |
| --- | ---: |
| Current 128-action training, preparation and recovery | 0.263 |
| Fixed 11-task comparison allocation | 0.978 |
| Current recipe: all training, validation attempts and diagnostics | 2.192 |
| All bounded attempts, including preceding stopped recipes | 3.585 |

The first two rows are included in the third, and the third in the fourth;
do not add the rows. These are allocated GPU hours, including failed checks and
CPU grading while the GPU remained allocated. Queue time, the historical full
run and the pre-existing baseline are excluded. The matched evaluation used one
model startup (204.17 seconds) and cached all 12 environment-control pairs; its
runner took 3,485.64 seconds. Repeated independent startup and environment
corrections explain why the whole investigation cost more than the 16-minute
training run. These H100 times are not four-L4 training measurements.

The earlier final continuation occupied four H100s for 9h50m11s, or 39.35 physical
GPU hours, excluding earlier preparation/recovery and release validation. Its
one-epoch encoded corpus contains 92,648,546 input tokens, including repeated
recent history, with 4,680,468 supervised target tokens. Two epochs repeat that
corpus; these token counts are not a two-epoch total. Masking a prompt's loss
does not remove its forward computation.

That implementation used full BF16 model replicas, gradient checkpointing,
one-example microbatches and DDP. The recorded rank-zero peak allocation was
71.65 GiB. Ordinary optimizer-step intervals averaged 8.30 seconds on four GPUs.
Intervals spanning validation/checkpoint boundaries averaged longer and totaled
about 27 minutes, including the subsequent training step; this is not a dedicated
profiler's estimate of checkpoint overhead. Most recorded time was in ordinary
training intervals. We have not measured GPU utilization or kernel-level efficiency.

The initial bounded recipe had 1,024 action presentations versus the earlier 33,982. Based on
the earlier throughput, training is expected to take roughly an hour on one H100;
this is an estimate, and preparation plus inference/verification add cost. The
initial training allocation is capped at 45 minutes and stops at update 32.
Only a successful autonomous probe permits the remaining one-epoch continuation,
capped at 90 minutes. That continuation was cancelled after the behavioral check.

The 512-action corrective run starts at frozen source
`e6522a6c90b3000de691346b37ee1a5695bc7196`, under
`runs/20261010-swe-lora-balanced/`. Jobs `1072507`–`1072510` cover preparation and
32 updates, partial behavior validation, the remaining one-epoch training, and
release validation in that order. Each later stage requires the preceding stage
to succeed. Repository checks and 76 tests passed in the initial allocation and
[CI run 38089579952](https://github.com/jwj1342/gemma-agent-lab/actions/runs/38089579952).
The initial allocation was shortened before starting from 45 to 30 minutes,
based on the measured 17m08s predecessor; the amendment is recorded. Remaining
training is capped at 45 minutes. No automatic two-epoch or full-corpus fallback
is authorized. The data size, update count and allocation limits describe a
budget, not a measured runtime or successful completion.

## Team access on four L4 GPUs

The competition runtime was observed with four L4 GPUs and 23,034 MiB available
per GPU. Our existing L4 inference check passed; L4 training has not been validated.
The current full-replica BF16 trainer cannot fit each replica on one L4, regardless
of a smaller dataset. Four separate GPU memories require explicit distribution.

Candidate training paths are PEFT with FSDP model sharding or a compatible
low-bit frozen-base implementation. Both need a load/backward/save/reload test
on the exact model and environment. Do not substitute an unrelated NF4 base for
the required QAT checkpoint or claim a Kaggle training recipe is ready based only
on H100 execution. H100 is the immediate protocol-debugging path; a teammate-facing
L4 recipe remains a separate compatibility requirement.

Primary references: [LoRA](https://arxiv.org/abs/2106.09685),
[PEFT FSDP](https://huggingface.co/docs/peft/en/accelerate/fsdp),
[PEFT quantization](https://huggingface.co/docs/peft/en/developer_guides/quantization),
and [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/).
These describe techniques and hardware, not a measured speedup for this run.
