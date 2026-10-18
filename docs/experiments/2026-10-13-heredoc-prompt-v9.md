# Heredoc-repair prompt v9 on official wheelhouse v29

## Question and comparison

- **Hypothesis:** On the official version-29 harness, multi-argument
  `edit_file` calls were frequently malformed while single-argument heredoc
  `run_command` calls arrived intact. A prompt that removes `edit_file`, builds
  replacements from text already read from the target file, and ends promptly
  after `submit_patch` may convert a correct working-tree change into a scored
  repair within the four-minute task limit.
- **Baseline and proposed change:** The reference is
  [`configs/baseline/`](../../configs/baseline/). The candidate is
  [`configs/experiments/heredoc-fix-v9/`](../../configs/experiments/heredoc-fix-v9/).
  Relative to the baseline, v9 removes `edit_file`, lowers
  `thinking_budget` from 2048 to 1024, and replaces the system prompt with a
  prescribed heredoc repair, targeted-test, submission, and termination
  sequence. Model, adapter policy, sampling seed, and task budgets are
  otherwise unchanged.
- **Task IDs / grouped split and data version:** Author-run public smoke tasks
  `rich_3278`, `rich_3894`, and `rich_2943`; official wheelhouse snapshot
  `official-v29-2026-10-08`, Kaggle dataset version 29. These tasks are not the
  sealed 70-task holdout.
- **Success metric:** Valid empty/reference controls followed by official
  verification with `agent_resolved=true` and a normally completed kernel.

## Reproduction

- **Source commit and configuration directory:** The configuration is introduced
  by the first commit in this PR under `configs/experiments/heredoc-fix-v9/`.
- **Preparation and execution:** Build the standalone archive with:

  ```bash
  python3 scripts/build_submission.py \
    --agent-dir configs/experiments/heredoc-fix-v9 \
    --output runs/heredoc-fix-v9/submission.zip
  ```

  A representative private-notebook command is:

  ```bash
  python3 scripts/prepare_kaggle_runtime.py \
    --owner losttrace \
    --slug gemma4-heredoc-v9-3278 \
    --agent-dir configs/experiments/heredoc-fix-v9 \
    --task-id rich_3278 \
    --wheelhouse-snapshot official-v29-2026-10-08 \
    --output runs/heredoc-fix-v9/rich_3278/notebook
  kaggle kernels push -p runs/heredoc-fix-v9/rich_3278/notebook \
    --accelerator NvidiaL4 --timeout 2700
  ```

  Reproduction now requires explicitly attaching the historical version-29
  wheelhouse dataset; the current live wheelhouse is a different environment.
- **Agent ZIP SHA256:** At reviewed commit `6b99f13`, the documented
  `build_submission.py` command produced
  `5c9b4dd6560b9a655c119700150f334bf35075af07200b824248d435a02f14ef`
  (3,716 bytes). The archive submitted to the competition was packaged
  differently and has SHA256
  `6b0a37e83112aa9105a19bbc35051b68c54084266c1e03123ab30e219eda9c96`
  (3,714 bytes); its member files were identical to that reviewed commit. This
  review update removes the trailing space from `prompts/system.md`, after
  which the documented command reproducibly produces
  `ba5fa676d8653d834b61f4e12835cfb7ee4a65f711cfbd5c03620ff210819eb3`
  (3,734 bytes). The cleanup does not change prompt behavior.
- **Model and adapter:**
  `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`; no adapter.
- **Environment and seed:** Official Kaggle notebook; `swegemma` 0.2.10,
  `adk-submission` 0.2.13; seed 42.
- **Platform and allocation:** Four NVIDIA L4 GPUs. Tensor parallelism was
  four. The compact archived records do not independently retain context-length
  or GPU-memory-fraction measurements for every run.
- **Budgets:** `max_time_minutes: 4`, `timeout_seconds: 60`,
  `max_tool_calls: 40`, `max_turns: 80`, `thinking_budget: 1024`, and
  `max_output_tokens: 8192`.

## Results

Empty-patch and reference-patch controls passed before the recorded candidate
runs. The same frozen v9 configuration was used for all five attempts.

| Task | Kernel | Terminal state | Agent time | Official result |
| --- | --- | --- | ---: | --- |
| `rich_3278` | `losttrace/gemma4-heredoc-v9-3278` | COMPLETE | 204.1 s | `resolved=true`; 23 tests passed |
| `rich_3278` | `losttrace/gemma4-heredoc-v9b-3278` | ERROR | 241.0 s | Timed out; not resolved |
| `rich_3278` | `losttrace/gemma4-heredoc-v9c-3278` | ERROR | 241.4 s | Timed out; not resolved |
| `rich_3894` | `losttrace/gemma4-heredoc-v9-r3894` | COMPLETE | 139.2 s | `resolved=true` |
| `rich_2943` | `losttrace/gemma4-heredoc-v9-r2943` | COMPLETE | 138.5 s | `resolved=true`; 27 tests passed |

The candidate resolved three of five attempts: one of three on the
escape-dense `rich_3278` task and both attempts on the other two tasks.

### Competition submission

The v9 archive was submitted to the competition and is the candidate's only
full-test-set evidence.

| Submission | Submitted (UTC) | Description | Status | Public score |
| --- | --- | --- | --- | ---: |
| `57067045` | 2026-10-11 07:48 | `Verified repair v2; ... ZIP 6b0a37e83112` | COMPLETE | 0.10 |

For comparison, baseline v1 (`02bb939095de`) and verified repair v1
(`c52a04379d88`) each scored 0.12. The v9 smoke resolved three of five
attempts, but its scored submission was below both comparisons.

### Transport observation

The author summarized the following call-level counts from exploratory v29
traces. They are diagnostic observations, not a randomized paired experiment.

| Channel | Observed calls | Intact calls |
| --- | ---: | ---: |
| Multi-argument `edit_file` | 18 | 0 |
| Single-argument heredoc `run_command` | 26 | 26 |

These observations motivated removing `edit_file` from this candidate. They
describe v29 and do not by themselves establish the behavior of wheelhouse
v31.

### Matched wheelhouse-v31 comparison

The team later ran the frozen v9 candidate on the same ten-task fast cohort and
identical H100 runtime as the three-way comparison in
[#23](https://github.com/jwj1342/gemma-agent-lab/pull/23) (job 1104757).

| Variant | Resolved | Tool errors | Missing-argument events | Longest error streak |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1/10 | 43 | 0 | 3 |
| protocol | 2/10 | 50 | 22 | 5 |
| recover | 3/10 | 68 | 15 | 3 |
| heredoc v9 | 2/10 | 33 | 0 | 2 |

V9 had the fewest tool errors, no missing-argument events, and the shortest
error streak. It resolved `rich_3278` and `fastapi_9753`. Four tasks ended in
`inference_error` after reaching the time limit, compared with two for
`recover`; the fixed repair-script sequence appears to spend too much time on
tasks that require more exploration. In v31, the remaining weakness is budget
use rather than malformed tool-call transport.

### Historical comparison

The exact baseline configuration resolved zero of two archived `rich_3278`
runs and three of three archived `rich_3894` runs. Across 22 pre-v9
`rich_3278` traces covering the baseline and several prompt variants, no run
resolved; four traces produced a correct patch but did not reach a scored clean
completion. The 22 traces must not be described as 22 baseline attempts.

This history is context rather than a matched multi-seed A/B comparison. The
candidate has only one stochastic attempt on `rich_3894` and `rich_2943`, and
three attempts on `rich_3278`; run-to-run variance may exceed the treatment
effect.

### Changes that did not help

- Additional prose that banned pre-verification or diagnosed contradictory
  escape-heavy scripts did not reliably prevent verification loops.
- Classifying scripts as either write-back fixes or import-based checks was
  bypassed by other generated script forms.
- A thinking budget of 512 reduced useful tool activity. A budget of 2048
  increased deliberation and produced larger, more fragile scripts within the
  same wall-clock limit. The candidate therefore retains 1024.
- `requests_7205` and `httpx_3672` were rejected by their reference controls
  because of sandbox dependency incompatibilities. They provide no evidence
  about candidate repair quality and are excluded from the result table.
- A five-minute follow-up was halted before producing a result and is not part
  of this candidate or its evidence.

Model-startup time, complete notebook elapsed time, and per-run quota deltas
were not retained consistently in the compact records and are therefore not
reported.

## Artifacts and next step

- **Artifact access:** The five raw inference traces behind the smoke results
  are archived in the private team data repository
  [`jwj1342/gemma-agent-lab-data`](https://github.com/jwj1342/gemma-agent-lab-data)
  under `experiments/heredoc-fix-v9-v29/`. Its README lists each trace's
  kernel, task, status, runtime, outcome and SHA256. The exploratory pre-v9
  traces behind the transport counts and the scratch notes remain local.
- **Summary and hashes:** This report is the team-visible compact summary; the
  raw v9 traces are in the data repository rather than in this repository.
- **Training:** Not applicable; no weights, optimizer state, or adapter were
  changed.
- **Conclusion:** v9 demonstrated a viable end-to-end repair path on three
  public tasks under official wheelhouse v29 and resolved three of five smoke
  attempts. Its full competition submission scored 0.10, below the 0.12
  baseline, and its matched v31 run resolved 2/10 tasks while four tasks reached
  the inference time limit. The evidence supports preserving v9 as a recorded
  transport experiment, but not adopting it as the team's submission policy.
  Any follow-up should retain the heredoc recovery mechanism while testing a
  less rigid sequence or a larger task budget in a frozen, environment-matched
  comparison. Keep the 70-task holdout sealed and coordinate any competition
  submission with the team.
