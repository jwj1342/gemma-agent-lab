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
- **Agent ZIP SHA256:**
  `6b0a37e83112aa9105a19bbc35051b68c54084266c1e03123ab30e219eda9c96`.
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

### Transport observation

The author summarized the following call-level counts from exploratory v29
traces. They are diagnostic observations, not a randomized paired experiment.

| Channel | Observed calls | Intact calls |
| --- | ---: | ---: |
| Multi-argument `edit_file` | 18 | 0 |
| Single-argument heredoc `run_command` | 26 | 26 |

These observations motivated removing `edit_file` from this candidate. They do
not establish the behavior of wheelhouse v31, whose rebuilt harness must be
evaluated separately.

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
reported. No competition submission was made.

## Artifacts and next step

- **Artifact access:** Raw run outputs remain on the author's laptop under the
  ignored Kaggle run-output directories and are not included in a repository
  clone. Teammates can identify the private Kaggle runs by the kernel names in
  the table and request the corresponding archive from the author. The frozen
  candidate is identified by its configuration commit and ZIP SHA256 above.
- **Summary and hashes:** This report is the team-visible compact summary. Raw
  timelines and scratch notes remain local rather than being committed as
  project documentation.
- **Training:** Not applicable; no weights, optimizer state, or adapter were
  changed.
- **Conclusion:** v9 demonstrated a viable end-to-end repair path on three
  public tasks under official wheelhouse v29 and resolved three of five smoke
  attempts. The evidence supports preserving the candidate and the v29
  transport observation; it does not establish causal superiority over the
  baseline or predict behavior on v31. Any expansion should use a frozen,
  environment-matched comparison. Keep the 70-task holdout sealed and do not
  submit to the competition without team coordination.
