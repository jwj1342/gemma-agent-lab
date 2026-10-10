# LoRA SFT versus baseline: the paired protocol and what it can actually detect

**Status: protocol and power analysis. No adapter has been trained and no paired run has been
executed for this experiment.** Prepared 2026-10-11.

## Why this document exists before training

The training side of this project is nearly ready: the offline stack trains and exports
([training stack manifest](../../resource/manifests/training-wheels-sft-v1.json)) and the adapter
configuration is wired
([lora-trajectory-v1](../../configs/experiments/lora-trajectory-v1/README.md)). The evaluation side is
the binding constraint, and it is worse than "we need more tasks": at the current task count the
experiment **cannot** produce an interpretable verdict at any realistic effect size. That is worth
knowing before a GPU hour is spent, so the protocol and its power are fixed here first.

## Control validity: what "measurable" means

A task is usable only when the unpatched repository fails and the reference patch passes. The exit
codes are not interchangeable, and reading them wrongly silently discards real tasks. Two defects of
exactly that kind were found and fixed while writing this note: the first version of the analysis
script required the empty arm to exit 0 and reported **zero** usable tasks, and the second version
still rejected empty-arm collection failures and reported three when four were usable.

| Arm | Required | Meaning |
| --- | --- | --- |
| empty | `resolved: false`, no error, and either exit 1 (tests ran and failed) or exit 2 caused by a collection error | The issue is real; the tests either fail or cannot import the symbols the fix introduces |
| gold | `resolved: true`, `test_exit_code == 0`, no error | The reference fix resolves it, so the task is solvable |

The second empty-arm shape matters more than it looks. `rich_3930` imports `CellSpan`,
`split_graphemes`, `cell_len`, `get_character_cell_size` and `split_text` from `rich.cells`, and that
task's own source patch adds exactly those names, so collection *must* fail before the fix. Requiring
a clean test failure would have discarded a perfectly measurable task. The rule that keeps this safe
is the gold arm: if collection also fails with the reference patch applied, the harness never
exercised the issue and the task is unusable.

Anything else is a different kind of evidence:

| Observed | Reading | Example |
| --- | --- | --- |
| collection error in **both** arms | Test-environment gap, not the issue; the gold control fails | `fastapi_13713`, `fastapi_14786` |
| timeout in both arms (exit 124) | Too slow to measure under this budget | `requests_7433` |
| empty exit 1, gold exit 1 | Base already passes; nothing to fix | `rich_3296` |

Measured on the existing control sweep (8 tasks attempted): **4 usable tasks — `rich_3454`,
`rich_3675`, `rich_3676`, `rich_3930` — all from a single repository.**

## Power: what three tasks can and cannot show

Pairwise scores are pass/fail, so only tasks where the two arms disagree carry information
(McNemar's exact test, alpha 0.05, target power 0.8). Computed by
[analyze_paired_power.py](../../scripts/analyze_paired_power.py):

| Assumed true gain (on tasks the baseline fails) | Tasks needed for 80% power |
| --- | --- |
| 5 points | 158 |
| 10 points | 78 |
| 20 points | 40 |
| 30 points | 26 |

With 4 tasks, power is effectively zero for every one of those gains and the minimum detectable gain
is undefined: the run would need the baseline to resolve nothing and the adapter to resolve all four
before the result clears the significance bar.

Two consequences:

1. **The public set cannot fund the experiment.** It holds 129 tasks, and the observed control yield
   so far is 4 of 8 attempted. Even a perfect yield across all 129 would leave a 10-point effect
   undetectable with a paired design.
2. **A single-repository task set cannot attribute the effect.** All usable tasks are `rich`, so a
   win could be repository memorisation rather than a transferable improvement. This is the same
   trap the SFT data already has
   ([SFT dataset builder](../../scripts/build_sft_dataset.py) warns that the four public repositories
   are hidden at scoring time).

## The protocol, frozen before results exist

1. **Task set.** The control-valid tasks only. Adding a task requires its own valid empty/gold pair;
   more attempted tasks is not the same as more measurable tasks.
2. **Arms.** `no-adapter baseline` versus `lora-trajectory-v1`, identical prompts, budgets, sampling,
   seed, and task set. The adapter is the only changed variable: the experiment directory copies the
   baseline prompts byte for byte.
3. **Controls per run.** Every arm re-establishes empty-fails and gold-passes before inference.
   Historical controls do not transfer to a new run.
4. **Metric.** Resolved-issue count per arm on the shared task set, reported as the paired 2x2
   (both, neither, baseline-only, adapter-only) with McNemar's exact p-value, never as two
   independent percentages.
5. **Acceptance.** Promotion requires the adapter arm to win on this paired design at the computed
   power. With the current task set that bar is unreachable, so the honest expected outcome is
   **"not measurable"**, not "no improvement".
6. **Cost.** Each paired task costs two full agent episodes with startup, sandbox, and test
   execution. The available measurements are partial: the L4 agent phase was 74.10 s per task in the
   October 5 smoke run, while the October 10 pair spent 118.9 s and 171.4 s of agent phase across its
   two arms; total wall time per task including environment startup reached many minutes. Scaling
   task count must be budgeted from measured wall time in the run that produces it, not from the
   agent-phase figure alone.

## What follows for the next step

The measurement instrument, not the adapter, is the gating item. Concretely, before a paired LoRA run
can mean anything, the project needs a control-valid task set sized from the table above, and the
cheapest way to grow it is to fix the failures that are fixable: collection errors (`exit 2`) suggest
environment gaps that a different test invocation may resolve, and the timed-out task may pass with a
larger per-task budget. Those are CPU-side investigations with no GPU cost, and they are worth doing
before spending quota on a result that cannot be read.
