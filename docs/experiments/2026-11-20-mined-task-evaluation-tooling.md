# Mined-task evaluation tooling and the per-task control gate (2026-11-20)

Status: working note for #34. Everything below is CPU-only; no GPU quota was used and nothing is submitted to
the competition.

## Problem

Two evaluation problems block a judgement about a candidate that differs by one or two tasks:

1. **Control validity does not transfer.** The October 13 manifest of 99 "control-valid" public tasks was only
   about half valid when rechecked in the current image (observed 5/10, 10/20 and 13/20 across three cohorts).
   The all-or-nothing gate in the batch notebook then discards the whole run, which wasted a full pair on
   2026-11-02.
2. **The task supply is too small and too noisy.** The 24/29-task dev cohorts carry a seed-to-seed standard
   deviation of about 1.6 tasks, so a 1-2 task difference cannot be resolved. A public dataset of 4,207 mined
   SWE-bench-style tasks (`mobassir/swe-agent-eval-validity-gemma4`) supplies holdouts, but it ships no
   repository snapshots and no offline wheels.

## What was solved

- `scripts/prepare_lora_eval_notebook.py --per-task-controls` measures only the tasks whose empty/gold
  controls both behave and records `excluded_invalid_controls`, instead of stopping the arm. The default
  all-or-nothing behaviour is unchanged for existing callers; `tests/test_lora_eval_notebook.py` covers the
  per-task path with a fixture where one of two tasks is valid (the suite has 20 tests).
- `scripts/build_mined_eval_bundle.py` rebuilds what the dataset does not ship: repository snapshots at the
  right `base_commit` (codeload tarballs, because the git protocol is blocked on this host) packed as
  `<instance_id>.tgz`, plus an offline wheel set per repository (`pip download`, cp312 manylinux), and a
  manifest with digests and the layout assumption.
- `scripts/make_mined_control_notebook.py` and `scripts/make_mined_sweep_notebook.py` generate CPU-only
  Kaggle notebooks that run the control pair for a handful of tasks or for a whole holdout split. The sweep
  embeds the 297 holdout tasks as a gzip+base85 payload (505 KB), caches one checkout per base commit, and
  appends `results.jsonl` after every task, so a truncated run still yields data.
- `scripts/collect_mined_sweep.py` downloads the finished kernel output, computes the per-repository validity
  rate with a Wilson 95% interval, writes `sweep-report.md`, and can post the table to #34.

## First control result (five tasks, three holdout repositories)

CPU kernel `junzhejiao/mined-task-controls-2026-11-20`, about two minutes, no GPU quota. Per task: download at
`base_commit`, apply the test patch and the gold patch (the `FAIL_TO_PASS` tests must pass), then re-download,
apply only the test patch and run the same tests (they must fail).

| Task | Repository | Gold | Empty | Verdict |
| --- | --- | ---: | ---: | --- |
| `lark_1540` | lark-parser/lark | pass | fail | valid |
| `ml_collections_09742cfcd9` | google/ml_collections | pass | fail | valid |
| `ml_collections_21f164575c` | google/ml_collections | pass | fail | valid |
| `click_2796` | pallets/click | fail | fail | invalid here |
| `click_3677` | pallets/click | pass | pass | invalid here (tests do not discriminate) |

Three of five are valid on our runtime. The dataset's own gating was done in its environment, so its labels do
not transfer: every task needs the in-kernel control gate, and the sweep enqueued for all 297 holdout tasks
measures the real per-repository rate before any 100-150 task arm is proposed.

## How to use

```bash
python scripts/build_mined_eval_bundle.py --tasks <tasks.jsonl> --ids <id> ... --out <bundle>
python scripts/make_mined_control_notebook.py --tasks <tasks.jsonl> --ids <id> ... --out <nb>
python scripts/make_mined_sweep_notebook.py --tasks <tasks.jsonl> --out <nb>
python scripts/collect_mined_sweep.py --kernel <owner/slug> --out runs/<date>-sweep [--post]
```

## Limitations

- The `.tgz` layout (repository content at the archive root) is our assumption; it works for patching and
  testing in the CPU controls but must be confirmed against a public snapshot in a kernel that mounts the
  competition data before inference runs depend on it.
- The control pair is a patch/test comparison, not the official verifier; parity with the evaluator is a
  separate check.
- The sweep payload is public task metadata only (patches, test lists); no competition data is involved.
