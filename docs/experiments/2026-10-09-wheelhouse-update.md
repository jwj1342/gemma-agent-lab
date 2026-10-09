# Official wheelhouse migration

Date: October 9, 2026. Tracking: [Issue #5](https://github.com/jwj1342/gemma-agent-lab/issues/5).

## Question and change

Can a freshly generated Kaggle notebook use the updated official wheelhouse while
preserving dependency provenance and the earlier environment?

The generator now accepts exact registered snapshots, with automatic selection by
default. An explicit `--wheelhouse-mode record` allows exploratory use of an
unregistered update, saves the actual hashes and versions, and labels it unreviewed.
Both paths reject changes between setup and installation. The agent configuration,
evaluation implementation, model handle, and previous H100 environment are unchanged.
See the [mode and migration guide](../wheelhouse-updates.md).

## Inventory and source review

The official dataset API reported version 29. A private
[CPU inventory notebook](https://www.kaggle.com/code/jwj1342/gemma4-wheelhouse-inventory-oct09-cpu),
version 1, completed and hashed all 41 mounted wheels. The filenames and byte sizes
match the pinned version-29 listing. The
[registry](../../resource/manifests/kaggle-wheelhouse-snapshots.json) retains the
October 5 inventory and adds this snapshot.

Changes are `adk_submission` 0.2.12 to 0.2.13 and `swegemma` 0.2.7 to 0.2.10.
`adk_eval_core` 0.1.0 was repacked: its archive SHA256 changed, but every extracted
member has identical content. The other 38 wheel names and hashes are unchanged.
Static source review found changes in graph lookup, tool-error feedback, prompt
construction, protected patch paths, and thought handling. These can change agent
behavior and grading; old aggregate results belong to their original environment.

## Reproduction

Prepare a private single-task notebook from this branch:

```bash
python3 scripts/prepare_kaggle_runtime.py \
  --owner YOUR_KAGGLE_USERNAME \
  --slug gemma4-wheelhouse-v29-smoke --title 'Gemma4 Wheelhouse V29 Smoke' \
  --agent-dir configs/baseline --task-id rich_3454 \
  --wheelhouse-snapshot official-v29-2026-10-08 \
  --output runs/wheelhouse-v29-smoke/notebook
kaggle kernels push -p runs/wheelhouse-v29-smoke/notebook \
  --accelerator NvidiaL4 --timeout 2700
```

The dataset snapshot selector checks the mounted bytes; it does not change the
attached dataset version. Attach version 29 if the live input has subsequently changed.
The generator retains the official pinned container and
`google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`, tensor parallel size 4,
GPU memory utilization 0.90, and baseline sampling seed 42. The unchanged ZIP hash is
`02bb939095ded0c970088fc9d19634e203e2d3119a36a5feff20bad55c9343fc`.

## Validation status

Repository checks and all **31 tests passed** in Slurm job `1062413` on October 9.
The tests cover old/new snapshot selection, deliberate record mode, actual version
capture, same-name byte changes, file additions/removals, rechecking before pip,
fresh-checkout preparation, and existing task/agent integrity checks.
The allocation used `StdEnv/2023`, `python/3.12.4`, and the existing compiler tooling
environment; the checks themselves used CPU only.

A private [L4 smoke notebook](https://www.kaggle.com/code/jwj1342/gemma4-wheelhouse-v29-smoke),
version 1, has been submitted. Compilation, empty/reference controls, inference,
and generated-patch verification are pending. No new competition entry has been
submitted, and repair quality with the updated harness has not yet been measured.

The notebook was prepared on parent revision `3054da842f794c854f438a5478ea9437b119e9bc`
with the uncommitted fix. Embedded helper SHA256 values in `preparation.json` and
the runtime manifest identify the exact executed source; the parent revision alone
does not identify those changes.

## Artifacts

Maintainer artifacts are under `runs/20261009-wheelhouse-update/`: the pinned file
listing, CPU inventory output, source diffs, preparation receipt, Slurm test log,
and generated L4 notebook. They are local artifacts excluded from Git. The private
Kaggle notebook links above require access; teammates can reproduce with their own
account. Preserve `runtime-smoke/inputs/wheelhouse.json`, the runtime manifest,
control results, and agent trace with any migrated experiment.
