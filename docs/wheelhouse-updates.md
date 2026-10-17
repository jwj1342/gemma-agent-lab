# Updating the Kaggle evaluation wheelhouse

Issue [#5](https://github.com/jwj1342/gemma-agent-lab/issues/5) reported that a new
notebook stopped at `Official wheelhouse inventory changed.` The generator embedded
October 5 hashes while Kaggle mounted newer official inputs. Wheel count alone did
not reveal the change: both snapshots have 41 wheels.

## Choose a mode

Normal preparation uses `--wheelhouse-mode verified --wheelhouse-snapshot auto`.
The notebook accepts any exact snapshot in the
[registry](../resource/manifests/kaggle-wheelhouse-snapshots.json), records which one
matched, and checks every file's SHA256. This permits known official updates while
retaining the historical snapshot.

To require a particular recorded environment, add:

```bash
--wheelhouse-snapshot official-v29-2026-10-08
```

This selects the expected fingerprints; it does not change Kaggle's attached dataset
version. Ensure the corresponding version is attached. A frozen agent/task run and
a frozen dependency environment are separate choices: `--run-dir` alone does not
force the old wheelhouse.

For an official update not yet in the registry, explicitly request record mode:

```bash
python3 scripts/prepare_kaggle_runtime.py \
  --owner YOUR_KAGGLE_USERNAME --slug gemma4-wheelhouse-exploration \
  --title 'Gemma4 Wheelhouse Exploration' \
  --agent-dir configs/baseline --task-id rich_3454 \
  --wheelhouse-mode record \
  --output runs/wheelhouse-exploration/notebook
```

Record mode permits differences from historical SHA256 values. It hashes the mounted
files, reads their package versions, checks that the core runtime packages exist,
and marks an unknown snapshot as `unreviewed_snapshot`. These are observed fingerprints,
not independent proof of authenticity or compatibility. Use the official input handle.

Both modes recheck the captured file set and hashes **before installation**. Replaced,
added, or removed files after setup still stop execution before pip or installation
cleanup. Agent ZIP, task, and helper-script provenance remain separate and unchanged.
An explicit named snapshot cannot be combined with record mode.

## Understand the diagnostic and receipts

An unmatched verified-mode input reports `missing`, `added`, and `changed_sha256`
filenames for the registered snapshots. Update the checkout and regenerate the
notebook, attach the requested historical version, or deliberately choose record
mode. A mismatch fails before the runtime output directory is created, so fixing
an input does not require deleting a partial run created by this check.

The notebook records the selected mode, matched snapshot, actual hashes, package
versions, and compatibility status in both `runtime-smoke/inputs/wheelhouse.json`
and `runtime-smoke/run-manifest.json`. `preparation.json` records the requested mode
and snapshot before the notebook runs. Preserve these files with the results.

## October 9 inventory and compatibility review

The official dataset reported version **29**, with files published October 8. A
private CPU-only Kaggle inventory run streamed all 41 mounted wheels through SHA256;
its names and sizes match the version-29 file listing. Original hashes remain in
[eval-assets-verification.json](../resource/manifests/eval-assets-verification.json).

| Component | Historical snapshot | Version 29 |
| --- | --- | --- |
| `adk_submission` | 0.2.12 | 0.2.13 |
| `swegemma` | 0.2.7 | 0.2.10 |
| `adk_eval_core` | 0.1.0 | 0.1.0, repacked archive |

Thirty-eight wheels have unchanged names and hashes. Two wheel names were replaced;
`adk_eval_core` retained its name but changed SHA256. Every member of its old and new
ZIP archives has identical content; the difference is in the archive packaging.
Package versions alone would miss this change. Recording hashes distinguishes the
exact inputs even when their extracted code is unchanged.

Static review of the official wheel sources found changes to graph-node name
resolution, graph relation matching, tool-error feedback, agent prompts, protected
patch paths, and handling of `include_thoughts=False`. These affect behavior and
verification, not just installation. Source compatibility checks do not establish
repair quality; preserve old results and rerun matched methods when migrating.

The new snapshot is registered by its verified bytes. GPU smoke status is recorded
separately in the registry and the
[migration validation report](experiments/2026-10-09-wheelhouse-update.md). The existing H100
environment and its version-specific dependency hook remain unchanged.

## October 13 inventory: version 31

The official files were re-published on October 12 at 19:51 UTC as dataset version **31**.
All 41 wheels were downloaded and hashed on a datamover. `swegemma` moved from 0.2.10 to 0.2.11, and
`adk_eval_core`, `adk_submission` and `vllm` kept their versions but changed SHA256. `swegemma` 0.2.11
removed `swegemma.models.discovery` and `swegemma.models.registry`; the harness scripts need the API
migration from [#21](https://github.com/jwj1342/gemma-agent-lab/pull/21) before they run on it.

The local H100 runtime now uses `/home/jwj/venvs/gemma4-eval-v31`, built with
`scripts/prepare_eval_environment.sh` and `scripts/install_eval_git.sh`. The version-29 environment
is kept for reproducing older runs. The dependency hook accepts 0.2.11 because
`swegemma/harness/container_setup.py` is identical in both versions, and the hook rebinds
`install_test_dependencies` in the two modules that import it (`agent_runner` and `verification`).
The [H100 report](experiments/2026-10-13-prompt-v31.md) records the first matched runs in this environment.

## Add another snapshot

1. Obtain the official dataset version and all wheel fingerprints in a disposable
   environment. Inspect metadata and relevant code changes; do not infer unchanged
   content from a filename or package version.
2. Append an entry with a unique ID, source/version, actual hashes, package versions,
   and an honest compatibility status. Preserve existing entries.
3. Run the preparation tests, then manually validate compilation, empty/reference
   controls, and one agent repair with the intended serving setup. Keep execution
   success separate from whether the repair resolves the issue.
4. Compare methods only within a recorded environment. Historical results remain
   evidence for their original snapshot.

Source: [official wheelhouse, version 29](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse/versions/29).
