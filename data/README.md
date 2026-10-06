# Data

`raw/gemma-4-developer-agent/` holds the unchanged original competition package:
524 files, 20.88 GiB in the audited local snapshot. `processed/` holds derived data
with explicit transformation scripts and version records. Both are excluded from Git.

The original `tasks.jsonl` contains reference patches and test patches. Separate
inference inputs from scoring labels, and group training/validation splits by repository
and base commit. Extract only the required repository snapshots into each run.
On Trillium, do not bulk-extract repositories or execute repository tests on login nodes.

See the [data audit](../resource/manifests/data-audit.json) and
[competition research](../docs/competition-research.md).
