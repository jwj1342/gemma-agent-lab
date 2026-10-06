# Repository navigation and Kaggle preparation

Date: October 6, 2026. Scope: team onboarding and local preparation, with the existing
evaluation implementation retained.

## Problem and change

The earlier Kaggle generator required a prepared local run and task data. A teammate cloning the
repository without the maintainer's storage could package an agent but could not directly prepare
our evaluation notebook. The new `--agent-dir` / `--task-id` mode packages the chosen configuration
and selects the public task from mounted Kaggle Inputs at runtime. The existing `--run-dir` mode
continues to support comparisons against a frozen H100 run.

The repository map, script catalog, configuration map, evidence catalog, Kaggle walkthrough, and
experiment template now connect the main entry points. All team-facing additions are in English.
Slides remain local and ignored; Overleaf is their separate collaboration location.

## Checks and observed results

| Requirement | Check | Result |
| --- | --- | --- |
| Find maintained files | Compare file lists against the script, configuration, and evidence catalogs | All 29 scripts, 12 configuration files, and 32 existing evidence files covered |
| Navigate the repository | Resolve relative Markdown links in changed/new team guides | Passed |
| Prepare from a fresh checkout | Copy only versioned preparation inputs into a temporary directory; generate outside the maintainer's workspace | Passed without data, weights, environments, or credentials |
| Prepare without Git installed | Run the generator with no Git executable on PATH | Passed; revision recorded as unavailable |
| Keep validation labels out of inference | Execute the generated setup cell against a tiny mounted-input fixture containing reference/test labels | Only public fields written; hash recorded |
| Preserve existing artifacts | Repeat preparation against an existing output directory and repeat runtime setup | Refused overwrite; prior bytes retained |
| Preserve the frozen-run path | Generate from a frozen fixture; alter hash/labels/mounted task | Valid input passed; each mismatch rejected |
| Preserve baseline identity | Compare generated ZIP with the measured baseline SHA256 | Unchanged |
| Preserve H100 behavior | Inspect changes to Slurm, container entry/mount runner, shared evaluation helpers, and baseline | No implementation changes |
| Retain runtime evidence | Recompute hashes for 17 H100 and 20 L4 evidence artifacts in local storage | All 37 matched |

The seven standard-library tests are in [test_kaggle_preparation.py](../../tests/test_kaggle_preparation.py):

```bash
python3 -m unittest discover -s tests -v
```

All seven passed. Python source parsing, shell syntax checks, and `git diff --check` also passed.
Both notebook preparation modes were additionally run against the real baseline and the existing
`rich_3454` H100 run. The frozen public task matched the current local original task record.

Baseline ZIP SHA256:
`02bb939095ded0c970088fc9d19634e203e2d3119a36a5feff20bad55c9343fc`.

## Artifacts and scope

The maintainer's generated preparation examples are local under
`runs/2026-10-06-repository-onboarding/{fresh-notebook,frozen-notebook}/`. They are not shared
downloads; teammates can regenerate them with their own account name using the
[Kaggle walkthrough](../kaggle-runtime.md).

This change did not launch another GPU session or make another competition submission. The generated
installation and evaluation cells, pinned image, official wheel hashes, L4 tensor parallelism, and
memory fraction retain the previously measured setup. The input-preparation change was exercised
locally; end-to-end GPU evidence refers to the October 5 baseline runs. Training, a full benchmark,
and new adapter-loading behavior remain outside this validation.
