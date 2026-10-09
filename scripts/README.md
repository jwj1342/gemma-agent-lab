# Maintenance and execution scripts

Start with the [Kaggle walkthrough](../docs/kaggle-runtime.md) or the
[Trillium walkthrough](../docs/cluster-runtime.md). Run commands from the repository root.
"Local" below means the preparation step needs only Python 3.10+ and the standard library.
The generated Kaggle notebook subsequently runs in Kaggle's evaluation environment.

## Common entry points

| Script | Where | Purpose |
| --- | --- | --- |
| [build_submission.py](build_submission.py) | Local | Deterministic agent ZIP and hash manifest; official compilation is separate |
| [check_repository.py](check_repository.py) | Development Python + PyYAML and Bash | CPU checks for tracked syntax, configuration references, deterministic agent packaging, and maintained-document links |
| [prepare_kaggle_runtime.py](prepare_kaggle_runtime.py) | Local | Private offline L4×4 notebook from `--agent-dir` plus `--task-id`, or an existing `--run-dir`; no local model/data needed in agent mode |
| [prepare_kaggle_upload.py](prepare_kaggle_upload.py) | Local | Private CPU notebook that emits an already reviewed small ZIP; no model inference |
| [prepare_eval_run.py](prepare_eval_run.py) | Local, with task data | Freeze agent, sanitized public task, and ZIP identity into a new run |
| [slurm_eval_smoke.sh](slurm_eval_smoke.sh) | Trillium GPU allocation | Existing H100 compile → controls → inference → independent tests workflow |
| [prepare_eval_suite.py](prepare_eval_suite.py) | Local, with task data | Freeze a candidate, public cohort, and runtime scripts for repeated evaluation |
| [build_prompt_candidate.py](build_prompt_candidate.py) | Any machine, without models/data | Build workflow, ICL and reflection ablations |
| [slurm_eval_suite.sh](slurm_eval_suite.sh) | Trillium GPU allocation | One model startup per batch, cached controls, independent grading, and paired comparison |
| [summarize_eval_suite.py](summarize_eval_suite.py) | Local, with completed runs | Coverage, resolution, cost, and task-by-task gains/regressions |
| [check_kaggle_quota.py](check_kaggle_quota.py) | Kaggle CLI environment | Read quota with complete duration values and save a dated JSON report |
| [check_submission_status.py](check_submission_status.py) | Kaggle CLI environment | Query existing submission/notebook state without uploading again |
| [submit_competition.py](submit_competition.py) | Prepared run + Kaggle CLI | Recheck evidence, ZIP, and upload route; dry-run by default, `--submit` sends an entry with a duplicate-prevention receipt |

## Shared evaluation helpers

| Script | Purpose |
| --- | --- |
| [wheelhouse_runtime.py](wheelhouse_runtime.py) | Match registered wheelhouse snapshots or explicitly record new inputs; recheck captured bytes before installation |
| [validate_agent.py](validate_agent.py) | Invoke the official compiler and restrictions in the prepared runtime |
| [evaluate_agent.py](evaluate_agent.py) | Separate inference from verification; explicit tensor parallelism and GPU-memory fraction |
| [prepare_eval_task.py](prepare_eval_task.py) | Export inference fields while removing reference/test patches and labels |
| [verify_eval_run.py](verify_eval_run.py) | Check frozen task identity/hash and agent/ZIP member hashes before execution or submission; standard library only |
| [summarize_eval_smoke.py](summarize_eval_smoke.py) | Check empty/gold controls and report pipeline execution separately from issue resolution |
| [define_eval_splits.py](define_eval_splits.py) | Freeze repository/snapshot groups after auditing known SWE/graph exposure |
| [eval_suite_lib.py](eval_suite_lib.py) | JSON, grouping, result integrity, control-cache validation, and descriptive intervals |
| [run_eval_suite.py](run_eval_suite.py) | Versioned asset identities, cache reuse, run locking, resume, and isolated execution phases |
| [local_eval_setup.py](local_eval_setup.py) | Version-checked local setup hook: isolated task dependencies, editable workspace install, and import-origin checks |
| [download_eval_test_wheels.py](download_eval_test_wheels.py) | Stage small pinned public-test dependency wheels using upstream hashes |

Kaggle embeds these evaluation helpers in its generated notebook. H100 runs them in the prepared
container. An inference result is not a competition score, and a packaging check is not a GPU test.

## Trillium installation and execution

These wrappers describe the maintainer's existing installation. Read their paths and the cluster
guide before using another account. Preparation involving large downloads/builds belongs on a datamover;
inference and repository tests belong in Slurm allocations.

| Script | Purpose |
| --- | --- |
| [kaggle.sh](kaggle.sh) | Load the maintainer's CLI environment and read credentials from HOME |
| [download_competition.sh](download_competition.sh) | Download the complete archive and verify extraction against its inventory |
| [download_eval_assets.sh](download_eval_assets.sh) | Download required model and evaluation wheelhouse |
| [prepare_eval_image.sh](prepare_eval_image.sh) | Build/inspect the pinned Apptainer image |
| [prepare_eval_environment.sh](prepare_eval_environment.sh) | Prepare the separate HOME evaluation environment |
| [install_eval_environment.py](install_eval_environment.py) | Install pinned official wheels and missing dependencies; record source hashes and known metadata conflicts |
| [prepare_eval_git.sh](prepare_eval_git.sh) | Prepare the container Git dependency and record provenance |
| [install_eval_git.sh](install_eval_git.sh) | Extract signed Ubuntu Git/PCRE2 packages and check Git behavior in the prepared environment |
| [run_eval_container.sh](run_eval_container.sh) | Apply explicit read-only input mounts and writable per-phase run paths; enforce Slurm for execution |
| [container_entry.sh](container_entry.sh) | Activate the prepared container environment and set offline/writable-cache settings |

The maintainer's lightweight CLI environment is:

```bash
module load StdEnv/2023 python/3.12.4
source /home/jwj/venvs/kaggle-cli/bin/activate
```

Use your own CLI on a laptop; these module commands are specific to Trillium.

## Collection, auditing, and documentation

| Script | Purpose |
| --- | --- |
| [extract_competition.py](extract_competition.py) | Check archive paths, names, sizes, and CRC; overwrites matching destination input files |
| [verify_eval_assets.py](verify_eval_assets.py) | Model/wheel hashes, ZIP CRCs, and safetensors offsets; potentially heavy I/O |
| [audit_data.py](audit_data.py) | Inspect task/graph schemas, NPZ headers, wheel metadata, and archive members; use a datamover |
| [collect_resources.py](collect_resources.py) | Capture official pages, public Code, discussion replies, and leaderboard CSV using Kaggle APIs |
| [collect_discussion_bodies.py](collect_discussion_bodies.py) | Capture topic opening posts, authors, and full reply trees |
| [index_resources.py](index_resources.py) | Regenerate English navigation/coverage; `--navigation-only` preserves existing receipt files |
| [render_notebook_reading.py](render_notebook_reading.py) | Render structured English reviews with source-hash and evidence-line checks |

Collectors cache files. Explicitly choose which snapshot to refresh; rerunning one command does not
refresh the entire collection. A complete update also requires separate competition-file, Models-page,
wheelhouse, and Code-structure inventories. Preserve dated evidence and inspect changes before updating conclusions.

## SFT pilot

Read [the first LoRA guide](../docs/first-lora.md) for data boundaries and validation gates.

| Script | Purpose |
| --- | --- |
| [download_swesmith_pilot.sh](download_swesmith_pilot.sh) | Download a pinned public trajectory shard and verify its upstream SHA-256 |
| [prepare_sft_data.py](prepare_sft_data.py) | Convert successful SWE-smith tool actions and group splits by repository/snapshot |
| [prepare_graph_sft.py](prepare_graph_sft.py) | Execute official graph tools to construct component demonstrations without gold patches |
| [prepare_terminal_sft.py](prepare_terminal_sft.py) | Execute official test/submission/status tools on controlled Git fixtures and label a final answer |
| [prepare_lora_run.py](prepare_lora_run.py) | Freeze training source and configuration before submitting |
| [train_first_lora.py](train_first_lora.py) | Assistant-action/terminal LoRA SFT, optional adapter initialization, full recovery state, held-out NLL, and export checks |
| [slurm_first_lora.sh](slurm_first_lora.sh) | Allocated, offline H100 pilot with a fresh-process resume |
| [check_lora_serving.py](check_lora_serving.py) | Official quantized-vLLM reload and fixed held-out action comparison |
| [slurm_check_lora.sh](slurm_check_lora.sh) | Isolated allocated serving compatibility check |
