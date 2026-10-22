# Evidence and inventory catalog

These records are dated observations. Check `recorded_at`, `checked_at`, or the filename before
using them. A successful file audit, notebook status, agent run, and official score prove different things.
Paths under `runs/`, data/model storage, HOME, and local slides refer to the maintainer's private
workspace; those artifacts are not included in a Git clone.

## Results and submission receipts

| File | Purpose |
| --- | --- |
| [competition-result.json](competition-result.json) | Completed official score for submission 56844353 |
| [competition-submission.json](competition-submission.json) | Original upload receipt and tested ZIP identity |
| [hpc-smoke.json](hpc-smoke.json) | H100 allocation, one-task results, controls, and artifact hashes |
| [local-evaluation.json](local-evaluation.json) | Matched baseline/LoRA batch outcomes, cache reuse, timings, and verified resume |
| [kaggle-runtime-smoke.json](kaggle-runtime-smoke.json) | L4×4 hardware, one-task results, timing, and artifact hashes |
| [kaggle-wheelhouse-v29-smoke.json](kaggle-wheelhouse-v29-smoke.json) | October 9 L4×4 validation of the updated official wheelhouse, controls, agent repair, and artifact hashes |
| [kaggle-pipeline.json](kaggle-pipeline.json) | CPU packaging notebook and byte-identical ZIP checks |
| [kaggle-evaluation-status.json](kaggle-evaluation-status.json) | Dated queries of existing notebook/submission state |
| [submission-route.json](submission-route.json) | Historical account eligibility and upload-route discovery |
| [pipeline-local-checks.json](pipeline-local-checks.json) | Local packaging/preparation checks before GPU evaluation |

## Data and reference inventories

| File | Purpose |
| --- | --- |
| [competition-files.json](competition-files.json) | Official competition filenames and sizes |
| [download-verification.json](download-verification.json) | Downloaded-file verification against the inventory |
| [data-audit.json](data-audit.json) | Combined structural audit of the competition data |
| [tasks-audit.json](tasks-audit.json) | Task-record fields and distributions |
| [graphs-audit.json](graphs-audit.json) | Code-graph structure |
| [embeddings-audit.json](embeddings-audit.json) | Embedding array headers and dimensions |
| [snapshots-audit.json](snapshots-audit.json) | Repository archive structure |
| [wheels-audit.json](wheels-audit.json) | Competition test-wheel inventory inspection |
| [sample-adapters.json](sample-adapters.json) | Supplied adapter examples; not team training evidence |
| [training-wheels-sft-v1.json](training-wheels-sft-v1.json) | Hashed offline SFT wheel inventory (Linux/CPython 3.12, `interpreter_abi: cp312`) and the version constraints measured by `scripts/check_training_stack.py`; `scripts/record_training_wheels.py` refuses to record a set whose binary wheels carry another cp3xx tag |
| model-modules.json (**not yet recorded**) | Real projection names and shapes of the loaded base model, from `scripts/probe_model_modules.py`; takes precedence over the sample adapter as LoRA target evidence once it exists |
| [first-lora.json](first-lora.json) | First actual SWE/graph SFT, recovery, export, and separately recorded evaluation checks |
| [harness-wheels.json](harness-wheels.json) | Official harness package metadata |
| [research-collection.json](research-collection.json) | Coverage of pages, notebooks, discussions, models, and rankings |

## Runtime installation and account observations

| File | Purpose |
| --- | --- |
| [wheelhouse-files.json](wheelhouse-files.json) | Official evaluation dependency inventory |
| [local-eval-test-wheels.json](local-eval-test-wheels.json) | Pinned supplemental wheels and upstream hashes for isolated local repository tests |
| [eval-assets-verification.json](eval-assets-verification.json) | Historical base model and wheel hashes; preserved for earlier runs |
| [kaggle-wheelhouse-snapshots.json](kaggle-wheelhouse-snapshots.json) | Registered old/new wheel inventories used by notebook preparation; compatibility status recorded separately |
| [eval-wheelhouse-2026-10-09.json](eval-wheelhouse-2026-10-09.json) | Historical October 9 official wheel hashes used by local adapter-notebook preparation before the registered snapshot workflow |
| [eval-git.json](eval-git.json) | Container Git installation, package identity, and basic behavior checks |
| [vllm-image.json](vllm-image.json) | Prepared Apptainer image identity and checks |
| [vllm-image-inspect.json](vllm-image-inspect.json) | Container inspection metadata |
| [vllm-wheel-metadata.txt](vllm-wheel-metadata.txt) | Raw dependency metadata from the official vLLM wheel |
| [accelerators-recheck.json](accelerators-recheck.json) | Historical account accelerator availability |
| [storage-2026-10-05.json](storage-2026-10-05.json) | Dated local storage observation |
| [diskusage-2026-10-05.txt](diskusage-2026-10-05.txt) | Initial account quota report |
| [diskusage-2026-10-05-post-download.txt](diskusage-2026-10-05-post-download.txt) | Quota report after data download |
| [diskusage-2026-10-05-runtime.txt](diskusage-2026-10-05-runtime.txt) | Quota report after runtime preparation |
| [workspace-verification.json](workspace-verification.json) | Historical workspace checks |
| [completion-audit.json](completion-audit.json) | Historical audit of the earlier research/slides work; some paths and remote-state facts have since changed |

Preserve historical receipts when creating a new run. Place new compact reports in
[docs/experiments/](../../docs/experiments/README.md), with commands, hashes, and accessible artifact
locations. Old account quota or submission allowance does not establish today's remaining resources.
