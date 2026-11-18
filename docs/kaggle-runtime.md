# Run the team baseline on Kaggle

You need a Kaggle account enrolled in the competition and access to its required inputs. You can
prepare the notebook on a laptop using Python 3.10+ and the standard library. Local model weights,
competition downloads, CUDA, and a Trillium account are unnecessary for this preparation step.

The workflow below runs one public task using the team's agent configuration and shared evaluation
scripts. The October 5 baseline passed on four L4 GPUs; see the [measured result](../resource/manifests/kaggle-runtime-smoke.json).
New agent changes still need their own run.

## 1. Generate your notebook without starting a GPU

From the repository root, replace `YOUR_KAGGLE_USERNAME` with your own account name:

```bash
python3 scripts/prepare_kaggle_runtime.py \
  --owner YOUR_KAGGLE_USERNAME \
  --slug gemma4-first-smoke \
  --agent-dir configs/baseline \
  --task-id rich_3454 \
  --output runs/first-kaggle-smoke/notebook
```

The output contains:

| File | Purpose |
| --- | --- |
| `runtime.ipynb` | Self-contained notebook with the small agent ZIP and shared evaluation helpers |
| `kernel-metadata.json` | Private notebook identity, pinned starter image, L4, offline mode, and input handles |
| `preparation.json` | Agent ZIP hash, source Git revision, chosen task, and preparation status |

The generator refuses to overwrite an existing output directory. Use a new path and a descriptive
slug for each experiment. To try a prompt change, copy the baseline to `configs/experiments/<name>/`
and pass that directory instead. Only small bundles can be embedded; adapters need a separately
implemented, versioned private-input workflow.

An existing frozen H100 run can also be prepared for Kaggle using `--run-dir runs/<run>` instead of
`--agent-dir` and `--task-id`. Its bundle/task hashes are checked, and the notebook compares the frozen
public task with the mounted competition data. This is useful for matched-input comparisons.

### Official wheelhouse updates

The default now accepts either registered official snapshot by exact fingerprints.
For a newer unregistered official update, add `--wheelhouse-mode record` to capture
its actual hashes and continue an explicitly exploratory run. To require one
snapshot, use `--wheelhouse-snapshot official-v29-2026-10-08` in default verified
mode. See [wheelhouse updates](wheelhouse-updates.md) for the distinction between
recording bytes and validating runtime compatibility.

## 2. Check your access and quota

Use your own Kaggle CLI installation and credentials on your laptop. Follow the
[official CLI setup](https://github.com/Kaggle/kaggle-cli#installation) if needed. The maintainer's
`scripts/kaggle.sh` wrapper is specific to Trillium and is unnecessary here.

```bash
kaggle quota
kaggle kernels push --help
```

The archived [competition accelerator page](../resource/competition/upgraded-accelerators.md) specifies
L4×4, Internet disabled, and twice the usual GPU-quota consumption. The account snapshot recorded
30 quota hours, corresponding to about 15 L4×4 wall hours before usage/reservations at that rate.
Your account's current entitlement, remaining time, queue, and rate may differ. Notebook quota is
separate from the competition's scoring-time limit. Stop idle interactive sessions.

## 3. Start one private L4 run

```bash
kaggle kernels push -p runs/first-kaggle-smoke/notebook \
  --accelerator NvidiaL4 --timeout 2700
kaggle kernels status YOUR_KAGGLE_USERNAME/gemma4-first-smoke
```

**Push creates a version and starts execution.** The timeout bounds the run to 45 minutes; queue
time and platform behavior are separate. Wait on the same version instead of pushing again when
it is queued. The [official CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
describes push, status, output, and accelerator selection. Metadata alone should not be used to
assume the selected hardware: the notebook records `nvidia-smi` output.

For a browser-first introduction, Copy & Edit the
[official starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent).
It demonstrates the official workflow. To reproduce this repository's experiment, use the generated
notebook and metadata together through the CLI. Uploading only the `.ipynb` through the UI does not
necessarily preserve the pinned image or attached inputs.

The metadata attaches these inputs:

| Input | Handle |
| --- | --- |
| Competition tasks, snapshots, graphs, and test wheels | `gemma-4-developer-agent` |
| Patched evaluation dependencies | `metric/gemma-4-developer-agent-wheelhouse` |
| Required model revision | `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2` |

Their standard mounted paths appear near the top of the generated notebook. It checks their presence,
selects the requested task, removes validation labels from the inference task, and verifies official
wheel inventory and hashes before installation. Verified mode accepts an exact registered snapshot;
record mode explicitly captures an unregistered update and its package versions. Installation uses
only the captured entries and rechecks their names and hashes before pip runs. No GitHub or Kaggle token
is embedded in the notebook. `git_revision` identifies the preparation checkout; `source_run_git_revision`
identifies the original frozen run, when supplied. Embedded script hashes record the exact helper code,
including any uncommitted edits present during generation.

## 4. Read the result and keep the evidence

After the version reaches `COMPLETE` or `ERROR`:

```bash
kaggle kernels output YOUR_KAGGLE_USERNAME/gemma4-first-smoke \
  -p runs/first-kaggle-smoke/output
```

| Output | What to inspect |
| --- | --- |
| `runtime-smoke/run-manifest.json` | Exact task, public-task hash, bundle hash, and intended resource settings |
| `runtime-smoke/inputs/` | Sanitized task, agent files, exact submission ZIP, and member-hash manifest for submission checks |
| `runtime-smoke/inputs/wheelhouse.json` | Selected policy, matched snapshot, actual wheel hashes, and package versions |
| `runtime-smoke/hardware.csv` | Actual allocated GPU names, memory, and driver |
| `runtime-smoke/installation.log` | Dependency preparation problems |
| `runtime-smoke/compile/validation.json` | Official agent compiler result |
| `runtime-smoke/controls-summary.json` | Original failure and reference-patch success |
| `runtime-smoke/infer/prediction.json` | Generated patch, call counts, errors, and agent-phase duration |
| `runtime-smoke/infer/results/` | Trace, model log, hardware details, and inference result |
| `runtime-smoke/verify/{empty,gold,agent}/` | Independent repository tests and detailed results |
| `runtime-smoke/smoke-summary.json` | `controls_valid`, `pipeline_executed`, and `agent_resolved` |
| `submission.zip` | The exact tested agent bundle, written after smoke completion |

Read the summary even if the notebook is `COMPLETE`: successful execution and a resolved issue
are separate outcomes. If execution fails, inspect the last phase log. A running session retains
its current outputs; export them and use a fresh session/output location before repeating setup.
Record results with the [experiment template](experiments/TEMPLATE.md). Coordinate an official
entry separately through the [submission guide](submission.md).

Check the downloaded run locally without consuming a submission:

```bash
python3 scripts/submit_competition.py \
  --run-dir runs/first-kaggle-smoke/output/runtime-smoke
```

The dry-run uses only the standard library. It verifies task identity, frozen input hashes, official
compiler results, controls, and completed agent tests. An unresolved repair can still pass the pipeline
check; inspect `agent_resolved` before selecting an entry. Add `--submit` only after the team selects
the candidate, using a Python environment with your own Kaggle CLI installation and credentials.

This complete output layout applies to notebooks generated after the October 6 repository audit.
The archived October 5 L4 run lacks some input manifests; preserve it as historical evidence and
generate a new experiment when the checked submission route is needed. Do not invent missing receipts.

## Compare agents on a task cohort

To compare agents on the scorer's hardware, freeze a cohort with `prepare_eval_suite.py` and pass the suite
directory as `--run-dir`. The notebook then runs empty/reference controls, one batch inference with a single
model start, and independent verification for every task. It writes `runtime-smoke/cohort-summary.json` with each
task's controls, outcome, tool calls and agent seconds.

```bash
python3 scripts/prepare_eval_suite.py --run-dir runs/<date>-<name>/<variant> --agent-dir <agent> --cohort fast
python3 scripts/prepare_kaggle_runtime.py --owner YOUR_KAGGLE_USERNAME --slug gemma4-<variant>-fast \
  --run-dir runs/<date>-<name>/<variant> --output runs/<date>-<name>/<variant>-notebook
```

Freezing the suite needs the local task file (`data/raw/.../tasks.jsonl`). A four-L4 session is charged at twice
its wall time, so a 12-task cohort at 8 minutes per task can use about 3–4 hours of quota. Compare variants on
the same frozen tasks and record the quota each run used.

## Resource choices and tuning

These are the existing single-task evaluation settings, not a training recipe or a promise that
every task fits. They preserve one shared agent bundle across both platforms.

| Setting | Kaggle L4×4 | Existing Trillium H100 |
| --- | --- | --- |
| GPU placement | Tensor parallel across 4 GPUs | One GPU |
| Tensor-parallel size | 4 | 1 |
| vLLM GPU-memory fraction | 0.90 per GPU | 0.80 |
| Model context limit | 32,768 tokens | 32,768 tokens |
| Baseline per-task agent budget | 4 minutes, 40 tools, 80 turns | Same |
| Runtime storage | `/kaggle/working/runtime-smoke/` | Run directory on SCRATCH |
| Dependencies | Official pinned Kaggle image and wheelhouse | Prepared Apptainer image and HOME environment |

L4×4 provides four separate GPU memories. Tensor parallelism distributes this model; a single
process does not automatically gain one contiguous 96 GB device. T4×2 is a different configuration
and is not supported by this tested runner.

Save quota by editing/packaging on CPU, validating one task before expanding, and comparing only
one change at a time. Preserve task, seed, context, and budget when comparing hardware. The current
runner starts a model per smoke experiment; use it for small checks before designing a batched
benchmark that can reuse a model process. Keep timing for model startup and agent execution separate.

For an out-of-memory error, locate whether it occurs during model loading, cache allocation, or
generation before changing a parameter. Record any shorter context or lower memory fraction as a
new runtime configuration and recheck behavior. Changing parallel size does not turn evaluation
into multi-GPU training.

For SFT/LoRA, start from the [training plan](training-plan.md). Use a separate offline training
notebook and test a few optimizer steps, save/resume, export, and evaluation reload before spending
the remaining quota. Training the required QAT model and reloading its adapter have not yet been
validated by this project; the evaluation wheelhouse is not a verified training environment.
