# Trillium evaluation runtime

This page records the project's local implementation. Read
`/home/jwj/.codex/cluster-guides/trillium.md` before Trillium operations.
Run model inference and repository tests in compute allocations, not on login nodes.
All results below are historical observations from October 5, 2026.

## Prepared components

- Required model v2: eight files. Official evaluation wheelhouse: 41 files. Sizes match the
  official inventory; SHA256 hashes, wheel CRCs, and safetensors offset bounds were checked.
  [Asset verification](../resource/manifests/eval-assets-verification.json) records structural checks, not model execution.
- Apptainer image preparation, HOME environment setup, isolated execution, and a single-GPU Slurm smoke script are available.
- The image uses a pinned amd64 digest of `vllm/vllm-openai:v0.19.1`; CPU Torch import passed.
  See [image manifest](../resource/manifests/vllm-image.json) and local `runs/prepare-eval-image.log`.
- The native `gemma4-compiler` environment passed import, directory-rule, and model-name checks only.
  It loads `StdEnv/2023 python/3.12.4 arrow/25.0.0`. Its Google GenAI/dependency combination
  supports preliminary checks and does not reproduce the full Kaggle runtime.
- The HOME container environment has a dependency lock and the known metadata exceptions below.
  Official compilation, single-task GPU inference, and independent patch tests passed.

## Preparation

Run large downloads and image/environment preparation on `trig-dm1`. Do not duplicate an active build.
These commands refer to the maintainer's existing workspace; renaming GitHub does not move it.

```bash
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/download_eval_assets.sh'
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/prepare_eval_image.sh'
# After the image succeeds, install official wheels and missing dependencies.
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/prepare_eval_environment.sh'
ssh -o BatchMode=yes trig-dm1 \
  'bash -l /scratch/jwj/kaggle_gemma4_agent/scripts/prepare_eval_git.sh'
```

Some wheel filenames omit CUDA local-version suffixes. The installer reads internal metadata and
uses temporary accurate filenames, preserving the originals. It skips reinstalling only when the
recorded source SHA256 matches the official file; matching version numbers alone cannot establish
that competition patches are installed. Validate the actual Torch/CUDA/vLLM combination.

### Known wheelhouse conflicts

The October 5 installation found that the official wheels cannot all be resolved together by pip:

- `google-adk 1.36.1` requires OpenTelemetry SDK <=1.41.1, while the supplied OTLP 1.44.0
  metapackage requires the 1.44 series. The adaptation retains the image's 1.41.0 runtime and disables telemetry.
- `prometheus-fastapi-instrumentator 8.1.0` declares Starlette >=1, while ADK requires <1.
- `xgrammar 0.2.5.post1` declares Transformers <5, while the wheelhouse supplies Transformers 5.13.1.

The official starter installs with `--no-deps`. Our adaptation also preserves the contents of all 41
wheels, separately resolves missing ADK/LiteLLM/test dependencies, and constrains official package
versions to avoid replacement by same-name PyPI releases. Cutlass DSL is pinned to the 4.6.2 required
by `quack-kernels`. Raw `pip check` results and explanations are in
[the dependency audit](../configs/eval-container.requirements.dependency-audit.json).
Unlisted new conflicts fail preparation. Unrelated image PyGObject/pycairo omissions are recorded separately.
These exceptions are not a clean dependency check. Successful execution covers the tested paths only,
not every extension, training configuration, or task. Kaggle verification retains the official image and install procedure.

Image inspection confirmed Torch 2.10.0+cu129 / CUDA 12.9; the observed GPU-login driver was
580.173.02. The allocated H100 run loaded and executed the model. The base image lacks pytest,
and the subprocess backend inherits its Python environment instead of performing Docker's per-task
installation. Preparation adds pytest 8.3.4 from the test wheelhouse and Dockerfile-specified
pytest-timeout 2.1.0. These local choices do not establish complete parity with private scoring dependencies.

The image provides `python3` / `python3.12`, with no `python` alias. Executing `/usr/bin/python3.12`
confirmed Python 3.12.13 and Torch 2.10.0+cu129. The original post-build check failed because it used
the absent alias, although SIF construction had succeeded. The entry point was corrected and checked
without rebuilding; see local `runs/inspect-eval-image-python3.log` and the image manifest.

Git was also absent. In job 1033766 the model edited code, but `submit_patch` could not invoke Git
and eventually timed out. Signed Ubuntu Git/PCRE2 packages were extracted without root into the
HOME venv's `vendor/git/`, exposed through a wrapper. The [Git manifest](../resource/manifests/eval-git.json)
records package hashes and init/add/commit/diff checks. Runtime prechecks now validate required commands
and patch export before loading the model.

## Isolation and writable paths

The runner uses `--containall --no-home`, disables default host/admin/cwd mounts, and binds only
required directories. Model, agent, scripts, prebuilt HOME venv, and inputs are read-only.
Inference receives only public fields from `prepare_eval_task.py`; verification separately receives labels.
Each phase writes only to its own run directory. HF/Torch/Triton/XDG caches and temporary files live
on SCRATCH. Inference does not mount Kaggle credentials, the complete project, or verification results.

The site exports `APPTAINER_BIND=/project,/scratch`; `--cleanenv --no-mount bind-paths` does not
remove those user-specified mounts. The dedicated runner clears inherited Apptainer/Singularity
bind/mount variables before passing its explicit list. October 5 checks confirmed read-only agent,
scripts, and venv; inaccessible credentials and complete project; writable run/tmp paths.
Evidence: local `runs/hpc-smoke-20261005/compile/runtime-filesystem.json`.

This is the official subprocess backend inside Apptainer. Its directory isolation does not establish
all Docker/gVisor scoring security properties. Mount behavior, GPU loading, and real single-task tests were checked.

## Reproduce the single-task check

Create a new run, freeze configuration, build the ZIP, and remove validation labels from inference inputs:

```bash
module load StdEnv/2023 python/3.12.4
gemma_root=$(pwd -P)
gemma_run="$gemma_root/runs/$(date -u +%Y%m%dT%H%M%SZ)-rich3454"
python scripts/prepare_eval_run.py --run-dir "$gemma_run" \
  --agent-dir configs/baseline --task-id rich_3454
```

Use a fresh directory. Existing runs preserve evidence; inspect them and choose a new name for a retry.
The runner mounts `runs/<experiment>/inputs/agent/`, so later baseline edits cannot change the run.
Preparation only copies and records inputs; it does not execute models or tests.

Submit from the GPU login node with an explicit account and writable SCRATCH logs. Leave the ordinary
GPU partition selection to Slurm. In the same shell, for this account:

```bash
sbatch --parsable --account=def-zhouyang \
  --output="$gemma_run/slurm-%j.out" --error="$gemma_run/slurm-%j.err" \
  "$gemma_root/scripts/slurm_eval_smoke.sh" "$gemma_root" "$gemma_run" rich_3454
```

Record the returned job ID. Use `squeue -j JOBID` while queued/running and
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList` afterward. The successful historical run
used `debug`; the command above uses the normal queue and omits temporary debugging node exclusions.
Missing logs alone are not a reason to resubmit.

After the job finishes:

```bash
python scripts/summarize_eval_smoke.py --run-dir "$gemma_run"
```

`pipeline_executed=true` means all stages ran; `agent_resolved=true` means the patch solved the issue.
For failures, inspect `slurm-*.err`, `infer/results/failure.json`, `infer/results/vllm.log`, and each
phase's `tests.log`. Fix environment problems in a new run and preserve failed evidence.

Other accounts must verify their own allocation, quotas, modules, and access. HOME paths in
`prepare_eval_environment.sh`, `prepare_eval_git.sh`, `run_eval_container.sh`, and `container_entry.sh`
need adaptation. Cloning does not copy environments, ignored data, or credentials.

Execution order is official compilation, empty/gold controls, inference, and independent agent-patch
verification. `summarize_eval_smoke.py --controls-only` requires genuine unpatched test failure and
a passing gold patch before model startup. Retain dependency locks, versions, memory/time measurements,
traces, patches, and tests before declaring the path verified.

## Measured result: October 5, 2026

Run `hpc-smoke-20261005-r4`, job **1033882**, used one H100 and 24 CPU cores on `trig0001`.
It finished `COMPLETED / 0:0` in **5 minutes 49 seconds**.

| Check | Result |
| --- | --- |
| Official compiler, isolation, and Git prechecks | Passed |
| Unmodified code with validation tests | 1 failed / 82 passed; reproduced the URL-highlighting issue |
| Reference patch with validation tests | 83 passed |
| Model inference | 19 LLM calls, 12 tool calls, 1,161-byte patch; 83.85 seconds including task preparation |
| Agent patch with independent verification | 84 passed, including one new regression test |

The patch adds `@` to the URL regex and a regression test, without removing or skipping existing tests.
Four edit calls initially omitted `old_string`; the model recovered. This is one observation about
call formatting, not a baseline-wide failure distribution. Reported weight loading was 19.62 GiB,
not peak VRAM. Settings: TP=1, BF16, 32,768 context. Training, LoRA, checkpoint/resume, and multi-GPU training remain unverified.

The [summary](../resource/manifests/hpc-smoke.json) references `compile/validation.json`,
`infer/prediction.json`, `infer/results/traces/`, `infer/results/vllm.log`,
`verify/{empty,gold,agent}/tests.log`, and `smoke-summary.json` in the original run.
For subsequent formal score and L4 evidence, see the [submission guide](submission.md).
Single-task success alone does not validate a full benchmark or the global L4 budget.

| Historical job | Outcome |
| --- | --- |
| 1033745 | Node prolog reported an unresponsive GPU; system cancellation, not a model result |
| 1033752 | Precheck found inherited binds exposed all SCRATCH; fixed by clearing them in the dedicated runner |
| 1033766 | Model edited code, but missing Git prevented patch export; added Git and a patch precheck |
| 1033882 | Complete single-task path passed after those fixes |
