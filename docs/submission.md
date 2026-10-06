# From agent configuration to an official submission

The current baseline passed local packaging, Kaggle CPU notebook output, official compilation,
H100 and L4×4 single-task inference, independent repository tests, and official scoring.
Submission **56844353** is **COMPLETE**, with public score **0.12** and no scoring error as checked
on October 5, 2026 at 18:14 UTC. Private score was unavailable. Training and a full local benchmark remain unverified.

## 1. Build the agent ZIP

`configs/baseline/` contains the root agent, analyzer, prompts, sampling, and budgets. It uses the
required model without LoRA; example adapters are not treated as trained team models.

```bash
python scripts/build_submission.py \
  --agent-dir configs/baseline \
  --output runs/pipeline-baseline/submission.zip
```

The builder checks root `agent.yaml`, extensions, symlinks, file count, expanded size, and ZIP CRC.
Fixed ordering, timestamps, and permissions make identical inputs produce identical bytes.
Official compilation is a separate step. `submission.manifest.json` records file and archive hashes.

Verified v1 SHA256: `02bb939095ded0c970088fc9d19634e203e2d3119a36a5feff20bad55c9343fc`.
Regenerate after editing configuration; do not carry forward this hash as if it described a changed bundle.

## 2. Compile and verify a real task

`scripts/validate_agent.py` invokes official `compile_submission`, tool registry, and limits.
It requires the runtime dependencies but does not load the model. `scripts/evaluate_agent.py`
separates inference, which receives sanitized task fields, from verification, which receives reference/test patches.

On October 5, official compilation passed in the isolated container. Evidence is in the local run
`runs/hpc-smoke-20261005-r4/compile/validation.json`. For `rich_3454`, unmodified code produced
1 failed / 82 passed, the reference patch produced 83 passed, and the agent patch produced 84 passed
including one added test. The [H100 manifest](../resource/manifests/hpc-smoke.json) records summaries and hashes.

Follow [runtime preparation](cluster-runtime.md). Retain prediction, trace, patch, complete tests,
and valid unpatched-failure/gold-success controls. Separate environment errors from unsuccessful repairs.
Any change to agent, budget, or adapter requires fresh timing and behavior checks.

### Verified Kaggle L4 run

To prepare a new experiment from a fresh clone, use the [Kaggle runtime walkthrough](kaggle-runtime.md).
The commands below for `jwj1342` refer to the maintainer's historical run.

Private notebook `jwj1342/gemma4-official-runtime-smoke`, version 1, uses the pinned official starter
image, four L4 GPUs, no Internet, and a 2700-second maximum. It checks the same `rich_3454` task and
41 official wheel hashes, then runs compilation, empty/gold controls, inference, and agent verification.

Downloaded logs show 1 failed / 82 passed for empty, 83 passed for gold, and 84 passed for the agent.
Compilation and wheel checks passed. Actual hardware was four NVIDIA L4s. The agent phase took
74.10 seconds with nine LLM calls and seven tool calls, excluding model startup. The notebook reached
smoke completion in about 655 seconds, excluding queue time and final output export.
The ZIP hash matches the formal submission. See [L4 evidence](../resource/manifests/kaggle-runtime-smoke.json).

This is one-task verification, not a new competition upload. `prepare_kaggle_runtime.py` generates
the notebook. Kaggle uses the official subprocess backend with platform-mounted inputs; its behavior
does not establish the same filesystem isolation as the local Apptainer adaptation.
One task does not validate aggregate performance or the full scoring budget.

Query or download the existing run without creating a new version:

```bash
bash -l scripts/kaggle.sh kernels status jwj1342/gemma4-official-runtime-smoke
# Download after COMPLETE or ERROR; do not push again merely because a run is queued.
bash -l scripts/kaggle.sh kernels output jwj1342/gemma4-official-runtime-smoke \
  -p runs/kaggle-runtime-smoke/kaggle-output
```

After `COMPLETE`, inspect actual tests and summaries. After `ERROR`, locate the failed stage:
installation, model startup, inference, or verification. Notebook status does not replace the official submission score.

## 3. Private CPU packaging notebook

`scripts/prepare_kaggle_upload.py --help` describes the generator. It embeds a reviewed small ZIP,
checks its bytes, and writes `/kaggle/working/submission.zip` without GPU or Internet.
Large adapters need versioned private inputs instead of inline embedding. This notebook performs packaging only.

The following v1 operations already completed; the push command is historical and creates a new version if rerun:

```bash
bash -l scripts/kaggle.sh kernels push -p runs/pipeline-baseline/kaggle -t 600
bash -l scripts/kaggle.sh kernels status jwj1342/gemma4-pipeline-baseline
bash -l scripts/kaggle.sh kernels output jwj1342/gemma4-pipeline-baseline \
  -p runs/pipeline-baseline/kaggle-output
```

The run reached `COMPLETE`, and its downloaded ZIP matches the local ZIP byte for byte.
See [packaging evidence](../resource/manifests/kaggle-pipeline.json) and the
[private notebook](https://www.kaggle.com/code/jwj1342/gemma4-pipeline-baseline).

## 4. Official competition upload

The snapshot allows one upload per day. Complete compilation, real-task checks, and budget review first.
The October 5 API reported `is_kernels_submissions_only=false`, open submissions, joined membership,
and one remaining submission. Direct ZIP upload was therefore supported without a packaging notebook.
The [route snapshot](../resource/manifests/submission-route.json) is historical; recheck before uploading.

The first ZIP was accepted at 06:08 UTC as submission **56844353**: no LoRA, 3695 bytes,
and the hash above. See [receipt](../resource/manifests/competition-submission.json).
The [result](../resource/manifests/competition-result.json) is COMPLETE with public score 0.12;
no private score was returned. The old remaining-submission count is not the current allowance.

```bash
bash -l scripts/kaggle.sh competitions submit --help
bash -l scripts/kaggle.sh competitions submissions gemma-4-developer-agent -v
```

`submit_competition.py` rechecks smoke evidence, task identity and hash, frozen configuration, ZIP and
member hashes. Upload mode additionally checks the current submission route and remaining allowance.
The dry-run works on a laptop with Python 3.10+ and the standard library. Use the complete downloaded
Kaggle `runtime-smoke/` directory or a completed H100 run:

```bash
python3 scripts/submit_competition.py --run-dir runs/first-kaggle-smoke/output/runtime-smoke
# After team selection, use your Python environment with Kaggle installed and authenticated.
python3 scripts/submit_competition.py --run-dir runs/first-kaggle-smoke/output/runtime-smoke --submit
# Query existing results without consuming another submission.
python3 scripts/check_submission_status.py --submission-ref 56844353
```

For H100, replace the example run directory with `runs/<completed-experiment>`. The container runner
also verifies frozen inputs before each phase. These checks detect accidental changes; they do not
authenticate results against a maliciously rewritten set of manifests. Keep reviewed experiment
receipts and artifact hashes. Historical October 5 L4 outputs predate the complete input layout;
see [Kaggle output compatibility](kaggle-runtime.md#4-read-the-result-and-keep-the-evidence).

Before upload it exclusively creates `competition-submission.json`, recording hash and request state.
An existing receipt prevents duplicate upload. After a network error or missing ID, inspect Kaggle history
before retrying; do not delete the receipt to bypass this check.

The current four-minute per-task cap implies about eight hours of generation for 120 tasks,
leaving time for initialization. This is a budget estimate, not a complete L4 measurement.
Record submission ID, model/data/harness versions, ZIP hash, status, score, and error logs.
Query existing uploads while waiting. Select final candidates explicitly; the last upload is not automatically the intended final choice.

Official sources: Evaluation, Rules, and Getting Started; archived in `resource/competition/`.
