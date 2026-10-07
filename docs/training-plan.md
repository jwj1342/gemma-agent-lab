# Training and evaluation preparation

The original plan below is now accompanied by the [first SWE/graph LoRA pilot](first-lora.md).
Use its measured report to distinguish completed training checks from planned benchmark comparisons.
`/home/jwj/venvs/kaggle-cli` is the maintainer's data-access environment. Trillium PyTorch,
SFT, and RL operating guidance is maintained in `/home/jwj/.codex/cluster-guides/trillium.md`.
All environment versions and resource figures below are dated October 5, 2026 unless stated otherwise.

## Separate environments

1. **Competition evaluation:** reproduce the official wheelhouse, patched vLLM, ADK, swegemma,
   model revision, and chat template. Establish offline generation and independent verification.
   Replacing this with the latest Alliance vLLM does not demonstrate scoring parity. The official
   notebook assumes many dependencies are already present in the Kaggle image.
2. **Training:** on Trillium, use modules, Alliance wheels, and a HOME venv. On Kaggle, prepare
   a separate versioned training notebook with dependencies and data attached for offline use;
   follow the resource constraints in the [Kaggle runtime guide](kaggle-runtime.md). Validate Torch,
   Transformers, PEFT, TRL, quantization, and attention extensions together for the target model.
   Exported adapters must load in the evaluation environment with compatible target modules/shapes,
   measurable effect, and a submission below 3 GiB.

A Dockerfile does not give Trillium users a Docker daemon. Apptainer or a controlled subprocess
backend are possible adaptations; subprocess is not the same isolation boundary as Docker/gVisor.
Limit generated commands to a dedicated workspace, omit download credentials, and prevent agent
writes to HOME or project source. Our Apptainer/subprocess path passed mount, writable-path,
compiler, offline-loading, and empty/gold checks, followed by patch generation and independent
verification on `rich_3454`. See [runtime status](cluster-runtime.md). This validates one-task evaluation, not training.

## Proposed sequence

1. Fix grouped public splits and establish unpatched failure plus gold-patch success for chosen
   tasks. Exclude incomplete or invalid environments. Group by repository/base commit to prevent snapshot overlap.
2. Stage the required v2 base, tokenizer/configuration, data, and caches. Use a short allocated
   H100 run or a Kaggle L4×4 session for inference and two-stage evaluation; retain patches, traces,
   timing, and failure categories.
3. Establish an analyzer/coder baseline without an adapter, then check timing near the L4 scoring environment.
4. Begin SFT with verified tool-use trajectories. Check chat templates, tool schemas, and loss masks;
   run a few steps, save, resume, export, and reload. Issue-to-reference-diff training alone does not teach the entire agent loop.
5. Consider DPO after obtaining reliable preference pairs, and GRPO after making generation and
   sandbox-test costs stable and affordable. Separate passing repairs, environment errors, empty patches,
   format errors, and timeouts in reward records; prevent rewards for changing the test environment.

| Official reference | What to verify |
| --- | --- |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) | Message/tool trajectories, assistant loss masks, packing |
| [PEFT quantization](https://huggingface.co/docs/peft/en/developer_guides/quantization) | Training preparation and LoRA target modules; W4A16 is not NF4 |
| [TRL DPO](https://huggingface.co/docs/trl/en/dpo_trainer) | Chosen/rejected pairs, reference models, memory budget |
| [TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer) | Multiple candidates, tool environments, custom rewards, SWE integration |
| [TRL vLLM](https://huggingface.co/docs/trl/en/vllm_integration) | Server/colocate modes, weight synchronization, version compatibility |
| [PyTorch torchrun](https://docs.pytorch.org/docs/main/elastic/run.html) | Ranks, process hierarchy, Slurm integration |
| [Lightning checkpoints](https://lightning.ai/docs/pytorch/2.5.0/common/checkpointing_basic) | Full-state recovery with `fit(..., ckpt_path=...)` |

At review time, TRL documentation listed vLLM 0.20–0.30 support, while the competition supplied patched
0.19.1. Recheck versions at implementation time and keep incompatible dependency sets separate.

## Account resources and storage

Track both byte and file-count quotas. October 5 `diskusage_report` limits were about 10% below
contemporaneous `df` limits and an earlier report. The cause remains unconfirmed; use the smaller
reported quota as the working budget. On other systems, `df` may describe the whole filesystem.

| Storage | Reported quota / files | Usage before model download, rounded | Purpose |
| --- | --- | --- | --- |
| HOME | 100 GiB / 1 million | 1.42 GiB / 14,000 | Environments/configuration; read-only on compute nodes |
| SCRATCH | 25 TiB / 10 million | 41 GiB / 2,731 | Data, weights, caches, checkpoints; no backup |
| PROJECT `def-zhouyang` | 1 TiB / 2 million, shared | 0 B / 4 | Important source/results; read-only on compute nodes |

At the same time, `df` showed HOME 110 GiB / 1.1 million files, SCRATCH 27.5 TiB / 11 million,
and PROJECT about 1126.4 GiB / 2.2 million. Do not budget the extra 10%. Disk space is separate
from CPU RAM and GPU VRAM. Subsequent model/container downloads changed the 41 GiB usage figure;
query `diskusage_report` again and allow for reporting delays.

Original data plus ZIP occupy about 41.30 GiB. Full snapshot extraction adds about 21.13 GiB and
238,000 entries. The v2 model package is about 21.70 GiB. Budget additional training weights,
conversion copies, optimizer/checkpoints, and per-run caches; on-disk weight size does not equal training VRAM.
After environment preparation, a report showed HOME 6094 MiB / 96K files, SCRATCH 81 GiB / 4885 files,
and PROJECT 0 B / 4 files. See `resource/manifests/diskusage-2026-10-05-runtime.txt`.
Temporary builds and reporting delays make these historical observations unsuitable as current free-space guarantees.

Compute nodes have no Internet and read-only HOME/PROJECT. RAM disks consume memory. Put runtime
HF/Torch/Triton/XDG caches, logs, and checkpoints in the run's SCRATCH path and disable online tracking.
Keep credentials outside team caches. Recheck GPU states and account access; this account's verified default was `def-zhouyang`.

## Completed prerequisites and remaining work

- Required v2 base and all official wheels are downloaded and verified. H100 offline loading succeeded;
  vLLM reported 19.62 GiB for weights, excluding KV cache and other peak usage.
- The container passed official compilation, empty/gold controls, and independent generated-patch tests.
  Missing Git was fixed. Job 1033882 completed with 84 passing agent-patch tests. Formal submission
  56844353 scored 0.12 publicly, and a Kaggle L4×4 single-task check also passed.
- The October 7 pilot prepared a separate HOME training environment, trained rank-8 LoRA on one H100,
  and validated full-state save/resume and export. See [the measured first LoRA report](first-lora.md).
- Multi-GPU training and broad held-out repair gains remain unvalidated. Pilot NLL gains do not establish them.
- Larger-corpus training, target-harness trajectory replay, and matched graph ablations remain the next research decisions.

Build grouped validation first, then decide SFT/RL scale. The measured L4 agent phase was 74.10 seconds,
excluding startup; this timing does not validate training.

## Lessons from public training examples

See all [individual reviews](../resource/notebooks/reading-notes.md). Evaluation reliability remains the first priority.

- The 31B QLoRA study has a real training entry and prompt loss mask, but uses gold hunks to locate
  context and a bitsandbytes 4-bit base rather than official QAT W4A16 serving. Evaluate known-location
  patch generation separately from full agent behavior with unknown locations.
- Issue-to-diff samples often omit source observations and tool results. Lower loss alone cannot
  demonstrate improved multi-step behavior. Estimate gold-file priors only from the training split.
- Renaming a Gemma 2 export or generating random A / zero B tensors does not establish compatible
  Gemma 4 training. Check trainable tensors, gradients, save/resume, and fixed-input outputs with adapters enabled/disabled.
- Start with a few steps and independent validation before scaling SFT. Community titles or placeholder
  weights do not justify large RL runs. Keep dependency failures, skipped required tests, and test-environment
  changes out of reward shortcuts.
