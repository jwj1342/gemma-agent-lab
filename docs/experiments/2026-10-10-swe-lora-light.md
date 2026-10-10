# Bounded SWE LoRA training and evaluation

The revised experiment uses 1,024 training actions and one epoch, with a complete
agent check after 32 updates. The user requested a small training/evaluation loop
before further scaling. The earlier four-GPU replacement jobs (`1071755`,
`1071772`, `1071773`, `1071774`) were cancelled before starting; their elapsed
compute time is zero. Existing artifacts and provenance are retained.

## Registered protocol

- Required base: `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`.
- Use the corrected native tool-handoff supervision, rank 16 / alpha 32,
  text attention and MLP projections, LR 5e-5, and a 4,096-token maximum.
- Reuse the pinned raw sources and snapshot exclusions. After encoding and
  deduplication, select 896 SWE, 112 graph, and 16 terminal actions. Round-robin
  over snapshot/tool strata with deterministic hashed ordering and seed 42.
  Preserve every selected action's recent context and complete target.
- Keep the validation population unchanged; use a fixed panel of up to 32
  examples per source for checkpoint selection. Keep the protected holdout sealed.
- One H100, microbatch one, accumulation four, one epoch: 256 optimizer updates.
  Save/restart at step eight, stop at step 32, and test the partial adapter before
  continuing. This is a new adapter and schedule; no old optimizer is reused.
- Native-tokenizer checks cover every retained sample. Record representative
  target prefixes/suffixes in `encoded-data/protocol-examples.json` for inspection.
  The official template ends pending calls at token 50 and final answers with
  `<turn|>` followed by a newline. Prompt tokens remain masked.

The partial adapter must pass the official loading/effect checks and autonomous
agent runs on the predeclared exposed tasks `rich_3454`, `fastapi_14455`, and
`requests_7502`, each with empty/reference controls. The gate requires valid,
nonempty, clean execution artifacts; issue resolution is reported separately.
A failed task is retained and investigated rather than replaced. Passing these
three probes only permits the remaining short epoch, not a performance claim.

After the epoch, the selected adapter must pass the existing release checks and
the unchanged fast cohort before a competition upload. Preserve the submitted
ZIP identity and actual upload receipt. Use the baseline prompt; the separate ICL
candidate is not promoted by its measured development result.

Configuration: [swe-lora-light.json](../../configs/training/swe-lora-light.json).
Launchers: [training](../../scripts/slurm_swe_lora_light.sh) and
[partial-agent validation](../../scripts/slurm_preflight_swe_lora_light.sh).
These launchers are Trillium-specific; dataset selection and the trainer are
separate Python modules.

## Cost evidence and budget

The earlier final continuation occupied four H100s for 9h50m11s, or 39.35 physical
GPU hours, excluding earlier preparation/recovery and release validation. Its
two epochs processed 92,648,546 input tokens, including repeated recent history,
although only 4,680,468 target tokens contributed to the loss. Masking a prompt's
loss does not remove its forward computation.

That implementation used full BF16 model replicas, gradient checkpointing,
one-example microbatches and DDP. The recorded rank-zero peak allocation was
71.65 GiB. Ordinary optimizer-step intervals averaged 8.30 seconds on four GPUs.
Intervals spanning validation/checkpoint boundaries averaged longer and totaled
about 27 minutes, including the subsequent training step; this is not a dedicated
profiler's estimate of checkpoint overhead. Most recorded time was in ordinary
training intervals. We have not measured GPU utilization or kernel-level efficiency.

The new recipe has 1,024 action presentations versus the earlier 33,982. Based on
the earlier throughput, training is expected to take roughly an hour on one H100;
this is an estimate, and preparation plus inference/verification add cost. The
initial training allocation is capped at 45 minutes and stops at update 32.
Only a successful autonomous probe permits the remaining one-epoch continuation,
capped at 90 minutes. No automatic two-epoch or full-corpus fallback is authorized.

## Team access on four L4 GPUs

The competition runtime was observed with four L4 GPUs and 23,034 MiB available
per GPU. Our existing L4 inference check passed; L4 training has not been validated.
The current full-replica BF16 trainer cannot fit each replica on one L4, regardless
of a smaller dataset. Four separate GPU memories require explicit distribution.

Candidate training paths are PEFT with FSDP model sharding or a compatible
low-bit frozen-base implementation. Both need a load/backward/save/reload test
on the exact model and environment. Do not substitute an unrelated NF4 base for
the required QAT checkpoint or claim a Kaggle training recipe is ready based only
on H100 execution. H100 is the immediate protocol-debugging path; a teammate-facing
L4 recipe remains a separate compatibility requirement.

Primary references: [LoRA](https://arxiv.org/abs/2106.09685),
[PEFT FSDP](https://huggingface.co/docs/peft/en/accelerate/fsdp),
[PEFT quantization](https://huggingface.co/docs/peft/en/developer_guides/quantization),
and [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/).
These describe techniques and hardware, not a measured speedup for this run.
