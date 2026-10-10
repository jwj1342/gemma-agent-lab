# Bounded SWE LoRA training and evaluation

The current corrective recipe uses 512 actions and one epoch, with termination
and complete-agent checks after 32 updates. The initial 1,024-action recipe was
stopped at update 32 when a real task exposed repeated actions after a successful
submission. The user requested a small training/evaluation loop before further
scaling. The earlier four-GPU replacement jobs (`1071755`,
`1071772`, `1071773`, `1071774`) were cancelled before starting; their elapsed
compute time is zero. Existing artifacts and provenance are retained.

## Initial 1,024-action protocol

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

## Preparation audit

The first allocation (`1072210`) stopped during the native-token audit, before
loading the model or starting training. Graph fallback fixtures lacked
`target_kind` metadata. The encoder now derives it from the actual assistant
target for every source and rejects conflicting metadata. Regression coverage
includes fixtures without that field. Its three dependent jobs were cancelled
without running. The preparation attempt used 7m39s on one H100.

The corrected retry uses a separate run directory, preserving the failed
preparation artifacts. An existing encoded-data directory without a completed
manifest is rejected rather than treated as a usable corpus. Data budgets,
training length, validation tasks, and release requirements remain unchanged.

## Corrected run: observed preparation and recovery

The corrected run is `runs/20261010-swe-lora-light-r2/`, frozen at training source
`55fb9b9b9e73c36021e46e3aeaadfb2c216e495c`. Repository checks and 72 tests passed
in its allocation and in
[CI run 38087195294](https://github.com/jwj1342/gemma-agent-lab/actions/runs/38087195294).

| Selected source | Actions | Snapshot groups | Input tokens | Target tokens |
| --- | ---: | ---: | ---: | ---: |
| SWE repair | 896 | 97 | 2,627,353 | 152,382 |
| Graph exercises | 112 | 10 | 91,028 | 4,355 |
| Terminal exercises | 16 | 16 | 12,567 | 261 |
| Total | 1,024 | — | 2,730,948 | 156,998 |

SWE actions cover 412 upstream trajectories and all 484 available snapshot/tool
strata; graph actions cover all 40 available strata. Every one of the 1,008 tool
targets ends at the supervised handoff token 50. The terminal targets retain the
native `<turn|>` and following newline. All 4,290 encoded validation actions are
preserved; the checkpoint-selection panel contains 32 SWE, 32 graph, and 19
terminal actions. These counts describe actions with context, not 1,024 complete
repair trajectories.

Both backward memory probes passed: the longest input (4,096 tokens) and longest
target (1,959 tokens). The highest Torch allocation recorded in the probes was
70.42 GiB. One allocator retry was logged; the backward operation completed.
Checkpoint eight was saved, loaded in a fresh process, and followed by a successful
update nine. Configuration/data/dependency contracts matched, and optimizer/RNG
state was restored. `recovery-gate.json` retains this evidence; exact equivalence
to an uninterrupted numerical run was not tested.

The initial allocation (`1072257`) completed successfully in 17m08s on one H100,
including full-corpus encoding, the memory probes, save/restart and 32 updates.
Those updates presented 128 selected actions and 20,008 supervised tokens.

| Fixed validation family | Actions | Initial NLL | Step 32 NLL |
| --- | ---: | ---: | ---: |
| SWE | 32 | 0.506931 | 0.243458 |
| Graph | 32 | 0.858223 | 0.099037 |
| Termination | 19 | 3.520047 | 1.596865 |

These teacher-forced losses establish progress on the training objective.
Autonomous interaction and repair outcomes are separate checks. The partial
adapter and its hashes are retained under `preflight-validation/`; this checkpoint
does not count as completing the selected-data epoch.

## Termination diagnosis and 512-action retry

Partial validation job `1072258` loaded the adapter in the official W4A16 runtime
and measured a weight effect against a repeated zero-update control. All eight
short tool-context requests produced schema-valid calls. However, all four
held-out submission contexts generated another tool call instead of a final
answer; the base ended correctly in all four. These short probes disable thinking
for both models; the autonomous agent retains the baseline thinking configuration.

On `rich_3454`, the autonomous adapter inspected and edited source, ran a small
test, and captured a 656-character source patch after about 24 seconds. It then
repeatedly ran invented test names after submission, exhausted 40 budgeted tool
calls, and timed out at four minutes. The patch has not been independently graded
at this stage. The run is not a clean completion or a repair success. The native
tool handoff worked across separate calls; post-submission termination remained
broken. The 14m48s validation allocation failed, and both dependent jobs were
cancelled without running. No adapter was uploaded.

Only 261 of the selected corpus's 156,998 supervised tokens were terminal answers
(0.17%). The first 32 updates saw three terminal actions. This motivates a bounded
corrective hypothesis, rather than proving a unique cause:

- Select 384 SWE, 64 graph and 64 terminal actions: 512 total, one epoch and 128
  optimizer updates. Keep the same raw sources, snapshot exclusions and seed.
- Average token loss within each action, then weight real actions equally within
  a global batch. Thus short completion answers receive the same example weight
  as long tool calls. Keep token-weighted validation NLL and also log token-weighted
  training loss; the two training objectives should not be compared as one curve.
- Reduce LR from 5e-5 to 2e-5. Keep rank 16, alpha 32 and the 4K context limit.
- Start a new adapter from the required frozen base. Preserve the stopped run;
  do not reuse its optimizer or alter its source contract.
- Save/restart at eight, stop at 32, and require all four terminal probes to end
  before running the same three real-task probes. Keep the empty/gold controls,
  nonempty-clean-artifact requirement, fast cohort and protected holdout unchanged.

Configuration: [swe-lora-light-balanced.json](../../configs/training/swe-lora-light-balanced.json).
The combined corrective experiment changes sampling, loss weighting and learning
rate, so its outcome cannot isolate their individual effects. The stopped task's
trace is used for diagnosis only; its patch or answer is not added to training.

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
