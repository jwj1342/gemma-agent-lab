# First complete SWE LoRA training run

This run targets the official version 29 environment (`swegemma 0.2.10`,
`adk-submission 0.2.13`). Its purpose is to complete a reproducible training and
submission cycle. An improvement in repair rate is a hypothesis until measured.

## Evidence behind the initial configuration

| Primary source | Relevant evidence | Decision for this run |
| --- | --- | --- |
| [SWE-smith, Appendix F](https://arxiv.org/html/2504.21798v1) | Successful repair trajectories are converted to the student's tool format; the main experiment uses full fine-tuning, LR 5e-5, up to three epochs and 32K context. | Retain upstream success provenance and explicitly convert tool arguments. Its benchmark score does not predict Gemma performance. |
| [SWE-Gym, Appendix B](https://arxiv.org/html/2412.21139v1) | The OpenHands study uses 491 successful trajectories. The separate 32B MoatlessTools setup uses LoRA rank 64, LR 5e-4, five epochs and 10,240-token context. | Treat rank 64 as a tested setting for that model/scaffold, rather than a universal requirement. |
| [LoRA Learns Less and Forgets Less, Section 4.6](https://arxiv.org/html/2405.09673v1) | Learning rate and target-module choice have substantial effects; attention plus MLP with rank 16 is a useful tradeoff in its experiments. | Start with rank 16 on text attention and MLP, alpha 32, LR 5e-5. These are engineering starting points, not a proven optimum. |
| [PEFT LoRA reference](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora) | Standard LoRA uses alpha/r scaling; rsLoRA changes scaling and DoRA introduces an additional adaptation mechanism. | Export standard PEFT LoRA with unchanged frozen base weights and no extra trained modules. |
| [Competition model and harness rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules) | Every agent uses the required W4A16 model; PEFT adapter directories with JSON configuration and safetensors weights are supported. | Train from the dequantized official checkpoint and verify the resulting adapter on the original quantized serving model. |

The authenticated competition pages were refreshed on October 9, 2026. They specify
the adapter artifact format, but do not separately approve every named LoRA variant.
The version 29 server infers adapter rank and rounds it to a supported vLLM bucket;
its fallback value of 128 is not an explicit competition rank limit. Standard rank
16 avoids additional conversion assumptions. QLoRA's usual NF4 base is not a substitute
for the competition's required QAT W4A16 checkpoint.

Direct inspection of the installed official vLLM 0.19.1 `lora/peft_helper.py`
confirms alpha/sqrt(rank) handling for `use_rslora`, while `use_dora=True` and
nonempty `modules_to_save` are rejected. This is runtime compatibility evidence,
separate from the competition's artifact-format permission. Standard LoRA remains
the first configuration; rsLoRA is a possible later ablation with its own loading
check. DoRA is unsuitable for this pinned runtime.

The organizer also clarifies that the evaluation system accepts LoRA adapters and
is not configured to execute other submitted weight-update mechanisms in the
[LoRA-only discussion](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743383).
The [adapter-effect discussion](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743508)
records a previously fixed adapter-zeroing bug. Therefore, seeing an adapter name in
the server model list is insufficient: verify a measurable effect as well as loading.
Both discussion threads were refreshed on October 9, 2026.

## Corpus construction

1. Read all eight pinned SWE-smith tool shards and keep up to 768 distinct successful issues.
   Reject unsupported editor actions and entire trajectories containing explicit
   dependency or network installation commands. Preserve actual upstream observations.
2. Exclude all frozen development and holdout snapshots. Partition remaining examples
   by repository and base commit, so mutations of one snapshot cannot cross splits.
3. Regenerate graph-neighbor, subgraph, retrieval and fallback demonstrations using
   the version 29 official tools on the permitted training snapshots. Select up to
   64 symbols per snapshot with a snapshot-dependent hash; selecting the same four
   symbols across nearby revisions produced mostly duplicates in the initial audit.
4. Execute controlled Git/test/submission fixtures and add brief final-answer targets.
   These teach termination; they are component exercises, not evidence of real repair.
5. Encode complete assistant targets with the official tokenizer and mask prompt and
   tool-output tokens. Keep up to six actions of history, shorten long observations
   explicitly, and remove complete older turns when needed. Reject targets that do
   not fit. Deduplicate encoded examples and check split separation again. If two
   snapshots yield identical encoded graph exercises across splits, remove their
   training copy and keep the validation copy.

The 8K window is an initial compute/context tradeoff. The longest encoded example
must pass a backward-memory check. It does not establish performance across the
full competition context window. The corpus manifest records retained/rejected
counts, input lengths, supervised tokens, source hashes and group counts.

## Training and checkpoint selection

The [configuration](../configs/training/swe-lora-v1.json) specifies two complete
shuffled epochs. Each accepted example appears once per epoch. Four DDP workers
with two accumulated microbatches produce an effective batch of eight actions;
zero-weight padding avoids duplicated supervision in the last batch. Loss is
normalized by supervised tokens across that batch. Graph and termination samples
retain their natural corpus frequency rather than replacing half the SWE examples.

Training uses 5% warmup, cosine decay to 10% of peak LR, BF16, gradient checkpointing,
and gradient clipping. BF16 autocast covers adapter projections while their master
weights remain FP32. The longest-example memory probe also reserves the footprint of
Adam's two moment tensors. A fixed validation panel of up to 128 actions per source is
evaluated before training, every 100 updates and at each epoch boundary. Selection
uses SWE validation NLL; graph and terminal losses are reported separately. Falling
training loss alone does not establish convergence or repair improvement.

The runner saves optimizer state, per-rank RNG state, deterministic data position,
hashes, and adapter weights, then deliberately restarts after eight updates to verify
recovery. It retains the latest resumable checkpoint and the best validation checkpoint.
Final exports contain only PEFT configuration and safetensors adapter weights.

## Evaluation and release gates

Check adapter loading and effect with the official quantized server. Freeze the
candidate, public task selection and archive hash. Require valid empty/gold controls
before repair evaluation, and report environment execution separately from resolved
issues. Keep the holdout protected from prompt/checkpoint tuning. A manual competition
submission follows successful artifact validation; record its receipt and eventual
score without interpreting a small local panel as a reliable leaderboard estimate.

Initial preparation artifacts live under `runs/20261009-swe-lora-v1/`; the expanded
corpus run uses `runs/20261009-swe-lora-v1-full/`. Actual corpus sizes,
training completion, measured losses and submission status belong in the final
experiment report; this document describes the initial protocol.
