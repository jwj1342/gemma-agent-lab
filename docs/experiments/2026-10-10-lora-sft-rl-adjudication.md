# PEFT, SFT, and RL for the Gemma 4 coding agent: adjudication and next-step options

**Status: analysis and decision record. Nothing here is a measured improvement.** No adapter has been
trained, no RL run has been started, and no new inference was executed for this note. Every claim is
labelled as an independently executed result, an organizer statement, static source analysis, or a
proposal. Prepared 2026-10-10.

## The three options are not alternatives at the same level

This is the first thing to settle, because the question "PEFT, SFT, or RL?" mixes two axes:

| Axis | Meaning here | What it can produce |
| --- | --- | --- |
| PEFT / LoRA | *How* any fine-tune is packaged so 31B weights fit the 3 GiB submission cap | A delivery format. On its own it changes nothing |
| SFT | Supervised fine-tuning of that LoRA on trajectory data | Imitates behaviour that appears in the data |
| RL (or preference learning) | Optimising the LoRA against a reward or preference signal | Behaviour that no demonstration contains but the reward prefers |

So PEFT is a prerequisite, not a competitor: both SFT and RL ship as PEFT adapters. The real fork in
the road is **SFT versus reward-driven optimisation (offline preference learning vs online RL)**, and
before either, whether an adapter is worth its serving cost at all.

## What is already established (with provenance)

| Fact | Provenance |
| --- | --- |
| The required base is `gemma-4-31b-it-qat-w4a16-ct` (compressed-tensors W4A16), fixed for every agent; adapters are optional and per-agent | Official competition page text in [model-selection-budget-and-harness-rules.md](../../resource/competition/model-selection-budget-and-harness-rules.md) |
| The harness derives `--max-loras` from the mounted adapter count and `--max-lora-rank` from the rank in `adapter_config.json`, rounded up to a vLLM bucket; fallback rank is 128 | Static analysis of [server.py](../../resource/harness-source/adk_submission/server.py) |
| The adapter's `lora_B` was all zero in the shipped sample adapters, so they cannot serve as an on/off smoke test; both samples are identical bytes | [sample-adapters.json](../../resource/manifests/sample-adapters.json) plus [reading notes](../../resource/notebooks/reading-notes.md) |
| The "adapters load but are silently wiped" bug (Gemma4 decoder layers registered twice) was fixed in wheelhouse v23+; organizer confirmed | [Discussion 743508](../../resource/discussions/743508.txt) (organizer reply) |
| Enabling LoRA costs KV cache. Measured on 4xL4 with one rank-16 adapter: 36,432 to 27,712 tokens at utilization 0.80 (0.76x). With `cudagraph_specialize_lora=False`, 34,288 tokens (0.94x) | [Discussion 744331](../../resource/discussions/744331.txt) (community measurement, not our own) |
| The organizer says training on Gemma-4-31b's own output is allowed; the answer for other models' outputs was still pending | [Discussion 743964](../../resource/discussions/743964.txt) |
| No local adapter training has ever completed here; no local GPU can hold 31B; quota is the binding constraint | [WORKLOG](../../WORKLOG.md), [model/README](../../model/README.md) |

Two consequences follow immediately.

1. **An adapter must out-earn its own KV-cache cost.** A submission carrying one adapter serves every
request through it and runs with a smaller cache at the same time. In the only complete paired
measurement we have, the agent took 19 tool calls and 191,500 tokens on one task
([WORKLOG](../../WORKLOG.md), `rich_3675`); that is well past 27.7k. Either the adapter must change the
trajectory enough to shorten it, or it pays a penalty on the tasks it does not help.
2. **A zero-effect or untrained adapter is worse than no adapter: it silently burns cache.** This is
why the new configuration in
[configs/experiments/lora-trajectory-v1](../../configs/experiments/lora-trajectory-v1/README.md)
carries an `adapter_config.json` but deliberately no weights.

## Expected improvement, ranked, with the reason

| Rank | Option | Expected effect on resolved-issue rate | Cost | Main risk | Confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | **Build the measurement instrument** (verify controls and freeze a task set large enough to detect an effect) | None by itself, but it is what makes any other option decidable | Low: CPU plus existing quota | Continued quota loss on unmeasurable runs | High |
| 2 | **SFT on complete agent trajectories** (tool-loop data, completion-only loss) | Small to moderate; the largest of the learning options, because the loop policy is what currently fails | Medium-high: needs rollouts or curated traces, then a paired A/B | Public repositories are likely disjoint from the hidden test repositories; format learning transfers, repository knowledge does not | Medium |
| 3 | **Offline preference learning on self-generated trajectories** (DPO-style, no environment rollouts at training time) | Small, and only after SFT has produced a usable policy; needs pass/fail pairs from our own runs | Medium: no sampling loop, but needs labelled pairs | Not enough valid pairs to separate signal from noise | Low-medium |
| 4 | **Online RL / GRPO** | Unknown, plausibly the highest ceiling, and the least measurable at our scale | Highest: 4-8 rollouts per prompt plus sandbox test execution per reward; our evaluation already took 10 kernel pushes to measure one pair | Non-reproducible infrastructure; reward hacking through test or environment edits | Low |
| 5 | **PEFT alone as "the change"** | None | Low | Shipping an inert adapter that costs KV cache | High (that it does nothing) |

Ordering rationale, in one paragraph: SFT beats RL *first* here because the binding failure we have
actually observed is policy and format, not preference. The agent produced a byte-identical patch with
and without a localization skill, and never called `run_skill_script` at all
([WORKLOG](../../WORKLOG.md), October 10). Imitating successful trajectories attacks exactly that.
RL only has an edge once the policy reliably executes the loop, because otherwise the reward signal
mostly encodes "did not finish" and the training cost buys noise. Online RL is last not because the
idea is weak but because every rollout needs a 4-minute, 40-tool-call agent episode plus sandboxed
test execution before it yields one bit of reward - the most expensive bit in the project.

## The measurement problem dominates

Our only complete paired comparison is n=1, and its two arms produced byte-identical patches (p=1.0).
Detecting a 10-point improvement in resolved rate with 80% power needs on the order of 40+ paired
tasks with valid empty/gold controls; with roughly 20 usable public tasks, effects below roughly 20
points are not separable from noise. **This is the strongest argument against starting with RL:** we
currently cannot tell a real gain from a lucky draw, so a training run's outcome would be
uninterpretable either way. Fixing this is cheap, CPU-side work, and it gates everything else.

## What is configured now (and what is still missing)

Configured, validated offline by `scripts/check_lora_adapter.py` (rank 32 -> bucket 32,
`max_loras` 1, 52,101,120 trainable parameters, target modules limited to the two projections with
recorded shape evidence):

- [agent.yaml](../../configs/experiments/lora-trajectory-v1/agent.yaml) and
  [code_analyzer.yaml](../../configs/experiments/lora-trajectory-v1/sub_agents/code_analyzer.yaml)
  declare `adapter: main_lora`.
- [adapter_config.json](../../configs/experiments/lora-trajectory-v1/adapters/main_lora/adapter_config.json)
  is the PEFT LoRA config the serving path reads for its rank.
- [scripts/train_lora_adapter.py](../../scripts/train_lora_adapter.py) is the training/export entry
  point: a CPU-only dry run by default, `--train` for real work, with target modules resolved against
  the loaded model and refusal on any unverified module name.

Missing, and not creatable in this workspace: the trained `adapter_model.safetensors`. There are no
base weights locally, no training environment, and no GPU that can hold 31B. Training must run on
Trillium or a Kaggle session, in the separate training environment the
[training plan](../training-plan.md) requires.

## Proposed next-step options

**Option A - Instrument first, then SFT on trajectories (recommended).** Finish the control-verified
task set, generate successful trajectories from the existing baseline on those tasks, filter to
passing repairs, train the configured adapter with completion-only loss on the tool loop, and run one
frozen paired A/B. Kill criterion: if the adapter arm does not beat the no-adapter arm on the frozen
set with valid controls, the adapter is not shipped and we keep the prompt/skill line.

**Option B - SFT on public reference patches (the cheap shortcut).** Train issue-to-diff or
issue-to-file supervision directly from `tasks.jsonl`. Cheapest real learning data available, and
several public notebooks did it. Expect weak transfer: the public set is four repositories that are
hidden at test time, and issue-to-diff samples carry no tool observations. Use it only as a fast
sanity check that the training and export path works end to end, not as the promoted experiment.

**Option C - Offline preference learning instead of SFT.** Skip SFT and build chosen/rejected pairs
from our own runs (paired base versus candidate, or passing versus failing patch on the same task).
Cheaper than online RL, but needs enough valid pairs to be statistically meaningful, which we do not
have yet. Reasonable as a follow-up to Option A.

**Option D - Online RL / GRPO.** Only after A or C produces a policy that reliably completes the
loop and after the instrument can measure it. Reward must separate passing repairs, environment
errors, empty patches, format errors, and timeouts, and must never reward editing tests or the test
environment ([training-plan](../training-plan.md) already states this rule).

**Option E - Do not ship an adapter at all.** Spend the compute on prompt/skill iteration and keep
the full 36k-token KV cache. Defensible right now: our only measured A/B of a prompt-side change was
inconclusive, which is an argument for better measurement, not necessarily for a different method.

## Recommendation

Take **Option A**, with Option B used strictly as an end-to-end plumbing check, and hold C and D until
the instrument and the policy are ready. Concretely: next research step = control-verified task set
plus trajectory collection on the existing baseline; the adapter configuration and training entry
point are already in place for that experiment.

## Open questions that need an organizer or a measurement, not an assumption

- Does the scoring environment pass `cudagraph_specialize_lora=False` when LoRA is enabled? If not,
  an adapter submission runs at roughly 0.76x KV cache rather than 0.94x.
- Are other models' outputs permitted for distillation? Pending as of discussion 743964.
- What is the real overlap between the public four repositories and the hidden test repositories?
  Our transfer estimate depends on it, and it is currently unknown.

## Resolved since this note was written (2026-10-10, later the same day)

| Earlier open item | Resolution | Evidence |
| --- | --- | --- |
| "Reachability, license, and size of the unquantized QAT checkpoint are unverified" | **Resolved.** `gemma-4-31b-it-qat-q4_0-unquantized` exists for 31B: Transformers/PEFT format, version 2, 62,578,682,779 bytes (~62.6 GB), Apache-2.0, Kaggle-hosted | `resource/models/google-gemma-4.json` |
| "A bnb-4bit QLoRA pipeline does not reproduce the served weight path" | **Still a risk, now precise.** Training on the *quantized* `w4a16-ct` or on an NF4 quantized base is not the served path; the unquantized QAT checkpoint is the closest training base in the same weight family. Whether a PEFT adapter trained on it loads cleanly under the scorer's compressed-tensors W4A16 serving remains **unverified** | Static analysis plus the model card |
| Local training feasibility | **Closed as infeasible.** The base cannot be shrunk (one permitted serving model, same QAT family for training). This laptop has 8,151 MiB VRAM and about 16 GiB RAM; a 4-bit 31B base needs ~15.6 GB just for weights, before activations, optimizer state, or checkpoints | `nvidia-smi`, memory counters |
| GPU quota available | **Measured: 102,698.778 s remaining of 108,000 s** (used 5,301.222 s), TPU 72,000 s untouched, refresh 2026-10-17 | `resource/competition/accelerator-quota-2026-10-10.json` |

The consequence for the ranking above is narrow but real: the first learning experiment now has a
concrete, licensed training base, so the question is no longer whether SFT is *possible* but whether
the adapter transfers through the W4A16 serving path at all. That transfer check deserves to be run
before any large training budget, because a negative answer invalidates the whole LoRA line.

