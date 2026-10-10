# LoRA trajectory experiment v1

This directory is the adapter experiment that competes with the untouched
[baseline](../../baseline/agent.yaml). It exists so that the official adapter path -- submission
`adapters/<name>/` plus `adapter: <name>` on an `LlmAgent` -- is configured, rank-checked, and
reproducible before any GPU time is spent.

**Status: configuration and validation only; no trained weights exist yet.** The repository has no
base weights, no GPU, and no training environment (see [training plan](../../../docs/training-plan.md)),
so this directory deliberately contains no `adapter_model.safetensors`. An untrained or zero-filled
adapter is not a neutral placeholder: mounting one sends every request through the adapter, which
cuts the KV cache and can make long tasks hang. Never copy the sample adapters from
`sample_submission/` or synthesize an A/B pair to fill this slot.

## What is configured

| Item | Value | Why |
| --- | --- | --- |
| Adapter name | `main_lora` | Discovery name is the `adapters/` subdirectory name |
| Rank `r` | 32 | Same vLLM bucket as 16, so a 32-rank adapter costs no more KV cache than 16 |
| `lora_alpha` | 64 | Keep the `alpha / r` scaling that the public 31B study used |
| `target_modules` | `q_proj`, `o_proj` | The only projections with recorded shape evidence in this repository |
| Trainable parameters | 52,101,120 | 0.168% of the 31B base; ~99.4 MiB bf16 |

Rank is the only knob that changes serving memory. The harness reads `r` from this file
(`adk_submission/discovery.py:infer_adapter_rank`) and rounds it up to a vLLM bucket
(`1, 8, 16, 32, 64, 128, 256, 320, 512`). A rank of 128 -- the harness fallback when the rank cannot
be read -- would reserve four times the LoRA slots for no measured benefit.

### Why only two target modules

`resource/manifests/sample-adapters.json` records tensor names and shapes for exactly two projections
of `gemma-4-31b-it-qat-w4a16-ct`: `self_attn.q_proj` (5376 -> 8192) and `self_attn.o_proj`
(8192 -> 5376). No other projection name or shape is verified anywhere in this repository, so
`target_modules` claims nothing beyond that evidence.

To close that gap, run `scripts/probe_model_modules.py` once in a session that has the base weights;
it writes `resource/manifests/model-modules.json`, which `scripts/check_lora_adapter.py` then prefers
over the sample adapter as its evidence source. `scripts/train_lora_adapter.py` independently resolves
every requested target against the loaded model before training and refuses to start if a target is
absent. Widen this list only after both agree.

## Validate before spending GPU time

```powershell
python scripts/check_lora_adapter.py --experiment-dir configs/experiments/lora-trajectory-v1
```

The check is CPU-only and offline. It re-derives the target shapes from
`resource/manifests/sample-adapters.json`, recomputes the parameter count, re-reads the rank-bucket
policy from the extracted harness source, and fails if the adapter config, the agent wiring, or the
harness policy disagree.

Two further offline checks need no base weights and no GPU, and each one targets a failure that
would otherwise be discovered only after a GPU session was already spent:

```powershell
# PEFT wiring and export layout on a tiny local model (needs torch and peft only)
python scripts/train_lora_adapter.py --self-check

# The actual SFT records, built from the public tasks already in this repository
python scripts/build_sft_dataset.py --output data/processed/sft-public-v1.jsonl
```

`--self-check` verifies that the configured `target_modules` resolve, that a LoRA layer is created and
receives gradients, that the export lands at `adapters/<name>/adapter_model.safetensors` — the exact
filename `adk_submission.discovery` groups a PEFT directory around — and that the exported adapter
reloads with matching weights. It proves wiring, never accuracy.

## Producing the weights

`scripts/train_lora_adapter.py` is the training and export entry point. Run it where the base model,
a GPU, and the training dependencies are available:

```bash
python scripts/train_lora_adapter.py --train \
  --agent-dir configs/experiments/lora-trajectory-v1 \
  --base-model "$GEMMA4_QAT_BASE" --dataset data/processed/trajectories.jsonl \
  --output-dir model/adapters/main_lora --max-steps 20
```

Then copy the exported `adapter_config.json` and `adapter_model.safetensors` into
`adapters/main_lora/`, re-run the check above, and keep the A/B control rule from the
[training-plan acceptance section](../../../docs/training-plan.md).

## Files

| Path | Purpose |
| --- | --- |
| [agent.yaml](agent.yaml) | Coder agent with `adapter: main_lora` |
| [sub_agents/code_analyzer.yaml](sub_agents/code_analyzer.yaml) | Analyzer agent, also on `main_lora` |
| [adapters/main_lora/adapter_config.json](adapters/main_lora/adapter_config.json) | PEFT LoRA config; rank source for the serving path |
| [prompts/](prompts/) | Prompts copied from the baseline plus the adapter note |
| [docs/training-plan.md](../../../docs/training-plan.md) | Environment, data, and acceptance requirements |
