# Serving-path risk: does LoRA actually apply to the W4A16 base?

**Status: external evidence gathered, local determination still open.** Prepared 2026-10-11. Nothing
here was measured on the competition stack; every claim is labelled with its source kind.

## Why this is checked before training

The competition serves exactly one model, `gemma-4-31b-it-qat-w4a16-ct`, a compressed-tensors W4A16
checkpoint. LoRA on a quantized base has a documented history of failing **quietly**: the adapter
loads, the server reports success, and the output is unchanged. Two independent reports already sit in
this repository's own collected material:

- The organizers' own vLLM build wiped adapters because Gemma4 registered each decoder layer twice,
  so `activate_adapter` called `reset_lora` on the alias and cleared the weights it had just set
  ([discussion 743508](../../resource/discussions/743508.txt)). Reported fixed in wheelhouse v23+.
- Independent reproduction on the then-current wheelhouse measured base and adapter logprobs as
  identical, through both `adk_submission.VllmServer` and direct vLLM ([same thread](../../resource/discussions/743508.txt)).

A third-party report on the same pattern exists outside the competition, which is why this is treated
as a property of the stack rather than a one-off competition bug.

## External evidence (untrusted sources, read as data)

| Observation | Source kind |
| --- | --- |
| vLLM's Gemma4 model file declares `SupportsLoRA` on the causal-LM class, and declares `packed_modules_mapping` with `qkv_proj: [q_proj, k_proj, v_proj]` and `gate_up_proj: [gate_proj, up_proj]` | Static source of upstream vLLM (`vllm/model_executor/models/gemma4.py`, `main`) |
| LoRA support for Gemma 4 was added by a recent upstream PR, and a second PR covers `Gemma4ForConditionalGeneration` | Upstream PR titles [#39560](https://github.com/vllm-project/vllm/pull/39560), [#39291](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/39291/overview) |
| A bug report states LoRA on Gemma 4 **MoE** fails at startup because the models do not implement `get_expert_mapping`, with a follow-up PR implementing it | Upstream issue [#53470](https://github.com/vllm-project/vllm/issues/53470), PR [#46772](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/46772/overview) |
| LoRA on a compressed-tensors **int4 (W4A16)** model: rank-32 all-layer adapters produced weak, non-reproducible outputs while rank-8 partial-coverage adapters worked, across versions 0.17.1-0.24.0 | Upstream issue [#50059](https://github.com/vllm-project/vllm/issues/50059) |
| A bugfix adds `select_gemm_impl` to `CompressedTensorsWNA16MoEMethod` so LoRA works on that quantized path | Upstream PR [#31453](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/31453/overview) |

The pattern across these: support exists but has been landing piecewise, and the failure mode is
silent rather than a startup error.

## What it implies for this project concretely

1. **The scoring container is a patched 0.19.1, and the patch is not on PyPI.** The exact LoRA state of
   the competition's vLLM build cannot be read from a published wheel; it ships inside the Docker image
   (`vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl`, sha256 `6b29fdc2…`, per
   [eval-wheelhouse-2026-10-09.json](../../resource/manifests/eval-wheelhouse-2026-10-09.json)). The
   `metric/gemma-4-developer-agent-wheelhouse` Kaggle dataset carries only 20 test-dependency files and
   no vLLM wheel, so this was confirmed by listing the dataset rather than assumed.
2. **Target-module naming is not obviously aligned.** PEFT training names layers as they exist in the
   Hugging Face model (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`),
   while vLLM packs them. Whether an adapter trained on the individual names is picked up correctly by
   the packed mapping is exactly what a serving check answers and what no amount of reading settles.
3. **Our current configuration is in the risky corner of the evidence.** `q_proj` and `o_proj`
   training is unremarkable, but issue #50059's finding that higher-rank, wider-coverage adapters
   degrade is a direct warning about a future `rank=32`, all-layer plan.
4. **Training base choice also interacts.** Training on the unquantized QAT checkpoint and serving on
   W4A16 crosses precision; training directly on the W4A16 base matches serving but is a harder
   optimization target. Either way the serving check is the gate.

## The cheapest resolution, which needs no training

A loud probe adapter is the decisive test: random `lora_A` **and** random `lora_B` at a scale that must
move the logits. If greedy logprobs are identical with and without it, LoRA is inert on that stack and
no trained adapter can help until it is fixed; if they change, the path works and training is worth
funding.

`scripts/make_lora_probe.py` builds one (rank 8, `q_proj` and `o_proj`, verified shapes taken from
`resource/manifests/sample-adapters.json`) and can validate that both sides carry signal, which is the
property the shipped example adapters lack: their `lora_B` is all zero, so they are incapable of
changing any output and are useless as an on/off control.

This test costs one short serving session on the competition model, with no training and no adapter
quality claims. It should run **before** any SFT budget is spent, and its result should be recorded
either way, because a negative result retires the whole LoRA line and saves the remaining work.

## Open questions that only a run can answer

- Does the competition's patched 0.19.1 apply LoRA to a compressed-tensors W4A16 Gemma 4, and does it
  survive the decoder-alias registration that caused the earlier wipe?
- Which target-module naming does it accept: the Hugging Face individual projections, the packed
  names, or both?
- Does the rank/coverage sensitivity in issue #50059 reproduce here, which would cap a useful rank?
