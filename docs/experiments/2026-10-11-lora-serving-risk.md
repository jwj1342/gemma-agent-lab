# Serving-path risk: does LoRA actually apply to the W4A16 base?

**Status: measured on the competition stack on 2026-10-11 -- LoRA is applied (see the result section).
The rank-32 plan is still unvalidated.** Nothing here is inferred; the positive result below carries
its own reproducibility control, and every negative or inconclusive statement is labelled as such.

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

## Result: LoRA is applied on the competition stack (measured 2026-10-11, kernel version 7)

Run: `junzhejiao/gemma4-lora-serving-probe` version 7, private, L4x4, competition model
`gemma-4-31b-it-qat-w4a16-ct` v2, wheelhouse vLLM 0.19.1 installed from the mount, `torch 2.10.0+cu128`,
`quantization=compressed-tensors`, `tensor_parallel_size=4`, `max_model_len=4096`,
`gpu_memory_utilization=0.90`, `enable_lora=True`, `max_loras=1`, `max_lora_rank=128`,
`enforce_eager=True`, `temperature=0`, `logprobs=3`.

| Measurement | Value |
| --- | --- |
| Base generated twice, logprob deltas | **0.0 maximum absolute delta** across 35 positions -- the engine is reproducible |
| Base versus adapter, logprob deltas | **1.187468 maximum absolute delta**, changed at **35 of 35** positions |
| Greedy text and token ids | identical in all three generations (`2, 3, 5, 7, 11, 13, 17, 19, 23, 29`, 35 tokens) |
| Verdict emitted by the probe | `applied: the engine reproduces its own logprobs exactly, and mounting the adapter moves them` |
| Adapter | rank 8, `q_proj` + `o_proj` at layer 0, random `lora_A` (std 0.02) and `lora_B` (std 1.0), SHA256 `b43e139956c95f2a…` |

What this settles: the aliasing wipe of [discussion 743508](../../resource/discussions/743508.txt) does
**not** reproduce on this wheelhouse, the adapter is honoured on a compressed-tensors W4A16 Gemma 4, and
a loud adapter moves the served distribution even when it does not change greedy text -- which is why an
unchanged *text* is not evidence of an inert adapter.

What it does not settle, stated so it cannot be over-read:

- **The planned adapter configuration is wider than the probe.** The probe is rank 8 on two projections
  of one layer; our experiment is rank 32 on `q_proj` and `o_proj` of every layer. Upstream
  [#50059](https://github.com/vllm-project/vllm/issues/50059) is precisely about a rank-32, wide-coverage
  adapter degrading on a compressed-tensors int4 base, so the rank-32 case still needs its own probe.
- **No trained adapter has been served.** This shows the mechanism works with random weights; it says
  nothing about whether trained weights help resolution, which is the paired-experiment question.
- **The probe used the model path directly, not `adk_submission.VllmServer`.** The harness's own launch
  path differs in flags (`max_model_len` 32768) and in how it enumerates adapters, so a submission can
  still fail where this notebook succeeded.
- The engine's own `Successfully loaded LoRA weights` / `No LoRA weights found` lines were **not**
  captured: vLLM loads adapters inside a spawned `EngineCore` process, so the in-notebook logging
  handler saw none of them (`lora_loaded_modules: 0`). Those log lines cannot be used as evidence from
  a notebook; the logprob comparison is what carries the result.

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

## The runnable probe (built 2026-10-11, not yet executed)

`scripts/prepare_lora_probe_notebook.py` turns that plan into one private L4x4 notebook. It embeds the
probe adapter as gzip+base85 (434,760 bytes, so no dataset publication is needed), proves its identity
by SHA256 before vLLM sees it, then loads the competition base directly with vLLM and compares greedy
logprobs between the base model name and the adapter name on one fixed prompt:

```python
LLM(model=MODEL, tensor_parallel_size=4, max_model_len=4096, gpu_memory_utilization=0.90,
    enable_lora=True, max_loras=1, max_lora_rank=128)
SamplingParams(temperature=0, max_tokens=40, logprobs=3)
llm.chat([...])                                    # base
llm.chat([...], lora_request=LoRARequest('main_lora', 1, ADAPTER))   # adapter
```

Those constructor arguments mirror the harness defaults in `resource/harness-source/adk_submission/server.py`
(`max_model_len` 32768 there, shortened here to buy KV cache) rather than inventing a different server.

```powershell
python scripts/prepare_lora_probe_notebook.py --adapter-dir artifacts/lora-probe `
    --owner <KAGGLE_USER> --slug gemma4-lora-serving-probe --output artifacts/lora-probe-notebook
# then push artifacts/lora-probe-notebook as a private kernel and read the cell log
```

Verdict semantics, all emitted in `probe-result.json` together with the first eight top-logprob
positions of both arms:

| Observation | Reading |
| --- | --- |
| Any compared position differs, or the token ids/text differ | LoRA is applied; training is worth funding |
| Logprobs identical, log shows loads **followed by** `No LoRA weights found for module ...decoder_layers.N` | The exact YOLO-alias wipe of [discussion 743508](../../resource/discussions/743508.txt) is still present in this wheelhouse |
| Logprobs identical, loads only | LoRA is dropped downstream of loading (the [#50059](https://github.com/vllm-project/vllm/issues/50059) pattern) |
| Logprobs identical, no load messages | The adapter never reached the engine |

What was verified offline, and what was not: the staging cell was executed locally and reproduced both
adapter files byte-identically (`adapter_model.safetensors` SHA256 `b43e139956c95f2a…`), the probe cell
fails with an explicit "base model mount is missing" message when the model input is absent, `--check`
reports `status: ok` and returns 1 on a one-byte tamper, and an incomplete adapter directory is
refused with no output written. **No GPU session has run**, so no verdict exists yet; the notebook also
cannot be exercised locally beyond that wall because vLLM is not installed here.

## Execution log on Kaggle (two failed versions, both environment, no verdict yet)

Running it produced two findings that no amount of local work could have surfaced. Both are recorded
because they apply to every future GPU notebook on this competition image, not just this probe.

| Version | Outcome | Cause, measured |
| --- | --- | --- |
| 1 | ERROR after ~30 s | `ModuleNotFoundError: No module named 'vllm'`. The starter and official GPU images ship vLLM **nowhere**; the patched 0.19.1 exists only inside the wheelhouse mount, so any notebook that needs vLLM must install that mount offline first |
| 2 | ERROR after ~10 s | `Wheelhouse mount is incomplete; missing 1: ['swegemma-0.2.10-py3-none-any.whl']` -- the organizers' in-place republish of 2026-10-09 removed exactly that wheel |
| 3 | ERROR, session lingered 3.5 h | Two of my own defects in the progress helper (`TypeError: unhashable type: 'dict'` from doubled braces), which killed the notebook 149 s in; the worker then kept holding the GPUs until it was reclaimed |
| 6 | COMPLETE in 8 min | First full measurement: identical greedy text, different top logprobs at all 35 positions -- but without a repeat-base control that is not yet evidence |
| 7 | COMPLETE in 8 min | Same, plus the repeat-base control: base noise 0.0, base-versus-adapter delta 1.187468. This is the measured result above |

The useful timing numbers, all from notebook output rather than estimate: staging the embedded adapter
8 s, wheelhouse inventory 16 s, installing 41 wheels / 880,842,253 bytes 53 s, `import vllm` 80 s,
engine initialisation 57 s after start, and each 35-token greedy generation 4-14 s. A full probe
therefore costs about eight minutes of L4x4 wall time once the notebook is correct.

Version 3 therefore installs the wheelhouse the way the reviewed runtime already does (drop the
`cutlass*.pth` shims, normalise the directory names from wheel METADATA, `pip install --no-index
--no-deps`, then prove `import vllm` in a fresh interpreter) and treats only vLLM and its runtime
dependencies as required, recording `swegemma`'s absence as drift. The wheelhouse is still the same
dataset version-1 mount, so the two failures are properties of the image and the dataset, not of the
probe logic: version 1 and 2 both proved the embedded adapter stages correctly on Kaggle, with hashes
matching the local probe byte for byte and the base model mount present at
`/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`.

Interpreter note for the SFT route, from the same logs: this GPU image reports
`/usr/local/lib/python3.12/dist-packages`, while the CPU control-sweep image reports `python3.13`. The
offline training wheel set is cp312, so it is correct for a GPU notebook on this image.

## Open questions that only a run can answer

- Does the competition's patched 0.19.1 apply LoRA to a compressed-tensors W4A16 Gemma 4, and does it
  survive the decoder-alias registration that caused the earlier wipe?
- Which target-module naming does it accept: the Hugging Face individual projections, the packed
  names, or both?
- Does the rank/coverage sensitivity in issue #50059 reproduce here, which would cap a useful rank?
