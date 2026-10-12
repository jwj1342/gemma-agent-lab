# Four-L4 LoRA training feasibility: plan, compatibility analysis and instrument

Task ownership: this is the model/adapter-compatibility and small-LoRA-reproducibility workstream
(issue #13), the four-L4 training-feasibility part of the review handoff in
[PR #8](https://github.com/jwj1342/gemma-agent-lab/pull/8).

**Status: four sessions have run, the fourth stopped at the base load on a measured version conflict, and
the fix for it is now in place.** Runs 1-4 established that the L4x4 allocation is gated on competition
attachment, verified the native handoff contract on four L4 GPUs, and then ended at
`ValueError: ... model type 'gemma4' but Transformers does not recognize this architecture` because the
image's `transformers 5.0.0` predates Gemma 4. Run 5 prepares the mounted model stack (transformers
5.13.1 plus torchao 0.16.0 and the rest, section 5) before it loads anything, so the measurement this
workstream exists for -- which strategy builds, the per-GPU peak and the seconds per update -- is the
next thing that happens. Preparation directories are `artifacts/l4-feasibility-notebook-v*/`; each run's
published outputs are fetched with `api.kernels_output` into `artifacts/l4-feasibility-v*-output/`.

## 1. The requirement

From issue #13, the acceptance criterion is:

> A teammate can reproduce the tiny train/save/reload loop without H100 access. A successful
> inference notebook alone does not meet this training acceptance criterion.

with these bounds: check quota first; cap the first feasibility session at one hour; stop on
incompatibility instead of launching a full run; keep the required base and the final artifact checks
intact; do not start a full-corpus or two-epoch run. The deliverable is a reviewed training entry
point **or** a specific measured compatibility issue.

## 2. Quota position and the arithmetic of one session

Measured this round with `scripts/check_kaggle_quota.py --output artifacts/quota-2026-10-12.json`:

| Field | Value |
| --- | ---: |
| Checked at | 2026-10-11T15:09:52Z (2026-10-11 23:09 Asia/Shanghai) |
| GPU seconds used | 81,601.38093 |
| GPU seconds remaining | 26,398.61907 |
| Refresh | 2026-10-17 |
| Earlier reading (18:36 Asia/Shanghai) | 59,248.937 remaining |

The archived competition accelerator page states that L4x4 notebooks consume GPU quota **at twice the
wall-clock rate** (`resource/competition/upgraded-accelerators.md`). So the remaining budget is about
**13,199 seconds of L4x4 wall time** (3.67 hours) before any further reservations.

That is the arithmetic for the session caps below. The ladder was rebalanced when Path D became a
notebook phase: four phase timeouts must now fit inside the session cap, so each one is smaller than it
was when there were three.

| Cap | Wall seconds | Quota seconds at 2x |
| --- | ---: | ---: |
| Packed probe hard timeout, including its checkpoint digests (`--packed-timeout`) | 600 | 1,200 |
| Four-GPU training subprocess hard timeout (`--train-timeout`) | 1,000 | 2,000 |
| Single-process sharded route hard timeout, including its load (`--device-map-timeout`) | 1,100 | 2,200 |
| Reload subprocess timeout (`--serving-timeout`) | 600 | 1,200 |
| Sum of the phase timeouts, which is what the session cap must cover | 3,300 | 6,600 |
| Session cap before the reload phase is skipped (`--session-limit`) | 3,300 | 6,600 |

The worker-level wall budgets are smaller than their subprocess timeouts (`--packed-budget` 420,
`--train-budget` 600, `--device-map-budget` 700) so a phase stops on its own budget with time left to
write its result, and `--load-budget` 480 stays below `--train-timeout` so a stalled base load is recorded
rather than killed.

One complete session therefore costs at most about 41 % of the 15,990.809 quota-seconds that remained
after run 4, and a session that stops early costs proportionally less. Measured sessions for comparison:
run 4 cost about 795 quota-seconds for roughly 320 s of wall time, and the six-minute run 6 cost 750
quota-seconds, both consistent with the 2x rate.

**The consumption is not a job, and it was still running when this was measured.** Three readings pin
the whole window, and they are linear at exactly the 2x rate:

| Reading (Asia/Shanghai) | used (quota s) | remaining | interval | rate |
| --- | ---: | ---: | --- | ---: |
| 18:36:07 | 48,751.063 | 59,248.937 | — | — |
| 20:00:10 | 58,837.494 | 49,162.506 | 10,086.431 / 5,043 s | 2.000 |
| 23:09:52 | 81,601.381 | 26,398.619 | 22,763.887 / 11,382 s | 2.000 |
| 23:33:49 | 84,475.066 | 23,524.934 | 302.054 / 151.1 s | 1.9986 |
| 23:42:25 | 85,507.321 | 22,492.679 | 262.427 / 131.3 s | 1.999 |
| 23:48:26 | 86,230.186 | 21,769.814 | 722.865 / 361.2 s | 2.001 |

So one uninterrupted 2x-rate session has been holding four L4s since before 18:36 and had not stopped
by 23:48, at about **120 quota-seconds per wall minute**; at that rate the remaining budget runs out in
about three hours. It is **not a kernel**: `kernels_list(mine=True, sort_by='dateRun')` reports the
newest run as `junzhejiao/gemma4-lora-serving-probe` at 2026-10-11T10:29:30Z with status `COMPLETE`,
and every other recent kernel is `COMPLETE` or `ERROR` with an older timestamp. The evidence is in
`artifacts/quota-drain-check.json` and `artifacts/quota-drain-check-2.json`, and the 20:00 reading was
written by another process to `runs/2026-10-11-rank8-pilot/quota-before.json`.

The `kaggle` and `kagglesdk` packages expose no session, stop or cancel call, so this can only be
stopped from the Kaggle web UI (an interactive session, or an orphaned worker of the kind this project
has already been bitten by: one earlier session held four GPUs for 3.5 hours after its notebook had
died). **This is now the blocking condition for all further spend.**

### Measured facts recovered from the existing probe output, at no quota cost

`api.kernels_output('junzhejiao/gemma4-lora-serving-probe')` downloads the completed probe's own
output files, so the L4x4 environment can be read without starting a session. Downloaded to
`artifacts/kaggle-probe-output/` (gitignored), it records:

- `visible_gpus: 4`, `torch_version: 2.10.0+cu128`, `vllm_version: 0.19.1`, and the mounted base path
  `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`.
- The wheelhouse inventory the probe installed: `compressed_tensors-0.15.0.1`,
  `transformers-5.13.1`, `vllm-0.19.1`, `bitsandbytes-0.46.1`, `safetensors-0.8.0`.
- **`CompressedTensorsConfig(dequantize=True)` requires the `compressed_tensors` package and names the
  required version.** Reproduced locally against `transformers 5.19.0`:

  ```
  ImportError: compressed-tensors>=0.15.0 is required for compressed-tensors quantization.
  Please install it with `pip install compressed-tensors>=0.15.0`.
  ```

  The mount's `compressed_tensors-0.15.0.1` satisfies that requirement exactly, which is a
  compatibility fact the plan did not have before.
- `lora_loaded_modules: 0` and `log_lines: 0` confirm the recorded caveat that the engine's own LoRA
  log lines are not capturable from a notebook.

The host's RAM and CPU count are **not** in that output, and the kernel log carries no `/proc/meminfo`
record, so host memory remains the single unknown that gates the FSDP loading strategy. The notebook
measures it in phase 0 before anything else is attempted.

## 3. Why the current recipe does not fit one L4
Recorded evidence, not inference:

| Quantity | Value | Source |
| --- | ---: | --- |
| H100 training peak allocation | 67.38 GiB | `training.peak_training_gib` in `resource/manifests/swe-lora-light.json` |
| Earlier full-replica peak | 71.65 GiB | `docs/experiments/2026-10-10-swe-lora-light.md` |
| L4 memory per GPU | 23,034 MiB (22.5 GiB) usable | measured in the served runtime |
| Trainable LoRA parameters | 122,429,440 | manifest |
| Base loading in the H100 trainer | `CompressedTensorsConfig(dequantize=True)` to BF16 | `scripts/train_swe_lora.py` on `experiment/swe-lora-v1` |

A BF16 replica of the base is roughly 58 GiB of the 67 GiB peak, so one L4 cannot hold a replica at
any dataset size. Four L4s do not pool memory by themselves: they need an explicit distribution.
How much of the 67 GiB is weights rather than activation is not recorded, and the transient terms
(the vocabulary projection and cross-entropy over the supervised positions) are the ones most likely
to decide the outcome at 22.5 GiB per GPU.

Note also that `scripts/train_swe_lora.py` refuses to start without `SLURM_JOB_ID` and selects
targets and a device map the way a single-node Trillium allocation expects. The trainer on PR #8 is
Trillium-only by construction; it cannot be pointed at a Kaggle session unchanged.

## 4. Candidate paths and the gate that decides each

Source-verified against a local `transformers 5.19.0` and `peft 0.21.2` checkout. The competition
image ships `transformers 5.0.0` (and the H100 recipe ran `5.18.0`), so every statement below is a
**prediction to measure**, not a result.

### Path A — BF16 base with FSDP sharding across four L4s (preserves the recipe's base loading)

The base is loaded as the H100 recipe loads it (dequantised to BF16, frozen) and FSDP shards it. Gate:
host RAM for the load, then the FSDP+PEFT build, then activation memory.

**The keyword that requests that loading mode changed name, and the mounted version takes the old one.**
`transformers 5.13.1`'s `CompressedTensorsConfig.__init__` is
`(..., run_compressed: bool = True, **kwargs)` and `get_loading_attributes()` returns
`{'run_compressed': ...}`; `5.18`/`5.19` take `dequantize: bool = False` and document
"`run_compressed=False` meaning `dequantize=True`". Passing `dequantize=True` to `5.13.1` therefore does
**not** fail: `**kwargs` is only consulted when `config_groups` or `kv_cache_scheme` is given (neither
is), so the keyword is dropped, `run_compressed` keeps its `True` default, and the checkpoint loads
**compressed** while the caller believes the base was dequantised. `from_dict` performs no renaming
either. A run in that state would record `dequantize: true` and train the packed weights, so
`resource/l4_feasibility/quantization_config_compat.py` now builds the config from the installed class's
signature, records which keyword it used and the effective mode, and raises rather than returning a
config whose mode disagrees with the request.

Two strategies are attempted in order, because they differ in how the adapter is exported:

| Strategy | How the adapter is exported | Memory requirement |
| --- | --- | --- |
| `fsdp-ignored` | LoRA modules are passed to FSDP as `ignored_modules`, so their parameters stay replicated and rank 0 holds the whole adapter; the export reads `named_parameters` directly, so no state-dict gather happens at all. Gradients are all-reduced explicitly, since FSDP does not synchronise ignored parameters, and the replicated parameters are broadcast from rank 0 after wrapping so every rank starts from the same adapter. | No parameter gather at all |
| `fsdp-all` | FSDP wraps the whole PEFT model; export gathers a full state dict with `FULL_STATE_DICT` + `rank0_only`. | A complete replica in rank-0 host RAM |

`fsdp-all` is skipped automatically when the host reports less than 70 GiB of RAM, and the skip is
recorded as a measurement rather than a silent fallback.

### Path B — the packed W4A16 checkpoint on one L4 (preserves the shipped base format)

Loading with `dequantize=False` keeps the checkpoint's own 4-bit packing, which is the format the
competition actually serves, and would fit a single L4. Four source facts say PEFT cannot train it:

1. `CompressedTensorsConfig.post_init` only forces `dequantize=True` when the checkpoint is *not*
   compressed, so for this QAT checkpoint `dequantize=False` is honoured and the weights stay packed.
2. `CompressedTensorsHfQuantizer._process_model_before_weight_loading` calls
   `compressor.compress_model(model)` for a compressed checkpoint, specifically so the packed
   checkpoint tensors have a module to load into. Those modules are compressed-tensors wrappers, not
   `torch.nn.Linear`.
3. `CompressedTensorsHfQuantizer.is_qat_trainable()` returns
   `dequantize or not is_quantization_compressed` — **False** for this checkpoint with
   `dequantize=False`, and `transformers/trainer_utils.py` raises
   "You cannot perform fine-tuning on purely quantized models" for that combination.
4. `peft 0.21.2` contains no compressed-tensors handling at all; its all-linear discovery is
   `isinstance(module, torch.nn.Linear)`, and its LoRA layer dispatch has no branch for a compressed
   wrapper.

The same file also needs the `compressed_tensors` package at construction time — `transformers` raises
`ImportError: compressed-tensors>=0.15.0 is required` without it — and the mount carries
`compressed_tensors-0.15.0.1`, which satisfies that requirement. So availability is **not** the
blocker for this path; PEFT's module-type handling is. The prepared model stack installs that wheel
together with the rest, since `compressed_tensors/logger.py` also imports `loguru` at module scope and
the image ships neither.

### Path C — NF4 re-quantisation of the QAT base (not used as the primary path)

`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')` would fit one L4 and is the
conventional QLoRA route, but it *changes the base*: the QAT calibration of a W4A16 checkpoint is
discarded, and the resulting adapter would be trained against weights that differ from the ones the
competition serves. It is therefore a distinct experiment requiring its own base-format and serving
compatibility checks, exactly as the task states. It is not substituted silently, and this notebook
does not use it.

### Path D — one process that shards the base across the four GPUs while loading (implemented)

The `torchrun` route materialises a **complete BF16 replica in host RAM on every rank** before FSDP shards
anything: the base dequantised to BF16 is about 58 GiB, four ranks need about 232 GiB, and this session's
host was measured at 188.69 GiB. Path D avoids that peak rather than working around it: one process loads
the dequantised base with `device_map='auto'` and a per-GPU budget, so each parameter lands on the device
accelerate assigns it and no replica is ever complete in one place. The frozen layers end up spread over
the four L4s (~14.5 GiB each at BF16), every LoRA parameter stays replicated and trainable, and the
adapter is exported by reading `named_parameters` directly — no gather, which is what `fsdp-ignored` was
for. The backward pass is ordinary accelerate hook plumbing, not hand-written pipeline code, so this is
closer to the reviewed design than the "manual layer-split with a hand-written backward" this section
originally sketched; that sketch is superseded.

It preserves everything the task requires: the same required QAT checkpoint, dequantised with the
loading-mode keyword this transformers version actually honours, the same native handoff contract, the
same supervised objective (imported from the four-GPU worker so both routes measure one thing), and the
same adapter export path that the verifier and the official-runtime reload already read.

`resource/l4_feasibility/train_l4_device_map.py` implements it and runs as its own notebook phase, after
the four-GPU attempt and only when no adapter exists yet. It records the `hf_device_map` placement, the
per-GPU peak allocated memory, seconds per update and the parameter devices, and it refuses its own
success in two cases that would otherwise look like one: more than 21.5 GiB on a single L4 (the base was
not spread) and most modules left in host RAM (the state this route exists to avoid).

## 5. The instrument

`scripts/prepare_l4_feasibility_notebook.py` generates a private Kaggle notebook with seven code
cells, each writing its result to disk as it runs, since a failed Kaggle session publishes nothing
until it ends:

| Phase | What it does | Output |
| --- | --- | --- |
| 1. Stage | Writes the five embedded workers after verifying their SHA256, records host RAM, CPU count, GPU names and memory, the base path and the visible wheel inventory, and stops when the session did not receive four L4s or the wheelhouse mount | `environment.json`, `plan.json` |
| 2. Prepare the model stack (same cell) | Runs `prepare_model_stack.py`: installs the mounted `transformers 5.13.1` with its `tokenizers`/`safetensors`, `compressed_tensors` and `loguru`, plus the `torchao 0.16.0` wheel `peft` demands, hash-checked against PyPI; then proves the chain in a fresh interpreter (`import transformers`, `CONFIG_MAPPING['gemma4']` resolving to `Gemma4Config`, `import peft`, `is_torchao_available()`, `CompressedTensorsConfig(dequantize=True)`, `AutoConfig.from_pretrained(base)`) and stops before any GPU work if it fails | `model-stack.json` |
| 3. Packed single-GPU probe | Loads the shipped W4A16 checkpoint under `dequantize=False` on one L4 and records what PEFT does with compressed module types | `packed-probe-result.json`, `packed-summary.json` |
| 4. Four-GPU training | `torchrun --standalone --nproc_per_node=<measured ranks>` under a hard timeout: native handoff contract, base load with the loading-mode keyword this transformers version understands, strategy ladder, one backward step, at most 8 updates, adapter export | `progress.jsonl`, `train-result.json`, `train-summary.json` |
| 5. Single-process sharded training | Only when no adapter exists yet and the session cap can still cover it (Path D): one process dispatches the dequantised base across the four L4s while loading, trains the same replicated adapter on the same example, and records the placement and per-GPU peaks | `device-map-progress.jsonl`, `device-map-result.json` |
| 6. Verify | Validates the exported adapter against the naming scheme the serving stack was measured to accept | `adapter-check.json` |
| 7. Reload | Installs the rest of the wheelhouse (vLLM and its dependencies) and reloads the adapter in the official vLLM runtime, comparing greedy top-logprobs with a repeat-base control | `serving-result.json` |
| 8. Summarise | Combines everything with the session wall time and the estimated quota charge | `result-summary.json` |

The model stack is prepared **before** any model is loaded, because the image cannot load the required
base at all: run 4 reached the base load and stopped there with
`ValueError: The checkpoint you are trying to load has model type 'gemma4' but Transformers does not
recognize this architecture` (`KeyError: 'gemma4'` from `CONFIG_MAPPING`), since the image ships
`transformers 5.0.0`. The wheelhouse's `transformers 5.13.1` contains `transformers/models/gemma4/` and
registers `("gemma4", "Gemma4Config")` in its generated `auto_mappings.py`. That upgrade cannot be made
alone: `peft 0.19.1`'s `is_torchao_available()` returns `False` only when torchao is *absent* and raises
`ImportError: Found an incompatible version of torchao ... only versions above 0.16.0 are supported` for
the image's `torchao 0.10.0`, so `torchao 0.16.0` is installed in the same step. The wheelhouse carries
no torchao wheel, so it is attached as a second dataset source and its bytes are checked against the PyPI
SHA256 before installation. The notebook kernel itself imports only the standard library, so none of this
can break the kernel that has to survive to the reload phase.

The five workers are real files under `resource/l4_feasibility/`, embedded byte-for-byte with their
hashes recorded, rather than strings inside the generator: brace escaping inside generated code has
already caused a lost GPU session in this project.

`resource/l4_feasibility/train_l4_feasibility.py` also validates the reviewed protocol contract on
the real tokenizer before training: the tool handoff token must be id 50, the template must not move
the prompt/target boundary, tokenization must not cross that boundary, and the supervised target must
end at the handoff token. That makes the notebook a check of the contract as well as of memory.

With four ranks and `gradient_accumulation: 4`, one epoch of the compact 128-action recipe is exactly
8 optimizer updates, so "at most eight updates" is one complete epoch of the registered recipe rather
than an arbitrary slice of it.

## 6. Measurement contract (deliverable 5)

Recorded by the notebook, for the report back to the task:

- **Package versions** — `torch`, `transformers`, `peft`, `accelerate`, `safetensors`, and the
  presence of `compressed-tensors`, `torchao`, `bitsandbytes`, plus the Python version, host RAM and
  CPU count. It also records whether `compressed_tensors` is importable and whether this
  `transformers` build's `CompressedTensorsConfig` accepts `dequantize`, because the image ships
  `transformers 5.0.0` while the H100 recipe ran `5.18.0` and the local analysis read `5.19.0`.
- **Per-GPU peak memory** — peak `torch.cuda.max_memory_allocated` **and** `max_memory_reserved` for
  each update, with the statistics reset at the *start* of the update so the peak covers the forward,
  the backward and the optimizer's lazily allocated state. A peak sampled after the forward would
  understate exactly the number this run exists to establish. The post-load figure is recorded too.
- **Seconds per update** — minimum, mean and maximum over the updates actually completed, measured
  across a whole update (forward included), not estimated from another device.
- **Physical GPU time versus quota use** — session wall seconds and the quota charge at the 2x L4x4
  rate, to be reconciled against a fresh `check_kaggle_quota.py` reading after the session.
- **Exact resume path** — every phase's file, the last `progress.jsonl` phase for a failed attempt,
  and the fact that no state is reused across sessions.
- **Adapter identity** — the exported adapter's SHA256, tensor count and validation verdict, and the
  official-runtime control and effect deltas.

The report must separate four outcomes that are easy to conflate: training memory feasibility, adapter
export validity, official-runtime honouring of the adapter, and agent repair performance. This
notebook measures the first three and says nothing about the fourth.

## 7. Planned setup to post before consuming shared compute

> **L4 training feasibility — planned setup (issue #13)**
>
> Owner: KaltistEsperanta. Base: the required `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`
> checkpoint, loaded exactly as the compact H100 recipe loads it (compressed-tensors dequantized to
> BF16, frozen). No NF4 re-quantisation is substituted.
>
> Quota at 2026-10-11 23:42 Asia/Shanghai: 22,492.7 s remaining of 108,000 s, refresh 2026-10-17 —
> **refresh this line with a fresh `check_kaggle_quota.py` reading before posting**, because a live
> session that is not ours was draining the account at the 2x rate when this was written and may have
> consumed more since. L4x4 charges at 2x wall time, so about 11,246 s of L4x4 wall time remain. The
> planned session is one private notebook, cap 3,300 s wall (~6,600 quota-seconds), with a hard 2,100 s
> timeout on the training subprocess so a failure cannot hold four GPUs. Nothing is launched until the
> draining session has stopped and the quota has been re-measured.
>
> Scope: one backward step, then at most 8 update steps (= one epoch of the 128-action recipe at four
> ranks), then save the adapter, then reload it in the wheelhouse vLLM runtime with a repeat-base
> control. No full corpus, no second epoch, no competition upload. Strategy order: FSDP with the LoRA
> modules ignored (no parameter gather), then whole-model FSDP only if the host reports enough RAM to
> gather a full replica. If both fail, the run records the exact exception and stops.
>
> What I will report: package versions, per-GPU peak memory, seconds per update, session wall time
> versus quota charge, the exact resume path, and either a reviewed training entry point or the
> specific measured incompatibility.

## 8. What was verified, and what was not

Verified offline, with results:

- `scripts/prepare_l4_feasibility_notebook.py --check` reports `status: ok`; every code cell compiles
  and all three embedded workers round-trip to their recorded SHA256.
- `tests/test_l4_feasibility.py`: 22 tests, 19 pass, 3 skipped with the reason recorded (this
  sandbox denies a confined process nested-directory creation, so the CLI-over-a-directory cases
  cannot run here; the decision rule they exercise is covered by the in-memory tests). The suite
  covers the key normalisation, the verifier's accept/reject rule, the trainable-versus-replicated
  parameter split that decides which gradients need an explicit all-reduce, the generator's embedding
  and scope guards, and the worker-to-notebook field names.
- The adapter verifier was run against the **real** served probe adapter
  (`artifacts/lora-probe`): 4 of 4 tensors match the scheme the competition stack was measured to
  accept, rank 8 from shapes, `in_features` 5376 and 8192 — independently reproducing the recorded
  shapes — and exit 0; the same adapter at an expected rank of 16 exits 1.
- The verifier was then run against the **real rank-32 all-layer adapter** the probe generated at
  runtime and that the served engine accepted (downloaded with `kernels_output`, 104,238,416 bytes):
  240 of 240 tensors match the served scheme, 120 `lora_A` and 120 `lora_B` across all 60 layers,
  rank 32 from shapes, `problems: []`, exit 0. The byte count matches the figure recorded in
  `WORKLOG.md` for that adapter exactly, so the naming scheme is now confirmed at two ranks and two
  very different file sizes rather than one.
- `CompressedTensorsConfig(dequantize=True)` was confirmed to fail without the `compressed_tensors`
  package, naming the required version `>=0.15.0` — the requirement the mount's `0.15.0.1` wheel
  satisfies. That is what justifies the notebook's single-wheel install step instead of assuming the
  image has the package.

Not verified, and not claimed: anything that needs the model weights, a GPU or vLLM. The torchrun
launch, the FSDP strategies, the memory arithmetic in section 3 and the vLLM reload have been
reasoned about and instrumented, not executed.

## 9. Open items

1. **Run 5 and reconcile it.** The environment conflict that stopped run 4 is fixed by the model-stack
   step; the session must now produce the training measurement (which strategy builds, per-GPU peak,
   seconds per update) or the next exact incompatibility. Reconcile the measured quota charge against
   the 2x L4x4 rule with a fresh `scripts/check_kaggle_quota.py` reading.
2. If `fsdp-ignored` builds, measure whether one graded 4K example fits; the synthetic example runs at
   2,048 tokens by default because the acceptance needs one backward step, not the full context.
3. If both FSDP strategies fail, implement Path D and re-measure.
4. Report two results to issue #13 that stand on their own: the native handoff contract verified on four
   L4 GPUs, and the measured `transformers 5.0.0` / `torchao 0.10.0` / `peft 0.19.1` version conflict
   together with its fix.
5. The attached torchao dataset is public and third-party. The session hashes the wheel against the PyPI
   digest before installing it, which is what makes that acceptable; if a first-party copy is ever
   published, switch the source and keep the same check.
6. The agent-side acceptance gates are unchanged by this work: this notebook measures training
   feasibility, not repair performance.
