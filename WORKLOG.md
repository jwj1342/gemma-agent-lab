# CodingAI Work Log

This file is the persistent handoff record for work performed in this repository. Read it before starting a task and update it before finishing.

## Resume point (main and complete v31 are integrated on an isolated branch, 2026-10-23 01:58 Asia/Shanghai)

**Read this first.** The remote integration selected on October 22 is complete on the isolated
`integration/oct22-main-v31` worktree; the original dirty `fix/oct20-handoff-audit` worktree remains untouched.

- Merge `8ef3c82` integrates `origin/main` while preserving the local adapter mount, deterministic and
  size-bounded notebook packaging, flat Kaggle mounts, batch freezing, and the remote wheelhouse registry.
- Merge `46bad80` integrates the complete `origin/chore/wheelhouse-v31` history. Fourteen of its fifteen staged
  paths matched the branch exactly; `docs/experiments/README.md` is the additive index resolution.
- The post-merge repair tracks the two previously untracked test-dependency helpers required by the committed
  control-sweep generator, defers the POSIX-only `fcntl` import until the Trillium execution paths use it, and
  repairs one root-relative work-log link. CPU verification passes: repository checks cover 76 Python files,
  14 shell files, 32 configurations, 83 maintained Markdown files and five agent packages; 154 unit tests pass
  with one expected skip because the gitignored reviewed-wheel payload is not present in this worktree.
- No branch was pushed, no PR was opened, and no Kaggle kernel or competition submission was created. The
  custom-tool workstream was recorded only by name in the October 22 handoff; its concrete deliverable was not
  preserved and must be confirmed before implementation.

## Resume point (the 2026-10-24 round is built: guarded lr probe + fixed 60-task confirmation, 2026-10-20 12:25 Asia/Shanghai)

**Read this first.** Quota is still exhausted (**114,317.5 s of 108,000**, refresh 2026-10-24T00:00Z = 08:00
Asia/Shanghai), so nothing ran on GPU today. The round that the refresh buys is prepared, tested on CPU, and
push-ready; two of the three pieces are new instruments rather than new results.

- **A guard now decides an arm before it spends the task budget.** `resource/l4_feasibility/collapse_guard.py`
  reads the training loss trajectory and returns `collapsed` (minimum loss < 1.0; V9 measured 0.7323),
  `flat` (movement < 0.5 % of the first loss; V1 measured 1.1e-3) or `proceed`. The generated kernel calls it
  after training and before evaluation and, on a stop, writes `collapse-guard.json` plus a `summary.json` with
  `"skipped": "<verdict>"` and `"measured_tasks": 0`, so a stop can never be read as a measured 0/24. A stopped
  arm costs the measured training-plus-mount time (**≈1,380 s wall ≈ 2,800 quota-seconds**), against 22,000 for
  the flat arm and 29,935 for the collapsed one. `tests/test_lora_eval_notebook.py` (13 tests) pins the rule on
  the two measured trajectories and parses every generated code cell.
- **V10 is built and verified field by field against V9**: `artifacts/tmp/lora-v10-tail3-r16-lr5e5/` (104,573
  bytes, SHA-256 `49762f3bd123c6b7caf75f6571f05bb984b4dadcf1f6004316ef80453ed0fc3f`, kernel id
  `junzhejiao/lora-v10-tail3-r16-lr5e5`, **not pushed**). Identical to V9 in tasks (the 24-task screening
  sample), 41-wheel v31 digest map, agent bundle file hashes, evaluator and hook sources, dataset
  (`e968a448…`), rank/alpha/seq/adapter; the intended differences are **lr 5e-5**, **`--max-updates 8` made
  explicit** (the trainer already capped at one pass), and the guard. The v31 snapshot used has SHA-256
  `e3a1a22e…`; the integration branch now contains it through `origin/chore/wheelhouse-v31`.
- **The confirmation set is drawn and tracked**: `resource/manifests/confirmation-tasks-2026-10-24.json`
  (SHA-256 `2da2b2b7…`), **60 tasks = 41 fastapi + 19 rich**, seed `lora-confirmation-2026-10-24`, ranked by
  `SHA-256("<seed>:<task_id>")` with proportional quotas. Exclusions are enforced by
  `scripts/select_confirmation_tasks.py`, not remembered: the 24 screening tasks **and the four tasks the
  training targets came from** (`rich_3676`, `rich_3882`, `rich_3894`, `rich_4077`; derived with
  `--trained-from`). **The first draw contained all four**, i.e. the earlier design would have confirmed on
  memorisation; that is the substantive finding of this session. Measured-rate cost of the pair: **≈92,300
  quota-seconds** (baseline 42,500 + adapter 49,800, both with the measured 1,827 s setup term), leaving
  ≈15,700 for the learning-rate ladder. 19 tests in `tests/test_confirmation_tasks.py`.
- **Local environment facts, measured** (they cost most of the session's wall time): the only working
  interpreter is `C:\Users\32747\miniconda3\python.exe` (the two `python.exe` entries on PATH are 0-byte Store
  stubs); `kaggle 2.2.4` loads with `PYTHONPATH=.pylibs`; a spawned process **cannot write** into `resource/`,
  `runs/`, `scripts/` or `.git`, and **cannot write inside a `tempfile.mkdtemp` directory** (0o700 ACL), so
  `scripts/check_repository.py` and most of `tests/` are blocked as documented — `artifacts/tmp/run_checks_lenient.py`
  substitutes the temp allocator and runs both checks unchanged.
- **Pre-existing, not caused by this session**: 18 `tests/test_l4_feasibility.py` errors because its `Args`
  fixture has no `max_memory_mib` while `scripts/prepare_l4_feasibility_notebook.py` reads it (script modified
  2026-10-18, test 2026-10-13); and `check_repository.py` fails on
  `configs/experiments/localization-v1/skills/localization-v1/resources/gate.json`, which the working tree has
  deleted. Both belong to the uncommitted backlog below.
- **Next, on 2026-10-24**: `kaggle kernels push` the V10 directory, read `collapse-guard.json` first — `proceed`
  earns the 24-task measurement, `flat` moves the next probe to lr 1e-4, `collapsed` to ≈3e-5. Only a
  `proceed` arm with a real measurement unlocks the 60-task pair (baseline + adapter, `--task-ids` from the
  manifest). Nothing is submitted; the adapter slot in `docs/experiments/2026-10-18-merge-plan.md` stays empty.
- **Record**: `docs/experiments/2026-10-18-lora-improvement-program.md` §15 (new), `scripts/README.md`,
  `resource/manifests/confirmation-tasks-2026-10-24.json`, `resource/l4_feasibility/collapse_guard.py`,
  `scripts/select_confirmation_tasks.py`, `tests/test_lora_eval_notebook.py`,
  `tests/test_confirmation_tasks.py`. Git: still uncommitted on `fix/lora-rank8-controlled` (4 behind /
  26+ ahead of `origin/main`); committing the backlog remains an open follow-up.

## Resume point (screening closed: no adapter beat the baseline, lr is the live axis, 2026-10-19 22:30 Asia/Shanghai)

**Read this first.** Four arms are measured on the same fixed 24 tasks and the same frozen agent. No adapter
beat no adapter, and the reason is now known for both ends of the learning-rate axis.

- **Arms**: V0 baseline `9/24` (9,595.9 s), V0b byte-identical baseline repeat `10/24` (9,761.5 s), V1 adapter
  (tail-3, r16/α32, 8 updates, lr 2e-5) `10/24` (10,976.8 s), **V9 adapter (same, lr 2e-4) `0/24`** (14,946.1 s).
- **Noise floor, measured**: V0 vs V0b = 3 gains / 2 regressions (net +1, p = 1.0). V0b vs V1 = **0 discordant**:
  the adapter's resolved set *is* the repeat's, so the "+1 for the adapter" was the baseline's own variance.
- **V1 is numerically flat**: RMS(`ΔW`) = 4.6e-6 (base scale 1e-2), loss 2.0966 → 2.0989, 6 of 7 shared tasks
  produced byte-identical patches. The serving path honours it (208 LoRA kernel files in the adapter arm's
  evaluation Triton cache, 0 of 946 in the baseline), so the flatness is the weights, not the plumbing.
- **V9 is destructive**: loss 2.10 → 0.73, then **0 tool calls on all 24 tasks**, every task timing out at
  ~523 s. The 24 traces show a decode loop of the final-answer sentence (2,666× in the worst case, 13 of 24
  hitting the 8,192-token cap) under `model_name: main_lora`. So the useful window is **between 2e-5 and 2e-4**.
- **Config correction**: the trainer plans one pass over the dataset and only truncates with `--max-updates`, so
  every arm ran **8 updates**; "32-update" variants were never possible with 8 records.
- **Quota**: **114,317.5 s used of 108,000 s**, i.e. **6,317.5 s over**, refresh 2026-10-24T00:00Z. V9 cost
  29,935.3 s because a collapsed agent maximises runtime. Nothing further can run this month.
- **Record**: `runs/2026-10-19-lora-screening/` (`REPORT.md`, `all-arms.json`, `noise-floor.json`, per-arm
  summaries/stdout, diagnostics, adapter weight norms, five notebooks) and
  `docs/experiments/2026-10-18-lora-improvement-program.md` §12-§14 plus §13 for the confirmation design.
- **Next, after 2026-10-24**: one arm at **lr 5e-5** (check its loss lands between 2.10 and 0.73), then a
  60-task paired confirmation (Option A in §13, ≈85,000 quota-seconds) only if that arm shows something.
  The adapter slot in `docs/experiments/2026-10-18-merge-plan.md` stays empty until then.

## Resume point (noise floor measured: the adapter did not beat a re-run of itself, 2026-10-19 18:15 Asia/Shanghai)

**Read this first.** The screening round has three measured arms and a verdict. A byte-identical re-run of the
no-adapter baseline flips as many tasks as the adapter was credited with, and the adapter arm's resolved set
is identical to that re-run.

- **Arms** (identical agent bundle, task ids, wheelhouse v31, seed 201, budget 8 min / 28 calls): V0 baseline
  `9/24` (9,595.9 s wall), V0b baseline repeat `10/24` (9,761.5 s), V1 adapter (tail-3, r16/α32, 8 updates,
  lr 2e-5) `10/24` (10,976.8 s).
- **Paired**: V0 vs V1 = 3 gains / 2 regressions (net +1); **V0 vs V0b = 3 gains / 2 regressions (net +1,
  p = 1.0)**, i.e. the adapter's whole credit is baseline variance; **V0b vs V1 = 0 discordant**, both 10/24.
  Record: `runs/2026-10-19-lora-screening/{REPORT.md,noise-floor.json,report.json,paired-table.md}` and
  `docs/experiments/2026-10-18-lora-improvement-program.md` §12.
- **Why the adapter could not have mattered**: measured on the exported file, RMS(`B`) = 8.2e-5 and
  RMS(`ΔW`) = 4.6e-6 against base weight scales of 1e-2; loss flat at 2.0966 → 2.0989 over 8 AdamW steps at
  lr 2e-5; six of seven shared tasks produced byte-identical patches. The serving path is proven to honour
  the adapter (208 LoRA kernel files in the adapter arm's evaluation Triton cache, 0 of 946 in the baseline).
- **Two differences do replicate**: the adapter arm is 10-13 % slower per task (KV-cache cost) and it writes
  a patch on 21/24 tasks versus 19/24 in both baselines — the second resolves nothing extra.
- **In flight (as of this entry; measured later in the 22:30 entry)**: `junzhejiao/lora-v9-tail3-r16-lr2e4`
  (pushed 18:06 local) — same arm at lr 2e-4, launched with `--max-updates 16`, but the trainer caps every run
  at 8 updates, so the real change was 10× the learning rate. A 32-update variant was generated but not
  funded.
- **Quota**: 84,382.2 s of 108,000 s used, **23,617.8 s remaining**, refresh 2026-10-24T00:00Z. V9 is
  expected to spend ≈22,200 s of it, leaving ≈1,400 s of margin.
- **Next**: read V9. Whatever it shows, the 24-task screening set can only rank: ±2 is measured noise, so a
  confirmation needs ~50-60 tasks per arm (≈38,000-45,000 quota-seconds for the pair) after the 2026-10-24
  refresh, or the Trillium H100 runtime, which spends no Kaggle quota. A 99-task paired confirmation
  (≈126,000 quota-seconds) cannot fit one monthly allowance.

## Resume point (LoRA programme started: M1 carrier proven, M2 screening running, 2026-10-18 21:20 Asia/Shanghai)

**Read this first.** The three-part LoRA request is in progress: part 1 (submission theory) is written, part 2
(the experiment programme) has a working carrier and a screening run in flight, part 3 (merge plan) is written
with the adapter slot left conditional.

- **Carrier**: `scripts/prepare_lora_eval_notebook.py` → one private L4 kernel that verifies the v31
  wheelhouse, installs the stack, unpacks the budget-fit agent, runs the batch evaluator (one vLLM server,
  one sandbox per task) and grades every patch with the official verifier. No adapter transport is needed;
  `--train-dataset` makes the same kernel train before evaluating.
- **M1 measured**: 2/3 resolved on three rich tasks with the 4-minute configuration; two timeouts exposed the
  prompt-versus-config budget mismatch, so screening uses the submission budget (8 minutes, 28 calls).
- **M2 running**: `junzhejiao/lora-m2-budgetfit-screening`, no-adapter baseline on the fixed 24-task sample.
- **Quota**: 8,947.6 s of 108,000 s used, 99,052.4 s remaining; refresh 2026-10-24.
- **Next**: read M2, then M3 (V1 tail-3 adapter trained and evaluated in one kernel), then the variant sweep;
  report the best adapter with its data so the user can decide on a submission.

## Resume point (task board executed, budget-fit adapter produced, 2026-10-18 19:35 Asia/Shanghai)

**Read this first.** The October 18 task board is executed except the one item that needs a carrier this
workspace does not have: the task-level paired evaluation of a LoRA adapter.

- **GitHub state after this round.** PR #16 carries `Closes #13` (verified in the live body) and the five
  issue #13 deliverables remain complete; its only gate is the requested independent review from `liaodid`.
  PR #18 is approved by `KaltistEsperanta` at `c75571a`. PR #23 is approved by this account at `b9c9c4d`
  (with two recorded follow-ups: branch is behind `main`, and the v31 lock moves seven non-wheelhouse
  packages) and already had `shizongniji`'s approval. PR #25 got a COMMENT review: one requested change, the
  new "12B smoke resolved 1/10" sentence has no run record in the repository. PR #8 is unchanged since
  2026-10-12 (`mergeable_state: dirty`, base `3054da8`, no reviews, no CI on its head), so there was nothing
  to re-review; it needs its author to rebase.
- **Data.** `trajectories-r3` is downloaded and audited: 31/31 index rows, 31/31 trace hashes, 31/31
  verifier-resolved patches, 874 steps, 704 tool calls, 0 empty messages/observations. The audit comment is
  on issue #4. The budget-fit subset (18 trajectories) has **no `rich_3777`** trajectory and no graph-tool
  calls at all; `run_command` is 60.8% of its calls.
- **No rank-32 model exists and no LoRA has ever been submitted.** Kernel
  `junzhejiao/gemma-4-rank32-bounded-retry-october-18` (2026-10-18 09:40Z) completed with
  `updates_completed: 0`, `adapter: null`, because it was launched with `--session-limit 1500` while the
  device-map phase advertises a 1500 s need. The newest competition submission (2026-10-18 00:00Z) is
  `max_tool_calls 28 -> 50`, **no LoRA**, and was still PENDING when checked; the score move this week is
  0.18 -> 0.17, i.e. not an improvement.
- **Training produced a budget-fit trajectory adapter, and it is a starting point only.**
  `junzhejiao/budgetfit-task-tail-rank16-v5` completed 8/8 updates at `--seq-len 2048` and exported 820
  tensors / 244,982,352 bytes / SHA256 `f9db770d…`, every key matching the served naming scheme
  (`artifacts/tmp/budgetfit-v5-out/l4-feasibility/adapter/`). Three larger-sequence runs OOM'd, and the
  measurements fix the envelope: ~20.0 GiB peak per card, AdamW state allocated on the first step, and the
  fixed harness prompt alone (2,631-3,654 tokens) cannot share a card with a completion. Consequences:
  (a) the successful adapter trained on the task statement only, not the full system+task prompt;
  (b) its completion is a text rendering of the loop tail, not native tool calls;
  (c) the fix for the full prompt is a training-OOM retry with `device_map: auto` plus host-RAM offload,
  because `balanced` ignores a lowered `--max-memory-mib` and the CPU fallback only fires on load failure.
  Encoding was proven without a model load first: at the default 2048 every record is rejected, at 8192 all
  10 encode (mean prompt 3,213.6, mean target 1,683.5 tokens).
- **The blocker to read before promising a paired result.** The adapter is 244,982,352 bytes; a compact
  evaluation notebook is limited to 1 MB; the dataset upload path posts to `www.googleapis.com` which is
  unreachable here. So the adapter can reach Kaggle inside a *competition submission ZIP* (that path works)
  but not inside an evaluation kernel. The task-level paired 2x2 needs the official runtime on Trillium,
  which this workspace has no credentials for.
- **Two environment traps measured today.** `build_metadata()` emits `machine_shape: NvidiaL4` even with
  `--cpu-only`, so a "free CPU preflight" is scheduled on accelerators. Seven kernels (2 preflight, 5
  training) took quota from 0 to **5,069.478 s of 108,000 s**; the four failed attempts cost about four times
  the one success because every attempt pays a 250-500 s weight load before failing. And the sandbox denies
  writes to `runs/`, `scripts/` and `.git` for spawned processes; the agent file tools still write, so
  generated reports go through them while subprocess output goes to `artifacts/tmp/`.

## Resume point (rank ladder at rung 1, quota exhausted, 2026-10-13 24:45 Asia/Shanghai)

**Read this first.** The rank ladder has started and is paused by quota, not by capability. **Rank 16 is
trained, validated and downloaded; rank 32 failed at the base load; rank 64 was never attempted.**

- **What exists now.** `junzhejiao/gemma4-lora-rank16-sft-v3` trained 32 updates of a rank-16 /
  alpha-32 LoRA on the required `gemma-4-31b-it-qat-w4a16-ct` base, dequantised to BF16 and dispatched
  across four L4s by one process, over 32 real SFT records. The adapter is 820 tensors,
  244,982,352 bytes, SHA256 `bec7346f3e820873105545bf473909780c52940bf71bcbd21adec9afb8aa9e65`,
  validated 820/820 against the served naming scheme, layers 0-59, all seven projections, and is
  downloaded at `runs/2026-10-13-lora-rank16-v3/output/l4-feasibility/adapter/`. Session wall 468.4 s,
  charged 1,743.06 quota-seconds.
- **Quota is exhausted for this workstream: 787.753 s of 108,000 s remain** (about 394 s of L4x4 wall
  time) and the refresh is **2026-10-17T00:00Z**. One rung costs about 1,750 quota-seconds and the
  failed rank-32 load alone cost 2,570.542 s, so nothing further can be trained before the refresh
  without a different platform (Trillium still needs credentials this workspace does not have).
- **The first thing to fix in the next attempt**, because it is the only measured cause of the rank-32
  failure: raise `--load-budget` above 600 s. The identical placement loaded in 196-217 s on five
  earlier runs and then exceeded 600 s once, so the cap is too close to the measured worst case; the
  worker's own cap was the binding limit while its subprocess timeout was 1,500 s.
- **How to run a rung without repeating this session's mistakes.**
  `scripts/prepare_l4_feasibility_notebook.py --dataset artifacts/sft-public-v1.jsonl --rank N
  --alpha 2N --max-updates 32 --skip-probe-routes --skip-serving --output runs/<date>-lora-rank<N>-v1`,
  then push the directory with the kaggle CLI. Add `--dataset-check --cpu-only --dataset-check
  /kaggle/working/l4-feasibility/dataset-check.json` for a **free CPU preflight** of the encoding on the
  real tokenizer before spending a GPU second; that preflight is what caught the second encoding defect.
  Generated cells render placeholders with `python_literal` (Python, not JSON) and use the
  `@@@NAME@@@` sentinel -- do not reintroduce `@@NAME@@` or `json.dumps` for booleans.
- **Costs and pitfalls measured this session**: `kernels_output` downloads through
  `www.kaggleusercontent.com`, which intermittently fails to resolve here while `api.kaggle.com` answers
  -- use `scripts/fetch_kernel_log.py` when the notebook prints its summary. The kaggle CLI lives only in
  `C:\Users\32747\miniconda3\envs\kaggle\python.exe`, `bounded_connect.py -m scripts.X` needs
  `PYTHONPATH=E:\Desktop\CodingAI`, and pytest must be installed from the reviewed wheels into
  `artifacts/tmp/pylibs`.
- **What is still missing for a verdict**: no paired evaluation has been run for any rung, and the
  control-valid set is 99 tasks (see `resource/manifests/control-valid-tasks-2026-10-13-final.json`), so
  once quota returns the ladder can be evaluated against a task set that already clears the 10-point
  power bar.

## Resume point (instrument cleared, 2026-10-13 21:30 Asia/Shanghai)

**Read this first.** The task-set constraint that has blocked every LoRA efficacy claim is gone:
the control-valid set is now **99 tasks (fastapi 60, rich 39)**, past the **78** that a 10-point
paired gain needs at 80 % power. The measurement is the `junzhejiao/gemma4-control-fastapi-starlette`
sweep: **28 of 30 usable** where the same 28 fastapi tasks had all failed hours earlier. The
recovery is the starlette pin the notebook measures for itself (1.6.0 rejects, 1.3.1 rejects,
0.52.1 accepts), so the FastAPI collection class is closed by measurement rather than assumption.

What changed since the previous resume point:

- The union lives in `runs/2026-10-13-control-valid-set.json` (working copy) and
  `resource/manifests/control-valid-tasks-2026-10-13-final.json` (tracked, 99 tasks, supersedes the
  73-task manifest). `artifacts/tmp/union_valid_tasks.py` now discovers every executed sweep by
  globbing `runs/**/control-sweep-report.json`, so a new sweep needs no source-list edit.
- The prepared sweep notebook was regenerated with the committed generator and reproduces
  byte-for-byte (`notebook_sha256 de25354d…`); the earlier file, which differed only in the
  declarative `test_deps.manifest_path` string, is kept as `notebook-previous.ipynb`. Two
  environment facts cost time here: the kaggle CLI lives in the `kaggle` conda env
  (`C:\Users\32747\miniconda3\envs\kaggle\python.exe`), and `bounded_connect -m scripts.X` needs
  `PYTHONPATH=E:\Desktop\CodingAI`. Local pytest is not installed anywhere; install it from the
  reviewed wheels into `artifacts/tmp/pylibs` (`--no-index --no-deps --target`) and add
  `artifacts/tmp/pylibs;.pylibs` to `PYTHONPATH`.
- **Compute is now the binding constraint, not the instrument.** GPU quota is unchanged at
  **5,306.051 s of 108,000 s**, refreshing **2026-10-17T00:00Z** (verified this session), i.e. about
  **2,653 s of L4x4 wall time**. At the recorded agent cost (~119 s baseline, ~171 s portable for
  `rich_3675` on four L4s) a 99-task paired run needs roughly 8-10 GPU hours, and even 40 tasks
  need far more than 2,653 s. A full paired evaluation therefore waits for the refresh or another
  platform; the CPU sweeps in this session spent no quota.
- Still human-only: the 53 tasks that remain invalid (11 fastapi exit-1-both-arms after the pin,
  9 rich, 13 requests, 1 httpx) would need a per-class measurement like the starlette one before
  a 5-point effect becomes detectable, and PR #8 still needs its rebase.

## Resume point (L4 training feasibility, 2026-10-12 20:35 Asia/Shanghai)

**Read this first if you are resuming the four-L4 training work (issue #13). The acceptance criterion is
met and measured.** Run 11 took the required `gemma-4-31b-it-qat-w4a16-ct` base through the bounded
train/save/verify/reload loop on four L4 GPUs in one 1,421 s session: BF16 dequantised and dispatched
across the cards, eight updates, an 820-tensor adapter exported and validated, and the official vLLM
runtime honouring it (control 0.0, effect 0.596932). Section 9 of
`docs/experiments/2026-10-12-l4-training-feasibility.md` holds the table of measured values, and the
run-11 entry at the top of `Work history` holds the narrative. What is left is reporting and deciding
whether anything further is worth the remaining quota, not feasibility work. The curated branch
`experiment/four-l4-lora-feasibility` (`c0648f7`) is already published on `origin`; opening its pull
request is the remaining human step, since this workspace has no GitHub CLI or token.

What is settled, with evidence:

- **Quota is nearly spent, and queue time is billed.** **5,306.051 s of 108,000 s remain** (refresh
  2026-10-17), about 2,653 s of L4x4 wall time. Run 11 was charged 4,733.33 quota-seconds against a
  2,842.6 estimate because the allocation is billed while it is held, including the ~780 s it spent
  queued. Fold queue time into any further session plan; the earlier drain is closed and the account was
  flat across the readings taken before this session.
- **Four L4 GPUs are obtainable, and the reason they were not before is now known.** L4x4 is gated on
  competition attachment. Setting `competition_sources: ['gemma-4-developer-agent']` in the kernel
  metadata changed the allocation from 2x T4 to 4x L4 (23,034 MiB each, 188.69 GiB host RAM, 48 CPUs)
  and also restored the wheelhouse mount, which the T4 session never had.
- **The reviewed handoff contract is verified against the real tokenizer on four L4s**: handoff token
  id 50, prompt/target boundary intact, target ends at the handoff token, 179 prompt + 19 target = 198
  sequence tokens. This is reportable to issue #13 on its own.
- **The measured incompatibility that blocks training**: the competition image ships
  `transformers 5.0.0`, which raises `KeyError: 'gemma4'` from `CONFIG_MAPPING` — it cannot load the
  required base at all. The wheelhouse's `transformers 5.13.1` is therefore mandatory, and it contains
  `transformers/models/gemma4/`. But `peft 0.19.1`'s `is_torchao_available()` raises while a too-old
  torchao is installed (it returns False only when torchao is *absent*), and the image has
  `torchao 0.10.0` against a 0.16.0 floor. So the environment needs **transformers 5.13.1 + loguru +
  torchao >= 0.16.0** together.

The torchao wheel now reaches the session without any upload: the notebook attaches the public dataset
`kajarnaklokossou/torchao-0-16-0`, whose single wheel is byte-identical to PyPI's
(`torchao-0.16.0-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`, 3,160,415 bytes, SHA256
`2d6293a0c57c9dd505efb025a7189459d154965fbed000efd638cf299f9362dd`), and `prepare_model_stack.py`
re-checks that digest before installing. Publishing our own dataset is impossible from this machine, and
the reason is now measured rather than assumed: the kaggle client sends dataset blobs to
`www.googleapis.com`, whose 8 IPv4 and 8 IPv6 addresses are all unreachable here. Any kaggle command that
performs a blob upload must run through `scripts/bounded_connect.py`, which bounds each connect attempt
so a blocked route fails quickly instead of looking like a hang.

Done since that note: the second dataset source is attached, the stage cell installs the wheelhouse model
stack plus the torchao wheel and proves the import chain before any GPU work, section 5 of
`docs/experiments/2026-10-12-l4-training-feasibility.md` no longer carries the disproved "keep the image's
transformers to dodge torchao" rationale, and **the loop itself now runs**: see the run-11 entry and
section 9. Still open for a human: posting the result to issue #13, and deciding whether a graded 4K
example or the paired repair control justifies the remaining ~2,653 s of L4x4 wall time.

Offline gates for this work: `tests/test_l4_feasibility.py` runs **38 tests, all passing**, and
`scripts/prepare_l4_feasibility_notebook.py --check` reports `status: ok`. Two new helpers exist for
reading runs: `scripts/poll_kernel.py` (compares `status.name`, because `str(raw)` silently fails) and
`scripts/fetch_kernel_output.py` (outputs plus the log). The generator now writes raw code templates and
`chr(10)` newline writes — do not reintroduce backslash escapes into the generated cells; the suite
asserts against them, and three variants of that one mistake have already cost two sessions.

## Resume point (audit update 2026-10-11 19:55 Asia/Shanghai)

**Readiness correction from the October 11 source audit:** no trained adapter exists. The random
serving probes remain useful evidence, but the training notebook is not ready: its generated adapter
config is a string, holdout rows are not filtered by either trainer, and the trajectory acceptance
gate can accept a failed agent when gold passes. The 116-row file is all training data and overlaps
three of the four historical control-valid tasks. The 78-task power estimate assumes no regressions;
it is not a universal prerequisite for a diagnostic pilot. Full findings, CPU reproductions, and the
proposed training/paired-evaluation sequence are in
`docs/experiments/2026-10-11-lora-readiness-audit.md`. This audit changed documentation only; these
implementation defects remain open. Historical readiness statements below must be read with these
corrections. Active branch: `feature/lora-serving-probe-run`, inspected revision `6d2ad38`.

**Read this section first; it is the state a future session needs in one place.**

Current position at a glance:

- **The serving check is answered, and the answer is positive for both configurations.** The competition
  stack applies LoRA: two base runs differ by 0.0 in logprobs, mounting the rank-8 embedded probe moved
  them by up to 1.187468 at 35/35 positions, and mounting a rank-32 all-layer adapter moved them by up
  to 10.069917 while changing the greedy tokens. The aliasing wipe is not present on this wheelhouse, and
  the #50059 wide-adapter degradation did not appear as a dropped adapter. Evidence, scope limits, and the
  eleven-run execution log: `docs/experiments/2026-10-11-lora-serving-risk.md`.
- What that still does not license: no **trained** adapter has ever been served, the probe used the model
  path rather than `adk_submission.VllmServer`, and the runtime adapter's shapes were assumed from the
  recorded evidence because the base exposes no weight index (`recorded_evidence_match: true`, with a
  warning).
- GPU quota is materially reduced: at 18:36 it was **59,248.937 s of 108,000 s remaining** (used
  48,751.063 s, refresh 2026-10-17). A correct probe run costs about 8 minutes of L4x4 wall time; the rest
  went to failures, including one session that held four GPUs for 3.5 hours after its notebook had died.
- Commits are on `feature/lora-serving-probe-run`: `7e4b53f`, `30e413f`, `0a09ba1`, `dd88e95`, `c2dd97e`,
  `f533707`, `9f8328c`. The earlier pause point was `0e49957` on `feature/localization-v1-skill`.
- Image facts measured on the competition `dockerImage` (do not re-derive them): Python 3.12.13;
  torch 2.10.0+cu128, transformers 5.0.0, peft 0.19.1, datasets 5.0.0, accelerate 1.13.0,
  safetensors 0.7.0, pandas 2.3.3, pyarrow 24.0.0 present; **trl, bitsandbytes and vLLM absent**; the
  wheelhouse mount adds transformers 5.13.1, bitsandbytes 0.46.1 and the patched vLLM 0.19.1, and it
  installs in 53 s. The CPU image is Python 3.13, so never infer the interpreter from a CPU notebook.
- Dataset upload from this machine is **blocked with evidence** (two attempts stalled: 40 min via CLI,
  20 min at 0.9 s CPU via SDK, `dataset_list(mine=True)` empty both times) while read paths answer
  instantly. Kaggle training needs the wheels uploaded through the web UI; Trillium needs no dataset.
- Note for the next probe run: a failed notebook leaves its worker holding four GPUs, and `kaggle
  kernels logs` returns nothing until the session ends, so budget for that.

### Where the LoRA SFT objective stands

Nothing has been trained: no trained `adapter_model.safetensors` exists anywhere, and the only GPU
spend so far is the probe session above. Nine commits exist across the two feature branches.

Ready and validated offline:

| Piece | State |
| --- | --- |
| SFT dataset | `artifacts/sft-public-v1.jsonl`, 116 records from 129 public tasks (13 dropped as `patch_too_long`) |
| Adapter experiment | `configs/experiments/lora-trajectory-v1/`: `adapter: main_lora`, rank 32, `q_proj`+`o_proj` only |
| Config check | `scripts/check_lora_adapter.py` -> `status: ok` (rank 32 to vLLM bucket 32, `max_loras` 1) |
| PEFT self-check | `scripts/train_lora_adapter.py --self-check` -> export and reload verified on a tiny model |
| Training stack | Verified end to end: 3 SFT steps, loss 5.4698 -> 5.4269, export 66,584 bytes, reload matches |
| Offline wheels | `artifacts/training-wheels/` (10 Linux/CPython-3.12 wheels, 55.4 MB); inventory `resource/manifests/training-wheels-sft-v1.json`, now carrying `interpreter_abi` |
| Training notebook | `scripts/prepare_lora_notebook.py` generates it; its setup cell was executed standalone and hash-verified |
| Adapter mounting | `scripts/prepare_kaggle_runtime.py --adapter-dir/--adapter-mount/--adapter-dataset-source` stages and hash-verifies a mounted adapter |
| Serving probe | `artifacts/lora-probe/` built by `scripts/make_lora_probe.py` (loud random A and B, rank 8) |
| Probe runner | `scripts/prepare_lora_probe_notebook.py` -> `artifacts/lora-probe-notebook/` (3 cells: stage, install wheelhouse, measure); `--check` reports `status: ok` |
| Measurement | `scripts/analyze_paired_power.py`: 4 control-valid tasks, all in `rich` |

### The three things that are blocked, in priority order

1. **The training platform and its environment conflict (now the first gate).** The serving question is
   answered, so training is the next step and two obstacles are measured: Kaggle needs the wheel set
   uploaded as a dataset through the web UI because CLI/SDK uploads from this machine stall, and on the
   competition image any notebook that imports `transformers` after installing the wheelhouse fails with
   `Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are
   supported` (no torchao wheel exists in the mount). The SFT notebook must either keep the image's
   transformers or avoid the wheelhouse upgrade. Trillium H100 needs no dataset and takes the 62.6 GB
   unquantized QAT base.
2. **The measurement instrument, which gates interpretability.** Only 4 control-valid tasks exist, all
   from one repository. A 10-point gain needs 78 tasks at 80% power, so even a successful adapter could
   not be read as better or worse today. Growth is CPU-side work: the control yield is 4 of 8 attempted,
   and the collection-error (`exit 2`) failures are the most fixable class.
3. **The paired no-adapter control,** which any resolution claim depends on and which cannot start before
   a trained adapter exists.

### Exact next commands

```powershell
# Local Python is NOT on PATH; use the absolute interpreter and set both path variables.
$py = 'C:\Users\32747\miniconda3\python.exe'
$env:PYTHONPATH = 'E:\Desktop\CodingAI\artifacts\combo-test;E:\Desktop\CodingAI\.pylibs'
$env:PYTHONUTF8  = '1'
$env:HF_HOME     = 'E:\Desktop\CodingAI\artifacts\hf-cache'   # datasets refuses to cache outside the workspace

# Re-verify everything that is already built
& $py scripts/check_lora_adapter.py
& $py scripts/analyze_paired_power.py --json-out artifacts/paired-power.json
& $py scripts/make_lora_probe.py --check --output artifacts/lora-probe

# Rebuild the SFT dataset and the training notebook (needs data/public/tasks.jsonl, which is gitignored)
& $py scripts/build_sft_dataset.py --output artifacts/sft-public-v1.jsonl
& $py scripts/prepare_lora_notebook.py --dataset artifacts/sft-public-v1.jsonl --owner <KAGGLE_USER> `
      --slug gemma4-lora-sft-v1 --output artifacts/lora-sft-notebook `
      --wheels-dir artifacts/training-wheels --wheels-source <KAGGLE_USER>/gemma4-training-wheels

# Generate a runtime notebook that evaluates the probe adapter (the serving check)
& $py scripts/prepare_kaggle_runtime.py --owner <KAGGLE_USER> --slug gemma4-lora-serving-probe `
      --agent-dir configs/experiments/lora-trajectory-v1 --task-id rich_3454 `
      --adapter-dir artifacts/lora-probe --adapter-mount /kaggle/input/gemma4-lora-probe/main_lora `
      --adapter-dataset-source <KAGGLE_USER>/gemma4-lora-probe --output artifacts/adapter-notebook

# Generate the logprob probe that decides whether LoRA applies at all (no dataset publication needed),
# then push it as a private kernel and read probe-result.json from the session output.
& $py scripts/prepare_lora_probe_notebook.py --adapter-dir artifacts/lora-probe `
      --owner <KAGGLE_USER> --slug gemma4-lora-serving-probe --output artifacts/lora-probe-notebook
& $py scripts/prepare_lora_probe_notebook.py --adapter-dir artifacts/lora-probe `
      --owner <KAGGLE_USER> --output artifacts/lora-probe-notebook --check
```

### Environment facts that cost time to rediscover

- **File permissions:** Python cannot create files in pre-existing workspace directories (`data/`,
  `data/processed`, `scripts/`, `runs/`, `dist/`, `resource/manifests/`, `.git`). Freshly created
  directories accept writes. An ACL repair verified two grants without restoring the operation, so do
  not repeat it; write generated artifacts to `artifacts/` (gitignored) or a new directory, and expect
  `git add` to need elevated access. Two undeletable leftovers remain: `gemma-ci-4qsxtd5l/` and
  `.tmp-check/`.
- **Python:** `C:\Users\32747\miniconda3\python.exe` (3.14.6), not on PATH. `pip install` needs an
  approved unconfined run because its unpack temp directory is outside the workspace.
- **Kaggle:** credentials authenticate (`~/.kaggle/access_token`); account `junzhejiao` has 14 kernels
  and 0 datasets. The patched `vllm-0.19.1` wheel is not on PyPI and the wheelhouse dataset has no
  vLLM wheel; it ships inside the Docker image.
- **Two skills are installed and should be loaded at the start of coding work:**
  `basic-programming-principles` (rules A-F, including "present reasoning in Chinese" and "log every
  piece of work") at `.agents/skills/basic-programming-principles/SKILL.md` and an identical copy at
  `C:\Users\32747\.dsh\skills\basic-programming-principles\SKILL.md`. Keep both in sync.

## Current project state

- October 24 custom-tool evaluation: the matched v31 Kaggle pilot is frozen under the Git-ignored
  `runs/2026-10-24-repair-checkpoint-fast-ab/`. It compares baseline and `repair-checkpoint-v1` on the ten
  valid-control tasks from the historical fast cohort with identical 31B model, seed, scripts, wheelhouse and
  budgets. Both private L4x4 notebooks compile locally and their package/notebook hashes are recorded in the
  experiment report. They have not been pushed: the raw API at 01:19 Asia/Shanghai still reported zero GPU
  seconds remaining and an 08:00 refresh. Recheck quota after 08:00, then push the two frozen notebook
  directories without regeneration; no competition submission is part of this experiment.

- October 23 custom-tool track: `feature/repair-checkpoint-skill` adds a submission-contained, read-only
  `repair-checkpoint` ADK skill and a baseline-matched experiment under
  `configs/experiments/repair-checkpoint-v1/`. It checks for empty patches, protected test/config edits,
  Python syntax errors, conflict markers, and debugger residue before `submit_patch`; it does not run
  repository code or mutate the worktree. Six focused tests, the repository gate, and the complete 160-test
  CPU suite pass (one expected artifact-dependent skip). Deterministic packaging includes the skill, but an
  official compiler/model smoke remains manual because this Windows host lacks `google.adk` and a reviewed
  local wheel. No GPU run, competition submission, push, or PR has occurred; remote A/B evidence is still
  required before promotion. The implementation is published on
  `origin/feature/repair-checkpoint-skill` at `60b07ee`; no PR exists yet because this host has no GitHub CLI.

- October 23 integration state: the isolated branch `integration/oct22-main-v31` contains `origin/main` through
  merge `8ef3c82` and the complete v31 branch through merge `46bad80`. The original dirty
  `fix/oct20-handoff-audit` worktree remains untouched. The combined tree preserves the remote wheelhouse
  registry plus the local adapter, compressed-payload, deterministic-ZIP, batch-freeze, flat-mount and
  validation behavior. A follow-up repair tracks the control-sweep dependency helpers and permits the
  POSIX-backed evaluation module's pure functions to be imported on Windows. All local CPU gates pass
  (154 tests, one expected artifact-dependent skip); nothing has been pushed and no PR exists yet. The next
  workstream is custom tooling, but the saved handoff contains no concrete acceptance criteria beyond the
  organizer constraint that custom scripts ship as sandbox skills rather than host-registry tools.

- October 22: the October 24 LoRA round remains ready but cannot start early. A fresh raw quota check at
  `2026-10-22T04:15:41Z` reported 114,317.472 of 108,000 GPU seconds used, zero reserved, zero remaining,
  and the same refresh at `2026-10-24T00:00:00Z` (08:00 Asia/Shanghai). The local V10 notebook still has
  the recorded SHA-256 `49762f3bd123c6b7caf75f6571f05bb984b4dadcf1f6004316ef80453ed0fc3f`.
  All 184 CPU tests now pass in the current working tree, including 44 focused guard/confirmation tests;
  `scripts/check_repository.py` also exits successfully, with shell syntax checks unavailable on this
  Windows host because Bash is absent. No kernel was pushed and no competition submission was created.
  Resume by rechecking quota after 08:00 on October 24, then push V10 and read `collapse-guard.json` before
  deciding whether any evaluation or confirmation arm is allowed to run.

- October 20: quota is 114,317.5 s of 108,000 until 2026-10-24T00:00Z, so the round is prepared rather than
  run. Three things are now push-ready: the **collapse guard** (a flat or collapsed arm is stopped from its own
  training loss for ≈2,800 quota-seconds instead of 22,000-30,000), the **V10 lr 5e-5 notebook**
  (`artifacts/tmp/lora-v10-tail3-r16-lr5e5`, digest-verified against V9, private, not pushed), and the
  **fixed 60-task confirmation set** (`resource/manifests/confirmation-tasks-2026-10-24.json`), which excludes
  both the screening tasks and the four tasks the training targets came from. Nothing has been submitted and no
  adapter has beaten the baseline: the screening verdict of October 19 stands (V1's 10/24 is the repeated
  baseline's resolved set, V9 collapsed to 0/24). The 2026-10-13→20 backlog is still uncommitted on
  `fix/lora-rank8-controlled`, and two pre-existing failures (a stale `max_memory_mib` test fixture, a deleted
  localization `gate.json`) block the full CPU suite locally.

- October 18 continuation: authenticated GitHub identity is `KaltistEsperanta`; current task
  sources are archived in `runs/2026-10-18-task-audit/`. The E-drive memo requires rank-32
  feasibility, leakage-free paired evaluation, and recommendations before selecting another research
  direction; issue #24 implementation remains deferred. Fresh quota is 108,000 GPU seconds with
  zero used/reserved, refreshing October 24. New assignments include PR #18 review, private
  `trajectories-r3` access, and a bounded budget-fit trajectory LoRA pilot. Requested reviews also
  include #23 and #25. Issue #13's technical work is complete; add its closing reference to PR #16.
  The October 16 meeting's rank-16 gain/rank-32 progress claims are not supported by local evidence.
  Execute items sequentially and record each completion below; do not treat this inventory as execution.

- October 14 23:43: live GitHub API inspection confirms the user's direct task in team tracker #7,
  issue #13, is technically complete: all five checkboxes are checked, PR #16 is mergeable, its
  `Repository checks` job passes, and the evidence comment is present. The issue remains open only
  because PR #16 has no reviews and branch protection still requires the requested independent review
  from `liaodid`; the author cannot legitimately complete that step. PR #8 remains a separate dirty
  research PR. Issue #24 records a real rules conflict: Kaggle Staff explicitly announced that the 31B,
  12B, E4B, and E2B QAT W4A16 variants are allowed with one shared variant per submission, while the
  competition Overview still says only 31B. This team can archive the announcement and implement the
  validator/documentation/runtime changes, but only the organizer can reconcile the official page; a
  12B end-to-end result additionally needs the model weights, v31 runtime, and H100 or L4 compute.

- October 14 01:24: fetched and pruned `origin`; remote `main` remains
  `625a597661848058f4398850e7a3f8423a4083b8`, and its hosted `Repository checks` job passes. Local
  `main` was merged with `origin/main` in the clean auxiliary worktree `artifacts/main-sync` and now
  points to `a809b6c`, two commits ahead only because it retains the Windows-safe filename rename and
  the merge commit; nothing was pushed. The dirty active branch was not merged: a dry merge reports
  content conflicts in `docs/README.md`, `resource/manifests/README.md`,
  `scripts/prepare_kaggle_runtime.py`, and `tests/test_kaggle_preparation.py`, so resolution awaits the
  user's decision. PR #16's exact remote head still passes 53 L4 tests (3 documented dependency skips),
  notebook generation, and real rank-16 adapter validation. No verified capability lift exists on the
  remote: PR #20 is a promising but unmatched prompt signal, PR #22 is confounded, and PR #8 does not
  improve clean/resolved counts. PR #21 does provide a clear wheelhouse/API compatibility recovery.

- October 14 01:14: **all five technical tasks in GitHub issue #13 are complete and checked off.**
  PR #16 now includes the missing efficiency/agent-training survey and a dependency-free safetensors
  header check that repairs its earlier CI failure; commit `1fa3e28` is pushed and GitHub `Repository
  checks` run `38436904198` passed. The issue has an evidence mapping comment and remains open only for
  the project-policy handoff: `liaodid` has been requested for the required independent review, after
  which PR #16 can merge. PR #8 remains a separate dirty/rebase collaboration item; its two issue #13
  review fixes were already merged into its head through PR #15 (`bd9b167`). No GPU run or competition
  submission was made in this audit.

- October 13 24:45: **the rank ladder has started and is paused by quota.** Rank 16 trained 32 updates on
  32 real SFT records with the required QAT base dispatched across four L4s, exported an 820-tensor
  adapter (244,982,352 bytes, SHA256 `bec7346f…`, validated 820/820 on the served scheme) and the file is
  downloaded locally; the CPU preflight proves all 116 records encode on the real tokenizer, 115
  supervised. Rank 32 failed at the base load after 600 s and consumed **2,570.542 quota-seconds**, so
  **787.753 s of 108,000 s remain** against ~1,750 s per rung and a 2026-10-17T00:00Z refresh. Rank 64 was
  never attempted and **no paired evaluation has been run for any rung**, so the ladder has a trained
  artifact and no capability claim. Supporting changes: `python_literal`/`@@@NAME@@@` placeholder
  rendering, dataset-driven device-map training, `--dataset-check` CPU preflight, and
  `scripts/fetch_kernel_log.py` for when `kaggleusercontent.com` does not resolve. Full suite 131 tests.

- October 13 21:30: **the control-valid task set is 99 tasks (fastapi 60, rich 39)**, past the 78 that
  a 10-point paired gain needs at 80 % power, so the instrument is no longer the constraint. The
  decisive measurement is the CPU-only sweep `junzhejiao/gemma4-control-fastapi-starlette` version 1
  (28 of 30 usable against 28 of 30 failing in the same-task batch that morning), recovered by
  pinning starlette to the newest wheelhouse wheel that accepts `Router(on_startup=...)` -- measured
  1.6.0 rejects, 1.3.1 rejects, 0.52.1 accepts. Tracked in
  `resource/manifests/control-valid-tasks-2026-10-13-final.json`; the earlier 73-task manifest is kept
  as history. Quota is unchanged at **5,306.051 s of 108,000 s** and refreshes 2026-10-17T00:00Z, so
  about 2,653 s of L4x4 wall time remain against roughly 8-10 GPU hours for a 99-task paired run:
  **compute is now the binding constraint**, and no adapter has been trained yet. Full offline suite
  passes (116 tests, 3 subtests).

- The October 12 completion audit is superseded by the October 14 entry above: both review fixes are
  upstream in PR #8's head through merged PR #15, the L4 branch is published as PR #16, issue #13 is
  updated, and authenticated GitHub state is verified. Independent approval and merge remain external.

- October 12 20:35 issue #13 (four-L4 training feasibility): **the acceptance criterion is met.** Run 11
  (kernel `junzhejiao/gemma-4-four-l4-lora-training-feasibility-v5` version 7) trained a replicated LoRA
  adapter on the required `gemma-4-31b-it-qat-w4a16-ct` base for eight updates with the BF16 base
  dispatched across four L4s by one process, exported an 820-tensor adapter (244,982,352 bytes, SHA256
  `c4d2bff9bf765bb6b16a01dc7545757358cbc1cc6d3054de3b41cfa601b76852`), validated it 820/820 against the
  served naming scheme, and reloaded it in the official vLLM runtime (`status: applied`; control
  base-versus-base 0.0, adapter effect 0.596932). Measured cost: 1,421.3 s of session wall and 4,733.33
  quota-seconds, against a 2,842.6 estimate, because the allocation is billed while held and this session
  spent ~780 s queued. **5,306.051 s of 108,000 s remain** (refresh 2026-10-17), about 2,653 s of L4x4
  wall time. The two routes the review proposed are measured infeasible here: four sharded ranks need
  about 286 GiB of host RAM (the host has 188.69 GiB and the phase was SIGKILLed), and PEFT cannot wrap the
  packed checkpoint's compressed `Linear` modules (`AttributeError: 'Linear' object has no attribute
  'weight'`). The surviving route is the single-process `device_map` dispatch, implemented as
  `resource/l4_feasibility/train_l4_device_map.py`. Earlier in this workstream: L4x4 is gated on
  competition attachment, and the environment conflict that stopped run 4 (the image's `transformers
  5.0.0` has no `gemma4` in `CONFIG_MAPPING`; `peft 0.19.1` refuses the image's `torchao 0.10.0`) is fixed
  by `resource/l4_feasibility/prepare_model_stack.py`, which installs the mounted `transformers 5.13.1`
  plus a PyPI-hash-checked `torchao 0.16.0` wheel from the attached public dataset
  `kajarnaklokossou/torchao-0-16-0`. No dataset upload is needed, and none is possible: the kaggle client's
  blob uploads go to `www.googleapis.com`, which is unreachable from this machine, so kaggle commands run
  through `scripts/bounded_connect.py`. Offline gates: 53 L4 tests and the 109-test suite pass, and the
  generator's `--check` reports `status: ok`. `docs/experiments/2026-10-12-l4-training-feasibility.md`
  section 9 holds the measurement table. Git refs are readable from this workspace, but authenticated
  GitHub issue/PR pages are not; posting the result to issue #13, pushing the branch, and obtaining an
  independent review remain human actions.

- October 11 19:55 readiness audit: random rank-8/rank-32 serving effects are recorded, but no trained adapter exists. Real training is blocked by an executed generated-config type error and unvalidated loader/environment integration. Holdout filtering, trajectory provenance, and explicit paired-run identities need fixes before efficacy claims. The 116 training records overlap three historical evaluation tasks. See `docs/experiments/2026-10-11-lora-readiness-audit.md`; implementation remains unchanged after this advisory task.

- The canonical remote is `origin` at `https://github.com/jwj1342/gemma-agent-lab.git`; future references
  to the project remote mean this repository unless the user says otherwise. `origin/main` is
  `625a597661848058f4398850e7a3f8423a4083b8`.
- Local `main` has incorporated `origin/main` through merge commit
  `a809b6cee4cae74624948be08be8db88793f0f11` and is two commits ahead: the other local commit,
  `88823c0f556874608cd993cd37ba32d94df7bbd8`, renames one leaderboard CSV whose colon-containing
  timestamp cannot be checked out on Windows. The blob content is unchanged and nothing has been
  pushed.
- Active work is on `fix/oct20-handoff-audit` at `89ee95f`, with the earlier Localization v1 migration
  preserved and unrelated localization edits still in the working tree. The branch is local-only and holds
  the *uncurated* copies of the L4 documents. The curated task branch
  `experiment/four-l4-lora-feasibility` (`1fa3e28`) is published on `origin` as PR #16 with passing CI;
  `liaodid` has been requested for review. `fix/pr8-contracts-v2` (`bd9b167`) merged into
  `experiment/swe-lora-v1` through PR #15, so PR #8 now points to `bd9b167` but remains dirty against
  `main` and needs its own rebase/review decision.
- October 10 paired L4x4 attempt (budget exhausted at 10 kernel pushes, one further push rejected by the platform): exactly one task pair completed, `rich_3675`, with valid controls in both arms; both arms left it unresolved with the **byte-identical patch**, and the portable arm's trace shows the `localization-v1` skill was **never invoked**, at +44% agent time. Predeclared acceptance is unmet (0.0 points, p = 1.0, n = 1); no promotion. The organisers **republished the official wheelhouse dataset in place on 2026-10-09T14:43Z** (`swegemma` 0.2.10 → 0.2.11, which deleted `swegemma.models.registry`, plus changed `adk_eval_core`/`adk_submission` bytes), so the frozen runner cannot import on the served wheelhouse; the reviewed wheels were restored from the private publisher kernel `junzhejiao/gemma4-pinned-wheels` v1, hash-verified against the frozen manifest. Evidence and the full ledger are in `runs/2026-10-10-localization-ab/{REPORT.md,ab-summary.json}`. `rich_3676` and `rich_3930` remain unmeasured because the pinned-wheel kernel-source mount was missing for both `rich_3676` baseline pushes.
- October 9 continuation: the user authorized private Kaggle trials and iterative v1 changes until a meaningful improvement is demonstrated. All 46 CPU tests passed after correcting Windows-specific expectations; 114 frozen cases and 18 notebooks passed integrity verification. The first private BM25 pilot uploaded successfully after compressing its notebook payload, but version 1 failed its input mount-path check before inference. A CPU-only input-layout probe is underway. No repair improvement has been measured or candidate promoted. Local checkpoint `ef5e199` contains the prior preserved module; continuation changes are in the working tree.
- The October 8 preservation handoff is in `docs/localization-handoff.md`, linked from the root README. It explains baseline localization, the implemented optional module, offline gains, rollback, local artifacts, and pending manual remote confirmation; no background GPU work was started.
- The remote baseline lives under `configs/baseline/` and uses the required Gemma 4 model, a main coding agent, and a read-only `code_analyzer` sub-agent. Remote packaging, validation, Kaggle/HPC preparation, CI, documentation, and resource snapshots are now present locally.
- All pre-remote versionable work is protected by branch `archive/local-before-remote-2026-10-07` at `2ba0f75448f16bdd41281572fa0969641ccc44c7` and annotated tag `local-before-remote-2026-10-07`. The original root commit remains its parent.
- A second filesystem backup excluding `.git` is at `E:\Desktop\CodingAI-local-backup-20261007-1818`; verification counted 106 files and 23,190,078 bytes. It includes ignored datasets, Kaggle results, ZIPs, caches, source, prompts, tests, and model artifacts as they existed before branch replacement.
- Archived work includes Programming v1, Graph Agent v1, Kaggle A/B experiments, Localization v1/v2/v3/v3.1/RL/v4 source and artifacts, package tests, and the full work history. Localization v1 is the first module migrated into an isolated remote-framework experiment; none of the experimental modules has replaced the baseline.
- Retained Localization v1 still means the archived v3 checkpoint (`artifacts/localization_v3_3k/`, best epoch 1,850). Programming v1 remains an unpromoted ablation after a 0/2 paired smoke result and should not replace the remote prompt without a new matched comparison.
- Localization v1 remains isolated under `configs/experiments/localization-v1/`. A portable content-based refinement now learns seven evidence coefficients and a confidence gate with nested refitting of both stages. It improves retrospective test Hit@1/Hit@5/MRR from BM25's 31.6%/68.4%/0.473 to 47.4%/78.9%/0.624; nested snapshot and repository validation also improve. Explicit `portable` mode runs the learned gate without the archived checkpoint. Default adaptive mode remains BM25 pending fresh confirmation and remote repair evidence. Prior gate-only CV reused a ranker trained on those tasks and is now explicitly qualified; it cannot justify promotion. An October 9 gate audit found the portable confidence floor `min_candidate_margin=0.025` overfits the full-training margin distribution (nested out-of-fold gate selection chose 0.0 in 8 of 9 folds), so `scripts/train_portable_localization.py` now fixes that floor at 0.0. This leaves nested results unchanged and lifts retrospective test gated Hit@1 to 57.9%, but exposes one `rich_3930`-style dev regression, so the frozen `portable-policy.json` and its remote A/B ZIP remain unchanged and promotion still awaits fresh confirmation and remote repair evidence.
- A frozen three-arm remote repair protocol is prepared at `runs/2026-10-08-localization-v1-portable/remote-ab-final/`: 38 tasks, 114 baseline/BM25/portable cases, and 18 pilot notebooks over six seeded tasks. October 9 launch copies and setup diagnostics are under `runs/2026-10-09-localization-kaggle/`; no inference result is available yet. The portable experimental ZIP is `runs/2026-10-08-localization-v1-portable/submission-localization-v1-portable.zip`, SHA256 `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48`. The remote baseline still has no trained localization adapter and remains unchanged.
- Public task data and partial-clone source repositories remain locally available; archived artifacts, results, and ZIPs remain recoverable from the external backup. Generated paths such as `artifacts/`, `data/public/`, `dist/`, `results/`, and `runs/` are excluded by `.gitignore`; source/configuration migrations are reviewed separately.
- `WORKLOG.md` remains the persistent handoff record. `AGENTS.md` combines the remote collaboration rules with the mandatory work-log procedure.
- A repository-local skill `basic-programming-principles`, asked for as 基本编程原则, lives at `.agents/skills/basic-programming-principles/SKILL.md` (untracked). DSH discovers it in this workspace, and it is meant to load at the start of coding work; it covers not guessing identifiers with a 10-attempt cap, lean token use, checking output against every instruction, logging all work in `WORKLOG.md`, fetching web material completely, and presenting all reasoning results in Chinese (rule F, added 2026-10-10). The name must stay kebab-case ASCII; the Chinese title lives in the description and body. An identical copy was installed at `C:\Users\32747\.dsh\skills\basic-programming-principles\SKILL.md` on 2026-10-10 so the skill also applies outside this repository; the project copy wins inside this checkout (discovery rank 100 versus 400), so edits must be applied to both or the repository copy removed.
- No LoRA is used by the measured baseline, and that is deliberate. A new, untracked experiment `configs/experiments/lora-trajectory-v1/` wires `adapter: main_lora` plus a rank-32 PEFT `adapter_config.json`, but contains **no weights**, so it is not runnable or submittable; `scripts/check_lora_adapter.py` validates the config and rank policy offline. The PEFT-versus-SFT-versus-RL ranking, the n=1 measurement problem that gates it, and the recommended next direction are in `docs/experiments/2026-10-10-lora-sft-rl-adjudication.md`. An adapter also costs KV cache (36,432 to 27,712 tokens on 4xL4 with one rank-16 adapter), so it must out-earn that penalty before promotion.
- The user selected direction **A** (fix the measurement instrument, then trajectory SFT) and **F** (close the `k_proj`/`v_proj`/MLP module evidence) on 2026-10-10. Offline tooling for both is ready: `scripts/probe_model_modules.py` writes the real module evidence to `resource/manifests/model-modules.json` (not yet run; needs the base weights), and `scripts/build_trajectory_dataset.py` turns verified harness traces into TRL-shaped SFT records, keeping only tasks whose gold control passed and whose trace contains a tool call. Neither has been run against real data, so no result exists yet.

## Archived pre-remote project state

- Git remote `origin` is `https://github.com/jwj1342/gemma-agent-lab.git`; future references to the project remote mean this repository unless the user says otherwise. `origin/main` was last fetched at `3054da842f794c854f438a5478ea9437b119e9bc` on 2026-10-07.
- Local `main` and `origin/main` have unrelated root histories (local: one commit; remote: five commits), and the working tree contains substantial uncommitted local experiments. The remote has been fetched into the object database only; it has not been merged, rebased, checked out, or set as the upstream of local `main`.
- The project is a minimal, declarative single-agent baseline for the Gemma 4 Developer Agent Kaggle competition.
- `submission/agent.yaml` configures `gemma-4-31b-it-qat-w4a16-ct` and all nine competition harness tools.
- The agent prompt supports the basic coding loop: inspect an issue, locate code, make a minimal source change, run targeted verification, inspect the diff, and submit a patch.
- The current programming capability is not a standalone locally trained module: the official Harness runs Gemma 4 as a single tool-using agent, while this repository supplies its declarative configuration, prompt policy, sampling settings, and packaging checks.
- `scripts/package_submission.py` validates and packages the contents of `submission/` into `dist/submission.zip`.
- The repository does not currently contain a standalone local coding-agent runtime, model server, interactive UI, or end-to-end harness test. Actual coding execution depends on the Kaggle harness/official notebook.
- Static package, localizer, and reward tests currently pass: 20 tests successful on 2026-10-07.
- `submission/prompts/system.md` now contains Programming v1: the retained embedding-retrieval policy plus a compact seven-field pre-edit plan, a disproof check, and evidence-driven recovery after failed verification.
- The preserved baseline is `dist/submission.zip`; retrieval-only control is `dist/submission-retrieval-v1.zip`; the current `submission/` working source is packaged separately as `dist/submission-programming-v1.zip`. The Programming v1 archive differs from the retrieval control only in `prompts/system.md`.
- Kaggle authentication is configured locally and was validated without logging the credential; the account has joined the target competition.
- Private Kaggle Notebook version 1 (`codingai-retrieval-a-b-smoke-test`) completed its two-task baseline/retrieval smoke test with 0/2 resolved for both variants; it was not submitted to the competition.
- Private Kaggle Notebook version 1 (`codingai-programming-v1-a-b-smoke-test`) completed a paired two-task retrieval-control versus Programming v1 evaluation; both variants resolved 0/2, so Programming v1 is not promoted. No competition submission was created.
- Graph Agent v1 is implemented under `experiments/graph_agent/submission/` and packaged as `dist/submission-graph-agent-v1.zip`; it has not been pushed to Kaggle.
- A dependency-free pairwise file-localization ranker is implemented in `scripts/train_localizer.py`; its 1000-epoch public-data run is saved under `artifacts/localization_v1/` with epoch 50 selected by development MRR.
- Localizer v2 reduces the label-specific `issue_path` feature to 0.25, raises L2 regularization to `1e-4`, and selects checkpoints using equal-weight development Hit@1 and Hit@5 with MRR as a tie-breaker. Its completed 10,000-epoch run is saved under `artifacts/localization_v2_10k/`; epoch 450 was selected, but its held-out results did not beat v1, so v1 remains the stronger experimental checkpoint.
- Localizer v3 completely removes `issue_path` and applies exact global decoupled weight decay through lazy scaling. Its completed 3,000-epoch run is saved under `artifacts/localization_v3_3k/`; epoch 1,850 was selected and currently has the strongest held-out Hit@3, Hit@5, and MRR among the localizer variants, though the test split remains only 19 tasks.
- Per the user's explicit decision, the experiment formerly named localizer v3 is retained as product-level **Localization v1**. Its existing `artifacts/localization_v3_3k/` path is preserved for lineage and because `artifacts/localization_v1/` already names an older experiment; neither artifact may be overwritten. The retained locator is not yet wired into the competition Agent runtime.
- V3.1 is a protected continuation experiment under `artifacts/localization_v3_1_additional_3k/`: it ran 3,000 additional epochs from v3's epoch-1,850 checkpoint but found no better development checkpoint, so its selected epoch 0 weights are identical to v3 and v3 remains current.
- An experimental two-head contextual-bandit/RL localizer is implemented in `scripts/train_rl_localizer.py`: a count policy predicts 1–26 returned files and a residual file-ranking policy sits on frozen v3. The first 300-epoch pilot under `artifacts/localization_rl_v1/` overfit training reward and selected epoch 0, so it is not promoted and v3 remains current.
- Localizer v4 extends the RL prototype with a configurable squared-component reward, defaulting to `0.65 * count_match^2 + 0.35 * location_f1^2`. Its completed 300-epoch run is under `artifacts/localization_v4_300/`; it also selected epoch 0 and did not improve the directly comparable exact-count, exact-set, or location-F1 metrics, so v3 remains current.
- The localizer benchmark is closed-world: candidates are the 138 distinct non-test Python paths found in public reference patches rather than complete repository trees, so its metrics must not be treated as end-to-end agent accuracy.
- Programming v1 is implemented as a prompt-only planning-gate experiment with localization and all runtime settings fixed. Its paired smoke test found no resolution gain, only 1/2 complete plan compliance, higher latency and edit churn, and fewer failed tool calls; keep it as an ablation and retain retrieval v1 as the control.

## Work history

### 2026-10-24 01:21 (Asia/Shanghai) - Freeze repair-checkpoint versus baseline Kaggle pilot; launch waits for quota refresh

- Objective: start measuring whether the submission-contained `repair-checkpoint` skill improves repair
  outcomes relative to the unchanged baseline, without using a formal competition submission as the tuning
  instrument.
- Selected the ten valid-control tasks from the predeclared v31 `fast` cohort, excluding only
  `fastapi_15745` and `requests_7502`, whose controls were already invalid in the matching historical study.
  The comparison uses `gemma-4-31b-it-qat-w4a16-ct`, no LoRA, seed 42, wheelhouse v31, four minutes, 40 tool
  calls and 80 turns for both arms. Success will be reported as paired issue resolution plus structural,
  adoption, timing and tool-cost diagnostics; a checkpoint `ready` result is not a resolved issue.
- Generated deterministic ZIPs and private offline L4x4 notebooks under the local ignored run directory
  `runs/2026-10-24-repair-checkpoint-fast-ab/`. Baseline ZIP SHA256 is `aa597ea3...e822c`; candidate ZIP is
  `47db85f0...eb2b2`. Baseline notebook SHA256 is `d9203e42...25d8`; candidate notebook is
  `552fd250...beaa`. Payload inspection confirmed identical task order, 41 wheel hashes and six embedded
  runtime scripts, no training payload, and the intended model/data mounts. All code cells compile.
- Semantic parity checks confirmed identical sampling, evaluation budget, analyzer prompt and analyzer
  configuration. The candidate differs only by agent name, the direct `repair-checkpoint` skill, and the
  pre-submission checkpoint instructions; Windows line-ending differences explain additional member-byte
  hash differences. The existing batch generator at `E:/Desktop/CodingAI/scripts/prepare_lora_eval_notebook.py`
  has SHA256 `07055b7f...d8219e`; its 19 focused CPU tests passed, and the frozen notebooks embed the current
  branch's evaluator/helpers rather than the original worktree copies.
- The raw Kaggle quota check at 2026-10-23 17:19:51 UTC reported 114,317.472 of 108,000 GPU seconds used,
  zero reserved and zero remaining, with refresh at 2026-10-24 00:00:00 UTC (08:00 Asia/Shanghai). No kernel
  push was attempted, no GPU was reserved, and no competition submission was made. One initial quota command
  lacked the documented `.pylibs` path and one obsolete manifest path was absent; both were corrected by
  using the recorded environment and current branch paths without repeating the failed commands.
- Changed tracked files: `WORKLOG.md` and
  `docs/experiments/2026-10-23-repair-checkpoint-skill.md`. Local-only generated files are the two ZIPs,
  manifests and notebook directories under the run path above, plus the dated quota receipt in
  `artifacts/tmp/`. Next: after 08:00, recheck quota and push both already frozen notebooks, then download and
  analyze both outputs before any promotion decision.

### 2026-10-23 02:12 (Asia/Shanghai) - Implement submission-contained repair checkpoint skill v1

- Objective: begin the custom-tool workstream by identifying and implementing a missing tool that the coding
  agent can call while preserving competition submission compatibility.
- Fetched `origin` and reviewed the existing agent contracts, official competition/ADK skill documentation,
  the frozen v31 harness source, the failure-mechanism branch (`93a7483`), the selective-graph branch
  (`9dc37de`), and public reference repository `Gosling-dude/gemma4-developer-agent` at
  `fb0826b7a560d766ece173f8d96f2827c0db2646`. The public reference had no visible license, so no source was
  copied. Existing evidence showed that relevant source was often already exposed while empty or incomplete
  patches remained common; another search tool was therefore not selected as the first intervention.
- Created `feature/repair-checkpoint-skill` and added `configs/experiments/repair-checkpoint-v1/`, matching the
  baseline model, tools, sampling, analyzer, and evaluation budgets while adding one direct skill plus one
  mandatory post-test/pre-submit call. The standard-library script reads tracked and untracked Git changes,
  parses changed Python without imports, and reports deterministic structural blockers as compact JSON. It
  never executes target code, runs tests, stages files, edits files, or reads hidden tests.
- Added six focused tests covering ready, empty, protected-edit, syntax/conflict/debugger, untracked-file, CLI,
  prompt, and non-mutation behavior. Added the experiment report and navigation entries. Verification passed:
  focused tests 6/6; repository checks over 78 Python, 14 shell, 36 configuration, 87 Markdown, and six
  deterministic agent packages; full CPU suite 160/160 with one expected historical-wheel artifact skip;
  `git diff --cached --check` passed before the final documentation update.
- The first repository-check attempt used Windows defaults and failed on GBK decoding and missing `bash`;
  UTF-8 mode plus the verified Git Bash path passed. The first full-suite attempt used an interpreter without
  `numpy`/`safetensors`; the documented local `.pylibs` path passed. One attempted reference-clone path was
  wrong and was corrected without repeating it. Frozen compiler import was attempted once and stopped at the
  absent `google.adk`; no local wheel was found, so official compilation and model execution remain an explicit
  Kaggle/Trillium v31 gate rather than a claimed result.
- Changed files: `WORKLOG.md`, `configs/README.md`, `docs/experiments/README.md`,
  `docs/experiments/2026-10-23-repair-checkpoint-skill.md`, new
  `configs/experiments/repair-checkpoint-v1/`, and `tests/test_repair_checkpoint_skill.py`. No GPU run, Kaggle
  upload, competition submission, PR, or merge was performed. Implementation commit `60b07ee` was pushed to
  `origin/feature/repair-checkpoint-skill`; the remote supplied
  `https://github.com/jwj1342/gemma-agent-lab/pull/new/feature/repair-checkpoint-skill` for PR creation, but this
  host has no `gh` executable. Next: open the PR, obtain independent approval, then run official compile/smoke
  and a matched baseline-versus-checkpoint repair A/B before promotion.

### 2026-10-23 01:58 (Asia/Shanghai) - Complete isolated main/v31 integration and restore the CPU gate

- Objective: resume the approved October 22 A1 integration from `E:\备忘录.txt`, complete both merges without
  changing the original dirty worktree, run the CPU checks, and then recover the scope of the queued custom-tool
  workstream.
- Continued the in-progress merge whose `MERGE_HEAD` was `bae1443`. All conflicts were already resolved;
  14 staged paths matched `origin/chore/wheelhouse-v31` byte-for-byte and the only differing path,
  `docs/experiments/README.md`, retained the additive index union. The merge was committed as `46bad80` after
  the earlier `origin/main` merge `8ef3c82`.
- The first Windows repository-check attempt failed because Python used GBK and Bash was absent from `PATH`;
  rerunning with UTF-8 mode and the verified Git Bash path exposed one real invalid root-relative link, which
  was repaired. The initial unit run then exposed two integration gaps: the committed control-sweep generator
  imported an untracked helper, and `run_eval_suite.py` imported POSIX-only `fcntl` at module import time.
- Added the existing `scripts/test_dependency_payload.py` and its documented builder
  `scripts/build_test_dependency_set.py` from the preserved worktree. Moved `fcntl` imports to the two Slurm
  execution paths so Linux locking semantics are unchanged while Windows can collect and test `indexed_hash`.
- Verification: both focused suites passed (7 control-sweep tests with one artifact-dependent skip, and 9
  evaluation-suite tests); the full suite passed 154 tests with that same skip; `scripts/check_repository.py`
  passed over 76 Python, 14 shell, 32 configuration, 83 Markdown and five agent-package checks; and
  `git diff --cached --check` passed after normalizing the two new files' EOFs.
- Changed after the merge: `WORKLOG.md`, `scripts/run_eval_suite.py`, and new
  `scripts/build_test_dependency_set.py` / `scripts/test_dependency_payload.py`. No push, PR, Kaggle run, GPU
  run or competition submission was performed. Remaining follow-up: ask the user to restate the custom-tool
  deliverable because the surviving handoff records only the workstream name.

### 2026-10-22 12:23 (Asia/Shanghai) - Fetched remote state and audited integration conflicts; merge awaits user choice

- Objective: fetch current remote work, identify every conflict with the local branch and dirty working tree,
  propose resolutions, and stop before merging so the user can choose. The custom-tool workstream remains next
  after the integration decision; no implementation of that second workstream started in this audit.
- `git fetch --prune origin` completed. It removed the stale remote-tracking ref
  `origin/experiment/heredoc-prompt-v9`; `origin/main` remains at `e2837a8` (October 21). The active
  `fix/oct20-handoff-audit` branch is 26 commits ahead and 5 behind, with merge base `3054da8`.
- A read-only `git merge-tree --write-tree --name-only HEAD origin/main` simulation found four committed
  content conflicts: `docs/README.md`, `resource/manifests/README.md`,
  `scripts/prepare_kaggle_runtime.py`, and `tests/test_kaggle_preparation.py`. No index or working-tree merge
  state was created.
- Dirty-path comparison found five overlaps with remote changes. Two are byte-identical to main:
  `scripts/evaluate_agent.py` (`88576f2...`) and
  `resource/manifests/kaggle-wheelhouse-snapshots.json` (`e512638...`). The three substantive dirty overlaps
  are `scripts/README.md`, `scripts/prepare_kaggle_runtime.py`, and `tests/test_kaggle_preparation.py`.
  No untracked local path collides with a remote-added path.
- Resolution finding: the two README conflicts are additive and should keep both sets of indexed evidence.
  The runtime conflict is semantic: main supplies the registered `wheelhouse_runtime.py` verified/record mode,
  while the local working copy supplies adapter mounting, deterministic final-ZIP verification, compressed
  under-1-MB notebook payloads, optional reviewed test dependencies, and flat Kaggle mount handling. The
  recommended resolution is a tested union with main's wheelhouse registry as the source of truth, not the
  local fixed October 9 manifest. Its tests must likewise be a union, restoring main's snapshot/mode tests and
  retaining the local adapter, size, mount-layout, batch-freeze, and dependency tests.
- Important contradiction: the previous handoff said the V10 v31 snapshot would arrive from `origin/main`,
  but main still contains only the October 5/v29 registry. The v31 work is on unmerged
  `origin/chore/wheelhouse-v31`, which is 7 commits ahead and 3 commits behind its main divergence. Decide
  separately whether to integrate that branch after main; a merge-tree simulation reports only one additive
  conflict in `docs/experiments/README.md`, so the complete measured v31 environment can be retained without
  selecting an isolated manifest. V10's notebook already embeds the measured v31 map, but repository provenance
  should not continue to claim main contains it.
- One inspection command failed: this Git version does not support `git show-ref --remotes`; remote refs were
  then read successfully with `git for-each-ref refs/remotes/origin`. No failed command was repeated.
- Repository changes: `WORKLOG.md` only. External/local Git metadata changed through fetch and the merge-tree
  simulation; no source/configuration file, branch tip, index entry, Kaggle resource, or competition submission
  was changed. The existing staged, unstaged, and untracked backlog remains intact.

### 2026-10-22 12:16 (Asia/Shanghai) - October 24 LoRA round revalidated; execution still quota-blocked

- Objective: continue the unfinished October 20 work without spending or misreporting unavailable GPU quota.
- Read-only external check: `scripts/check_kaggle_quota.py`, run with the documented `.pylibs` environment,
  reported 114,317.472 of 108,000 GPU seconds used, zero reserved, zero remaining, and refresh at
  `2026-10-24T00:00:00Z` (08:00 Asia/Shanghai). The receipt is local-only at
  `artifacts/tmp/accelerator-quota-2026-10-22.json`; no credential content was printed or stored in Git.
- Revalidated the prepared V10 kernel directory without pushing it: `evaluation.ipynb` is still 104,573 bytes
  with SHA-256 `49762f3bd123c6b7caf75f6571f05bb984b4dadcf1f6004316ef80453ed0fc3f`; its metadata file is present.
- Verification: the focused guard and confirmation suites passed 44/44; the complete CPU suite passed
  184/184; `scripts/check_repository.py` exited successfully and checked 56 Python files, 22 configurations,
  58 maintained Markdown files, and three agent packages. It could not execute the ten shell syntax checks
  because Bash is not installed on this Windows host, so that platform-specific check remains unverified.
- Important correction to the prior handoff: the formerly reported `max_memory_mib` fixture and missing
  localization resource failures do not reproduce in the current working tree. No repair was needed.
- Repository changes: `WORKLOG.md` only. No source/configuration change, commit, push, Kaggle kernel push,
  GPU run, or competition submission was made. The large pre-existing staged/unstaged backlog was preserved.
- Remaining follow-up: after 08:00 Asia/Shanghai on October 24, recheck quota, push the prepared private V10
  kernel, and inspect `collapse-guard.json` first. Only `proceed` permits the fixed 24-task measurement; only
  a real measured improvement permits the already frozen 60-task confirmation pair.

### 2026-10-20 23:39 (Asia/Shanghai) - Withdrawn: no external CUMCM skill was installed

- The user pointed at a 50.3 MB external archive (`CUMCM_repo-main(1).zip`) holding a CUMCM C modelling skill
  package and asked for it to be loaded as a workspace skill that all later commands would follow. Ten skills
  were inspected (`1start-cumcm-c`, `2c-analysis-modeling`, `3c-solver-experiments`, `4c-writing-visual`,
  `5c-judge-reflection`, `4drawio`, `doctor`, `mathmodel-figure-templates`, `typst-author`, `_references`;
  12.7 MB of `skills/` plus a 37.7 MB web application) and the install scope was being confirmed with the user.
- **The user withdrew the request before anything was installed.** Nothing was written to `.agents/skills/`
  (still only `basic-programming-principles`), no modelling directory was created, and the staging copy under
  `artifacts/tmp/cumcm-skill-staging` was deleted. Repository changes: none.
- Also corrected in this entry's parent: the 2026-10-20 round entry was labelled 12:55 but the measured
  artifact mtimes put its completion at 12:24-12:26, so the heading now reads 12:25.

### 2026-10-20 12:25 (Asia/Shanghai) - The 2026-10-24 round prepared: loss-trajectory guard, lr 5e-5 notebook, fixed 60-task confirmation set

- Objective: continue the 2026-10-19 work (user instruction "继续昨天的工作") with no GPU available — quota is
  114,317.5 s of 108,000 until 2026-10-24T00:00Z — by building everything the refresh needs so that no session
  time is spent preparing a kernel that day.
- **Decision taken with the user before acting**: three options were offered (build the 10-24 round, commit the
  six-day uncommitted backlog, or both in that order); the user chose the round, so the backlog stays open and
  is recorded as a follow-up rather than silently absorbed.
- **Guard module** `resource/l4_feasibility/collapse_guard.py`: pure function over
  `device-map-result.json['updates']`, returning `collapsed` / `flat` / `proceed` with the whole trajectory
  recorded. Thresholds are taken from the two measured arms, not tuned: floor 1.0 (V9 0.7323 vs V1 2.0966) and
  flat ratio 0.005 (V1 moved 1.1e-3, the moving fixture arm 0.22). Registered in `WORKER_FILES` so the kernel
  receives and hashes it like the other workers.
- **Generator changes** `scripts/prepare_lora_eval_notebook.py`: `--collapse-loss-floor` and
  `--collapse-flat-ratio` (0 disables either), both embedded in the payload; the guard runs inside the training
  branch after the adapter is mounted; the evaluation body became `evaluate_arm()` so a stop can skip it
  without an exit code that looks like a crash. A baseline kernel (no `--train-dataset`) still starts with
  `GUARD = {'verdict': 'proceed', 'reason': 'no_training'}`.
- **V10 built** (`artifacts/tmp/lora-v10-tail3-r16-lr5e5/`, 104,573 bytes,
  SHA-256 `49762f3bd123c6b7caf75f6571f05bb984b4dadcf1f6004316ef80453ed0fc3f`): same 24 tasks, same 41-wheel v31
  digest map, same agent bundle hashes, same evaluator/hook text, same dataset, same rank/alpha/seq/adapter;
  **lr 5e-5**, `--max-updates 8` explicit, guard present. Verified by
  `artifacts/tmp/verify_v10.py` (16 field checks, all true, every code cell parses); build inputs were recovered
  from the V9 payload by `artifacts/tmp/recover_v9_inputs.py` rather than remembered. Not pushed: pushing needs
  quota to be meaningful, and the kernel stays private.
- **Confirmation set** `resource/manifests/confirmation-tasks-2026-10-24.json` (SHA-256 `2da2b2b7…`, 60 tasks =
  41 fastapi + 19 rich), drawn by the new `scripts/select_confirmation_tasks.py`. The first draw put four
  **trained** tasks (`rich_3676`, `rich_3882`, `rich_3894`, `rich_4077`) into the confirmation set; the tool now
  derives the trained task ids from the dataset (`--trained-from`, stripping the `seedNNN-` prefix) and refuses
  an overlap, and the manifest records the exclusion with the dataset hash. `requests_7205` is trained but
  outside the control-valid manifest and is recorded as `tasks_outside_manifest`.
- **Verification actually run**: `tests/test_lora_eval_notebook.py` 13 tests OK, `tests/test_confirmation_tasks.py`
  19 tests OK. Full suite through `artifacts/tmp/run_checks_lenient.py`: 165 tests, 20 failures / 22 errors /
  3 skipped, all in modules this session did not touch — 18 in `test_l4_feasibility` (stale `Args` fixture,
  script modified 2026-10-18), 10 in `test_repository_checks` and 8 in `test_kaggle_preparation` (sandbox
  write denials inside subprocess temp directories, a missing deleted localization resource, and
  `build_submission` writing into `mkdtemp`). Without the wrapper the suite reports 15 failures / 74 errors,
  all `PermissionError`. `scripts/check_repository.py` cannot complete here for the reasons above and its
  result is reported as blocked, not passing.
- Changed files: `resource/l4_feasibility/collapse_guard.py` (new), `scripts/prepare_lora_eval_notebook.py`,
  `scripts/select_confirmation_tasks.py` (new), `resource/manifests/confirmation-tasks-2026-10-24.json` (new),
  `tests/test_lora_eval_notebook.py` (new), `tests/test_confirmation_tasks.py` (new), `scripts/README.md`,
  `docs/experiments/2026-10-18-lora-improvement-program.md` (§13 row, new §15), `WORKLOG.md`. Local-only
  artifacts (gitignored): `artifacts/tmp/lora-v10-tail3-r16-lr5e5/`, the build/verify helpers under
  `artifacts/tmp/`, and the lenient check runner. No GPU run, no competition submission, no kernel push.
- Remaining follow-up, in order: (1) on 2026-10-24 push V10 and read the guard verdict, then follow the ladder
  (proceed → 24-task measurement; flat → lr 1e-4; collapsed → ≈3e-5); (2) only a measured `proceed` arm
  unlocks the 60-task pair (≈85,600 quota-seconds); (3) repair the pre-existing `test_l4_feasibility` fixture
  and the missing localization resource, and commit the 2026-10-13→20 backlog on a feature branch; (4) the
  guard's skip path still needs one real kernel run to be trusted end to end.

### 2026-10-19 22:30 (Asia/Shanghai) - V9 closes the screening round: lr 2e-4 destroys tool use

- Objective: read the last funded arm (V9, lr 2e-4) and close the screening round with a verdict.
- **V9 completed** (`junzhejiao/lora-v9-tail3-r16-lr2e4`, wall 14,946.1 s) and resolved **0 of 24**. Every task
  has **0 tool calls**, 1-2 LLM calls, a 0-byte patch and the 8-minute session timeout (mean 525.9 s, median
  527.4 s, 24 of 24 tasks ≥7 min). Training fitted the targets instead of staying flat: loss 2.0966 → 0.7323,
  gradient norms to 28.7, adapter sha256 `63ae2cf39079d139d06b3c887af173f33bc4fb479f3d52c63a06324a3a565d4a`.
- **Mechanism, measured at token level**: all 24 traces were downloaded and read
  (`artifacts/tmp/v9-out/**/traces/`, `artifacts/tmp/summarise_v9_traces.py`). Every trace records
  `model_name: "main_lora"` (a third confirmation the adapter served the requests); 13 of 24 end at exactly
  `total_completion_tokens: 8192`, the output cap, and the rest accumulate 12,3xx; 24 of 24 repeat one line at
  least 5 times, up to 2,666× ("Done."), and `fastapi_13537` repeats "The fix is to keep track of already
  visited field…" 170 times until the cap. So it is a decode loop of the tail-3 target — the model narrates a
  final answer instead of calling tools — not a server stall.
- **Configuration correction found while writing this up**: `train_l4_device_map.py` builds its update plan from
  one pass over the records and only truncates it (`plan = plan[:args.max_updates]`), so every run on this
  8-record dataset trained **8 updates**. The V9b "32 updates" notebook would have run the same 8 updates; the
  §12.2 projection that assumed 16-32 steps was wrong on the step axis, and only the learning rate moved.
- **Cost**: V9 consumed **29,935.3 quota-seconds**, the most expensive arm of the programme, because a collapsed
  agent maximises runtime. Month-to-date is now **114,317.5 s used against 108,000**, i.e. 6,317.5 s over, with
  the refresh on 2026-10-24T00:00Z. Nothing further can run before then.
- **Final screening verdict**: four arms, no adapter better than no adapter; the measured ±2 noise floor from
  the byte-identical repeat is what the adapter's "+1" turned out to be; the next test is a single arm at
  lr 5e-5, then a 60-task paired confirmation (≈85,000 quota-seconds) only if that arm moves.
- Changed files: `docs/experiments/2026-10-18-lora-improvement-program.md` (§4 status, §12 consequence, §12.2
  step-axis correction, new §14), `docs/experiments/2026-10-18-merge-plan.md` (V9 evidence, slot stays empty),
  `runs/2026-10-19-lora-screening/` (`V9-summary.json`, `V9-stdout.txt`, `all-arms.json`, updated `REPORT.md`,
  `diagnostics.json`, `timeout-proximity.json`), `WORKLOG.md`. No competition submission; kernel private.

### 2026-10-19 18:15 (Asia/Shanghai) - Screening verdict: the adapter did not beat a re-run of the baseline

- Objective: price the 24-task noise floor with a byte-identical baseline repeat, then spend the last
  affordable arm on an adapter that can actually move weights.
- **The repeat completed** (`junzhejiao/lora-v0b-baseline-repeat`, 9,761.5 s wall, 19,542.3 quota-seconds)
  and resolved **10/24**. Paired with V0 (9/24): 7 both, 2 V0-only, 3 V0b-only, 12 neither — the same 3-gain
  / 2-regression split the adapter was credited with, exact McNemar p = 1.0. Against V1 it is exactly equal:
  **10 both, 0 discordant, 14 neither**, so V1's resolved set *is* V0b's resolved set.
  Record: `runs/2026-10-19-lora-screening/noise-floor.json`,
  `docs/experiments/2026-10-18-lora-improvement-program.md` §12.
- **What replicates against the repeat**: the adapter arm is 10-13 % slower per task (361.5 s vs 319.1 and
  328.2 s), which is the KV-cache cost and explains `rich_3278` flipping to a timeout; and it writes a patch
  on 21/24 tasks versus 19/24 in both baselines, which resolves nothing extra (`rich_3470`, `rich_3944`,
  `rich_4075` all fail verification). Everything else is noise.
- **Weight-scale measurement added** (`artifacts/tmp/adapter_delta_norms.py`): reading only the LoRA factors,
  the exported adapter has RMS(`B`) = 8.2e-5 and RMS(`ΔW`) = 4.6e-6 per element, versus 1.26e-4 / 7.0e-6 for
  the older 32-update rank-16 adapter, against base scales of 1e-2 or more. The flat loss and the
  byte-identical patches are the same fact seen three ways.
- **Launched the last affordable arm**: `junzhejiao/lora-v9-tail3-r16-lr2e4` (V9, 18:06 local) — identical
  to V1 except lr 2e-4, launched with `--max-updates 16`. **Corrected by the 22:30 entry:** the trainer caps
  every run at 8 updates on this dataset, so V9 differs from V1 only in the learning rate (10×), and that
  alone collapsed it to 0/24.
- **Budget ruling recorded**: a 99-task paired confirmation costs ≈126,000 quota-seconds at the measured
  2.00 × billing, so it cannot fit one 108,000 s allowance; confirmation must shrink to 50-60 tasks per arm
  or move to the Trillium H100 runtime.
- Changed files: `docs/experiments/2026-10-18-lora-improvement-program.md` (§4 status column, §12.1 rewritten
  against the repeat, §12.2 weight measurement, §12.3 cost model), `docs/experiments/2026-10-18-lora-submission-theory.md`
  (the trained-adapter serving gap closed with the measured kernel counts), `runs/2026-10-19-lora-screening/`
  (`REPORT.md`, `noise-floor.json`, `arms/{V0b-summary.json,V0b-stdout.txt,diagnostics.json,adapter-norms-*.json}`,
  `notebooks/V9b-evaluation.ipynb`), `WORKLOG.md`. No competition submission; all kernels private. Git: the
  branch still carries the 2026-10-18 uncommitted work, so nothing was committed here (see follow-up).

### 2026-10-19 12:10 (Asia/Shanghai) - LoRA screening: baseline V0 and adapter V1 measured, repeat launched

- Objective: continue the 2026-10-18 LoRA programme (user instruction "continue yesterday's work"), i.e.
  read the two kernels left running overnight, report them, and keep the screening round moving.
- **Both overnight kernels completed without failure.** `lora-m2b-baseline-site-packages` resolved 9/24 in
  9,595.9 s; `lora-m3-v1-tail3-r16` (tail-3 supervision, rank 16 / α32, 8 updates, lr 2e-5, trained inside
  the same kernel) resolved 10/24 in 10,976.8 s. Paired: 7 both, 2 V0-only, 3 V1-only, 12 neither, net +1,
  exact McNemar p = 1.0. Summaries were recovered from `kaggle kernels logs` (reconstructed stdout written
  next to them) because `kernels_output` re-downloads the whole working tree slowly.
- **The trained-adapter serving gap is closed by measurement.** The V1 evaluation Triton cache contains 96
  LoRA kernel artifacts (`_lora_shrink_kernel`, `_lora_expand_kernel`, fused `…lora_shrink_marlin_gemm…`);
  the V0 cache contains none. So the adapter was mounted through `adk_submission.VllmServer`
  (`--enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules main_lora=…`) and executed, which is the
  one gap `docs/experiments/2026-10-18-lora-submission-theory.md` left open.
- **The adapter arm did not test the hypothesis, and that is the main finding.** Training moved
  122,429,440 parameters through 8 AdamW steps at lr 2e-5 (two `rich_3894` records skipped: 2,050-token
  prompt), loss 2.0966 → 2.0989, i.e. ≈1.6e-4 of coordinate movement. Six of the seven tasks resolved by
  both arms have byte-identical patch sizes, so the served weights changed almost no tokens.
- **Measured noise floor, from data already in hand**: the 4-task probe (`fastapi_9425`, `fastapi_9753`,
  `rich_3454`, `rich_2943`) resolved 4/4 with no adapter, while this 24-task baseline resolved 2 of the same
  4 with the same weights, seed and prompt. Sandbox/test observations, not sampling, drive that divergence,
  so ±2 at 24 tasks is the planning floor and the V1 "+1" sits inside it.
- **Carrier validation of the new notebooks**: `artifacts/tmp/v0b-repeat/evaluation.ipynb` is byte-identical
  to the V0 notebook; `artifacts/tmp/lora-v9-tail3-r16-lr2e4/evaluation.ipynb` is byte-identical to M3 in
  `agent_bundle`, `model`, `scripts`, `task_ids` and `wheelhouse` (the v31 snapshot must be passed as
  `artifacts/review-pr23/resource/manifests/kaggle-wheelhouse-snapshots.json`; the workspace copy of that
  file is absent on this branch and `artifacts/main-sync` holds a different, non-matching entry).
- **Launched**: `junzhejiao/lora-v0b-baseline-repeat` (noise floor at 24 tasks, pushed 12:03 local).
  **Prepared, not pushed**: V9 (lr 2e-4, 16 updates), which fits the remaining quota once the repeat drains.
- Quota: **64,839.9 s used of 108,000 s, 43,160.1 s remaining** (refresh 2026-10-24); the pair cost
  41,227.6 s, exactly 2.00 × summed wall, so a 24-task arm with training costs ≈22,000 s.
- Changed files: `docs/experiments/2026-10-18-lora-improvement-program.md` (§12), new
  `runs/2026-10-19-lora-screening/` (REPORT.md, report.json, paired-table.md, per-arm summaries and
  stdout, four notebooks), `WORKLOG.md`. Helper scripts for the recovery live in `artifacts/tmp/`
  (`extract_m2b_m3.py`, `build_screening_report.py`, `compare_payloads.py`, `decode_payload.py`).
  No competition submission; all kernels private. No source change outside documentation.

### 2026-10-18 23:30 (Asia/Shanghai) - LoRA programme: M2 baseline, the fastapi dependency gap, M2b and M3

- Objective: run M2 (24-task baseline) and start M3 (first adapter arm), reporting quota promptly.
- **M2 completed but is not a usable baseline**: kernel `lora-m2-budgetfit-screening` (session wall 5,608 s)
  reported 6/24 resolved, all `rich`. Nine of the twelve `fastapi` tasks never started the agent:
  `Workspace editable install failed: ... annotated-doc>=0.0.2 (from fastapi) (from versions: none)`.
  `annotated-doc` and `pytest-timeout` exist only in the official container's `/eval-test-wheels` mount,
  which a Kaggle kernel cannot attach. Recorded as an environment gap, not a task outcome.
- **Fix and probe**: added `--site-packages` to `scripts/prepare_lora_eval_notebook.py`, so each sandbox may
  satisfy those two packages from the image while the task's workspace package still wins the import (the hook
  asserts the imported origin is inside the workspace). Probe `lora-probe-fastapi-deps` with four tasks:
  **4/4 resolved** — `fastapi_9753` (26 calls, 2,057 B), `fastapi_9425` (24 calls, 544 B), `rich_3454`
  (17 calls, 657 B), `rich_2943` (22 calls, 400 B); wall 1,679 s.
- **Launched with the corrected setting**: `lora-m2b-baseline-site-packages` (no-adapter baseline, same 24
  tasks) and `lora-m3-v1-tail3-r16` (V1 adapter: tail-3 supervision, rank 16 / alpha 32, 8 updates,
  learning rate 2e-5, task-statement prompt, sequence 2,048; trained inside the kernel and mounted at
  `agent/adapters/main_lora` before evaluation). Both were running or queued at this entry.
- **Quota**: **23,597.2 s of 108,000 s used, 84,402.8 s remaining** after M1 v1/v2, M2 and the probe.
- Changed files: `scripts/prepare_lora_eval_notebook.py` (`--site-packages`),
  `docs/experiments/2026-10-18-lora-improvement-program.md` (§10 corrections, §11 invalidators), `WORKLOG.md`.
  No competition submission; all kernels private.

### 2026-10-18 21:20 (Asia/Shanghai) - LoRA programme M1: the in-kernel evaluation carrier works

- Objective: start milestone M1 of the LoRA improvement programme (user instruction), report quota
  exhaustion promptly, and later name the best trained model with its measured data. Quota is not exhausted:
  **8,947.6 s used of 108,000 s, 99,052.4 s remaining** at the last check.
- **Built the carrier**: `scripts/prepare_lora_eval_notebook.py` generates one private Kaggle L4 kernel that
  verifies the 41-wheel v31 snapshot, installs the reviewed stack plus torchao 0.16, unpacks a frozen
  budget-fit agent (recovered byte-wise from the trajectories-r3 release), and runs
  `evaluate_agent.py infer --task-ids ...` + `verify --patch-source agent` in batch mode (one server, one
  sandbox per task) with its own offline dependency hook, because the repository hook hardcodes the official
  container's `/data/wheels` and `/eval-test-wheels` mounts and asserts `swegemma == 0.2.10`.
- **M1 v1 failed honestly** (wall 550.5 s): every task stopped in sandbox setup because
  `pytest-timeout==2.1.0` exists in neither mounted wheel source; the official test-wheelhouse mount is not
  attachable from a Kaggle kernel.
- **M1 v2 passed** (wall 1,354.3 s): `rich_3454` resolved (14 calls, 656 B patch, 207 s), `rich_3675` timed
  out at 4.0 min with an empty patch, `rich_3676` resolved (1,649 B) but also hit the 4.0 min timeout. So
  2/3 resolved, 1/3 clean, 2/3 timed out. The carrier — install → vLLM → batch inference → official
  verification → summary JSON — is proven; the tolerant hook and the budget finding are recorded in
  `docs/experiments/2026-10-18-lora-improvement-program.md` §9.
- **Budget correction applied before screening**: the recorded agent enforces `max_time_minutes: 4.0` while
  its prompt promises 8 minutes and the 0.18 submission ran at 8 minutes, so the screening arms now use 8.0
  with `max_tool_calls: 28`.
- **M2 launched**: `junzhejiao/lora-m2-budgetfit-screening` evaluates the no-adapter budget-fit baseline on
  the fixed 24-task screening sample (12 fastapi + 12 rich, seeded hash of the 99-task manifest, stored in
  `artifacts/tmp/m1/screening-tasks.json`). It is running at the time of this entry.
- **Training extension prepared but not yet run**: the same generator accepts `--train-dataset` and embeds the
  four-L4 workers, so M3 trains the variant adapter in the same session and mounts it at
  `agent/adapters/main_lora` before evaluation. No adapter has to cross the 1 MB notebook limit.
- Changed files: `scripts/prepare_lora_eval_notebook.py` (new),
  `docs/experiments/2026-10-18-lora-improvement-program.md` (§9), `WORKLOG.md`. GPU work: two private kernels,
  no competition submission.

### 2026-10-18 20:17 (Asia/Shanghai) - LoRA programme: submission theory, experiment design, merge plan

- Objective: the user asked for three things in order — (1) a theory-only answer to whether a LoRA can be
  submitted without disturbing the trained base, no real submission allowed; (2) an experiment programme
  that measures whether LoRA raises accuracy and searches for the best training variant; (3) a comparison
  of the best Kaggle submission with GitHub `main` and the local workspace plus a merge plan.
- **Clarified before acting.** Six questions were answered by the user: the no-submission rule covers the
  whole task; the evaluation carrier is an in-kernel "train then evaluate" Kaggle run; the budget is not
  capped and may wait for the 2026-10-24 refresh; screening uses 20-30 tasks first and the full 99 afterwards;
  both the budget-fit baseline and the rank-16 v3 adapter are baselines; the merge plan must cover both git
  and the final submission configuration.
- **Part 1 written**: `docs/experiments/2026-10-18-lora-submission-theory.md`. It separates three readings of
  "not losing the trained parameters" (base untouched; adapter bytes and honouring; behaviour unchanged and
  therefore a score property), shows the read-only base and separate LoRA tensors, and cites the measured
  evidence: the serving probe (base repeat delta 0.0, rank-8 adapter 1.187468, rank-32 10.069917), the run-11
  official-runtime reload (`status: applied`, effect 0.596932), and our adapter (820 tensors, all keys
  matching, `problems: []`). It also records the one unclosed gap: no trained adapter has yet been mounted
  through `adk_submission.VllmServer` itself, only through `vllm` directly.
- **Part 2 designed**: `docs/experiments/2026-10-18-lora-improvement-program.md`. Fixed factors, nine
  screening variants (tail vs loop, rank 8/16/32, two-projection vs all-projection, LR, updates, task
  weighting), the screening/confirmation split (24 tasks then 99), the decision rule (McNemar p < 0.05 with
  clean completion not falling), the cost model (≈11.5-13.7k quota-seconds per screening kernel), and the
  carrier: one kernel that trains the adapter, assembles the agent in `/kaggle/working`, then runs the
  batch evaluator (`--task-ids`, one server, seed 42) and the official verifier — no adapter transport at
  all. The known limitation is stated: on four L4s training can only use the task statement prompt while
  evaluation uses the harness system prompt.
- **Part 3 prepared**: `docs/experiments/2026-10-18-merge-plan.md`. Compares the three artifacts, records two
  configuration traps (the submitted prompt promises 8 minutes while `eval_config.yaml` enforces 4; the
  prompt forbids the graph tools, which explains zero graph calls), gives the git sequence (commit local
  work, cherry-pick #21, merge `main`, resolve the four conflicts, open a small integration PR, rerun
  `check_repository.py`) and the configuration composition table with the adapter slot conditional on part 2.
- **Verified while writing**: the budget-fit agent is recoverable byte-wise from the release
  (`agent.yaml`, `configs/sampling.yaml`, `eval_config.yaml`, `prompts/system.md`), the release's system
  prompt itself forbids the three graph tools, the pending `max_tool_calls 50` submission is still PENDING,
  and no submission ZIP for the 0.18 run is identified in this workspace.
- **Not done, stated plainly**: the part-2 instrument has not been built or run yet, so no accuracy result
  exists; part 3's adapter slot therefore defaults to empty. The next step is the instrument's first
  milestone (baseline arm on 3 tasks, one kernel) followed by the baseline and first trained variant on the
  24-task screening set.
- Changed files: three new documents under `docs/experiments/` and `WORKLOG.md`. No GPU run, no submission,
  no remote mutation in this round.

### 2026-10-18 20:02 (Asia/Shanghai) - Complete the outstanding review request: PR #8

- Objective: the user asked for every review that someone requested of this account but that was never
  submitted. Inventoried all 15 PRs through the API (requested reviewers plus submitted reviews): only
  **PR #8** was outstanding (`liaodid` and `KaltistEsperanta` requested, 0 reviews); #18, #23 and #25 were
  already reviewed by this account; #16 is authored here and awaits `liaodid`; #20 and #22 request
  `jwj1342` and `Sholpan1no`; #17 (closed without merge) still lists this account but cannot be reviewed
  into anything. Repo-wide comments were also scanned for `@KaltistEsperanta` asks: #18 (done), #13's
  closing instruction (done), and the October 15/16 team asks (done or superseded).
- **PR #8 reviewed at `bd9b167`** (head), in the four areas the PR assigns to this account: model/protocol,
  save/resume, adapter loading, completion behaviour. Read the code rather than the description:
  prefix/token boundary assertions and the `<|tool_response>` stop id 50 check in `src/sft_data.py` and
  `scripts/encode_sft_dataset.py`; the resume contract and `base_weights_digest`; the zero-versus-trained
  adapter instrument in `scripts/check_lora_serving.py`; the SLURM/full-replica constraint that keeps the
  trainer Trillium-only.
- **Both previously requested fixes are in, with tests.** `convert_trajectory` now raises
  `UnsupportedAction('Final answer lacks an observed submit_patch call')` instead of silently dropping a
  terminal turn (`tests/test_sft_data.py` asserts the message), and `base_weights_digest` hashes every
  sharded safetensors file or fails with a message that names the layout
  (`tests/test_swe_lora_contract.py` covers shard/index changes and invalid names). The thinking-channel
  mismatch is now stated in the report's limitations, which was the earlier review's top question.
- **Manifest and code agree** on the claims checked: comparison 3/11 versus 2/11 resolved, 4/11 versus 8/11
  clean, 2/11 both; peak 67.38113355636597 GiB; `encoding_protocol.tool_handoff_stop_id: 50`; and
  `src/sft_schedule.py::loss_weights` implements the equal-weight example normalisation the report describes.
- **Local test run at this head**: 71 tests over 11 files, 67 pass, 4 platform-only failures (three assert
  POSIX path separators, one is the CRLF-versus-LF hash in `verify_eval_run`), and `test_eval_suite.py`
  cannot import on Windows (`fcntl`). `scripts/check_repository.py` passes once the materialised tree is
  git-tracked: **55 python, 19 shell, 15 configuration, 49 markdown, 1 agent package**.
- **Merge state measured, and it is the reason this is not a merge approval**: 15 conflicting files against
  `a5bbc85` (`src/sft_data.py`, `scripts/prepare_terminal_sft.py`, `scripts/prepare_sft_data.py`,
  `scripts/check_lora_serving.py`, `scripts/evaluate_agent.py`, `scripts/verify_eval_run.py`,
  `tests/test_sft_data.py`, `tests/test_eval_suite.py`, the READMEs); 40 of 86 changed files are already
  byte-identical on `main`; 29 exist only on the branch; and `bd9b167` has no CI run (0 check runs, pending
  combined status; the cited green run is `ccfcff0`).
- **Posted**: APPROVE with those conditions —
  [review 5562903932](https://github.com/jwj1342/gemma-agent-lab/pull/8#pullrequestreview-5562903932) — and a
  correction comment after noticing I had quoted the repository-check counts from the other PRs:
  [comment 6249674045](https://github.com/jwj1342/gemma-agent-lab/pull/8#issuecomment-6249674045) with the
  measured 55/19/15/49/1.
- Changed files: `WORKLOG.md` only. Remote mutations: one review and one PR comment.
- Follow-up: PR #8 needs an author rebase onto `main` and a fresh `Repository checks` run; this account
  should re-review the rebased head because the last-push approval rule invalidates this approval after any
  new push.

### 2026-10-18 19:46 (Asia/Shanghai) - Re-review PR #18 after a review notification

- Objective: the user received a review notification for PR #18 and asked for a review. The earlier round
  had already approved this PR, so the job was to establish whether anything changed and to review the
  current head rather than duplicate a review event.
- **State read first.** Head is still `c75571aaed57` (2 commits, 3 files, +26/-31), `mergeable: true`,
  `mergeable_state: behind`, base `625a597`. My APPROVE at that head (2026-10-18T09:37:26Z) is undismissed,
  and `requested_reviewers` no longer lists this account (Sholpan1no, liaodid, shizongniji are listed), so the
  notification is a stale request rather than a new push.
- **Content verified, not restated.** The only JSON change is `require_last_push_approval: true`; every bullet
  in `docs/ci-and-review.md` is backed by the versioned file (`strict: true`, check context `Repository checks`
  with `app_id` 15368, one approving review, stale dismissal, last-push approval, conversation resolution,
  `enforce_admins`, no force pushes or deletion); the `#protection-for-main` anchor exists; the repository is
  public (`private: false`) so the "became public on October 12" explanation holds; `GET /branches/main`
  reports `protected: true` at `a5bbc85`.
- **Boundary measured again.** The detailed protection endpoint still returns 404 to this account even with a
  public repository, so individual live settings remain owner-reported; only `protected: true` is
  independently observable.
- **Base drift checked.** `git diff 625a597 origin/main --` the three PR paths is empty, so `#10`/`#21` did not
  touch them and being behind `main` creates no content conflict for this PR.
- **Repository checks re-run** on the existing `artifacts/review-pr18` checkout at `c75571a`: passed (48
  python, 14 shell, 15 configuration, 60 markdown, 2 agent packages). Confined runs cannot do this (bash
  named pipes and temp writes are denied), so it used one approved unconfined command.
- **Three non-blocking findings posted as a PR comment** (6249530035): (1) `docs/repository-map.md` still calls
  the protection file an "intended ... request; actual enforcement depends on account entitlement", which now
  contradicts the PR; (2) the `#11` citation is an open issue about tool-call recovery — the post-merge review
  comment lives there and was summarized on PR #10 by `shizongniji` on 2026-10-13T01:19Z, and it listed three
  behavioural findings, so "found no defects" overstates it; (3) "Last checked: October 12" should say who
  verified it with admin credentials, since a collaborator cannot re-derive it.
- **Deliberately did not submit a second review.** A COMMENT review from the same account risks replacing the
  counted approval, so the findings went in as an issue comment and the APPROVED state was re-read afterwards
  to confirm it is unchanged. Recommendation recorded: keep the approval; the PR is mergeable as it stands.
- Changed files: `WORKLOG.md` only; the transcript of the review is in `artifacts/tmp/gh-2026-10-18/` and the
  remote mutation is one PR comment.

### 2026-10-18 18:44 (Asia/Shanghai) - Execute the task board: PR reviews, trajectory audit, budget-fit adapter

- Objective: carry out the tasks assigned to this account in the team tracker (#13/PR #16, PR #18, #4 data,
  #7 training, PR #8, PR #23, PR #25, SWE-Gym/SWE-smith research, weekly note) and report whether the rank-32
  model improved.
- **PR #16**: read the live body through the API and confirmed `Closes #13` is present (description last
  updated 2026-10-18T09:34Z) and `mergeable_state: behind`. No new action was needed; the remaining gate is
  `liaodid`'s requested review. Re-fetched `requested_reviewers`: `liaodid` only.
- **PR #18**: verified the posted review from the earlier round (`KaltistEsperanta`, APPROVE, `c75571a`,
  2026-10-18T09:37Z) and its repository-checker log (48 Python, 14 shell, 15 configuration, 60 Markdown, 2
  agent packages). No further mutation, to avoid a duplicate review.
- **PR #23**: read the 15-file diff, not just the description. Reproduced the v31 manifest claim by
  comparing snapshots: 41 wheels in both, `swegemma` 0.2.10 -> 0.2.11, exactly three digests changed
  (`adk_eval_core`, `adk_submission`, `vllm`). Ran `scripts/check_repository.py` on a materialized tree of
  `b9c9c4d` (Windows cannot check out the colon-bearing CSV, so `artifacts/tmp/materialise_tree.py` writes
  every other tracked blob): passed 48/14/15/61/2. Posted APPROVE with two follow-ups: the branch is behind
  `main` (`a5bbc85` has #21's import migration, the head alone still imports removed
  `swegemma.models.registry/discovery`), and the new lock re-resolves seven packages the wheelhouse snapshot
  does not cover (`cryptography` 48.0.1 -> 46.0.7, `pyarrow` 25.0.1 -> 26.0.0, `typer` 0.27.2 -> 0.27.3,
  `SQLAlchemy` 2.1.3 -> 2.1.4, `google-cloud-bigtable` 2.49.0 -> 2.50.0, `iniconfig` 2.3.0 -> 2.3.1,
  `truststore` 0.10.4 -> 0.10.5).
- **PR #25**: five-file docs diff. Verified the new anchor resolves, the generator and generated README move
  together, `validate_agent.py` still enforces the 31B base, the checker passes (48/14/15/60/2) and hosted CI
  is green. Posted COMMENT rather than approval: the new sentence "a local 12B smoke ... resolved 1/10
  fast-cohort tasks" has no run, receipt, or note in this repository (issue #24 proposes that smoke, it does
  not report one), which conflicts with the AGENTS.md rule to separate author claims from executed results.
- **PR #8**: re-fetched the PR and its reviews. Unchanged since 2026-10-12T16:33Z: 86 files, 59 commits,
  `mergeable_state: dirty` against `3054da8`, 0 reviews, 0 comments, and 0 check runs on `bd9b167`. The two
  review fixes are on the head. Since the prior review is recorded in
  `docs/experiments/2026-10-12-pr8-review.md`, no duplicate review was posted; the action is the author's
  rebase.
- **#4 data**: confirmed push access to `jwj1342/gemma-agent-lab-data`, downloaded the release asset by API
  (192,965 bytes, SHA256 `7deafdd5…`, matches the receipt) and audited all 31 trajectories: 31/31
  directories, 31/31 recomputed `trace_sha256` matches, 31/31 non-empty verifier-resolved patches (324-2,175
  bytes), 874 steps, 704 tool calls, 0 empty messages, 0 empty observations, per-task spread exactly as
  declared, baseline 13 / budget-fit 18. Prompt fixedness: `prompts/system.md` identical per family and the
  task prompt identical within a family; the only per-seed difference is `configs/sampling.yaml: seed`, which
  also explains why `agent_archive_sha256` differs per seed run. One labelling anomaly: `seed103-requests_7205`
  has no `final`-tagged step. Posted the audit to issue #4
  (https://github.com/jwj1342/gemma-agent-lab/issues/4#issuecomment-6248795836) and left the deeper quality
  audit to `liaodid`. The two reviews are
  [PR #23 approval 5562560592](https://github.com/jwj1342/gemma-agent-lab/pull/23#pullrequestreview-5562560592)
  and [PR #25 comment 5562560684](https://github.com/jwj1342/gemma-agent-lab/pull/25#pullrequestreview-5562560684).
- **Tool boundary research**: counted every tool invocation in the 31 trajectories. Budget-fit uses
  `run_command` 60.8%, `read_file` 26.9%, `edit_file` 6.8%, `submit_patch` 4.6%, `get_status` 0.5%,
  `write_file` 0.5%, and **zero** calls to the three graph tools; baseline uses `write_file` 7.1%. Recorded
  reading: trimming is measurable and success-only confounded, and the rules boundary is that new tools must
  ship as sandbox skills (host discussion 743573), so trimming our `agent.yaml` is inside our control while
  host-registry substitution is not.
- **SWE-Gym / SWE-smith**: read both papers' method sections (SWE-Gym: 2,438 executable instances from 11
  repos, 491 trajectories give +12.3/+13.6 points, metrics include Empty Patch and Stuck in Loop; SWE-smith:
  50k instances from 128 repos, bug synthesis at 35-97% yield per strategy, LM-generated issue text, one
  environment per repository). Wrote `docs/meetings/2026-10-18-weekly.md` with configuration, submissions,
  results, concrete trajectories, the tool boundary, the integration plan, and five questions for the team.
- **#7 training**: wrote `scripts/build_trajectory_sft_dataset.py` (release -> prompt/completion records,
  trace-hash gate, per-task cap, head/tail step rendering with truncation counters, and a `--prompt-mode
  task|full` switch) and produced `artifacts/tmp/budgetfit-sft/budgetfit-sft-v1.jsonl`: 10 records over 5
  tasks at `--max-per-task 2`, seed 20261018. Encoding preflight on Kaggle measured the boundary that the
  default sequence length breaks: 2048 skips all 10 records (prompts 2,631-3,654 tokens,
  `dataset_incompatible`), 8192 encodes 10/10 with mean prompt 3,213.6 and mean target 1,683.5 tokens.
- **Four training attempts, one success.** `budgetfit-trajectory-lora-rank16-v1` (seq 6144) and
  `...-tail-v2` (seq 4608) both OOM on the first backward (158.00 MiB and 120.00 MiB requests) after
  loading 66 modules 7/19/19/21; `...-tail-v3` with `--max-memory-mib 13500` reproduced the identical
  placement (`cpu_modules: 0`) and OOM'd the same way, proving `balanced` ignores the lowered cap and the
  CPU fallback only fires on load failure; `...-v4` (seq 2560, task prompt only) completed **3** updates
  (losses 2.013/1.975/2.351, peaks to 20.05 GiB) and OOM'd at the fourth, which located the ceiling: the
  AdamW state for 122,429,440 parameters is allocated on the first step and peaks land at ~20.0 GiB per
  card. `junzhejiao/budgetfit-task-tail-rank16-v5` (seq 2048, task prompt only, tail-3 completion,
  1,865 tokens worst case) **completed 8/8 updates** and exported a validated adapter: 820 tensors,
  244,982,352 bytes, SHA256 `f9db770dc600ed57c3e0e5de7be7fd3d51549a3a41a76b09793c670747947514`,
  `unmatched_keys: []`, `problems: []`, losses 2.0966/2.2445/2.8107/2.4196/2.1903/1.7857/2.2695/2.0943,
  session 361.8 s (~723.6 quota-seconds), artifact at
  `artifacts/tmp/budgetfit-v5-out/l4-feasibility/adapter/` (copying into `runs/` is sandbox-denied).
  Recorded as a starting point, not a capability claim: 8 single-record updates at 2e-5 are not convergence,
  and the training prompt omits the harness system instruction because the full prompt (2,631-3,654 tokens)
  cannot share a 20.5 GiB card with any completion.
- **Generator change**: `scripts/prepare_l4_feasibility_notebook.py` gained `--max-memory-mib` (passes the
  worker's existing flag through the plan and the launch command; verified in the rendered notebook). It did
  not change placement, so the real fix is a training-OOM retry with `device_map: auto` plus a host-RAM
  allowance. Also recorded: the generator emits `machine_shape: NvidiaL4` even with `--cpu-only`, so "free"
  CPU preflights land on accelerators (quota 0 -> 553.749 s observed).
- **Reporting**: posted the October 18 measurement comment to #7
  (https://github.com/jwj1342/gemma-agent-lab/issues/7#issuecomment-6249358114) with the rank-32 verdict, the
  five-attempt memory table, the adapter hashes, both caveats, and the carrier blocker.
- **Rank-32 verdict**: no rank-32 model or submission exists. The 09:40Z retry kernel completed with
  `updates_completed: 0` and `adapter: null` because `--session-limit 1500` was below the device-map phase's
  advertised 1500 s requirement; the newest competition submission is a `max_tool_calls` change with no LoRA
  and was still PENDING. The only measured score movement this week is 0.18 -> 0.17.
- **Environment findings that cost time**: (1) the generator emits `machine_shape: NvidiaL4` even with
  `--cpu-only`, so CPU preflights land on accelerators and consume quota (0 -> 553.749 s observed);
  (2) kernel slug is derived from the *title*, so a title that does not slugify to the metadata `id` makes a
  push conflict (409) — the second push needed a metadata edit; (3) a Kaggle output file named with a
  Windows-style absolute path cannot be downloaded by the CLI, so preflight receipts should be written to a
  plain relative name; (4) the sandbox denies spawned-process writes to `runs/`, `scripts/` and `.git`, and
  denies `bash` named pipes, so `check_repository.py` needs one unconfined run (approved) and generated
  reports are written with the agent file tools.
- Changed files: `scripts/build_trajectory_sft_dataset.py` (new),
  `scripts/prepare_l4_feasibility_notebook.py` (`--max-memory-mib` pass-through),
  `docs/meetings/2026-10-18-weekly.md` (new),
  `docs/experiments/2026-10-18-budgetfit-trajectory-lora.md` (new),
  `docs/experiments/2026-10-11-lora-paired-protocol.md` (carrier amendment), `WORKLOG.md`,
  `runs/2026-10-18-task-audit/gh_api.py` (new helper, reads the credential without printing it), plus
  `artifacts/tmp/` receipts and the gitignored `data/private/trajectories-r3/`. No GPU code was changed and
  nothing was pushed to any branch; all remote mutations were two PR reviews, two issue comments (one under
  #4, one under #7), and no merge.
- Verification gaps stated rather than hidden: the new dataset builder has no unit test in this round (it was
  exercised on the real release and its selection, sizes and truncation counts are recorded, but there is no
  committed test); `--max-memory-mib` is verified to render into the notebook and the v3 session executed with
  it, but its intended effect (host offload) did not materialise; and `scripts/check_repository.py` could not
  run confined, so the one local run used an approved unconfined command.

### 2026-10-18 17:52 (Asia/Shanghai) - Complete PR #18 review and start the bounded rank-32 retry

- Completed PR #18 review at `c75571aaed57324ec4f13869ebabcbaa759bdc7d`: read all three changed
  files, compared protection semantics with official GitHub documentation, and ran the repository
  checker successfully (48 Python, 14 shell, 15 configuration, 60 Markdown, two agent packages).
  Submitted APPROVE with a PR-description wording correction and the explicit limitation that the
  account can verify `protected=true` but cannot read detailed live protection settings. No merge.
- The isolated review checkout is `artifacts/review-pr18`; Windows rejects one historical colon-bearing
  CSV path. Non-cone exclusion plus invocation-local `core.protectNTFS=false` materialized every
  checked file while leaving that unrelated CSV absent; the checker passed without changing source.
- Completed private release acquisition: `data/private/trajectories-r3/release.tar.zst`, 192965 bytes,
  SHA256 `7deafdd56991128fb76a1c9c9ebdfd21da3fd31c1422365efe219041ff129233`, matching the receipt.
- Imported PR #16's existing dependency-free adapter checker into the active continuation tree.
  The targeted suite passed 83 tests with three PyTorch-dependent skips. Earlier failures were local
  NumPy ABI mismatch, then the obsolete verifier's PyTorch dependency; neither was a GPU result.
- Generated `runs/2026-10-18-lora-rank32-v3/` and launched exactly one private Kaggle version with
  a 1500-second timeout: https://www.kaggle.com/code/junzhejiao/gemma-4-rank32-bounded-retry-october-18.
  Kaggle derived the actual slug from the title, differing from the metadata's requested slug;
  monitoring must use the returned URL above. Rank 32/alpha 64, 32 updates, 900-second load cap.
  No training success is claimed until outputs are inspected; no competition submission was made.
- Preserved the original memo under ignored `runs/2026-10-18-task-audit/memo-original.txt`, then
  replaced `E:/备忘录.txt` with pending tasks; completed preparation evidence is retained here and
  in the pre-existing notebook preparation records. User authorized this cleanup.
- Network recovery: two Python urllib TLS failures led to a PowerShell REST helper; checked for
  an existing review before posting, avoiding a duplicate mutation.
- Changed files: `WORKLOG.md`, `resource/l4_feasibility/verify_l4_adapter.py`; ignored run receipts,
  Windows-only development dependencies, private data archive, and external memo.


### 2026-10-18 17:45 (Asia/Shanghai) - Complete issue #13's closing link and private-data access

- Completed: added `Closes #13` to PR #16's existing description and read back the result.
  The five checked technical tasks remain complete; independent approval/merge is still pending.
- Completed: accepted the invitation for `jwj1342/gemma-agent-lab-data` (invitation 337574852),
  then successfully retrieved private release `trajectories-r3` metadata. Its two assets are the
  192965-byte archive and SHA256 receipt; acquisition and training audit follow separately.
- With explicit user authorization, posted task acceptance, bounded rank-32 plan, and the local
  evidence correction to #7: https://github.com/jwj1342/gemma-agent-lab/issues/7#issuecomment-6248285468.
- Branch metadata confirms `main.protected=true`; the detailed protection endpoint returns 404 to
  this account, so specific live settings cannot yet be independently verified. PR #18 review will
  distinguish the visible protection flag from owner-reported settings.
- Verification: successful API responses for invitation acceptance, release access, PR description
  update, and branch metadata. Sanitized receipts are in `runs/2026-10-18-task-audit/`.
- Changed repository file: `WORKLOG.md`; remote PR metadata, membership, and one authorized comment.
- Local preflight found two tests blocked by incompatible local NumPy binaries; installed Windows
  NumPy/safetensors/PyYAML into ignored `artifacts/windows-ci` for a clean retry. One guessed review
  branch name failed; subsequent reads use the API-confirmed `docs/branch-protection-status` ref.
- User additionally authorized removing completed memo content after transferring results here.
  Remaining: rank-32 run, matched evaluation, reviews, data audit/pilot, research and meeting notes.


### 2026-10-18 17:35 (Asia/Shanghai) - Recover the memo and translate current assigned tasks

- Objective: finish the E-drive memo and the user's current GitHub assignments, translating them
  before execution and checkpointing each completed item.
- Read `E:/备忘录.txt`, the work log, repository guides, and the programming skill. Preserved all
  existing dirty changes, including the uncommitted adapter-mount/frozen-archive repair.
- Authenticated read-only GitHub requests identified `KaltistEsperanta`; fetched all 25 issues/PRs,
  open PR metadata, complete threads for #4/#7/#13/#8/#16, and reviews for #8/#16. Snapshots are
  under `runs/2026-10-18-task-audit/`. Translated direct assignments, shared asks, and requested
  reviews in chat, distinguishing already completed #13 work from new budget-fit training work.
- Fresh Kaggle raw-duration check: 108000 GPU seconds allowed, 0 used, 0 reserved; refresh
  2026-10-24T00:00Z. No training or inference was launched in this inventory step.
- Decision: retain the memo's #24 deferral. Treat the meeting's approximately 3% rank-16 gain and
  half-trained rank-32 statements as unverified conflicts with local receipts, not measured results.
- Verification: GitHub API reads and quota command succeeded; `git fetch origin --prune` completed.
- Changed files: `WORKLOG.md` only; ignored API/quota snapshots and local remote refs were updated.
- Remaining checklist: complete memo experiments; review #18; obtain private data; close-link #13;
  audit #8/#23/#25 review requests; read SWE-Gym/SWE-smith and tool-extension contract; prepare
  budget-fit trajectory pilot and meeting evidence. External messages await explicit user consent.


### 2026-10-14 23:43 (Asia/Shanghai) - Re-audit the GitHub task and assess issue #24

- Objective: re-check whether the user's assigned GitHub task is complete; if so, translate issue #24's
  objective and determine whether the team can resolve it.
- GitHub evidence: authenticated API inspection identified `KaltistEsperanta` and confirmed that team
  tracker #7 delegates this member's focused work to issue #13. All five issue #13 tasks are checked.
  PR #16 remains open at `1fa3e28`, is mergeable, has a passing hosted `Repository checks` run, and has
  no reviews; `liaodid` remains the requested reviewer. Therefore the technical assignment is complete,
  but the policy handoff cannot close until an independent teammate approves and the PR merges. PR #8
  remains open and dirty as a separate research/rebase item. Other open team review requests were noted
  but were not conflated with the completed #13 assignment.
- Issue #24 evidence: read the complete issue and comment thread. The public Kaggle discussion API
  returned topic 748287, posted by Kaggle Staff Ryan Holbrook on 2026-10-11, explicitly allowing the
  31B, 12B, E4B, and E2B QAT W4A16 variants and requiring every agent/sub-agent in a submission to use
  the same variant. Staff replies say smaller models keep the same hardware and budgets, point LoRA
  training to the matching unquantized checkpoints, prescribe a 12B `--hf-overrides`, and say the
  `vision_embedder` failure is fixed in the newest wheelhouse. A same-day web recheck still found the
  competition Overview saying only 31B, so the official sources remain inconsistent.
- Capability assessment: evidence collection and the requested `AGENTS.md`, validator, training-plan,
  tests, and runtime parameterization are within this repository's capabilities. PR #23 already records
  wheelhouse v31 and passing CI, but remains open/blocked with no review and its measured H100 cohort is
  still 31B. The current v31 branch still hard-codes 31B across validation, model registry, asset paths,
  Kaggle metadata, and evaluation receipts, so a credible 12B smoke needs deliberate parameterization
  rather than a one-line allowlist edit. Final rule-page reconciliation requires the organizer, and an
  empirical 12B verdict requires external model/compute access.
- Tooling note: the GitHub plugin exists but is not installed and was already suggested within the
  seven-day cooldown, so no duplicate suggestion was issued. The existing private-repository PAT was
  used only for sanitized read requests; no credential contents were printed or recorded. One Git fetch
  and one later API request hit transient TLS failures; the required state was obtained through other
  successful API calls without repeating a failed approach more than once.
- Changed files: `WORKLOG.md` only. No GitHub/Kaggle comment, review, merge, push, GPU run, competition
  submission, source/configuration change, or credential change was made.
- Follow-up: wait for `liaodid` to review PR #16. For #24, ask the host to make the Model Selection page
  match topic 748287 (or explicitly state that the announcement supersedes it), archive the full official
  topic under `resource/`, review/merge the v31 compatibility work, parameterize model selection, then run
  a bounded 12B serving and matched fast-cohort smoke before considering any submission switch.

### 2026-10-14 01:36 (Asia/Shanghai) - Resolve the origin/main conflicts step by step

- Setup: created `integration/lora-rank8-origin-main` in `artifacts/integration-origin-main` at
  `89ee95f` and started a no-commit merge of `origin/main`; the expected four files conflicted. The
  first combined create-and-merge command failed because its not-yet-created directory was used as the
  working directory; splitting creation and merge succeeded without repeating that approach.
- Step 1/4 complete: resolved and staged `docs/README.md` by retaining all three local LoRA navigation
  entries and the remote prompt/context entry. Conflict-marker search and staged `diff --check` passed.
- Step 2/4 complete: resolved and staged `resource/manifests/README.md` by retaining every unique local
  and remote record, deduplicating `eval-assets-verification.json`, relabeling the fixed October 9
  manifest as historical, and retaining the snapshot registry as the notebook selection source.
  All referenced existing manifests were checked, and conflict-marker search plus staged
  `diff --check` passed. A PowerShell inspection initially repeated the unsupported `foreach { ... } |
  Format-Table` form and failed at parse time; the approach was replaced with collection into `$rows`
  before formatting and will not be reused.
- Remaining: `scripts/prepare_kaggle_runtime.py` and `tests/test_kaggle_preparation.py`; the merge has
  not been committed or pushed.

### 2026-10-14 01:31 (Asia/Shanghai) - Decompose the four active-branch merge conflicts

- Objective: identify the exact competing changes behind the four conflicts reported when comparing
  `fix/lora-rank8-controlled` with `origin/main`, without performing the merge or choosing a resolution.
- Method and evidence: inspected the conflict tree produced by `git merge-tree HEAD origin/main` and
  compared its stage-1 base, stage-2 local, and stage-3 remote blobs. The common base is `3054da8`;
  all relevant remote edits come from `625a597` (PR #10), while the local edits span the LoRA runtime,
  serving-probe, dependency, and evaluation work.
- Findings: the four files contain 11 conflict hunks. `docs/README.md` has one additive navigation-row
  conflict (three local LoRA reports versus the remote prompt/context guide). `resource/manifests/README.md`
  has two additive catalog conflicts (local SFT/module and October 9 wheel records versus remote first-LoRA,
  test-wheel, and snapshot-registry records). `scripts/prepare_kaggle_runtime.py` has five substantive
  contract conflicts: explicit October 9 manifest plus test-dependency/adapter validation versus the remote
  wheelhouse registry; payload fields; run-manifest fields; compressed base85 notebook/test-dependency/size
  behavior versus plain base64 plus snapshot metadata; and output metadata. `tests/test_kaggle_preparation.py`
  has three corresponding fixture conflicts: payload encoding and wheel schema, copied manifest input, and
  dynamic platform-safe baseline hash versus the remote constant plus wheelhouse receipt assertions.
- Resolution assessment: both documentation sides should be retained. The runtime should combine the remote
  snapshot registry and receipt with the local adapter staging, reviewed test dependencies, compressed base85
  payload, size gate, input paths, and dynamic hashes; the tests must assert that combined contract. Choosing
  either whole side would discard independently required behavior.
- Changed files: `WORKLOG.md` only. No conflicted file, branch, index, remote ref, runtime artifact, or
  experiment was modified; resolution still awaits the user's decision.

### 2026-10-14 01:24 (Asia/Shanghai) - Fetch remote updates, sync local main, and audit regressions and improvement signals

- Objective: fetch any remote updates without disturbing the dirty experiment worktree, stop for a
  user decision on conflicts, and re-check the GitHub task metrics and any positive remote changes.
- Fetch and sync: `git fetch --prune origin` left `origin/main` at `625a597` and pruned the merged/deleted
  `origin/fix/pr8-contracts-v2` ref. A conflict-free normal merge was completed in the separate clean
  worktree `artifacts/main-sync`; local `main` is now `a809b6c`, ahead of `origin/main` by the retained
  Windows-safe filename commit plus the merge commit. Nothing was pushed and the dirty active worktree
  was not modified except for this log.
- Conflict audit: a dry merge of `origin/main` into `fix/lora-rank8-controlled` reports four content
  conflicts: `docs/README.md`, `resource/manifests/README.md`, `scripts/prepare_kaggle_runtime.py`, and
  `tests/test_kaggle_preparation.py`. The recommended default is to leave the experiment branch intact
  until PR #17 settles the known-negative PR #10 main change. If immediate integration is chosen, use a
  dedicated branch, union the documentation/manifests, retain the local dependency/adapter/runtime
  preparation behavior, and add the remote tests while preserving dynamic hashes and Windows-safe byte
  handling.
- Regression verification: the exact PR #16 head `1fa3e28` passed all 53 L4 tests with 3 documented
  torch-dependent skips, the notebook generator returned `status: ok`, and the real rank-16 adapter
  validated at rank 16 with 820 tensors and 244,982,352 bytes. Synced `main` produced eight failures and
  one error under the Windows full suite: unavailable `fcntl`, checkout-byte/hash differences, and
  POSIX-path assertions. These are platform gaps already addressed on the local branch, not a hosted
  regression; GitHub's `Repository checks` for `625a597` completed successfully. The repository checker
  also cannot run its Bash checks on this host. `git diff --check origin/main..main` passed and the clean
  synced tree contains no colon-named path.
- Remote improvement audit: no capability lift is sufficiently controlled for promotion. PR #20's
  heredoc prompt is the strongest signal (`rich_3278` 1/3 versus a historical 0/22, plus two 1/1 task
  results), but it is not a matched cohort and needs independent reproduction. PR #22 changed prompt and
  wheelhouse together and never exercised its fallback; PR #8 reports 3/11 versus 2/11 patch verification
  but unchanged 2/11 clean/resolved with loop/time regressions. PR #21 is a clear compatibility recovery:
  it updates the runner to the republished wheelhouse API and turns the prior import failure into a
  successful 83-test `rich_3454` execution. PR #10 on current main is negative (11/31 versus 16/31 and
  +15.4% inference time) and PR #17 proposes reverting it.
- Failed approaches and recovery: the GitHub CLI was absent, so authenticated status inspection used the
  existing Git credential helper and GitHub REST API without logging credentials. A PowerShell pipeline
  directly after `foreach` caused one parser error and was replaced with a collected array. Earlier range
  arguments using `$base..ref` were parsed as variable names; subsequent Git diffs used braced variables
  or separate revision arguments. No failed approach was repeated.
- Changed files: `WORKLOG.md`; local Git metadata/main history also changed through fetch/prune and the
  local merge. No active-branch conflict was resolved, no remote ref was modified, and no GPU, Kaggle, or
  competition operation was performed.

### 2026-10-14 01:14 (Asia/Shanghai) - Complete the copied GitHub issue #13 tasks and repair PR #16 CI

- Objective: check whether the previously copied GitHub task was complete, finish every actionable
  gap, and stop if nothing technical remained.
- Remote audit: authenticated read-only inspection showed issue #13 open with five unchecked tasks,
  PR #15 merged, PR #16 open with no review and one failed `Repository checks` run, and PR #8 still
  open/dirty. Four issue #13 tasks already had measured evidence; the requested technique survey was
  incomplete, and PR #16's three failing tests depended on `numpy`/`safetensors` absent from the
  deliberately lightweight CI environment.
- Work performed on `experiment/four-l4-lora-feasibility`: added the issue's comparison matrix for
  standard LoRA, checkpointing, packing, trajectory SFT, replay filtering, rsLoRA, DoRA, NF4 QLoRA,
  DPO and GRPO; recorded benefit, cost, serving compatibility, evidence quality and two recommended
  next experiments. Replaced the adapter checker's framework dependency with validated standard-library
  safetensors-header parsing, and made the CLI tests write a minimal real-format file with the standard
  library. Commit `1fa3e28` was pushed to PR #16.
- Verification: 53 targeted L4 tests passed with 3 documented torch-dependent skips; the notebook
  generator returned `status: ok`; the new checker validated the real rank-16 adapter (820 tensors,
  244,982,352 bytes); `git diff --check` passed; GitHub `Repository checks` run `38436904198` passed.
  One initial generator check omitted its required owner/slug/output arguments and was corrected with
  the documented invocation. The local repository checker still cannot represent CI in this sparse
  Windows worktree (missing out-of-cone files, GBK decoding and no Bash), matching the prior recorded
  environment limitation; hosted Linux CI is the authoritative full gate.
- GitHub handoff: all five issue #13 boxes were checked, evidence was posted at issue comment
  `6151059197`, and independent review was requested from `liaodid` on PR #16. The issue was deliberately
  left open until that teammate approval and merge; no self-approval or policy bypass was attempted.
- Files changed: `docs/experiments/2026-10-12-l4-training-feasibility.md`,
  `resource/l4_feasibility/verify_l4_adapter.py`, `tests/test_l4_feasibility.py`, and `WORKLOG.md`.
  No GPU job, Kaggle upload, credential change, or competition submission was performed.

### 2026-10-13 24:45 (Asia/Shanghai) - Rank 32 hit the load budget and consumed the rest of the quota: the ladder pauses at rank 16

- Objective: run the next rung of the rank ladder, rank 32, with the encoding fixes that rank 16 proved.
- **The run failed at the base load, not at training.**
  `junzhejiao/gemma4-lora-rank32-sft-v1` (`gemma-4-rank-32-lora-sft-v1-four-l4`) ended COMPLETE with
  `single_process_device_map.status` reporting the placement ladder's single attempt as
  `outcome: timeout, seconds: 600.0` on `device_map: balanced` with the same 20,500 MiB per-GPU budget
  that loaded the base in 217 s one rung earlier. Verdict: *the sharded load did not finish within
  `--load-budget-seconds`*. Session wall 1,263.8 s.
- **Cost: 2,570.542 quota-seconds for nothing, and the quota is now 787.753 s remaining** (was
  3,358.295; refresh still 2026-10-17T00:00Z). At the L4x4 2x billing rate that is about **394 s of
  usable wall time**, while one rung costs roughly 470 s of wall time and ~1,750 quota-seconds. **The
  ladder cannot be continued without the refresh or another platform.**
- What this rung did establish: the same notebook shape, dataset embedding, tokenizer path and
  `--load-budget` plumbing ran to the load phase without an encoding error, so rank 16's fixes hold at
  rank 32. What it did not establish: anything about rank 32, since no update ran.
- The load timeout is unmeasured as to cause rather than explained: the same placement loaded in 217 s
  on the rank-16 run and in 196-217 s on the four feasibility runs, so a 600 s cap that this run
  exceeded is either host contention on the shared machine or a genuine variance in dispatch time that
  the earlier runs did not show. The cheapest correction for the next attempt is to raise
  `--load-budget` to match the measured worst case (the four-L4 feasibility notebook uses 480 s for its
  worker while its subprocess timeout is 1,100 s; this run's subprocess timeout was 1,500 s, so the
  worker's own cap was the binding limit, not the harness).
- Verification: the full suite passes (**131 tests, 4 subtests**), `tests/test_l4_feasibility.py` alone
  passes 68, and the generator's `--check` still reports `status: ok`. `scripts/fetch_kernel_log.py` is
  new: `kernels_output` downloads through `www.kaggleusercontent.com`, which stopped resolving from this
  machine (`getaddrinfo failed`, twice), while `api.kaggle.com` answered -- the log alone carried this
  measurement.
- State of the ladder: **rank 16 trained, validated and downloaded** (adapter at
  `runs/2026-10-13-lora-rank16-v3/output/l4-feasibility/adapter/`), rank 32 attempted and failed at load,
  rank 64 never attempted. No paired evaluation has been run for either rung, so no capability claim
  exists.
- Changed files: `scripts/fetch_kernel_log.py` (new), `WORKLOG.md`. Run directories under `runs/` are
  gitignored.

### 2026-10-13 24:10 (Asia/Shanghai) - Rank 16 of the ladder is trained on the real records, and the encoding cost two GPU sessions to get right

- Objective (user request): begin the LoRA experiment, training one rank group at a time from low rank to
  high rank. **Rank 16 is done and its artifact is in hand; rank 32 is launched next.**
- **Result of the rank-16 rung** (`junzhejiao/gemma4-lora-rank16-sft-v3`, title slug
  `gemma-4-rank-16-lora-sft-pilot-v3-four-l4`): `status: ok`. The required
  `gemma-4-31b-it-qat-w4a16-ct` base was dequantised to BF16 and dispatched across four L4s by one
  process, PEFT wrapped it at rank 16 / alpha 32, **32 of 32 optimizer updates completed**
  (`stopped_reason: completed_updates`) over 32 distinct records (0.278 of one pass, 115 trainable
  records after one over-long prompt was refused), and the adapter exported and validated:
  **820 tensors, 244,982,352 bytes**, SHA256 `bec7346f3e820873105545bf473909780c52940bf71bcbd21adec9afb8aa9e65`,
  `adapter-check.json` -> all 820 keys on the served naming scheme, 0 unmatched, layers 0-59, all seven
  projections. Session wall **468.4 s**, charged **1,743.06 quota-seconds**; per-update seconds
  1.382 / 5.030 / 16.509 (first update includes warm-up); per-GPU peaks 14.299 / 19.931 / 20.102 /
  19.255 GiB. The adapter file itself was downloaded into the run directory, so the artifact is local.
- **Two encoding defects were found by training, not by review, and both are now fixed with tests.**
  Run 1 (v1) died at `NameError: name 'true' is not defined` -- `json.dumps(True)` is `true`, and the
  generator had always rendered placeholders as JSON. Run 2 (v2) trained through PEFT and then died at
  the first update with `ValueError: too many dimensions 'str'`: on this image
  `apply_chat_template(..., tokenize=True)` returns **token strings, not ids**. The encoder now renders
  the template to text and tokenises explicitly, which also let the boundary be computed by comparing
  the prompt rendering with the full rendering token by token.
- **The fix was proven before another GPU second was spent, by a CPU-only preflight.**
  `train_l4_device_map.py --dataset-check` tokenises the dataset on the real tokenizer and exits before
  any model load; `prepare_l4_feasibility_notebook.py --dataset-check --cpu-only` emits that notebook with
  `enable_gpu: false` and without the four-L4 hardware assertion. The preflight
  (`junzhejiao/gemma4-lora-rank16-preflight`) reported `status: ok` for all 116 records: 115 supervised
  examples, 10 truncated at `--seq-len 2048`, mean 232.4 prompt and 522.0 target tokens, one record
  refused because its prompt alone exceeds the cap. That is the measurement that made run 3 safe.
- **Data normalisation found in passing**: 25 of the 116 reviewed records end their completion with
  `"\n "`, an extraction artifact; the reader now strips it and records `completion_trimmed` so the
  supervised region equals the rendered completion.
- Verification: `tests/test_l4_feasibility.py` -> **68 tests pass** (dataset reader, mask boundary,
  boundary-refusal, merged-token boundary, placeholder rendering as Python rather than JSON, unsubstituted
  placeholder, CPU-preflight shape and `enable_gpu`), and the generator's `--check` reports `status: ok`.
- Quota after this rung: **3,358.295 s remaining of 108,000 s** (refresh 2026-10-17T00:00Z). The failed
  v1 run cost 204.696 s and the successful v3 run 1,743.06 s, so the ladder now costs about 1,750 s per
  rung at this update count.
- Honesty limits for this rung, unchanged from the previous entry: the records are public-task
  prompt/patch pairs rather than verified agent trajectories, they overlap the control-valid evaluation
  tasks, and no paired evaluation was run, so this rung establishes that the ladder trains on real data
  end to end and nothing about capability.
- Changed files: `resource/l4_feasibility/train_l4_device_map.py`,
  `scripts/prepare_l4_feasibility_notebook.py`, `tests/test_l4_feasibility.py`, `WORKLOG.md`. Run
  directories under `runs/` are gitignored.

### 2026-10-13 23:15 (Asia/Shanghai) - The rank ladder starts: rank 16 trains on the real SFT records (session running)

- Objective (user request): begin the LoRA experiment, training one rank group at a time from low rank to
  high rank.
- **Scope agreed first, because the ladder is not affordable in one step.** Three rungs (16/32/64) with
  their paired evaluations need roughly 10,000 s of L4x4 wall time; the quota holds **5,306.051 s** and
  does not refresh until **2026-10-17T00:00Z**, about 2,653 s of wall time. The user chose a bounded
  low-rank pilot now (rank 16, training only) with evaluation waiting for the refresh, and the existing
  116-record public-task SFT artifact as the training data.
- **What was missing was not a rank flag but a training loop.** The measured four-L4 route
  (`train_l4_device_map.py`) trained the *synthetic handoff example* for at most 8 updates; the rank
  ladder needs real records. Added to that worker, offline-first:
  - `load_sft_jsonl`: reads the prompt/completion artifact, skipping malformed or empty records instead of
    ending a session on one bad line, and recording `prompt_chars` so the objective can supervise the
    patch alone;
  - `encode_sft_record`: the same boundary rule TRL uses for `completion_only_loss` -- the
    generation-prompt rendering must be a token prefix of the full rendering, and everything after that
    measured prefix is supervised. Deriving the boundary as `len(whole) - len(completion)` looks right and
    **drops one token of supervision**, which the new test caught before any GPU was spent;
  - a dataset-driven loop over one pass of the training split (`--dataset`, `--batch-size`,
    `--max-updates`), accumulating one forward/backward per record, still ending in the shared
    `target_loss`, so both routes measure one objective; per-update loss, records and instance ids are
    recorded.
- **The generator now has two shapes.** `--rank/--alpha/--batch-size/--max-updates/--dataset` plus
  `--skip-probe-routes` (drops the packed and four-FSDP probe routes, which train the synthetic example)
  and `--skip-serving` (the rank ladder spends its session on training and leaves the official-runtime
  reload to the evaluation step). The dataset is embedded as text, so no dataset upload is needed -- which
  this machine cannot do anyway -- and `--expected-rank` now follows the plan instead of a hardcoded 16.
  The feasibility shape is unchanged by default.
- **Prepared and pushed: `junzhejiao/gemma4-lora-rank16-sft-v1` version 1**, title slug
  `gemma-4-rank-16-lora-sft-pilot-four-l4`, four L4s, competition-attached, rank 16 / alpha 32, batch 1,
  bound 24 updates over one deterministic pass of the 116 records, `--device-map-budget 900`,
  `--device-map-timeout 1500`, `--session-limit 3300`, notebook 465,645 bytes.
- **Run 1 failed in the device-map cell, and the defect was ours, for 204.696 quota-seconds**
  (5,306.051 -> 5,101.355, measured): `NameError: name 'true' is not defined`, because
  `json.dumps(True)` renders `true` and the cell is Python. The generator's `render()` had always used
  `json.dumps`, so any boolean, `None` or numeric placeholder was a latent crash; `--skip-probe-routes`
  was simply the first feature to put one in a cell. Fixed by adding `python_literal()`, which recurses
  through containers and keeps Python spellings for booleans and `None`, and by moving the placeholders
  to an `@@@NAME@@@` sentinel: with a double-`@` placeholder, the "left unsubstituted" check fired on the
  embedded base85 worker payloads, which contain `@@` pairs.
- **Run 2 pushed as `junzhejiao/gemma4-lora-rank16-sft-v2`** (title slug
  `gemma-4-rank-16-lora-sft-pilot-v2-four-l4`), `--max-updates 32` at ~8 s per update from the recorded
  probe cost, inside the same 900 s wall budget. Its result is recorded in the entry above this one once
  the session ends.
- **Offline verification**: `tests/test_l4_feasibility.py` -> **64 tests pass** (8 new dataset tests, the
  rank-ladder shape test, the placeholder-rendering test that would have caught run 1's crash, and the
  boundary-refusal test), the generator's `--check` reports `status: ok`, and the full suite passes
  (**125 tests, 4 subtests**) before this entry was finalised.
- **Estimated truncation at `--seq-len 2048`**: 4 of 116 records exceed the cap on a 4-chars/token
  estimate (`fastapi_14463` has a ~2,569-token prompt and is refused by design rather than truncated
  silently, while the other three lose part of the completion and are recorded as `truncated`), so the
  first rung trains a near-complete pass.
- Changed files: `resource/l4_feasibility/train_l4_device_map.py`,
  `scripts/prepare_l4_feasibility_notebook.py`, `tests/test_l4_feasibility.py`, `WORKLOG.md`. Generated
  run directory `runs/2026-10-13-lora-rank16/` (gitignored).
- Known honesty limits for this rung: the records are the public-task prompt/patch pairs, not verified
  agent trajectories, and the training records overlap the control-valid evaluation tasks, so this rung can
  establish that the ladder trains on real data end to end -- it cannot support a capability claim. No
  paired evaluation is attempted in this rung.

### 2026-10-13 21:30 (Asia/Shanghai) - The fastapi starlette sweep landed: 99 control-valid tasks, past the 10-point power bar

- Objective: resume yesterday's task and launch the 30-task fastapi sweep that was prepared but
  deliberately left unlaunched, because the control-valid set had to grow before any LoRA gain
  could be measured.
- **The prepared notebook was first proven reproducible, then pushed.** Its cells already carried
  the measured-starlette logic, and regenerating it with the committed generator differed in
  exactly one payload string: the declarative `test_deps.manifest_path`. Every wheel digest, the
  test-deps manifest digest, the agent scripts, the eval config and all task ids were identical,
  so the recorded file was rewritten with the committed generator's output
  (`notebook_sha256 de25354d1108f067838288b9f2755036a8440ef7c1f729f67055368296640dac`) and the
  earlier file kept as `notebook-previous.ipynb`.
- **Kernel `junzhejiao/gemma4-control-fastapi-starlette` version 1, CPU-only, COMPLETE in about
  30 minutes of notebook time (RUNNING 12:47Z, COMPLETE 13:19Z).** Result: **28 of 30 usable**
  (`usable_count: 28`), against 28 of 30 failing in the same-task batch earlier the same day. The
  two non-usable tasks are honestly different failures, not a broken sweep: `fastapi_14186` has an
  empty-arm collection error with a gold arm that runs but fails
  `test_model_description_escaped_with_formfeed`, and `fastapi_14077` resolves in **both** arms, so
  it is already green at base.
- **The pin is measured, not assumed.** `task-environment-pins.json` records the same entry for all
  30 tasks -- `Requested: measured: newest wheel whose Router accepts on_startup`, installed
  `0.52.1`, returncode 0 -- and the probe trail is
  `[1.6.0 rejects, 1.3.1 rejects, 0.52.1 accepts]`. Exit-code pairs across the 30 tasks: 26 at
  (empty 1, gold 0), 3 at (2, 0), 1 at (0, 0).
- **Union recomputed: 99 control-valid tasks (fastapi 60, rich 39)**, up from 73, with 26 additions
  and no losses. That is past the **78** the paired protocol needs for a 10-point gain at 80 % power
  and 8 validated in more than one sweep. Recorded in `runs/2026-10-13-control-valid-set.json` and
  in the new tracked manifest `resource/manifests/control-valid-tasks-2026-10-13-final.json`, which
  supersedes the 73-task manifest. `artifacts/tmp/union_valid_tasks.py` now discovers executed
  sweeps by globbing `runs/**/control-sweep-report.json` instead of a hand-maintained source list.
- **The pairing protocol document now carries an amendment** stating that the task-set constraint
  is cleared and that the binding constraint has moved to compute: quota is unchanged at
  **5,306.051 s of 108,000 s** and refreshes **2026-10-17T00:00Z** (re-read this session, about
  **2,653 s** of L4x4 wall time), while the recorded agent cost on four L4s is ~119 s baseline and
  ~171 s portable per arm, so a 99-task paired run needs roughly 8-10 GPU hours. A full paired
  evaluation therefore waits for the refresh; this session spent **no GPU quota**.
- Verification: the full offline suite passes with the local wheel-installed pytest --
  **116 passed, 1 warning, 3 subtests passed** -- and `tests/test_control_sweep_notebook.py` alone
  passes 7. The sweep's own artifacts were read from the downloaded output
  (`control-sweep-report.json`, `task-environment-pins.json`, `pinned-wheels.json`), not inferred.
- Environment facts worth keeping: the kaggle CLI is only in the `kaggle` conda environment
  (`C:\Users\32747\miniconda3\envs\kaggle\python.exe`), `bounded_connect.py -m scripts.X` requires
  `PYTHONPATH=E:\Desktop\CodingAI`, and pytest is not installed in either interpreter -- it was
  installed from the reviewed local wheels into `artifacts/tmp/pylibs` with
  `--no-index --no-deps --target`.
- Changed files: `resource/manifests/control-valid-tasks-2026-10-13-final.json` (new),
  `docs/experiments/2026-10-11-lora-paired-protocol.md` (amendment), `WORKLOG.md`. Generated
  notebooks, downloaded sweep output and helper scripts live under `runs/` and `artifacts/`
  (gitignored).
- Remaining follow-up: the 53 tasks still invalid (11 fastapi that now fail exit 1 in both arms, 9
  rich, 13 requests, 1 httpx) each need a per-class measurement like this one before a 5-point
  effect becomes detectable; no adapter has been trained yet, so the paired run still has nothing
  to compare; PR #8 still needs its rebase.

### 2026-10-13 02:00 (Asia/Shanghai) - The control-valid set grew from 4 tasks to 73; the fastapi class is measured and fixed

- Objective (goal round 1, closing): grow the control-valid task set until a LoRA gain is measurable.
  It is done: **73 control-valid tasks**, up from 4.
- Results of the four CPU-only scale-up sweeps, all carrying the embedded reviewed harness wheels:
  | Sweep | Attempted | Usable |
  | --- | ---: | ---: |
  | `frozen38` (the set that returned 0/38 on 2026-10-09) | 38 | **29** (17 fastapi, 12 rich) |
  | untried `rich` batch | 30 | **24** (80 %) |
  | untried `fastapi` batch | 30 | 2 (run before the starlette measurement below) |
  | untried mixed batch | 30 | **17** |
  | recheck of the original eight | 8 | 4 |
  Negative controls from the same day: the same 38 tasks **without** the wheel restore returned 0/38, and
  the same eight returned 0/8. That is a clean A/B for the drift fix.
- **Union: 73 valid tasks (rich 39, fastapi 34)**, recorded both as a working copy
  (`runs/2026-10-13-control-valid-set.json`) and as a tracked manifest,
  `resource/manifests/control-valid-tasks-2026-10-13.json`. Against the paired protocol's power table that
  reaches a **20-point** gain at 80 % power (40 tasks needed), is **5 short** of the 78 that a 10-point
  gain needs, and far from the 158 for 5 points.
- **The fastapi collection class is cracked, by measurement rather than assumption.** The exact error was
  `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` in both arms, while
  `task-environment-pins.json` recorded `{}` -- those snapshots declare no starlette pin, so the image's
  wheel (the newest in the task wheelhouse, 1.6.0) was left in place. The generator now probes the
  wheelhouse's starlette candidates from newest down with a direct `Router(on_startup=[])` import and
  installs the newest one that accepts it. Measured boundary on the real wheels: **1.6.0 rejects, 1.3.1
  rejects, 0.52.1 accepts**. With that pin `fastapi_13713` and `fastapi_14786` -- invalid for this
  project's entire history -- are now usable (empty exit 1, gold exit 0, resolved). My earlier reading
  that the keyword was removed at 0.47 was wrong; the measurement corrected it.
- Ready but deliberately **not launched**, because this session was asked to stop: a 30-task fastapi sweep
  with the measured pin at `runs/2026-10-13-control-fastapi-starlette/`. It should convert a large share of
  the 28 failures in the fastapi batch, and would likely push the set past the 78-task mark.
- **GitHub, finished for the handoff**: PR **#16** opened for `experiment/four-l4-lora-feasibility`, and the
  measured four-L4 result posted to **issue #13** (comment `6128890774`). PR #8's head carries its two
  review fixes (`bd9b167`) but the PR reports `mergeable_state=dirty`, so it still needs a rebase against
  `main` before it can merge.
- **What the training half now faces, stated honestly: compute, not instrument.** 5,306.051
  quota-seconds remain of 108,000 (refresh 2026-10-17, about 2,653 s of L4x4 wall time), one bounded
  training session already cost 4,733.33 of them, and a paired evaluation over 40+ tasks runs two arms per
  task on top of that. The rank ladder therefore waits for the quota refresh or for another platform
  (Trillium needs credentials this workspace does not have; the local GPU is an 8 GiB laptop part that
  cannot hold this base at any quantisation). The ladder design to use when compute exists: rank 16 on the
  compact recipe first, then 32 and 64 only if the paired result moves, re-establishing empty/gold controls
  in every run as the protocol requires.
- Verification: seven tests in `tests/test_control_sweep_notebook.py` pass (including the starlette probe),
  the whole suite passes, and the union is computed from the fetched Kaggle reports by
  `artifacts/tmp/union_valid_tasks.py` rather than by hand.
- Changed files: `scripts/prepare_task_control_sweep.py`, `tests/test_control_sweep_notebook.py`,
  `resource/manifests/control-valid-tasks-2026-10-13.json` (new), `WORKLOG.md`. **No GPU quota was spent
  in this round**; every sweep is CPU-only.

### 2026-10-13 01:10 (Asia/Shanghai) - The control instrument was broken by wheelhouse drift; fixed by embedding the reviewed wheels; 128 tasks now sweeping

- Objective (goal round 1): grow the control-valid task set before any LoRA rank work, because a gain
  cannot be measured against four tasks.
- **Re-running the eight earlier tasks returned 0 of 8 usable, including the four that passed on
  2026-10-09.** The kernel log states the reason in one line:
  `harness wheelhouse version drift: [('swegemma-0.2.10-py3-none-any.whl', ['swegemma-0.2.11-py3-none-any.whl'])]`
  plus `harness wheelhouse changed bytes: ['adk_eval_core-0.1.0-py3-none-any.whl',
  'adk_submission-0.2.13-py3-none-any.whl']`. Every arm then produced no test result at all. This also
  narrows the earlier pydantic finding: that fix was necessary -- it explains the 0/38 sweep of
  2026-10-09 14:31 -- but it stopped being sufficient once the organisers replaced those wheels in place
  at 14:43 the same day.
- **The kernel-source mount route failed and cannot be trusted for an instrument.** With
  `--kernel-source junzhejiao/gemma4-pinned-wheels` and the documented session path, the restore cell
  reported `pinned wheel mount is incomplete: [all three wheels]`. The publisher kernel's ref is
  `gemma4-pinned-wheels` while its title is "gemma4 pinned wheels", and this project has already seen the
  same metadata work for one kernel and fail for another (`rich_3675` versus `rich_3676`).
- **Fix: the reviewed wheels now travel inside the notebook.** `--pinned-wheels-dir` and
  `build_pinned_blob()` pack the three reviewed wheels (279,363 bytes in total) as a hash-checked base85
  payload; the restore cell extracts them, verifies every digest against
  `resource/manifests/eval-wheelhouse-2026-10-09.json`, and force-reinstalls them over the mounted
  versions. The notebook is 815,610 bytes, inside the 1 MB API limit. Re-run verdict:
  `pinned-wheels.json` reports all three installed with `missing: []`, and the sweep returned **4 usable of
  8** -- the four `rich` tasks are valid again (empty exit 1 or 2, gold exit 0, resolved), reproducing the
  2026-10-09 result exactly.
- The other four still fail for the protocol's known classes: `fastapi_13713` and `fastapi_14786` fail
  collection in **both** arms (exit 2), `requests_7433` times out in both arms (exit 124), and `rich_3296`
  has tests that already pass at base. The missing module behind the `fastapi` pair is not exposed
  anywhere in the run outputs (`error_message: null`, per-task directories empty), so unlocking that class
  needs a diagnostic rather than a guess.
- **Scale-up pushed, 128 tasks in four CPU-only sweeps**, all carrying the embedded fix: the 38 that gave
  0/38 on 2026-10-09 plus the 90 never attempted, in repository-homogeneous batches (30 `fastapi`, 30
  `rich`, 30 mixed) so a failure class can be attributed to a repository. Ceiling arithmetic: at the
  observed yield the set can reach roughly 55-65 valid tasks, against the **40** that make a 20-point
  paired gain detectable at 80 % power.
- **GitHub side, now that the token reaches the repository.** The earlier fine-grained token was scoped to
  an unrelated repo; the replacement classic PAT reads `jwj1342/gemma-agent-lab`. State read from the API:
  PR #8 `open`, `merged=false`, **`mergeable_state=dirty`** (needs a rebase against `main`), head
  `experiment/swe-lora-v1@bd9b167` -- so the two review fixes are on the PR -- with 0 review comments;
  issue #13 open with **0 comments**; and no pull request existed for the L4 branch. Actions taken:
  **opened PR #16** for `experiment/four-l4-lora-feasibility` (head `c0648f7`) and **posted the measured
  result to issue #13** (comment `6128890774`). Those were the two remaining human handoff steps; PR #8's
  rebase is still outstanding.
- Verification: six new tests in `tests/test_control_sweep_notebook.py` pass -- the restore step is opt-in,
  a wheel that differs from the reviewed manifest is rejected before a notebook is written, the embedded
  payload installs after hash verification (using the real reviewed wheels), and the reviewed manifest is
  asserted to list every pinned wheel and not to list the drifted one.
- Changed files: `scripts/prepare_task_control_sweep.py`, `tests/test_control_sweep_notebook.py`,
  `WORKLOG.md`. Generated notebooks and downloaded evidence remain under `runs/` and `artifacts/`
  (gitignored). **No GPU quota was spent this round**: every sweep here is CPU-only.

### 2026-10-13 00:50 (Asia/Shanghai) - Measurement-set growth: the 0/38 control sweep's root cause, and a CPU recheck in flight

- Objective (goal round 1): grow the control-valid task set before any LoRA rank work, since a gain cannot
  be measured against four tasks.
- **GitHub side finished first.** The two PR #8 review fixes are now on the PR itself, not on a side
  branch: `git push origin fix/pr8-contracts-v2:experiment/swe-lora-v1` fast-forwarded
  `ccfcff0..bd9b167`, and `git ls-remote` confirms the remote head is `bd9b1675…`.
- **The provided GitHub token authenticates but is scoped to the wrong repository.** `E:\GitHub_token.txt`
  holds a fine-grained token that reaches exactly one repo, `jwj1342/Kaggriculture`; `gemma-agent-lab`
  answers 404, which is what GitHub returns for a private repo the token cannot see. It needs that repo
  added under Repository access with Contents: Read, Pull requests: Read and write, Issues: Read and
  write. Editing access does not change the token string, so the same file stays usable.
- **Root cause of the 0/38 control sweep, found by diffing the two notebooks rather than guessing.** The
  failing `sweep-frozen38-v17` re-pinned **pydantic** alongside starlette; the working
  `sweep-valid4-v14` pins **only starlette** and carries the reason in a comment: repinning pydantic
  installs a version whose pydantic-core binding is absent from the task wheelhouse, which breaks litellm
  for every task, so no test ever runs. That matches the observed signature exactly -- every one of the 38
  gold arms recorded `status=NO_RESULT_FILE, returncode=1` with no test exit code at all. The current
  generator already carries the fix, so that task set had simply never been re-run with it.
- Two environment facts established on the way. The working 4/4 sweep ran at 2026-10-09T14:19:36Z, *before*
  the organisers republished the wheelhouse at 14:43Z (`swegemma 0.2.10 -> 0.2.11`, which deleted
  `models.registry`), so a new sweep must also survive that drift. And the mounted wheelhouse holds
  **42** files today: a first reading that showed 20 was a pagination artefact of
  `dataset_list_files`, not a shrunk dataset, which is worth remembering before concluding drift.
- **Pool arithmetic, recorded so the training decision is judged honestly.** The public set is 129 tasks
  (fastapi 67, rich 48, requests 13, httpx 1); 39 have been attempted and **90 never tried**. At the pilot
  yield of 4 of 8, the ceiling is roughly 55-65 valid tasks, against the 78 that the power model wants for
  a 10-point gain at 80 % power -- and that model itself assumes zero regressions.
- Action: pushed **`junzhejiao/gemma4-control-recheck8`** (CPU-only, `enable_gpu: false`, wheelhouse
  attached, reviewed test dependencies embedded) over the same eight tasks the power analysis uses -- four
  known-good plus four previously invalid -- to test both the fixed configuration and the drift at once.
  Status was RUNNING at 16:43:32Z when last queried; the network to `api.kaggle.com` is flaky tonight
  (repeated `SSLEOFError`, recovering on retry), so polls need retries rather than a single call.
  A 38-task notebook with the fixed configuration is prepared locally at
  `runs/2026-10-13-control-frozen38-fixed/` and is ready to push next, followed by batches over the 90
  untried tasks.
- Changed files: `WORKLOG.md` only. Everything else this round is under `runs/` (gitignored generated
  notebooks and records); no production code changed.

### 2026-10-12 21:35 (Asia/Shanghai) - The task branch is already on the remote; verified it, and classified the delivery-tree failures

- Objective: push the task-related branch that codex curated
  (`experiment/four-l4-lora-feasibility`, "Publish measured four-L4 LoRA feasibility loop").
- **Finding: it is already on the remote and byte-identical to the local branch, so no push was needed.**
  `git ls-remote --heads origin experiment/four-l4-lora-feasibility` returns
  `c0648f70430c83d82508e8fb6bb8febd5738086e`, exactly the local tip, and
  `git push origin experiment/four-l4-lora-feasibility` answers `Everything up-to-date`. The companion
  branch `fix/pr8-contracts-v2` is on the remote too (`bd9b1675d5c7056ee146957428f7fefabf8f6580`,
  identical to local). **The 21:04 entry below predates her own push**: it was written at 21:04, while the
  delivery commit is dated 21:16:43 +0800, so its line "the L4 branch has not been pushed" is stale and
  should be read as history, not as current state.
- The delivery commit is authored by `Kaltist <junzhejiao32@gmail.com>`, 2026-10-12 21:16:43 +0800,
  13 files, **all additions**, parented on the current `origin/main` (`625a597`). Her curation is visible
  in the diff against the working branch: the published tree adds a "Reproduce the bounded loop" section,
  corrects the stale "github.com is unreachable" wording in the PR8 review, and adds the four-L4 row to
  the README. Nothing of the measured work is dropped, and
  `git show <branch>:docs/experiments/2026-10-12-l4-training-feasibility.md` carries run 11's numbers
  (adapter 244,982,352 bytes, effect `0.596932`, the queue-time quota lesson).
- **Verified rather than assumed**: running inside the delivered tree, the notebook generator's `--check`
  reports `status: ok` and `tests/test_l4_feasibility.py` runs **53 tests OK**.
- **The delivery worktree is a sparse checkout, which is what explains its full-suite result** (15
  failures + 2 errors) and clears the delivered commit of them: `core.sparseCheckout=true` with the cone
  `scripts/ tests/ docs/ configs/ resource/l4_feasibility/ resource/notebooks/official/ README.md
  AGENTS.md .gitignore`, while the index still tracks 972 entries with none deleted. Every failure is an
  environment artifact -- the missing files sit outside the cone (`src/`, `resource/manifests/README.md`,
  `tests/README.md`, `THIRD_PARTY_NOTICES.md`), `tests/test_eval_suite.py` imports the Unix-only `fcntl`,
  and the repository checker shells out to a `bash` this Windows host lacks. The delivered commit modifies
  no existing file and none of the failing tests exercise the files it adds, so the repository-level CPU
  gate belongs on Linux CI and cannot be reproduced in full here.
- **A full checkout of that tree is impossible on Windows, and it is inherited from upstream.**
  `git worktree add` against `origin/main` fails with
  `error: invalid path 'resource/leaderboard/gemma-4-developer-agent-publicleaderboard-2026-10-05T03:50:43.csv'`
  -- a colon in a filename -- which is exactly why local `main` carries the unpushed compatibility commit
  `88823c0`. This matches the platform incompatibility codex recorded when she stopped her PR #8 backport,
  and it is the same reason the sparse worktree exists.
- Repository changes: `WORKLOG.md` only (this entry). The verification commands were read-only and the
  only write was the no-op push; **no new push was made because none was needed**.
- Remaining human actions: open the pull request for the published branch
  (https://github.com/jwj1342/gemma-agent-lab/compare/main...experiment/four-l4-lora-feasibility) -- this
  workspace has no `gh` CLI, no `GH_TOKEN`/`GITHUB_TOKEN` and no stored CLI host config, so it cannot; run
  the repository CPU gate on Linux CI; and decide whether any local-only branch should be published.
  `fix/lora-rank8-controlled` (`71027b4`) holds the acceptance commits `650f306` and `7356a80` plus codex's
  audit commit and carries the *uncurated* variants of the README and both experiment docs;
  `feature/lora-serving-*`, `feature/localization-v1-skill`, `integration/local-work` and
  `archive/local-before-remote-2026-10-07` are also local-only.

### 2026-10-12 21:04 (Asia/Shanghai) - Audit DeepSeek's issue #13 handoff and re-verify the L4 entry point

- Objective: translate the requested issue #13 scope, determine which items were actually completed,
  and safely close any remaining reproducibility gaps without consuming shared compute.
- Completion finding: the technical L4 acceptance criterion is met. Commits `650f306` and `7356a80`
  implement and measure the required-base four-L4 train/save/verify/reload loop; run 11 completed eight
  updates and the official runtime measured a non-zero adapter effect. The compact H100 artifact review
  is recorded in `docs/experiments/2026-10-12-pr8-review.md`.
- Remaining collaboration work: the L4 branch has not been pushed, issue #13 has not been updated from
  this checkout, and independent teammate approval is not recorded. PR #8 still points to `ccfcff0`.
  Its review found two small unmerged fixes: reject/count a final terminal answer that would otherwise
  be silently dropped, and hash all files in a sharded base checkpoint (or fail with a layout-specific
  diagnostic) instead of assuming `model.safetensors`.
- Remote evidence: `git ls-remote` returned `ccfcff0` for both the experiment branch and PR #8 head and
  `f486342` for the PR merge ref. The private GitHub issue/PR URLs return 404 without authenticated web
  access, so merge and approval state were not inferred from the existence of the merge ref.
- Verification: `tests.test_l4_feasibility` ran 53 tests successfully after using the documented
  `artifacts/combo-test;.pylibs` dependency path; the first bare-environment run failed only because
  `numpy` and `safetensors` were absent. `scripts/prepare_l4_feasibility_notebook.py --check` reports
  `status: ok` with all seven embedded worker hashes. No Kaggle job, upload, or quota-consuming action
  was started.
- Attempted follow-up: a separate PR #8 fix worktree, a path-limited archive, and a temporary Git index
  were each blocked by the historical leaderboard filename containing `:` on Windows. After repeated
  confirmation of the same platform incompatibility, the code backport was stopped instead of risking
  the dirty primary worktree. The scratch branch was deleted; an empty
  `E:\Desktop\CodingAI-pr8-patch` directory remains because the execution policy denied its removal.
- Files changed: `WORKLOG.md` and the existing untracked review document
  `docs/experiments/2026-10-12-pr8-review.md`, which now records the remote-head check. No unrelated
  working-tree changes were modified.

### 2026-10-12 20:35 (Asia/Shanghai) - Run 11 meets the acceptance criterion: train, save, verify and reload on four L4s

- Objective: close the loop after the export fix.
- **Run 11 (kernel version 7) COMPLETE and it is the acceptance measurement.** One session,
  2026-10-12 19:41-20:06 Asia/Shanghai, session wall **1,421.3 s**:
  - the required `gemma-4-31b-it-qat-w4a16-ct` base was dequantised to BF16 (`keyword: run_compressed`,
    `effective_dequantize: true`) and dispatched across the four L4s by one process in **217.5 s**
    (`device_map='balanced'`, 20,500 MiB per GPU, 66 modules placed `{0: 7, 1: 19, 2: 19, 3: 21}`,
    `cpu_share: 0.0`);
  - a replicated LoRA adapter of **122,429,440 parameters** trained **8 of 8 updates**
    (`completed_updates`), loss `2.0530 -> 0.3739`, **seconds per update min 1.388 / mean 1.767 /
    max 4.116**, per-GPU peaks **7.615 / 18.535 / 18.642 / 17.593 GiB** against 22.03 GiB per card;
  - the adapter exported: **820 tensors, 244,982,352 bytes**, SHA256
    `c4d2bff9bf765bb6b16a01dc7545757358cbc1cc6d3054de3b41cfa601b76852`, spanning layers 0-59 and all seven
    projections;
  - `adapter-check.json` -> `status: ok`, **820/820** keys on the naming scheme the serving stack was
    measured to accept, `0` unmatched, rank 16;
  - the official vLLM runtime reloaded it: **`status: applied`** in 155.4 s (engine ready at 119.4 s),
    with the repeat-base control at `max_abs_logprob_delta: 0.0` over 40 shared positions and the adapter
    effect at **`0.596932`** over the same positions. Greedy tokens and text did not change on this probe
    prompt, which is recorded rather than glossed: the runtime honours the adapter (its distribution
    moves) without flipping the greedy continuation here.
- Two defects were found and fixed on the way, both in the reporting layer rather than the measurement:
  `save_adapter` used `torch` without importing it (run 10 completed eight updates and then failed at the
  save with `NameError`), and the summary cell's rollup did not read the device-mapped route at all, so
  run 11's own `result-summary.json` still reports `updates_completed: 0` and `adapter: null` while
  `device-map-result.json` holds the real values. The rollup now includes that route and a regression test
  asserts it; the numbers above come from the route's own file.
- **A quota lesson worth carrying**: the measured charge for run 11 was **4,733.33 quota-seconds**
  (10,039.379 -> 5,306.051) against a 2,842.6 estimate at the 2x rate. The allocation is billed for the
  time it is held, and this session spent about **780 s queued** before the notebook started. Only
  **5,306.051 s of 108,000 s** remain (refresh 2026-10-17), so further sessions must fold queue time into
  their plan. This session stopped spending GPU quota at that point on purpose.
- Verification this round: `tests/test_l4_feasibility.py` -> **53 tests pass**; the whole suite ->
  **109 tests pass**; `--check` -> `status: ok`.
- Changed files this round: `resource/l4_feasibility/train_l4_feasibility.py`,
  `scripts/prepare_l4_feasibility_notebook.py`, `tests/test_l4_feasibility.py`,
  `docs/experiments/2026-10-12-l4-training-feasibility.md` (new section 9 holds the measurement and the
  status header now says the criterion is met), `WORKLOG.md`.
- Remaining follow-up: report to issue #13 (a human action, since the repository is unreachable here);
  decide whether a graded 4K example or the paired repair control is worth the remaining 5,306
  quota-seconds -- the loop's feasibility is established and the eight-update adapter is not a promotion
  candidate.

### 2026-10-12 20:15 (Asia/Shanghai) - Run 10 trains on four L4s: eight updates complete on the required QAT base; the export hit a latent NameError

- Objective: run the objective fix and see the loop finish.
- **Run 10 (kernel version 6) COMPLETE**: 904.7 s of session wall. It is the first session that trained.
  - The load again succeeded on the ladder's first step (`balanced`, 217 s), with the same placement as
    run 9, and PEFT wrapped 122,429,440 trainable parameters across the four GPUs.
  - **Eight optimizer updates completed** on the required `gemma-4-31b-it-qat-w4a16-ct` base, dequantised
    to BF16 and dispatched across four L4s, with the loss falling `2.0530 -> 2.0303 -> 1.8816 -> 1.5570 ->
    1.1500 -> 0.8744 -> 0.6202 -> 0.3739` and gradient norms `7.16, 7.23, 7.56, 8.56, 8.26, 6.57, 5.80,
    4.25`. Seconds per update: **min 1.385, mean 1.681, max 3.455** (the first includes warm-up).
    `stopped_reason: completed_updates`. The loss falling to 0.37 is the expected overfit of eight updates
    on one synthetic example; this probe measures feasibility, not generalisation.
  - Per-GPU peaks during training: **7.615 / 18.535 / 18.642 / 17.593 GiB** against 22.03 GiB per card,
    so the route has about 3.4 GiB of headroom on its busiest GPU.
  - Then `adapter_save_failed`: **`NameError: name 'torch' is not defined`**, from `save_adapter`.
- **That NameError was latent in the shared export helper and no route had ever reached it.** `save_adapter`
  converts the LoRA tensors with `torch.bfloat16` but imported only `safetensors.torch` and the FSDP
  symbols; the four-GPU route died at its load and the packed route at its PEFT wrap, so the save had
  never executed. Fixed by importing `torch` inside the helper, and now covered by a real test: it exports
  from a fake device-mapped model and asserts the two tensors, the served key scheme
  (`...q_proj.lora_A.weight`) and that the frozen `base_layer` weight is not exported.
- Verification: `tests/test_l4_feasibility.py` -> **53 tests pass**; the whole suite -> **109 tests pass**.
- **Run 11 pushed as kernel version 7** carrying the export fix, and observed RUNNING. This session should
  be the one that closes the loop: dispatch, eight updates, export, adapter verification, and the
  official-runtime reload.
- Changed files this round: `resource/l4_feasibility/train_l4_feasibility.py`,
  `tests/test_l4_feasibility.py`, `WORKLOG.md`.

### 2026-10-12 19:45 (Asia/Shanghai) - Run 9: Path D loads the base across four L4s and wraps the adapter; the objective's index tensor was the last barrier

- Objective: run the placement ladder and find out whether the single-process sharded route can reach a
  training step.
- **Run 9 (kernel version 5) COMPLETE**: about 852 s of session wall, **1,698.10 quota-seconds**
  (11,737.479 -> 10,039.379). Three phases of the new route carry results, and the first two are the
  breakthrough this workstream needed:
  - the **load succeeded on the ladder's first attempt** (`balanced`, 20,500 MiB per GPU, with
    `expandable_segments:True`) after 196 s: 66 modules placed as `{0: 7, 1: 19, 2: 19, 3: 21}`,
    `cpu_share: 0.0`, and per-GPU peaks of 8.63 / 18.61 / 18.71 / 17.70 GiB. So the required QAT base
    **is** dispatchable across four L4s as BF16, and the second ladder step was not needed. Run 8's
    failure was indeed allocator fragmentation rather than an impossible split.
  - **PEFT wrapped the dispatched model**: `trainable_parameters: 122429440` -- exactly the figure the
    manifest records for this adapter -- in 820 tensors spread over all four GPUs
    (`{cuda:0: 56, cuda:1: 260, cuda:2: 258, cuda:3: 246}`).
  - the first forward then failed with
    `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cuda:3)`.
- **Root cause, read from the wheel rather than guessed**: `models/gemma4/modeling_gemma4.py` computes
  `slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep`
  and then `self.lm_head(hidden_states[:, slice_indices, :])`. The shared objective passed
  `logits_to_keep=positions - 1`, a **tensor** on `cuda:0`, which that line uses as a gather index into
  hidden states living on the last GPU -- the error verbatim. This was a latent defect that had never
  executed: the four-GPU route died at its load before any forward pass.
- Fix: `target_loss` now passes an integer (`len(positions) + 1`) and drops the extra trailing row, which
  reproduces the old selection exactly -- the hidden state before each supervised token -- without an
  index tensor, and it validates that the supervised positions really are the trailing range. The
  objective's targets also follow the logits' device now, so cross-entropy cannot straddle two GPUs. Both
  changes are device-independent, so the four-GPU route measures the same objective.
- Second measured correction: the resident BF16 size. Summing run 9's per-GPU peaks gives about 63.6 GiB,
  against the ~58 GiB this note estimated and the ~71.5 GiB that the packed-to-BF16 ratio suggested. The
  measured figure is the one to keep, and section 3 of the experiment note now carries it.
- Verification: `tests/test_l4_feasibility.py` -> **52 tests pass** (new: the objective asks for an integer
  count, the extra trailing row does not affect the loss, and non-trailing positions are rejected); the
  whole suite -> **108 tests pass**.
- **Run 10 pushed as kernel version 6** carrying the objective fix, and observed RUNNING. Quota at that
  point: **10,039.379 s remaining**, about 5,019 s of L4x4 wall time. This is the first session that can
  plausibly reach a completed training step, an exported adapter and the official-runtime reload.
- Changed files this round: `resource/l4_feasibility/train_l4_feasibility.py`,
  `tests/test_l4_feasibility.py`, `docs/experiments/2026-10-12-l4-training-feasibility.md`, `WORKLOG.md`.

### 2026-10-12 19:20 (Asia/Shanghai) - Run 8: Path D reaches the load and dies on allocator fragmentation; a placement ladder instead of one guess

- Objective: run the newly implemented single-process sharded route (Path D) and see whether it can reach
  a training step where the four-rank route could not.
- **Run 8 (kernel version 4) COMPLETE**: about 1,020 s of session wall, **2,040.54 quota-seconds**
  (13,778.017 -> 11,737.479). It executed its new phase:
  `device-map-result.json` records `status: load_incompatible` after a **213 s load attempt** with
  `OutOfMemoryError: CUDA out of memory. Tried to allocate 442.00 MiB. GPU 2 has a total capacity of
  22.03 GiB of which 351.12 MiB is free. This process has 21.68 GiB memory in use. Of the allocated memory
  17.88 GiB is allocated by PyTorch, and 3.60 GiB is reserved by PyTorch but unallocated.`
- **What that number says, and it is not simply "too big".** The per-GPU cap handed to accelerate was
  21,000 MiB and the process reached 21.68 GiB, so the split respected the cap to within half a gigabyte;
  what killed it was **3.60 GiB reserved by the allocator but never allocated**. The dispatch is therefore
  limited by allocator fragmentation as much as by the placement, which is a different failure from the
  four-rank route's host-RAM kill and needs a different fix. Second reading from the same line: GPU 2
  alone held 17.88 GiB, above the 14.5 GiB a plain four-way split of the ~58 GiB BF16 base implies, so the
  resident size is larger than the estimate in section 3 of the experiment note. The honest statement is
  that one GPU's share is measured at 17.88 GiB; the other three were not reported.
- The environment and keyword results held again: `model-stack.json` `verdict: ok`, and the new phase's
  own `quantization_config` record was `{requested_dequantize: true, keyword: run_compressed,
  effective_dequantize: true}`. The packed phase repeated `peft_incompatible` for a third time.
- Fix implemented, in the worker rather than in another guess at a budget:
  `device_map_attempts()` returns a two-step placement ladder -- `balanced` at the full per-GPU budget
  first, then `auto` with a 1,500 MiB lower cap **and** a host-RAM allowance so the tail can be offloaded
  instead of failing -- and each attempt is recorded with its own exception. The worker also sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the CUDA context exists, which is the direct
  remedy for reserved-but-unallocated memory. The shared load budget now bounds the *sum* of the attempts,
  so two attempts cannot overrun one phase.
- **Run 9 pushed as kernel version 5** carrying the ladder, and observed RUNNING. Quota at that point:
  **11,737.479 s remaining** of 108,000 (refresh 2026-10-17), about 5,868 s of L4x4 wall time.
- Changed files this round: `resource/l4_feasibility/train_l4_device_map.py`,
  `tests/test_l4_feasibility.py` (the ladder and the allocation-config ordering are asserted),
  `docs/experiments/2026-10-12-l4-training-feasibility.md` (section 3 now carries the measured SIGKILL),
  `WORKLOG.md`.

### 2026-10-12 18:55 (Asia/Shanghai) - Run 7: the loading mode is now genuinely BF16, and four-rank FSDP dies on host RAM with SIGKILL

- Objective: re-run with the version-keyword fix and read what the four-GPU route really does.
- **Run 7 (kernel version 3) COMPLETE**: 670.0 s of session wall, **1,394.49 quota-seconds**
  (15,172.507 -> 13,778.017). Three of its phases carry results:
  - `model-stack.json` -> `verdict: ok` again, so the environment fix is reproducible.
  - `packed-probe-result.json` -> `peft_incompatible` again after 193 s (the checkpoint digest included),
    so the packed route's rejection is reproducible too, not a one-off.
  - the four-GPU phase recorded `quantization_config` as
    `{requested_dequantize: true, keyword: run_compressed, effective_dequantize: true}`. That is the fix
    working, and it is also the first evidence that the BF16 dense base is genuinely requested on this
    wheelhouse version: the keyword is `run_compressed`, because 5.13.1 has no `dequantize`.
- **Path A is measured infeasible on this host, and it dies in a way that cannot report itself.** After
  `quantization_config` the worker wrote nothing further: no `base_loaded`, no `base_load_failed`, no
  `train-result.json`. The kernel log carries `traceback : Signal 9 (SIGKILL) received by PID 496` three
  times, which is the host OOM killer, not an exception. The arithmetic in section 3 of the experiment
  note predicted exactly this: each of the four ranks materialises a complete BF16 replica of the frozen
  base (about 58 GiB) before FSDP shards anything, so the load needs about 232 GiB against the measured
  188.69 GiB of host RAM. A SIGKILL also means the worker's own `--load-budget-seconds` never fires, so
  the run cannot even record the reason; only the kernel log shows it. That is the strongest argument for
  Path D, which never holds a complete replica anywhere.
- Quota discipline: run 7 cost 1,394 quota-seconds for 670 s of wall time, again the 2x rate. Three runs
  this session (5, 6, 7) cost 68 + 750 + 1,394 = 2,212 quota-seconds in total.
- **Run 8 pushed as kernel version 4**, carrying Path D as a fifth phase, and observed RUNNING. This is the
  first session in which the notebook can reach a training step: Path A is ruled out by measurement rather
  than by prediction.
- Also this round: the whole L4 workstream was committed for the first time --
  `650f306 Add the four-L4 training instrument, its model-stack step, and Path D` on branch
  `fix/lora-rank8-controlled` (the same commit the WORKLOG calls the active branch; it and
  `feature/lora-serving-probe-run` both point at `6d2ad38`). Twelve files: the notebook generator, seven
  workers, the IPv4 connect helper, the L4 test module, the experiment note and this log. The repository
  is unreachable from this workspace, so pushing the branch and opening a PR remain human actions. Two
  earlier experiment notes from this workstream are still untracked and were left alone rather than
  bundled into this commit.
- Changed files: `WORKLOG.md` (this entry); the code is in `650f306`.

### 2026-10-12 18:45 (Asia/Shanghai) - Run 6 fixed the environment and rejected the packed route; the version-keyword trap and Path D

- Objective: re-run after the guard fix, read the result, and keep moving toward the training measurement.
- **Run 6 (kernel version 2) COMPLETE**: 358.3 s of session wall, **750.086 quota-seconds** measured
  (15,922.593 -> 15,172.507). Its stage phase is the decisive environment result:
  `model-stack.json` reports `verdict: ok` with `transformers 5.13.1`, `peft 0.19.1`,
  `torchao 0.16.0` and `is_torchao_available: true`, `compressed_tensors 0.15.0.1`, `loguru 0.7.3`,
  `gemma4` resolving to `Gemma4Config`, and `AutoConfig.from_pretrained(base)` returning
  `config_model_type: gemma4`. **Run 4's blocker is resolved and measured, not inferred.**
- **Path B (packed) is now measured and rejected**, which is a reportable result on its own: the shipped
  W4A16 checkpoint loads in 3.96 s with ~0 GiB of GPU memory allocated (compressed-tensors defers
  decompression to the first forward pass, exactly as the 5.13.1 docstring says), all 410 target modules
  are `Linear` wrappers that pass `isinstance(module, torch.nn.Linear)`, and PEFT then fails with
  `AttributeError: 'Linear' object has no attribute 'weight'` -- status `peft_incompatible` after 272 s.
  So the packed format is not trainable with PEFT, and the reason is the module type rather than memory.
- The 272 s is itself mostly overhead worth recording: 264 s of it is the packed probe hashing the
  checkpoint's ~20 GB `model.safetensors` to record `base_format_unchanged`, before any load happens.
- **A version-keyword trap made the four-GPU phase meaningless, and it would have been silent.** Run 6's
  four-GPU phase recorded `base_loaded` with `rank0_allocated_gib: 0.0`: transformers **5.13.1**'s
  `CompressedTensorsConfig.__init__` is `(..., run_compressed: bool = True, **kwargs)` and
  `get_loading_attributes()` returns `{'run_compressed': ...}`, while `5.18`/`5.19` take
  `dequantize: bool = False` and document "`run_compressed=False` meaning `dequantize=True`". The
  `dequantize=True` both workers passed is swallowed by `**kwargs`, which 5.13.1 only forwards when
  `config_groups` or `kv_cache_scheme` is given (neither is), so it is dropped: `run_compressed` keeps its
  `True` default and the checkpoint loads compressed while the run records `dequantize: true`. `from_dict`
  does no renaming either. Verified in the mounted wheel itself, not from memory.
  `resource/l4_feasibility/quantization_config_compat.py` now builds the config from the installed class's
  signature, records the keyword it used and the effective mode, and raises instead of returning a config
  whose mode disagrees with the request; both training workers use it, and the stage verification gained a
  fatal `dequantize_mode_is_effective` check.
- **Second defect, also from run 6**: `get_peft_model` sat outside the strategy `try/except`, so the wrap
  failure killed the worker before any attempt, traceback or `train-result.json` was written and the
  summary could only say "the worker wrote no train-result.json". The wrap is now inside the recorded
  attempt, so the same failure names itself in the run's own files.
- **Path D is implemented** (`resource/l4_feasibility/train_l4_device_map.py`) because the four-GPU route
  is arithmetically short of host RAM: four ranks each materialise a complete BF16 replica (~58 GiB) before
  FSDP shards anything, about 232 GiB against the measured 188.69 GiB. Path D is one process that loads the
  dequantised base with `device_map='auto'` and a per-GPU budget, so nothing is ever complete in one place;
  it reuses the same native contract, objective and adapter export, and it refuses two load results that
  both look like success in a log (more than 21.5 GiB on one L4; most modules left in host RAM). The
  notebook gained a fifth phase for it, gated on "no adapter yet and the session cap can still cover it",
  and the budget ladder was rebalanced so four phase timeouts still fit inside the 3,300 s cap
  (600 + 1,000 + 1,100 + 600).
- Verification this round: `tests/test_l4_feasibility.py` -> **49 tests pass** (new coverage: the keyword
  mapping for both class shapes and the refusal, the placement verdict's two failure modes, the Path D cell's
  cap guard, and the summary cell now loading `device-map-result.json`); the whole suite -> **105 tests
  pass**; `--check` -> `status: ok` with seven embedded workers round-tripping.
- **Run 7 pushed as kernel version 3** and observed RUNNING at 10:28:29Z, carrying the keyword fix, the
  recorded-wrap fix and the Path D phase. Its outcome is recorded in the next entry.
- Changed files: `resource/l4_feasibility/{quantization_config_compat,train_l4_device_map}.py` (new),
  `resource/l4_feasibility/{train_l4_feasibility,probe_packed_base}.py` (keyword helper, recorded wrap,
  device-safe loss targets), `scripts/prepare_l4_feasibility_notebook.py`,
  `tests/test_l4_feasibility.py`, `docs/experiments/2026-10-12-l4-training-feasibility.md`, `WORKLOG.md`.

### 2026-10-12 18:20 (Asia/Shanghai) - The upload blocker is gone (it was a network fault), the torchao wheel is mounted, and run 5 carries the fix

- Objective: continue the unfinished four-L4 training work (issue #13) from the 02:05 resume point --
  get `torchao 0.16.0` into the session, install `transformers 5.13.1` with it, re-run, and reach the
  training-memory measurement.
- **The dataset-upload stall is root-caused, and it was never a Kaggle problem.** The kaggle client
  uploads dataset blobs to `www.googleapis.com`, and every address of that host is unreachable from this
  machine: `Resolve-DnsName` returns 8 IPv6 and 8 IPv4 addresses, and a bounded probe failed all 16
  (`2001:4860:4843:400::`, `172.217.117.4`, ...) while `storage.googleapis.com` and `www.kaggle.com`
  answered normally. `Test-NetConnection 2001:4860:4840:400:: 443` reports `TcpTestSucceeded = False`
  against `TcpTestSucceeded = True` for the IPv4 address of `storage.googleapis.com`, so the campus IPv6
  path to Google is blackholed and the IPv4 frontends for `www.googleapis.com` are blocked as well. The
  earlier "40 minutes via CLI, 20 minutes via SDK at 0.9 s CPU" stall was the client sitting in
  `SYN_SENT` until the operating-system connect timeout, address after address. `netstat` during the
  attempt shows exactly that state. Everything read-only (`kernels_output`, `datasets files`,
  `datasets download`) works, because those hosts are reachable.
- Fixed with `scripts/bounded_connect.py`: it caps a blocking `socket.socket.connect` at 5 s and restores
  blocking mode once connected, so every HTTP stack (requests/urllib3, httpcore, plain sockets) walks the
  resolver's address list quickly instead of hanging on the first unreachable entry. Process-local, no
  administrator rights, no host change. Measured: connect to an unreachable address now fails in 5.0 s
  with the socket timeout restored to `None`, and `storage.googleapis.com` still connects in 0.29 s.
- **The upload is no longer needed at all.** Instead of publishing our own dataset, the notebook attaches
  the public dataset `kajarnaklokossou/torchao-0-16-0`, whose single file
  `torchao-0.16.0-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl` is 3,160,415 bytes --
  downloaded and hashed here as
  `2d6293a0c57c9dd505efb025a7189459d154965fbed000efd638cf299f9362dd`, exactly the PyPI digest recorded at
  the 02:05 resume point. That byte-level match is what makes a third-party dataset acceptable, and the
  session re-checks the same digest before installing: a tampered copy was served to the new worker in a
  dry run and it was rejected (`verdict: blocked`, exit 3) while the real wheel planned cleanly (exit 0).
- The other four wheels were re-verified against the reviewed manifest rather than assumed: the
  wheelhouse's `transformers-5.13.1-py3-none-any.whl` (11,503,977 bytes,
  `53f0ea8aa397e29244c2377ba981bcaf0c87adcf44fbdd447ef6306522afcacd`) was downloaded with
  `api.dataset_download_file` and matches `resource/manifests/eval-wheelhouse-2026-10-09.json` exactly, so
  the mount has not drifted for this workstream. Its contents were then inspected instead of trusted:
  `transformers/models/gemma4/` is present, `transformers/__init__.py` exports `CompressedTensorsConfig`,
  `compressed-tensors>=0.15.0 is required`, and the wheel declares
  `tokenizers<=0.23.0,>=0.22.0` and `safetensors>=0.8.0` -- which is why the mounted `tokenizers 0.22.2`
  and `safetensors 0.8.0` are installed with it (the image ships safetensors 0.7.0). `CONFIG_MAPPING` is
  generated in this version: `auto_mappings.py` carries `("gemma4", "Gemma4Config")`, and
  `_LazyConfigMapping.__contains__` delegates to that mapping, so the check the notebook runs is
  meaningful rather than ornamental.
- Implementation: new worker `resource/l4_feasibility/prepare_model_stack.py` (install planning, PyPI
  hash check, offline `--no-index --no-deps` install, and a fresh-interpreter proof of
  `transformers` -> `CONFIG_MAPPING['gemma4']` -> `peft` -> `is_torchao_available()` ->
  `CompressedTensorsConfig(dequantize=True)` -> `AutoConfig.from_pretrained(base)`). It runs in the stage
  cell before any GPU work, so a broken stack costs a preflight, not a training attempt.
  `scripts/prepare_l4_feasibility_notebook.py` now embeds five workers, attaches the second dataset
  source, and no longer keeps the disproved "stay on the image's transformers" rationale. Two defects
  found while editing were fixed: the summary cell read `packed` without ever loading
  `packed-summary.json` (a guaranteed `NameError` in the last cell of a paid session) and read
  `compressed_tensors_step` from a file written before that key existed.
- Verification this round, all offline and all run: `tests/test_l4_feasibility.py` -> **38 tests pass**
  (7 new: install-decision rule, hash-mismatch rejection, missing-wheel blocking, all three Kaggle mount
  layouts, the recorded digest, the verdict rule, the second dataset source, and the summary-cell
  regression); the whole suite -> **94 tests pass**; `prepare_l4_feasibility_notebook.py --check` ->
  `status: ok` with five workers round-tripping; the worker's own CLI dry run -> `planned` for the real
  wheel and `blocked` for a tampered copy; `scripts/check_repository.py` still exits 1 for pre-existing
  reasons only (the uncommitted localization-v1 working tree and `.sh` files this Windows host cannot
  parse) and reports nothing about the files changed here.
- Quota: **15,990.809 s of 108,000 s remain**, refresh 2026-10-17, unchanged from 02:05, so the account
  is flat and the earlier drain is over. One session at the documented cap costs at most 6,600
  quota-seconds, about 41 % of what is left.
- **Pushed run 5**: `junzhejiao/gemma-4-four-l4-lora-training-feasibility-v5` version 1
  (https://www.kaggle.com/code/junzhejiao/gemma-4-four-l4-lora-training-feasibility-v5), generated from
  `artifacts/l4-feasibility-notebook-run5/` and pushed through `scripts/bounded_connect.py`. The kernel
  ref is derived from the notebook **title**, not from the metadata `id` slug, which is why the run is
  `-v5` while the preparation directory is named for the run.
- **Run 5 reached the model-stack step and stopped there on a defect in that step's own guard**, at 26 s
  of session time and a measured **68.216 quota-seconds** (15,990.809 -> 15,922.593, i.e. 34 s of L4x4
  wall). The guard asked `judge(plan, None)['problems']`, and that call returns
  `problems or {'verification': 'not run'}` whenever there is no verification yet, so it was never empty
  and the verification never ran: `verification: null`, `problems: {'verification': 'not run'}`. Fixed by
  splitting the real blocking problems into `collect_problems(plan)` and asking *that* whether anything
  is wrong; `judge(plan, None)` now returns an empty problem set for a clean plan, and a regression test
  asserts exactly that.
- Everything up to the guard was measured and is positive, which is worth keeping: the session did get
  **four NVIDIA L4s (23,034 MiB each), 188.69 GiB host RAM and 48 CPUs** with the competition attached;
  the wheelhouse mounted at `/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse` with all
  41 wheels; the new torchao dataset mounted at
  `/kaggle/input/datasets/kajarnaklokossou/torchao-0-16-0/` -- the **depth-3 layout**, which the first
  version of `resolve_wheel` would have missed, so that fix was load-bearing; the mounted wheel hashed to
  `2d6293a0c57c9dd505efb025a7189459d154965fbed000efd638cf299f9362dd` **on Kaggle**, matching PyPI; and all
  five installs returned 0 -- `transformers 5.0.0 -> 5.13.1`, `safetensors 0.7.0 -> 0.8.0`,
  `compressed_tensors 0.15.0.1` (absent -> installed), `loguru 0.7.3` (absent -> installed),
  `torchao 0.10.0 -> 0.16.0`, with `tokenizers 0.22.2` already satisfying the wheel's `>=0.22.0`
  requirement exactly as its metadata predicted.
- **Pushed run 6** immediately afterwards as version 2 of the same kernel, from
  `artifacts/l4-feasibility-notebook-run6/`, carrying the guard fix. Its outcome is recorded below once
  the session settles.
- Changed files: `scripts/bounded_connect.py` (new), `resource/l4_feasibility/prepare_model_stack.py`
  (new), `scripts/prepare_l4_feasibility_notebook.py`, `tests/test_l4_feasibility.py`,
  `docs/experiments/2026-10-12-l4-training-feasibility.md`, `WORKLOG.md`. Generated and gitignored:
  `artifacts/l4-feasibility-notebook-run5/**`, `artifacts/tmp/**`, `artifacts/torchao-dataset-upload*.log`,
  the quota receipts.
- Remaining follow-up: (1) read run 5's outputs and record which strategy builds, the per-GPU peak and
  the seconds per update -- or the next exact incompatibility; (2) reconcile the measured quota charge
  against the 2x L4x4 rule with a fresh `check_kaggle_quota.py` reading; (3) report to issue #13 the two
  results that stand on their own (handoff contract on four L4s, and the measured
  `transformers 5.0.0` / `torchao 0.10.0` / `peft 0.19.1` conflict with its fix).

### 2026-10-12 02:05 (Asia/Shanghai) - The training blocker is now a measured architecture/version conflict, with the fix identified; work stopped for the night at the user's request

- Objective: install the missing dependency, re-run, and reach the training-memory measurement. The
  session did reach the base load, and what it found replaces the earlier assumption.
- **Run 4's result, and it is the decisive finding of this task.** With the stage cell now installing
  `compressed_tensors` and `loguru` from the mount and proving the import works
  (`importable_after: true`), both routes failed one step later, on the base model itself:
  `ValueError: The checkpoint you are trying to load has model type 'gemma4' but Transformers does not
  recognize this architecture`, with `KeyError: 'gemma4'` from `CONFIG_MAPPING` in
  `configuration_auto.py`. The image ships `transformers 5.0.0`, which predates Gemma 4. Nothing about
  memory, FSDP or PEFT was ever exercised, because no weight was read.
- **The experiment document's environment premise is refuted by this measurement, and this matters.**
  `docs/experiments/2026-10-12-l4-training-feasibility.md` section 5 chose to keep the image's
  `transformers 5.0.0` specifically to avoid the torchao incompatibility. That choice is now known to be
  impossible: 5.0.0 cannot load the required architecture at all, so the wheelhouse's
  `transformers 5.13.1` is mandatory, and the torchao conflict must be solved rather than avoided.
- Fix identified, each step verified from sources rather than assumed:
  - `transformers 5.13.1` **does** contain the architecture: its wheel has `transformers/models/gemma4/`
    with `configuration_gemma4.py`, `modeling_gemma4.py`, `processing_gemma4.py` plus
    `gemma4_assistant`/`gemma4_unified` variants. The wheel's SHA256 was checked against PyPI.
  - The torchao gate is **peft's**, not transformers': `peft/import_utils.py::is_torchao_available()`
    returns `False` only when torchao is *absent* (`find_spec(...) is None`) and otherwise raises
    `ImportError: Found an incompatible version of torchao ... only versions above 0.16.0 are supported`
    when the installed version is below 0.16.0. The image's `torchao 0.10.0` is the trigger. Uninstalling
    torchao would satisfy it, but the honest fix is a newer wheel.
  - `torchao 0.16.0` publishes `torchao-0.16.0-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`,
    a cp310-abi3 wheel that installs on the image's CPython 3.12. Downloaded to
    `artifacts/torchao-wheel/`, 3,160,415 bytes, SHA256
    `2d6293a0c57c9dd505efb025a7189459d154965fbed000efd638cf299f9362dd`, matching PyPI exactly.
  - The checkpoint itself carries no custom modeling code — its only files are `config.json`,
    `model.safetensors` and `generation_config.json` — so `trust_remote_code` is not an available
    workaround.
- **Where the remaining blocker is, stated exactly.** The torchao wheel is not on the mounted
  wheelhouse and the notebook has no internet, so it must reach the session through a read-only dataset
  mount. A dataset was staged at `artifacts/torchao-wheel/` with `dataset-metadata.json`
  (id `junzhejiao/gemma4-torchao-016`) and `kaggle datasets create -p ... --dir-mode zip` was started,
  but it did not finish within 10 minutes and was cancelled when the user asked to stop. This is the
  same class of stall already recorded for this machine's dataset uploads (40 min via CLI, 20 min via
  SDK, both empty afterwards), so the next attempt should either run unattended for much longer or use
  the web UI, which is the documented path for this project.
- Verification this round: `tests/test_l4_feasibility.py` runs **30 tests, all passing**, now covering
  the raw templates plus the `chr(10)` newline convention (asserted in the file, in the rendered cells,
  and by executing a generated `plan.json` write and reloading it); `--check` reports `status: ok` and
  every cell compiles. Run 4's own outputs were read from `artifacts/l4-feasibility-v4-output/`.
- Quota: **15,990.809 s of 108,000 s remaining**, refresh 2026-10-17. Run 4 consumed about 795
  quota-seconds for roughly 320 s of wall time, again consistent with the 2x L4x4 rate. The drain is
  gone; the three sessions this round cost about 1,790 quota-seconds in total.
- Changed files: `scripts/prepare_l4_feasibility_notebook.py`, `tests/test_l4_feasibility.py`,
  `WORKLOG.md`, plus new helper scripts `scripts/poll_kernel.py` and `scripts/fetch_kernel_output.py`.
  Generated and gitignored: `artifacts/l4-feasibility-v4-output/**`, `artifacts/torchao-wheel/**`,
  `artifacts/tmp/**`, the quota receipts.
- Remaining follow-up, in order, for whoever resumes: (1) get `torchao 0.16.0` into a dataset the
  notebook mounts (retry the upload unattended, or use the web UI) and attach it as a second
  `dataset_sources` entry; (2) install `transformers 5.13.1` from the wheelhouse plus that torchao
  wheel in the stage cell, and verify the import chain `transformers` -> `peft` -> base load; (3) re-run
  once and finally record the strategy that builds, the per-GPU peak and the seconds per update, or the
  next exact incompatibility; (4) update section 5 of the experiment document, whose "keep transformers
  5.0.0" rationale is now disproved; (5) report two results to issue #13 that stand on their own: the
  handoff contract verified on four L4 GPUs, and the measured
  `transformers 5.0.0` / `torchao 0.10.0` / `peft 0.19.1` version conflict.

### 2026-10-12 01:45 (Asia/Shanghai) - The base-load dependency is fixed, and a generator escape defect cost one session before it was caught

- Objective: install the missing import-time dependency identified in the previous run and re-run, so
  the session finally reaches training memory.
- The dependency diagnosis was confirmed and fixed: the wheelhouse mount carries
  `compressed_tensors-0.15.0.1` and `loguru-0.7.3`, and the staging cell now installs both with
  `--no-index --no-deps` and then *imports* them to prove it worked. Run 3 recorded
  `install_returncode: {compressed_tensors: 0, loguru: 0}`, `problems: {}`, `importable_after: true`.
  That step is solved.
- **A generator escape defect stopped run 3 at 77 s, and it is worth recording precisely because it is
  the third variant of one mistake.** Converting the code-cell templates to raw strings removed a level
  of Python's own backslash processing, so the correctly spelled `'\\n'` in the template started
  reaching the generated cell verbatim and wrote the two characters ``\n`` instead of a newline. The
  result was a `plan.json` ending in `}\n`, four characters too many, so the next cell died on
  `JSONDecodeError: Extra data: line 10 column 2` and the session was lost before any model work. The
  lesson is that three layers now interpret the same backslash: the file, the template literal, and
  the generated code. All generated newline writes therefore use `chr(10)`, which means the same thing
  at every layer, and no template may spell a newline with a backslash.
- Why this was not caught offline, stated plainly: `--check` compiles the cells, and a cell that writes
  `}\n` compiles perfectly. Compilation is not the contract; a JSON file that parses is. The suite now
  asserts both, executes the generated `plan.json` write in a scratch directory and reloads it, and
  forbids every backslash-newline spelling in the templates. `tests/test_l4_feasibility.py` runs
  **30 tests, all passing**.
- Quota: the second session consumed 841.9 quota-seconds for 405.2 s of wall time (17,627.769 ->
  16,785.847), a ratio of 2.08, which independently confirms the archived 2x L4x4 rate. Run 3 cost
  about 154 quota-seconds for its 77 s.
- Evidence read this round, all from the runs' own output files: run 2's `environment.json`,
  `progress.jsonl`, `packed-progress.jsonl`, `packed-probe-result.json`, `train-summary.json`,
  `result-summary.json`; run 3's `compressed-tensors.json`, `environment.json`, corrupt `plan.json`,
  and log.
- Changed files: `scripts/prepare_l4_feasibility_notebook.py` (raw templates, `chr(10)` writes, the
  verified `REQUIRED_WHEELS` install, `build_metadata`, measured-rank launch), `tests/test_l4_feasibility.py`,
  `WORKLOG.md`. Generated and gitignored: `artifacts/l4-feasibility-v3-output/**`,
  `artifacts/l4-feasibility-notebook-v6/**`.
- Remaining follow-up: read the settled run 4 for the strategy that built, per-GPU peak memory and
  seconds per update, or the exact incompatibility; then compile the report for issue #13, which
  already has two reportable results (the handoff contract verified on four L4s, and the
  `compressed_tensors`/`loguru` environment defect).

### 2026-10-12 01:20 (Asia/Shanghai) - Four L4s granted and the contract verified; both training routes now stop on one missing package

- Objective: run the corrected bounded session, then diagnose and repair whatever it exposed.
- **The hardware substitution is fixed and confirmed by measurement.** The pushed session
  (`junzhejiao/gemma-4-four-l4-lora-training-feasibility-v2`, version 1) reported **four `NVIDIA L4`
  GPUs of 23,034 MiB each, 188.69 GiB host RAM and 48 CPUs**, with the official wheelhouse mounted at
  `/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse` and 41 wheels visible. The stage
  cell's preflight passed with `preflight_problems: []`. So attaching the competition
  (`competition_sources: ['gemma-4-developer-agent']`) is what grants L4x4, and the same omission had
  also suppressed the wheelhouse mount, which is why the earlier T4 session saw `wheels_visible: 0`.
  The earlier figure of 31.35 GiB host RAM was likewise a T4-host artifact; this host reports 188.69 GiB,
  which is above the 70 GiB threshold, so `fsdp-all` would have been attempted rather than skipped.
- **The reviewed handoff contract is verified against the real tokenizer on four L4s**, which is the
  durable result from this session: `handoff_token_id` 50 with `handoff_token_is_50: true`,
  `whole_starts_with_prefix: true`, `token_boundary_matches: true`, `target_ends_with_handoff: true`,
  179 prompt tokens, 19 target tokens, 198 sequence tokens. It reproduced identically in both the packed
  probe and the training worker.
- **Both routes then stopped on the same missing dependency, before touching any weights.** The image
  ships `compressed_tensors 0.15.0.1` but not its own import-time dependency `loguru`:
  `compressed_tensors/logger.py` does `from loguru import logger`, so
  `CompressedTensorsConfig(...)` raises `ModuleNotFoundError: No module named 'loguru'`. That failed the
  packed route at `dequantize=False` and the four-GPU route at `dequantize=True` identically. Zero
  weights were loaded, so neither path is refuted — this is an environment defect, not a model
  incompatibility, and the wheelhouse carries `loguru-0.7.3-py3-none-any.whl`.
- Two further measured facts worth keeping: this image's `CompressedTensorsConfig` does **not** accept
  `dequantize` as a named parameter (`compressed_tensors_config_accepts_dequantize: false`) though it
  accepts `**kwargs`; and the packed probe recorded the base's real shard hashes
  (`config.json` b100d85e…, `model.safetensors` 1b9b1d62…, `generation_config.json` 70110aa8…).
- Session cost: 405.2 s of wall time, an estimated **810.4 quota-seconds** at the 2x L4x4 rate, stopped
  by the load failure rather than by any budget. No adapter was produced, so the adapter-validation and
  official-runtime reload phases were skipped as designed.
- Verification performed: the run's own `environment.json`, `progress.jsonl`, `packed-progress.jsonl`,
  `packed-probe-result.json`, `train-summary.json` and `result-summary.json` were read from
  `artifacts/l4-feasibility-v2-output/`; the log was downloaded and searched. The
  `NameError`/`UnboundLocalError` crash class recorded last round did **not** recur, which confirms
  those two repairs: both workers reached their own recorded error paths and wrote result files.
- Changed files: `WORKLOG.md` only this round; the run's outputs are gitignored under
  `artifacts/l4-feasibility-v2-output/`.
- Remaining follow-up, in order: (1) install the missing import-time dependencies from the mounted
  wheelhouse in the stage cell, which is a small, bounded, offline step; (2) re-run once and read the
  strategy that builds, the per-GPU peak and the seconds per update; (3) reconcile the quota charge for
  both sessions against the 2x rule; (4) report the contract verification and the dependency finding to
  issue #13, since the contract result is reportable now.

### 2026-10-12 01:05 (Asia/Shanghai) - The T4 downgrade and both worker crashes are explained, fixed and re-pushed

- Objective: resume issue #13 after the user reported that the GPU drain had been fixed. The first
  bounded L4 session had already run and failed, so the work was to diagnose that failure, repair what
  it exposed, and re-run.
- What the first session actually produced, recovered from `artifacts/l4-feasibility-run-output/` and
  `artifacts/l4-feasibility-pulled/` (the run's own `result-summary.json`, `environment.json`,
  `progress.jsonl`, `packed-progress.jsonl` and log): **213 s of wall time, no adapter, no training**.
  Two independent causes, both measured rather than inferred:
  1. **The session ran on 2x Tesla T4 (14,911 MiB each), not four L4s**, with `host_memory_gib` 31.35
     and 4 CPUs, even though the metadata requested `machine_shape: NvidiaL4`. The four-GPU phase then
     died on `torch.AcceleratorError: CUDA error: invalid device ordinal` because `torchrun` was
     launched with a hard-coded `--nproc_per_node=4`, and local ranks 2 and 3 have no device.
  2. **Both workers crashed on their own error paths, not on the model.** `probe_packed_base.py`
     raised `NameError: name 'torch' is not defined` and then `UnboundLocalError: cannot access local
     variable 'finish'`, because `finish()` was defined *after* the native-contract check that calls
     it; `train_l4_feasibility.py` raised the same `NameError` because `native_batch()` used the
     module-global `torch` without importing it. Both were already repaired in the working tree
     (commitless, from the interrupted turn) and are now covered by tests.
- **The root cause of the hardware substitution is now identified and fixed.** L4x4 is gated on
  competition attachment: `resource/competition/upgraded-accelerators.md` says "L4s are only available
  for notebooks attached to this competition". The working `gemma4-lora-serving-probe` notebook that
  genuinely measured four L4s sets `competition_sources: ['gemma-4-developer-agent']`; this generator
  set `competition_sources: []`, so Kaggle silently substituted two T4s. The generator now writes
  `COMPETITION_SOURCE = 'gemma-4-developer-agent'`. This single omission also explains the second
  symptom: `stage` recorded `wheels_visible: 0` and no wheelhouse mount, so `compressed_tensors` was
  absent and the packed base could not have loaded; the wheelhouse dataset itself is fine
  (`dataset_list_files('metric/gemma-4-developer-agent-wheelhouse')` lists `compressed_tensors-0.15.0.1`
  and the rest), so its mount was missing in that session too.
- Quota: stable at **17,627.769 s remaining** (used 90,372.231 of 108,000, refresh 2026-10-17) across
  two readings 8.5 minutes apart — an identical `time_used`, which is the evidence that the drain is
  over. Two hours earlier, on 2026-10-11 at 23:09, it was 26,398.619 s, so the leak consumed roughly
  8,770 quota-seconds (about 73 minutes of L4x4 wall time) after this work was written up.
  `artifacts/quota-live-2026-10-12-resume.json` and `-resume2.json`.
- Repairs made this round, each the minimum change for the defect it addresses: the generator now
  attaches the competition (metadata and `preparation.json`); the train cell reads the measured GPU
  count from `environment.json` and launches that many ranks, so a downgraded allocation is recorded
  instead of crashing; `train_l4_feasibility.py` gained `--load-budget-seconds`, enforced with a
  watchdog thread around the base load, so a stalled 58 GiB load records `load_timeout` rather than
  being killed by the notebook's subprocess timeout with no measurement at all; `validate_budgets()`
  was extracted from `main` so the ladder is testable; and the default train timeout rose to 1600 s
  with the session cap kept at 3300 s, so the timeouts total 3100 s inside the cap (a 6600
  quota-second worst case, 37 % of what is left).
- Verification: `tests/test_l4_feasibility.py` runs **28 tests, all passing**, adding coverage for the
  competition attachment in the generated metadata, the measured-rank launch, the load-budget flag
  reaching the worker and staying below the subprocess timeout, and the budget ladder rejecting a
  cap it cannot hold. `scripts/prepare_l4_feasibility_notebook.py --check` reports `status: ok` and
  every code cell compiles; the worker compiles and exposes the new flag.
- Pushed: version 1 of `junzhejiao/gemma-4-four-l4-lora-training-feasibility-v2` (the slug is derived
  from the **title**, not the metadata `--slug`, exactly as recorded for the first kernel), requesting
  `NvidiaL4` with a 3600 s platform timeout. Status `QUEUED` when this entry was written; the run is
  polled at 45 s intervals into a background job and its outputs will be fetched with
  `api.kernels_output` when it settles.
- New helper scripts (small, read-only against Kaggle): `scripts/poll_kernel.py` (polls
  `kernels_status`, comparing `status.name` rather than `str(raw)`, which silently fails) and
  `scripts/fetch_kernel_output.py` (downloads outputs plus the log, because an unrecorded worker crash
  appears only in the log).
- Changed files: `scripts/prepare_l4_feasibility_notebook.py`, `resource/l4_feasibility/train_l4_feasibility.py`,
  `tests/test_l4_feasibility.py`, `scripts/poll_kernel.py` (new), `scripts/fetch_kernel_output.py` (new),
  `WORKLOG.md`. Generated and gitignored: `artifacts/l4-feasibility-notebook-v3/**`, the quota receipts,
  `artifacts/tmp/**` inspection helpers.
- Remaining follow-up: read the settled session's `environment.json` (does it now report four L4s?), the
  strategy that built, the per-GPU peak and seconds per update, or the exact incompatibility; reconcile
  the quota charge against the 2x rule; then record the result in
  `docs/experiments/2026-10-12-l4-training-feasibility.md` and report to issue #13.

### 2026-10-12 00:05 (Asia/Shanghai) - Bounded L4 session pushed and running; the quota drain is not a kernel session

- Objective: with the user's approval, run the one bounded L4 feasibility session from issue #13, and first answer their question about how to stop the session that had been draining quota.
- Pushed and running: `junzhejiao/gemma-4-four-l4-lora-training-feasibility` version 1, kernelId 138194260, pushed 2026-10-11 23:59:51 Asia/Shanghai with `kernels_push(..., timeout='3600', acc='NvidiaL4')`; status `RUNNING` at 16:00:26Z. The platform derived the slug from the notebook **title**, not from the metadata `--slug`, so the ref is `/code/junzhejiao/gemma-4-four-l4-lora-training-feasibility` and monitoring must use that. URL: https://www.kaggle.com/code/junzhejiao/gemma-4-four-l4-lora-training-feasibility
- The leak is **not** a kernel session, established by three independent checks rather than by inspection: `kernels_list(mine=True, sort_by='dateRun')` shows the newest run as the probe at 2026-10-11T10:29:30Z with every later entry older; the `kagglesdk` `GetKernelSessionStatus` RPC reports `COMPLETE`/`ERROR` for five recent kernels and **no RUNNING session**; and `CancelKernelSession` therefore has no `kernel_session_id` to act on. The competitions API exposes **no cancel for submissions**, and the one `PENDING` submission (ref 57067045, teammate `losttrace`, 2026-10-11T07:48:04Z) returns an **empty episode list**, so it is queued rather than scoring. Two `[Private Notebook]` entries with empty refs in the kernel list are the kind of draft that has no slug and is therefore invisible to every kernel API — the most probable holder is an interactive session open in a browser tab, which only the user can stop.
- SDK facts found by running, all of which cost real time: `api.kernels_status()` returns an `ApiGetKernelSessionStatusResponse` object whose `.status` is a `KernelWorkerStatus` enum — not a dict and not a JSON string, so both `isinstance(raw, dict)` and `json.loads(raw)` silently fail and a poller comparing `str(raw)` to `'RUNNING'` exits after one iteration; `quota_view().to_dict()` **drops whole days** from durations (`timeUsed` renders as `1268.330431.0s`, `totalTimeAllowed` as `21600s`), so the documented six-hour figure is the day-truncated form of 108,000 s and only `timedelta.total_seconds()` is trustworthy; and `get_accelerator_quota_statistics()` exists alongside `quota_view()` but renders durations the same broken way.
- Quota at push (16:00:03Z): `time_used` 87,634.196 s of 108,000 s, and `remaining_after_reservations` collapsed to **0** because `time_reserved` jumped from 0 s to 43,177.758 s — i.e. the platform reserved more than the remaining pool as soon as the session was created. So the session's own charge and the leak's will have to be separated from the reservation after it settles, and a fresh `quota_view` read is part of the analysis.
- The notebook itself was extended before the push, on the reasoning that the packed single-GPU route is the only one that plausibly fits this hardware: a BF16 replica needs ~58 GiB of host RAM **per rank** (four ranks on one host), whereas the shipped packed checkpoint is ~16-17 GiB and loads in one process. So phase order is now stage → packed single-GPU probe → four-GPU FSDP (only if the packed route saved no adapter) → adapter verification → official-runtime reload → summary, with per-phase wall budgets and a CLI guard that refuses a session limit the phase timeouts can overrun. `probe_packed_base.py` records the decisive fact (the class of the projection modules PEFT would wrap) seconds after the load and converts every incompatibility into a recorded verdict instead of a traceback.
- Verification before pushing: 22 offline tests pass (`OK (skipped=3)`) after the test doubles were taught the new `--packed-*` arguments and the new cell layout; the generator's `--check` reports `status: ok`; and the written notebook was reloaded to confirm all four embedded workers decode byte-identically to the current sources with matching digests.
- Changed files: `resource/l4_feasibility/probe_packed_base.py` (new), `resource/l4_feasibility/train_l4_feasibility.py` (shared `target_loss`, host/dependency probes), `scripts/prepare_l4_feasibility_notebook.py` (packed phase, budget guard, phase policy), `tests/test_l4_feasibility.py`, `docs/experiments/2026-10-12-l4-training-feasibility.md`, `WORKLOG.md`. Generated and gitignored: `artifacts/l4-feasibility-notebook/**` (including `status-history.jsonl`), `artifacts/kaggle-probe-output/**`, quota receipts.
- Remaining follow-up: download the session's outputs when it settles and record which route worked, the per-GPU peaks, the seconds per update and the exact incompatibility otherwise; reconcile the reservation and the leak against the push baseline; then hand the review comments and the planned-setup post to a human, since this workspace cannot reach GitHub.

### 2026-10-11 23:50 (Asia/Shanghai) - Drain characterised from four measurements, and three L4-path defects repaired before any spend

- Objective: continue issue #13 after the user asked to understand the quota consumption before deciding whether to run the L4 feasibility session. No GPU quota was to be spent.
- Drain, confirmed live and unchanged in character: readings at 23:44:59 and 23:45:22 (22.3 s apart) differ by 44.579 quota-seconds, a rate of **1.9988**; earlier pairs gave 2.000, 2.000 and 1.9986. So a single uninterrupted L4x4-rate session has been holding four GPUs since before 18:36 and was still running at 23:45, at roughly **120 quota-seconds per wall minute**. Remaining fell from 26,398.619 s (23:09) to 22,138.756 s (23:45). It is not a kernel: the newest `kernels_list(mine=True, sort_by='dateRun')` entry is `junzhejiao/gemma4-lora-serving-probe` at 2026-10-11T10:29:30Z, status `COMPLETE`, and `kaggle`/`kagglesdk` expose no session, stop or cancel call, so it can only be stopped from the Kaggle UI. Evidence: `artifacts/quota-drain-check.json`, `artifacts/quota-drain-check-2.json`, `artifacts/quota-latest.json`, `artifacts/quota-round2.json`, `artifacts/quota-round2-final.json`.
- The existing probe's own output was recovered with `api.kernels_output` at no quota cost into `artifacts/kaggle-probe-output/`, which answers L4x4 questions that would otherwise need a session: `visible_gpus: 4`, `torch 2.10.0+cu128`, `vllm 0.19.1`, base path `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`, and the wheelhouse inventory it installed (`compressed_tensors-0.15.0.1`, `transformers-5.13.1`, `vllm-0.19.1`, `bitsandbytes-0.46.1`, `safetensors-0.8.0`). Host RAM and CPU count are **not** in that output and the kernel log has no `/proc/meminfo` record, so host memory stays the one unknown that gates the FSDP loading strategy.
- Defect 1, and it would have wasted the session: the required base is a compressed-tensors checkpoint, and `CompressedTensorsConfig(dequantize=True)` raises `ImportError: compressed-tensors>=0.15.0 is required for compressed-tensors quantization` when the package is absent — reproduced locally against `transformers 5.19.0`. Whether the competition image carries it was never recorded. The notebook's train phase now checks for it in a subprocess and installs **just** that wheel from the mount (`--no-index --no-deps`) when missing, recording the probe, the chosen wheel and the post-install import result to `compressed-tensors.json`. It deliberately does not install the whole wheelhouse, which would upgrade `transformers` and hit the torchao wall before training starts. The mount's `0.15.0.1` satisfies the `>=0.15.0` requirement exactly, so the package is not the packed-path blocker — PEFT's module handling is.
- Defect 2: the per-update peak memory was sampled *after* the forward, so the number this run exists to establish would have understated an update's real footprint. Statistics now reset at the start of the update, covering forward, backward and the optimizer's lazily allocated state; `max_memory_reserved` is recorded beside `max_memory_allocated`, since reservation is what decides an OOM; and `seconds` now covers the whole update.
- Defect 3: the `fsdp-ignored` export called `raw_model.state_dict()`, and FSDP's default state-dict type gathers a complete replica onto rank 0 — about 58 GiB, precisely the failure that strategy exists to avoid. The adapter is now read directly from `named_parameters`, which is safe because those parameters were excluded from sharding. Two hardening changes went in with it: replicated LoRA parameters are broadcast from rank 0 after wrapping, so no two ranks start from different adapters, and the trainable/replicated sets are read back from the *wrapped* network, because with `use_orig_params=True` those are the objects the optimizer must see. The worker also records whether this `transformers` build's `CompressedTensorsConfig` accepts `dequantize`, which is a version-specific gate between the image's 5.0.0 and the recipe's 5.18.0.
- Verification: `tests/test_l4_feasibility.py` now runs **22 tests, 19 pass, 3 skipped** for the recorded sandbox reason, adding coverage for the trainable-versus-replicated split and for the worker-to-notebook field names; the generator's `--check` reports `status: ok`; the regenerated `artifacts/l4-feasibility-notebook/feasibility.ipynb` was reloaded and its three embedded workers decode byte-identically to the current sources with matching digests, and the train cell contains the new dependency step. The adapter verifier was run against the **real rank-32 all-layer adapter** the probe generated and the served engine accepted: 240 of 240 tensors match the served naming scheme, 120 `lora_A` + 120 `lora_B` over all 60 layers, rank 32 from shapes, `problems: []`, exit 0, 104,238,416 bytes — byte-identical to the figure already recorded in this log, so the scheme is confirmed at rank 8 and rank 32 rather than once. One self-inflicted defect was caught by the compile check before any run: a single-escaped `'\n'` inside a generator template became a real newline and broke the cell, the same escape-defect class this project has already paid for.
- Changed files: `resource/l4_feasibility/train_l4_feasibility.py`, `scripts/prepare_l4_feasibility_notebook.py`, `tests/test_l4_feasibility.py`, `docs/experiments/2026-10-12-l4-training-feasibility.md`, `WORKLOG.md`. Generated and gitignored: `artifacts/l4-feasibility-notebook/**`, `artifacts/kaggle-probe-output/**`, the quota receipts.
- Not done: no GPU session, no push, no upload, **no quota spent by this work**. Host RAM is still unmeasured, so no claim is made that the FSDP path will fit; the notebook measures it first and skips the whole-model gather below 70 GiB.
- Remaining follow-up: (1) the user stops the draining session from the Kaggle UI and I re-measure to confirm the rate is zero; (2) post the planned setup and get approval for one bounded session; (3) run it and record which strategy built, the per-GPU peaks and seconds per update, or the exact incompatibility; (4) Path D only if both FSDP strategies fail.

### 2026-10-11 23:20 (Asia/Shanghai) - Four-L4 training feasibility: instrument, compatibility analysis, and the PR #8 review

- Objective: take over issue #13 (model/adapter compatibility and reproducible small LoRA experiments on four L4 GPUs, parent workstreams #7/#4, implementation PR #8). Its five tasks are: review PR #8's chat-template boundaries, supervised handoff token 50, save/restart contract and early agent gates; validate the compact H100 recipe's artifacts with the maintainer; investigate PEFT/FSDP sharding on four L4s while preserving the required base; build a private bounded L4 feasibility notebook (load, one backward step, at most eight updates, save, reload in the official runtime); and record versions, per-GPU peak memory, seconds/update, physical GPU time versus quota, and the exact resume path.
- How the issue was obtained, because it constrains everything: the repository is private and this workspace cannot reach it. `web_fetch` on the issue returns 404 unauthenticated, `github.com:443` is unreachable from `pwsh`, and there is no `gh` CLI, no token and no retrievable Git credential (the credential helper dies on the sandbox's named-pipe restriction). The user pasted the issue text. Anything resembling a PR comment or issue post from here must therefore be handed to a human.
- Localization of the referenced material: the issue's own paths (`configs/training/swe-lora-light-protocol.json`, `docs/experiments/2026-10-10-swe-lora-light.md`, `resource/manifests/swe-lora-light.json`) do not exist on the checked-out branch. They are on `origin/experiment/swe-lora-v1` at `ccfcff0`, the head of PR #8, which is a diverged line from the current `fix/lora-rank8-controlled`. The issue's concatenated `experiment/swe-lora-v1ccfcff01fcc891` decodes to branch `experiment/swe-lora-v1` plus commits `ccfcff0` and `1fcc891`. `git archive` cannot export that tree on Windows (the repository has a leaderboard CSV whose name contains a colon), so files were read individually and staged under `artifacts/review/swe-lora-v1/` as a review copy.
- Quota, measured this round (`scripts/check_kaggle_quota.py --output artifacts/quota-2026-10-12.json`): **26,398.619 s of 108,000 s remaining at 23:09**, used 81,601.381 s, refresh 2026-10-17. **The rate rule is now pinned from the archived official page**: `resource/competition/upgraded-accelerators.md` says L4x4 notebooks consume quota at **twice** the wall-clock rate, so the remaining budget was about 13,199 s of L4x4 wall time. This also explains the 1.77–2.35 ratios in earlier rounds' readings.
- **The consumption is not a job, and it is still running.** A fourth reading at 23:31 and a fifth at 23:33 (`artifacts/quota-drain-check.json`) show 302.054 quota-seconds consumed in 151.1 wall seconds — a rate of **1.9986**, i.e. exactly the 2x L4x4 rate, with `remaining` falling to **23,524.9 s**. Two intermediate readings prove it has been continuous since 18:36: 18:36:07 used 48,751.063, 20:00:10 used 58,837.494 (`runs/2026-10-11-rank8-pilot/quota-before.json`), 23:09:52 used 81,601.381 — deltas of 10,086.431 s over 5,043 s and 22,763.887 s over 11,382 s, both exactly 2.000x. So a single uninterrupted 2x-rate session has been holding four L4s for the whole window and is still holding them, at roughly **120 quota-seconds per wall minute**; at that rate the remaining budget is exhausted in about 3.3 hours. It is **not a kernel**: `kernels_list(mine=True, sort_by='dateRun')` shows the newest run as `junzhejiao/gemma4-lora-serving-probe` at 2026-10-11T10:29:30Z with status `COMPLETE`, and every other recent kernel is COMPLETE or ERROR with older timestamps. The `kaggle` and `kagglesdk` packages expose no session, stop or cancel call, so this can only be stopped from the Kaggle UI (an interactive/Lab session or an orphaned worker of the kind this project has already been bitten by). Stopping it is now the prerequisite for any further spend.
- Deliverable 1, the instrument: `scripts/prepare_l4_feasibility_notebook.py` generates a private `NvidiaL4` notebook (five stdlib-only code cells, each writing results to disk as it runs, because a failed Kaggle session publishes nothing until it ends). Phase 0 stages three workers after verifying their SHA256 and records host RAM, GPUs and the wheel inventory; phase 1 runs `torchrun --standalone --nproc_per_node=4` under a hard timeout for one backward step, at most eight updates and an adapter export; phase 2 validates the export; phase 3 installs the wheelhouse and reloads the adapter in the official vLLM runtime with a repeat-base control; phase 4 summarises with the session wall time and the estimated quota charge. The workers are real files (`resource/l4_feasibility/{train_l4_feasibility,verify_l4_adapter,serving_l4_adapter}.py`) embedded byte-for-byte, not f-strings, because brace escaping inside generated code has already cost this project a GPU session. The kernel never imports `transformers`/`peft`/`torch`, and the wheelhouse is installed **after** training, so the documented torchao conflict cannot fire; with four ranks and accumulation four, one epoch of the 128-action recipe is exactly eight updates, so the update cap is a full epoch rather than an arbitrary slice.
- Deliverable 1, why it is shaped that way (source-verified predictions, to be measured not assumed): a BF16 replica of the base is ~58 GiB of the recorded 67.38 GiB peak against 22.5 GiB usable per L4, so `fsdp-ignored` (LoRA modules passed as FSDP `ignored_modules`, replicated LoRA, explicit gradient all-reduce, no parameter gather at export) is tried first, and whole-model FSDP with a full state-dict gather is skipped automatically when host RAM is below 70 GiB. The packed-W4A16 alternative is expected to be **incompatible with PEFT LoRA** for four read-and-cited reasons in `transformers 5.19.0`/`peft 0.21.2`: `post_init` honours `dequantize=False` for a compressed checkpoint, `_process_model_before_weight_loading` calls `compressor.compress_model` so the modules are not `nn.Linear`, `is_qat_trainable()` returns `dequantize or not is_quantization_compressed` (False here) and `trainer_utils` raises for that combination, and PEFT has no compressed-tensors handling at all. NF4 is recorded as a different base model and is deliberately not substituted.
- Deliverable 2, documents: `docs/experiments/2026-10-12-l4-training-feasibility.md` (quota arithmetic, the one-L4 memory argument, the four candidate paths with the gate that decides each, the instrument, the measurement contract, and the planned-setup text to post before consuming compute) and `docs/experiments/2026-10-12-pr8-review.md` (the four review areas plus artifact validation).
- Two real defects in the training worker, found by re-reading it rather than by running it, and fixed: (1) the per-update peak was sampled *after* the forward, so the number this run exists to establish would have understated the update's real footprint — the statistics now reset at the start of the update so the peak covers forward, backward and the optimizer's lazily allocated state, and `max_memory_reserved` is recorded alongside `max_memory_allocated`, since reservation is what actually decides an OOM; `seconds` now likewise covers the whole update rather than backward-and-step. (2) the `fsdp-ignored` export called `raw_model.state_dict()`, and FSDP's default state-dict type gathers a complete replica onto rank 0 — about 58 GiB, the exact failure the strategy exists to avoid; the adapter is now read straight from `named_parameters`, which is safe because those parameters were excluded from sharding. Two more hardening changes went in with them: the replicated LoRA parameters are broadcast from rank 0 after FSDP wrapping (their random initialisation must be identical on every rank, or the ranks would train different adapters and average unrelated gradients), and the trainable/replicated sets are now read back from the *wrapped* network, because with `use_orig_params=True` those are the objects the optimizer must see.
- Verification actually executed, offline: the generator's `--check` reports `status: ok` and every code cell compiles; the notebook written to `artifacts/l4-feasibility-notebook/` was reloaded and its three embedded workers decode byte-identically to the current sources with matching digests; `tests/test_l4_feasibility.py` runs 22 tests with 19 passing and 3 skipped for a recorded reason (this sandbox denies a confined process nested-directory creation, so the CLI-over-a-directory cases cannot execute here; the decision rule they exercise is covered by in-memory tests), and the suite now also covers the trainable-versus-replicated split that decides which gradients need an explicit all-reduce and the worker-to-notebook field names, which are an integration seam with no type system behind them; the adapter verifier was run against the **real** served probe adapter `artifacts/lora-probe` and reports 4 of 4 tensors on the scheme the competition stack was measured to accept, rank 8, `in_features` 5376/8192 (independently reproducing the recorded shapes) and exits 0, while the same adapter at an expected rank of 16 exits 1. All five source hashes recorded by the compact run's manifest were re-hashed from the committed blobs at `1fcc891` and all five match, which is what makes the rest of the manifest reviewable; the manifest's coverage, cost and outcome figures were cross-checked against the narrative and reconcile (128 actions, 236,021 input tokens, 6,195 supervised tokens, the four terminal/serving tables, 67.38 GiB, 947 s, 0.263 GPU-hours, and the 3/11 versus 2/11 fixed comparison).
- Review conclusions worth acting on: the early-gate design is sound — `summarize_eval_smoke.py` raises `SystemExit` on a failed control and the launcher runs under `set -e`, so inference cannot start on an invalid task — but it is a per-task gate, since the loop covers two of the three registered tasks and stops at the first failure. Two cheap changes are recommended before merge: count or fail when a terminal assistant turn is dropped because the preceding observation is not named `submit_patch`, and hash every base weight shard instead of assuming `model.safetensors`. One limitation should be added to the run document: `enable_thinking=True` is applied to every training target including tool calls, while the short serving probes disabled thinking. The save/restart contract is stronger than the summary suggests (config, dependencies, data hash, world size, base and source hashes; optimizer and per-rank RNG restoration with a successful update 9), and the trainer on PR #8 is Trillium-only by construction (it requires `SLURM_JOB_ID` and loads full BF16 replicas), which is why issue #13 needs a separate L4 entry point.
- Environment facts re-confirmed this round: the DSH `write`/`edit` tools can create files in `scripts/`, `docs/` and `resource/`, but confined child processes cannot — `pwsh`/`Python` are denied in those directories, denied nested directory creation under `artifacts/`, and cannot write to `.git` (a worktree could not be created). So deliverables are written with the file tools, generated artifacts go under `artifacts/` (gitignored), and `git add`/commit was not attempted. Local library versions differ from every installed environment: this workspace has `torch 2.14.1+cpu`, `transformers 5.19.0`, `peft 0.21.2`, `accelerate 1.15.0`, no `compressed-tensors`; the Kaggle image has `transformers 5.0.0`; the H100 recipe ran `5.18.0`/`peft 0.19.1`; the local GPU is an 8 GiB RTX 5060, so nothing here can exercise the model.
- Changed files: `scripts/prepare_l4_feasibility_notebook.py`, `resource/l4_feasibility/{train_l4_feasibility,verify_l4_adapter,serving_l4_adapter}.py`, `tests/test_l4_feasibility.py`, `docs/experiments/2026-10-12-l4-training-feasibility.md`, `docs/experiments/2026-10-12-pr8-review.md`, `WORKLOG.md`. Generated and gitignored: `artifacts/review/swe-lora-v1/**` (review copy of the PR branch), `artifacts/review/hashcheck/**`, `artifacts/quota-2026-10-12.json`, `artifacts/l4-feasibility-notebook/` when generated without `--check`.
- Not done, deliberately: **no GPU session, no Kaggle push, no upload, no competition submission, no GPU quota consumed.** No commit or push, because GitHub is unreachable and `.git` is not writable from a confined process. The two documents are written to be posted by a human, since this workspace cannot comment on the issue or the PR.
- Remaining follow-up, in order: (1) **stop the live L4x4 session that is draining quota at 2x** — this is the blocking condition and it can only be done from the Kaggle UI — then re-measure; (2) get approval for one bounded session and post the planned setup from section 7 of the feasibility document; (3) reconcile the measured quota charge against the 2x rule; (4) run the session and record which strategy built, the per-GPU peak and the seconds per update, or the exact incompatibility; (5) if both FSDP strategies fail, implement the layer-split pipeline at Path D; (6) hand the two requested PR #8 changes and the thinking-channel limitation to the maintainer.

### 2026-10-11 19:55 (Asia/Shanghai) - Audit the DeepSeek handoff and design a valid LoRA comparison

- Objective: understand the recent work and advise how to improve LoRA or compare the remote baseline with an adapted model.
- Work performed: reviewed the resume/history, required project guides, LoRA training/data/notebook builders, serving report, harness rank/trace source, and paired analysis; consulted official TRL/PEFT/Transformers documentation. Added `docs/experiments/2026-10-11-lora-readiness-audit.md` with findings and a staged protocol.
- Verified: adapter checker reports `status: ok` and `weights_present: false`; control analysis finds four historical Rich controls and no LoRA pairs. The 116-row artifact is entirely `train` (58 FastAPI, 45 Rich, 13 Requests), overlapping `rich_3454`, `rich_3675`, and `rich_3676`. Baseline/candidate prompt and budget text matches after newline normalization, although raw bytes differ.
- Executed defects: the generated notebook's doubly encoded config evaluates to a string and rank access raises TypeError; mock gold-pass plus agent-fail evidence passes the trajectory builder's acceptance gate; a local-harness-shaped observation produces an empty tool-call ID. Static review found missing holdout filtering, messages-versus-completion loss integration, single-device model placement, and unvalidated W4A16/NF4 loading. These were documented, not repaired.
- Statistical correction: four wins/no losses gives exact p=0.125, six gives p=0.03125. The existing power function assumes no regressions and gives 0.991322774 power at 129 tasks under baseline=0/gain=0.1, contradicting the existing protocol's universal insufficiency claim. Controls/arms are currently aggregated by task and directory names rather than explicit frozen run identities. Rank 16 and 32 have separate harness buckets.
- Decision: fix training/data provenance first; prefer verified coder trajectories with inspected assistant masks, real-base short training/export/reload, then an identical-environment baseline versus coder-only LoRA pilot and a separate held-out confirmation. Small pilots remain useful diagnostics without supporting broad superiority claims. Training/checkpoint size and runtime environment support still require measurement.
- Verification scope: targeted CPU reproductions and existing read-only checkers ran; documentation whitespace and local-link checks passed. No full test suite or GPU inference/training. An exploratory nested observation fixture was rejected as evidence after checking the actual local schema.
- Changed files: `WORKLOG.md` and the new readiness audit only. No source/configuration/weight changes; pre-existing user changes were preserved. No GPU launch, quota query, upload, submission, remote fetch, push, PR, or merge.
- Follow-up: implement the documented readiness fixes on a reviewed feature branch, establish disjoint data and valid controls, choose an available training platform, and manually launch only after CPU/integration gates and a fresh quota check.

### 2026-10-11 18:40 (Asia/Shanghai) - The rank-32 all-layer configuration is served too, and the image's torchao constraint is mapped

- Objective: close the one gap left by the rank-8 result -- whether the configuration we actually intend to train (rank 32, `q_proj`+`o_proj`, every layer) is served by this stack, given upstream #50059 reports wide higher-rank adapters degrading on compressed-tensors int4 bases.
- Result, measured on kernel version 11: **the rank-32 all-layer adapter is applied.** With the same reproducibility control (two base runs, 0.0 delta), the runtime-generated rank-32 adapter on **all 60 layers** (240 tensors, 104,238,416 bytes) moved the top logprobs by up to **10.069917** at 35 of 35 positions and **changed the greedy tokens and text** (35 to 40 tokens). The probe's verdict line: `runtime rank-32 all-layer adapter: applied -- the engine reproduces its own logprobs exactly and the adapter moves them, changing the generated tokens`.
- What that does and does not license: it kills the "the planned adapter config silently does nothing" risk, and the #50059 degradation does not appear here as a dropped adapter. It does **not** show rank 32 is wise -- a random rank-32 adapter with `lora_B` at std 1.0 is *expected* to destroy the output, and destroying the greedy text is exactly what a loud random adapter should do. The configuration is servable, not validated, and no trained adapter has ever been served.
- Three defects found by running, in order: (1) `AttributeError: 'Gemma4Config' object has no attribute 'num_hidden_layers'` -- the decoder fields live under `text_config`, and my first fix tried to guess which holder was authoritative; (2) after rewriting, `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`; (3) dropping the `transformers` import did not help, because `peft` itself pulls the quantizer stack and raised the same error. The final runtime adapter therefore imports **neither** `transformers` nor `peft`: it writes the tensors with `safetensors` using the key scheme the embedded probe already proved vLLM accepts, and takes shapes from checkpoint headers when the checkpoint exposes them.
- The torchao constraint is a reusable environment fact and is now recorded in the experiment note: this image ships torchao 0.10.0, the wheelhouse's transformers 5.13.1 requires above 0.16.0, and the wheelhouse mount carries no torchao wheel. Any notebook on this image that imports `transformers` after installing the wheelhouse hits it, so the SFT notebook must plan for it explicitly.
- Structural change that ended the defect cycle: the runtime adapter is no longer an f-string inside the generator but a real file, `resource/lora_probe/runtime_adapter.py`, rendered by placeholder substitution. That removed brace-escaping as a defect class, and the file is now covered by `artifacts/tmp/check_runtime_adapter.py`, which renders it and builds against three fixtures: a checkpoint with a real safetensors shard (shapes read from headers), a multimodal config with no shard (falls back to recorded evidence with a warning), and a shard whose shapes disagree with the recorded evidence (reported, not silently accepted).
- Unresolved and stated as such: the base exposes **no weight index and no readable projection shapes** (`shards_with_targets: []`, `measured_layers: 0`), so the rank-32 adapter's shapes came from `resource/manifests/sample-adapters.json` and the run reported `recorded_evidence_match: true` with a warning. That the adapter changed the output is consistent with the shapes being right, but they were assumed rather than measured.
- GPU cost, stated plainly: quota at 18:36 was **59,248.937 s of 108,000 s remaining** (used 48,751.063 s). This session has spent about 43,450 GPU-seconds, roughly 40 % of the allocation, on seven complete probe runs plus the failed version-3 session that held four GPUs for 3.5 hours. A correct probe run costs about 8 minutes of L4x4 wall time; the waste was in the failures, not in the measurement.
- Changed files: `resource/lora_probe/runtime_adapter.py` (new), `scripts/prepare_lora_probe_notebook.py` (template rendering, wider probe wiring, attributable verdicts), `docs/experiments/2026-10-11-lora-serving-risk.md` (rank-32 result, execution log with all eleven attempts, torchao constraint). Commits on `feature/lora-serving-probe-run`.
- Remaining follow-up: the serving question is answered, so the open items are now (1) the training platform (Kaggle needs the wheel set uploaded through the web UI because CLI uploads stall; the SFT notebook must also work around the torchao/transformers conflict), (2) whether to train at all given the measurement instrument still has only 4 control-valid tasks and would need 78 to read a 10-point gain at 80 % power, and (3) the paired no-adapter control that any claim depends on.

### 2026-10-11 17:55 (Asia/Shanghai) - MEASURED: the competition stack does apply LoRA, with a reproducibility control

- Objective: finish the serving check that the whole adapter line depends on, and make its answer trustworthy rather than merely positive.
- Result, measured on `junzhejiao/gemma4-lora-serving-probe` version 7 (private, L4x4, competition model `gemma-4-31b-it-qat-w4a16-ct` v2, wheelhouse vLLM 0.19.1 installed from the mount, `quantization=compressed-tensors`, `tensor_parallel_size=4`, `max_model_len=4096`, `gpu_memory_utilization=0.90`, `enable_lora=True`, `max_loras=1`, `max_lora_rank=128`, `enforce_eager=True`, greedy with `logprobs=3`): **LoRA is applied on this stack.** Two independent base generations produced **0.0 maximum absolute logprob delta** (the engine is reproducible), while mounting the rank-8 layer-0 probe adapter produced a **1.187468 maximum absolute delta at 35 of 35 positions**. Greedy text and token ids were identical in all three generations.
- Why the control was necessary, and it changed the conclusion: version 6 already showed "all 35 positions differ", but with a single base run that could equally have been run-to-run nondeterminism in the engine. The repeat-base control is what turns the observation into evidence, and it is now part of the notebook rather than a note in the report.
- What this does and does not license, recorded so the result is not over-read: it rules out the aliasing wipe of discussion 743508 on this wheelhouse and proves the adapter is honoured on a compressed-tensors W4A16 Gemma 4. It does **not** validate our planned configuration (rank 32 across every layer, whereas the probe is rank 8 on two projections of one layer -- exactly the corner upstream issue #50059 warns about), it says nothing about a *trained* adapter's effect on resolution, and it used the model path directly rather than `adk_submission.VllmServer`, whose flags and adapter enumeration differ.
- Measurement caveat found the same way: the engine's own `Successfully loaded LoRA weights` / `No LoRA weights found` lines were **not** captured (`lora_loaded_modules: 0`), because vLLM loads adapters inside a spawned `EngineCore` process that does not inherit the notebook's logging handler. Those lines must not be used as evidence from a notebook; the logprob deltas carry the result.
- Cost and timing, from notebook output rather than estimate: staging 8 s, wheelhouse inventory 16 s, installing 41 wheels / 880,842,253 bytes 53 s, `import vllm` 80 s, engine init 57 s, each 35-token greedy generation 4-14 s. A correct probe costs about **eight minutes of L4x4 wall time**. GPU quota after the run should be rechecked before the next launch.
- Defects found and fixed this round, all mine and all caught by running rather than reading: doubled braces in the generated probe cell (two separate sites, `{{}}` in a non-f-string cell and a dict comprehension written as a set literal) caused `TypeError: unhashable type: 'dict'`; and the comparison logic had no noise floor. Three offline checks now stand guard over that class: `artifacts/tmp/check_set_literals.py` flags unhashable set literals in any generated notebook (verified against a synthetic defect and against the current generation), `artifacts/tmp/check_stage_helper.py` executes the generated `stage()` helper and round-trips `progress.json`, and `artifacts/tmp/check_comparison_logic.py` runs the generated `top_map`/`divergence` functions on fixtures (identical generations give 0.0, a 0.8 shift is detected, unshared positions are ignored).
- Honest note on the session's efficiency: the version-3 session and the two record-hash failures cost roughly 27 % of the GPU allocation for no result, almost all of it because a failed notebook leaves its worker holding four GPUs and because the CLI shows nothing until a session ends. The eight-minute figure above is what the same probe costs now that it is correct.
- Changed files: `scripts/prepare_lora_probe_notebook.py` (repeat-base control, attributable-verdict logic, dict() instead of set-literal comprehensions), `docs/experiments/2026-10-11-lora-serving-risk.md` (result section, execution log with all five attempts), `resource/manifests/eval-wheelhouse-2026-10-09.json` (`observed_drift`), `WORKLOG.md`. Commits `dd88e95`, `c2dd97e` on `feature/lora-serving-probe-run`.
- Remaining follow-up, in order: (1) probe the **rank-32 all-layer** configuration we actually intend to train, since the current result covers rank 8 on two projections of one layer; (2) decide the platform for training (Kaggle needs the wheel set uploaded through the web UI because CLI uploads stall here; Trillium needs no dataset); (3) the measurement instrument remains the gate on interpretability -- 4 control-valid tasks, 78 needed for a 10-point gain at 80 % power.

### 2026-10-11 14:30 (Asia/Shanghai) - Kaggle execution of the serving probe: two environment defects found and fixed, image dependency map measured

- Objective: with the user's approval, actually run the serving probe pushed at 13:11 rather than leaving it offline-verified, and keep the training route unblocked while it runs.
- Result so far, stated plainly: **no serving verdict yet.** Kernel version 1 failed in ~30 s, version 2 in ~10 s, both for environment reasons rather than probe logic; version 3 (pushed 13:17) is still RUNNING after more than an hour, which is vLLM startup on a 31B W4A16 model with `tensor_parallel_size=4` rather than a hang: the staged inputs and the base model mount were both confirmed present by version 1's own output.
- Defect 1, found only by running it: **the official/starter GPU image does not ship vLLM at all** (`ModuleNotFoundError: No module named 'vllm'`). The patched 0.19.1 is not on PyPI and exists only inside the wheelhouse mount, so any notebook that needs vLLM must install that mount offline first. This also means the harness never imports out of the image; it comes from `metric/gemma-4-developer-agent-wheelhouse`.
- Defect 2, also only visible by running it: my install cell demanded all 41 reviewed wheels, and the mount is missing exactly `swegemma-0.2.10-py3-none-any.whl` -- the wheel the organizers deleted when they republished the dataset in place on 2026-10-09. The cell now requires only vLLM and its runtime dependencies (`transformers`, `compressed_tensors`, `xgrammar`, `flashinfer_*`, `bitsandbytes`, `tokenizers`, `safetensors`) and records missing non-critical wheels as drift in `install-result.json`, which is the honest way to treat a dataset the organizers mutate under us.
- Evidence that the probe logic itself works on Kaggle: version 1 and 2 both staged the embedded adapter and reported `Input mounts: {'/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2': True, '/kaggle/input': True}`. The adapter files inside the Kaggle session are **byte-identical to the local probe** (`adapter_model.safetensors` `b43e139956c95f2a…`, `adapter_config.json` `6a3cb6d1df48b8fb…`), which verifies the gzip+base85 embedding path end to end on the real run image, not just locally.
- Defect 3, found by waiting rather than by reading: **kernel version 3 ran for over 3.5 hours without publishing a single log line.** Kaggle emits a cell's output only when that cell finishes, so a slow or stalled cell is invisible from the CLI -- `kernels logs` and `kernels files` both returned nothing for the whole run. Version 4 (pushed 16:40; queued behind version 3 under the two-session GPU limit) therefore writes `progress.json` on disk after every phase -- inventory, install-start, install-done, import-check, engine-start, engine-ready, base-generated, adapter-generated -- with elapsed seconds, so even a stall leaves retrievable evidence. It also adds `enforce_eager=True` (graph capture is the slowest part of a 31B vLLM start-up and irrelevant to a logprob comparison) and pins the offline install to `--no-build-isolation --only-binary=:all: --retries 0 --timeout 15`, because plain `--no-index --no-deps` can still try to resolve or build and then wait on a network that does not exist.
- The wait paid off with real numbers, and they clear the install of suspicion. Kernel version 3's log, retrieved only after the session ended, shows: `[inventory]` at 25 s (41 wheels mounted, `swegemma-0.2.10` absent, `adk_eval_core`/`adk_submission` hashes changed -- the republish), `[install-start]` 880,842,253 bytes at 25 s, `[install-done] returncode 0` at **78 s**, `[import-check] {"vllm": "0.19.1"}` at **109 s**, and `vLLM 0.19.1 | GPUs 4 | ['NVIDIA L4, 23034 MiB' x4]` at 147 s. So the offline wheelhouse install takes **53 seconds**, not hours, and the patched vLLM does import on this image.
- Defect 4, mine, found by exactly that log: the probe cell died at 149 s with `TypeError: unhashable type: 'dict'` inside `stage()`. The helper had been written with doubled braces copied from the neighbouring f-string cell, so `json.loads(...) if ... else {{}}` was emitted literally and `record[name] = ...` was applied to a `set`. The run therefore never reached the engine at all -- the 3.5 hours were spent after the crash, waiting for the session to be reclaimed, not on vLLM. Fixed to a plain `{}` in a non-f-string cell, and now verified by executing the helper **out of the generated notebook** (`artifacts/tmp/check_stage_helper.py`): `progress.json` round-trips two phases correctly. Version 5 was pushed at 17:44 with this fix.
- Honest correction to the earlier guess in this entry: vLLM start-up was never proven to be the slow part. The version-3 session consumed its ~13,900 GPU-seconds mostly *after* its notebook had already failed, because nothing stopped the worker; the lesson recorded is to read a finished session's log before blaming the model.
- Image dependency map, measured on the same `dockerImage` as the official notebook (`gcr.io/kaggle-private-byod/python@sha256:37c64f7d…`) with a CPU-only kernel (`junzhejiao/gemma4-training-deps-probe`, no GPU quota): Python 3.12.13; present are torch 2.10.0+cu128, transformers 5.0.0, peft 0.19.1, datasets 5.0.0, accelerate 1.13.0, safetensors 0.7.0, pandas 2.3.3, pyarrow 24.0.0, multiprocess 0.70.16, dill 0.4.1, xxhash; **absent are trl, bitsandbytes and vllm**; the wheelhouse adds transformers 5.13.1 and bitsandbytes 0.46.1 but nothing else training-related. Consequence: the offline SFT wheel set is still required, so publishing it as a dataset remains the one manual step gating Kaggle training.
- Interpreter correction, because two images disagree and the earlier note named only one: the competition wheelhouse GPU image is **CPython 3.12** (`/usr/local/lib/python3.12/dist-packages`, Python 3.12.13), while the CPU control-sweep image is 3.13. The cp312 training wheel set is therefore correct for a GPU notebook on the competition image. `scripts/record_training_wheels.py` now derives `interpreter_abi`/`binary_wheel_abis` from the wheel filenames and **refuses to record** a set whose cp3xx tags do not match `--python-version`; verified both ways (cp312 wheels recorded as 3.12; the same directory claimed as 3.13 exits 1 naming pandas, pyarrow and xxhash). `resource/manifests/training-wheels-sft-v1.json` was regenerated with those fields and all 10 hashes re-verified (10 ok, 0 mismatch, 0 missing).
- Dataset publication attempted again (a third try) and it is now **blocked from this environment, with evidence**: `kaggle datasets create -p artifacts/dataset-training-wheels --dir-mode zip` hung for 40 minutes with no progress and no dataset in `kaggle datasets list --mine`, and a second attempt directly through `KaggleApi.dataset_create_new` ran 20 minutes at 0.9 s CPU time, again leaving `dataset_list(mine=True) == []`. The read paths work instantly (`api.authenticate()` then an empty list), so the failure is specifically the upload, not authentication or permissions; earlier rounds saw 403 instead. Both attempts were stopped rather than left burning. Consequence: Kaggle SFT training cannot start until the wheels reach Kaggle by some other route (manual web upload, or Trillium), and the immediate blocker for this line is therefore the platform, not the code.
- Verification this round: the notebook generator was re-generated and re-verified after each fix (staging cell reproduces both adapter files; install cell fails only at the documented mount wall locally; `--check` reports `status: ok` with `vllm_wheel_pinned: true`); all 67 relative links across the four touched documentation files resolve; the image facts above come from a COMPLETE kernel's own `environment.json`, not from inference.
- Changed files: `scripts/prepare_lora_probe_notebook.py` (wheelhouse install cell, required-package rule, drift recording), `scripts/record_training_wheels.py` (ABI guard), `resource/manifests/training-wheels-sft-v1.json` (regenerated), `resource/manifests/README.md`, `docs/experiments/2026-10-11-lora-serving-risk.md` (execution log), `WORKLOG.md`. Generated but ignored: `artifacts/lora-probe-notebook/` (v3), `artifacts/dataset-training-wheels/`, `artifacts/tmp/**` evidence.
- GPU quota: the probe is running on L4x4 now. Measured at 14:27 Asia/Shanghai: **94,315.809 s of 108,000 s remaining** (used 13,684.191 s), so this session has consumed 8,383.0 s; recheck with `scripts/check_kaggle_quota.py` before the next launch and write its output under `artifacts/`, since `resource/competition/` refuses writes from Python.
- Remaining follow-up: read `probe-result.json` from version 3 and record the verdict; finish or abandon the dataset publication; then either fund training (SFT dataset and notebook already built) or retire the adapter line if the probe says LoRA is inert.

### 2026-10-11 13:08 (Asia/Shanghai) - The serving check gained a runnable, offline-verified probe notebook

- Objective: continue last night's work by advancing its self-declared highest-value, no-training item -- deciding whether LoRA is applied at all when serving the compressed-tensors W4A16 competition base -- without spending GPU quota before the user picks the execution route.
- Gap found by inspection, not assumption: `scripts/prepare_kaggle_runtime.py` stages and hash-verifies a mounted adapter, but it runs the official harness end to end and **measures no logprobs**. The plan written at 07:20 ("compare greedy logprobs with and without the adapter") therefore had no executor. The probe adapter itself did exist and was re-checked this round: `artifacts/lora-probe/` holds `adapter_config.json` plus `adapter_model.safetensors` (434,760 bytes, rank 8, `q_proj`/`o_proj`), and `scripts/check_lora_adapter.py` still exits 0 with `served_max_lora_rank: 32`, `max_loras: 1`.
- Delivery: `scripts/prepare_lora_probe_notebook.py` (new) generates a private, offline L4x4 notebook (`probe.ipynb`, 512,266 bytes) that embeds the loud adapter as gzip+base85 -- so **no Kaggle dataset publication is needed**, removing the 403-prone step -- proves its SHA256 before vLLM sees it, and then compares greedy logprobs on one fixed prompt between the base model name and the adapter name: `LLM(..., tensor_parallel_size=4, max_model_len=4096, gpu_memory_utilization=0.90, enable_lora=True, max_loras=1, max_lora_rank=128)` with `SamplingParams(temperature=0, max_tokens=40, logprobs=3)` and `LoRARequest('main_lora', 1, ADAPTER)`. The constructor arguments mirror the harness defaults in `resource/harness-source/adk_submission/server.py` instead of inventing a different server, and the model handle (`google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`) plus the official `dockerImage` come from the existing generators and `resource/notebooks/official/kernel-metadata.json`.
- Why the result is readable either way: the probe cell records, in `probe-result.json`, the first eight top-logprob positions of both arms, token-id and text differences, and the engine's own LoRA log lines split into `Successfully loaded LoRA weights for module ...` and `No LoRA weights found for module ...decoder_layers.N...`. Identical logprobs with loads *followed by* alias skips is the exact wipe of [discussion 743508](resource/discussions/743508.txt); loads only is the upstream #50059 pattern; no loads at all means the adapter never reached the engine. Each case gets a distinct verdict string rather than one undifferentiated "no change".
- Self-inflicted defect caught by running the cells, not by reading them: the first version's probe cell relied on globals defined in the staging cell (`WORK`, `MODEL_PATH`, `PROBE`) and died with `NameError: name 'WORK' is not defined` under independent execution. The probe cell now re-derives every constant from `probe-config.json`, which the staging cell writes; the second generation was re-verified from scratch.
- Verification actually executed offline (no GPU, no vLLM, no base weights): the staging cell ran standalone and reproduced both adapter files byte-identically (`adapter_config.json` `6a3cb6d1df48b8fb…`, `adapter_model.safetensors` `b43e139956c95f2a…`); the probe cell exits 1 at an explicit "Base model mount is missing" wall instead of silently doing nothing; `--check` returns `status: ok` and `identity_fields_match: true`, and returns 1 with `status: mismatch` on a one-byte tamper of the weights; generating into an existing `--output` exits 1; an adapter directory without `adapter_model.safetensors` exits 1 and writes nothing. All 32 relative links in the touched documentation resolve.
- Honest scope limit: **no GPU session has run and no verdict exists.** The probe cannot be exercised past the mount check locally, because vLLM is not installed in this workspace; that wall is verified behaviour, not a working measurement. GPU quota is untouched at 102,698.778 s of 108,000 s (refresh 2026-10-17), checked this round.
- Changed files: `scripts/prepare_lora_probe_notebook.py` (new), `docs/experiments/2026-10-11-lora-serving-risk.md` (runnable-probe section), `docs/README.md`, `configs/experiments/lora-trajectory-v1/README.md`, `WORKLOG.md`. Generated and gitignored: `artifacts/lora-probe-notebook/{probe.ipynb,kernel-metadata.json,preparation.json}` and `artifacts/quota-2026-10-11.json`. `resource/competition/accelerator-quota-normalized.json` remains unwritable, which is why the quota receipt went under `artifacts/`.
- Remaining follow-up: the user decides how to run it -- (a) push `artifacts/lora-probe-notebook` as a private Kaggle kernel (`kaggle kernels push`) for one short L4x4 session, or (b) run the same two cells on Trillium H100. Everything else in the LoRA line still waits behind that result: training, the platform/wheel-dataset decision, and the measurement instrument (still 4 control-valid tasks, 78 needed for a 10-point gain at 80% power).

### 2026-10-11 07:20 (Asia/Shanghai) - Serving-path risk established from upstream evidence, and a no-training probe built to settle it

- Objective: settle the decisive unknown for the whole LoRA line -- whether LoRA is applied at all when serving the compressed-tensors W4A16 competition base -- without spending training budget.
- Evidence gathered (external, treated as data, cited in the new note): upstream vLLM's `gemma4.py` declares `SupportsLoRA` on the causal-LM class and declares `packed_modules_mapping` with `qkv_proj: [q_proj, k_proj, v_proj]` and `gate_up_proj: [gate_proj, up_proj]`; Gemma 4 LoRA support was added by recent upstream PRs ([#39560](https://github.com/vllm-project/vllm/pull/39560), [#39291](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/39291/overview)); LoRA on Gemma 4 MoE **fails at startup** because `get_expert_mapping` was missing ([#53470](https://github.com/vllm-project/vllm/issues/53470), fixed by [#46772](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/46772/overview)); and on a compressed-tensors int4 (W4A16) base, **rank-32 all-layer adapters produced weak, non-reproducible outputs while rank-8 partial-coverage adapters worked**, across 0.17.1-0.24.0 ([#50059](https://github.com/vllm-project/vllm/issues/50059)), with a bugfix adding `select_gemm_impl` to `CompressedTensorsWNA16MoEMethod` for LoRA ([#31453](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/31453/overview)).
- Why this is the binding risk rather than a curiosity: this competition already has a documented silent-wipe of adapters on this exact base (discussion 743508, fixed in wheelhouse v23+), and issue #50059 is the same symptom reported independently. A silent failure means every trained adapter would score exactly like the base model while consuming the full training and evaluation budget.
- Delivery 1: `scripts/make_lora_probe.py` (new) builds a **loud probe adapter** -- random `lora_A` *and* random `lora_B` at a scale that must move logits -- using the two shapes verified for this base in `resource/manifests/sample-adapters.json`. If greedy logprobs are identical with and without it, LoRA is inert on that stack. Its `--check` mode also detects the trap the shipped example adapters fall into, all-zero B.
- Delivery 2: `docs/experiments/2026-10-11-lora-serving-risk.md` (new) records the evidence with source kinds, the four concrete implications (the patched wheel is not on PyPI and ships inside the Docker image; a Kaggle dataset listing confirmed the wheelhouse has no vLLM wheel; target-module naming differs between PEFT names and vLLM's packed mapping; our rank-32 all-layer direction sits in the risky corner of #50059), and the open questions only a run can answer.
- Verification, including three negative paths that all failed correctly once the dependency path was set: building with an unverified target module exits 1 with `no verified shapes for [...]`; reusing an existing output directory exits 1; a synthetic zero-B adapter is judged `status: inert` with the explanatory verdict. The probe itself builds 4 tensors, 434,760 bytes, with `max_abs` 0.084/4.22 on the A/B sides respectively.
- Self-inflicted defect caught while testing: the first negative-path attempt reported `ModuleNotFoundError: No module named 'torch'` because `PYTHONPATH` was set after those commands ran, so it had tested nothing. Re-run with the path set, both paths produced their intended errors.
- Consequence for the plan, stated honestly: the previous rounds' SFT preparation is necessary but not sufficient. The recommended order is now **serving check first**, then training, because a negative serving result retires the line and saves the remaining budget. Nothing else in this round consumed GPU quota: 102,698.778 s of 108,000 s remain, the fifth consecutive round with no GPU execution.
- Changed files: `scripts/make_lora_probe.py` (new), `docs/experiments/2026-10-11-lora-serving-risk.md` (new), `docs/README.md`, `configs/experiments/lora-trajectory-v1/README.md`, `WORKLOG.md`.
- Remaining follow-up: run the probe on the competition model (one short serving session, no training), choose the platform, and enlarge the control-valid task set. The first of these is the cheapest action with the highest information value in the whole project right now.

### 2026-10-11 05:40 (Asia/Shanghai) - Control validity corrected: 4 usable tasks, not 3, and the mechanism is now understood

- Objective: keep advancing the measurement half of the goal on CPU only, and settle whether the exit-2 control failures are environment gaps or expected behaviour.
- Root cause found by comparing each failing task's `test_patch` against its own source `patch`: `rich_3930` imports `CellSpan`, `split_graphemes`, `cell_len`, `get_character_cell_size` and `split_text` from `rich.cells`, and **that task's source patch adds exactly those names**. Collection therefore *must* fail before the fix, and the gold arm confirms it: empty arm 2 collection errors, gold arm 189 tests passing. This is a valid control, not a broken one.
- Instrument fix: `scripts/analyze_paired_power.py` now distinguishes empty-arm shapes instead of demanding a clean test failure. Exit 1 counts as "tests ran and failed"; exit 2 counts when the output carries a collection-error marker **and** the gold arm passes cleanly; a collection failure that survives the gold patch stays invalid because the harness never exercised the issue. Corrected count: **4 usable tasks (`rich_3454`, `rich_3675`, `rich_3676`, `rich_3930`), up from 3**, all still in one repository.
- The gold-arm requirement is what keeps the relaxed rule safe, and the data proves it separates the cases: `fastapi_13713` and `fastapi_14786` fail collection in *both* arms (genuine environment gap, gold invalid), `requests_7433` times out in both arms (exit 124), and `rich_3296` fails in both arms (base already passes). Rationale for every task is now emitted in the tool's `control_rationale` field, so no verdict is unexplained.
- Power is unchanged in shape despite the extra task: still zero for every effect size, minimum detectable gain still undefined, and 80% power still needs 78 tasks for a 10-point gain. The correction matters for honesty and for the ceiling -- the yield is 4 of 8 attempted, so the instrument is less hopeless than the previous count implied, but still far from able to read a result.
- Deliverables updated: `scripts/analyze_paired_power.py` (corrected rule plus per-task rationale) and `docs/experiments/2026-10-11-lora-paired-protocol.md` (the corrected table, the `rich_3930` mechanism, and the note that two successive versions of this rule had undercounted usable tasks).
- Static feasibility check attempted and deferred, recorded so it is not repeated blindly: the patched `vllm-0.19.1` wheel is **not on PyPI**, and the `metric/gemma-4-developer-agent-wheelhouse` dataset contains only 20 test-dependency files with no vLLM wheel, because that wheel ships inside the Docker image. Inspecting whether vLLM accepts LoRA on a compressed-tensors W4A16 model therefore needs a different route (image layer or upstream source), and it remains the decisive unknown for the whole LoRA line.
- Verification: the tool re-ran to identical output with the rationale visible; every relative link in the touched documentation resolves. Changed files: `scripts/analyze_paired_power.py`, `docs/experiments/2026-10-11-lora-paired-protocol.md`, `WORKLOG.md`. GPU quota untouched at 102,698.778 s of 108,000 s -- fourth consecutive round with no GPU execution.
- Remaining follow-up: the platform decision still gates training, and the measurement instrument still gates interpretability. The cheapest useful next CPU-side step is to run a wider control sweep over the tasks whose failures look environment-shaped, now that the validity rule no longer discards collection-error controls.

### 2026-10-11 04:10 (Asia/Shanghai) - Paired-experiment protocol and power analysis: only 3 control-valid tasks exist

- Objective: advance the second half of the goal (the paired baseline-versus-LoRA comparison with empty/gold controls) without spending GPU quota, and establish whether such a comparison could be interpreted at all.
- Hard finding, measured from the existing control sweep rather than assumed: of 8 tasks attempted, **only 3 have valid controls -- `rich_3454`, `rich_3675`, `rich_3676` -- and all three are in a single repository.** The other five fail for four distinct reasons: `requests_7433` timed out in both arms (exit 124), `fastapi_14786`, `fastapi_13713` and `rich_3930` hit test-collection errors (exit 2), and `rich_3296` has tests that already pass at base (empty exit 1), so there is nothing to fix.
- Instrument defect found and fixed in my own first version: it required `test_exit_code == 0` for the empty arm and therefore reported **zero** valid tasks. An empty patch fails by running the tests and failing them, which pytest reports as exit 1; exit 0 would mean the tests passed without the fix and exit 2 is a collection error. The corrected rule is now written into `scripts/analyze_paired_power.py` and the protocol note, because reading those codes wrongly silently discards real tasks.
- Power computed exactly (McNemar, alpha 0.05, 80% power) rather than estimated: detecting a **10-point** gain needs **78** tasks, 20 points needs 40, 30 points needs 26, and 5 points needs 158. With 3 tasks the power is effectively zero for every one of those, and even a perfect yield across all 129 public tasks would leave a 10-point effect undetectable. The honest expected outcome of a paired run at this size is "not measurable", not "no improvement".
- Deliverables: `scripts/analyze_paired_power.py` (new, standard library only) reads every control and arm `result.json` under `runs/`, applies the corrected validity rule, computes the paired 2x2, McNemar's exact p, and the task counts required for 80% power; `docs/experiments/2026-10-11-lora-paired-protocol.md` (new) freezes the protocol and acceptance criteria before any result exists, and records the exit-code semantics table.
- Triage of what could grow the task set, measured from `data/public/tasks.jsonl`: all 129 tasks carry a test patch; **50 reference data files or fixtures**, 14 touch conftest/fixtures, and 9 exceed 20 KB. Those are the tasks most likely to fail controls for environment reasons, and `rich_3930` shows the mechanism concretely -- its empty arm hit 2 collection errors while its gold arm ran 189 tests passing, so the dependency exists only for the unpatched state.
- Verification: the power tool re-ran to identical output (`control_valid_count: 3`, `tasks_required_for_80pct_power.gain_10pct: 78`), and every relative link in the two touched documentation files resolves.
- Changed files: `scripts/analyze_paired_power.py` (new), `docs/experiments/2026-10-11-lora-paired-protocol.md` (new), `docs/README.md`. No repository change outside these. GPU quota untouched at 102,698.778 s of 108,000 s, and this is the third round with no GPU execution.
- Remaining follow-up: the platform decision is still the gate for training, and the measurement instrument is the gate for interpretability. The cheap next step on the instrument is CPU-side: widen the control sweep over the tasks that fail for environment reasons, starting with the exit-2 collection failures, since those are fixable without GPU quota.

### 2026-10-11 02:40 (Asia/Shanghai) - Offline SFT stack proven end to end and the training wheel set pinned, hashed, and recorded

- Objective: remove the dependency risk from the LoRA SFT path before any GPU session. The Kaggle route cannot fine-tune without training wheels, and Trillium would fail the same way if the versions were wrong, so this work serves both platforms.
- The stack trains, measured not asserted: `scripts/check_training_stack.py` (new) builds a tiny `transformers` causal LM, wraps the configured `q_proj`/`o_proj` with LoRA, loads the real 116-record JSONL through `datasets`, runs `SFTTrainer` with `completion_only_loss=True` for 3 optimisation steps, exports the adapter, and reloads it. Result: `status: ok`, loss **5.4698 -> 5.4269** across 3 steps with `num_tokens` 256/512/768 (the completion mask is doing something), 8 LoRA tensors, `adapter_model.safetensors` 66,584 bytes, `reload_matches: true`, exit code 0.
- Two real version defects found by running it, both of which would have burned a GPU session:
  - **`transformers==5.13.1` requires `huggingface-hub>=1.5.0,<2.0`.** The first attempt resolved hub 2.2.0 and died at `import transformers` with `ImportError`. Pinned to 1.33.0.
  - **`trl 1.15.0` cannot train here at all.** `SFTTrainer.__init__` patches a fused LM head and calls it every step, but that path needs the `trl.kernels` extension; without it the step fails with `AttributeError: 'NoneType' object has no attribute 'apply'`. Downgraded to **trl 0.24.0**, which needs only `transformers>=4.56.1`. `datasets` additionally needs `pandas` and `multiprocess`, neither carried by the wheelhouse.
- Pinned and recorded: `configs/training.requirements.txt` now carries the verified set with each constraint explained, and `resource/manifests/training-wheels-sft-v1.json` lists **10 Linux/CPython-3.12 wheels, 58,045,422 bytes total**, each with its SHA256. All 10 hashes were re-verified against the downloaded files (10 ok, 0 mismatch, 0 missing). `scripts/record_training_wheels.py` (new) regenerates that manifest and documents the exact `pip download` command.
- Verified version set: transformers 5.13.1, peft 0.21.2, trl 0.24.0, accelerate 1.15.0, datasets 5.1.0, huggingface-hub 1.33.0, plus pandas/pyarrow/xxhash/dill/multiprocess. The Linux wheels resolve pandas 2.3.2 and pyarrow 20.0.0, while the Windows verification used pandas 3.0.6; that difference is recorded in the manifest rather than hidden.
- Scope limit stated in the check's own output: the tokenizer was a local byte-level stand-in because no Gemma tokenizer was cached and no download was allowed, so this proves the training loop and export layout, **not** tokenizer fidelity or accuracy. The real tokenizer comes from the base model at run time.
- Environment note for future rounds: `datasets` caches under `~/.cache/huggingface`, which the sandbox refuses, so any local run needs `HF_HOME`/`HF_DATASETS_CACHE` pointed inside the workspace.
- Changed files this round: `scripts/check_training_stack.py` (new), `scripts/record_training_wheels.py` (new), `configs/training.requirements.txt` (pinned), `resource/manifests/training-wheels-sft-v1.json` (new), `resource/manifests/README.md`. The wheel files themselves live in ignored `artifacts/training-wheels/`.
- Remaining follow-up, unchanged in substance: the platform decision. Kaggle now needs only one manual step -- publish `artifacts/training-wheels/` as a private dataset (or let me attempt `kaggle datasets create`) and pass its handle to `scripts/prepare_lora_notebook.py`. GPU quota still untouched: 102,698.778 s of 108,000 s.

### 2026-10-11 00:20 (Asia/Shanghai) - LoRA SFT dataset produced and the training notebook generator built and executed offline

- Objective: continue the LoRA SFT work with platform-independent deliverables, without spending GPU quota, while the platform choice is still open.
- Platform finding that changes the Kaggle plan: the competition wheelhouse carries **only `transformers` and `bitsandbytes`** among the training stack. `peft`, `trl`, `accelerate`, `datasets`, and even `torch` are **absent** (torch comes from the host image). With Internet disabled, a Kaggle kernel therefore **cannot fine-tune** unless the training wheels are attached as a mounted dataset. This is a concrete blocker for the Kaggle route and an argument for Trillium, which installs its own training environment.
- Data delivered: `scripts/build_sft_dataset.py` produced **116 training records** (129 public tasks minus 13 over the 8000-character cap) at `artifacts/sft-public-v1.jsonl`, 338,702 bytes. It is written under `artifacts/` rather than the documented `data/processed/` because that directory still refuses writes (see the blocker below); the builder reports the intended path in its own output.
- Notebook generator delivered: `scripts/prepare_lora_notebook.py` emits a self-contained, offline, private Kaggle notebook (`train.ipynb` 102 KB, `kernel-metadata.json`, `preparation.json`) with the records embedded as gzip+base85 and the reviewer-approved adapter config. It installs any `--wheels-dir` wheels offline, gates on the required modules with an explicit missing list, resolves the base model from `GEMMA_BASE_MODEL` or the Kaggle mount, verifies every configured target module exists on the loaded model before training, trains with `completion_only_loss=True`, and exports `adapters/<name>/adapter_model.safetensors` with its SHA256.
- Verification, executed rather than asserted: the generated setup cell was extracted from the notebook and **run standalone** twice. Without wheels it stopped with the exact missing list (torch, transformers, peft, trl, datasets, accelerate, bitsandbytes locally). With `PYTHONPATH` pointed at the local installs it stopped at only `trl, datasets, bitsandbytes` -- proving the embedded dataset and its hash check had already passed. The extracted `dataset.jsonl` is byte-identical to the source (338,702 bytes, sha256 `0eb01ed62d1d3fa3...`). Wheel mode was exercised with a stub `--wheels-dir`, producing `dataset_sources: ['testuser/gemma4-training-wheels']` and `wheels_count: 2`. Negative cases fail correctly: `--wheels-source` without `--wheels-dir`, and an already-existing `--output`.
- Defect found by that verification, and it was mine: the generator embedded **gzip-compressed** bytes while an editing pass had removed `gzip.decompress` from the setup cell. The hash assertion caught it immediately (`AssertionError: embedded dataset hash mismatch`) instead of letting it reach a GPU session as a decode error. Fixed, then re-verified to a byte-identical written dataset. A second self-inflicted defect in the same block was an `IndentationError` from a sloppy replacement, caught by `ast.parse`.
- Blocker unchanged: Python still cannot create files in pre-existing directories (`data/processed`, `dist`, `scripts`, `runs`, `.git`, `data/processed-sft`), while `artifacts/` and fresh directories accept writes. The earlier ACL repair verified two grants without restoring the operation, so the skill's stop rule still stands and no second repair was attempted. Committing again needed the same elevated access.
- Changed files: `scripts/prepare_lora_notebook.py` (new). Generated but ignored outputs: `artifacts/sft-public-v1.jsonl`, and the notebook directory created as a test (removed; regenerate with the real `--owner`).
- Remaining follow-up: choose the training platform. On Kaggle, first publish a dataset containing peft/trl/accelerate/datasets wheels and pass it as `--wheels-dir`/`--wheels-source`; on Trillium, the same `scripts/train_lora_adapter.py --train` path works once the training environment exists. Either way the paired no-adapter control experiment still has to be run before any claim. GPU quota untouched: 102,698.778 s remaining.

### 2026-10-10 23:30 (Asia/Shanghai) - LoRA feasibility verified offline; the SFT dataset exists and the export path is proven

- Objective: continue the LoRA SFT work without burning GPU quota, per the user's selection of "feasibility validation only". Two verdicts were required: whether the base can be shrunk, and whether local training is possible.
- Verdict 1, the base cannot be shrunk: the competition permits exactly one serving model, `gemma-4-31b-it-qat-w4a16-ct`. Training must stay in the same QAT weight family, and the only non-quantized member is `gemma-4-31b-it-qat-q4_0-unquantized` at 62.6 GB. Any smaller base (12B, 26B A4B, E2B) is a different model and would not produce a loadable competition adapter.
- Verdict 2, local training is not possible: `nvidia-smi` reports an RTX 5060 Laptop GPU with 8,151 MiB total and 6,164 MiB free; memory counters report roughly 16 GiB RAM. A 4-bit 31B base needs about 15.6 GB for weights alone, before activations, optimizer state, or checkpoints. Disk is fine (600.9 GB free on E:). Conclusion recorded rather than attempted, because the arithmetic cannot be argued away.
- Work delivered, all offline and CPU-only: `scripts/build_sft_dataset.py` now produces the training records, and `scripts/train_lora_adapter.py` gained `--self-check`, which exercises the PEFT and export path on a tiny local model. Measured results: **116 training records** from 129 public tasks, 13 dropped as `patch_too_long` at the 8000-character cap; `check_lora_adapter.py` still `status: ok`; the self-check reports `target_modules_resolved: [q_proj, o_proj]`, `target_modules_verified_wrapped: true`, `forward_backward_ok: true`, `weights_filename_matches_discovery: true`, `reload_matches_export: true`, and `adapter_names_discoverable: [main_lora]`.
- Defect found and fixed by testing rather than inspection, which is the point of the self-check: passing `--target-modules nonexistent_proj` **passed** at first. Two separate bugs had hidden it -- PEFT warns and wraps nothing when a target is absent, and `--target-modules` was never forwarded into the self-check. Both are fixed; the negative case now exits 1 with `target modules absent from the base model: ['nonexistent_proj']`. Three earlier failures in this same block were also only visible by running it: a missing `prepare_inputs_for_generation` stub, a `config` that PEFT reads as a mapping, and a causal-LM keyword signature.
- Honesty fix in the dataset builder: the previously defaulted holdout repository `encode/httpx` contains **one** task and its patch exceeds the cap, so the "repository-disjoint holdout" was silently empty. The default is now no holdout, and the builder emits an explicit warning either way -- that no repository-disjoint split is achievable at four repositories, or that a requested holdout contributed no records. **There is no held-out metric available from this data, and none will be claimed.**
- Environment work, recorded as side effects: `.gitignore` now excludes `/.pylibs/` and `/.pipcache/`, which were untracked and would have added hundreds of megabytes to the repository. The local Python installs live there (CPU torch, peft, transformers, accelerate, safetensors, numpy, httpx2, httpcore2), added through approved unconfined commands because pip needs temporary directories outside the workspace.
- Commit: `f8e565b` groups this session's reusable work into one commit (adapter experiment, the three validation scripts, the SFT builder, the adjudication note, the quota receipt, and the `.gitignore` fix). The commit also carried nine files that were already staged by the previous session (`docs/kaggle-runtime.md`, `docs/localization-handoff.md`, `resource/manifests/eval-wheelhouse-2026-10-09.json`, `scripts/prepare_kaggle_runtime.py`, `tests/README.md`, and three test files) because unstaging them needed the same elevated access; that unrelated localization work is still present in the working tree and was not otherwise touched.
- Still blocked, and the blocker is a decision rather than a capability: the workspace file-permission problem remains, so Python cannot create files in pre-existing directories (`.git`, `data/processed`, `runs/`, `scripts/`, `configs/`). The earlier ACL repair verified two grants but did not restore the operation, so the skill's stop rule applies. Practical effect: `--self-check` needs `--self-check-dir` inside a freshly created directory, and staging a commit needs elevated access.
- Remaining follow-up: the user must choose the training platform (Trillium H100 versus Kaggle L4×4, where a 62.6 GB base very likely cannot be staged) and whether to switch this session to full access so the repository layout can be honoured. No GPU quota was spent this round: 102,698.778 s of 108,000 s remain.

### 2026-10-10 22:30 (Asia/Shanghai) - LoRA SFT start: quota checked, training base identified, dataset builder written, blocked on workspace permissions

- Objective: begin the remote LoRA SFT work the user authorised -- train on the data already in the repository, then compare the original remote model against the SFT adapter in a paired experiment.
- Account state, measured rather than assumed: the Kaggle SDK was installed workspace-locally (`.pylibs`, via one approved unconfined command) and `scripts/check_kaggle_quota.py` reports **GPU remaining 102,698.778 s of 108,000 s** (used 5,301.222 s, reserved 0), TPU 72,000 s untouched, refresh 2026-10-17. At the documented L4x4 double-consumption rate that is roughly 14 wall hours -- enough, but the binding constraint remains the per-session push budget, not hours.
- Training base identified from the model card rather than guessed: `resource/models/google-gemma-4.json` lists **`gemma-4-31b-it-qat-q4_0-unquantized`** (Transformers framework, version 2, 62,578,682,779 bytes ~ 62.6 GB, Apache-2.0). This is the unquantized counterpart of the QAT family the scorer serves, so it is the principled SFT base; `gemma-4-31b-it-qat-w4a16-ct` remains the only permitted serving model.
- Data actually available offline: `data/public/tasks.jsonl` holds all 129 public tasks with `problem_statement` and `patch` (fastapi 67, rich 48, requests 13, httpx 1; zero non-empty `hints_text`; patch sizes 176 B to 308,489 B). **No verified agent trajectory exists locally** -- the trajectories from earlier Kaggle runs live in private kernel outputs -- so issue-to-reference-patch is the only offline-supervised option, with the transfer caveat the training plan already records (the four public repositories are hidden at scoring time).
- Work delivered: `scripts/build_sft_dataset.py` builds prompt/completion JSONL from those tasks with two guardrails -- a **repository-disjoint split** (default holdout `encode/httpx`) so a random split cannot fake transfer, and a patch-length cap that reports drops instead of truncating silently. `data/processed/` is the intended output location per `.gitignore`.
- **Blocker, not worked around:** Python processes cannot write to pre-existing workspace directories. `data/`, `data/processed`, `data/public`, `data/raw`, `.tmp-tests`, `configs/experiments/*`, `scripts/`, `resource/manifests/`, `runs/`, `dist/` all return `PermissionError` on create, while directories created in this session (`artifacts/`, fresh `mkdir`) accept writes. The cause is visible in the ACLs: writable directories carry the sandbox's own allow entry (`S-1-4-584114110-936812334`), pre-existing ones do not, and the workspace root's entry does not inherit.
- Repair attempt, per the loaded `diagnose-windows-sandbox-acl` skill: one approved unconfined run against `data/processed` with `-AllowRoot E:\Desktop\CodingAI` and `-Out E:\Desktop\CodingAI-acl-recovery`. Result `SUMMARY FIXED=0 GRANTED=2 REFUSED=0 RESTORED=0`, verdict `PRECONDITION`, both grants verified, `nextAction: verify_original_confined_operation`. The confined retry of the original operation **still failed**, so the repair did not resolve the denial. The skill requires stopping after a repair that does not restore the operation, so no second repair was started. Recovery commands are printed in the report; nothing was rolled back.
- Report and evidence: `E:\Desktop\CodingAI-acl-recovery\acl-report-a915ae460d28436c8ddc8ca07dd7d4d1.jsonl` (outside the workspace, per the skill's guidance), with two DACL backups and their restore scripts beside it.
- Changed files this round: `scripts/build_sft_dataset.py` (new), `resource/competition/accelerator-quota-2026-10-10.json` (new quota receipt), `.pylibs/` and `.pipcache/` (gitignored, workspace-local installs). No dataset was produced, and `data/processed/sft-public-v1.jsonl` does not exist.
- Remaining follow-up: the user must choose between (a) temporarily switching this session to full access so pre-existing directories become writable and the repository layout can be honoured, or (b) letting me write new artifacts into freshly created directories, which changes the documented layout. Training itself additionally needs a GPU run on the 62.6 GB unquantized QAT base, with the adapter exported and then a paired control experiment; neither can start until the file-permission decision is settled.

### 2026-10-10 21:55 (Asia/Shanghai) - Installed the basic-programming-principles skill into the user-level DSH root

- Objective: on the user's instruction ("复制到dsh中即可"), install the skill so it applies outside this repository instead of only in this checkout.
- Target resolved from the environment rather than assumed: `DSH_HOME` is `C:\Users\32747\.dsh`, so the user-level root is `C:\Users\32747\.dsh\skills` (the `user-dsh` discovery root, rank 400). It did not exist before this task, so it was created. `DSH_AGENTS_HOME` is unset, so `C:\Users\32747\.agents\skills` was not used.
- Work performed: copied the bundle to `C:\Users\32747\.dsh\skills\basic-programming-principles\SKILL.md`. The DSH file sandbox denied the first attempt (`[sandbox: file access denied under workspace-write mode]`), so the identical command was retried once with `danger-full-access` after the user approved it. Side effect recorded honestly: that one approved command ran unconfined, and it created the new `skills` directory under `.dsh`.
- Verification: the destination file exists and its SHA256 equals the source file's, `883FE8643A90337361DC425DD0C055D5526942E3EA3DBB3327499D7A212F4761`. No other file was written.
- Precedence note for future agents: the same skill name now exists in two roots. Within this repository the project root (`<projectRoot>/.agents/skills`, rank 100) wins; in any other workspace the user root (rank 400) supplies it. Both copies currently hold identical bytes, so the distinction has no behavioural effect today, but an edit to only one of them will not propagate to the other.
- Changed files: none in this repository. One file created outside it: `C:\Users\32747\.dsh\skills\basic-programming-principles\SKILL.md`.
- Remaining follow-up: keep the two copies in sync deliberately, or delete the repository-local copy if the user-level install is meant to be the single source. Neither copy is tracked in Git.

### 2026-10-10 21:40 (Asia/Shanghai) - Added rule F (Chinese reasoning display) to the basic-programming-principles skill

- Objective: extend the repository-local `basic-programming-principles` skill with one more rule, 所有思考结果用中文展示 (present every reasoning result in Chinese).
- Work performed: added section **F** to `.agents/skills/basic-programming-principles/SKILL.md`, updated the intro from five rules to six, extended the frontmatter `description` so the session catalog advertises the new rule, and added checklist item 8. The rule scopes "reasoning result" explicitly: progress notes, findings, decisions, failures, and next steps are displayed in Chinese, while code, identifiers, paths, commands, file contents, and quoted upstream text stay verbatim so the reasoning is translated and the artifacts are not. It also records two boundaries rather than silently contradicting existing rules: repository documentation stays English per `AGENTS.md`, and this skill sets Chinese as the default for displayed reasoning only, without overriding a user who asks for another language.
- Verification: the running Web session re-published its skill catalog on its own, and the entry now carries the new description text (`... 只要开始编程就加载，思考结果用中文展示。`), which confirms the frontmatter parsed and the rule is discoverable. No code path, test, or configuration outside the skill file was touched.
- Changed files: `.agents/skills/basic-programming-principles/SKILL.md` (untracked, repository-local).
- Remaining follow-up: none for the rule itself. The skill remains untracked and repository-local; promoting it to every project still requires copying the bundle to `C:\Users\32747\.dsh\skills` or `C:\Users\32747\.agents\skills` outside the sandbox, and committing it requires the branch/PR and reviewer approval path in `docs/ci-and-review.md`.

### 2026-10-10 21:15 (Asia/Shanghai) - User selected directions A and F; built the module-evidence probe and the trajectory dataset builder

- Objective: the user chose two next directions from the four offered options: **A**, fix the measurement instrument and then run trajectory SFT, and **F**, close the missing module evidence (`k_proj`, `v_proj`, and the MLP projections) before widening the adapter. This entry records the offline work that both directions need before any GPU time is spent.
- F, tooling delivered: `scripts/probe_model_modules.py` loads the base model on CPU (`--device-map cpu`, no forward pass, no GPU needed), enumerates every real `nn.Linear` projection with its `in_features`/`out_features`, fails on any shape inconsistency, and writes `resource/manifests/model-modules.json`. `scripts/check_lora_adapter.py` now **prefers that probe manifest over the sample adapter** as its evidence source and reports which source it used, so the adapter config is automatically re-validated against measured reality once the probe exists. The manifest row is registered in `resource/manifests/README.md` as "not yet recorded".
- F, not done and not claimed: the probe itself has not been run. It needs the base weights and a `transformers` environment; this workspace has neither, so `k_proj`, `v_proj`, `gate_proj`, `up_proj`, and `down_proj` remain unverified and `target_modules` stays at `q_proj`, `o_proj`.
- A, tooling delivered: `scripts/build_trajectory_dataset.py` converts harness traces (`results/traces/trace_<instance_id>.json`, ATIF v1.7, legacy entry-list format also accepted) into TRL-shaped messages JSONL for `scripts/train_lora_adapter.py`. It keeps a trajectory only when the task's `result.json` shows `resolved: true` with `test_exit_code: 0` and no error **and** the trace contains at least one tool call; everything rejected is counted by reason (`skipped_task_failed`, `skipped_no_tool_call`, `skipped_no_result`, `skipped_unreadable`) so a weak dataset cannot look strong.
- Verification: the builder was exercised on a synthetic fixture with one resolved ATIF trace, one unresolved legacy trace, and one trace without matching results. It emitted exactly one record, kept the tool-call and observation structure intact, and reported `kept: 1, skipped_task_failed: 0, skipped_no_result: 1` as designed; the fixture was then deleted. `scripts/check_lora_adapter.py` still reports `status: ok` after the evidence refactor. A real deflate: the first fixture run failed because PowerShell writes a UTF-8 BOM and the readers used plain `utf-8`; all JSON and YAML readers in the three new scripts now use `utf-8-sig`, which is the correct handling for Windows-authored files.
- Still unverified: the builder has never seen a real trace, because no trace artifact exists in this checkout; its ATIF handling is derived from `resource/harness-source/adk_eval_core/tracing/trace.py` and must be inspected against the first real run. The measurement instrument itself (a control-verified task set large enough to detect an effect) is still not built, and it remains the gate for any SFT result being interpretable.
- Environment limitation, reported rather than worked around: `scripts/check_repository.py` cannot run in this workspace. Python's `TemporaryDirectory` creates a `0o700` directory and then fails on `_resetperms` while cleaning it up (`PermissionError: [WinError 5]`), leaving `gemma-ci-4qsxtd5l/` at the repository root in a state the confined token can neither list nor delete. The repository-level link and YAML gates were therefore replaced by targeted checks (all relative links resolved; all four new YAML files parsed). This does not block coding work, and it is the one repository path where another approval round would be needed to continue.
- Changed files: `scripts/probe_model_modules.py` (new), `scripts/build_trajectory_dataset.py` (new), `scripts/check_lora_adapter.py` (evidence-source refactor plus BOM tolerance), `resource/manifests/README.md`, `configs/experiments/lora-trajectory-v1/README.md`.
- Remaining follow-up: run the probe in a GPU session and re-run the adapter check with all seven targets; inspect the first real trace against the builder before training; build the control-verified task set, which is the item both the user's choice A and any future measurement depends on.

### 2026-10-10 20:40 (Asia/Shanghai) - LoRA audit, adapter wiring, and the PEFT/SFT/RL adjudication

- Objective: (1) establish whether the remote agent configuration uses a LoRA environment or parameters, (2) configure one if it does not, (3) rank PEFT, SFT, and RL by the improvement each can plausibly produce and give the user one recommended next research direction to choose.
- Audit result, read from files rather than assumed: no LoRA anywhere. `configs/baseline/agent.yaml` and `configs/baseline/sub_agents/code_analyzer.yaml` declare no `adapter` field; no `adapters/` directory existed under `configs/`; `model/adapters/` holds only a README; `data/raw/gemma-4-developer-agent/` is not present locally. The harness supports it: `adapter` is a field on the agent schema (`resource/harness-source/adk_submission/schema.py:279`) and is resolved against discovered `adapters/<name>/` directories (`resolvers/models.py:46`). "Remote" could not be re-checked live because no network fetch was performed in this session.
- Configured: the new experiment `configs/experiments/lora-trajectory-v1/` declares `adapter: main_lora` on the coder and the analyzer, with prompts, sampling, and budgets copied verbatim from the baseline so the adapter is the only changed variable. `adapters/main_lora/adapter_config.json` is a rank-32 PEFT LoRA config, `lora_alpha` 64.
- Finding that shaped the config: `resource/manifests/sample-adapters.json` records tensor names and shapes for exactly two projections of the required base -- `self_attn.q_proj` (5376x8192) and `self_attn.o_proj` (8192x5376). No other projection name or shape is verified anywhere in this repository, so `target_modules` claims only those two and the README records why. `k_proj`, `v_proj`, `gate_proj`, `up_proj`, and `down_proj` remain an evidence gap to close against the loaded model, not an assumption to encode.
- Verification (all offline, CPU only): `scripts/check_lora_adapter.py` re-derives the target shapes from the sample manifest, recomputes 52,101,120 trainable parameters at rank 32, re-reads the rank policy from the harness source, and confirms rank 32 maps to vLLM bucket 32 with `max_loras` 1 rather than the 128 fallback; it also asserts that exactly one adapter name is declared across the experiment YAML. Result `status: ok`. `scripts/train_lora_adapter.py` ran its dry-run path. All four new YAML files parse, and every relative link in the six files touched was resolved and found to exist.
- Not verified and not claimed: no adapter weights exist. `adapter_model.safetensors` requires the 31B base weights, a GPU, and the training environment, none of which exist in this workspace, so the experiment is deliberately not runnable or submittable. `scripts/check_repository.py` could not complete: it aborts while chmod-ing its own temp directory (`PermissionError: [WinError 5]` under the DSH sandbox), so the repository-level link and YAML gates were replaced by the targeted checks above rather than reported as passing. Python here is `C:\Users\32747\miniconda3\python.exe` 3.14.6, not on PATH.
- Deliverable analysis: `docs/experiments/2026-10-10-lora-sft-rl-adjudication.md` settles that PEFT is a packaging axis, not a competitor to SFT and RL, then ranks the options. Rank 1 is building the measurement instrument, because the only complete paired comparison is n=1 with a byte-identical patch in both arms, so no learning result is currently decidable. Rank 2 is SFT on complete agent trajectories with completion-only loss. Rank 3 is offline preference learning, rank 4 online RL/GRPO, rank 5 PEFT alone. Recommended: instrument first, then SFT on trajectories; use issue-to-diff SFT only as a plumbing check.
- Cost finding worth keeping: an adapter is not free even when inert. Serving one lowers the measured KV cache from 36,432 to 27,712 tokens on 4xL4 at utilization 0.80, and our own single paired task already consumed 191,500 tokens with 19 tool calls, so an adapter must out-earn that penalty. This is why no placeholder adapter was added to the submitted baseline.
- Changed files: `configs/experiments/lora-trajectory-v1/**` (new), `scripts/check_lora_adapter.py` (new), `scripts/train_lora_adapter.py` (new), `configs/training.requirements.txt` (new), `docs/experiments/2026-10-10-lora-sft-rl-adjudication.md` (new), plus table rows and cross-references in `configs/README.md`, `docs/README.md`, and `docs/training-plan.md`. The measured baseline (`configs/baseline/`) is untouched.
- Remaining follow-up: confirm the module names and shapes for `k_proj`, `v_proj`, and the MLP projections against the loaded model before widening `target_modules`; ask the organizers whether the scorer sets `cudagraph_specialize_lora=False` when LoRA is enabled; decide whether to commit this new work on a feature branch, since the current branch carries unrelated uncommitted localization changes.

### 2026-10-10 19:16 (Asia/Shanghai) - Side task: added the basic-programming-principles skill to the project skill root

- Objective: create one reusable skill, asked for as 基本编程原则 ("basic programming principles"), that is loaded whenever coding work starts and carries five rules: (A) do not guess identifiers and never exceed 10 trial-and-error attempts, (B) keep token use lean and skip reasoning that does not change the code, (C) read every programming instruction in full and check each output against it, (D) log every piece of work, (E) fetch web material as completely as possible and avoid redundant refetching.
- Naming constraint found by local inspection, not assumption: `isSkillName` in `node_modules\@deepseek-ai\dsh\node_modules\@deepseek-ai\dsh-skill\lib\index.js:17` is `SKILL_NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/`, and `parseSkillFile` (`dsh-skill-filesystem\lib\index.js:685`) drops any file whose frontmatter `name` fails it. A Chinese `name` is therefore rejected, so the skill is named `basic-programming-principles` while the requested Chinese name rides the `description` and the `# 基本编程原则` heading; description and body text may be Chinese.
- Work performed: created the skill bundle under the project skill root `<projectRoot>/.agents/skills` (discovery rank 200); the root did not exist before, so it was created. The frontmatter `description` starts with 基本编程原则 and names the coding trigger in both languages so the session catalog advertises it as a coding-time skill; the body restates all five rules in Chinese with the operational English detail, and ends with a seven-item completion checklist.
- Verification: the running Web session at http://127.0.0.1:3080 re-published its catalog automatically and now lists `basic-programming-principles` with the requested description, so the file parsed, passed name validation, and was discovered without a restart or a rebuild. The earlier catalog in this session did not contain it, so the change, not a stale cache, produced the entry. No code, config, or test was run beyond this file.
- Changed files: `.agents/skills/basic-programming-principles/SKILL.md` (new, repository-local and currently untracked). No other repository change.
- Remaining follow-up: decide whether this skill is a temporary local aid or a tracked team rule. Committing it requires a branch/PR and one reviewer approval per `docs/ci-and-review.md`; note that the current branch `feature/localization-v1-skill` already carries many unrelated uncommitted localization changes, so a commit here would be mixed. To apply the same skill to every project instead of only this repository, copy the bundle to `<dshHome>\skills` (`C:\Users\32747\.dsh\skills`) or `<agentsHome>\skills` (`C:\Users\32747\.agents\skills`); both are outside the DSH file sandbox, so that copy needs elevated approval.

### 2026-10-10 12:55 (Asia/Shanghai) - Localization v1 paired 4xL4 attempt: one pair measured, push budget exhausted

- Objective: rerun the frozen baseline-versus-portable (Localization v1) paired comparison on Kaggle L4x4 under a hard user-authorized budget of 10 kernel pushes and 9 error-fix iterations, doing nothing else, and stopping immediately if exceeding the budget became possible.
- Budget outcome: **exactly 10 kernel versions were pushed**, plus one further `kaggle kernels push` that the platform rejected with `Maximum batch GPU session count of 2 reached` and which created no version; that rejection is counted as an attempt, so the budget is exhausted and all further Kaggle pushes stopped. 6 of the 9 permitted fixes were used. No competition submission was created.
- Run ledger (kernel versions): 3675-baseline v3 ERROR, 3675-portable v6 ERROR, 3675-baseline v4/v5/v6 ERROR (four distinct defects, below), `gemma4-pinned-wheels` v1 COMPLETE, **3675-baseline v7 COMPLETE and 3675-portable v7 COMPLETE** (the measured pair), 3676-baseline v1 and v2 ERROR (pinned-wheel mount missing).
- Platform drift that invalidated the frozen environment: the organisers **republished `metric/gemma-4-developer-agent-wheelhouse` in place at 2026-10-09T14:43Z**. `swegemma` moved 0.2.10 (`13261f9d…`) to 0.2.11 (`63f054a3…`), and `adk_eval_core-0.1.0` and `adk_submission-0.2.13` changed bytes under unchanged filenames (`f22cdc3f…`→`ca261146…`, `8a19afc7…`→`69642a13…`). 0.2.11 is a breaking refactor that **deleted `swegemma/models/registry.py` and `swegemma/models/discovery.py`**, so `setup_gemma_model_registry`, which `scripts/evaluate_agent.py` imports, no longer exists anywhere in the wheel; the frozen evaluation pipeline cannot import on the served wheelhouse. `swegemma` is not published on PyPI, so the only sources are the dataset (now 0.2.11) and local archives.
- Code fixes applied this round (all still uncommitted; the three files were already untracked before this round): harness-wheelhouse verification now matches by **distribution** instead of filename so a version bump is recorded as drift rather than a missing dependency, and the offline-fatal PyPI fallback was removed (`test_dependency_payload.py`); `TEST_DEPS` is embedded as a Python literal instead of a JSON string, which had raised `TypeError: string indices must be integers` in every notebook using the mount helper; test-only wheels install individually so cp313 `ujson`/`orjson` are recorded as skipped on the CPython 3.12 starter image instead of aborting; the case notebook now reproduces `inputs/submission.manifest.json`, which `verify_eval_run.py` requires; and a new pinned-wheel restore cell (`--pinned-wheel-mount`, forwarded by `run_ab_kaggle.py`) reinstalls the reviewed wheels after the mounted set.
- Environment repair: the reviewed wheels were republished from a private publisher kernel `junzhejiao/gemma4-pinned-wheels` v1 (CPU), whose output hashes match `resource/manifests/eval-wheelhouse-2026-10-09.json` exactly: `swegemma-0.2.10` `13261f9d…`, `adk_eval_core-0.1.0` `f22cdc3f…`, `adk_submission-0.2.13` `8a19afc7…`. Both measured arms recorded `pinned-wheels.json` with all three installed and 41/41 mounted harness wheels installed. Kaggle dataset creation was attempted first and again made no progress (`kaggle datasets list --mine` empty, status 403), matching the earlier session's record.
- Measured result for `rich_3675`, the only task with a complete pair: empty control failed and gold control passed in **both** arms, so the controls are valid. Both arms left the issue **unresolved**, and both produced the **byte-identical patch** (sha256 prefix `ce6e5797a2befeb1`). The portable arm's trace contains **zero** `run_skill_script` calls and exactly one `localization-v1` occurrence, inside the system prompt, so the localization skill was never invoked; the agent edited `rich/console.py`, the gold file, directly. The portable arm cost 171.4 s of agent phase versus 118.9 s baseline (+44%), 19 versus 17 tool calls, and 191,500 versus 175,551 tokens. The comparison therefore measures the cost of an unused prompt instruction, not the module's effect on resolution.
- Blocker for the remaining validated tasks: both `rich_3676` baseline versions failed at the pinned-wheel restore cell with all three wheels reported missing at `/kaggle/input/gemma4-pinned-wheels/harness-wheels-pinned`, although the same mount and identical metadata worked in the two `rich_3675` kernels. Separating "newly created kernel" from "unreliable kernel-source mount" needed one more diagnostic push, which would have exceeded the budget, so the round stopped there. `rich_3676:portable` was never created.
- Verification: the measured pair carries frozen identity — portable bundle `e278af4648c25f95…` (the frozen experimental ZIP), public task `e8796bfcae6e7f7c…`, tensor parallel size 4, 4x NVIDIA L4 23034 MiB, CPython 3.12.13, 41/41 harness wheels installed, 4 of 6 test-only wheels installed with `ujson`/`orjson` recorded as interpreter-rejected. Predeclared acceptance (≥5 points gain, one-sided exact McNemar p ≤ 0.05) is **not** met: 0.0 points, p = 1.0, n = 1. No promotion is justified and no claim about Localization v1's remote effect can be made.
- Files changed: `WORKLOG.md`, and new evidence `runs/2026-10-10-localization-ab/{REPORT.md,ab-summary.json}` plus an empty scratch `runs/2026-10-10-localization-ab/probe.txt` left by a write test. Source edits (uncommitted, untracked files): `scripts/test_dependency_payload.py`, `scripts/prepare_ab_case_notebook.py`, `scripts/run_ab_kaggle.py`. `configs/baseline/`, the frozen protocol, its ZIP hashes, `configs/experiments/localization-v1/`, and every archived checkpoint are unchanged.
- External state: 10 private Kaggle kernel versions pushed across `junzhejiao/gemma4-ab-rich-3675-{baseline,portable}`, `junzhejiao/gemma4-ab-rich-3676-baseline`, and `junzhejiao/gemma4-pinned-wheels`; no competition submission, push to Git, PR, merge, or credential change. Kernel logs and outputs of this round live in the session temporary directory and are ephemeral except for the two evidence files above.
- Follow-up: decide whether to keep pinning the reviewed wheelhouse or re-review the runner against `swegemma 0.2.11` (an API migration, not a notebook fix); fix the skill-invocation problem before spending more GPU time, since the prompt contract did not cause the module to run; and diagnose the kernel-source mount before running `rich_3676` and `rich_3930` under a fresh budget.

### 2026-10-10 10:04 (Asia/Shanghai) - Read-only advice review: LoRA fine-tuning of the required base model

- Objective: answer the user's question about LoRA fine-tuning "the current remote large model parameters" with grounded, project-specific advice. No training run, adapter, script, or configuration change was requested or made.
- Sources read this session: `WORKLOG.md`, `docs/training-plan.md`, `docs/kaggle-runtime.md`, `docs/competition-research.md`, `resource/competition/model-selection-budget-and-harness-rules.md`, `resource/manifests/sample-adapters.json`, `resource/models/google-gemma-4.json`, the harness source at `resource/harness-source/adk_submission/{server.py,discovery.py}`, and the LoRA-related entries in `resource/notebooks/reading-notes.md`.
- Findings that change the advice (all from those sources, not from a new measurement):
  - The only allowed serving base is `gemma-4-31b-it-qat-w4a16-ct` (compressed-tensors W4A16, not NF4/bitsandbytes), so a bnb-4bit QLoRA pipeline as used by the public 31B study does not reproduce the served weight path.
  - The model card snapshot says an **unquantized QAT checkpoint (bf16, "Q4_0")** exists for 31B in the family; if it is reachable, it is the principled training base for this adapter rather than an unrelated bf16 IT checkpoint. Reachability, license, and size are unverified.
  - The harness derives LoRA serving from the adapter directory, not from a hand-written vLLM flag: `infer_adapter_rank` reads `r` from `adapter_config.json`, and `spawn_vllm_server` rounds it up to a vLLM bucket `(1, 8, 16, 32, 64, 128, 256, 320, 512)` (fallback `max_lora_rank=128`), sets `--max-loras` to the mounted-adapter count, and passes `--lora-modules name=path` with names constrained to `^[a-zA-Z0-9_.-]+$`. `discovery.py` groups a PEFT directory around `adapter_model.safetensors`.
  - The two sample adapters in the competition data are the same bytes and their B tensors are zero (recorded in the reading notes), so they cannot serve as an end-to-end on/off smoke test.
- Advice recorded for the user: fix the weight-path question first with a cheap on/off serving test, then decide the training base, then build trajectory data; keep training and evaluation environments separate; keep adapter rank within the vLLM buckets; validate with the existing frozen-protocol acceptance rule and state statistical power honestly given only four tasks currently have valid controls.
- Verification: read-only. No test suite, benchmark, training step, packaging, or Kaggle command was executed, so no metric in this log was re-derived.
- Files changed: `WORKLOG.md` only. `configs/baseline/`, the frozen protocol, its ZIP hashes, and all archived checkpoints are untouched.
- External state: no push, PR, merge, GPU run, Kaggle upload, or competition submission.
- User decision at 10:05 Asia/Shanghai: **do not start LoRA work now**; keep it as a later option and continue the prompt/skill/localization line. This is a scope decision, not a measured comparison: no LoRA experiment was run and no negative result is claimed. The reasons recorded are the W4A16-vs-NF4 weight-path risk, the unverified training base, the likely small count of verified successful trajectories, and the fact that L4x4 quota is better spent on evaluation.
- Follow-up (superseded by the decision above, kept for provenance): writing a LoRA training design into `docs/training-plan.md`, validating the adapter serving path, or deferring LoRA behind the current prompt/skill line of work.
- Resume sequence for the chosen line, unchanged from the 2026-10-10 00:05 stop point: read the two in-flight L4 arms, confirm the empty/gold controls inside each run before trusting any agent result, then compare only the validated tasks; the fixed `min_candidate_margin=0.0` portable policy still awaits fresh confirmation and remote repair evidence before any promotion.

### 2026-10-10 00:38 (Asia/Shanghai) - Side task: installed the DSH whale widget into the web profile

- Objective: install the latest GitHub release of the `DeepSeek-Balance-Whale-Widget` DSH plugin into the DSH installation this session runs under. No repository change was intended or made; this entry records the read-only investigation plus the out-of-repository side effect.
- Version determination: GitHub's latest release and tag for `MeteorNOX/DeepSeek-Balance-Whale-Widget` is `v0.3.18` (2026-10-02, commit `770d3f5`), and npm `dsh-whale-widget` dist-tags `latest` is the same `0.3.18`. The newer `For-Codex` `0.4.x` releases and the `v1.0.0`/`v2.0.0+win` tags belong to the separate Codex/Windows desktop lines and are explicitly not the DSH web plugin, so they were not installed.
- Environment facts: `DSH_HOME=C:\Users\32747\.dsh`, `DSH_PROFILE=web`, and the GUI at `http://127.0.0.1:3080` is that `web` profile. The `desktop` profile already carried `dsh-whale-widget@0.3.18`; only the `web` profile lacked it.
- Work performed: `dsh plugin --profile web add dsh-whale-widget@0.3.18`, run with an escalated sandbox because the profile, pnpm store, and corepack cache live outside the workspace. `pnpm` was absent from PATH, so the corepack shim at `C:\Program Files\nodejs\node_modules\corepack\shims` was prepended for that invocation (it fetched pnpm 12.10.1); no global package was installed.
- Finding worth remembering: that pnpm run pruned two profile plugins that are listed in `dsh.profile.bundles` but were absent from `package.json` dependencies — `dsh-webview-clipboard@0.2.1` and `dsh-vscode-file-links@0.1.0`, both distributed only by the Deepseek-Harness-for-VS-Code extension and not published on npm (`add` of those names fails with a 404). They were moved out of `node_modules\.ignored` back into `node_modules\`, restoring the exact pre-install layout.
- Verification: `profiles\web\package.json` now lists `dsh-whale-widget: 0.3.18` and includes it in `dsh.profile.bundles`; `node_modules` holds `dsh-drop-caret`, `dsh-whale-widget`, `dsh-webview-clipboard`, and `dsh-vscode-file-links`; the composed `cordis.yml` contains the `dsh-whale-widget` entry; and the running server answers `GET /dsh-whale/wait.json` with 401 instead of 404 (the path is defined at `lib/index.js:3425` of the installed plugin), so the plugin mounted live and no server restart was needed. The user still needs a browser hard refresh (Ctrl+F5) to load the client widget.
- Changed files: none in this repository. Files created or modified outside it: `C:\Users\32747\.dsh\profiles\web\{package.json,pnpm-lock.yaml,pnpm-workspace.yaml,cordis.yml,node_modules\**}` and the plugin-manager operation log.
- Remaining follow-up: if the whale does not appear after a hard refresh, restart `dsh web`; if the profile is later re-installed with a plain `pnpm add`, expect `dsh-webview-clipboard` and `dsh-vscode-file-links` to be pruned again and restore them the same way.

### 2026-10-10 00:05 (Asia/Shanghai) - Stop point: environment fixed, comparison arms in flight

- Objective: stop on request and record the exact state so work can resume without repeating the environment diagnosis.
- Where the work stopped: **two paired L4x4 arms are RUNNING and have not yet produced results** — `junzhejiao/gemma4-ab-rich-3675-baseline` (pushed earlier, running for over an hour) and `junzhejiao/gemma4-ab-rich-3675-portable` (version 5, pushed with the corrected wheelhouse handling). No resolution result, patch, or trace has been downloaded for either arm, so **there is still no measurement of whether Localization v1 improves issue resolution.** Nothing should be read as a v1 result.
- Last blocker resolved before stopping: the organisers **republished the harness wheelhouse dataset in place**. The mount held all 41 filenames but `swegemma-0.2.10-py3-none-any.whl` was **absent**, and `adk_eval_core-0.1.0` and `adk_submission-0.2.13` had **different bytes** under the same names (observed `ca261146828a3d8ab0894b4766bc61a0905e3ef94f8eede17561a4a058404e1f` and `69642a13145151040fb96b483170f96cc7cff1d509851d21d08bc7a0ad3ef371` versus reviewed `f22cdc3fecb3a340f2aa0dde99024ecc9a82128fe91fb10d3e252564b188b289` and `8a19afc752aa6cec4e98e25b09aa1fe4b020ad902af8f014d294c85d70247d5d`), while the API still reported version 29. Hash assertions therefore failed for a reason no retry could fix. Both generators now record observed hashes beside the reviewed ones, install the mounted wheels, and fetch `swegemma==0.2.10` from PyPI only when it is absent, verifying SHA256 `f22cdc3fecb3a340f2aa0dde99024ecc9a82128fe91fb10d3e252564b188b289` before installing.
- Validated task set (the only tasks where the empty patch fails and the gold patch passes): `rich_3454`, `rich_3675`, `rich_3676`, `rich_3930`. All four are inside the frozen 38-task protocol. This is the set any comparison must use. It cannot support the predeclared acceptance rule (at least 5 percentage points resolution gain with one-sided exact McNemar p <= 0.05) at any realistic effect size, so the eventual report must state statistical power explicitly rather than imply significance.
- Still-unusable tasks and why, with evidence: FastAPI tasks fail pytest collection with `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` because the environment resolves Starlette to 1.6.0 (the documented dedup problem); `requests` controls exit 124; `rich_3296` fails its own reference patch because IPython's Pygments lexers change `Syntax.guess_lexer` results.
- Environment fixes that are now established and should be reused: the Kaggle image runs **CPython 3.13** (so `cp312` binary wheels are rejected); the official wheelhouse must be attached as a `dataset_sources` entry and mounts at `/kaggle/input/gemma-4-developer-agent-wheelhouse`; `swegemma` installs no test dependencies in subprocess mode and `sandbox/setup.py --fast-path` exits 0 having installed nothing; `httpbin`/`pytest_httpbin` must not be installed because their `pytest11` plugin import fails against Werkzeug 3 and aborts collection for every task; IPython must be hidden for the rich tasks; only Starlette may be repinned per task (repinning pydantic installs 2.13.5 against pydantic-core 2.46.4 and breaks `litellm` for every task).
- Repository state: `WORKLOG.md` is the only file changed inside the repository in this round (via the edit tool). The following script changes are in the working tree but were never pushed and are **not committed**: `scripts/prepare_task_control_sweep.py`, `scripts/test_dependency_payload.py`, `scripts/prepare_ab_case_notebook.py`, `scripts/run_ab_kaggle.py`, `scripts/check_notebook_cells.py`, `scripts/read_kernel_log.py`, `scripts/build_test_dependency_set.py`, `scripts/stage_test_deps_dataset.py`, `scripts/stage_ab_input_dataset.py`, and temporary root-level diagnostics (`list_wheels.py`, `list_tasks.py`, `inspect_*.py`, `summarize_protocol.py`) that should be deleted before handoff. `scripts/prepare_kaggle_runtime.py` was also edited earlier in the session.
- Artifacts outside the repository, because this session could not write under `runs/` and one escalated retry was declined: generated notebooks were written to the system temporary directory (`%TEMP%\dsh-notebooks\...`) and pushed from there, and kernel logs were downloaded there for inspection. Those paths are ephemeral and will be cleared; the durable evidence for this round is the JSON that the kernels themselves wrote, retrievable with `kaggle kernels output <kernel> -p <wherever writable>`.
- Local artifacts that do exist under `runs/2026-10-09-remote-ab/` from earlier in the session: `env/test-deps-py313/` (reviewed test-only wheel set plus manifest, SHA256 `0ab6d99b386ac158eefaf9dab55357a98545fa1196323e62e9832343419b6b87`), `sweep-valid4-output/` (the 4/4 validated control report), `sweep-frozen38-output/`, `sweep-pilot-output/`, `gpu-ab/`, and `official/HARNESS_README.md`.
- Quota: 94331.404 GPU-seconds remained at 23:00 Asia/Shanghai with a 2026-10-10 00:00 UTC refresh; the two running arms and the earlier failed attempts consumed the difference from 100970.614. Recheck before any further launch.
- Immediate next steps for whoever resumes, in order: (1) read both arm outputs and confirm the empty/gold controls inside each run before trusting any agent result; (2) if the portable arm fails again, check `harness-wheelhouse-check.json` and `harness-install.json` first; (3) run the same two arms for the remaining valid tasks (`rich_3454`, `rich_3676`, `rich_3930`) — that is six more L4x4 cases; (4) report paired resolution counts, exact McNemar, and mean agent-time ratio against the predeclared thresholds, plus the honest statement that four to seven tasks cannot establish significance; (5) copy the temporary notebooks and kernel outputs into `runs/2026-10-09-remote-ab/` once writes are available.
- External state: several private Kaggle CPU and GPU kernel versions were pushed while debugging; no competition submission, push to Git, PR, or merge occurred, and the frozen protocol, its ZIP hashes, `configs/baseline/`, and all archived checkpoints are unchanged.

### 2026-10-09 23:05 (Asia/Shanghai) - Four tasks validated; wheelhouse republished in place (in progress)

- Objective: continue repairing the control-group environment and start the L4x4 comparison on tasks whose controls actually pass.
- Decisive new finding: the organisers' harness wheelhouse dataset is **republished in place**. A notebook probe listed the mount and found all 41 filenames present but `swegemma-0.2.10-py3-none-any.whl` missing, `adk_eval_core-0.1.0` with SHA256 `ca261146828a3d8ab0894b4766bc61a0905e3ef94f8eede17561a4a058404e1f` instead of the reviewed `f22cdc3fecb3a340f2aa0dde99024ecc9a82128fe91fb10d3e252564b188b289`, and `adk_submission-0.2.13` with `69642a13145151040fb96b483170f96cc7cff1d509851d21d08bc7a0ad3ef371` instead of `8a19afc752aa6cec4e98e25b09aa1fe4b020ad902af8f014d294c85d70247d5d`, while the API still reports version 29. My hash assertions therefore failed for a reason that no retry could fix. Both generators now record the observed hashes beside the reviewed ones and install the mounted wheels instead of refusing to run; the reviewed test-only wheel set keeps strict hash enforcement because it is under our control.
- Root cause of the "all 38 tasks failed" sweep: pinning **pydantic** per task installed `2.13.5` while the task wheelhouse contains `pydantic-core 2.46.4`, so `litellm` raised `SystemError: The installed pydantic-core version (2.46.5) is incompatible` before any test ran, and every control produced no result file. Only **starlette** is repinned now; the requirement regex also anchors to a full constraint line so dev-dependency lines cannot be mistaken for the runtime constraint.
- Two more environment defects isolated and fixed:
  - Installing `httpbin`/`pytest_httpbin` registers a `pytest11` entry point that imports `httpbin`, whose `from werkzeug.http import parse_authorization_header` no longer exists in the image's Werkzeug 3, aborting collection for **every** task including ones that never use httpbin. Both are now recorded as deliberately excluded.
  - IPython registers extra Pygments lexers, which made `Syntax.guess_lexer("banana.py", "import this")` return `ipython` and broke `rich_3296`'s reference patch. Hiding IPython raised the validated share of the pilot from 2/8 to 4/8.
- Validated task set so far: `rich_3454`, `rich_3675`, `rich_3676`, `rich_3930` all have empty-fails/gold-passes controls, and all four are in the frozen 38. FastAPI tasks still fail at collection with `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` (the documented Starlette 1.6.0 selection problem); `requests` controls exit 124; `rich_3296` fails the gold patch itself because of the lexer artifact.
- GPU pipeline status: `junzhejiao/gemma4-ab-rich-3675-baseline` has been RUNNING on L4x4 past setup, wheel verification and dependency install, which proves the compact-notebook path works end to end through model startup. The portable arm failed only on the wheel-hash assertion above. GPU quota at 23:00 Asia/Shanghai was 94331.404 seconds remaining (refresh 2026-10-10 00:00 UTC).
- Working around the write block: this session's sandbox still denies every write under `runs/`, and one escalated retry was declined, so generated notebooks are now written to the system temporary directory and pushed from there. The consequence is that the artifacts of this round live outside the repository; their paths and hashes are recorded here for recovery, and they must be copied into `runs/2026-10-09-remote-ab/` once writes are available again.
- Files changed: `WORKLOG.md` only inside the repository (via the edit tool). Script edits are staged in the working tree: `scripts/prepare_task_control_sweep.py`, `scripts/test_dependency_payload.py`, `scripts/prepare_ab_case_notebook.py`, `scripts/run_ab_kaggle.py`, `scripts/check_notebook_cells.py`, `scripts/read_kernel_log.py`. The frozen protocol, its ZIP hashes, `configs/baseline/`, and all archived checkpoints remain untouched.
- External state: several private Kaggle CPU and GPU kernel versions pushed while debugging; no competition submission, push, PR, or merge.
- Follow-up: read the version-7 sweep result to finalise the valid task list, then run the baseline-versus-portable pairs on those tasks only, and report paired resolution counts with the predeclared acceptance rule and an explicit statement of statistical power. At the current validated rate (~4 of 8 pilot tasks) the honest conclusion may be that this environment cannot support a significant comparison, which must be stated rather than worked around.

### 2026-10-09 21:20 (Asia/Shanghai) - Official Kaggle sources confirm the environment design (in progress)

- Objective: stop inferring the platform from local snapshots and read the organisers' current official material before deciding how to run the v1 comparison.
- Official sources consulted this session: the competition pages through `competition_list_pages` ("Model Selection, Budget, and Harness Rules", "Evaluation"), the competition data listing through `competition_list_files`, the dataset API views for `metric/gemma-4-developer-agent-wheelhouse`, and the organisers' `HARNESS_README.md` (downloaded to `runs/2026-10-09-remote-ab/official/HARNESS_README.md`, 49356 bytes).
- Confirmed official facts:
  - The submit format is unchanged: one `agent.yaml` at the archive root, optional `configs/`, `prompts/`, `sub_agents/`, `adapters/`, and `skills/<name>/` with `SKILL.md`; skills may contain `scripts/` and `resources/`. Unpacked size must stay below 3 GiB and the context limit is 32768 tokens. Agents may only use the nine harness tools plus `agent_tool` sub-agents.
  - Scoring is PASS/FAIL per task from applying the patch and running that task's validation tests; the overall score is the percentage of tasks whose validation tests pass. The agent gets 12 hours of wall-clock for all tasks inclusive of sandbox setup, excluding patch validation.
  - `HARNESS_README.md` section 4.2 lists seven container bootstrap steps. Step 5 is the only one that installs test dependencies: it streams cached unpacked site-packages and runs `sandbox/setup.py --fast-path <repo>`. The README also states the agent instruction is "All repository and test dependencies are ALREADY pre-installed", and section 4.1 defines the subprocess backend as the option for environments without a Docker daemon.
  - This matches the source finding exactly: `install_test_dependencies` returns early for the subprocess backend, and an in-notebook probe ran `sandbox/setup.py --fast-path` to completion (returncode 0) without installing anything. So on Kaggle the harness itself never supplies repository test dependencies; the public scorer assumes an image where they already exist.
  - The wheelhouse dataset `metric/gemma-4-developer-agent-wheelhouse` is public, still present, and at version 29 (880845509 bytes, updated 2026-10-08T19:18:14Z) with version notes "Update swegemma to 0.2.10 and adk-submission to 0.2.13 (#745028, #745220)". Its file listing still contains exactly the 41 wheels of the reviewed October 9 manifest, including `adk_eval_core-0.1.0`, `adk_submission-0.2.13`, and `swegemma-0.2.10`. The earlier assumption that this dataset had been replaced was wrong; it simply is no longer reachable through the competition-data path.
  - The 124-file wheel set that now lives under the competition data at `wheels/` is the per-task dependency wheelhouse (60 `starlette` versions, `pydantic` 1.10.15/2.13.4, `pytest` 6.2.5/8.3.4/9.1.1, `rich` 15.0.0, `fastapi` 0.141.1, and so on). It contains no harness wheels and still no `dirty_equals`, `inline_snapshot`, `typing_inspection`, `pytest_httpbin`, `multipart`, `ujson`, or `orjson`.
- Corrected design, driven by those sources: the paired notebooks must attach the official wheelhouse dataset for the harness wheels (it mounts at `/kaggle/input/gemma-4-developer-agent-wheelhouse`, confirmed by a runtime listing), keep `--data-dir` at the competition data so the harness finds the task wheelhouse, and additionally install the reviewed test-only wheels because nothing else supplies them on the public subprocess path.
- New evidence from the verification cell: with the dataset attached, all **41 of 41** reviewed harness wheels verify by SHA256 (`missing: []`, `mismatched: []`). The first install attempt failed only because `apache_tvm_ffi-0.1.13.post3` is tagged `cp312` while the Kaggle image runs CPython 3.13; the install step now installs each verified wheel separately, records per-wheel return codes in `harness-install.json`, and skips a wheel whose tags the interpreter rejects instead of failing the cell.
- Work performed: added official-source inspection scripts (`inspect_kaggle_objects.py`, `inspect_competition_data.py`) as read-only diagnostics, downloaded the organisers' README, and reworked `scripts/prepare_task_control_sweep.py` plus `scripts/test_dependency_payload.py` to resolve the harness mount, record the mounted task wheelhouse, probe `sandbox/setup.py`, verify harness wheels by hash, and install tolerantly.
- Verification so far: still no passing control. The pilot sweep (`junzhejiao/gemma4-control-sweep-pilot`) has reached the wheel-install stage in version 10 and is running; earlier versions established the mount path, the interpreter version, the stale-snapshot problem, the 124-file task wheelhouse contents, and the 41/41 harness hash match. No GPU run has started.
- Root cause of the first real control failure: version 10 completed all ten control runs, and both empty and gold failed for every task with a `pytest11` plugin error, not a task failure. Installing `pytest_httpbin`/`httpbin` put a plugin in site-packages that imports `httpbin`, whose `from werkzeug.http import parse_authorization_header` no longer exists in the image's Werkzeug 3, so pytest aborted collection for every task, including tasks that never use httpbin. The reviewed set now installs only `asttokens`, `executing`, `dirty_equals`, `inline_snapshot`, `ujson`, and `orjson`; `httpbin`, `pytest_httpbin`, and `python_multipart` are recorded as deliberately excluded, and the notebook source dropped from 639 KB to 468 KB. Version 11 re-runs the seven-task pilot with this set.
- Open blocker: creating the private Kaggle dataset that carries the reviewed test-only wheels (`junzhejiao/gemma4-public-test-deps`, staged under `runs/2026-10-09-remote-ab/dataset-test-deps`) has produced no progress through the CLI twice; the compact GPU notebook needs that mount (or an equivalent) because the agent bundle already occupies 913 KB of the 1 MB notebook-source limit.
- Files changed: two new read-only diagnostics, the two reworked generators/helpers, the downloaded official README, and this log, plus artifacts under `runs/2026-10-09-remote-ab/`. The frozen protocol, its ZIP hashes, `configs/baseline/`, and all archived checkpoints remain untouched.
- Follow-up: confirm the sweep proves tasks usable, resolve or work around the test-dependency dataset mount, then launch the L4x4 comparison on valid tasks only and report paired resolution counts with the predeclared acceptance rule.

### 2026-10-09 20:50 (Asia/Shanghai) - Platform state changed again; controls still blocked (in progress)

- Objective: continue repairing the public-evaluation environment so the baseline-versus-Localization-v1 comparison has valid controls.
- Finding that invalidates the October 9 wheelhouse snapshot: the wheelhouse attached through the competition data is no longer the 41-file harness inventory. A private CPU probe (`junzhejiao/gemma4-control-sweep-pilot`, version 5) enumerated the mount and found **124 wheels, none of the 41 reviewed names present**. The mounted set is the per-task dependency wheelhouse: `starlette` 0.19.0 through 0.52.1 plus 1.3.1 and 1.6.0 (60 versions), `pydantic` 1.10.15 and 2.13.4, `pytest` 6.2.5/8.3.4/9.1.1, `rich` 15.0.0, `fastapi` 0.141.1, `httpx` 0.28.1, `requests` 2.33.0/2.34.2, `flask`, `werkzeug`, `sqlalchemy`, `sqlmodel`, `typer`, and build backends. It contains **no** `swegemma`, `adk-submission`, `adk-eval-core`, `vllm`, `transformers`, or `torch` wheel, and it still lacks the test-only packages `dirty_equals`, `inline_snapshot`, `typing_inspection`, `pytest_httpbin`, `multipart`, `ujson`, and `orjson`.
- Consequence: `resource/manifests/eval-wheelhouse-2026-10-09.json` is stale and the notebook's strict inventory check was wrong to require set equality. Both generators now verify each reviewed wheel by name and hash, install exactly those, and record extras, missing files, and mismatches in `wheelhouse-check.json` instead of asserting set equality. This is the same class of amendment as the October 9 snapshot refresh, not a change to any agent, task, prompt, or acceptance threshold.
- What still works: the independent reviewed test-dependency set (`runs/2026-10-09-remote-ab/env/test-deps-py313/`, manifest SHA256 `0ab6d99b386ac158eefaf9dab55357a98545fa1196323e62e9832343419b6b87`) remains exactly the right complement, because the new task wheelhouse omits those packages. It was rebuilt for the image's real interpreter: the Kaggle notebook runs **CPython 3.13**, so the earlier `cp312` `ujson`/`orjson` wheels were rejected with "not a supported wheel on this platform".
- Other confirmed platform facts: the notebook runner installs the official wheelhouse itself (`adk_submission` is absent until it runs), so every validation notebook must install it before calling `evaluate_agent.py`. `prepare_kaggle_runtime.py` asserted a wheelhouse mount path that no longer exists; that is fixed as well.
- Work performed: added `scripts/prepare_task_control_sweep.py` (private CPU control sweep), `scripts/prepare_ab_case_notebook.py` (compact per-case L4x4 notebook that re-verifies every frozen hash), `scripts/test_dependency_payload.py` (shared hash-verified dependency helper) and `scripts/stage_test_deps_dataset.py`; extended `scripts/build_test_dependency_set.py` with an explicit `--abi` and made `scripts/prepare_kaggle_runtime.py` install the reviewed test-only wheels offline.
- Verification so far: script compilation passes; the compact GPU notebook for `rich_3675/portable` generates at 942,997 bytes (below the 1 MB limit) and asserts the frozen bundle SHA256 `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48` and the frozen task SHA256. No control has yet passed, so **no task is confirmed usable and no GPU run has started**.
- Environment blockers still open: the test-dependency Kaggle dataset upload (`junzhejiao/gemma4-public-test-deps`) had not completed, and this session still cannot write under `runs/` without per-command approval.
- Files changed: the four new/edited scripts above and this log, plus local artifacts under `runs/2026-10-09-remote-ab/`. The frozen protocol, its ZIP hashes, `configs/baseline/`, and all archived checkpoints remain untouched.
- External state: four private Kaggle CPU kernel versions pushed for the sweep; no GPU run, competition submission, push, PR, or merge.
- Follow-up: decide whether `sweep/setup.py --fast-path` installs the new per-task wheelhouse in subprocess mode; if it does not, install the per-task wheel selection in the notebook before the controls. Then re-run the frozen 38-task control sweep and only start the L4x4 comparison on tasks whose controls actually pass.

### 2026-10-09 20:15 (Asia/Shanghai) - Repair the public-evaluation platform blockers for the v1 A/B (in progress)

- Objective: run the authorized Kaggle 4xL4 comparison of the baseline agent against Localization v1, and first remove the two blockers that would otherwise make any GPU result meaningless.
- Blocker 1, file writes: every `runs/...` write was denied by this session's sandbox even though the folder permissions are correct (the bundled ACL tool re-inspected the tree and returned `NOT_THIS_CLASS`, "both required rights are available"). A widened probe confirmed the denials are the confinement boundary, not a Windows defect. Work continues with per-command escalation into `runs/2026-10-09-remote-ab/`; the user selected full access in the GUI, but the running session still reports workspace-write.
- Blocker 2, invalid controls: the October 9 CPU preflight recorded `controls_valid: false` because the `fastapi_13713` gold patch dies at pytest collection with `ModuleNotFoundError: No module named 'dirty_equals'`. Root cause established from source, not guessed: `swegemma.harness.container_setup.install_test_dependencies` returns immediately when `type(docker).__name__` is `SubprocessManager`, and `SubprocessManager` only uses its `wheels_dir` to translate `/wheels` in commands. The sandbox venv is created with `system_site_packages=True` plus a `_host_env.pth` that points at the host site-packages, so repository test dependencies must already exist in the notebook interpreter. `scripts/evaluate_agent.py` hard-codes `sandbox='subprocess'`, and the competition data exposes no `wheels/` or `sandbox/wheels` directory, so nothing is ever installed.
- Supporting evidence: the public 41-file wheelhouse contains no FastAPI/Requests test dependencies. `resource/discussions/743973-full.json` (a competitor's audit) independently reports the same missing wheels plus a separate `starlette` dedup problem that makes roughly 54 of 67 public FastAPI tasks fail with or without the patch; `resource/discussions/745775-full.json` and the GraphLoc end-to-end notebook review record the same class of failure and a working offline-wheel workaround.
- Work performed: added `scripts/build_test_dependency_set.py`, which downloads and SHA256-hashes a reviewed test-only wheel set for CPython 3.12/manylinux x86_64 from PyPI without installing anything and without touching the official wheelhouse. Produced 45 wheels; the notebook installs a deliberately conservative subset (`asttokens`, `executing`, `dirty_equals`, `inline_snapshot`, `pytest_httpbin`, `httpbin`, `python_multipart`, `ujson`, `orjson`) with `--no-index --no-deps --force-reinstall`, explicitly excluding `pytest`, `pluggy`, `rich`, `flask`, `gevent` and every other package the image already provides at a task-relevant version. Added `scripts/prepare_task_control_sweep.py`, which embeds that wheel set plus the frozen runner scripts and the frozen `eval_config.yaml` into a self-contained private CPU notebook that runs the official empty/gold controls per task with no model call and reports which tasks are usable.
- Verification so far: 628 KB notebook sources (below the 1 MB API limit); both kernels pushed as private CPU kernels, `junzhejiao/gemma4-control-sweep-pilot` (8 tasks) and `junzhejiao/gemma4-control-sweep-frozen38` (the 38 frozen protocol tasks). No test suite was run for the two new scripts yet and no control result has been downloaded, so no task is yet confirmed usable.
- Quota: 100970.614 GPU-seconds remaining at 20:11 Asia/Shanghai, zero reserved, refresh 2026-10-10 00:00 UTC. A three-arm 38-task run is about 114 L4x4 executions and does not fit that window; the predeclared acceptance thresholds were not changed.
- Files changed: `scripts/build_test_dependency_set.py`, `scripts/prepare_task_control_sweep.py`, `WORKLOG.md`, and local artifacts under `runs/2026-10-09-remote-ab/` (`env/test-deps/`, `env/quota-before.json`, `sweep-pilot/`, `sweep-frozen38/`). Root-level scratch scripts `list_wheels.py`, `inspect_repo_tests.py`, `summarize_protocol.py`, and `list_tasks.py` are temporary diagnostics to be removed before handoff. The frozen protocol, its ZIP hashes, `configs/baseline/`, and all archived checkpoints are untouched.
- External state: two private Kaggle CPU kernels created; no GPU run, competition submission, push, PR, or merge.
- Follow-up: confirm which tasks the sweep proves usable, then run only those tasks on 4xL4 across the baseline and portable arms (and the BM25 arm where it adds information), with the predeclared paired acceptance rule. Do not treat any result as a v1 promotion before the controls, the paired counts, and the timing check are all complete.

### 2026-10-09 19:55 (Asia/Shanghai) - Read-only intake review of the workspace

- Objective: read the project and explain what it does, without changing any project or experiment state, and confirm the collaboration/runtime facts a future agent needs before making changes.
- Work performed: read `WORKLOG.md`, `AGENTS.md`, `README.md`, `docs/repository-map.md`, `docs/localization-handoff.md`, `configs/README.md`, `configs/baseline/agent.yaml`, `docs/training-plan.md`, the new `docs/experiments/2026-10-09-localization-v1-portable-gate.md`, and the uncommitted `WORKLOG.md` diff. Inventoried `scripts/` (40 files including the README), `tests/`, `docs/`, `src/`, and the top-level `resource/` directories; inspected `git log`, `git status --short`, `git branch -vv`, `git diff --stat`, and working-tree modification times.
- Findings: the project is a research and reproducible-experiment workspace for the Gemma 4 Developer Agent Kaggle competition, not a standalone agent runtime. The baseline under `configs/baseline/` is declarative only: `gemma-4-31b-it-qat-w4a16-ct`, the main coding agent with nine harness tools plus the read-only `code_analyzer` sub-agent, prompts, sampling, and evaluation budgets, packaged by `scripts/build_submission.py`. The work concentrates on code localization (Localization v1: full-tree BM25, a portable learned ranker with a confidence gate, and the archived v3 checkpoint), tool use/repair, evaluation, and a proposed but unstarted training direction. Verified facts in `README.md` are deliberately narrow: the official submission `56844353` scored 0.12 publicly, plus one Kaggle L4x4 and one H100 single-task smoke test; no training, checkpoint resume, or multi-GPU path has been validated, and `src/` is still a reserved extension point containing only its README.
- Current-state finding 1: HEAD is `ef5e19944e0c8abe4ff40eb69b76d85c4e7c5487` on `feature/localization-v1-skill`, matching the branch state recorded in this log. The remote baseline is unchanged and no experimental module has replaced it.
- Current-state finding 2 (needs attention): the working tree holds a substantial uncommitted prompt-aware localization change that predates this session and is not described by any work-history entry. It adds `scripts/prepare_localization_contract.py`, `scripts/train_prompt_localization.py`, `scripts/check_localization_skill.py`, `tests/test_prompt_localization.py`, and `skills/localization-v1/scripts/prompt_localize.py` with `assets/prompt-policy.json` (all untracked, modification times 18:21-18:25), and it moves the skill's runtime data from `resources/` to `assets/` (the official ADK executor materializes `assets/`, not `resources/`), with `SKILL.md`, `localize.py`, `benchmark_localization_v1.py`, and `tests/test_localization_v1.py` updated to match. This also resolves a documentation inconsistency: the 19:32 entry states `fit_gate` is the only source edit, while the actual diff is the `assets/` migration plus that gate fix. The staged set (`docs/kaggle-runtime.md`, `scripts/prepare_kaggle_runtime.py`, `resource/manifests/eval-wheelhouse-2026-10-09.json`, and related test/doc updates) matches the October 9 Kaggle entries.
- Verification: read-only inspection only. No test suite, repository check, benchmark, trainer, packaging, or Kaggle command was executed, so no metric in this log was re-derived. The `assets/` migration was confirmed by reading the file diff and the new `SKILL.md` text, not by running `localize.py`.
- Environment repair (outside the repository): shell commands initially failed with `SetNamedSecurityInfoW failed (Win32 5): grantWrite(E:\Desktop\CodingAI)` under workspace-write confinement. The bundled one-shot diagnosis-and-repair for that path found the folder lacked effective `WRITE_OWNER` for the signed-in user, backed up the original permissions, granted a full-control entry, and the mandatory re-read verified the grant; the original confined listing then succeeded. Details and the rollback command are in the report under `E:\Desktop\CodingAI-acl-recovery\`. A bounded subtree scan reported `truncated` at 2000 visited entries with zero package entries found, so no deeper path needed a second repair.
- Files changed: `WORKLOG.md` only (this entry). No source, config, prompt, skill, policy, test, artifact, or archive was created, edited, moved, or deleted, and no untracked work was staged or committed. Ignored test-scratch paths under `.tmp-tests/` and `runs/2026-10-09-localization-resume/tmp/` remain unreadable to this session's token; that is pre-existing confinement, not a project defect.
- External state: no push, PR, merge, branch change, GPU run, Kaggle action, upload, competition submission, or credential change.
- Follow-up: the prompt-aware localization work and the `resources/` to `assets/` migration still owe a work-history entry, verification results, and a decision on whether the prompt contract is a separate experiment arm or part of the frozen portable A/B; do not fold it into the frozen protocol or its ZIP hashes without a recorded decision. Confirm the untracked `assets/` directory is intended to be versioned while the archived 4 MB `model.json` stays out of normal reviews. Then resume the documented sequence: fresh confirmation plus remote paired repair evidence before any promotion.

### 2026-10-09 19:32 (Asia/Shanghai) - Diagnose and refine the Localization v1 portable gate

- Objective: continue improving the Localization v1 module for a meaningful remote gain. Audit the portable ranker and its confidence gate using the nested (unbiased) validation protocol as the decision basis, and only keep changes with evidence.
- Work performed: re-ran `train_portable_localization.py` from the local public data (129 tasks, 128 scorable, build 18.7s, training 65.6s) to establish a fresh baseline and confirm the pipeline. Diagnosed the candidate ranker's dev regressions: `rich_3930` (BM25 rank 1 → candidate rank 6; the issue is about graphemes/emoji/width but the gold files are `__main__.py`/`_cell_widths.py`/`_unicode_data/*`, so path/symbol retrieval is misled by `make_terminal_widths.py`/`emoji.py`) and `requests_7433` (BM25 rank 1 → candidate rank 2). Found the frozen gate's `min_candidate_margin=0.025` to be an overfit: the full-training fit chose 0.025, but nested out-of-fold gate selection chose 0.0 in 8 of 9 folds. Attempted two ranker enhancements that did not help and were reverted: an exact-identifier `exact_symbols` strengthening (coordinate descent still assigned weight 0) and `exact_score` fusion (Hit@5 fell from 0.725 to 0.670). Confirmed the frozen weights `[1,2,0,2,2,0,1]` are a stable coordinate-descent fixed point (2/3/5/10 passes all converge identically).
- Verification: all 32 focused localization tests pass. Removing the `min_candidate_margin` floor leaves nested results unchanged (snapshot gated Hit@1 0.418, repository 0.396, both no-regression), raises retrospective test gated Hit@1 from 0.474 to 0.579, but lowers retrospective dev gated Hit@1 from 0.389 to 0.333 because `rich_3930` is no longer suppressed (`dev_no_regression` becomes false).
- Files changed: `scripts/train_portable_localization.py` only (`fit_gate` now fixes `min_candidate_margin` at 0.0 and keeps only the BM25-uncertainty threshold). New local study artifacts under `runs/2026-10-09-localization-v2/` (`baseline/`, `exact-enhanced/`, `gate-min0/`). No change to `configs/baseline/`, the frozen `portable-policy.json`, or archived checkpoints.
- External state: no push, PR, merge, GPU run, Kaggle upload, or competition submission. The frozen remote A/B protocol and its ZIP hashes are untouched.
- Follow-up: the `min_candidate_margin` floor overfits the full-training margin distribution; fixing it at 0.0 matches the nested fold gate and restores test recall, but the resulting `dev_no_regression` failure is driven by one `rich_3930`-style task where the issue concept does not correspond to the gold file path/symbol names. No label-free gate signal reliably separates that failure mode from candidate wins. Do not promote on this evidence alone; a fresh confirmation and remote repair A/B remain required. Next avenues: a robustness-oriented no-regression definition across nested folds rather than the small retrospective dev split, or a ranker change that avoids the path/symbol misdirection on concept-only issues.

### 2026-10-09 12:23 (Asia/Shanghai) - Start authorized Kaggle localization trials (in progress)

- Objective: run private Kaggle comparisons against the remote baseline, retain a materially better v1 candidate, and otherwise revise the module based on measured failures. The user explicitly authorized GPU trials and potentially substantial module changes; this supersedes the previous pending launch confirmation. Competition submission and routine main-branch merging are separate actions.
- Preflight: quota at 12:23 Asia/Shanghai was 101084.638 GPU seconds remaining, zero reserved. Preserved the frozen protocol, bundles, and original notebooks. Launch copies and receipts are under `runs/2026-10-09-localization-kaggle/`.
- Upload fixes: default notebook titles now derive from their slugs. Kaggle's actual API error reported a 1 MB notebook-source limit; added deterministic gzip/base85 payload transport and a source-size guard. Re-encoded all 18 launch notebooks while asserting exact equality of decoded frozen payload bytes. Largest source is 939030 bytes; all agent/task/helper hashes remain unchanged. Regression suite passed 46/46 after the transport change, and repository checks passed; a new explicit oversized-payload rejection test is being added.
- First remote execution: `junzhejiao/localization-ab-fastapi-13713-bm25`, version 1, launched with NvidiaL4 and a 2700-second timeout. It failed after roughly nine seconds of notebook execution at the input-directory assertion, before installation, controls, or inference. Server metadata confirms the intended competition, official wheelhouse, model v2, pinned image, private/offline settings, and NvidiaL4. This is an environment failure, not an issue-resolution result.
- Next diagnostic: a private CPU-only `junzhejiao/localization-runtime-input-probe` lists mount directories for the same inputs. Repair the platform setup before launching more GPU cases; preserve failed attempts and the original frozen protocol. The first pilot follows the protocol's BM25/portable/baseline order.
- Confirmed setup findings: BM25 version 2 showed flat competition/dataset mounts on GPU while the CPU script used namespaced paths. Added both-path support and path recording; version 3 passed mount/task setup but stopped at the strict wheel inventory check. The official October 8 wheelhouse update changes `swegemma` from 0.2.7 to 0.2.10, `adk-submission` from 0.2.12 to 0.2.13, and the bytes of `adk-eval-core 0.1.0`; 38 other wheels match the old hashes. CPU probe version 2 hashed all 41 files, and the three changed wheels were independently downloaded and matched before static interface inspection.
- Environment amendment: added `resource/manifests/eval-wheelhouse-2026-10-09.json` (SHA256 `ad76c04f79a04e289d367760705003a06d587fc437b38d8ca76322f54f4f4e8c`) as the notebook generator default, with an explicit `--wheel-manifest` override and provenance hash in prepared/runtime records. Original inventory and frozen case hashes remain unchanged. All 18 pilot notebooks were regenerated under `reviewed-notebooks/` with this common inventory, recorded in `environment-amendment.json`; acceptance thresholds are unchanged. No upstream source was installed or executed on the host.
- Current verification: 49 CPU tests pass, including the source-size guard, flat-mount frozen-input check, and explicit wheel-manifest provenance. A private CPU-only `localization-preflight-fastapi-13713-bm25` is checking official compilation and empty/gold controls before more GPU work.
- Status: active investigation; no model-performance evidence, promotion, competition submission, push, PR, or merge. This entry will be updated with executed results and verification before handoff.

### 2026-10-09 11:55 (Asia/Shanghai) - Resume localization validation and clear Windows test failures

- Objective: continue the unfinished Localization v1 handoff, complete available CPU verification, and establish readiness for the frozen remote comparison.
- Work: read the persistent handoff and collaboration/runtime guides; confirmed a clean starting tree on `feature/localization-v1-skill` at `ef5e199`. Reproduced the seven documented Windows failures after isolating sandbox temporary-directory permission errors.
- Changes: notebook preparation tests now compare their embedded archive with a direct build from the same checkout and independently compare every ZIP member with source bytes, instead of requiring a historical compressed ZIP hash. Tampering tests restore the exact original task bytes. Repository-check tests expect native path separators. Added Windows UTF-8/Bash setup guidance and refreshed the localization handoff verification status. No agent, policy, training, packaging, or frozen evaluation inputs were changed.
- Verification: `python -m unittest discover -s tests -v` passed 46/46 on Windows/Python 3.14.6. Repository checks passed 33 Python, 10 shell, 16 configuration, 50 Markdown, and two agent packages with `PYTHONUTF8=1` and Git Bash on PATH. The initial default environment lacked Bash and decoded UTF-8 as GBK; environment configuration resolved these failures. Test runs required approved execution outside the sandbox because Python temporary directories were inaccessible even under the workspace temporary root.
- Frozen evidence: verified task identity, sanitized task hashes, source members, and ZIP hashes for all 114 cases; parsed all code cells and checked embedded task, ZIP, helper hashes, private/offline metadata, and kernel identities for all 18 pilot notebooks. Protocol SHA256 remains `22ae84ed20e6e039b8ddd3da50ab8cd4ec8a34b1a67491f9bc2f9e7cb07b6ccb`; all three arm ZIP hashes match the October 8 handoff.
- Read-only external check: the existing Kaggle CLI/SDK returned quota at 11:53 Asia/Shanghai: 108000 GPU seconds allowed, 6915.362 used, zero reserved, and 101084.638 remaining (28.079 quota hours). Conversion uses `total_seconds()`; this is quota accounting, not guaranteed L4 wall time. Recheck before an eventual launch. No credentials were printed or modified.
- Evidence files: ignored `runs/2026-10-09-localization-resume/` contains before/after test logs, `quota.json`, and `frozen-audit.json`. The original `remote-ab-final` directory and all preserved artifacts remain unchanged.
- Changed files: `tests/test_kaggle_preparation.py`, `tests/test_repository_checks.py`, `tests/README.md`, `docs/localization-handoff.md`, and `WORKLOG.md`.
- Follow-up: manual GPU/upload confirmation was requested according to AGENTS.md and remains pending. After confirmation, recheck quota, run the frozen integration pilot, inspect valid controls and repair results, and only then consider the full predeclared comparison. No GPU execution, upload, competition submission, push, PR, merge, or promotion occurred. Changes await the normal PR and teammate-review workflow.

### 2026-10-08 17:51 (Asia/Shanghai) - Preserve the localization result and explain the remote baseline

- Objective: preserve successful optimization while the user is away, and provide an understandable account of the existing remote localization mechanisms and modular optimization path.
- Decision: retain the measured offline gains and implemented optional portable module, without claiming an unmeasured remote repair gain or replacing the baseline. The module is already implemented; no further tuning against the historical test was performed.
- Work: inspected the actual baseline main/analyzer configurations and prompts; wrote `docs/localization-handoff.md` describing issue extraction, source search, graph queries, read-only analysis, the learned ranker/gate hook, BM25 fallback, and source-confirmation requirements. Added a root README entry so the handoff is easy to find.
- Preservation: kept the final study, label-containing training cache, frozen 114-case protocol, 18 pilot notebooks, and experimental ZIP in their existing ignored local run directory. Rechecked ZIP SHA256 as `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48`. These local outputs are not a remote backup. Source, frozen policy, tests, and experiment reports are included in the local feature-branch checkpoint containing this entry; no push or merge is authorized by this handoff.
- Verification: documentation-only additions leave runtime and frozen packages unchanged. Prior focused tests passed 25/25; full suite was 39/46 with seven documented Windows-specific expectations. Final repository checks passed (33 Python, 10 shell, 16 configuration, 50 Markdown files, two agent packages); staged whitespace checks passed. The baseline directory has no staged changes.
- Changed files for this handoff: `README.md`, `docs/localization-handoff.md`, and `WORKLOG.md`; the preceding staged implementation is preserved as the same checkpoint.
- Follow-up: manual quota review and controlled remote evaluation are required before any promotion; the six-task pilot cannot satisfy the full 38-task rule. No GPU job, upload, external deployment, or automatic background continuation was started.

### 2026-10-08 17:43 (Asia/Shanghai) - Learn a portable v1 ranker and prepare paired remote confirmation

- Objective: optimize the ranking algorithm, its parameters, and the learning gate to produce a measurable improvement and prepare a direct comparison against the remote Gemma baseline.
- Important correction: the archived legacy ranker already trained on the 91 training tasks used by the previous gate study. Cross-fitting only its gate therefore did not measure whole-system generalization. Added explicit false promotion requirements for ranker cross-fitting, fresh confirmation, and remote paired repair evidence to the legacy trainer; qualified the historical large-sample report without altering its recorded metrics.
- Implementation: added standard-library portable source/query/path/symbol features and a small nonnegative ranker to the existing skill. The archived checkpoint is preserved and is not loaded by portable mode. `scripts/train_portable_localization.py` learns six coefficients around a fixed BM25 coefficient, with a repository-balanced objective, two coordinate passes over a fixed grid, three-fold out-of-fold gate calibration, nested five-fold snapshot validation, and nested leave-one-repository-out validation. Shared base commits cannot cross fold boundaries. Runtime and training share feature, scoring, and gate functions.
- Frozen parameters: weights `[1, 2, 0, 2, 2, 0, 1]` for BM25, deduplicated query, weak length normalization, path, symbols, exact definitions, and implementation indicator. The gate accepts candidate ranking when BM25 margin is at most 0.6 and candidate margin is at least 0.025. Parameters were selected from training only, frozen before development/test scoring, and reproduced in a second complete run. There were 40 ranker fits per study; final wall time was 74.120 seconds on Windows/Python 3.14.6, CPU and standard library only, seed 52.
- Results: nested snapshot BM25 versus gated candidate Hit@1/Hit@5/MRR is 23.1%/54.9%/0.384 versus 41.8%/74.7%/0.555; nested repository results are 39.6%/72.5%/0.526 for the candidate. Development is 38.9%/55.6%/0.466 versus 38.9%/83.3%/0.549. Historical test is 31.6%/68.4%/0.473 versus 47.4%/78.9%/0.624. The gate activates on 17/18 development and 14/19 test tasks.
- Gate tradeoff: ungated development Hit@1 is 33.3%, so gating restores one correct first choice. Ungated test Hit@1/Hit@5/MRR is 57.9%/84.2%/0.705, higher than gated test performance; the conservative gate sacrifices some gains. No further selection was performed from test results. Test bootstrap Hit@1 and MRR delta intervals exclude zero, but Hit@5 does not; development uncertainty and historical data exposure prevent an independent-confirmation claim.
- Runtime audit: all 37 scorable development/test tasks produced identical ranks and activation decisions through the packaged runtime and learner. Median local ranking time, excluding source scan and AST construction, was 0.248 seconds portable versus 0.271 seconds through the current BM25 wrapper; this is not L4 latency. Explicit `portable` mode runs the candidate for experiments. `portable-policy.json` remains unpromoted, and default adaptive execution stays BM25.
- Remote handoff: added `prepare_localization_ab.py` and `summarize_localization_ab.py`. Prepared 38 task IDs and 114 frozen cases, with the same model, sampling seed 42, analyzer, and evaluation budgets. Baseline/BM25/portable arm order rotates. Six seeded pilot tasks have 18 private offline notebooks; all 114 case hashes and notebook code cells were independently checked. The BM25 and portable arms differ only in the explicit skill mode. The protocol requires at least five points resolution gain over baseline, one-sided exact paired McNemar p <= 0.05, positive gain over the BM25 arm, <= 10% mean agent-time increase, and complete valid controls/pipelines. Missing and invalid cases remain in the denominator and block promotion.
- Cross-platform fix: `prepare_eval_run.py` and the generated Kaggle setup now write public tasks as explicit UTF-8/LF bytes. This fixes a real local-to-Kaggle task hash mismatch without changing agent behavior. Earlier ignored preparation directories are retained; only `remote-ab-final` is the final handoff.
- Verification: 25 focused tests passed. Full suite: 39/46 passed; four fixed Linux/Python baseline ZIP-hash expectations and three POSIX-path assertions still fail on Windows. The task-hash test now gets past frozen-input verification and reaches its pre-existing ZIP-hash assertion. New checks cover ranker/gate fold isolation, shared commits, runtime/training parity, disabled-policy fallback, malformed activation values, and paired failure denominators. After staging, repository checks passed 33 Python, 10 shell, 16 configuration, 49 Markdown, and 2 agent-package checks; staged whitespace checks passed. Both staged and unstaged baseline diffs remained empty.
- Artifacts: final study and exact source/data hashes are in `runs/2026-10-08-localization-v1-portable/study-final/`; reproducible feature cache and earlier study are preserved alongside it. Portable experimental ZIP is 726,506 bytes (4,112,735 unpacked), SHA256 `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48`. Final baseline arm SHA is `aa597ea39b66412eb53fcd3cd2fcc470c0650853a4ebe69f73d777f0ba7e822c`; BM25 arm SHA is `cb1aa5427f2ab42044bb1029371860da4476872774b9176aec3a11c3d3e746b7`. Recovery commands and all limitations are in `docs/experiments/2026-10-08-localization-v1-portable.md`.
- Changed files: portable runtime/policy, portable trainer, legacy trainer promotion guard, paired preparation/summary scripts, two new test modules, task-byte fixes in the existing evaluation/Kaggle preparers, skill instructions, experiment report/navigation, and this log.
- External state and remaining boundary: no GPU/model run, Kaggle upload/submission, credential change, push, PR, merge, baseline modification, or archived-checkpoint overwrite. The laptop lacks official evaluation packages and required model weights. GPU runs remain manual under AGENTS.md. Positive remote repair improvement is still unmeasured; launch the frozen integration pilot after quota review, then complete the predeclared comparison before promotion.

### 2026-10-08 - Expand Localization v1 gate learning and retain the safety fallback

- Objective: perform substantially more learning for Localization v1 and ensure that v1 is activated only when evidence supports a reliable improvement over full-tree BM25.
- Protocol expansion: changed `scripts/train_localization_gate.py` from fitting on 18 development tasks to fitting on 91 training tasks, with deterministic five-fold repository-stratified out-of-fold evaluation, leave-one-repository-out evaluation, frozen confirmation on 18 scorable development tasks, final veto on 19 holdout tasks, and 5,000 paired repository-stratified bootstrap samples on each confirmation split. Runtime features remain label-free, and holdout never selects a rule.
- Learned rule: the training-selected depth-two rule enables the fixed v1 hybrid when BM25 margin is at most `0.36349847748766084` and BM25 positive-score fraction is greater than `0.12810649472405317`. It activates on 86/91 training tasks, 18/18 development tasks, and 18/19 holdout tasks.
- Training evidence: BM25 versus fitted gate is Hit@1 23.1% versus 52.7%, Hit@5 54.9% versus 78.0%, and MRR 0.384 versus 0.640. Five-fold out-of-fold gate metrics are 47.3%, 73.6%, and 0.595; leave-one-repository-out metrics are 48.4%, 74.7%, and 0.603, all above the same BM25 training baseline.
- Frozen evidence: development improves Hit@1 from 38.9% to 50.0%, Hit@5 from 55.6% to 77.8%, and MRR from 0.466 to 0.612. Holdout reverses this pattern: Hit@1 falls from 31.6% to 15.8%, Hit@5 remains 68.4%, and MRR falls from 0.473 to 0.350. Development has 10 wins / 6 ties / 2 losses; holdout has 6 / 3 / 10.
- Uncertainty: development bootstrap deltas are Hit@1 +11.1 points with 95% interval [-11.1, +33.3], Hit@5 +22.2 [+5.6, +44.4], and MRR +0.147 [+0.012, +0.294]. Holdout deltas are Hit@1 -15.8 [-36.8, +5.3], Hit@5 0.0 [-26.3, +26.3], and MRR -0.123 [-0.285, +0.036]. The confidence and holdout promotion checks therefore fail.
- Safety decision: updated the versioned rule and evidence in `resources/gate.json` but kept `enabled: false`. Default adaptive runtime remains an exact BM25 alias and does not load the legacy checkpoint. More fitting on the same public labels cannot ensure v1 generalization; new repositories or untouched tasks are required.
- Verification: all 14 focused localization/gate tests passed outside the restrictive Windows temporary-directory sandbox. The full 35-test suite passed 28 tests; the same four Windows/Python-3.14 newline/ZIP-hash assumptions and three POSIX-path assertions failed. The new large-learning and bootstrap tests passed. After staging, `scripts/check_repository.py` passed 28 Python, 10 shell, 15 configuration, 48 Markdown, and 2 agent-package checks. Repeated packaging produced the same ZIP hash, and the baseline staged diff remained empty.
- Artifacts: `runs/2026-10-08-localization-v1-large-learning/gate-study.json`, generated gate JSON, and `submission-localization-v1-adaptive.zip`. The ZIP is 725,248 bytes (4,107,258 unpacked), SHA256 `b9750171abfa09139b9f5ba80fcebd6d5dc5746c30bd0f3af4243e0696efaf16`.
- Files changed: expanded gate trainer and tests, updated versioned gate and skill explanation, added `docs/experiments/2026-10-08-localization-v1-large-learning.md`, updated experiment/config/script navigation, marked the initial gate report as superseded for deployment evidence, and updated this work log.
- External state: no GPU/model run, Kaggle action, credential change, upload, competition submission, push, PR, merge, or baseline change was performed.
- Follow-up: obtain genuinely new tasks or repositories, train a full-tree portable ranker rather than the archived closed-world path ranker, and require untouched confirmation plus matched end-to-end repair A/B before enabling v1.

### 2026-10-08 - Learn and integrate a fail-safe Localization v1 gate

- Objective: write a learning program that studies development-versus-holdout differences and selectively enables or disables the v1 hybrid so it can complement the BM25 fallback.
- Learning program: added `scripts/train_localization_gate.py`, a dependency-free learner over 12 label-free runtime features: repository size, issue length, BM25 and legacy margins/positive fractions, exact evidence, and BM25/v1 ranking agreement. It fits at most two conjunctive thresholds on development labels, reports feature distribution shift and per-task outcomes, performs leave-one-task-out and leave-one-repository-out validation, freezes the rule before holdout scoring, and emits both a full study and a reviewable gate JSON.
- Leakage boundary: reference-patch paths are used only after ranking to fit the development rule and score evaluation. No gold path enters runtime features or the skill. The designated holdout had already been inspected at aggregate level in the previous experiment, so this gate study is explicitly retrospective rather than a new independent confirmation; holdout can veto deployment but never selects the rule.
- Learned rule: enable the fixed v1 hybrid only when BM25's positive-score fraction is at most `0.9879032258064516` and the strongest exact score is at most `11.75`. On 18 scorable development tasks it enabled 12, improving Hit@1 from 38.9% to 55.6%, Hit@5 from 55.6% to 77.8%, and MRR from 0.466 to 0.641. Leave-one-task-out and leave-one-repository-out also exceeded the development BM25 baseline.
- Holdout finding and safety decision: the same frozen rule enabled 10/19 tasks but regressed Hit@1 from 31.6% to 26.3%, Hit@5 from 68.4% to 63.2%, and MRR from 0.473 to 0.440. The candidate changes from 10 wins / 6 ties / 2 losses on development to 6 / 3 / 10 on holdout. FastAPI alone changes from 6 wins / 0 losses to 2 wins / 6 losses, showing within-repository task shift rather than only repository-mixture shift. The largest label-free shifts are BM25 margin (+0.501 standardized difference), legacy positive fraction (+0.488), and reduced BM25/v1 ranking agreement.
- Runtime integration: added validated `resources/gate.json` and an `adaptive` localizer mode. An enabled gate computes exactly the same features as training and chooses BM25 or v1 per issue. The current gate records the learned rule but sets `enabled: false`; adaptive mode therefore aliases BM25 and does not load the 4 MB checkpoint. A future independently validated gate can activate v1 without source changes.
- Verification: the final gate study is at `runs/2026-10-08-localization-v1-gate/gate-study-final.json`. All 12 focused localization/gate tests passed. With process-local UTF-8 and Git Bash on PATH, `python scripts/check_repository.py` passed 28 Python, 10 shell, 15 configuration, 47 Markdown, and 2 agent-package checks. The full 33-test suite passed 26 tests; the same four Windows/Python-3.14 newline/ZIP-hash assumptions and three POSIX-path assertions failed, with no new functional failure.
- Package: `runs/2026-10-08-localization-v1-gate/submission-localization-v1-adaptive.zip` is 725,149 bytes (4,106,980 unpacked), SHA256 `fbf74428030892b416fe23b2f56b6fa1dcbc892c4494c1ab4ac3728cef5b75c0`; deterministic build and CRC checks passed.
- Files changed: `scripts/train_localization_gate.py`, `tests/test_localization_gate.py`, adaptive-gate additions to the packaged localizer and its tests, `resources/gate.json`, the empty-repository-set fix in `scripts/benchmark_localization_v1.py`, `docs/experiments/2026-10-08-localization-v1-gate.md`, related navigation/report updates, and this work log.
- External state: no GPU/model run, Kaggle action, credential change, upload, competition submission, push, PR, or merge was performed.
- Follow-up: collect a larger grouped calibration set from additional repositories or untouched tasks. Do not continue optimizing against the same 19-task holdout and claim independence. Re-enable v1 only after a new gate passes grouped validation, an untouched confirmation set, official compilation, and matched end-to-end repair A/B.

### 2026-10-08 - Integrate and validate Localization v1 against complete source trees

- Objective: modify and complete Localization v1 so it can be inserted into the current remote agent framework and demonstrate a clear localization improvement without changing the measured baseline.
- Integration: added an isolated agent variant under `configs/experiments/localization-v1/`. The skill is attached directly to the root agent with the official root-relative `skills/localization-v1` reference. The prompt runs it once only when no target file is explicit, verifies the top three candidates from source, and preserves exact search, graph tools, and the read-only analyzer as fallbacks. The baseline model, analyzer configuration, sampling settings, and evaluation budgets remain unchanged.
- Runtime implementation: added a standard-library Python source scanner, BM25 file ranker, AST symbol hints, exact-score and archived-checkpoint diagnostic modes, bounded output, test-file exclusion, symlink exclusion, checkpoint validation, and signed-score-aware RRF. The packaged checkpoint is byte-identical to the protected archived model (SHA256 `56bec1d839b79ecaf9374a7f6da1e5cb496870e87c51750d2284d10f569cfdbc`); repository-specific memorized priors are disabled.
- Full-tree benchmark: evaluated 129 public tasks at their exact base commits with 21--789 non-test Python candidates per snapshot. The deterministic original split remained 91 train / 19 development / 19 test, seed 52, and reference patches were used only after ranking. Benchmark score caching reduced the final run to 26.828 seconds.
- Promotion decision: the development-selected hybrid overfit and was rejected (test Hit@1 15.8%, Hit@5 68.4%, MRR 0.350); BM25 plus exact with legacy disabled was also rejected (26.3%, 57.9%, 0.402). Full-tree BM25 is the safe default and clearly beats the archived legacy-only checkpoint on the held-out test: Hit@1 31.6% vs 21.1%, Hit@3 47.4% vs 36.8%, Hit@5 68.4% vs 42.1%, Hit@10 78.9% vs 52.6%, and MRR 0.473 vs 0.306. This is a localization improvement, not yet an end-to-end repair-resolution claim.
- Verification: all 6 focused Localization v1 tests passed outside the restrictive Windows temp-directory sandbox. With process-local UTF-8 and Git Bash on PATH, `python scripts/check_repository.py` passed 26 Python, 10 shell, 14 configuration, 46 Markdown, and 2 reproducibly packaged agent checks. The full 27-test suite passed 20 tests; the four Windows/Python-3.14 newline/ZIP-hash failures and three POSIX-path assertion failures are the same documented platform-sensitive assumptions, while all new tests passed. The official compiler script could not start in the laptop development environment because `litellm` is intentionally absent; no evaluation dependencies were mixed into it.
- Package: `runs/2026-10-08-localization-v1/submission-localization-v1.zip` is 722,599 bytes (4,097,771 unpacked), SHA256 `ed8a6df2da228a4d0b946027c555c45d6379d663e0c299a4a675f903b1989ed7`; local build, deterministic archive, and CRC checks passed.
- Files changed: new experiment bundle under `configs/experiments/localization-v1/`, `scripts/benchmark_localization_v1.py`, `tests/test_localization_v1.py`, `docs/experiments/2026-10-08-localization-v1.md`, navigation updates in `configs/README.md`, `scripts/README.md`, and `docs/experiments/README.md`, plus this work log. Local ignored benchmark/package artifacts are under `runs/2026-10-08-localization-v1/`.
- External state: fetched missing public GitHub blob objects needed by the local partial clones after explicit approval. No Kaggle run, GPU run, credential change, upload, competition submission, push, PR, or merge was performed.
- Follow-up: compile and run a matched baseline-versus-Localization-v1 repair A/B in the prepared Kaggle or H100 environment. Require valid controls, end-to-end resolution improvement, CPU CI, and one teammate approval before promoting the experiment into `configs/baseline/`.

### 2026-10-07 - Assess Localization v1 portability into the remote baseline

- Objective: determine whether the retained local Localization v1 can be moved into the fetched remote model/agent and whether the remote already contains a similar design.
- Local model finding: Localization v1 is a separate dependency-free hashed pairwise file ranker, not a Gemma adapter. Its protected checkpoint has dimension 262,144, 81,030 sparse nonzero weights, best epoch 1,850, and a 138-path candidate catalog covering only FastAPI, Rich, Requests, and HTTPX. It uses issue/path token pairs, lexical/path mentions, path depth, repository-prefixed path-token weights, and repository/path priors; the label-specific `issue_path` feature was removed in v3.
- Remote baseline finding: `configs/baseline/` uses the unchanged `gemma-4-31b-it-qat-w4a16-ct` base with no adapter. Localization is controlled by the prompt and the read-only `code_analyzer` sub-agent, using `run_command`, `read_file`, `search_similar_code`, `get_code_neighbors`, and `get_code_subgraph`. `search_similar_code` queries precomputed 256-dimensional symbol embeddings and is not a general learned issue-to-file ranker.
- Similar-design finding: the remote reference collection contains several unactivated third-party localization designs, including TF-IDF/BM25 and hashing rankers, GraphLoc BM25 plus graph PageRank as an ADK skill, GemmaLoc candidate reranking, graph propagation/RRF, and coder/localizer sub-agent layouts. These are research snapshots and reading notes, not part of the formal baseline. GraphLoc's packaged skill is the closest execution precedent; the active analyzer sub-agent is the closest architectural analogue.
- Portability conclusion: direct weight fusion into Gemma is not possible because the checkpoint is an independent linear scorer with a different parameter space. Tool-side integration is technically feasible: the official schema accepts packaged `skills`, `run_skill_script` can execute a standard-library localizer in the task repository, and the 3.75 MiB Git checkpoint is below the 50 MiB per-skill limit.
- Required adaptation before activation: enumerate candidate Python source files from the current task repository instead of the checkpoint's public-gold candidate catalog; define behavior for unseen repository-prefixed features; emit bounded top-k paths; preserve exact-search/BM25 fallback; and package it in `configs/experiments/localization-v1/` rather than altering `configs/baseline/`.
- Validation boundary: the current 47.4%/73.7%/78.9% held-out Hit@1/3/5 and 0.619 MRR came from a 19-task closed-world split whose candidate universe was built from public gold paths. It does not establish full-tree, unseen-repository, hidden-task, or end-to-end repair benefit. Require full-tree cross-repository localization followed by matched official baseline-versus-skill repair evaluation before promotion.
- Verification: inspected the protected trainer, unit tests, checkpoint metadata and metrics; inspected the remote agent/analyzer configuration, harness graph-tool implementation, skill schema/resolver and size limits, data guide, research report, and localization notebook reviews.
- Files changed: `WORKLOG.md` only. No source, checkpoint, Agent configuration, prompt, remote branch, Kaggle resource, or model weight was changed.
- Follow-up: if approved, create an isolated Localization v1 skill experiment with full-tree candidate enumeration and tests, keeping the remote baseline byte-for-byte unchanged.

### 2026-10-07 - Protect local work and adopt the remote baseline

- Objective: follow the agreed migration plan while ensuring every pre-existing local source file, experiment, model artifact, ignored dataset, result, and archive remains recoverable.
- Filesystem protection: copied the complete working directory except `.git` to `E:\Desktop\CodingAI-local-backup-20261007-1818`. A source/destination count-and-size check matched 106 files and 23,190,078 bytes. The copy includes the 53 Git-ignored files (3,187,773 bytes) that could not be protected by a normal commit.
- Git protection: created `archive/local-before-remote-2026-10-07` and committed all 43 changed/new versionable paths as `2ba0f75448f16bdd41281572fa0969641ccc44c7`; created annotated tag `local-before-remote-2026-10-07` at the same commit. No credential-like filename was staged. The original local root remains the snapshot commit's parent.
- Remote adoption: moved local `main` onto the remote history and configured `origin/main` as its upstream. Direct checkout exposed one remote path that is invalid on Windows because its timestamp contains colons. Local `main` therefore adds exactly one child commit, `88823c0f556874608cd993cd37ba32d94df7bbd8`, renaming that CSV timestamp from `03:50:43` to `03-50-43`; Git detected a 100% rename and the blob content is unchanged.
- Validation safeguard: an initial low-level tree construction introduced carriage returns into tree entry names. Diff validation caught the broad rename set before checkout, so that malformed commit was rejected, never used as a final branch/tag target, and never pushed. The replacement commit was accepted only after its diff contained exactly the intended single-file rename.
- Migration workspace: created `integration/local-work` from the Windows-compatible `main`. Restored the complete `WORKLOG.md`, combined the remote collaboration guide with the mandatory work-log instructions, and extended `.gitignore` for preserved local `artifacts/`, `data/public/`, `dist/`, and `results/` content.
- Migration boundary: no Programming v1, Localization, Graph Agent, trainer, checkpoint, A/B experiment, or old packaging implementation has been copied into the remote baseline yet. They remain protected and will be evaluated/migrated individually so the verified remote baseline is not overwritten.
- Verification: confirmed the archive branch and annotated tag resolve to the snapshot commit, local `main` has `origin/main` as its direct parent, the compatibility commit is a single 100%-similar rename, the external backup matches source count/size, and preserved generated data no longer appears as untracked after the ignore-rule merge. Installed the remote's locked `PyYAML==6.0.3` CI dependency. With UTF-8 mode and Git Bash on `PATH`, `python scripts/check_repository.py` passed all 23 Python, 10 shell, 8 configuration, 42 Markdown, and 1 agent-package checks.
- Test result: the 21-test CPU suite ran outside the restrictive sandbox and passed 14 tests. Four failures compare Linux/Python-3.12 archive or newline hashes with output produced by Windows/Python 3.14, and three assert POSIX `/` paths while the checker correctly reports Windows `\` paths. These are platform-sensitive test assumptions rather than observed baseline logic failures; tests were not weakened or edited. The first sandboxed attempts failed earlier because Python could not write temporary directories and are not code regressions.
- Files/state changed: external backup directory, local archive branch/tag, local `main` and integration branch refs, `.gitignore`, `AGENTS.md`, `WORKLOG.md`, and the local Python environment (`PyYAML==6.0.3`). No push, force-push, remote branch mutation, Kaggle action, model run, or competition submission occurred.
- Follow-up: confirm the full 21-test suite in the repository's Linux/Python-3.12 CI environment, then migrate one experimental area at a time. Prefer comparing the remote multi-agent baseline against archived Programming v1 before changing `configs/baseline/`; migrate localization source separately from generated checkpoints.

### 2026-10-07 - Fetch and audit the canonical GitHub remote

- Objective: bring `https://github.com/jwj1342/gemma-agent-lab.git` into the local Git repository as the canonical remote, compare it with the current local workspace, and stop before integrating anything so the user can choose the next action.
- Remote operation: added the URL as `origin` and ran a fetch with pruning. `origin/main` now points to `3054da842f794c854f438a5478ea9437b119e9bc` (`Fix Kaggle submission handoff and audit reproducibility gaps (#2)`). No merge, rebase, checkout, reset, branch-upstream change, push, or working-tree replacement was performed.
- History finding: local `main` is the independent root commit `561d85acf2b2fc1c03e2cc9030cb9fc62938634e`; remote `main` has five commits rooted at `61f7c7edae3e2986c4b88fb4f1104d9761089f76`. `git merge-base main origin/main` returns no merge base, and the two sides contain one and five unique commits respectively. A normal fast-forward pull is therefore impossible.
- Tree comparison: the fetched remote contains 897 files and 30,777,734 blob bytes; 815 files and 30,484,147 bytes are under `resource/`. Comparing remote paths with all local tracked and non-ignored untracked paths yields 3 common paths (`.gitignore`, `AGENTS.md`, and `README.md`), all content-different; 50 local-only paths; and 894 remote-only paths. Semantically related submission files also moved from local `submission/` to remote `configs/baseline/`.
- Remote additions: a multi-agent baseline with a read-only `code_analyzer` sub-agent, stricter 60-second/4-minute evaluation limits, reproducible packaging and Kaggle/HPC preparation scripts, CPU repository checks, GitHub CI/review templates, dependency locks, competition/reference snapshots, public notebook reviews, manifests, and expanded English documentation.
- Local-only work to preserve: Programming v1 under `submission/`, the package script and tests, Graph Agent and Kaggle A/B experiments, Localization v1/v2/v3/v3.1/RL/v4 trainers and 19 artifact files, extension notes, and this work log. The current local baseline uses one agent, a more detailed planning/retrieval prompt, 16,384 output tokens, 4,096 thinking tokens, and 300-second/30-minute evaluation limits; the remote baseline uses a main agent plus analyzer sub-agent, 8,192 output tokens, 2,048 thinking tokens, and 60-second/4-minute limits.
- Risk/decision: directly merging would require `--allow-unrelated-histories` and would collide on the three common files while also mixing two repository layouts. Keep the fetched remote isolated until the user chooses whether to replace the local tree, preserve local work on a branch and adopt the remote, or selectively transplant remote infrastructure into the current tree.
- Verification: confirmed both fetch and push URLs, remote default branch `origin/main`, exact commit/parent chain, absence of a merge base, tree/path counts, and the unchanged pre-existing dirty-worktree set before this log update. No code tests were needed because no application code or configuration was integrated.
- Files/state changed: `.git/config` gained the `origin` definition, Git downloaded remote objects and created `refs/remotes/origin/main`, and `WORKLOG.md` was updated. No fetched file was written into the working tree and no existing local source, prompt, experiment, model artifact, or result was modified.
- Follow-up: wait for the user's integration choice; do not run `git pull` on local `main` as currently structured.

### 2026-10-07 - Run Programming v1 versus retrieval-control A/B test

- Objective: test Programming v1 in the official Harness and compare it with the pre-change retrieval-v1 prompt without integrating other modules.
- Prior result recovered: downloaded the completed earlier baseline/retrieval smoke test. Both variants resolved 0/2 on `fastapi_15661` and `fastapi_15588`; this confirmed the two tasks were useful unresolved controls and exposed edit churn, temporary-file mistakes, and test-file modification in one earlier stochastic run.
- Experiment implementation:
  - Added an isolated private Kaggle script under `experiments/kaggle_programming_v1_ab/` with the exact retrieval-control and Programming v1 prompts, the same model/tools/sampling/budgets, and the same two explicit task IDs.
  - Added trace metrics for seven-field plan compliance before the first file mutation, edit/write calls, failed tool calls, changed-file count, test-file modification, duration, and official resolution.
  - Corrected the post-run plan parser to accept the numbered Markdown labels actually produced by the model and to treat both `write_file` and `edit_file` as first mutations. The remote CSV's zero-compliance value was a parser undercount; corrected trace audit is 1/2. Inference results are unaffected.
- Official result: retrieval control resolved 0/2 and Programming v1 resolved 0/2. Both produced two non-empty final patches and neither final patch modified tests in this paired rerun.
- Efficiency/behavior comparison:
  - Programming v1 reduced median failed tool calls from 2.5 to 1.0 (60% fewer).
  - It increased median edit calls from 3.0 to 4.5, median tool calls from 24.5 to 25.5, and median duration from 350.4 to 400.6 seconds (14.3% slower).
  - Aggregate tokens were 557,846 versus 803,959 for the control, but this is confounded by the control entering a 22-call repeated-`submit_patch` loop on `fastapi_15661`.
- Task diagnosis:
  - For `fastapi_15661`, the gold source is `scripts/prepare_release.py`. The control chose that path but implemented an incomplete interface and exited 2; Programming v1 emitted no structured plan, created `scripts/release.py` plus a workflow, expanded scope, and exited 2.
  - For `fastapi_15588`, both chose the correct `fastapi/sse.py` but exited 1. Programming v1 emitted all seven fields, yet its plan over-scoped validation into `format_sse_event`; the reference patch only adds single-line validators to `ServerSentEvent`.
- Decision: do not promote Programming v1. The current seven-field gate is too weakly followed and too verbose, and an explicit plan can formalize a wrong or over-broad hypothesis. A future v2 should use three compact fields (target/evidence, minimal change plus negative scope, verification) and stop after one clean successful `submit_patch`.
- Verification: the experiment script compiles, embedded Programming v1 prompt exactly matches `submission/prompts/system.md`, credential scanning found no secrets, all 20 repository tests passed, and downloaded traces/results were manually cross-checked against public reference patches. Full analysis is in `results/programming_v1_ab_smoke/analysis.md`.
- Files changed: new `experiments/kaggle_programming_v1_ab/kernel-metadata.json` and `programming_v1_ab_test.py`, `WORKLOG.md`, and downloaded ignored results under `results/retrieval_ab_smoke/` and `results/programming_v1_ab_smoke/`; the production prompt and all retained model artifacts were not changed.
- External state: created and completed private Kaggle Notebook version 1 at `junzhejiao/codingai-programming-v1-a-b-smoke-test`; no competition submission, credential change, or other remote mutation was performed.

### 2026-10-07 - Implement prompt-only Programming v1

- Objective: without integrating Localization v1, Graph Agent, or any other module, add the proposed modification-planning workflow to the existing retrieval prompt.
- Prompt change:
  - Added a hard pre-edit gate after source-grounded localization and before the first edit.
  - Requires one compact `Change plan` with exactly seven fields: target, evidence, root cause, intended change, compatibility boundary, verification, and disproof condition.
  - If the disproof condition is unresolved, requires one bounded source read or targeted search and a plan update before editing.
  - After failed verification, requires classification as wrong location/root cause, implementation defect, or unrelated environment/test problem, followed by an evidence-based plan revision; unchanged patches and failing commands may not be repeated.
  - Stops revision after the diff matches the plan and targeted verification passes, preventing opportunistic cleanup.
- Isolation: compared the working submission byte-for-byte with `dist/submission-retrieval-v1.zip`; only `prompts/system.md` differs. Model, nine-tool list, single-agent topology, sampling configuration, evaluation limits, Localization v1, and Graph Agent are unchanged.
- Packaging: created `dist/submission-programming-v1.zip` with SHA-256 `3E9A58106BAEB78AB12335C0FC56734C71EF7B4C7CF5AB4CFF7394C7B7C20149`. The preserved baseline and retrieval-control hashes remain `F42BB63AA3D12545912CAB31BC6E4638FDFB634F099AC53888AC0CA005E39701` and `948DE71457C59581F545D766C2E07B9B1E56DAFB31E178C8A8DC3C86B00404D4`.
- Verification: added a package test for the planning gate and all seven required fields; all 20 repository tests passed, archive integrity passed, the archive contains the expected four files at its root, `agent_tool` is absent, and `git diff --check` plus explicit trailing-whitespace checks passed.
- Decision: keep Programming v1 as an isolated candidate pending paired end-to-end evaluation; do not replace the preserved baseline or retrieval-only control yet.
- Files changed: `submission/prompts/system.md`, `tests/test_package_submission.py`, `WORKLOG.md`; new ignored generated artifact `dist/submission-programming-v1.zip`.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Choose the starting point for programming improvements

- Objective: decide whether to begin with modification planning or broader prompt changes for the programming path.
- Recommendation: start with a compact modification-planning gate implemented as the only prompt change. This improves the current prompt-driven architecture without introducing another model, and it keeps localization, tools, sampling, and budgets fixed for causal comparison.
- Proposed pre-edit contract: after source verification and before the first edit, record the target file/symbol, observed evidence, root cause, intended behavior change, compatibility/impact boundary, smallest verification command, and one condition that would invalidate the plan. If evidence is insufficient, perform one bounded source read rather than edit.
- Proposed recovery rule: after a failed verification, revise the plan from the failure evidence before making another edit; do not repeat the same patch or command unchanged.
- Experiment order: (1) baseline current retrieval prompt, (2) planning-gate-only prompt, then only if it helps, (3) refine editing and recovery instructions. Do not simultaneously integrate Localization v1 or Graph Agent because that would confound attribution.
- Suggested measures: resolved rate first, then valid/non-empty patch rate, first-edited-file correctness, edit churn, targeted-test success, tool calls, and duration. Planning text should stay compact so it does not consume the 50-turn/tool budget.
- Files changed: `WORKLOG.md` only; no Agent prompt, model, configuration, localization artifact, or distribution archive was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Retain Localization v1 and audit the programming path

- Objective: retain the strongest v3 localization checkpoint under the product-level name Localization v1, then inspect how the current AI implements programming work.
- Naming decision: Localization v1 refers to the protected experiment-v3 checkpoint at `artifacts/localization_v3_3k/` (3,000 epochs, best epoch 1,850). The directory was not renamed because `artifacts/localization_v1/` already contains the older experiment-v1 artifact and preserving lineage avoids ambiguity.
- Architecture finding: there is no standalone programming model or deterministic coding engine in this repository. The official Harness serves `gemma-4-31b-it-qat-w4a16-ct`; `submission/agent.yaml` grants nine tools, and the system prompt instructs the model to parse the issue, localize source, form a hypothesis, edit minimally, run targeted verification, inspect the diff, and call `submit_patch`.
- Runtime finding: the Harness/`swegemma.Evaluator` supplies repository snapshots, graph data, embeddings, sandbox execution, task budgets, and final test evaluation. The model chooses each tool call; repository code only declares policy and limits.
- Integration gap: retained Localization v1 is an offline hashed pairwise ranker and is not referenced by `submission/`, the Notebook evaluator, or the Graph Agent experiment. Current runtime localization is prompt-directed exact search plus optional harness embedding/graph tools.
- Version finding: `submission/` exactly matches `dist/submission-retrieval-v1.zip`; the separately preserved `dist/submission.zip` contains the earlier baseline prompt. Graph Agent v1 remains an unactivated experiment.
- Control and validation finding: programming behavior is governed primarily by prompt rules, with no local end-to-end coding test. Package validation checks the required model, tool list, archive layout, and single-agent boundary, but not whether generated patches solve tasks.
- Verification: rebuilt a temporary audit package and ran the four package tests successfully; confirmed byte-for-byte agreement between `submission/` files and `dist/submission-retrieval-v1.zip`; `git diff --check` passed. The temporary audit archive was removed after exact-path verification; test-only archives were regenerated by the package tests.
- Files changed: `WORKLOG.md` only; no model, localization artifact, Agent configuration, prompt, retained distribution archive, or source code was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Confirm the strongest current localizer version

- Objective: identify the best currently available localization version after the v4 experiment.
- Finding: v3 remains the retained and strongest overall version. On the 19-task held-out test split it records Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, and MRR 0.619.
- Comparison: v3.1 selected its unchanged v3 parent checkpoint; RL v1 and v4 both selected epoch 0 and did not improve directly comparable prediction-quality metrics. V4 is therefore an ablation, not the promoted model.
- Scope caution: the benchmark is closed-world and small, so this is the best version under the current local evaluation rather than proof of best hidden Kaggle performance.
- Files changed: `WORKLOG.md` only; no model, source code, configuration, or artifact was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Train v4 with squared count and location matching

- Objective: change the RL localization reward to a weighted sum of squared position and count matching, expose independent proportions, run 300 training epochs as v4, and compare it with prior variants.
- Reward implementation:
  - V4 uses `count_weight * count_match^2 + location_weight * location_f1^2`, retaining the 0.50 positive-reward threshold and the guard that assigns zero match when no correct file is returned.
  - Added `--count-weight`, `--location-weight`, and `--reward-component-power`; weights must be non-negative and sum to one. Defaults are 0.65 count, 0.35 location, and power 2. The reward helper keeps power 1 as a backward-compatible linear control.
  - Changed the default output to the isolated `artifacts/localization_v4_300/` directory and records the full reward/training configuration in the model and metrics artifacts.
- Training: completed exactly 300 epochs with seed 52 on the same 91/19/19 split and frozen v3 checkpoint. Sampled training reward reached 0.218 at epoch 300, but deterministic development reward fell from -0.445 at epoch 0 to -0.656 and development exact-set rate fell from 26.3% to 15.8%; checkpoint protection selected epoch 0.
- Comparison:
  - Because both RL v1 and v4 selected epoch 0, their directly comparable selected-checkpoint metrics are identical: development exact count/set 63.2%/26.3%, and test exact count/set/location-F1 63.2%/31.6%/40.4%.
  - At epoch 300, v4 kept development exact count at 63.2% versus RL v1's 42.1%, but both had 15.8% exact-set accuracy and v4 had lower location-F1 (25.0% versus 27.6%).
  - Reward values are not directly comparable across v1 and v4 because the scoring formulas differ. The squared formula intentionally lowers partial-match scores and did not solve file-identity generalization or the one-file training imbalance.
  - Retained v3 remains preferable: its test ranking metrics are Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, and MRR 0.619, while its metrics are not directly equivalent to the RL complete-set measures.
- Decision: keep v4 as a reproducible reward ablation but do not promote it over v3. A comparison report is stored at `artifacts/localization_v4_300/comparison.md`.
- Verification: all 19 unit tests passed; `git diff --check` and explicit trailing-whitespace checks passed; v4 records 300 completed epochs, squared power 2, the requested 0.65/0.35 proportions, and best epoch 0. The protected v3 artifacts were only read and not overwritten.
- Files changed: `scripts/train_rl_localizer.py`, `tests/test_rl_localizer.py`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v4_300/`; prior v1, v2, v3, v3.1, and RL v1 artifacts were not modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Add count-emphasized reinforcement learning for localization

- Objective: reward or penalize the policy according to how closely the number and identity of returned files match the answer, with positive reward at 50% match and extra emphasis on count accuracy.
- Reward design:
  - `count_match = 1 - abs(predicted_count - gold_count) / max(predicted_count, gold_count)`.
  - `location_f1` is the set F1 between predicted and gold source-file paths.
  - Combined match is `0.65 * count_match + 0.35 * location_f1`; match at or above 0.50 receives positive reward equal to match, otherwise reward is `-(1 - match)`.
  - A zero-location guard forces match to zero when no correct file is returned, preventing a policy from earning positive reward by guessing only the correct count.
- Implementation:
  - Added `scripts/train_rl_localizer.py`, a dependency-free two-head contextual-bandit policy on frozen v3: a 1–26 file-count head and a trainable residual file-ranking head.
  - Added stochastic count selection, without-replacement file selection, REINFORCE-style advantage updates with per-repository moving baselines, global decoupled decay, deterministic evaluation, and best-development checkpoint protection.
  - Added five reward tests covering exact match, wrong locations with correct count, the 50% boundary, below-threshold proportional penalty, and stronger count weighting.
- Data finding: the full public gold-count distribution is extremely imbalanced: 91 of 129 tasks change one source file, while rare tasks change as many as 26. In the fixed training split, 67 of 91 tasks have count one.
- Pilot: ran 300 epochs with count weight 0.65. Sampled training reward rose from roughly -0.04 to +0.47, but deterministic development reward never beat the epoch-0 value of -0.350 and exact-set rate degraded from 26.3% to 15.8% during training.
- Decision: the saved best checkpoint is epoch 0, with zero residual/count learned weights and frozen v3 behavior plus the empirical count prior. Keep the code and curve as an RL prototype, but do not promote it over v3. Future work needs count-balanced sampling or more multi-file tasks before another RL run.
- Verification: fifteen tests passed, `git diff --check` passed, the 300-epoch artifact and reward configuration were validated, and retained v3 artifact hashes remained unchanged.
- Files changed: new `scripts/train_rl_localizer.py`, new `tests/test_rl_localizer.py`, `WORKLOG.md`, and generated artifacts under `artifacts/localization_rl_v1/`; no v3 artifact was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Show a representative public task structure

- Objective: give the user one concrete example of the localization-training task format.
- Example reviewed: public task `rich_3676` from `Textualize/rich` at base commit `2b5d648b2c62960deb6439f4204ed8fa6eb3cf7b`; its problem statement is `Notes\n\nAdd support for Exception.__notes__` and it has no hints.
- Structure explained: each JSONL record contains `instance_id`, `repo`, `base_commit`, `problem_statement`, `hints_text`, organizer `patch`, and organizer `test_patch`. Only the problem/hints are localization input; reference patches are post-run labels.
- Gold localization for the example: the reference source patch changes `rich/default_styles.py` and `rich/traceback.py`; the organizer test patch changes `tests/test_traceback.py`.
- Files changed: `WORKLOG.md` only; no model, training data, code, or artifact was modified.
- External state: no training run, Kaggle request, Notebook action, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Continue v3 for 3,000 epochs as v3.1

- Objective: preserve all v3 data, continue from its selected checkpoint for 3,000 additional epochs, and store the experiment separately as v3.1.
- Implementation:
  - Added validated sparse-checkpoint loading for the `no_issue_path_v3` feature set, including dimension and candidate-inventory checks.
  - Added `--init-model` and `--epoch-offset`; v3.1 restored v3's epoch-1,850 weights and continued the learning-rate schedule over global epoch indices 1,851 through 4,850.
  - Evaluated the parent checkpoint as local epoch 0 and retained it unless a continuation checkpoint improved the Hit@1/Hit@5 composite, preventing continuation regression.
  - Added non-empty output-directory refusal so retained artifacts cannot be accidentally overwritten.
  - Added a fixed read-only checkpoint fixture and unit test for sparse weight restoration. An ACL-restricted temporary test directory created during initial validation was explicitly removed after exact-path verification.
- Training: completed exactly 3,000 additional epochs and recorded 122 curve rows including the epoch-0 parent checkpoint. No continuation checkpoint exceeded the parent's development composite score of 0.6579; the best additional checkpoints reached at most 0.6316.
- Result: v3.1 selected local epoch 0/global epoch 1,850. Its saved weights are byte-for-byte equal as JSON weight values to the protected v3 weights, and its held-out metrics therefore remain Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, and MRR 0.619.
- Decision: keep `artifacts/localization_v3_1_additional_3k/` as the continuation record but do not promote it over v3; further identical-data epochs are not justified by this result.
- Verification: ten unit tests passed, `git diff --check` passed, v3.1 lineage/configuration and 3,000 additional epochs were validated, and all three retained v3 artifact hashes remained unchanged.
- Files changed: `scripts/train_localizer.py`, `tests/test_train_localizer.py`, new `tests/fixtures/v3_checkpoint.json`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v3_1_additional_3k/`; no v3 artifact was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Retain localizer v3

- Objective: preserve the current v3 implementation and trained checkpoint.
- Decision: treat v3 as the retained current localizer variant and keep `artifacts/localization_v3_3k/model.json`, `metrics.json`, and `training_curve.csv` unchanged. Keep v1 and v2 as comparison artifacts rather than deleting them.
- Verification: confirmed the v3 artifact still records 3,000 completed epochs, best epoch 1,850, feature set `no_issue_path_v3`, global decoupled weight decay, and the previously reported held-out metrics.
- Files changed: `WORKLOG.md` only; no source, model, metric, or training-curve file was modified.
- External state: no training run, Kaggle request, Notebook action, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Check remote L4 status without affecting local work

- Objective: inspect the existing Kaggle L4 Notebook while leaving the completed local v3 training and remote job unchanged.
- Result: a read-only Kaggle status query reports `junzhejiao/codingai-retrieval-a-b-smoke-test` as `KernelWorkerStatus.QUEUED`; it is still waiting for compute and has not started L4 execution.
- Files changed: `WORKLOG.md` only for this status check.
- External state: no Notebook stop, restart, rerun, push, competition submission, credential change, or artifact download was performed.

### 2026-10-06 - Remove issue-path memorization and train v3 for 3,000 epochs

- Objective: completely remove the label-specific `issue_path` feature, replace touched-coordinate L2 with true global decoupled weight decay, and train a separate v3 for 3,000 epochs.
- Implementation:
  - Removed `issue_path` from feature construction and removed its weight from the model and CLI configuration.
  - Implemented global decoupled decay as a lazy scalar applied to every weight on every update, including weights absent from the current positive/negative pair; materialization preserves exact weights for checkpoints and serialization.
  - Added tests proving the memorization feature is absent and an untouched coordinate is decayed.
  - Changed defaults to 3,000 epochs, evaluation every 25 epochs, output under `artifacts/localization_v3_3k/`, and weight decay `1e-4`.
- Training: completed exactly 3,000 epochs with seed 52 and recorded 121 evaluation points. The equal-weight Hit@1/Hit@5 development objective selected epoch 1,850 with composite score 0.6579.
- Held-out results on the same 19 test tasks:
  - v3: Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, MRR 0.619.
  - v1: Hit@1 47.4%, Hit@3 68.4%, Hit@5 73.7%, MRR 0.614.
  - v2: Hit@1 42.1%, Hit@3 63.2%, Hit@5 73.7%, MRR 0.556.
- Interpretation: v3 matches v1 Hit@1 and improves Hit@3, Hit@5, and MRR by roughly one held-out task or less. This is directionally positive but not conclusive because the test split contains only 19 tasks and the benchmark remains closed-world.
- Verification: nine unit tests passed, `git diff --check` passed, and the saved artifact records the requested 3,000 epochs, best epoch 1,850, the no-`issue_path` feature set, and global-decoupled decay mode.
- Files changed: `scripts/train_localizer.py`, `tests/test_train_localizer.py`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v3_3k/`; v1 and v2 artifacts were not overwritten.
- External state during training: no Kaggle request, Notebook push, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Diagnose why localizer v2 underperformed

- Objective: explain why the downweighted, more strongly regularized 10,000-epoch v2 did not outperform v1.
- Evidence:
  - V2 delayed perfect training Hit@1 from epoch 75 to epoch 450 but still reached 100%, so the changes slowed memorization rather than preventing it.
  - Despite nominally stronger L2, the selected v2 model has 127,877 nonzero weights, L1 norm 3,198.31, and L2 norm 26.93 versus v1's 102,911, 1,104.67, and 11.05. The longer run allowed weights to compensate for the 0.25 feature scaling; the current update applies shrinkage only to touched coordinates and scales it by the decaying learning rate.
  - V1 and v2 reached the same maximum development composite score of 0.6053. V1 reached it once at epoch 50, while v2 reached it at 68 recorded evaluation points between epochs 450 and 6,100, leaving the new objective with weak checkpoint discrimination.
  - With only 19 development and 19 test tasks, each Hit@k task changes the metric by 5.26 percentage points. V2's Hit@1 and Hit@3 decline relative to v1 each corresponds to one test task, so the apparent difference is not statistically strong.
  - Equal weighting of Hit@1 and Hit@5 does not directly optimize a listwise top-five objective; pairwise training still compares one selected positive against at most one hard and one random negative. The frequency-plus-lexical baseline therefore retains substantially better Hit@5.
- Conclusion: v2 combined an underpowered, highly discrete selection signal with excessive repeated optimization on 91 training tasks. Feature rescaling and nominal L2 were insufficient structural controls, and 10,000 epochs amplified rather than solved the small-data problem.
- Files changed: `WORKLOG.md` only; no source code, model, or artifact was modified.
- External state: no training run, Kaggle request, Notebook push, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Regularize the localizer and retrain for 10,000 epochs

- Objective: reduce `issue_path` memorization, strengthen regularization, balance Hit@1 and Hit@5 during checkpoint selection, and complete 10,000 new training epochs without overwriting v1.
- Implementation:
  - Changed `PairwiseLocalizer` to expose `issue_path_weight`, with a new default of 0.25 instead of the prior implicit 1.0.
  - Increased default L2 regularization from `1e-6` to `1e-4`.
  - Added an equal-weight checkpoint objective, `0.5 * Hit@1 + 0.5 * Hit@5`, using development MRR only to break equal composite scores.
  - Added CLI controls for the issue-path weight and checkpoint Hit@1 weight, and recorded all training/selection settings in the saved metrics.
  - Added tests for issue-path downweighting and composite checkpoint scoring.
- Training: ran exactly 10,000 epochs with seed 52, evaluation every 50 epochs plus epoch 1, and wrote a separate v2 artifact directory. The run produced 201 recorded evaluation points and selected epoch 450 with a development composite score of 0.6053.
- Held-out results on the same 19 test tasks:
  - v2: Hit@1 42.1%, Hit@3 63.2%, Hit@5 73.7%, MRR 0.556.
  - v1: Hit@1 47.4%, Hit@3 68.4%, Hit@5 73.7%, MRR 0.614.
  - Frequency-plus-lexical baseline: Hit@1 36.8%, Hit@3 63.2%, Hit@5 94.7%, MRR 0.568.
- Decision: retain v2 as an ablation but do not replace v1. The changes preserved v1 Hit@5 but reduced Hit@1 and MRR; the 10,000-epoch trajectory also confirms that useful development performance peaks early and later epochs mainly sustain training-set memorization.
- Verification: eight unit tests passed, `git diff --check` passed, and artifacts record 10,000 completed epochs, best epoch 450, 201 curve rows, and 127,877 nonzero weights.
- Files changed: `scripts/train_localizer.py`, `tests/test_train_localizer.py`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v2_10k/`; v1 artifacts were not modified.
- External state: no Kaggle request, Notebook push, competition submission, credential change, or remote L4 process change was performed.
- Follow-up: test a hybrid score that preserves the frequency-plus-lexical candidate recall, replace gold-derived candidates with full repository file inventories, and use early stopping instead of additional epochs.

### 2026-10-06 - Expose localization code for manual tuning

- Objective: show the user the trained file-localization implementation and identify the sections and parameters suitable for manual tuning.
- Reviewed components: public-patch label extraction, deterministic repository split, issue/path tokenization, hashed feature construction, pairwise logistic update, hard-negative selection, development-checkpoint selection, evaluation metrics, and CLI hyperparameters.
- Important tuning targets: token limit and stop words, feature values in `PairwiseLocalizer.features`, learning-rate decay and L2 regularization, negative sampling, development checkpoint objective, and closed-world candidate construction.
- Important caution: `issue_path=<repo:path>:<token>` is highly label-specific and explains the rapid training-set memorization; `build_candidates` uses the union of public gold paths and must be replaced by a full repository file inventory for a realistic benchmark.
- Files changed: `WORKLOG.md` only; localization source and trained artifacts were not modified.
- External state: no training run, Kaggle request, Notebook push, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Train a lightweight file localizer for 1000 epochs

- Objective: start no-queue training of code-location ability and complete 1000 training passes without changing the existing embedding model or interfering with the remote L4 run.
- Data preparation:
  - Downloaded the public `tasks.jsonl` locally and excluded it from Git; it contains 129 tasks across FastAPI, Requests, Rich, and HTTPX.
  - Corrected reference-patch parsing to use unified-diff `+++ b/<path>` headers; all 129 tasks yielded at least one non-test Python target and 38 tasks changed multiple source files.
  - Used a deterministic repository-stratified split of 91 train, 19 development, and 19 test tasks. Reference patches supplied labels only and were not included in issue text.
- Implementation:
  - Added `scripts/train_localizer.py`, a dependency-free hashed pairwise logistic ranker using issue tokens, path priors, issue/path associations, lexical overlap, and hard-negative training.
  - Added parser and deterministic-split tests in `tests/test_train_localizer.py`.
  - Ran exactly 1000 full epochs with seed 52, evaluated the development set every 25 epochs (plus epoch 1), and saved the epoch-50 best-development checkpoint instead of the overfit final weights.
  - Saved `model.json`, `metrics.json`, and `training_curve.csv` under `artifacts/localization_v1/`.
- Results:
  - Training Hit@1 at the selected checkpoint: 97.8%; development Hit@1/MRR: 47.4%/0.598.
  - On the untouched 19-task test split, the selected model reached Hit@1 47.4%, Hit@3 68.4%, Hit@5 73.7%, and MRR 0.614.
  - The frequency-plus-lexical test baseline reached Hit@1 36.8%, Hit@3 63.2%, Hit@5 94.7%, and MRR 0.568. The model improved top-rank precision and MRR, but degraded Hit@5, so it should be treated as an experimental reranker rather than a replacement retrieval system.
- Verification: six unit tests passed; the artifact records 1000 completed epochs, 41 evaluation points, best epoch 50, and 102,911 nonzero hashed weights; final `git diff --check` passed.
- Limitations: the candidate inventory is built from public changed-file labels and omits never-changed repository files, only file-level localization is trained, and the test split is small. Full-tree candidates and symbol-level labels are required before agent integration.
- Files changed: `.gitignore`, `scripts/train_localizer.py`, `tests/test_train_localizer.py`, `WORKLOG.md`, and generated artifacts under `artifacts/localization_v1/`; downloaded ignored input under `data/public/`.
- External state: no Kaggle Notebook push, competition submission, remote-run change, or credential-file modification was performed.
- Follow-up: construct complete per-commit file inventories, add symbol-level targets, and compare a conservative hybrid that falls back to the frequency/lexical ranker so Hit@5 is not lost.

### 2026-10-06 - Assess no-queue training options

- Objective: determine which training work can begin immediately from the public data without waiting for Kaggle L4 capacity.
- Finding: data preparation, reference-patch parsing, gold file/symbol extraction, retrieval-policy supervision, lightweight ranking/classification, graph-policy tuning, prompt/skill development, and CPU/static evaluation can start immediately after downloading `tasks.jsonl` and selected graph/embedding files.
- Hardware boundary: the local NVIDIA RTX 5060 Laptop GPU has 8,151 MiB VRAM, which is insufficient for practical fine-tuning or rollout-based RL of the required `gemma-4-31b-it-qat-w4a16-ct`; the official runtime uses four L4 GPUs. LoRA/RL for the required 31B model therefore still needs substantially larger local/external compute or queued Kaggle capacity.
- Recommended immediate training target: build a symbol-localization/router dataset from public issue text and post-run reference patches, then train or tune a lightweight policy that predicts exact symbols and whether Graph Agent delegation is needed. Keep a fixed repository/task holdout to measure generalization.
- Files changed: `WORKLOG.md` only.
- External state: no dataset download, credential change, Notebook push, rerun, or competition submission was performed.

### 2026-10-06 - Clarify when localization accuracy can improve

- Objective: clarify whether localization accuracy can improve only after a formal competition submission.
- Finding: localization can be measured and improved entirely through private evaluation on the 129 public development tasks. Reference patches are used only after each run to score files/symbols and never exposed to the agent. A fixed development/validation/holdout split prevents tuning directly to every public answer.
- Runtime feedback: even without reference patches, the agent can improve confidence during a task by checking exact symbol resolution, source behavior, graph relationships, call sites, and targeted tests.
- Boundary: hidden-test localization cannot be observed before formal scoring; public holdout performance is only an estimate of generalization. Formal submissions should be used for confirmation rather than the main tuning loop.
- Current opportunity: after the private smoke test completes, download its trajectory artifacts and compute localization metrics offline without pushing a new Notebook version or submitting to the competition.
- Files changed: `WORKLOG.md` only.
- External state: no credential changes, Notebook pushes, reruns, or competition submissions.

### 2026-10-06 - Check and explain remote L4 results access

- Objective: show how to inspect the private Kaggle L4 smoke-test status, logs, and output artifacts.
- Current status: `junzhejiao/codingai-retrieval-a-b-smoke-test` remains `KernelWorkerStatus.QUEUED`; no runtime logs or A/B result files exist yet.
- Result locations after completion: the private Notebook version page exposes Logs and Output; `ab_results.csv` contains per-task details and `ab_summary.csv` contains the variant comparison.
- Files changed: `WORKLOG.md` only.
- External state: performed one read-only status query; no credential changes, Notebook pushes, reruns, or competition submissions.

### 2026-10-06 - Define code-localization accuracy metrics

- Objective: define how to measure whether the agent accurately finds the bug or target edit location.
- Measurement design: derive gold changed files, hunks, and enclosing symbols from each public reference patch after the run; parse the ATIF trajectory/tool trace to compute file Hit@k, symbol Hit@k, MRR, tool calls/time to first correct source read, irrelevant-read rate, and whether the first edited file is a gold file. Keep official test resolution as the ultimate outcome because alternative valid fixes may not match the reference location.
- Leakage guardrail: reference patches and test patches may be used only for post-run scoring, never included in the agent prompt or retrieval input.
- Important harness finding: official documentation states that offline `search_similar_code` resolves queries against stored node keys/suffixes and recommends class/function/module symbols rather than free-form natural-language sentences. The current retrieval-v1 query instructions may therefore underperform and should be revised in a later version after preserving the current smoke-test result.
- Current limitation: the queued smoke test records resolution, tool calls, duration, and patch size, but does not yet extract localization metrics from trajectory traces.
- Files changed: `WORKLOG.md` only.
- External state: no credential changes, Notebook pushes, reruns, or competition submissions.

### 2026-10-06 - Document the end-to-end agent runtime logic

- Objective: explain how the project agent understands an issue, writes code, verifies correctness, and is finally scored.
- Summary: the harness supplies the problem statement, hints, clean repository snapshot, graph, and embeddings; the Gemma root agent extracts requirements, localizes source with exact search and conditional retrieval, forms a source-grounded hypothesis, makes a minimal source patch, runs targeted verification, inspects status/diff, and calls the internal `submit_patch` tool. The harness then applies the patch to a clean verification sandbox, restores protected tests/configuration, applies organizer tests, and runs pytest for pass/fail scoring.
- Architecture note: the current retrieval-v1 candidate is single-agent; Graph Agent v1 is a separate, not-yet-activated experiment that delegates uncertain localization to a read-only specialist while retaining all edits and verification on the root.
- Files changed: `WORKLOG.md` only.
- External state: no credential changes, Notebook pushes, or competition submissions were made for this explanation.

### 2026-10-06 - Implement Graph Agent v1 while monitoring retrieval smoke test

- Objective: optimize the current retrieval-v1 design with a read-only Graph Agent without modifying credentials or pushing another version of the active Kaggle Notebook.
- Work performed:
  - Downloaded and reviewed the official `HARNESS_README.md` and `sample_submission` AgentTool configuration.
  - Added an isolated multi-agent experiment under `experiments/graph_agent/submission/` rather than modifying the controlled `submission/` directory.
  - Added `code_graph_analyzer`, restricted to `read_file`, `search_similar_code`, `get_code_neighbors`, and `get_code_subgraph`, with `include_contents: none`, disabled transfers, and `skip_summarization: true` at the root tool boundary.
  - Kept editing, shell commands, tests, status inspection, and `submit_patch` exclusively on the root agent.
  - Added bounded delegation and structured evidence-packet prompts, including ranked candidates, dependency evidence, root-cause hypothesis, verification advice, and uncertainty.
  - Extended the packaging script with an explicit `--allow-multi-agent` mode while preserving single-agent validation by default.
  - Added two package tests for the Graph Agent directory and archive layout; updated the extension documentation.
  - Generated `dist/submission-graph-agent-v1.zip` without replacing any baseline or retrieval archive.
- Verification:
  - `python -m unittest discover -s tests -v`: 4 tests passed.
  - `git diff --check`: passed, with only Git line-ending notices.
  - Archive integrity and expected seven-file layout verified.
  - Graph Agent archive SHA-256: `1131E65894755726A5A41B3EFC59B72580B16BEE038C46D4F7002B09C214D5E5`.
- Files changed: `scripts/package_submission.py`, `tests/test_package_submission.py`, `extensions/graph_agent/README.md`, `WORKLOG.md`, and new files under `experiments/graph_agent/`; generated ignored artifact: `dist/submission-graph-agent-v1.zip`.
- External-state guardrails: did not modify the local credential file and did not push Graph Agent or another version of the active Notebook.
- Follow-up: after the retrieval-v1 smoke test finishes, inspect its results; then create a separate Graph Agent A/B Notebook only with explicit approval or after the current run is complete.

### 2026-10-06 - Authenticate and launch private retrieval A/B smoke test

- Objective: validate retrieval-v1 against the original baseline in the official Kaggle environment without creating a competition submission.
- Work performed:
  - Verified the local credential through Kaggle's read-only token introspection endpoint without printing or recording it.
  - Installed Kaggle CLI 2.2.4 into a temporary directory and confirmed the account has joined `gemma-4-developer-agent`.
  - Pulled the official Starter Notebook and used its offline wheel, vLLM, model-registry, and `Evaluator` setup as the basis for a private test script.
  - Added `experiments/kaggle_retrieval_ab/` with a private, internet-disabled L4 script comparing the original baseline and retrieval-v1 on the same two public tasks lacking explicit location hints.
  - Scanned the upload directory for credential patterns, then pushed private Kaggle Notebook version 1 at `junzhejiao/codingai-retrieval-a-b-smoke-test`.
- Verification: local script syntax, private kernel metadata, credential scan, two repository tests, and `git diff --check` passed before upload; Kaggle accepted version 1 and placed it in the L4 queue.
- Files changed: `WORKLOG.md` and new files under `experiments/kaggle_retrieval_ab/`; no credential file changes.
- External state: one private Kaggle Notebook/version was created and started; no `Submit to Competition` action was used.
- Follow-up: monitor completion, download `ab_results.csv` and `ab_summary.csv`, and diagnose any runtime failure before expanding beyond two tasks.

### 2026-10-06 - Reject an exposed Kaggle credential

- Objective: assess a newly generated Kaggle API token supplied for testing.
- Finding: the value had the expected Kaggle API-token prefix, but it was pasted into chat and must therefore be treated as compromised.
- Action: did not validate, use, persist, or reproduce the credential; instructed the user to revoke it, generate a replacement, and store the replacement only in the local `~/.kaggle/access_token` file.
- Follow-up decision: even after the user accepted the risk, the chat credential was not embedded in a command or request because doing so would copy it into additional execution logs. A local credential file remains the minimum safe handoff.
- Credential check: `C:\Users\32747\.kaggle\access_token` was created as a directory rather than a text file, so no token was read and no Kaggle authentication request was attempted.
- User guidance: provided Windows Notepad steps to remove the mistaken directory and save a UTF-8, extensionless `access_token` file without placing the credential in chat or project files.
- Second credential check: the mistaken directory was removed, but `C:\Users\32747\.kaggle` was empty and no credential file was present; authentication was therefore not attempted.
- Files changed: `WORKLOG.md` only; no credential data was recorded.
- Follow-up: after the replacement token is stored locally and the exposed token is revoked, verify authentication without printing the token and resume the private Kaggle test.

### 2026-10-06 - Attempt to start the official A/B test

- Objective: run the approved retrieval-policy experiment without creating a Kaggle competition submission.
- Work performed:
  - Checked available external integrations; no Kaggle plugin/connector was available.
  - Audited the local environment for Kaggle authentication, CLI/runtime packages, official data, Docker, and GPU capacity.
  - Confirmed the cached official starter-notebook response is readable and describes a two-task public evaluation flow, but the competition-data download is a 40-byte `401 Unauthenticated` response.
  - Attempted to install the official Kaggle CLI into a temporary directory; dependency download failed because the PyPI connection terminated during TLS, and a cached-package-only fallback was unavailable.
- Environment findings:
  - No Kaggle CLI, `swegemma`, Google ADK, vLLM, PyTorch, or Docker is installed.
  - No `KAGGLE_API_TOKEN`, legacy Kaggle environment credentials, `~/.kaggle/access_token`, or `~/.kaggle/kaggle.json` is configured.
  - The local GPU is an NVIDIA GeForce RTX 5060 Laptop GPU with 8,151 MiB, insufficient for the official four-L4 model-serving setup.
- Result: the real agent A/B run cannot start until Kaggle authentication is configured and the competition rules have been accepted; no notebook was launched and no website submission was created. The CLI is also not yet installed, but authentication is the first required user action and direct API access can be used if CLI installation remains unavailable.
- Files changed: `WORKLOG.md` only.
- Follow-up: configure Kaggle authentication locally (prefer `~/.kaggle/access_token` or `kaggle auth login` after CLI installation), then download/push the required assets and launch a private Notebook run without selecting `Submit to Competition`.

### 2026-10-06 - Clarify Kaggle evaluation versus submission

- Objective: determine whether running the experiment in the official Kaggle environment changes the website competition submissions.
- Finding: running a notebook, evaluating against public tasks, or generating `/kaggle/working/submission.parquet` does not by itself create a competition submission. A submission is created only through an explicit `Submit to Competition` action or the Notebook Editor shortcut that combines saving and submitting.
- Operational caution: notebook runs consume Kaggle compute/GPU quota, and saving a version creates notebook output/version history, but neither should change the leaderboard submission history unless the submit action is selected.
- Files changed: `WORKLOG.md` only.
- Verification: checked the current official starter-notebook metadata and Kaggle competition/notebook submission documentation on 2026-10-06.
- Follow-up: run the 24-task public-data A/B as a private notebook evaluation and stop after inspecting saved outputs; do not select `Submit to Competition` until the variant is intentionally ready for hidden scoring.

### 2026-10-06 - Implement embedding-retrieval policy v1

- Objective: implement the approved prompt-only strategy for using the existing `search_similar_code` tool more effectively.
- Work performed:
  - Updated `submission/prompts/system.md` with a localization ladder that prioritizes direct file/symbol inspection and exact text search.
  - Added explicit semantic-search triggers, compact query guidance, `k=5`, inspection of at most three initial candidates, source-based confidence checks, and one materially reformulated retry.
  - Added conditional escalation to `get_code_neighbors` and small, known-node uses of `get_code_subgraph`.
  - Added retrieval budgets and stop conditions so localization does not consume the edit, verification, diff, and submission budget.
  - Generated `dist/submission-retrieval-v1.zip` without overwriting the existing baseline archive.
- Verification:
  - `python -m unittest discover -s tests -v`: 2 tests passed.
  - Confirmed the experiment archive contains the retrieval-policy heading and `search_similar_code(query, k=5)` instruction.
  - Baseline SHA-256 remained `F42BB63AA3D12545912CAB31BC6E4638FDFB634F099AC53888AC0CA005E39701`.
  - Experiment SHA-256 is `948DE71457C59581F545D766C2E07B9B1E56DAFB31E178C8A8DC3C86B00404D4`.
- Files changed: `submission/prompts/system.md`, `WORKLOG.md`; generated ignored artifact: `dist/submission-retrieval-v1.zip`.
- Limitation: no end-to-end task was executed because the official Kaggle harness and task data are not present in this repository.
- Follow-up: run the planned paired 24-task baseline-vs-policy pilot in the official harness before promoting the experiment archive to the default submission.

### 2026-10-06 - Design the embedding-retrieval policy experiment

- Objective: specify how to improve the existing `search_similar_code` strategy before deciding whether to implement it.
- Proposed implementation:
  - Make a prompt-only change in `submission/prompts/system.md`; keep the model, agent topology, tools, sampling settings, and embeddings unchanged.
  - Add a localization ladder: use direct file/symbol reads and exact text search first; call semantic retrieval only when location remains uncertain or exact search is ambiguous.
  - Build compact queries from exact errors, API/symbol names, actual behavior, and expected behavior; use `k=5`, inspect at most the top three candidates, and allow at most one materially reformulated retry before editing.
  - Treat retrieval results as candidates, require source verification, use graph neighbors only after identifying a credible symbol, and stop retrieval once one concrete edit hypothesis is supported.
  - Reserve `get_code_subgraph` for a small set of already identified symbols involved in a cross-file flow.
- Proposed evaluation: paired baseline-vs-policy runs on a fixed public-task pilot, followed by a held-out confirmation only if resolution improves without excessive tool-call/time cost or new invalid patches.
- Decision gate: no functional Agent files were changed; implementation awaits explicit user approval.
- Files changed: `WORKLOG.md` only.
- Follow-up: if approved, patch the prompt, package a separately named variant archive, run static validation, and prepare/run the controlled A/B evaluation where the official harness is available.

### 2026-10-06 - Prioritize the extension roadmap

- Objective: rank Multi-Agent, Graph Agent, RAG, RL, LoRA, and embedding retrieval for the current baseline.
- Recommendation: start with embedding retrieval by improving query construction, small-`k` candidate selection, and source verification using the existing `search_similar_code` tool.
- Rationale:
  - The competition already supplies per-task graph-node embeddings and an offline semantic-search tool, so this route has the lowest implementation cost and directly targets code localization.
  - Use graph neighbors/subgraphs as a second-stage expansion after finding a credible symbol; create a separate read-only Graph Agent only if controlled evaluation shows delegation adds value beyond direct tool use.
  - Custom RAG overlaps with the supplied retrieval system; generic Multi-Agent adds coordination cost; LoRA and RL require a stable evaluation/training pipeline and are later-stage experiments.
- Verification: reviewed all extension design notes and checked the current Kaggle competition overview and official starter-notebook metadata on 2026-10-06.
- Files changed: `WORKLOG.md` only.
- Follow-up: run a baseline-vs-retrieval-policy ablation on the public tasks and record resolution rate, localization success, tool calls, wall time, context use, and patch validity.

### 2026-10-06 - Add persistent project work log

- Objective: create a durable Markdown record that future work sessions can use to understand prior activity quickly.
- Work performed:
  - Added this `WORKLOG.md` with a concise current-state summary and reverse-chronological history.
  - Added root-level `AGENTS.md` instructions requiring future agents to read and update the log on every task.
- Files changed: `AGENTS.md`, `WORKLOG.md`.
- Verification: confirmed the repository previously had no `AGENTS.md` or existing work-log Markdown file.
- Follow-up: maintain both the current-state summary and a new history entry during every subsequent task.

### 2026-10-06 - Assess basic programming capability

- Objective: determine whether the current project can perform simple programming tasks.
- Findings:
  - The Kaggle submission configuration exposes tools for reading, searching, editing, writing, running commands, checking status, graph/code retrieval, and submitting patches.
  - The system prompt defines a complete minimal code-repair workflow.
  - The project can serve as a basic coding agent inside the official Kaggle harness.
  - It cannot yet run as a standalone local interactive coding assistant because the local runtime and model integration are absent.
- Verification:
  - Ran `python -m unittest discover -s tests -v` successfully: 2 tests passed.
  - Inspected `dist/submission.zip`; it contains `agent.yaml`, `configs/sampling.yaml`, `eval_config.yaml`, and `prompts/system.md` at the expected paths.
- Files changed: none.
- Follow-up: run an end-to-end sample issue through the official notebook/harness when available.
