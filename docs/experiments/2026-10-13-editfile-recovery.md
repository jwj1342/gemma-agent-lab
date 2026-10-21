# Edit-file recovery prompt: first-error heredoc fallback

## Question and comparison

- Hypothesis: entering the intact single-argument `run_command` heredoc channel on
  the **first** malformed `edit_file` error (instead of after two repeated errors)
  shortens repeated-error streaks, reduces wasted editor retries and improves
  clean completion. The fallback is given as a copyable call form instead of prose.
- Baseline and proposed change: `protocol` prompt (workflow + editor argument
  guidance + prose recovery after two identical errors) vs `recover` (same base,
  one recovery paragraph replaced with a first-error heredoc form). Model, tools,
  sampling, task budget and adapters unchanged. See issue #11 for the exact diff
  and hypothesis.
- Task IDs / grouped split and data version: public smoke task `rich_3454`,
  mounted Kaggle competition inputs. Wheelhouse: **live official wheelhouse
  re-published 2026-10-12 19:51 UTC** (record mode; see the environment note).
- Success metric: missing-argument events, longest repeated-error streak, clean
  completion, issue resolution, tool calls, elapsed time.

## Reproduction

- Source commit and configuration directory: the smoke used the candidate frozen
  at `96a98ab` (branch `experiment/editfile-recovery`), built from
  `configs/prompt-icl` with `--variant recover` into
  `runs/editfile-recovery/candidate`. The embedded harness scripts come from
  `4028a6b` (branch `fix/live-wheelhouse-2026-10-12`), which adapts
  `validate_agent.py` and `evaluate_agent.py` to the October 12 official API
  migration; the agent bundle itself is unchanged.
- Exact preparation and execution commands:

  ```bash
  python scripts/build_prompt_candidate.py --variant recover \
    --output runs/editfile-recovery/candidate
  python scripts/prepare_kaggle_runtime.py --owner losttrace \
    --agent-dir runs/editfile-recovery/candidate --task-id rich_3454 \
    --slug gemma4-recover-smoke --wheelhouse-mode record \
    --output runs/editfile-recovery/notebook-livefix
  kaggle kernels push -p runs/editfile-recovery/notebook-livefix \
    --accelerator NvidiaL4 --timeout 2700
  ```

- Evaluated and revised prompt: the smoke and fast-cohort results used the
  pre-correction prompt preserved in `96a98ab`/`be754dc`, LF-normalized text SHA256
  `375b96b66675f0e9ff2faf7db7edcee7eb617e2543f5fb6836c3567b1fe83977`. The
  corrected fallback recipe in this revision builds LF-normalized text SHA256
  `4f22136d22565058ab6b60df5f756baadde12c6134b1f8073c87c8b08c507190`; it has
  not been run, and the historical results are not attributed to it. To rebuild
  the evaluated candidate, check out `be754dc` and run the command above.
  These are the builder's UTF-8 text hashes before platform newline conversion,
  not necessarily file-byte hashes. The retained smoke ZIP contains a CRLF
  `prompts/system.md` with raw SHA256
  `0436696854b4b4d895703e86f8c14d8ab223caf6436e6a5ef35ef8605b452695`;
  replacing CRLF with LF gives the historical `375b96b6…` hash above.
- Agent ZIP SHA256: the evaluated candidate is
  `025322b7cf5f25edcb22d77d6a64cf4dcefcd474094cc0491c4e4dc91323b7f5`
  (4,219 bytes; identical to the frozen candidate announced in issue #11). The
  revised candidate builds
  `5714a9abc0a33f73055ae25c3d73bb9d8c0beb0ac8fba0fe18fa3266090016f8`
  (4,423 bytes).
- Model handle/revision and adapter hash: `gemma-4-31b-it-qat-w4a16-ct` v2,
  no adapter.
- Environment/image, dependency versions, and seed: official starter image
  pinned in kernel metadata; official wheelhouse mounted by the platform
  (unreviewed record-mode snapshot, all 41 wheel SHA256 captured in
  `inputs/wheelhouse.json`). Seed 42, temperature 0.2, thinking budget 2048
  (the frozen `inputs/agent/configs/sampling.yaml` records 2048; no runtime
  override).
- Platform and actual GPU/CPU allocation: Kaggle L4×4, offline, private notebook
  `losttrace/gemma4-recover-smoke-rich-3454` version 3.
- Context, parallelism, memory fraction, and time/tool budgets: tensor parallel 4,
  GPU memory utilization 0.90, 4 min / 40 calls / 80 turns / 60 s per task.

## Environment note

The official wheelhouse was re-published 2026-10-12 19:51 UTC (all 41 wheels
re-created). Versus snapshot `official-v29-2026-10-08`: `adk_eval_core`,
`adk_submission` and `vllm` changed SHA256; `swegemma` moved 0.2.10 → 0.2.11 and
removed `swegemma.models.discovery`/`swegemma.models.registry` (the harness fix
branch adapts to this; see issue #5). This run therefore uses
`--wheelhouse-mode record` on the live wheelhouse — an unreviewed snapshot, not
the recorded v29 environment the protocol smoke used. The environment-controlled
protocol-vs-recover comparison was originally planned for the local version-29
runtime; the team subsequently ran a matched version-31 cohort (see the v31
update below). A later comparison of all 41 retained wheel hashes matches
`official-v31-2026-10-12` exactly. The recover smoke therefore used v31 wheels,
not v29; its Kaggle L4 runtime remains distinct from the local H100 runtime.

## Results

- Original-failure and reference-success controls: valid — empty patch fails the
  tests, gold patch passes (83 tests).
- Pipeline execution / issue resolution: full pipeline executed; the official
  compiler passed on the live wheelhouse; the agent produced a real nonempty
  patch (1,482 chars) and the verification passed: **resolved**, test exit 0,
  83 passed in 0.12 s.
- Model startup, agent time, total elapsed time, and quota used: inference
  152.2 s (agent 151.2 s) against the 4-minute budget; whole kernel ~13 min
  wall; Kaggle GPU quota consumed ~1.6 h (L4×4).
- Measured metrics, failure stages, and limitations:
  - Tool calls: the retained trace records 16 invocations — `read_file` 5,
    `run_command` 5, `edit_file` 5 and `submit_patch` 1. `submit_patch` is
    budget-free (1 call); the harness's budget counter reported 13 consumed
    calls, two fewer than the 15 recorded calls to budgeted tool types. The
    retained receipts do not explain the two-call gap, so both numbers are
    reported separately rather than combined. `get_status` was not called. LLM
    calls: 17 (harness counter; the trace records 18 agent turns, including the
    final message).
  - Missing-argument events: **0**. Tool error events: **0**. Longest
    repeated-error streak: **0**. Nonzero `run_command` results: 0. Unparsed
    observations: 0.
  - The heredoc fallback never fired: all five `edit_file` calls were accepted
    by the live harness. The protocol smoke on v29 had three malformed editor
    calls on the same task. Two variables changed at once (prompt and official
    environment), so this clean result cannot be attributed to the `recover`
    paragraph alone — the rebuilt harness wheels may have changed the
    serialization behavior. The matched v31 cohort below is the available
    attribution experiment: malformed-call events still occur on v31 for
    protocol and recover, but not for baseline in that comparison.
- Comparison with the baseline on matching tasks: not measured in the same
  recorded environment; see the note above. Protocol smoke (v29, rich_3454):
  3 malformed editor calls, 0 fallback uses, resolved (per the #10 report).

### Matched v31 update (2026-10-13)

The team ran the frozen recover candidate on the fast cohort (10 of 12 tasks
had valid controls) in the
version-31 environment, together with baseline and protocol
([PR #23](https://github.com/jwj1342/gemma-agent-lab/pull/23), jobs
1090123–1090125). Results: baseline 1/10, protocol 2/10, recover 3/10; recover
recorded 15 missing-argument events and a longest error streak of 3, versus 22
and 5 for protocol. This is a single-attempt comparison on local H100, not a
repeat of the Kaggle L4 smoke. The original recover smoke results are retained;
they used live v31 wheels, whereas the earlier protocol smoke used v29.

## Artifacts and next step

- Private artifact location and access instructions: Kaggle kernel
  `losttrace/gemma4-recover-smoke-rich-3454` version 3 (private), outputs
  downloaded to `runs/editfile-recovery/output-v3/` (result JSONs, trace,
  wheelhouse fingerprints, logs; triton caches excluded).
- Summary, trace, patch, and log paths/hashes: trace
  `runtime-smoke/infer/results/traces/trace_rich_3454.json` (new
  `adk_eval_core` session schema); wheelhouse receipt
  `runtime-smoke/inputs/wheelhouse.json` (status `unreviewed_snapshot`, 41
  wheels — fingerprints for the registry entry in issue #5).
- For training: not applicable.
- Conclusion and next experiment: the evaluated recover candidate completed the
  official smoke cleanly and resolved `rich_3454`; the matched v31 cohort
  resolved 3/10 versus protocol 2/10 and baseline 1/10, a single-attempt
  comparison. The corrected fallback recipe in this revision is a new candidate
  that has not been run and must be evaluated separately before its results can
  be compared with these historical numbers. Next: run the revised candidate on
  a frozen, environment-matched cohort; keep the 70-task holdout sealed.
