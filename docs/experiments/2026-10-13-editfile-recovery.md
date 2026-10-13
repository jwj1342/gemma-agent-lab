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

- Source commit and configuration directory: candidate frozen at `96a98ab`
  (branch `experiment/editfile-recovery`), built from `configs/prompt-icl` with
  `--variant recover` into `runs/editfile-recovery/candidate`. The embedded
  harness scripts come from `4028a6b` (branch `fix/live-wheelhouse-2026-10-12`),
  which adapts `validate_agent.py` and `evaluate_agent.py` to the October 12
  official API migration; the agent bundle itself is unchanged.
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

- Agent ZIP SHA256: `025322b7cf5f25edcb22d77d6a64cf4dcefcd474094cc0491c4e4dc91323b7f5`
  (identical to the frozen candidate announced in issue #11).
- Model handle/revision and adapter hash: `gemma-4-31b-it-qat-w4a16-ct` v2,
  no adapter.
- Environment/image, dependency versions, and seed: official starter image
  pinned in kernel metadata; official wheelhouse mounted by the platform
  (unreviewed record-mode snapshot, all 41 wheel SHA256 captured in
  `inputs/wheelhouse.json`). Seed 42, temperature 0.2, thinking budget 1024.
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
protocol-vs-recover comparison belongs to the local version-29 fast cohort
(pending @jwj1342 coordination).

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
  - Tool calls: 13 total — `read_file` 5, `run_command` 5, `edit_file` 5,
    `submit_patch` 1. LLM calls: 17.
  - Missing-argument events: **0**. Tool error events: **0**. Longest
    repeated-error streak: **0**. Nonzero `run_command` results: 0. Unparsed
    observations: 0.
  - The heredoc fallback never fired: all five `edit_file` calls were accepted
    by the live harness. The protocol smoke on v29 had three malformed editor
    calls on the same task. Two variables changed at once (prompt and official
    environment), so this clean result cannot be attributed to the `recover`
    paragraph alone — the rebuilt harness wheels may have changed the
    serialization behavior. The v29 fast-cohort comparison is the attribution
    experiment; a fresh v29 baseline trace would settle whether the live
    wheelhouse still exhibits the corruption at all.
- Comparison with the baseline on matching tasks: not measured in the same
  recorded environment; see the note above. Protocol smoke (v29, rich_3454):
  3 malformed editor calls, 0 fallback uses, resolved (per the #10 report).

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
- Conclusion and next experiment: the recover candidate completes the official
  smoke cleanly on the live environment and resolves `rich_3454`; whether the
  recovery paragraph matters can only be judged on the frozen v29 fast cohort
  where the corruption was observed. Next: fixed fast cohort on the local v29
  runtime with @jwj1342 (issue #11 step 3), paired against protocol/baseline.
