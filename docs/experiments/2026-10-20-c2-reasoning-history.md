# Proposed C2 ablation: omit retained thought parts

Status: CPU preparation, not an evaluated improvement. Follow-up to
[issue #12](https://github.com/jwj1342/gemma-agent-lab/issues/12).

## Why this experiment

The owner's [difficulty breakdown](https://github.com/jwj1342/gemma-agent-lab/issues/7#issuecomment-6281286094)
shows that C2's L4 gains were concentrated on easier tasks. That supports reporting the
hard slice separately; it does not establish that the hidden tasks have the same distribution.
One leaderboard draw per candidate cannot distinguish a small gain from variance.

A local audit of the [r5 traces](https://github.com/jwj1342/gemma-agent-lab-data/releases/tag/kaggle-l4-r5)
separated attempted edits from accepted edits and final patches. In the strict subset
that received reference-hunk lines but produced no reference-file patch, 13 task-runs
across eight tasks ended with empty patches: six agent timeouts, six call-limit stops,
and one replacement-matching failure after a correct diagnosis. A further partial
multi-file repair was kept separate. These are retrospective diagnostic labels, not
permitted inference inputs or a representative estimate for the hidden benchmark.
Graph replay on selected symbols had mixed utility; it does not justify unconditional graph calls.
Neither timing gaps nor repeated hypotheses prove that compaction caused these failures.

The frozen C2 package retains thought parts with a positive 4096-token budget.
A cheap, isolated question is whether omitting those parts from subsequent history
helps the agent reach and execute a source edit within the same limits.

## Intervention and provenance

Change only `thinking_config.include_thoughts: true` to `false` in the exact
`kaggle-l4-r5/c2/agent` directory. Keep thinking budget 4096, output budget 8192,
temperature 0.2, top-p 0.95, prompt, tools, and evaluation limits unchanged.
This does **not** disable current-turn reasoning or halve its budget.
The builder rejects any control whose file inventory or hashes differ from
[the pinned manifest](../../configs/experiments/c2-r5-control.json), then records
both inventories and creates the deterministic ZIP outside the agent directory.
The older baseline on `main` is not a substitute for this control.

The public budget-fit prompt is credited to `honghanhhh` (Apache 2.0), with the
team's C2 modification. This PR does not vendor that prompt, task data, or traces;
retain upstream attribution when distributing the generated derivative.

Static inspection and a local helper-only probe used `adk_submission 0.2.13`,
wheel SHA256 `f156f0bd84580ccb9334726d114d7a5481f16d3a6e2887cc4d276ced0ef8d99d`.
Its generation helpers keep current reasoning enabled for a positive budget,
remove prior reasoning fields, and filter response thought parts while preserving
visible text, native tool calls and usage metadata in the probed fixtures.
The probe extracted four official functions and stubbed global patch installation;
it was not an ADK/vLLM integration test or an official compilation result.
Actual request/history handling remains a required smoke check.

## Reproduce CPU preparation

Install the repository's CI requirements in a development environment. Obtain the
frozen C2 directory from the r5 release and use its actual local path:

```bash
python scripts/build_history_candidate.py \
  --source /path/to/gemma-agent-lab-kaggle-l4-r5/c2/agent \
  --output runs/2026-10-20-c2-no-thought-history
```

The output contains `agent/`, `submission.zip`, `submission.manifest.json`, and
`ablation.json`; only agent files enter the ZIP. Existing run directories are
rejected. Expected candidate ZIP SHA256:
`aba592678322f993d1360b9362975283ac12d5da335295be2db35b2477a7b942`.
Local packaging and CRC checks are distinct from the official compiler gate.
Executed locally: repository checks and all 68 CPU unit tests passed. The builder
also reproduced the candidate ZIP hash above from the frozen release files.
No official compilation, GPU inference or leaderboard submission was performed.

## Matched L4 test, manual execution only

1. Coordinate quota and the route in #12 before GPU execution. Use the same
   runtime/model revisions as the control. Compile both agents with the official
   runtime, run the existing exposed smoke, and inspect actual history/request
   handling, native tool calls and accounting. Stop if the intervention is not isolated.
2. Freeze both agents, ZIPs, public task inputs, runtime identity and seeds using
   the existing evaluation preparation workflow. Do not insert reference patches,
   hindsight labels or task-specific audit findings into either agent's inputs.
   Keep the protected holdout sealed.
3. Compare C2 and candidate on the unchanged dev cohort, seed 42; counterbalance
   execution order. If worth following, repeat a matched seed-1 pair rather than
   comparing against an unmatched historical run. Preserve every outcome.
4. Report paired wins/losses, valid-control coverage, hard-slice and total resolution,
   empty final patches, accepted edits, tool errors/rejections, clean completion,
   agent/suite wall time and prompt/completion tokens where available. Use the
   owner's direct historical difficulty manifest; any reconstructed mapping must
   be labeled as inferred. Inspect reasons for hard-task regressions.
5. Advance only with repeatable unhinted repair gains and acceptable total runtime;
   shorter history or fewer empty patches alone are insufficient. Coordinate any
   later leaderboard slot with the owner. No submission is triggered by this PR.

Thought parts may be absent in candidate traces by design; that is not evidence
that current reasoning was disabled. Lost working memory, repeated recomputation,
and degraded repair are plausible negative outcomes. Retain a negative result.
If this does not help, prioritize the task-balanced, native-tool repair transfer
experiment in [#27](https://github.com/jwj1342/gemma-agent-lab/issues/27), evaluated
without hints on task-disjoint cases; keep selective graphs a separate ablation.
