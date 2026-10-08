# Localization handoff

## Decision: retain the offline improvement; do not promote yet

The portable v1 ranker and learned gate improve offline file localization and are retained on
`feature/localization-v1-skill`. The historical test improves Hit@1 from 31.6% to 47.4% and Hit@5
from 68.4% to 78.9% against BM25. Nested snapshot and repository-held-out validation also improve.
The public test was previously inspected: this is retrospective evidence, not a fresh blind test.

There is **no measured remote repair improvement yet**. Better file ranking does not guarantee
better patches. The gate also loses some historical-test gains compared with the ungated candidate.
Do not repeatedly tune against that test or declare remote success from these numbers.
See the [full study](experiments/2026-10-08-localization-v1-portable.md) for metrics, limitations,
training recovery commands, frozen hashes, and the remote acceptance protocol.

## What the existing remote agent uses for localization

This describes the checked-in baseline configuration, not a newly executed remote trace.
The baseline has no separately trained file ranker. Gemma chooses tools according to its prompts.

| Stage | Existing mechanism | Practical limitation |
| --- | --- | --- |
| Read the issue | Extract paths, functions, classes, CLI commands, and error text | Ambiguous reports may not identify a component |
| Search source | Targeted filename/text search through `run_command`, then `read_file` | Literal matches can miss synonyms and indirect causes |
| Follow symbols | `search_similar_code`, `get_code_neighbors`, and `get_code_subgraph` | Precomputed embeddings do not guarantee free-text search; missing graph results do not establish absence |
| Analyze cause | Delegate to the read-only `code_analyzer` agent | It uses the same source/graph tools and is not an independent learned ranker |
| Verify repair | Main agent edits source and runs a targeted test | Correct localization alone does not establish a correct fix |

The analyzer returns file paths, line numbers, a root-cause explanation, and a proposed minimal change.
The prompt tells it to search source when graph lookup fails, including for async functions.
Evidence: [agent configuration](../configs/baseline/agent.yaml),
[main prompt](../configs/baseline/prompts/system.md),
[analyzer configuration](../configs/baseline/sub_agents/code_analyzer.yaml), and
[analyzer prompt](../configs/baseline/prompts/analyzer.md).

## The optimized module already implemented

The isolated [experiment bundle](../configs/experiments/localization-v1/agent.yaml) adds a
[localization skill](../configs/experiments/localization-v1/skills/localization-v1/SKILL.md).
It leaves the baseline model, analyzer, and sampling settings intact.

1. If the issue already identifies a target, read it directly. Otherwise invoke the skill once.
2. Give it only the issue and mounted source tree, never reference patches or evaluation labels.
3. Enumerate Python files and score content, path, and symbol evidence. The portable mode does not
   load the archived checkpoint or memorize repository-specific path identities.
4. Apply the learned confidence gate: accept portable ranking only when its frozen rule matches;
   otherwise return BM25 ranking. This fallback disables learned ranking, not the whole skill.
5. Return ranked paths, line numbers, and symbols. Gemma confirms candidates by reading source;
   if candidates are unhelpful, its existing exact-search and graph tools remain available.

The module is read-only and standard-library-only. It is not a replacement for root-cause reasoning
or the repair loop, and its existing-file Python scope does not cover every possible task.
Default `adaptive` mode remains BM25 while the promotion switches are disabled. Explicit `portable`
mode exercises the learned candidate and its gate for the controlled experiment.

## Preserved artifacts and next action

Local artifacts remain under `runs/2026-10-08-localization-v1-portable/`:

- `study-final/study.json`: final results, parameters, folds, input/source hashes, and environment.
- `features-final.json`: hash-checked training cache, containing labels; never package it for inference.
- `remote-ab-final/`: the current frozen baseline/BM25/portable comparison, 38 tasks and 114 cases.
- `remote-ab-final/notebooks/`: 18 prepared pilot notebooks over six tasks.
- `submission-localization-v1-portable.zip`: the explicit portable-mode experimental bundle.

The ZIP SHA256 was rechecked at handoff:
`e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48`.
These ignored run artifacts are local, not a remote backup and not included in a clone. Source,
policy, tests, and reports are versioned. Older preparation attempts are retained; use only
`remote-ab-final` for the frozen comparison.

GPU launches and uploads remain manual under the project guide. Check quota and follow the
[recorded launch/recovery procedure](experiments/2026-10-08-localization-v1-portable.md#frozen-remote-comparison).
Valid empty/gold controls must precede inference. The predeclared acceptance rule requires at least
5 percentage points repair improvement over baseline, paired significance, improvement over the
BM25 skill, acceptable timing, and complete valid cases. A six-task pilot alone cannot pass it.

If remote results fail that rule, retain the experiment as an ablation and keep the original baseline.
Use the failures to distinguish missed files from incorrect reasoning or edits before changing the
module. Any newly tuned candidate needs separate confirmation; do not rewrite the frozen protocol.
No upload, GPU job, remote deployment, or automatic background continuation was started at handoff.

## Verification status

The preceding implementation checks passed all 25 focused tests; full-suite results were 39/46,
with seven Windows-dependent ZIP/path expectations still failing. Repository syntax, configuration,
packaging, and documentation checks passed. Official compilation and model execution remain pending.
The handoff changes documentation only and do not change the frozen experiment or its ZIP.
