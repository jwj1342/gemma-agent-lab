# Prompt and context adaptation: initial execution record

Status on October 9, 2026: research, portable candidate construction and repository
CI are complete. The first version 29 smoke executions are recorded below.
Multi-task effectiveness is not yet measured, and no weight updates are used.

## Question and comparison

Can clearer workflow instructions, short examples, execution feedback or compact
playbooks improve repair success within the same inference budget? The
[research review](../prompt-route.md) separates these methods from gradient-based
test-time training and records the competition's supported interfaces.

All candidates use the required model, baseline generation settings, a four-minute
per-task deadline and the same frozen evaluation runtime. `reflect` additionally
registers a packaged skill. The fixed fast cohort contains 12 tasks; its baseline
must be measured before interpreting any candidate score as an improvement.

## Smoke observations

The exposed integration task is `rich_3454`. It is excluded from training. Empty
patch verification failed the intended test and the reference patch passed under
version 29. Runs used one H100, the required W4A16 base model and seed 42.

| Candidate | Official compilation | Patch passes tests | Clean agent completion | Schema-invalid editor calls |
| --- | --- | --- | --- | ---: |
| `workflow` | Passed | Yes | No; four-minute timeout | 20 |
| `icl` | Passed | No | No; four-minute timeout | 32 |
| Original `reflect` | Failed skill frontmatter validation | Not run | Not run | Not measured |
| Corrected `reflect-v2` | Passed | Yes | Yes; agent time 90.69 seconds | 7 |

Both executed candidates repeatedly omitted the mandatory `old_string` argument
from `edit_file`. The harness returned that error each time. Workflow eventually
left a passing patch through a different editing tool, while ICL did not. Neither
trace called graph tools. This single task cannot establish ICL's overall value or
the value of graph retrieval.

Workflow used 34 LLM calls and 14 budget-counted tool calls; ICL used 45 and 13.
Invalid invocations appear in the trace but are not all counted as executed tools.
Count both when diagnosing loops. Recorded model input/output totals were 338,823
tokens for workflow and 446,834 for ICL, including repeated conversation context.

The playbook failure was repaired by using the official loader's kebab-case name
`repair-playbook` and its `references/` directory. The corrected `reflect-v2`
package passed official compilation and smoke verification. It made 24 LLM calls,
15 budget-counted tool calls and one submission; total inference including setup
took 120.45 seconds. It did not call a graph or skill tool, so this success does not
establish a benefit from playbook retrieval. Editor invocation errors remain.

The independent trace audit counted longest identical-error streaks of nine,
thirteen and four calls for workflow, ICL and corrected reflection respectively.
Reflection also edited a protected test file while debugging its generated case.
The official submission/verification code strips protected test changes; its graded
patch contained the implementation change and a temporary reproduction file.
Thus, repair success does not establish instruction compliance or a minimal patch.
Keep generated checks separate from the independent grading result.

Run `scripts/audit_tool_traces.py` after a cohort finishes to reproduce these
diagnostics. It writes trace/source hashes and distinguishes missing arguments from
nonzero command results. It operates outside the frozen evaluation runtime.

A separate `protocol` candidate adds the three editor argument names and a bounded
recovery policy: after a repeated invocation error, use an asserted source rewrite
through `run_command`, inspect the diff, test and finish. This is generic tool-use
guidance derived from the exposed smoke trace, with no task-specific solution.
Its smoke and fast-cohort evaluations are pending.

## Provenance and next measurements

- Branch: `experiment/prompt-icl-route`; [Issue #9](https://github.com/jwj1342/gemma-agent-lab/issues/9)
  and [PR #10](https://github.com/jwj1342/gemma-agent-lab/pull/10).
- Initial candidates: `f3cebb5`; corrected skill: `26bada6`; protocol candidate: `4c64dd9`.
- Local artifact root: `runs/20261009-prompt-icl/` in the main workspace.
- Preserve the failed original `reflect/` and the two timeout traces; use
  `reflect-v2/` for the corrected package.
- Corrected reflection smoke: job `1063875`; baseline/workflow/ICL/reflection
  fast comparisons: job `1063878`; protocol smoke and fast: job `1063897`.
- Raw summaries, traces, patch verification and frozen hashes live in each
  candidate's `smoke/` and `fast/` directory. These ignored artifacts are not in a clone.

Report paired repaired/regressed tasks, invalid calls, timeouts and inference cost.
Keep the holdout sealed. These are research-inspired prompt ablations, not complete
reproductions of GEPA, ACE, SWE-Replay or any gradient-based TTT algorithm.
