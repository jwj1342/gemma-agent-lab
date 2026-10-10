# Prompt and context adaptation: initial execution record

Status on October 10, 2026: research, portable candidate construction, official
compilation, controlled smoke tests and the first matched fast-cohort comparison
are complete. ICL resolved 3/11 eligible tasks versus the baseline's 2/11, with two
gains and one regression. One attempt per task on this small public cohort does not
establish a reliable improvement. All candidates use zero weight updates.

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
| `protocol` | Passed | Yes | Yes; agent time 73.52 seconds | 3 |

The first two candidates repeatedly omitted the mandatory `old_string` argument
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
Its smoke passed with 22 LLM calls, 16 budget-counted tool calls and 214,220
input/output tokens including repeated context. Inference including sandbox setup
took 108.52 seconds. Three malformed editor calls remained, and the trace did not
use the suggested command-based editing fallback. It also edited a protected test
file, then submitted twice after noticing that the submission tool filtered that
file. The final graded patch changed only the implementation file. This gives a
concrete remaining target: explain protected-file filtering and successful
termination clearly. It does not prove that the recovery policy caused the repair.

All four candidate packages passed official compilation. The registered five-way
comparison has now finished. Prompt packages stayed frozen throughout execution.

### Fast-cohort coverage

Under version 29, all five runs have valid empty/reference
controls for **11 of the fixed 12 tasks**. `fastapi_15745` remains listed but is
ineligible: its reference patch yields 33 passed and 3 skipped tests, including a
required asynchronous test. The official verifier rejects that skipped required
test even though pytest exits successfully. Keep the verifier unchanged and report
the 11/12 coverage alongside each score; do not substitute another task.

## Completed fast-cohort comparison

The [versioned result record](../../resource/manifests/prompt-fast-v29.json) contains
task outcomes, agent/runtime hashes, paired comparisons and trace-audit hashes.
All suites used one H100, seed 42, the required W4A16 model and identical
four-minute / 40-tool / 80-turn limits. The comparator checked task membership,
control masks, model, verifier, benchmark and budget identities before pairing.

| Candidate | Resolved / eligible | Clean and resolved | Clean completion | Timeouts | Inference minutes | LLM calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2/11 | 2 | 8/11 | 2 | 39.89 | 413 |
| Workflow | 2/11 | 2 | 7/11 | 4 | 42.97 | 418 |
| ICL | 3/11 | 3 | 7/11 | 2 | 38.62 | 434 |
| Reflection | 2/11 | 1 | 6/11 | 3 | 40.70 | 468 |
| Protocol recovery | 2/11 | 1 | 7/11 | 3 | 41.21 | 373 |

Resolution uses independent verification of the recovered patch. Reflection and
protocol both left a passing `httpx_3672` patch after timing out; those outcomes
are not clean completions. Inference minutes include per-task setup but exclude
model startup, controls, grading and queue time. Whole-run timing is affected by
control-cache reuse and should not be used directly to rank prompt efficiency.
These H100 timings do not establish L4 timing.

| Candidate | Newly resolved versus baseline | Regressions |
| --- | --- | --- |
| Workflow | `rich_3278` | `requests_7502` |
| ICL | `rich_3278`, `fastapi_9753` | `requests_7502` |
| Reflection | `fastapi_9753` | `requests_7502` |
| Protocol recovery | None | None |

ICL's net gain is one task, or 9.1 percentage points on the eligible cohort.
Its descriptive 95% binomial interval is wide; task correlation, stochastic
generation and adaptive development further limit interpretation. Keep it as a
candidate for a larger frozen comparison, with the holdout still sealed.

### Tool behavior and cost

| Candidate | Missing-argument events | Longest repeated-error streak | Graph calls | Skill calls | Total model tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0 | 3 | 0 | 0 | 4,046,028 |
| Workflow | 31 | 13 | 5 | 0 | 3,851,510 |
| ICL | 13 | 4 | 4 | 0 | 4,100,594 |
| Reflection | 104 | 65 | 1 | 0 | 4,564,302 |
| Protocol recovery | 12 | 4 | 0 | 0 | 3,559,466 |

Token totals include repeatedly supplied conversation context. Every eligible
task had a trace; missing-argument events are observed tool feedback, rather than
a complete schema validator. Nonzero commands may be intentional reproductions.
Reflection entered a 65-call malformed-write loop on `rich_4079`. Its optional
playbook was never retrieved in these runs, so its utility remains unmeasured.
Graph use occurred in some candidates, but no matched graph-only ablation was run;
these counts do not establish a graph benefit.

The protocol candidate used about 12% fewer model tokens than baseline while
resolving the same tasks, but took slightly longer and had fewer clean successes.
ICL used about 1.3% more tokens. None of the prompts consistently improved repair,
tool validity and completion together. A practical next comparison should retain
ICL, address malformed calls with task-independent schema guidance, and measure
regressions on a larger development cohort before a final holdout decision.

## Provenance and next measurements

### Registered development expansion, October 10

Job `1067715` evaluates baseline then ICL on the pre-existing 40-task development
cohort. Both agent ZIPs are byte-identical to their fast-cohort packages; no prompt
was revised after inspecting fast outcomes. Both reuse the exact version 29 runtime
and generation budget from the frozen fast baseline. Results are pending.

The development cohort contains the original 12 fast tasks and 28 additional
tasks. Report all 40 with control coverage and also report the additional 28
separately: the nested full cohort cannot serve as wholly independent confirmation
of a candidate selected on fast results. Keep the 70-task holdout sealed. This
comparison still has one stochastic attempt per task and does not establish
cross-repository or leaderboard generalization.

Local artifacts and the pre-inference registration are in
`runs/20261010-prompt-dev/`. The registration lists the exact additional task IDs,
unchanged agent hashes are recorded in each run manifest, and the runner reuses
eligible cached controls while generating fresh predictions for each candidate.

### Development control coverage

The baseline control phase completed before inference with **31/40 eligible
tasks**: 11/12 from the fast cohort and 20/28 from the additional development
tasks. Nine tasks are excluded from repair-rate denominators under the frozen
control policy. This is reduced coverage, not nine model failures.

| Excluded tasks | Observed reason |
| --- | --- |
| `rich_3472` | Both controls fail collection because `attr` is unavailable. |
| `fastapi_14791`, `fastapi_11194`, `fastapi_14953` | Both controls require the missing `python-multipart` dependency. |
| `fastapi_14964` | Both controls fail collection because `sqlalchemy` is unavailable. |
| `requests_6757` | The reference patch fails an mTLS test with an expired-certificate SSL error. |
| `fastapi_15745` | The official verifier rejects a skipped required asynchronous test. |
| `fastapi_14978` | The reference patch passes; the empty patch has collection errors for the missing `strict_content_type` API. The local eligibility gate excludes this failure form. |
| `fastapi_14512` | The reference patch passes; the empty patch has Pydantic fixture/setup errors. The local eligibility gate excludes this failure form. |

The last two cases concern our conservative negative-control gate, rather than
reference-patch failure. Any later change to dependency preparation or control
eligibility needs a separately frozen comparison for both candidates. This run
keeps its original runtime, task list, control cache and gate unchanged.
`runs/20261010-prompt-dev/control-audit.json` records the source control hash,
individual outcomes and this classification. Inference and paired results remain
pending; the holdout remains sealed.

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
