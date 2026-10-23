# Matched r5 traces: where successful and failed attempts diverge

## Selection fixed before this comparison's semantic reading

Use the lexicographic first three task IDs with valid controls, traces and
predictions in all five r5 arms, and discordant C2 / C2-seed1 outcomes. This
selects three tasks and all 15 associated attempts, including successes and
failures. The selector uses outcome metadata, not trace contents. Historical
familiarity with this release exists; this is not a blinded or prospective study.
The comparison is deliberately outcome-selected and cannot estimate mechanism
prevalence, population effect sizes or causality. H100 r4 is not pooled with L4.

Cases A, B and C follow that sorted order. The private comparison retains task
IDs, source observations, exact reasoning quotes, trace/prediction hashes and
both reader files. The tracked receipt uses task-identity hashes and event
positions, so benchmark source and solution excerpts stay out of reusable code
and public reports. This supplements, rather than replaces, the full-cohort
[exposure audit](2026-10-22-failure-mechanisms.md).

## Milestone definitions

- **Source:** first numbered returned observation covering a removed/replaced
  reference old line, before any editor attempt or flagged possible shell mutation.
  This is a positional exposure proxy, not proof of diagnosis. Unnumbered reads,
  pure additions and uncertain source positions remain unknown.
- **Hypothesis A/B:** each reader's earliest explicit actionable local correction
  supported by the observed implementation. A suspicion, neighboring improvement,
  or a promise to fix something does not suffice. A local correct hypothesis need
  not be a complete repair; later drift is possible. The label is a semantic
  judgment, not a replayed proof.
- **Attempt / accepted:** first `edit_file` or `write_file` invocation / first
  such invocation returning `status=ok`. These include rejected scratch writes;
  the private timeline lists paths and statuses. Shell mutations are flagged
  separately and not silently credited as accepted editor calls.
- **Command write A/B:** an independently noticed and manually checked source
  write through `run_command`; preserve this separately from editor-tool success.
- **Check A/B:** first runtime verification attempt after an accepted editor call
  or a reader-confirmed source command write.
  An import error is an attempt, not a passing test. A diff/status inspection or
  `submit_patch` is not a behavioral check. No established source mutation means this milestone remains unknown, even if
  there were pre-edit checks.

Numbers below are one-based tool-event ordinals, with trace step IDs retained
in the receipt. Hypothesis text precedes its numbered call; source is returned
by its call. Rejected calls are included, so ordinals may exceed the 28-call
execution budget. `unknown` means not established by this measurement; it does
not establish that the event never happened. Outcomes are released valid-control
verifier results, not tests rerun here.

| Case | Arm | Resolved | Source | Hyp A / B | Attempt | Accepted editor | Command write A / B | Check A / B |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| A | budgetfit | no | 27 | 28 / 28 | 30 | unknown | unknown / unknown | unknown / unknown |
| A | c2 | yes | 13 | 17 / 17 | 22 | 22 | unknown / unknown | 23 / 23 |
| A | c2-seed1 | no | unknown | unknown / unknown | 31 | unknown | unknown / unknown | unknown / unknown |
| A | c2-recover | yes | 3 | 9 / 10 | 27 | 27 | unknown / unknown | unknown / unknown |
| A | c3 | yes | 10 | 12 / 14 | 11 | 14 | unknown / unknown | 15 / 15 |
| B | budgetfit | no | 10 | 11 / 11 | 20 | 20 | unknown / unknown | unknown / unknown |
| B | c2 | no | unknown | unknown / unknown | 23 | 23 | unknown / unknown | 26 / 26 |
| B | c2-seed1 | yes | 7 | 8 / 8 | 14 | 16 | unknown / unknown | 28 / 28 |
| B | c2-recover | no | unknown | unknown / unknown | 16 | unknown | 22 / 22 | 24 / 24 |
| B | c3 | yes | 7 | 15 / 15 | 17 | 17 | unknown / unknown | 25 / 25 |
| C | budgetfit | yes | 17 | 18 / 18 | 24 | 24 | unknown / unknown | 25 / 25 |
| C | c2 | yes | 7 | 10 / 10 | 21 | 25 | unknown / unknown | 27 / 27 |
| C | c2-seed1 | no | 7 | 12 / 12 | 24 | unknown | unknown / unknown | unknown / unknown |
| C | c2-recover | yes | 5 | 8 / 8 | 15 | 15 | unknown / unknown | 16 / 16 |
| C | c3 | yes | 4 | 7 / 7 | 12 | 12 | unknown / unknown | 13 / 13 |

## Independent readings and disagreements

Reader A is the primary AI analyst. Reader B is a separately tasked AI reader
who received the selected identities and criteria without reader A's labels or
draft. Both inspected the original release and reference behavior. Exact quote
membership and trace identities are checked mechanically; the checker cannot
certify semantic truth or the earliest possible interpretation. These are two
AI readings, not independent human approval or evidence of inter-rater reliability
in a larger population. Teammate semantic review remains pending.

Two of the 15 attempts have hypothesis-timestamp disagreements; neither
involves agreement being substituted for an unknown label. In Case A recovery,
A accepts the corrective proposal before call 9 and B requires the more explicit
proposal before call 10. In Case A C3, A accepts the new-list proposal before
call 12, while B requires the concrete edit plan before call 14. Both timestamps
remain in the timeline; the discrepancy is about commitment specificity, not
an estimated rate of execution failures. Verification timestamps agree after
the extraction corrections below.

Reader B exposed two measurement problems in the initial A pass. A missed
`python3 script.py` in Case A C2 because the initial command shortlist emphasized
`-c` checks; its first post-edit check is call 23 and fails assertions. The later
revised reproduction at call 28 passes, which does not make the earlier check
successful. In Case B recovery, accepted editor calls remain unknown, but a
successful command writes source at call 22, a second command changes behavior
at 23, and a local annotation check runs at 24. This does not verify the issue's
causal dispatch behavior. Both initial reader files are retained with hashes;
normalization/corrections are explicit in the final private files. Shell writes
are separately human-readable evidence, not automatically inferred success.

A second limitation concerns hypothesis quality: Case B C3's actionable direction
coexists with false intermediate claims about Python inspection. We accept the
local correction timestamp, not every statement in that reasoning. Case C C2's
first accepted edit also removes unrelated fields before a later correction;
acceptance is not patch correctness. C3 keeps exploring even after a successful
edit/check in Case C, so repeated reading alone is not a failure discriminator.

## Three matched examples

**Case A: late correct diagnosis and a competing wrong hypothesis.** Budget-fit
first exposes the changed old line at call 27 and states the local correction
at call 28; its only editor attempt is call 30, rejected. C2 exposes the line at
13 and lands an accepted edit at 22, with a released resolved outcome. C2 seed1
has no established positional exposure or accepted edit and pursues a different
explanation. This pair does not support treating all failures as execution-only.
The successful recovery variant also waits until call 27 for its accepted edit:
late editing is a risk signal, not a sufficient explanation of failure.

**Case B: an accepted edit can miss the causal operation.** C2 accepts edits at
calls 23–25, but its patch targets a neighboring part of the implementation and
its released outcome is unresolved. Its first source-supported corrective
hypothesis for the reference causal operation remains unknown under reader A's
criterion. C2 seed1 sees that operation at call 7, proposes a correction before
call 8, recovers from a rejected first edit at 14, and accepts an edit at 16;
its outcome is resolved. Budget-fit also identifies a relevant correction but
finishes unresolved: recognizing a local fix is not proof of completing it.
These comparisons support separating wrong-target work, matching recovery and
incomplete implementation rather than equating file membership with diagnosis.

**Case C: correct intent, unsuccessful edit construction.** C2 and C2 seed1 both
receive the changed line at call 7. Both readers can identify an actionable
correction in each trace, with any timestamp disagreement retained below. C2's
first edit at 21 fails, then calls 25 and 26 succeed; its outcome is resolved.
C2 seed1 waits until call 24, receives an ambiguous-match error, and gets a
no-match error at 25 after supplying context that does not match the observed
source. No accepted editor call or nonempty final patch is recorded. Recovery
and C3 accept edits at 15 and 12 and resolve. This is concrete motivation to test
edit construction; it does not prove a prompt intervention will be followed.

## One next experiment: observed contiguous edit context

Register [c2-observed-edit-context-v1](../../configs/experiments/observed-edit-context.json).
Replace only C2's first edit-construction paragraph: construct `old_string` from
one contiguous returned source block, preserving order, whitespace and intervening
lines; add adjacent observed lines when the target is duplicated. This targets
the specific ambiguous/no-match behavior in Case C. The control already asks for
exact unique matches; the intervention makes the construction method explicit.
No claim is made that repeating an instruction will fix compliance.

This becomes the single proposed next prompt comparison; the earlier diagnosis
checkpoint remains a historical packaged candidate and is deferred, not an
additional arm in this experiment. Retained-reasoning ablation #28 remains deferred.

Use both frozen C2 variants with their original member identities and explicit
server-seed contract (42 / 1), the full frozen 29-task registration, identical
valid-control intersections, 28 calls, eight minutes, output/thinking settings
and no adapters. Report candidate-only and control-only resolutions by task and
seed. Also measure matching errors per attempted `edit_file` and accepted source
edit coverage. Selected historical cases are not a promotion set.

Mechanistic support requires fewer matching errors in both seed comparisons;
no net resolved-task loss in either seed is required before proposing a further
run. Any gain remains exploratory until independently repeated. Passing matching
checks alone cannot promote a submission. Freeze hashes, obtain teammate review,
and run the official compiler and Kaggle 4×L4 suite preflight (#26) before manual
inference or spending a submission slot. Stop on a preflight exception. Do not
change budgets or add another rule mid-comparison.

Both seed packages were built on CPU with only `prompts/system.md` changed.
The receipt records member and archive identities. No GPU inference, official
verifier, real tokenizer or competition submission was run for this comparison.

## Reproduction and private review handoff

```bash
python -m scripts.compare_matched_traces \
  --release /path/to/gemma-agent-lab-kaggle-l4-r5 \
  --tasks /path/to/tasks.jsonl \
  --reader runs/20261023-matched-review/reader-a.json \
  --reader runs/20261023-matched-review/reader-b.json \
  --output runs/20261023-matched-review/new-comparison
```

The comparison refuses overwrites and validates source-map identity, reviewer
trace hashes, exact quotes and verification ordering. Reader files have
`reader` and `readings` keys; each `arm/task` entry includes `trace_sha256`,
`hypothesis`, optional `source_write`, and `verification`. A milestone is null or contains `step_id`,
`quote`, and `rationale`. Hypothesis quotes come from `message`; verification
quotes come from JSON-serialized decoded observations; source-write quotes come
from command arguments and require a successful recorded exit status. Those
statuses do not by themselves certify the command changed the intended source. The checked example
files remain in ignored local runs and need private transfer to another machine;
they have not been uploaded to a shared release. A teammate can inspect each
quote in the frozen source release and supply a third reader file. Differences
remain visible rather than being replaced with a consensus label.

## CPU validation receipt

The 85 unit tests and repository checks pass. The comparison validates all 15
reader pairs and exact quotes against frozen traces. Both candidate packages
change one prompt member only. Input/code hashes, milestone positions and package
receipts are in [the tracked receipt](../../resource/manifests/matched-traces-20261023.json).
No new repair outcomes are claimed.
