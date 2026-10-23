# Evidence review: from source exposure to a verified repair

## Decision and scope

Keep the scored budget-fit package as the release comparator. Prepare one
diagnosis-checkpoint experiment, but do not promote it or allocate GPU time from
source-overlap counts alone. Before expanding trajectory training, preserve
explicit reasoning and verify the actual tokenizer/protocol boundary.

This review builds on the team's useful work: frozen runs, independent controls,
L4 batches, trace releases, and negative-result reporting make the next question
testable. The remaining question is specific: **what prevents available evidence
from becoming a correct, applied and verified change?** More search, more editing
instructions and more training examples are different hypotheses about that gap.

Related work: [#7](https://github.com/jwj1342/gemma-agent-lab/issues/7),
[#12](https://github.com/jwj1342/gemma-agent-lab/issues/12),
[#14](https://github.com/jwj1342/gemma-agent-lab/issues/14),
[#27](https://github.com/jwj1342/gemma-agent-lab/issues/27).
Review baseline: main `e2837a847a89` on October 22, 2026. The downloadable
artifacts below were inspected on CPU; issue-verification tests were not rerun.

## What the results support

The r5 release reproduces the following valid-control L4 outcomes. Each arm has
29 registered attempts and the same 21 valid-control tasks; exclusions remain in
the coverage report rather than disappearing from the record.

| Arm | Resolved | Failed | Reference-hunk exposed failures | Exposed changed-old-line failures |
| --- | ---: | ---: | ---: | ---: |
| budget-fit | 6/21 | 15 | 13 | 9 |
| C2 | 11/21 | 10 | 6 | 4 |
| C2, seed 1 | 9/21 | 12 | 9 | 5 |
| C2 recovery adaptation | 9/21 | 12 | 9 | 6 |
| C3 | 11/21 | 10 | 8 | 6 |

Across these arms, 45 of 59 failures overlap a reference hunk, but only 30 have
observed positional coverage of a removed/replaced old line. Forty-nine failures
have any changed old lines at all; pure-addition references have no such line.
These are **59 attempts over 16 tasks**, not 59 independent bugs. Old-line
coverage is a stronger exposure signal than context-hunk overlap, but remains
positional evidence, not a semantic diagnosis or a complete source-read audit.
Unnumbered shell reads and changed intermediate source positions require review.

The separate H100 r4 release reproduces 12/14/12 resolutions on 29 valid tasks
per run: 49 failures over 22 tasks, 39 hunk-exposed failures and 32 with exposed
changed old lines. Do not pool H100 timing or task masks with r5's L4 results.

The local C2 gain has not established leaderboard improvement. The reported
public observations are budget-fit 0.18, C2 0.17 and C3 0.13. Neither distribution
shift nor stochastic variation has been isolated as the explanation. C2 and C3
are different packages, not repeated draws of the same treatment.
[Submission updates](https://github.com/jwj1342/gemma-agent-lab/issues/7).

The latest training report also changes the next-step decision: budget-fit's
12/14/12 dev counts compare with 4 for actions-only training, 13 with thoughts
retained, 12 with one-line synthetic data and 10 for the 190-trace mixture.
These are author-reported H100/L4-emulated results, not independently reproduced
L4 measurements or multiple seeds of one adapter. Their detailed local run files
were unavailable in main and the listed data releases. They do not demonstrate a
training gain, but they do justify auditing the learning interface before scaling.
[Latest report](https://github.com/jwj1342/gemma-agent-lab/issues/27#issuecomment-6331909020).

## Separate mechanism judgments from observations

The new CPU auditor records returned ranges, reference-hunk overlap, changed-old-line
coverage, attempted/accepted editor calls, final diff paths, missing artifacts and
reported termination. It never executes trajectory commands. A filename match,
an attempted edit and a final patch are different observations. Deleted source
paths and shell-produced final patches are counted; missing predictions remain unknown.

Mechanism labels require a private, identified review with an exact trace hash,
prediction hash, step IDs, verbatim excerpts and a rationale:

- `no_decisive_source_exposure`: review found the needed source was not received.
- `diagnosis_unresolved`: relevant source was available, but the causal diagnosis
  remained unsupported or unsettled.
- `recognized_not_applied`: a source-supported change was stated but not applied.
  This requires both reasoning and source-observation excerpts and an explicit
  source-support judgment.
- `wrong_or_incomplete_patch`: an observed nonempty patch is wrong or incomplete.
- `verification_or_completion`: review identifies a verification/completion problem.
- `insufficient_evidence`: missing, ambiguous or not yet reviewed evidence.

Without annotations, all semantic labels remain `insufficient_evidence`.
Timeout and budget rejection are separate flags; they are not root-cause labels.
There are three initial private case interpretations exercising the annotation
path: unresolved diagnosis, recognized-but-unapplied change and partial
implementation. These are selected examples, not prevalence estimates, and have
not received independent teammate review. No full-cohort semantic percentages or
dominant-mechanism claim is made here.

The review still needs a stratified second pass: include absent exposure,
source-exposed empty patches, nonempty wrong patches, multi-file work and missing
evidence. Record disagreement rather than forcing a diagnosis. Freeze the original
historical difficulty manifest before reporting hard/easy strata; the previous
audit inferred those bands from published aggregates and outcomes.

## Experiment A: one conditional diagnosis checkpoint

[The specification](../../configs/experiments/diagnosis-checkpoint.json) pins both
released C2 member inventories. The builder accepts only exact controls, rejects
extra members and changed bytes, and replaces one generic paragraph. It changes
only `prompts/system.md`; model, tools, limits, sampling and adapters remain identical
to the matching control. It creates an isolated agent, deterministic ZIP and hash
receipt without private task text, traces or reference solutions in the package.

The new paragraph asks the agent to compare implementation with expected behavior,
identify the conflicting operation and minimal justified change, then edit. If
uncertain, it permits one check discriminating two hypotheses. This targets the
condition the existing early-edit rule assumes has already been satisfied.
It may still be ignored or induce a premature wrong edit. It is a hypothesis.

Both frozen seed variants were packaged on CPU. The candidate prompt SHA256 is
`dc162f5264c27f64a59480cb71613ff32ca05290a952467aa37662a34d33a5fe`.
The archive digests produced here are:

- C2 control variant: `4c2c26e0e569b0e0118a0010af703b77670304d92f3d8e03ca061b297b24b18f`.
- C2 seed-1 control variant: `0d2c3763f78e9025ff1aa78ce1c07aa70fd841738e2ac35d43b5b9cddefdfa3b`.

ZIP reproducibility is checked within the same environment; member hashes are the
portable comparison across compression-library versions. Official compilation,
ADK/vLLM integration and repair evaluation remain pending.

Manual evaluation registration:

1. Finish/review the mechanism labels. Start this arm only if they support a
   semantic decision bottleneck. If application failures dominate, propose a
   separately reviewed application/recovery arm instead; do not silently substitute it.
2. Freeze runtime/wheel hashes, task/base identities and controls using the existing
   preparation pipeline. Use the full 29-task r5 dev cohort and recheck controls;
   report the historically valid 21-task mask and any new mismatch explicitly.
3. Compare matching C2/control and checkpoint/candidate at seeds 42 and 1 on four
   L4s, interleaving methods. C2's default sampling file omits a seed, so explicitly
   freeze server seed 42; the seed-1 package explicitly sets 1. Do not change only
   one arm's seed or compare a new runtime with an old cached control result.
4. Report verified paired gains/losses, hard and easier strata, empty final patches,
   accepted edits, call count, clean completion, tokens and full-run elapsed time.
   Time to first edit is a diagnostic, not the success criterion.
5. Do not promote without repeatable paired repair improvement and explained
   regressions. Before a submission, compare the candidate with frozen budget-fit
   and check setup/startup plus all-task inference against the 12-hour allowance.
   Use the sealed holdout at a preregistered milestone, not during prompt iteration.

This does not revive #28's deferred history-removal experiment, change graph
policy, shorten reasoning or raise the tool cap. Those would be separate treatments.

## Experiment B: repair the supervision contract before more training

Main's converter deliberately drops assistant prose for tool-call messages and
previously discarded explicit reasoning too. `--preserve-reasoning` now copies
plain-text `reasoning` or `reasoning_content` into the assistant `reasoning` field
for retained tool actions and the real terminal answer. Conflicting/non-text
formats are rejected. Missing thoughts are not invented, and ordinary `content`
is not reinterpreted as reasoning. Intermediate thought-only messages are still
outside this action-level converter's supported format.

The default remains historical actions-only conversion. Export manifests and
per-trajectory metadata record the selected policy and number of assistant
messages with reasoning. Native tool arguments, actual observations, call IDs and
submission/terminal behavior remain unchanged. This is an input-contract fix;
it does not reproduce the unpublished converter used in the latest training runs.

Before training, obtain that run's exact converter/trainer revision and dataset
hashes. On the pinned real tokenizer, inspect serialized examples with and without
reasoning; confirm thinking-channel rendering, assistant-only loss, native call
boundaries, final stop tokens and deployed prompt compatibility. Record rejected
rows by length and task, and probe longest target as well as longest input for
memory. The added CPU tokenizer fixture verifies our masking contract, not the
real Gemma template. **Real-template validation is still pending.**

Then register a small task-balanced comparison: thought-preserving natural
successes versus natural plus source-grounded recovery examples, with the same
student prompt/tools, fixed training-token budget, seed and adapter recipe.
Disclose coverage differences. Preserve teacher/hint provenance; no explicit hint
mention is not proof of grounding. Exclude dev/holdout snapshots and related
mutations. Evaluate unhinted non-training tasks on L4 with matching no-adapter
controls. Do not expand rank/data sweeps or stronger-teacher collection without a
repeatable transfer result. Lower NLL and adapter loading are intermediate checks.

## Reproduce the CPU work

Obtain the existing private data releases rather than committing their contents.
The release SHA256 identities and aggregate CPU receipts are indexed in
[the evidence manifest](../../resource/manifests/failure-mechanisms-20261022.json).
All trace-derived outputs belong under ignored `runs/`.

```bash
python -m scripts.audit_failure_mechanisms \
  --release /path/to/gemma-agent-lab-kaggle-l4-r5 \
  --runs budgetfit c2 c2-seed1 c2-recover c3 \
  --tasks /path/to/competition/tasks.jsonl \
  --output runs/20261022-failure-mechanisms/new-r5

# Optional: --annotations /path/to/private-review-annotations.json
# JSON object keyed by run/task; values contain category, reviewer, rationale,
# trace_sha256, prediction_sha256, evidence [{step_id, field: message|observation, quote}], and
# source_supported: true for recognized_not_applied. Keep this file private.

python -m scripts.build_diagnosis_candidate \
  --source /path/to/gemma-agent-lab-kaggle-l4-r5/c2/agent \
  --output runs/20261022-diagnosis-checkpoint/new-seed42
# Repeat with c2-seed1/agent and a new output for seed 1.

python -m scripts.prepare_sft_data \
  --parquet /path/to/pinned-shard.parquet \
  --tasks /path/to/competition/tasks.jsonl \
  --preserve-reasoning --output runs/20261022-sft-reasoning/new-dataset

python scripts/check_repository.py
python -m unittest discover -s tests -v
git diff --check
```

No model inference, GPU job, training, competition submission or task verifier
was launched by this work. A merge records tested tooling and a registered
experiment; it does not change the team's selected submission.

## CPU validation

All 80 repository unit tests passed locally with Python 3.14 and the pinned
PyYAML 6.0.3 development dependency. Repository checks passed for 54 Python,
14 shell, 20 configuration and 66 Markdown files, plus three static agent
packages. The staged diff passes whitespace checks. The private r4/r5 runs and
both candidate packages above were generated successfully. Hosted CI uses
Python 3.12 and is a separate integration check.

## Requested review

@jwj1342: please review the distinction between hunk exposure, changed-old-line
coverage and semantic diagnosis; the exact C2/control identities and seed
contract; and whether explicit-field reasoning preservation matches the intended
training path. The three private sample interpretations need a second reader
before they guide GPU allocation. Please also confirm access to the actual
thought-preserving training revision before Experiment B proceeds.

## October 23 reviewer corrections

The reviewer identified three valid issues: header-like source lines could be
misparsed inside diff hunks; the packaging fixture depended on Windows newline
translation; and missing patch fields were treated as explicit empty diffs.
The shared parser now consumes declared old/new hunk lengths before recognizing
headers, including in reference-map reconstruction. Fixtures use LF bytes.
Missing/non-string patch fields remain unknown with or without a trace.
Regression tests cover syntactically valid Python docstrings via difflib and
both missing-evidence paths. All 82 CPU tests and repository checks pass.
Both private r4/r5 audits were recomputed: every published summary counter is
unchanged. Historical receipts above describe the original run, not the revised
parser. No official verifier, real tokenizer or GPU experiment was rerun.

### Additional prediction-format guard

Only an explicitly empty `agent_patch` string is classified as an empty patch.
Nonempty input without recognized file headers, and input that fails counted-hunk
parsing, now produce unknown final-patch membership. This includes headerless Git
rename/binary changes; support for extracting their paths is not claimed. Regression
cases exercise these inputs with and without traces. This is a conservative input
guard, not a general Git-diff validator. The 82 tests and repository checks pass;
recomputed r4/r5 summary counters remain unchanged. No runtime or GPU validation
was added. Private rerun outputs are in `runs/20261023-patch-evidence/`.
