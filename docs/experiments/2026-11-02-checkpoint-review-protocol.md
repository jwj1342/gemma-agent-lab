# Diagnosis checkpoint: protocol and candidate freeze

No inference, training, GPU allocation or submission is launched by this protocol.
The November 3 owner decisions and November 5 package freeze are recorded in
[the freeze handoff](2026-11-05-observation-checkpoint-freeze.md).
This supersedes the older two-seed evaluation suggestion for the checkpoint;
existing JSON registrations remain historical CPU packaging receipts.

## One candidate and hypothesis

Compare frozen C2 against C2 with only its immediate-edit paragraph replaced:

> You have just read code that you believe conflicts with the issue. State the
> conflicting operation and smallest justified correction, then edit next. If
> uncertain between two specific hypotheses, run at most one command whose
> output would differ between them (the issue reproduction or a targeted test),
> then decide from that observation. Do not repeat an unchanged check.

This is an untested hypothesis about reducing delayed action, not an improvement
claim. There is no second candidate or automatic promotion of the earlier
observed-edit-context candidate.

## Freeze and controls

Code baseline for this draft: `aa76bb0` (full revision in Git history). Protocol
text: the commit updating this document. The owner must approve both revisions before preparing
an immutable run manifest; any implementation of this paragraph needs a new
explicitly recorded full commit before inference. This draft does not claim that
an executable refined candidate has already been frozen.

Data identities: the committed receipts at `a0263801634174a3eb803767d19260818632ba00`,
`resource/manifests/failure-mechanisms-20261022.json` and
`resource/manifests/matched-traces-20261023.json`, identify historical evidence.
For the proposed H100 evaluation, freeze the owner's complete 29-task development
inventory, task/reference hashes, frozen C2 agent members, runtime, model,
tokenizer and sampling configuration in the run manifest before execution.
Those H100 inputs are not available in this checkout: their final hashes require
owner confirmation and must not be substituted with the 15 selected trace cases.
Keep references offline for scoring only; never expose them to the agent.

Use four paired seeds per arm: 42, 1, 2, 3. Keep 28 calls, eight-minute base budget,
4096 thinking tokens, no adapters and identical recorded H100 scaling in both
arms. Require empty/gold controls; use the task/seed intersection valid in both
arms. Report exclusions and missing artifacts by arm. Four seeds are a bounded
pilot, not a claim of adequate statistical power; tasks and seeds are correlated.

## Prespecified endpoints and decision

Primary mechanism endpoint, no-patch-after-exposure: the proportion of attempts
with positional exposure of at least one removed/replaced reference old line
that finish with an explicitly empty `agent_patch` string. The denominator is
all paired valid-control task/seed attempts in that arm with established old-line
exposure and a known, completely parsed final patch. Include successes and
failures. Pure-addition references cannot establish old-line exposure.

Missing traces, missing/invalid patches, unsupported diff sections and absent
reference evidence are unknown, never empty. Exclude unknowns from this rate,
report their counts and reasons separately, and report a sensitivity comparison
restricted to pairs eligible in both arms. Zero denominators, a zero control
rate, or any unknown endpoint among otherwise valid pairs make promotion
inconclusive rather than a pass.

Target: at least **30% relative reduction**, `(control_rate - candidate_rate) /
control_rate >= 0.30`, not 30 percentage points. Report rates, numerators and
denominators per seed and pooled; also report the eligible-pair sensitivity.

Pooled resolution non-inferiority margin: at most **one fewer resolved task per
29-task seed on average**, i.e. **four net lost resolved attempts out of 116**
with four seeds and complete controls (3.45 percentage points). On a smaller
paired valid-control intersection of N attempts, use the same rate margin,
`delta >= -N/29` (integer losses at most floor(N/29)); do not silently keep a
four-task allowance after exclusions. Delta is candidate minus control resolved
attempts. This is a chosen practical tolerance, not a margin inferred from the
reported between-seed sd of about 1.6/29. The owner must approve it before launch.

For an inferential non-inferiority claim, require the one-sided 95% lower bound
for the pooled paired resolution-rate difference to exceed `-1/29`. Prespecify
10,000 paired bootstrap resamples of task IDs (all paired seeds of a task move
together), analysis RNG seed 20261102, percentile lower bound. This exploratory
interval captures task clustering but four seeds cannot characterize seed
variation reliably. Report the interval and its assumptions; failure to clear
it is inconclusive evidence, not proof of harm. The empirical margin alone must
not be described as statistically established non-inferiority. Per-seed outcomes
and paired gains/losses are reporting only, with no per-seed zero-loss gate.

Success requires the 30% relative mechanism target in both pooled and eligible-
pair analyses, the pooled empirical resolution margin and interval criterion,
complete endpoints, and no new compilation/runtime failure. Failure of the
mechanism target or an empirical resolution loss beyond the margin fails the
promotion gate. Missing inputs/controls/endpoints or an interval crossing the
margin are inconclusive. Do not select a favorable seed or change the margin
after seeing results. Four seeds remain a pilot with uncertain power.

Secondary metrics, fixed before running:

- Call index of first accepted `edit_file` or `write_file` (`status=ok`), plus
  no-accepted-editor count; separately report confirmed shell source writes.
- Repeated unchanged checks: repeated exact command text with identical returned
  stdout/stderr/status and no intervening accepted editor or confirmed shell
  source write. Report unknowns when observations or mutation attribution are
  incomplete; this is a conservative proxy, not semantic equivalence.
- Edits outside every reference file: a known, nonempty final patch whose complete
  affected-path inventory has no reference-file overlap. Unsupported diffs are
  unknown, never counted as outside-reference edits.

## Owner decisions and remaining preflight

[Owner confirmation](https://github.com/jwj1342/gemma-agent-lab/pull/29#issuecomment-6597549165)
on November 3 settles C2 as sole control, budget-fit as historical reference,
seeds 42/1/2/3, one observation-grounded paragraph and the pooled margin design.
H100 scaling is 0.5 (eight base minutes become four); server seed 42,
TP 1, memory utilization 0.8, 32K context, concurrency 1, 28 calls and 240-second
command timeout are fixed in both arms. The exact bootstrap procedure above is
prespecified, with no later outcome-based adjustment.

The November 5 executable packages supersede the draft's missing-candidate gate.
The earlier code-baseline paragraphs describe the draft's historical state.
Full private runtime/model/tokenizer/task/reference hashes, semantic second review,
official compilation and runtime preflight remain required in the run manifest.
Abbreviated owner hashes are provenance hints, not verified SHA-256 identities.
The owner's 29-task mask is the analysis ceiling; run the full frozen development
quarters and retain newly invalid controls/exclusions rather than selecting cases.

The historical `audit_failure_mechanisms.py` report covers failures only. It must
not supply this protocol's primary denominator, which includes successful attempts.
The runner must export per-attempt endpoints for both arms, all four seeds,
unknown reasons and paired valid-control masks, and implement the prespecified
paired analysis before interpreting results. No H100 metric implementation or
bootstrap execution is claimed by the CPU package freeze.

XML trajectory conversion and GPU execution remain separate work. Real-template
checks validate only tested native formats; they do not establish training readiness.
