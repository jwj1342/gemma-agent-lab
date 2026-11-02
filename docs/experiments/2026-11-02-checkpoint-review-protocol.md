# Diagnosis checkpoint: draft protocol for owner confirmation

No inference, training, GPU allocation or submission is authorized by this draft.
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

Code baseline: `d961cba` (full revision in Git history). Protocol text: the commit
introducing this document. The owner must approve both revisions before preparing
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

No loss in resolution means **zero allowed net lost resolved attempts** on the
full paired valid-control intersection, pooled and in each seed. Report paired
gains/losses separately; this observed threshold does not prove non-inferiority.
Success requires the mechanism target in both pooled and eligible-pair analyses,
the resolution condition, complete endpoints, and no new compilation/runtime
failure. Otherwise the candidate fails the promotion gate or is inconclusive
when inputs/controls/endpoints are incomplete. Do not select a favorable seed.
Report first accepted editor call (and no-edit count), repeated unchanged checks
with a frozen definition, and final edits outside all reference files as secondary
metrics; shell mutations require separate attribution. No significance or
population-level benefit claim follows from this pilot.

## Owner questions before registration

Confirm C2 as the sole control, seed list, complete H100 inventory and hashes,
time scaling, and the strict zero-loss / 30%-relative promotion gate. Confirm the
paragraph and independent annotation review before freezing implementation.
XML trajectory conversion, real-tokenizer validation, official compilation and
GPU execution remain separate work or preflight gates, not completed results.
