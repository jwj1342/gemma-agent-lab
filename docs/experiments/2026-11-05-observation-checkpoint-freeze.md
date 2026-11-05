# Observation-grounded checkpoint: candidate freeze and handoff

## Decision

Freeze the executable candidate for one H100 pilot, not a winning submission.
The [owner confirmed the design on November 3](https://github.com/jwj1342/gemma-agent-lab/pull/29#issuecomment-6597549165)
and reiterated available H100 capacity in [the November 5 team update](https://github.com/jwj1342/gemma-agent-lab/issues/7).
C2 is the sole experimental control. Historical budget-fit is a descriptive
reference; its 0.18 score is team-reported, not independently rerun here.

The [new specification](../../configs/experiments/observation-checkpoint.json)
implements the agreed observation-grounded paragraph. Historical diagnosis and
observed-edit-context specifications remain unchanged. No graph policy, tool,
model, adapter, thinking budget or recovery rule is added to this treatment.
The seed change is identical within each pair.

## CPU packages and evidence

Source C2 came from the locally retained private r5 release; the complete archive
SHA-256 was independently checked against the committed historical receipt:
`8efa27ea037e3df1fb00f2fe5ce1695d25abcc0468ed71502e91552c7e60c808`.
A fresh release download timed out, so no fresh download is claimed.
The builder also verifies every C2 member byte against the historical inventory.

Eight packages were built: control/candidate for labels 42, 1, 2 and 3.
Server seed stays 42; label 42 omits a sampling seed, while 1/2/3 set it explicitly.
The generated seed-1 control matches the pre-existing C2 seed-1 inventory exactly.
Within each pair only `prompts/system.md` differs. The candidate prompt SHA-256 is
`d00405d6ba826c8beeb922046cd047e296984d39c4a5a54e4947cdc88f6fd8d0`.
Member and ZIP identities, the 29-task analysis mask, benchmark hash and runtime
provenance are in the [freeze receipt](../../resource/manifests/observation-checkpoint-20261105.json).
ZIP identities are environment-dependent across compression-library versions;
member hashes are the portable identity. The package test checks deterministic
rebuilds, seed pairing, unchanged sources and rejection of altered controls.

Reproduce from the exact private release control, without credentials or a GPU:

```bash
python -m scripts.build_checkpoint_pilot \
  --source /path/to/gemma-agent-lab-kaggle-l4-r5/c2/agent \
  --output runs/20261105-observation-checkpoint
```

Keep packages and raw traces under ignored `runs/`; the commit contains generic
prompt text, tooling and compact hashes only. Use the full Git commit containing
this report as the candidate code revision; record `git rev-parse HEAD` in the
owner's immutable run manifest. Any later package/spec change needs a new freeze.

## Required run handoff

1. Record full runtime, wheel, model, tokenizer, task/base and reference hashes.
   The owner supplied abbreviated private runtime/quarter digests; these cannot
   be expanded or verified from this checkout. The full benchmark digest does
   match the owner's prefix/suffix. Run the frozen full dev quarters and analyze
   only the paired valid-control intersection with the recorded 29-task mask.
2. Obtain the independent semantic second review and perform official compilation
   plus H100 preflight on both arms. Fix integration failures before spending the
   full pilot; any changed candidate must be refrozen.
3. Export endpoints for successes and failures, unknowns and all exclusions.
   The historical failure-only auditor is insufficient for the denominator.
   Implement/check the paired metric aggregation and task-cluster bootstrap
   against the [prespecified protocol](2026-11-02-checkpoint-review-protocol.md).
4. Run two arms by four seeds with the owner's quarter setup: TP 1, memory 0.8,
   32K context, concurrency 1, server seed 42, 0.5 time scaling (four minutes),
   28 calls and an unchanged 240-second command timeout. Retain every seed.
5. Apply the 30% relative mechanism target and pooled resolution non-inferiority
   gates, including the interval and eligible-pair sensitivity. Missing evidence
   is inconclusive. A pilot pass justifies measured 4xL4 confirmation and total
   runtime accounting before considering a competition slot.

## New evidence and scope boundaries

The November 5 team report says specification-evidence checks produced no real
gaps online, skills can miss local repositories after ADK changes cwd, and the
full-data LoRA submission exceeded the scorer runtime despite a seven-minute cap.
These are owner-reported findings; private runs were not independently repeated.
They support keeping this pilot prompt-only, measuring behavior and runtime, and
requiring actual L4 evidence before submitting heavier candidates. Skill-path
fixes belong to their owning branch, not this one-factor comparison. XML-only
conversion remains [issue #30](https://github.com/jwj1342/gemma-agent-lab/issues/30).

This change runs no inference, verifier, training, cluster job or submission.
CPU checks establish package/tooling integrity, not agent effectiveness.
Main still requires CI and a teammate approval through PR #29 before merging.

## Validation on November 5

With Python 3.14 and pinned PyYAML 6.0.3, repository checks passed (61 Python,
14 shell, 22 configuration, 71 Markdown files and three static agent packages).
The full CPU suite reported 90 tests, OK with one optional real-tokenizer
integration class skipped because no local tokenizer directory was supplied.
This does not rerun or supersede the earlier real-template evidence in PR #29.
Staged whitespace checks passed. Hosted CI for the new commit is separate.
