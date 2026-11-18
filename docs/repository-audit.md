# Repository collaboration audit — October 6, 2026

An independent AI reviewer inspected revision `1e09d7b` with no conversation history. The review
used a new CS teammate's perspective, read the project instructions and guides, and reproduced
failure cases in temporary fixtures. This is a tooling review, not a teammate's GitHub approval.

## Findings and changes

| Finding | Reproduction | Change |
| --- | --- | --- |
| Downloaded Kaggle runs could not use checked submission | Generated setup plus smoke-result fixtures failed because the input ZIP/manifest were missing | Notebook setup now exports both alongside the frozen task and agent; a CPU test covers the complete dry-run handoff |
| Packaged agents could reference files outside their ZIP | A tracked `../shared.md` include passed CI but was excluded from the bundle | References must stay within the owning agent directory; nested `!include` and `config_path` rules are checked separately |
| Installation could use unverified wheels | Adding an extra wheel passed setup and entered the mocked pip command | Exact wheel inventory/hash checks run before installation; pip arguments come from the recorded inventory |
| Frozen task changes were accepted | Editing a prepared problem statement still passed submission checks | A shared validator checks task identity/hash and agent/archive member hashes before execution and submission; results must match the prepared task |
| Force-added excluded files bypassed CI | A dummy ignored `access_token` passed the original check | CI rejects tracked files matching ignore rules; agent Python files also receive syntax checks |
| Frozen-run notebook provenance was ambiguous | Old run revision was retained while current scripts were embedded | Preparation and source-run revisions are separate; exact embedded script hashes are recorded |

The four workflow defects were rated P2 by the reviewer. The artifact guard and provenance record
were additional improvements. Findings concern tooling guarantees, not evidence that an earlier
score or measured smoke result was corrupted.

## Validation and scope

The local CPU suite passes 21 tests, including fresh-source preparation without local data/models,
label removal, configuration boundaries, wheel additions, frozen-input changes, and submission
handoff. The existing H100 evidence passes the strengthened submission dry-run. The baseline ZIP
remains `02bb939095ded0c970088fc9d19634e203e2d3119a36a5feff20bad55c9343fc`.
All 37 artifact hashes listed in the historical H100 and L4 manifests were rechecked and match.
The [hosted audit CI run](https://github.com/jwj1342/gemma-agent-lab/actions/runs/37521076367)
also passed. The same independent reviewer then verified all six fixes, reproduced the Kaggle
submission handoff and wheel allowlist behavior, and reported no new actionable defects.

No training, model inference, Kaggle notebook launch, competition upload, or dependency installation
was performed for this audit. New guards and output metadata have CPU coverage; the changed generated
notebook has not received a fresh end-to-end L4 run. The October 5 runtime measurements remain
historical evidence for their original notebook and scripts.

## Remaining collaboration limits

- **Main protection:** at audit time, the private repository's GitHub entitlement rejected branch
  protection with HTTP 403. Resolved on October 12, 2026: the repository became public and the
  [documented rule](ci-and-review.md#protection-for-main) is now enforced on `main`.
- **Artifacts and access:** a fresh clone contains code and receipts, not raw data, weights, private
  run outputs, or another member's credentials. Use the [Kaggle walkthrough](kaggle-runtime.md)
  for independent execution and record shared artifact locations in experiment reports.
- **Evidence:** hashes detect accidental drift; they are not signatures or a complete record of
  every runtime dependency. Single-task checks do not establish benchmark or training performance.
- **Exclusions:** CI applies filename rules rather than scanning all contents for secrets. Review
  staged diffs. Slides remain excluded from the current tree; earlier commits retain their history.
- **External code and isolation:** preserve third-party attribution and review upstream material
  before execution. Kaggle platform inputs remain mounted; the notebook is not a strong isolation
  boundary for untrusted agent code.

Use the [one-review workflow](ci-and-review.md) for normal contributions. See the
[submission guide](submission.md) for the manually selected competition-upload step.
