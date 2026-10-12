# CI and review for a small research team

Our working sequence is **feature branch → pull request → CPU checks → one teammate's approval → squash merge**.
Keep `main` usable for the next person cloning the repository. Experiments can explore freely on their
own branches; the PR explains what changed and what the evidence supports.

## Day-to-day workflow

1. Create a short-lived branch from current `main`, such as `experiment/tool-budget`,
   `fix/notebook-inputs`, or `docs/training-notes`.
2. Keep one coherent question or change in the PR. Link an Issue for research experiments and
   include the configuration, measured outcomes, and accessible artifact locations.
3. Stage the intended files, run the local checks below, then push the branch and open a PR.
4. Wait for the **Repository checks** job to pass. Ask one teammate with write access to review.
   The author does not count as the approving reviewer. Resolve discussion threads before merging.
5. After new code is pushed, ask the reviewer to check the revised version. Merge using **Squash and merge**
   after checks and approval, then update your local `main`. The feature branch can be deleted.

The reviewer checks whether the change is understandable, reproducible, and supported by the reported
evidence. A prompt experiment does not need to improve the leaderboard to be worth recording.
State negative results and limitations clearly. CI verifies tooling and repository structure;
GPU experiments establish model behavior.

## Automated checks

One workflow runs on pull requests targeting `main`, pushes to `main`, and manual dispatch.
Its stable check name is **Repository checks**. It uses one GitHub-hosted Ubuntu CPU runner,
Python 3.12, and a five-minute timeout. A newer update to the same PR cancels the older run.

| Check | Coverage |
| --- | --- |
| Python and shell syntax | Team code under `scripts/`, `src/`, `tests/`, and agent bundles; scripts are parsed rather than launched |
| Configuration | Team YAML/JSON syntax and tracked references confined to each agent bundle, including nested path resolution |
| Agent packaging | Build each versioned agent directory twice and compare ZIP bytes; verify archive structure/CRC |
| Artifact exclusions | Reject tracked files that match ignore rules, including force-added excluded files |
| Documentation links | Local file/directory targets in maintained Markdown; ignored local artifacts cannot satisfy a link |
| Preparation tests | Notebook generation, checked submission handoff, frozen-input integrity, wheel inventory, and output preservation |

The workflow file is `.github/workflows/ci.yml`; `scripts/check_repository.py` performs the static
checks and packaging. The test suite is under `tests/`. CI installs only the small dependency in
`configs/ci.requirements.txt`. It never executes upstream notebooks, installs training dependencies,
loads model weights, starts Slurm/Kaggle jobs, or submits competition entries. The workflow has read-only
repository permissions and uses no Kaggle or cluster secrets.

Upstream snapshots remain unchanged. CI inspects our navigation READMEs within `resource/`, while
excluding original source/page bodies and historical reading-note excerpts from Markdown validation.
The link check covers inline local destinations and reference definitions; it skips code blocks,
external URLs, and heading anchors. It is not a full Markdown renderer or external-link crawler.
YAML checks verify syntax and references; the official competition compiler remains a separate runtime gate.
The ignore-rule gate checks filenames, not credential contents. Review diffs before pushing and keep
credentials out of source files even when their names are allowed. An ignored-file check cannot undo
a credential leak after a commit has already reached GitHub.

## Run the checks locally

From the repository root, in a development environment with Python 3.10+ and Bash:

```bash
# Install into your development virtual environment, not a training/evaluation environment.
python -m pip install -r configs/ci.requirements.txt
git add <specific-files>
python scripts/check_repository.py
python -m unittest discover -s tests -v
git diff --cached --check
```

Stage new files before running the checker: it deliberately checks the Git file list so that a
private local file cannot hide a missing file in a teammate's clone. It reads current working-tree
contents; stage any final corrections before committing. On Trillium, follow the cluster's environment
rules and use an existing suitable tooling environment or install PyYAML through the Alliance wheelhouse.
These small preparation checks fit the login-node lightweight-test limits; model/repository evaluation uses Slurm.

To debug a failed PR, open its **Checks** tab, read the first failed step, reproduce that command
locally, and push the fix to the same branch. A successful CPU run does not replace an L4/H100
experiment when changing inference, dependencies, adapters, or evaluation behavior.

## Protection for main

`main` is protected by GitHub, so the workflow above is enforced rather than voluntary:

- A pull request with **one approving review** is required. The author's own approval does not count.
- New commits dismiss stale approvals, and the most recent push must be approved by someone other than its pusher.
- **Repository checks** from the GitHub Actions app (`15368`) must pass, and the branch must be up to date with `main`.
- Review conversations must be resolved.
- The requirements apply to administrators as well; nobody can merge around them.
- Force pushes to `main` and deletion of `main` are blocked.

Repository settings allow only squash merging and automatically delete merged feature branches.
There is no automatic merge, release deployment, or competition upload in this workflow.

The applied REST configuration is versioned in [.github/main-protection.json](../.github/main-protection.json).
The file does not apply itself: after changing it, the owner must apply it and verify the live rule:

```bash
gh api --method PUT repos/jwj1342/gemma-agent-lab/branches/main/protection \
  --input .github/main-protection.json
gh api repos/jwj1342/gemma-agent-lab/branches/main/protection
```

## Enforcement status

Last checked: October 12, 2026. The configuration above is live on `main`. The first
[hosted PR run](https://github.com/jwj1342/gemma-agent-lab/actions/runs/37518341796) on October 6 passed
all repository checks, so the required check name is established. The repository became public
on October 12, 2026, which made GitHub branch protection available; earlier, the private repository's
entitlement returned HTTP 403 for the protection API.

Before protection was enabled, review was team policy only. The initial CI installation (#1) and audit
fixes (#2) were owner-authorized without independent approval. PR #10 was merged on October 12 without
a submitted review; it is reverted by PR #17, and its branch remains available for a reviewed resubmission.

Official references: [protected branch availability](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches),
[workflow configuration](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax),
and [Actions allowances](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
