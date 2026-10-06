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
| Python and shell syntax | Team code under `scripts/`, `src/`, and `tests/`; scripts are parsed rather than launched |
| Configuration | Team YAML/JSON syntax and tracked local `!include` / `config_path` targets |
| Agent packaging | Build each versioned agent directory twice and compare ZIP bytes; verify archive structure/CRC |
| Documentation links | Local file/directory targets in maintained Markdown; ignored local artifacts cannot satisfy a link |
| Preparation tests | Existing notebook generation, task-label filtering, frozen-input checks, and output preservation |

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

## Intended protection for main

The repository owner should enable these settings once the account's private-repository entitlement permits them:

- Require a pull request with **one approving review**.
- Dismiss stale approvals when new commits change the reviewed code.
- Require **Repository checks** to pass and the branch to be up to date.
- Require review conversations to be resolved.
- Apply the requirements to administrators as well.
- Block force pushes and deletion of `main`.

The workflow check must have run successfully before selecting it as required. Repository settings
should allow squash merging and automatically delete merged feature branches. There is no automatic
merge, release deployment, or competition upload in this workflow.

The intended REST configuration is versioned in
[.github/main-protection.json](../.github/main-protection.json). It binds the required check to
the GitHub Actions app (`15368`), rather than accepting an identically named check from any app.
Once entitlement is available and the workflow has passed, the owner can apply and verify it:

```bash
gh api --method PUT repos/jwj1342/gemma-agent-lab/branches/main/protection \
  --input .github/main-protection.json
gh api repos/jwj1342/gemma-agent-lab/branches/main/protection
```

The configuration file describes the requested rule; its presence alone does not enable protection.

## Enforcement status

Last checked: October 6, 2026. The first [hosted PR run](https://github.com/jwj1342/gemma-agent-lab/actions/runs/37518341796)
passed all repository checks and 13 tests. The workflow is configured for PRs, `main`, and manual runs.
Squash merging and automatic deletion of merged feature branches are enabled; other merge methods
and automatic merging are disabled. The initial installation is owner-authorized and has no independent
reviewer approval; subsequent routine changes follow the one-review policy above.

GitHub's protection/ruleset API returned HTTP 403 with an instruction to upgrade to GitHub Pro or
make the repository public. This repository stays private. Until an entitlement upgrade allows the
protection rule, PRs, passing CI, and one-person review are **team policy**, not a GitHub-enforced
merge barrier. A collaborator with write access can technically bypass the convention.

Official references: [protected branch availability](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches),
[workflow configuration](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax),
and [Actions allowances](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
