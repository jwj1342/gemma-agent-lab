# Participation and collaboration

1. Open the [prediction competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)
   with your own Kaggle account, accept its rules, and complete required verification.
   The October 4 account snapshot confirms `jwj1342` joined; each teammate completes their own enrollment.
2. Join the formal Kaggle team through its Team page. The snapshot allows five members and sets
   the merger deadline to November 25, 2026. Follow team-sharing rules when exchanging code or data.
3. Read the [overview](overview.md), [data guide](data.md), and official rule snapshots.
   Keep your own token in `~/.kaggle/access_token` with mode 0600; do not share it with teammates.
4. Put team implementations in `src/`, configuration in `configs/`, and unchanged references in `resource/`.
   Record versions, splits, model revisions, dependencies, budgets, seeds, and results for every experiment.
5. On Trillium, read the cluster guide first. Edit/submit on login nodes, use datamovers for large
   transfers/extraction, and use Slurm for inference/training. Compute nodes read HOME/PROJECT and write SCRATCH.

## Kaggle development resources

Copy & Edit the [official starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent).
Check Inputs for the competition data, official wheelhouse, and required model version.
Choose L4×4 under Session options with Internet disabled. CPU-only ZIP packaging does not need a GPU;
start L4 sessions for model execution and verification.

The October 5 raw API check confirmed a 30-hour GPU quota for the maintainer's account, equivalent to
about 15 hours of L4×4 wall time at the current 2× rate. An early six-hour report was a duration-serialization
bug that dropped a whole day; it did not describe a quota increase. Before running, use `kaggle quota`
or the raw API to check actual usage, reservations, and remaining allowance. Treat the scoring
12-hour limit and personal notebook quota as separate constraints.

## Prediction and paper tracks

The snapshot allows one prediction submission per day. Check artifacts and evidence before uploading,
and coordinate submissions across the team. Two final submissions may be selected; verify the final
selection page instead of assuming the latest upload is automatically selected.

The optional [paper track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
requires separate enrollment and closes November 12, 2026. It does not replace prediction-track submission.
Dates, prizes, and licensing obligations are preserved in `resource/competition/timeline.md`,
`prizes.md`, `rules.md`, and `foundational-rules.md`. Each member accepts their own legal terms.

## GitHub collaboration

The private repository is [jwj1342/gemma-agent-lab](https://github.com/jwj1342/gemma-agent-lab),
with default branch `main`. Use your own GitHub account and accept the owner's invitation.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for cloning, lightweight checks, and the PR workflow.

Keep data, weights, tokens, caches, full logs, and checkpoints outside Git. Include conclusions and
reproducible configuration in PRs. GitHub preserves pushed source and small references; ignored SCRATCH
artifacts still need separate backups. Trillium ACLs, GitHub access, and Kaggle team membership are independent.
