# Project collaboration guide

- This project enters `gemma-4-developer-agent`; the paper track is a separate competition.
- On Trillium, read `/home/jwj/.codex/cluster-guides/trillium.md` before cluster operations.
  Other systems do not depend on that local path. Start with `README.md`,
  `docs/competition-research.md`, and `docs/training-plan.md`.
- Use `docs/repository-map.md` to locate files and `docs/kaggle-runtime.md` for the laptop-to-Kaggle path.
  Keep the shared agent/evaluation logic independent of platform-specific installation and launch scripts.
  Notebook preparation from `--agent-dir` must work without local data, model weights, or Kaggle credentials.
- Write all team-authored documentation, comments, templates, and generated navigation in English.
  Preserve original third-party sources, quoted metadata, and provenance; do not rewrite upstream notebooks.
- Track code, configuration, and experiment reports in Git. Exclude data, weights, caches,
  checkpoints, credentials, and run outputs according to `.gitignore`. Put adaptations in `src/` or `configs/`.
- Use `runs/<date>-<short-name>/` for each experiment. Record Git revision, modules,
  dependencies, data/model versions, seeds, resources, metrics, and the complete recovery path.
- On Trillium, use Slurm allocations for evaluation, training, and repository tests;
  use datamovers for large transfers and bulk extraction. Keep login-node work lightweight.
- Treat resources as dated snapshots. Prioritize current rules and explicit organizer
  clarification, then official package behavior, then public examples. Distinguish author
  claims, static source analysis, and independently executed results.
- Official evaluation currently requires `gemma-4-31b-it-qat-w4a16-ct`. Other Models-page entries do not imply approval.
- Keep evaluation and training environments separate. Do not directly execute installation
  or deletion cells from third-party notebooks on the host.
- Keep Kaggle credentials in HOME, outside the workspace. Training jobs should not carry them.
- Members must join the formal Kaggle team. Grant sharing permissions as needed; never open all of SCRATCH.
- The private repository is `jwj1342/gemma-agent-lab`; onboarding is in `CONTRIBUTING.md`.
  Local data, models, HOME environments, and historical run paths are not included in a clone.
- Check Kaggle quota with `scripts/check_kaggle_quota.py` or the CLI. kagglesdk 0.1.37
  `to_dict()` loses whole days in durations. The early six-hour figure was incorrect;
  the raw API confirmed 108000 seconds (30 hours). Use `total_seconds()` and recheck usage/reservations before running.
- Before evaluation, freeze the agent, public task, and ZIP hashes with `prepare_eval_run.py`.
  The runner reads that snapshot. Require valid empty/gold controls before inference;
  report execution success separately from issue resolution.
- Presentation sources and outputs remain local in `slides/` and in the separate Overleaf project.
  Keep the entire `slides/` directory out of this repository; `slides/main.tex` is the local entry point.
