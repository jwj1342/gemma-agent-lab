# Working together

This is the private team workspace for the **Gemma 4 Developer Agent prediction
competition**. You can contribute from a regular laptop and use Kaggle for GPU
experiments. A Trillium account is optional.

## First visit

1. Ask the repository owner to invite your GitHub account, then accept the invitation.
2. Use your own Kaggle account to accept the competition rules and join the formal
   Kaggle team. GitHub access and Kaggle team membership are separate.
3. Read the [English kickoff slides](slides/gemma4-kickoff/main.pdf), then the
   [competition overview](docs/overview.md), [data guide](docs/data.md), and
   [research notes](docs/competition-research.md). Most detailed notes are currently in Chinese.

```bash
gh repo clone jwj1342/kaggle_gemma4_agent
cd kaggle_gemma4_agent
python3 scripts/build_submission.py \
  --agent-dir configs/baseline \
  --output runs/local-packaging/submission.zip
```

The packaging command needs Python 3.10+ and its standard library only. It checks
the ZIP structure and records file hashes; it does not run a model or submit to
Kaggle. Configure Git with your own name and email before committing.

## Choose a starting point

| Goal | Start here |
| --- | --- |
| Understand coding agents and our plan | [Slides](slides/gemma4-kickoff/main.pdf) |
| Modify prompts, tools, or agent settings | [Baseline configuration](configs/baseline/) |
| Run one task without a local GPU | Copy & Edit the [official Kaggle starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent) and follow [participation](docs/participation.md) |
| Study public approaches | [Notebook reading notes](resource/notebooks/reading-notes.md) |
| Understand our measured results | [Submission guide](docs/submission.md) and [L4 evidence](resource/manifests/kaggle-runtime-smoke.json) |
| Plan fine-tuning | [Training plan](docs/training-plan.md); training has not yet been validated |
| Edit the presentation | [Slide source and build guide](slides/gemma4-kickoff/README.md) |

In Kaggle, attach the competition inputs, required model, and official wheelhouse
before starting an L4×4 session, with Internet disabled. Check your own quota and
start with one public task. Preserve needed outputs and stop the session when done.

## One change, one branch, one pull request

```bash
git switch main
git pull --ff-only
git switch -c experiment/short-description
# Edit files, then review exactly what will be committed.
git status --short
git add <specific-files>
git diff --cached
git commit -m "Describe the change and its purpose"
git push -u origin HEAD
gh pr create --base main
```

Create an Issue for the hypothesis or task, and link it in the PR. A useful PR
explains the change, the baseline it compares against, what was checked, and what
remains untested. Ask another teammate to review before merging. This is our
working convention; it is not a claim that branch protection is enabled.

Keep the known baseline intact when exploring: put alternative agent directories
under `configs/experiments/<name>/` and implementations under `src/`. Build bundles
with `scripts/build_submission.py --agent-dir <your-config> --output <run-path>`.
Coordinate formal competition submissions with the team; a PR does not upload a submission.

## Record experiments without committing large artifacts

Use `runs/<date>-<short-name>/` for local outputs. Keep a small report in
`docs/experiments/<date>-<short-name>.md` with:

- Question, baseline, and the single change being evaluated.
- Code commit, configuration, task/split identifiers, random seed, and model/data versions.
- Environment, hardware, time budget, elapsed time, and evaluation method.
- Results and failures, distinguishing measured results from estimates.
- Artifact location, hashes, and checkpoint/resume instructions when training.

An artifact location must be accessible to the teammates who need it. A path on
someone else's laptop or Trillium account is not a shared download link.

Do not commit tokens, raw competition data, base weights, adapters, runtime caches,
full logs, or checkpoints. The directory READMEs remain in Git to explain where
those files belong. Use the approved Kaggle inputs or agreed private artifact
storage, and keep provenance and access instructions in the experiment report.

## Environment and source boundaries

- `scripts/kaggle.sh`, download wrappers, Slurm scripts, and the documented HOME
  virtual environments describe the maintainer's Trillium setup. Inspect these
  before using them on another system; teammates can begin in the browser instead.
- Files containing `+computecanada` dependency versions are environment snapshots
  for Alliance wheels, not generic requirements to install on every laptop.
- `resource/` contains dated reference snapshots and lightweight evidence. Raw
  inputs, base models, and the runs named by manifests are not included in a clone.
- Preserve upstream notebooks and source snapshots. Put adaptations in our own
  code directories and record attribution. See [third-party notices](THIRD_PARTY_NOTICES.md).
- Edit slides in `sections/` and `figures/`, rebuild `main.pdf`, and coordinate
  Overleaf synchronization so browser edits and Git changes do not overwrite each other.
