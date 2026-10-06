# Repository map

This repository contains the team's competition configuration, reusable execution scripts, reference
material, and small evidence records. GPU runs use the same agent bundle on Kaggle L4×4 or Trillium H100.
Choose the platform at the execution entry point; the agent configuration is shared.

## Find the right starting point

| Question | File or directory |
| --- | --- |
| What are we building, and what works today? | [Project homepage](../README.md) |
| What does the competition require? | [Overview](overview.md), then the original [official pages](../resource/competition/pages.json) |
| Where do I edit the agent? | [Configuration map](../configs/README.md) |
| How do I run it with no local GPU? | [Kaggle runtime](kaggle-runtime.md) |
| How do I use our existing H100 setup? | [Trillium runtime](cluster-runtime.md) |
| What does a particular script do? | [Script catalog](../scripts/README.md) |
| Where are the results behind a claim? | [Evidence catalog](../resource/manifests/README.md) |
| Where are public notebooks and discussions? | [Research resource index](../resource/README.md) |
| How do I record my own results? | [Experiment report template](experiments/TEMPLATE.md) |

## Top-level files

| File | Purpose |
| --- | --- |
| `README.md` | Project introduction, quick start, verified results, and navigation |
| `CONTRIBUTING.md` | Team access, branches, pull requests, and experiment conventions |
| `AGENTS.md` | Instructions for coding assistants working in this checkout |
| `THIRD_PARTY_NOTICES.md` | Attribution and licensing boundaries for collected sources |
| `.gitignore` | Exclude credentials, large inputs, runtime outputs, and local slides |
| `.gitattributes` | Mark reference snapshots as vendored in GitHub language statistics |
| `.github/ISSUE_TEMPLATE/experiment.md` | Propose a hypothesis and its resource budget |
| `.github/pull_request_template.md` | Explain a change and its validation to reviewers |

## Working directories and ownership

| Directory | What belongs here | Who changes it? |
| --- | --- | --- |
| `configs/` | Agent YAML/prompts and named environment snapshots | Experiment authors; preserve the measured baseline |
| `src/` | New team implementations such as retrieval or training utilities | Implementation authors; currently a reserved extension point |
| `scripts/` | Existing build, evaluation, preparation, and collection entry points | Maintainers and experiment authors |
| `tests/` | Lightweight checks of our preparation tools | Contributors changing those tools |
| `docs/` | Human-readable guides and compact experiment reports | Everyone |
| `resource/` | Unchanged upstream material, reading notes, and dated evidence | Reviewers/collectors; preserve source attribution |
| `data/raw/` | Original competition downloads, excluded from Git | Download/preparation tools |
| `data/processed/` | Derived splits and training records, excluded from Git | Data preparation tools |
| `model/base/`, `model/adapters/` | Local weights and exports, excluded from Git | Download/training tools |
| `runs/` | Per-experiment inputs, logs, outputs, caches, and checkpoints, excluded from Git | Execution tools |
| `slides/` | Maintainer's local presentation, entirely excluded from Git | Separate Overleaf workflow |

Directory READMEs stay in Git where `.gitignore` explicitly allows them. A clone provides configuration
and documentation; each platform supplies its own data and weights. Kaggle mounts these as Inputs.
The current Trillium installation keeps them under `data/`, `model/`, and `resource/downloads/`.

## How a change reaches an experiment

1. Copy `configs/baseline/` to `configs/experiments/<name>/`, then edit prompts or settings there.
   Keep notes in `docs/experiments/`; a Markdown note inside the agent directory would enter its ZIP.
2. For Kaggle, generate a notebook with `prepare_kaggle_runtime.py --agent-dir … --task-id …`.
   It packages the chosen configuration locally and reads the public task from Kaggle Inputs at runtime.
3. For H100, use `prepare_eval_run.py` to freeze the agent and public task, then use the existing Slurm entry point.
4. Both routes call `validate_agent.py`, `evaluate_agent.py`, and `summarize_eval_smoke.py`.
   They check original failure, reference-patch success, and the agent's generated patch separately.
5. Save an experiment report and artifact locations. Coordinate any official upload through the
   [submission workflow](submission.md); running a notebook alone does not submit an entry.

The execution engines and installers remain separate because Kaggle and Trillium have different
environments. Shared evaluation logic stays in the same small set of scripts.

## Reading the reference collection

| Path pattern | Meaning |
| --- | --- |
| `resource/competition/*.md` | Original official page text; filenames identify topics |
| `resource/competition/pages.json`, `metadata.json` | Original page collection and competition metadata |
| `resource/competition/accelerator-quota*.json` | Dated account checks; see participation for the historical duration bug |
| `resource/notebooks/official/` | Original starter notebook and metadata |
| `resource/notebooks/public/<owner>/<slug>/source.ipynb` or `source.py` | Downloaded upstream notebook/script |
| `resource/notebooks/public/<owner>/<slug>/readable.txt` | Text rendering used for line references in our reviews |
| `resource/notebooks/public/<owner>/<slug>/response.json` | Original download response and metadata |
| `resource/notebooks/reading-notes.json` → `.md` | Editable structured team reviews → generated reading guide |
| `resource/notebooks/index.json`, `review-features.json` | Collection listing and static source features |
| `resource/discussions/` | Topic listing, captured opening posts/replies, and generated navigation |
| `resource/models/<owner>/<slug>/` | Original model/instance cards; listing does not imply evaluation approval |
| `resource/models/wheelhouse/` | Official dependency-dataset metadata |
| `resource/leaderboard/` | Captured CSV and summary; a snapshot rather than live rankings |
| `resource/harness-source/` | Extracted official package source for inspection; runtime installs come from wheels |
| `resource/manifests/` | Named receipts, file inventories, audits, and measured results |

Use the subdirectory indexes to select an individual notebook, discussion, model, or result. Original
sources stay unchanged; write our interpretation in reading notes or `docs/`. Dates on snapshots matter.
