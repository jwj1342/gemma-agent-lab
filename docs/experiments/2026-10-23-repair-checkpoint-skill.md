# Repair checkpoint skill v1

## Question and comparison

- Hypothesis: a deterministic post-edit checkpoint can reduce empty patches and structurally invalid
  submissions in cases where the model already inspected relevant source but did not complete a valid repair.
- Baseline and proposed change: the baseline tools and budgets are unchanged. The candidate adds one
  standard ADK skill and one mandatory pre-submission call in the main prompt.
- Task IDs / grouped split and data version: remote repair A/B not measured. The design target is the
  frozen L4 r5 failure-mechanism analysis from 2026-10-22.
- Success metric: fewer empty or structurally invalid submissions without reducing resolved-task rate or
  materially increasing timeout rate.

## Design and provenance

The skill is read-only and standard-library-only. It compares the worktree with `HEAD`, includes untracked
files, parses changed Python files without importing them, and reports an empty patch, protected test/config
edits, syntax errors, merge-conflict markers, and debugger residue. Logical failures are returned as JSON
with process exit code zero so the model can act on the finding. The script does not run tests, execute
repository code, stage changes, edit files, or inspect hidden tests.

The official competition overview documents custom tools and skills in submission archives, and the official
ADK skills documentation defines `SKILL.md`, `run_skill_script`, and sandboxed skill execution:

- <https://www.kaggle.com/competitions/gemma-4-developer-agent/overview>
- <https://github.com/google/adk-docs/blob/main/docs/skills/index.md>
- <https://github.com/google/adk-python/blob/main/src/google/adk/tools/skill_toolset.py>

The public `Gosling-dude/gemma4-developer-agent` repository at
`fb0826b7a560d766ece173f8d96f2827c0db2646` was reviewed as an architectural reference because it also
packages standard-library skill scripts. No source was copied: the repository did not expose a license at
the reviewed commit.

## Reproduction

- Source commit and configuration directory: working branch `feature/repair-checkpoint-skill`;
  `configs/experiments/repair-checkpoint-v1/`.
- Focused test: `python -m unittest tests.test_repair_checkpoint_skill`.
- Repository gate: `python scripts/check_repository.py`.
- Package command: `python scripts/build_submission.py --agent-dir configs/experiments/repair-checkpoint-v1 --output <outside-agent-path>.zip`.
- Agent ZIP SHA256: not frozen; build output is a disposable validation artifact until remote A/B promotion.
- Model handle/revision: `gemma-4-31b-it-qat-w4a16-ct`; no adapter.
- Platform and resources: CPU-only static/unit validation on Windows; model execution not measured.

## Results

- Original-failure and reference-success controls: not measured.
- Pipeline execution / issue resolution: not measured.
- Static and CPU validation: six focused checkpoint tests passed. The repository gate passed over 78 Python
  files, 14 shell scripts, 36 configurations, 87 maintained Markdown files, and six agent packages. It built
  every agent twice and confirmed deterministic ZIP bytes. The complete CPU suite passed 160 tests with one
  expected skip for an unstaged historical wheelhouse artifact.
- Official compiler validation: not run on this Windows host. The frozen v31 compiler source imports
  `google.adk`, but the package and an auditable local wheel are absent. This is an explicit runtime gate,
  not evidence of a compiler failure.
- Comparison with the baseline on matching tasks: not measured.

## Artifacts and next step

- Versioned implementation: `configs/experiments/repair-checkpoint-v1/`.
- Versioned focused tests: `tests/test_repair_checkpoint_skill.py`.
- Next step: compile and smoke-run the package in the frozen Kaggle or Trillium v31 runtime, then run a
  matched remote A/B on the frozen public tasks. Record checkpoint outcomes, tool cost, empty-patch rate,
  execution success, and issue resolution separately. Do not promote the experiment on structural-check
  results alone.
