# Repair checkpoint skill v1

## Question and comparison

- Hypothesis: a deterministic post-edit checkpoint can reduce empty patches and structurally invalid
  submissions in cases where the model already inspected relevant source but did not complete a valid repair.
- Baseline and proposed change: the baseline tools and budgets are unchanged. The candidate adds one
  standard ADK skill and one mandatory pre-submission call in the main prompt.
- Task IDs / grouped split and data version: the remote repair A/B is prepared but not yet measured. It uses
  the ten valid-control tasks from the frozen `fast` cohort; `fastapi_15745` and `requests_7502` remain
  excluded because their controls were invalid in the matching v31 study. The design target is the frozen
  L4 r5 failure-mechanism analysis from 2026-10-22.
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
- Prepared run: `runs/2026-10-24-repair-checkpoint-fast-ab/` (local and Git-ignored). The task order is
  `rich_4079`, `rich_3506`, `rich_3468`, `rich_3278`, `httpx_3672`, `fastapi_14455`, `fastapi_14266`,
  `fastapi_15800`, `fastapi_14609`, and `fastapi_9753`.
- Agent ZIP SHA256: baseline `aa597ea39b66412eb53fcd3cd2fcc470c0650853a4ebe69f73d777f0ba7e822c`;
  repair checkpoint `47db85f004f4ce5c7943a819a2d5343e686775d2ac3f10e0c875e30d1d0eb2b2`.
- Private Kaggle notebooks: baseline
  `junzhejiao/repair-checkpoint-fast-baseline` (`d9203e42093e388ebe09c4cf91fa105586999456edae2b1d48f9e617e92925d8`)
  and candidate `junzhejiao/repair-checkpoint-fast-v1`
  (`552fd2503a730e2c481585440cf99fdff78dee06016d6369b54bd1df3ebabeaa`). These are prepared files, not
  uploaded notebook versions.
- Batch generator: local staged backlog at `E:/Desktop/CodingAI/scripts/prepare_lora_eval_notebook.py`,
  SHA256 `07055b7f488824e1d4efedfb9be514c1e4a0a241e2254c2b54167ed2d3d8219e`; its 19 CPU tests passed. The
  generated notebooks embed the current branch's evaluator and freeze all six runtime-script hashes.
- Model handle/revision: `gemma-4-31b-it-qat-w4a16-ct`; no adapter.
- Platform and resources: CPU preparation on Windows; planned private Kaggle L4x4 runs with wheelhouse v31,
  seed 42, and unchanged four-minute / 40-tool / 80-turn budgets. Model execution is not yet measured.

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
- Launch gate: the raw Kaggle quota API at 2026-10-23 17:19:51 UTC reported 114,317.472 of 108,000 GPU
  seconds used, zero reserved, zero remaining, and refresh at 2026-10-24 00:00:00 UTC (08:00
  Asia/Shanghai). No notebook was pushed before the refresh.

## Artifacts and next step

- Versioned implementation: `configs/experiments/repair-checkpoint-v1/`.
- Versioned focused tests: `tests/test_repair_checkpoint_skill.py`.
- Next step: after 08:00 Asia/Shanghai on October 24, recheck quota and push both prepared private notebooks
  without rebuilding them. Record control coverage, official compilation, task-level gains/regressions,
  checkpoint calls and verdicts, empty-patch rate, wall time, tool cost, and issue resolution separately.
  Do not promote the experiment on structural-check results alone.
