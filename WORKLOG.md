# Gemma Agent Lab Work Log

## Current project state

- November 25: infrastructure implementation and CPU verification complete. Official v31 CPU verifier
  `junzhejiao/mined-verifier-smoke-20261125` version 2 passes both sqlparse_8b0342730b and lark_1540 controls;
  94/94 Windows CPU tests pass. The prior pip bootstrap ordering failure and skipped required regex test
  are corrected and preserved as separate failed evidence. #34's initial CPU deliverables are satisfied
  with a provisional 105-task proposal; full runtime control approval and final-candidate L4 timing are
  not claimed. Keep #34 open. User requests #39 next after this closeout.
- November 24 evaluation infrastructure closeout is on `fix/mined-eval-infra-20261124`, based on PR #42
  at `426b9c8b` plus #40's byte-writing fix (`a14431d` locally). Windows preparation passes 14/14;
  the full CPU suite passes 93/93, repository checks pass with Bash available, and staged whitespace
  checks pass. No model inference, GPU allocation or competition upload.
- Strict mined controls now require executed JUnit tests, gold success and an ordinary empty assertion
  failure. Targeted CPU rerun rejects all six disputed historical controls; both comparison tasks pass.
  The default-image diagnostic ran Python 3.13; future notebooks pin the official starter image.
- A shared corrected 105-task manifest is provisional, from 220 historical rc=0/1 rows. Old 105-task
  metadata is preserved. Revalidate every selected task on the frozen runtime before inference.
- Official-verifier smoke is a separate CPU job, `junzhejiao/mined-verifier-smoke-20261124`, version 4.
  It pins v31 hashes and the starter image, freezes agent/task/ZIP inputs, and tests two mined tasks.
  Results and publication status are recorded in the closeout report when collected. Earlier failed
  preparations remain separate; none establish a successful verifier run.
- See `docs/experiments/2026-11-24-evaluation-infrastructure-closeout.md`. Raw evidence remains in the
  original workspace's ignored `runs/2026-11-24-eval-infra-closeout/`. Progress checks are user-directed;
  this task reads job evidence directly and sets up no periodic polling.

## Work history

### 2026-11-25 01:44 (Asia/Shanghai) - Complete official controls and assess #34 before starting #39

- Read the original and feature-worktree logs, resumed the pending CPU verifier, and re-read live #34/#39
  requirements once at the user's request. No recurring monitoring. #34 and #39 remain open remotely.
- Fixed offline bootstrap ordering: install the build backend before resolving editable metadata.
  Added an ordering regression test. The next official run passed sqlparse but rejected Lark's gold
  despite rc=0 because a required regex test was skipped; added regex to the offline bundle and actual
  sandbox installation. Include build backends, transitive wheels and the observed sitecustomize wrapt
  dependency in bundle preparation. Final official verifier now passes both task/control pairs.
- Recorded successful receipt in `resource/manifests/mined-verifier-smoke-20261125.json`; raw traces,
  notebook hashes and failed-attempt recovery are under the original workspace's
  `runs/2026-11-25-infra-and-teacher/`. No LLM/model calls, GPU allocation, competition submission or
  current-quota query. Verified 94 CPU tests; static and whitespace checks are part of publication.
- Updated the closeout report with a deliverable-by-deliverable #34 assessment and an explicitly
  assumption-based H100 estimate. Initial CPU feasibility work is complete; leave the issue open for
  acceptance/sharing and the conditional final-candidate L4 timing. All 105 official controls remain
  a runtime gate, not an inferred result from the old rc-only screen. Proceed to repository-disjoint
  teacher mechanism work for #39; no independent teacher API is configured in the inspected environment.

### 2026-11-24 02:39 (Asia/Shanghai) - Finish the evaluation infrastructure

- Objective: close the Windows preparation, control acceptance, billing and reproducible handoff gaps
  identified in the user's preceding GitHub audit. Read the original workspace WORKLOG and project
  guidance first; use the basic-programming-principles skill for implementation.
- Isolated PR #42's head in a feature worktree. Windows rejected a historical CSV containing colons;
  sparse materialization also initially inherited conflicting worktree settings. Recovered by loading
  the index and materializing Git blob bytes, leaving that one CSV off disk. Initial checkout CRLF
  conversion caused baseline hash mismatches, fixed by preserving original bytes. No original workspace
  implementation or staged backlog was edited.
- Re-ran exact #40 head: production byte-writing fix works; one test restored its fixture in text mode.
  Changed the fixture to byte restoration and checked LF preservation. Deferred Linux-only fcntl imports
  from portable helpers and standardized checker diagnostic paths. Windows preparation now 14/14.
- Added shared strict JUnit controls, detailed log collection, chronological control selection, explicit
  provisional legacy screening, source-task hashes and portable manifests. Snapshot/wheel bundles export
  separate verification/public task files, resolve transitive wheels, retain wheel hashes, reject missing
  IDs/existing outputs, and fail on partial builds. Corrected 105-task and 50-task-pair billing figures.
- CPU diagnostic: all six disputed tasks rejected; sqlparse_8b0342730b and lark_1540 accepted. The
  diagnostic has 8 selected tasks, not a representative validity estimate. Recorded metadata in
  `resource/manifests/mined-control-audit-20261124.json`; full test tails remain ignored.
- Added a no-inference official-verifier notebook. First launch lacked kernelspec; another preparation
  used the old local snapshot registry. Fetched the actual PR #23 inventory and pinned v31. Version 3
  verified wheelhouse/setup/frozen inputs but stopped at asyncio.run inside a notebook event loop;
  version 4 uses native await. Preserve failures separately and report actual verification outcomes.
- Validation: 93 CPU tests passed on Python 3.14.6 / Windows; checker passed (58 Python, 14 shell,
  19 configuration, 68 Markdown files, three static packages); whitespace check passed. Tests exercise
  collection/timeout/skip rejection, stale-control replacement, output preservation, redaction, and
  self-contained CPU notebook preparation. No training, inference, GPU quota query or competition entry.
- Changed files: control/sweep/collector/selector/bundle/verifier scripts, three portability fixes,
  focused tests, five metadata manifests, report/navigation and this log. Independent teammate approval
  remains required before a routine merge; no approval is inferred from CI.
