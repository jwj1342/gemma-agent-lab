# Gemma Agent Lab Work Log

## Current project state

- Infrastructure closeout is published in PR #42 at `5d234fd`; hosted Repository checks pass. The
  successful official v31 CPU smoke accepts two mined tasks after offline sandbox setup. The initial
  #34 CPU feasibility deliverables are complete with qualifications. Keep #34 open for acceptance,
  artifact sharing and the conditional final-candidate L4 timing; all-cohort runtime approval is pending.
- The corrected 105-task manifest is provisional from 220 historical rc=0/1 rows. Old metadata is
  preserved; every selected task needs official controls on the frozen runtime before inference.
- Current branch `experiment/teacher-evidence-20261125` starts #39 separately. Twelve teacher tasks over
  four repositories are disjoint from the seven mined evaluation repositories. Cases 1 and 3 are executed and
  verified: more_itertools_757 (iter_index stop bound) and more_itertools_1157 (serialize generator support,
  private `_iterator`/`_lock`, `send`/`throw`/`close` forwarded) both pass official controls and independent
  grading; case 2 (more_itertools_1116) is deferred because the terse statement does not name the new API and
  the four plausible names already exist as single-draw recipes. The graph added a novel locator only in case 3
  (the existing `TestSerialize` reference). No student instruction or adapter is promoted.
- All 97 project CPU tests pass. The prototype is not the official ADK graph and no independent
  graph-on/off teacher-model comparison or Gemma gain is claimed. The user is unsure about teacher API
  configuration; the current-session mechanism pilot uses no separate paid API.
- Publication: infrastructure PR #42 at 5d234fd has successful hosted CI; teacher experiment commit
  4da6631 is published as draft PR #43, stacked on #42's branch. Neither PR is merged, and neither
  research issue is closed. Human teammate approval remains required before routine merging.
- Reports: November 24 infrastructure closeout and November 25 teacher source-graph pilot under
  `docs/experiments/`. Raw evidence is under the original workspace's ignored
  `runs/2026-11-25-infra-and-teacher/` and `runs/2026-11-25-issue39-teacher/`. No model inference inside
  Kaggle, GPU allocation, competition submission, merge or recurring status monitor.

## Work history

### 2026-11-25 19:30 (Asia/Shanghai) - Teacher cases 2 and 3: one deferred for statement ambiguity, one verified

- Objective: continue the #39 mechanism study past case 1 on the frozen 12-task teacher development set.
- Case 2 `more_itertools_1116` ("Add random ordered combinatoric functions"): official controls are valid
  (kernel `junzhejiao/teacher-controls-20261125-case2`, empty fails, gold passes), but the statement does not
  name the new API and the four plausible names already exist in `more_itertools/recipes.py` as single-draw
  recipes. No patch was frozen; the case is recorded as deferred with the statement-specification limit.
- Case 3 `more_itertools_1157` (serialize generator support): the graph located the unpatched
  `serialize.__init__/__iter__/__next__` and the existing `TestSerialize.test_concurrent_calls` reference — the
  first case where the structural query surfaced something new. The frozen patch privatised `_iterator`/`_lock`
  and forwarded `send`/`throw`/`close` under the same lock; local smoke checks passed and the official verifier
  (`junzhejiao/teacher-verification-20261125-case3`) returned valid controls and `teacher_resolved: true` with
  zero model calls.
- Deviation recorded: case 3 controls were run after the teacher opened the source; no reference or injected
  test patch was read.
- Changed files: `docs/experiments/2026-11-25-teacher-source-graph-pilot.md`,
  `resource/manifests/teacher-mechanism-cases-20261125.json`,
  `resource/manifests/teacher-verifier-case3-20261125.json`, `WORKLOG.md`. Raw evidence stays in the ignored
  `runs/2026-11-25-issue39-teacher/case-2-*` and `case-3-*` directories.
- Verification: case 3 verifier log hash recorded in the receipt; no GPU, no competition submission, no merge.

### 2026-11-25 01:57 (Asia/Shanghai) - Execute the first repository-disjoint #39 teacher mechanism trial

- Objective: after infrastructure closeout, start the stronger-teacher work requested by the user.
  User is unsure whether a separate API is configured; use a current-session mechanism trial and
  distinguish it from an independent paired teacher evaluation. No extra teacher API was invoked.
- Froze 12 public development tasks (three each from more-itertools, tenacity, w3lib and mistune),
  excluding all seven mined evaluation repositories and leak-flagged tasks. Source, statement and
  license provenance remain explicit. First case/order was fixed before opening the teacher packet.
- Implemented a read-only partial AST source graph, with explicit symbol IDs, same-file syntactic
  links, unresolved calls, source hashes and trace output. Two targeted tests reject dynamic/shadowed
  links and ambiguous node shortcuts. It is not an ADK graph replacement or deployed competition skill.
- more_itertools_757 passed official empty/gold controls first. Teacher read only public statement,
  base source/tests and the Python 3.12 itertools contract, then froze a minimal stop-bound change.
  Five existing tests and 108 independent boundary combinations pass. Independent v31 grading in
  `junzhejiao/teacher-patch-verification-20261125` v1 resolves the teacher patch with valid controls.
  No reference/test patch was shown to the teacher before its patch was frozen.
- Two graph queries cost 0.6035 local process seconds / 1,025 bytes. They exposed a caller already
  found by literal search; critical external/dynamic calls remained unresolved. No marginal diagnostic
  fact, causal speedup, teacher A/B effect or Gemma gain is established. Do not force graph use or
  promote a student procedure from this case; preserve the null result and the frozen remaining cases.
- Changed the verifier notebook generator to grade a selected task's teacher prediction after valid
  controls, with task isolation and patch hashes. Added three metadata receipts, the graph prototype,
  focused tests and the English pilot report/navigation. Raw patches, source snapshots and logs remain
  ignored under `runs/2026-11-25-issue39-teacher/` in the original workspace.
- Verification: 97 CPU tests pass; static/whitespace checks are run for the published experiment branch.
  Infrastructure PR #42's hosted CI is successful at 5d234fd. Independent teammate review remains
  required before merging; neither #34 nor #39 is closed and no model/GPU competition run was launched.
- Publication completed: teacher experiment commit 4da6631 is on draft PR #43, targeting PR #42's
  infrastructure branch. Static checks passed (60 Python / 14 shell / 19 configuration / 70 Markdown /
  three packages), as did whitespace checks. PR #42's body/head were confirmed updated. Follow-up:
  reviewer acceptance for infrastructure; continue predeclared teacher cases only as a mechanism
  study until an independent controlled teacher comparison and a held-out Gemma A/B are available.

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
