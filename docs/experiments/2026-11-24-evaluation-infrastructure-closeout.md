# Evaluation infrastructure closeout (2026-11-24)

## Scope

Finish the #34 evaluation infrastructure before another model comparison. Based on PR #42 at
`426b9c8b1741d928c12996701476b015dad434d7`, with #40's one-line byte-writing fix included separately.
Development and Windows tests run on Python 3.14.6; remote smoke jobs are CPU only. No model inference,
GPU allocation, LoRA training, or competition submission is part of this change.

## Windows preparation

The exact #40 head was materialized from Git blobs, preserving LF bytes. Its requested preparation suite
passed 13/14 tests. The remaining failure came after the test's own corruption check: restoring the task
with `write_text` converted LF back to CRLF. Restore the original bytes instead and assert LF preservation;
the full preparation suite then passes 14/14, including the submission dry-run and tamper rejection.
No competition entry is sent by that dry-run.

Two existing Windows CPU-suite problems were also fixed: defer POSIX file-lock imports until Linux runner
execution, so portable hashing helpers can be imported; format repository-check diagnostics with POSIX
paths, matching the documented paths on both operating systems. Linux locking behavior is unchanged.

## Control acceptance

The historical sweep accepted `gold_rc == 0` and any nonzero empty return code. Re-parsing the saved source
logs reproduces 226 accepted IDs, but six had usage/collection errors or a timeout instead of demonstrated
test failures:

| Task | Historical empty rc | In old 105-task cohort |
| --- | ---: | --- |
| lark_1452 | 4 | yes |
| click_3637 | 4 | no |
| click_3695 | 4 | yes |
| tomlkit_509 | 124 | yes |
| mobly_932 | 4 | yes |
| arrow_1201 | 2 | yes |

The new shared gate requires JUnit evidence: gold exits zero with executed, non-failing tests; empty exits
one with at least one test failure; neither side has collection errors or all tests skipped. Failed
installation, missing reports, no tests, and timeouts are rejected. The sweep keeps the exact FAIL_TO_PASS
node IDs and does not silently broaden them to whole-file tests. Detailed per-task records are printed
to stdout and appended to JSONL so the collector can retain diagnostics without fetching cached repositories.
Sample controls use the same generator and gate as a full sweep.

The CPU rerun `junzhejiao/mined-infrastructure-controls-20261124` v1 independently rejected all six disputed
tasks and accepted both comparison tasks (`lark_1540`, `sqlparse_8b0342730b`). Five disputed tasks failed
collection/selection and `tomlkit_509` timed out. arrow now fails its gold control too. The captured task
test counts and statuses are tracked in `mined-control-audit-20261124.json`; full logs remain local.
This diagnostic run used Kaggle's default Python 3.13 image, not the scorer image. Future control notebooks
pin the official starter image, and full-cohort validation remains necessary on that runtime.

The selector uses the latest supplied result per task, so a later control failure overrides an earlier
pass. Legacy logs are unverified by default. `--legacy-screen` permits rc=0/1 rows for planning only and
sets `evaluation_ready: false`. It cannot turn historical labels into official-verifier evidence.

## Shared cohort handoff

Three metadata-only manifests are tracked under `resource/manifests/`:

- `mined-cohort-20261121-historical.json`: old draw preserved and explicitly marked historical.
- `mined-control-screen-20261124.json`: 297 task IDs, repository IDs, recorded statuses and source hashes;
  no gold patches, tests, repository contents or credentials.
- `mined-cohort-20261124-provisional.json`: deterministic new 105-task draw from the 220 rc=0/1 pool,
  seed `mined-cohort-2026-11-24`, with source task/log hashes and `evaluation_ready: false`.

The new draw has click 34 / tomlkit 26 / mobly 21 / lark 8 / sqlparse 7 / arrow 5 / ml_collections 4;
statement mix terse 21 / concrete 76 / design 8; leak-flagged 27 / clean 78. No model outcomes were used
to select it. Its controls must be rechecked on the exact runtime before it becomes an inference cohort.
The old draw is not overwritten or silently reinterpreted.

With the source dataset downloaded to a local path:

```bash
python scripts/select_mined_cohort.py --tasks /path/to/tasks.jsonl \
  --logs resource/manifests/mined-control-screen-20261124.json --legacy-screen \
  --seed mined-cohort-2026-11-24 --size 105 --out /new/path/cohort.json

python scripts/make_mined_sweep_notebook.py --tasks /path/to/tasks.jsonl \
  --ids lark_1452 click_3637 click_3695 tomlkit_509 mobly_932 arrow_1201 \
  --slug mined-infra-controls --title 'mined infra controls' --out /new/path/cpu-controls

python scripts/build_mined_eval_bundle.py --tasks /path/to/tasks.jsonl \
  --ids lark_1540 sqlparse_8b0342730b --out /new/path/bundle
```

The bundle includes `tasks.jsonl` for verification, `tasks.public.jsonl` with reference/test labels removed,
snapshots, test wheels including transitive dependencies, and content hashes. It rejects unknown task IDs,
existing outputs, and partial download success. Wheel versions are resolved when building, then frozen by
the manifest; rebuilding later is a new bundle, not an assumption of identical dependencies.

## Official-verifier smoke

`make_mined_verifier_notebook.py` prepares a separate CPU-only notebook that verifies the official v31
wheelhouse digests, rebuilds two snapshots, resolves offline wheels, freezes each agent/public task/ZIP
with `prepare_eval_run.py`, and calls the unmodified official `verify_task` for empty and gold patches.
The dependency hook installs into isolated sandbox environments from the captured wheels and checks that
task imports resolve inside the workspace. The official starter image is pinned by digest. No model or
tokenizer is loaded. Verification results are checked using the shared `control_checks` function.

```bash
python scripts/make_mined_verifier_notebook.py --tasks /path/to/tasks.jsonl \
  --ids lark_1540 sqlparse_8b0342730b \
  --out /new/path/verifier
```

The tracked `mined-verifier-runtime-20261124.json` retains the reviewed 41-wheel v31 digests; an explicit
`--snapshot` may override it for a separately reviewed runtime. The generator refuses an unmatched mounted
wheelhouse. The downloaded source dataset and wheelhouse remain external inputs.
Inference, model serving, GPU runtime, and all-cohort official parity are separate gates.

## Billing and analysis

The November 21 specification now distinguishes wall seconds from 2x billed L4 seconds. With its stated
historical rates, 105 tasks cost 84,231-92,169 quota seconds per arm; a 50-task pair costs
84,048-91,608. H100 timing has not been measured by these calculations. Current quota was not queried.
The CPU sweep collector produces control reports only; a paired resolution analysis must use actual
inference/verifier rows and is not supplied by that collector.

For #34's requested H100 planning estimate, the owner's #26 description reports about 2x faster
generation on H100. If agent time halves, verification stays at 60 seconds per task, and setup stays at
1,827 seconds, 105 tasks would take `105 * (323.7/2 + 60) + 1827` to
`105 * (361.5/2 + 60) + 1827`, approximately 25,121-27,106 wall seconds (7.0-7.5 hours) per arm.
This is an assumption-based estimate from an owner-reported speed ratio, not an independently measured
H100 runtime. Tool subprocess time does not necessarily halve. Calibrate a small H100 batch and split
allocations according to the actual queue limits before launching a full arm.

## Verification record

Local artifacts: `runs/2026-11-24-eval-infra-closeout/` in the original workspace. Focused tests passed
40/40; the subsequent full Windows CPU suite passed 92/92 before the final notebook-preparation test
was added. The final CPU checks and remote outcomes are recorded below when collected. Initial materialization
failures came from a tracked colon-containing historical CSV and Windows checkout line endings; source
blobs were materialized without altering the upstream bytes, with only that CSV excluded from disk.

The first official CPU notebook failed before execution because kernelspec metadata was absent. The
generator now supplies it and a regression test checks that a prepared notebook declares its kernel.
That failed launch supplies no verifier result.
The second preparation selected the old local v29 registry; the subsequent preparation used PR #23's
actual v31 inventory, verified the mounted wheelhouse, rebuilt snapshots and wheels, and froze the two
tasks, but stopped before verification because a notebook already has a running event loop. The final
cell uses native top-level await. These failed attempts are preserved and do not count as verifier passes.

November 25 follow-up: the next launch reached the dependency hook but failed because pip inspected
editable metadata before installing hatchling. Bootstrap packages and editable installation are now
separate offline steps, with a regression test for this ordering. The first completed verifier pair
accepted sqlparse_8b0342730b. For lark_1540, gold exited zero but the official verifier rejected a skipped
required regression test (`test_recursive_pattern`). The missing optional `regex` test dependency is now
included in both the snapshot bundle and sandbox installation; a zero exit is never substituted for the
official resolved flag. Final rerun outcomes are recorded separately.

Final November 25 result: `junzhejiao/mined-verifier-smoke-20261125`, kernel version 2 (local attempt 6),
passes both official control pairs. sqlparse_8b0342730b and lark_1540 both have empty rc=1 with actual test
failures and gold rc=0 with `resolved=true`. The successful Lark run executes the required regex test.
The receipt, source/stream hashes and per-control outcomes are tracked in
`resource/manifests/mined-verifier-smoke-20261125.json`. No model calls occurred. Raw evidence is in
`runs/2026-11-25-infra-and-teacher/official-verifier-v6/` in the original workspace, with the complete
runtime input receipt retained in the kernel output. Final local validation: 94/94 CPU tests, repository
checks and whitespace checks. No recurring job-status monitor was installed.

## Issue #34 acceptance assessment

| Requested deliverable | Evidence and status |
| --- | --- |
| License and shipped-content audit | Delivered in PR #41; tasks/tests ship, snapshots and offline wheels must be rebuilt |
| Offline evaluator feasibility and small gold/empty sample | Two tasks now pass the official v31 verifier using isolated offline-installed test environments |
| A stratified 100-150 task proposal and H100 wall estimate | Versioned 105-task proposal, source hashes, selector and qualified 7.0-7.5 hour planning estimate delivered; the proposal remains provisional |
| Numeric go/no-go | GO for rebuilding a larger evaluation supply; require official controls per task before inference. NO-GO for treating all 220 historical rc=0/1 rows as approved runtime controls |
| Later final-candidate 120-task L4 runtime | Not triggered or completed; no final candidate or runtime measurement is supplied here |

The initial CPU feasibility deliverables are complete with these qualifications. The issue should remain
open for acceptance, bundle sharing and the conditional final-candidate timing check. This work does not
establish power to resolve a one- or two-task gain, and no claim is made that all 105 controls already
pass the official verifier. The requested next workstream is #39; derive its teacher procedures outside
the seven evaluation repositories, and do not promote them without a separate controlled comparison.
