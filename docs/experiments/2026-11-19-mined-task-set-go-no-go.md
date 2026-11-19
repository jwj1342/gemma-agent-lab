# #34 go/no-go: can the mined task set enlarge our local evaluation? (2026-11-19)

CPU-only assessment. Data downloaded from Kaggle dataset `mobassir/swe-agent-eval-validity-gemma4`
(companion to *Too Small to See*, Mobassir 2026). Nothing was uploaded, no GPU run was made.

## What the dataset actually ships

| Item | Value |
| --- | --- |
| `data/tasks.jsonl` | 50.9 MB, **4,207 tasks**, 64 repositories |
| Schema | `instance_id, repo, base_commit, patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS, problem_statement, problem_statement_source, gold_files, gold_lines, statement_type, leak_flag, leak_evidence, upstream_license, paper_split` |
| Labels | `statement_type` concrete 2,714 / terse 977 / design 516; `leak_flag` true 1,813 (43 %) |
| Splits | mined-197 197, google-100 100, google-reserve 63, train-mined 1,838, train-google 746, train-refreshed 1,263 |
| Licence | code/labels/results Apache-2.0; task content upstream MIT 1,654 / Apache-2.0 1,609 / BSD-3 900 / BSD-2 44, with `LICENSE_AUDIT.csv` |
| **Not shipped** | **repository snapshots and per-repo environment wheels.** README: "snapshots are not redistributed. Rebuild them with `git clone` and `git checkout <base_commit>`." |

## Our evaluator's contract

`scripts/evaluate_agent.py` builds `EvalConfig(snapshots_dir=<data>/snapshots, wheels_dir=<data>/wheels, ...)` and
reads `<data>/snapshots/<instance_id>.tgz` for every task. The competition mount supplies both for the 129
public tasks; **neither exists for the 4,207 mined tasks**. The verification path also needs `swegemma`,
which is not installed in this workspace (it only runs inside the Kaggle wheelhouse).

So the mined set is schema-compatible but **not runnable as-is**, and controls cannot be executed here.

## Connectivity measured (why a rebuild is still possible)

| Route | Result |
| --- | --- |
| `codeload.github.com/<owner>/<repo>/tar.gz/<sha>` | **works**: cyclopts @ `ff79b6c` fetched, HTTP 200, 2.5 MB |
| `pypi.org/simple` with `--index-url` | **works**: `rich==13.7.1` wheel downloaded |
| `git` protocol to `github.com` | blocked (connection reset) |
| Kaggle dataset upload from this host | blocked historically (googleapis.com unreachable) |

A rebuild script can therefore fetch snapshots (codeload) and wheels (PyPI) here, but publishing them as a
dataset needs a UI upload or a networked kernel, and the control runs need `swegemma`, i.e. a Kaggle kernel.

## Cost model (measured rates, not estimates)

Formula used for every row (so the numbers are reproducible):

```text
wall_seconds_per_arm = N × (agent_seconds_per_task + verify_seconds_per_task) + setup_seconds
quota_seconds_per_arm = 2 × wall_seconds_per_arm          # measured 2.0x billing on 4xL4
```

Inputs, all measured earlier in this repository: `agent_seconds_per_task` 323.7 s (baseline arm) to
361.5 s (adapter arm); `verify_seconds_per_task` 60 s; `setup_seconds` 1,827 s. The 11-18 review quoted
363-390 s per task, which is the same computation with the agent and verify terms added per row:

| N tasks | wall using 363 s/task | wall using 390 s/task | quota @363 | quota @390 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 19,977 s | 21,327 s | 39,954 | 42,654 |
| 100 | 38,127 s | 40,827 s | 76,254 | 81,654 |
| 120 | 45,387 s | 48,627 s | 90,774 | 97,254 |
| 150 | 56,277 s | 60,327 s | 112,554 | 120,654 |

The earlier table in the issue comment quoted the upper end (390 s/task) rounded; both ends are inside
"one arm only" for 100-120 tasks and outside the monthly allowance for 150, so the conclusion is unchanged.

The mined set solves the *task-supply* problem (4,207 tasks, 64 repos) but not the *budget* problem: a
100-150 task paired comparison does not fit our monthly 4×L4 allowance. It fits the owner's H100 queue
(separate compute) and the one-off 120-task runtime check that #36 requires before a final candidate.

## Cohort proposal (if a run is authorised)

Best clean supply is the repo-disjoint holdouts: **297 tasks over 9 repositories** (mined-197: click 73,
tomlkit 56, humanize 23, lark 18, sqlparse 15, arrow 12; google-100: mobly 65, openhtf 24, ml_collections 11),
of which terse 56 / concrete 212 / design 29 and leaky 105.

For a 120-task cohort: take all 56 terse holdout tasks, then 64 more stratified by repository from the
non-holdout pools (preferring terse, then concrete), keeping `leak_flag` as a reported strata rather than a
filter, and reporting the repository mix. Expected effect resolution: with our measured 29-task seed sd of
~1.6 tasks, 120 tasks gives roughly 0.8-task resolution for a one-arm comparison, enough to see the 3-5 point
differences the plan cares about, but only on H100-class compute.

## Verdict

- **GO as a task supply**: schema-compatible, licensed, labelled, 4,207 tasks over 64 repositories.
- **NO-GO as an offline drop-in here**: no snapshots, no wheels, no `swegemma` in this workspace; controls
  cannot be executed locally.
- **Next actions**: (1) build snapshots with codeload and wheels with `pip download --index-url
  https://pypi.org/simple` for the 9 holdout repositories, then run gold/empty controls on 3-5 tasks inside a
  Kaggle kernel with the wheelhouse (CPU, no GPU quota); (2) if controls pass, publish the snapshot/wheel
  bundle as a dataset from the kernel output (UI step, a human); (3) authorise 100-150 task arms only on H100,
  and keep 4×L4 arms at ~50-60 tasks.
