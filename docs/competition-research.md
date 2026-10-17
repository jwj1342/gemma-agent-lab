# Competition research and operating constraints

Collection began on October 4, 2026 in the evening; API timestamps are October 5 UTC.
This report distinguishes official rules, organizer clarifications, community reports, and local static inspection.
**Overview and Data description were retrieved in full** and are preserved in `resource/competition/`.
This is a dated research snapshot; recheck dynamic rules before new submissions.

## Task, score, and submission

- The agent receives a repository snapshot and issue, edits source offline, and returns a patch.
  Independent verification applies the patch and tests; the score is the fraction of issues resolved.
- Upload `submission.zip` with root `agent.yaml`. Harness-registered tools and sandbox skill scripts
  are supported; submitted Python cannot provide an arbitrary host entry point.
- Prompts, ADK configuration, skills, and optional PEFT LoRA are allowed. Conservatively keep the
  expanded package below **3 GiB**, including adapters. Allowed extensions are
  `.yaml/.yml/.md/.txt/.py/.json/.safetensors`; exclude symlinks, escaping paths, pickle weights, and nested archives.
- All agents/subagents currently require **`gemma-4-31b-it-qat-w4a16-ct`**. Forum plans for smaller models are not current approval.
- The snapshot permits **one submission per day**, **two final selections**, and **five team members**.
- Approximately **120 hidden tasks**, roughly half public/half private scoring, share one 12-hour run.
  Hidden tasks come from private repositories, not random held-out tasks from the four public repositories.
- The 12 hours include sandbox setup and exclude subsequent patch verification. Per-task agent budgets
  exclude setup. Organizers confirmed sequential tasks. At 120 tasks, six minutes each would consume
  all 12 hours before overhead; an initial 4–4.5-minute cap is a hypothesis to test on L4.
- A September 28 clarification still described global timeout as an error. We found no explicit confirmation
  that automatic zero-scoring of unfinished tasks had shipped; do not depend on a planned change.
- Deadlines in 2026: optional paper November 12, entry/team merger November 25, final prediction
  December 2, at official 23:59 UTC deadlines. Prizes: $65,000 prediction and $35,000 separate paper track.

Sources: [Overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview),
[Rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules),
[task count](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744258),
[hidden repositories](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744951),
[ordering and budget](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743063),
and [timeout/parallelism](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964).

## Official compute: L4×4

The competition provides **four NVIDIA L4 GPUs, 24 GB each, 96 GB in aggregate**.
Copy & Edit the official starter, attach competition data, the official wheelhouse, and required model,
and select L4×4 with Internet disabled. A saved packaging notebook must actually output
`/kaggle/working/submission.zip` before that output can be submitted. Direct ZIP upload is also
supported, as verified in the [submission guide](submission.md). UI labels may change.

- L4×4 consumes quota at **twice** the older T4×2/P100 reference rate; organizers may adjust it.
- Before our L4 smoke run, the October 5 raw API showed **108000 GPU seconds (30 hours)**,
  zero used/reserved, and **72000 TPU seconds (20 hours)**. Thirty reference hours correspond to
  about 15 L4×4 wall-clock hours, not a guarantee that one session can run for 15 hours.
- The early six-hour snapshot was incorrect: kagglesdk 0.1.37 `TimeDeltaSerializer.serialize`
  used `timedelta.seconds`, dropping one day and turning 30 hours into 21600 stored seconds.
  `accelerator-quota.json` is retained for provenance only. Use raw durations and `total_seconds()`,
  recording both usage and reservations. This was a serialization correction, not a quota increase.
- The reported refresh was `2026-10-10T00:00:00Z`. Recheck actual remaining quota before running;
  a quota allowance does not guarantee immediate machine availability.
- Personal notebook quota and the 12-hour scoring budget are different constraints.
- H100 measurements do not establish L4 timing or memory behavior. Validate near the scoring hardware.

Sources: [accelerator rules](../resource/competition/upgraded-accelerators.md),
[L4 specifications](https://docs.cloud.google.com/compute/docs/gpus), and
[raw quota response](../resource/competition/accelerator-quota-raw.json).

## Data inventory

The complete package has 524 files, **22416766904 bytes (20.88 GiB)**; its ZIP has 21929289087 bytes.
Every filename/size matches the official inventory, and extraction checked CRC. Structural audits covered
all tasks, graphs, NPZ headers, wheel metadata, and TGZ members without executing repository code.
See the [audit](../resource/manifests/data-audit.json).

| Component | Count | Observations |
| --- | ---: | --- |
| `tasks.jsonl` | 129 | FastAPI 67, Rich 48, Requests 13, HTTPX 1; issue, hints, base commit, reference/test patches |
| `snapshots/` | 129 TGZ | 237,734 members; full extraction about 21.13 GiB; extract on demand |
| `graphs/` | 127 JSON | Unique repo/commit pairs; 711–4597 nodes per graph; observed edge type only `calls` |
| `embeddings/` | 127 NPZ | 256-dimensional float32 per symbol; headers inspected without unnecessary pickle loading |
| `wheels/` | 124 | Multiple versions of 41 repository-test packages; separate from serving/training wheels |
| `docker/` | 4 | Two Python 3.13 Dockerfiles and imp/telnetlib compatibility files |
| `sandbox/setup.py` | 1 | Editable installs, dependency selection, working-directory injection, test configuration; sandbox only |
| `sample_submission/` | 10 | Analyzer/coder YAML, prompts, sampling, budgets, and two example adapters |
| `HARNESS_README.md` | 1 | 671-line manual; submission, tools, budgets, two-stage evaluation, and local CLI reviewed |

All 129 tasks match snapshots, graphs, and embeddings. The description's 256-file graph/embedding
counts included older task-name hardlinks; the current download has 127 commit-name files of each type.
This is a packaging discrepancy, not evidence of missing downloads. Mappings are in `tasks-audit.json`.
Both example safetensors contain four tensors and valid offset bounds; [headers](../resource/manifests/sample-adapters.json)
record shapes, dtypes, and hashes. Structural validity does not prove training or benefit.

Repository-test wheels omit `typing-inspection`, `inline-snapshot`, `dirty-equals`, and
`python-multipart`. Community collection failures do not establish the same defect in private scoring.
Before assigning rewards, require genuine unpatched failure and passing gold controls on the chosen task.
Keep `patch` and `test_patch` out of agent inference inputs.

The separate [evaluation wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)
lists 41 files, including patched `vllm 0.19.1`, `swegemma 0.2.7`, `adk-submission 0.2.12`,
`adk-eval-core 0.1.0`, `google-adk 1.36.1`, and `transformers 5.13.1`.
Pin dataset versions and file hashes because organizers may patch a release without changing its version.
Three small harness wheels were initially extracted for inspection; all 41 files were subsequently
verified, with separate native prechecks and container evaluation. The latter passed complete compilation,
real generation, and independent tests; see [runtime evidence](cluster-runtime.md).

Source inspection found adapter-count/rank-derived LoRA settings, thinking-budget/seed forwarding,
and graph-output truncation. Older forum bugs cannot automatically be treated as still open; source code
alone also cannot validate every runtime path. In `adk-submission 0.2.12`, `include_thoughts=False`
sets `enable_thinking=False`, changing generation rather than merely hiding text. See
`adk_submission/resolvers/generation.py`, the [community report](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745059),
and [wheel hashes](../resource/manifests/harness-wheels.json).

Further evidence: [Data description](../resource/competition/data-description.md),
[file list](../resource/manifests/competition-files.json), and [verification](../resource/manifests/download-verification.json).
The original `HARNESS_README.md` remains in the separately downloaded competition data.

## Code: 112 public entries

Competition-filtered pagination returned 100+12 entries. All downloads completed; three anonymous
401 responses were resolved through authenticated access, and one entry contained empty source.
Source, metadata, hashes, and readable text are retained. Reviews inspect structure, dependencies,
configuration, evaluation, training entry points, packaging, retrieval, and data quality.
No third-party notebook was executed or independently reproduced, and this is not a line-by-line security audit.

All **112 individual reading records** were completed on October 5, including the empty-source limitation.
The [reading notes](../resource/notebooks/reading-notes.md) record execution paths, default switches,
actual ZIP content, reusable ideas, caveats, and source lines; JSON retains source hashes.
Only logic included in the submitted agent ZIP runs during official scoring; surrounding notebook analysis may not.

Useful references:

- [Official starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent):
  offline wheels, sample configuration, vLLM, two-task local evaluation, packaging. Its global reinstalls
  and cutlass `.pth` deletion belong to its disposable notebook assumptions, not an HPC host.
- [Local evaluator](https://www.kaggle.com/code/hitarthjain0/gemma-4-agent-run-the-real-evaluator-locally):
  per-task traces, repeated calls, and failure diagnostics useful for experiment records.
- [Reported 0.13 configuration](https://www.kaggle.com/code/hsiaosuan/gemma-developer-agent-0-13-submission):
  read-only planner then coder, no LoRA, five minutes / 60 calls / 100 turns. The score is an author claim,
  not our reproduction or a causal architecture ablation. Decoded YAML sets `include_thoughts: false`
  in both phases, so nonzero thinking budgets do not imply thinking is enabled in this harness.
- GraphLoc, Wold coverage/localization, and code-graph-audit expose async, name-resolution, coverage,
  and localization problems. Use them as testable retrieval baselines with gold labels excluded from inference.
- [31B QLoRA study](https://www.kaggle.com/code/denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study):
  real training/export code requiring base/module/version, split, and platform compatibility checks.
- [TPU KerasHub tutorial](https://www.kaggle.com/code/nursrijan/gemma-4-devagent-kerashub-lora-training-on-tpu):
  fallback random tensors and string-renamed export layers do not establish competition adapter compatibility.
- [TFD-Agent](https://www.kaggle.com/code/rajrajak99/tfd-agent-autonomous-swe-bench-resolution-with-ge):
  architecture/file-operation simulation; its plotted 0.18 is a fixed estimate, not a measured run result.

Common pitfalls include obsolete hardlink fixes, mock validators presented as compilation,
10–20-minute task budgets that exceed global limits, and packaging presented as model execution.
Titles such as Top 1 or SOTA need current leaderboard and experiment evidence.

| Candidate direction | Source evidence | Next check |
| --- | --- | --- |
| Bounded source localization | Evidence Scout BM25/RRF and GraphLoc lexical/graph ablations | Fixed issue/source inputs against grep/lexical baselines; resolution and time |
| Fewer ineffective loops | Lean/reviewer/full profiles, repeated-call counts, counterexample probes | Matched budgets/tasks; prove probes reproduce the issue before testing repair |
| Reliable evaluation evidence | Shape of Doubt ZIP hashes, fixed-patch replay, endpoint health, JUnit | Isolate environment changes and retain failures so dependency fixes are not credited to prompts |
| SFT / LoRA | Real QLoRA entry and some diff-only masks | Compatible base, short training/resume, then cross-repository evaluation without gold localization |

These are untested team hypotheses. Specific source findings include:

- Some retrieval experiments restrict candidates to all gold files or use target embeddings/symbols as seeds;
  their metrics do not measure unknown-issue localization. Others align new-file diff lines to old-snapshot ASTs;
  those labels need correction. Some graph-rebuilder classes are defined but never used in the comparison.
- A TPU tutorial starts from Gemma 2 2B and relabels an export as Gemma 4 31B. Other examples emit random A,
  all-zero B, or invalid placeholder files. These do not establish trained, compatible competition LoRA.
- `RUN_HF_SMOKE=True` and `RUN_COMMAND_SCREEN=True` may contradict CPU-only/default-off prose.
  Conversely, a training function's presence does not prove that it ran.
- The `119 of 129 sound` statistics come from historical author audit tables/attached CSVs. The notebook
  mainly recomputes summaries/localization. The raw required-node surrogate is 117; 119 includes an
  interpretation of skips. Neither is our per-task reproduction or a private-scoring performance ceiling.
- Black Cat v13 retains a 0.12 score from its v10 anchor; the embedded v13 L4 pilot timed out during server
  startup and completed zero paired tasks. A new default candidate does not inherit the anchor's score.
- `complete EDA & ADK starter kit` writes reference `patch` directly into its final parquet. That is answer
  export, not agent prediction. Our submission remains an agent ZIP checked by official compilation and inference.

Per-entry sources and locations are in the [reading notes](../resource/notebooks/reading-notes.md).
None of these third-party candidates was added to the formal baseline or executed during review.

## Models: four entries in the captured page

`ListCompetitionModelInstances` returned no further page token.

| Instance | Version / package size | Interpretation |
| --- | --- | --- |
| `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct` | v2; 23,297,590,856 bytes, about 21.70 GiB | Required base; budget the whole package, not only theoretical weight estimates |
| `jek1wantaufik/buddy/Transformers/gemma4` | v1; 8,989 bytes | ModelBuddy references an agent bundle; not permission to replace the base |
| `mousumiroy250/mt5-base/Transformers/default` | v1; about 6.99 GB | Attached entry without a body model card; not an approved substitute |
| `muhammadukasha09/bert-base-multilingual-cased/Transformers/default` | v1; about 717 MB | Attached entry without a body model card; not an approved substitute |

Metadata, instance descriptions, and available cards were retained. Only the required Gemma weights were
downloaded later. Family-card 128K/256K capabilities do not override the **32,768-token** competition limit.

**Update, 2026-10-11.** A [host announcement](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287) added `gemma-4-12b-it-qat-w4a16-ct`,
`gemma-4-e4b-it-qat-w4a16-ct` and `gemma-4-e2b-it-qat-w4a16-ct` as allowed bases. Every agent and sub-agent in one
submission must use the same variant, scoring conditions are unchanged, and adapters for the smaller variants
should be trained on the unquantized checkpoints. The 12B also needs the host's `--hf-overrides` serving flag.
As of 2026-10-16 the Model Selection page still listed only 31B (the host agreed to update it), and two teams
reported that otherwise identical 12B submissions ended in "Notebook Threw Exception" while 31B scored
([discussion 749259](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749259)). The team
therefore keeps 31B for submissions; a local 12B smoke with the baseline prompt resolved 1/10 fast-cohort tasks
and exhausted its tool budget early ([#24](https://github.com/jwj1342/gemma-agent-lab/issues/24)).
Models-page `bestPublicScore` describes its associations, not the entire leaderboard maximum.
Sources: [Models](https://www.kaggle.com/competitions/gemma-4-developer-agent/models) and
[page inventory](../resource/models/competition-tab.json).

## Discussion: 116 topics and 370 messages

Opening posts and complete reply trees were saved with authors/timestamps and matched API counts.
See the [discussion index](../resource/discussions/README.md). External links, image text, Discord,
and deleted/private content are outside this coverage claim.

Organizer clarifications reviewed:

- [ADK version](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743800):
  use the competition wheelhouse; ADK 2.x is unsupported and participants cannot modify scoring infrastructure.
- [LoRA fix](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743508):
  organizers reported the zeroed-adapter issue fixed in wheelhouse v23.
- [October 1 status](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744794):
  organizers reported dynamic adapter/rank KV-cache allocation and fixes for double JSON, lost thoughts,
  thinking-budget forwarding, and seed forwarding. Old fixed max_loras=8/max_rank=128 prose does not
  describe the dynamic policy; still verify changed outputs with adapters enabled/disabled.
- [read_file types](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744678):
  organizers reported a wheelhouse fix.
- [Public task quality](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743817):
  public tasks received less review than hidden tasks. Hidden-gold guarantees do not validate our public local environment.
- [Graph files](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911):
  empty embedding packaging was fixed; missing async functions were still unresolved at that point.
- [External teacher outputs](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807):
  permitted subject to model licenses and competition rules; record source, license, and generation process.
- [Custom tools](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743573):
  package scripts as sandbox skills rather than arbitrary host-registry entries.

Continue tracking global-timeout recovery, compaction losing tool results, undeclared tools,
missing public-test dependencies, ambiguous issues such as `fastapi_15661`, chat templates, and L4 queues.
Some reports are unconfirmed community observations rather than established current official bugs.

## Leaderboard distribution

Official CSV: **October 5, 2026, 03:50:43 UTC**. There are **1,755 scored teams**, maximum **0.24**,
median **0.08**, and mean approximately **0.08199**. The leading team includes `samson8`; ranks 2–7
all display 0.17. Values retain the CSV's precision.

| Displayed score | Teams |
| --- | ---: |
| 0.24 | 1 |
| 0.17 | 6 |
| 0.15 | 41 |
| 0.13 | 139 |
| 0.12 | 307 |
| 0.10 | 344 |
| 0.08 | 316 |
| 0.06 | 200 |
| 0.05 | 148 |
| 0.03 | 65 |
| 0.01 | 55 |
| 0.00 | 133 |

Many teams tie. A few tasks can change rank substantially. Community inference of 58 public tasks
from score steps is not official confirmation; organizers specified approximately 120 total with a
roughly equal split. Public rank does not determine final private performance.
Sources: [leaderboard](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard)
and [statistics](../resource/leaderboard/summary.json).

## What has been established

The data package, main text pages, and enumerated public resources were collected; neither Overview
nor Data description was blocked. Binary assets received inventory/format checks, not a line-by-line
review of all repository snapshots. Public notebooks received structure review and individual execution-path
notes, not full security audits or execution reproductions. External projects and image attachments remain incomplete.

Subsequent work verified the required model and 41 wheel files, built the no-LoRA baseline, and matched
local ZIP bytes against the private CPU notebook. The container, separate environment, and Git tools
supported completed job 1033882: compilation, offline GPU generation, valid empty/gold controls, and
84 passing agent-patch tests on `rich_3454`. Training and full local benchmarking remain unverified.

The October 5 18:14 UTC query confirmed submission **56844353 COMPLETE**, public score **0.12**,
private score unavailable. L4×4 single-task verification also completed with valid controls and 84 passing
tests; the agent phase took 74.10 seconds excluding startup. This was the existing submission's returned
result, not a duplicate upload. See [official result](../resource/manifests/competition-result.json),
[L4 evidence](../resource/manifests/kaggle-runtime-smoke.json), [submission steps](submission.md),
[runtime adaptation](cluster-runtime.md), and [asset verification](../resource/manifests/eval-assets-verification.json).
