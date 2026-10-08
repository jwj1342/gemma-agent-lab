# CodingAI Work Log

This file is the persistent handoff record for work performed in this repository. Read it before starting a task and update it before finishing.

## Current project state

- The canonical remote is `origin` at `https://github.com/jwj1342/gemma-agent-lab.git`; future references to the project remote mean this repository unless the user says otherwise. `origin/main` is `3054da842f794c854f438a5478ea9437b119e9bc`.
- Local `main` is based directly on `origin/main` and is one compatibility commit ahead: `88823c0f556874608cd993cd37ba32d94df7bbd8` renames one leaderboard CSV whose colon-containing timestamp cannot be checked out on Windows. The blob content is unchanged and nothing has been pushed.
- Active work is on `feature/localization-v1-skill`, branched from `integration/local-work`, for the reviewed migration and full-tree validation of Localization v1. The remote baseline remains unchanged.
- The October 8 preservation handoff is in `docs/localization-handoff.md`, linked from the root README. It explains baseline localization, the implemented optional module, offline gains, rollback, local artifacts, and pending manual remote confirmation; no background GPU work was started.
- The remote baseline lives under `configs/baseline/` and uses the required Gemma 4 model, a main coding agent, and a read-only `code_analyzer` sub-agent. Remote packaging, validation, Kaggle/HPC preparation, CI, documentation, and resource snapshots are now present locally.
- All pre-remote versionable work is protected by branch `archive/local-before-remote-2026-10-07` at `2ba0f75448f16bdd41281572fa0969641ccc44c7` and annotated tag `local-before-remote-2026-10-07`. The original root commit remains its parent.
- A second filesystem backup excluding `.git` is at `E:\Desktop\CodingAI-local-backup-20261007-1818`; verification counted 106 files and 23,190,078 bytes. It includes ignored datasets, Kaggle results, ZIPs, caches, source, prompts, tests, and model artifacts as they existed before branch replacement.
- Archived work includes Programming v1, Graph Agent v1, Kaggle A/B experiments, Localization v1/v2/v3/v3.1/RL/v4 source and artifacts, package tests, and the full work history. Localization v1 is the first module migrated into an isolated remote-framework experiment; none of the experimental modules has replaced the baseline.
- Retained Localization v1 still means the archived v3 checkpoint (`artifacts/localization_v3_3k/`, best epoch 1,850). Programming v1 remains an unpromoted ablation after a 0/2 paired smoke result and should not replace the remote prompt without a new matched comparison.
- Localization v1 remains isolated under `configs/experiments/localization-v1/`. A portable content-based refinement now learns seven evidence coefficients and a confidence gate with nested refitting of both stages. It improves retrospective test Hit@1/Hit@5/MRR from BM25's 31.6%/68.4%/0.473 to 47.4%/78.9%/0.624; nested snapshot and repository validation also improve. Explicit `portable` mode runs the learned gate without the archived checkpoint. Default adaptive mode remains BM25 pending fresh confirmation and remote repair evidence. Prior gate-only CV reused a ranker trained on those tasks and is now explicitly qualified; it cannot justify promotion.
- A frozen three-arm remote repair protocol is prepared at `runs/2026-10-08-localization-v1-portable/remote-ab-final/`: 38 tasks, 114 baseline/BM25/portable cases, and 18 pilot notebooks over six seeded tasks. No GPU inference or upload has run. The portable experimental ZIP is `runs/2026-10-08-localization-v1-portable/submission-localization-v1-portable.zip`, SHA256 `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48`. The remote baseline still has no trained localization adapter and remains unchanged.
- Public task data and partial-clone source repositories remain locally available; archived artifacts, results, and ZIPs remain recoverable from the external backup. Generated paths such as `artifacts/`, `data/public/`, `dist/`, `results/`, and `runs/` are excluded by `.gitignore`; source/configuration migrations are reviewed separately.
- `WORKLOG.md` remains the persistent handoff record. `AGENTS.md` combines the remote collaboration rules with the mandatory work-log procedure.

## Archived pre-remote project state

- Git remote `origin` is `https://github.com/jwj1342/gemma-agent-lab.git`; future references to the project remote mean this repository unless the user says otherwise. `origin/main` was last fetched at `3054da842f794c854f438a5478ea9437b119e9bc` on 2026-10-07.
- Local `main` and `origin/main` have unrelated root histories (local: one commit; remote: five commits), and the working tree contains substantial uncommitted local experiments. The remote has been fetched into the object database only; it has not been merged, rebased, checked out, or set as the upstream of local `main`.
- The project is a minimal, declarative single-agent baseline for the Gemma 4 Developer Agent Kaggle competition.
- `submission/agent.yaml` configures `gemma-4-31b-it-qat-w4a16-ct` and all nine competition harness tools.
- The agent prompt supports the basic coding loop: inspect an issue, locate code, make a minimal source change, run targeted verification, inspect the diff, and submit a patch.
- The current programming capability is not a standalone locally trained module: the official Harness runs Gemma 4 as a single tool-using agent, while this repository supplies its declarative configuration, prompt policy, sampling settings, and packaging checks.
- `scripts/package_submission.py` validates and packages the contents of `submission/` into `dist/submission.zip`.
- The repository does not currently contain a standalone local coding-agent runtime, model server, interactive UI, or end-to-end harness test. Actual coding execution depends on the Kaggle harness/official notebook.
- Static package, localizer, and reward tests currently pass: 20 tests successful on 2026-10-07.
- `submission/prompts/system.md` now contains Programming v1: the retained embedding-retrieval policy plus a compact seven-field pre-edit plan, a disproof check, and evidence-driven recovery after failed verification.
- The preserved baseline is `dist/submission.zip`; retrieval-only control is `dist/submission-retrieval-v1.zip`; the current `submission/` working source is packaged separately as `dist/submission-programming-v1.zip`. The Programming v1 archive differs from the retrieval control only in `prompts/system.md`.
- Kaggle authentication is configured locally and was validated without logging the credential; the account has joined the target competition.
- Private Kaggle Notebook version 1 (`codingai-retrieval-a-b-smoke-test`) completed its two-task baseline/retrieval smoke test with 0/2 resolved for both variants; it was not submitted to the competition.
- Private Kaggle Notebook version 1 (`codingai-programming-v1-a-b-smoke-test`) completed a paired two-task retrieval-control versus Programming v1 evaluation; both variants resolved 0/2, so Programming v1 is not promoted. No competition submission was created.
- Graph Agent v1 is implemented under `experiments/graph_agent/submission/` and packaged as `dist/submission-graph-agent-v1.zip`; it has not been pushed to Kaggle.
- A dependency-free pairwise file-localization ranker is implemented in `scripts/train_localizer.py`; its 1000-epoch public-data run is saved under `artifacts/localization_v1/` with epoch 50 selected by development MRR.
- Localizer v2 reduces the label-specific `issue_path` feature to 0.25, raises L2 regularization to `1e-4`, and selects checkpoints using equal-weight development Hit@1 and Hit@5 with MRR as a tie-breaker. Its completed 10,000-epoch run is saved under `artifacts/localization_v2_10k/`; epoch 450 was selected, but its held-out results did not beat v1, so v1 remains the stronger experimental checkpoint.
- Localizer v3 completely removes `issue_path` and applies exact global decoupled weight decay through lazy scaling. Its completed 3,000-epoch run is saved under `artifacts/localization_v3_3k/`; epoch 1,850 was selected and currently has the strongest held-out Hit@3, Hit@5, and MRR among the localizer variants, though the test split remains only 19 tasks.
- Per the user's explicit decision, the experiment formerly named localizer v3 is retained as product-level **Localization v1**. Its existing `artifacts/localization_v3_3k/` path is preserved for lineage and because `artifacts/localization_v1/` already names an older experiment; neither artifact may be overwritten. The retained locator is not yet wired into the competition Agent runtime.
- V3.1 is a protected continuation experiment under `artifacts/localization_v3_1_additional_3k/`: it ran 3,000 additional epochs from v3's epoch-1,850 checkpoint but found no better development checkpoint, so its selected epoch 0 weights are identical to v3 and v3 remains current.
- An experimental two-head contextual-bandit/RL localizer is implemented in `scripts/train_rl_localizer.py`: a count policy predicts 1–26 returned files and a residual file-ranking policy sits on frozen v3. The first 300-epoch pilot under `artifacts/localization_rl_v1/` overfit training reward and selected epoch 0, so it is not promoted and v3 remains current.
- Localizer v4 extends the RL prototype with a configurable squared-component reward, defaulting to `0.65 * count_match^2 + 0.35 * location_f1^2`. Its completed 300-epoch run is under `artifacts/localization_v4_300/`; it also selected epoch 0 and did not improve the directly comparable exact-count, exact-set, or location-F1 metrics, so v3 remains current.
- The localizer benchmark is closed-world: candidates are the 138 distinct non-test Python paths found in public reference patches rather than complete repository trees, so its metrics must not be treated as end-to-end agent accuracy.
- Programming v1 is implemented as a prompt-only planning-gate experiment with localization and all runtime settings fixed. Its paired smoke test found no resolution gain, only 1/2 complete plan compliance, higher latency and edit churn, and fewer failed tool calls; keep it as an ablation and retain retrieval v1 as the control.

## Work history

### 2026-10-08 17:51 (Asia/Shanghai) - Preserve the localization result and explain the remote baseline

- Objective: preserve successful optimization while the user is away, and provide an understandable account of the existing remote localization mechanisms and modular optimization path.
- Decision: retain the measured offline gains and implemented optional portable module, without claiming an unmeasured remote repair gain or replacing the baseline. The module is already implemented; no further tuning against the historical test was performed.
- Work: inspected the actual baseline main/analyzer configurations and prompts; wrote `docs/localization-handoff.md` describing issue extraction, source search, graph queries, read-only analysis, the learned ranker/gate hook, BM25 fallback, and source-confirmation requirements. Added a root README entry so the handoff is easy to find.
- Preservation: kept the final study, label-containing training cache, frozen 114-case protocol, 18 pilot notebooks, and experimental ZIP in their existing ignored local run directory. Rechecked ZIP SHA256 as `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48`. These local outputs are not a remote backup. Source, frozen policy, tests, and experiment reports are included in the local feature-branch checkpoint containing this entry; no push or merge is authorized by this handoff.
- Verification: documentation-only additions leave runtime and frozen packages unchanged. Prior focused tests passed 25/25; full suite was 39/46 with seven documented Windows-specific expectations. Final repository checks passed (33 Python, 10 shell, 16 configuration, 50 Markdown files, two agent packages); staged whitespace checks passed. The baseline directory has no staged changes.
- Changed files for this handoff: `README.md`, `docs/localization-handoff.md`, and `WORKLOG.md`; the preceding staged implementation is preserved as the same checkpoint.
- Follow-up: manual quota review and controlled remote evaluation are required before any promotion; the six-task pilot cannot satisfy the full 38-task rule. No GPU job, upload, external deployment, or automatic background continuation was started.

### 2026-10-08 17:43 (Asia/Shanghai) - Learn a portable v1 ranker and prepare paired remote confirmation

- Objective: optimize the ranking algorithm, its parameters, and the learning gate to produce a measurable improvement and prepare a direct comparison against the remote Gemma baseline.
- Important correction: the archived legacy ranker already trained on the 91 training tasks used by the previous gate study. Cross-fitting only its gate therefore did not measure whole-system generalization. Added explicit false promotion requirements for ranker cross-fitting, fresh confirmation, and remote paired repair evidence to the legacy trainer; qualified the historical large-sample report without altering its recorded metrics.
- Implementation: added standard-library portable source/query/path/symbol features and a small nonnegative ranker to the existing skill. The archived checkpoint is preserved and is not loaded by portable mode. `scripts/train_portable_localization.py` learns six coefficients around a fixed BM25 coefficient, with a repository-balanced objective, two coordinate passes over a fixed grid, three-fold out-of-fold gate calibration, nested five-fold snapshot validation, and nested leave-one-repository-out validation. Shared base commits cannot cross fold boundaries. Runtime and training share feature, scoring, and gate functions.
- Frozen parameters: weights `[1, 2, 0, 2, 2, 0, 1]` for BM25, deduplicated query, weak length normalization, path, symbols, exact definitions, and implementation indicator. The gate accepts candidate ranking when BM25 margin is at most 0.6 and candidate margin is at least 0.025. Parameters were selected from training only, frozen before development/test scoring, and reproduced in a second complete run. There were 40 ranker fits per study; final wall time was 74.120 seconds on Windows/Python 3.14.6, CPU and standard library only, seed 52.
- Results: nested snapshot BM25 versus gated candidate Hit@1/Hit@5/MRR is 23.1%/54.9%/0.384 versus 41.8%/74.7%/0.555; nested repository results are 39.6%/72.5%/0.526 for the candidate. Development is 38.9%/55.6%/0.466 versus 38.9%/83.3%/0.549. Historical test is 31.6%/68.4%/0.473 versus 47.4%/78.9%/0.624. The gate activates on 17/18 development and 14/19 test tasks.
- Gate tradeoff: ungated development Hit@1 is 33.3%, so gating restores one correct first choice. Ungated test Hit@1/Hit@5/MRR is 57.9%/84.2%/0.705, higher than gated test performance; the conservative gate sacrifices some gains. No further selection was performed from test results. Test bootstrap Hit@1 and MRR delta intervals exclude zero, but Hit@5 does not; development uncertainty and historical data exposure prevent an independent-confirmation claim.
- Runtime audit: all 37 scorable development/test tasks produced identical ranks and activation decisions through the packaged runtime and learner. Median local ranking time, excluding source scan and AST construction, was 0.248 seconds portable versus 0.271 seconds through the current BM25 wrapper; this is not L4 latency. Explicit `portable` mode runs the candidate for experiments. `portable-policy.json` remains unpromoted, and default adaptive execution stays BM25.
- Remote handoff: added `prepare_localization_ab.py` and `summarize_localization_ab.py`. Prepared 38 task IDs and 114 frozen cases, with the same model, sampling seed 42, analyzer, and evaluation budgets. Baseline/BM25/portable arm order rotates. Six seeded pilot tasks have 18 private offline notebooks; all 114 case hashes and notebook code cells were independently checked. The BM25 and portable arms differ only in the explicit skill mode. The protocol requires at least five points resolution gain over baseline, one-sided exact paired McNemar p <= 0.05, positive gain over the BM25 arm, <= 10% mean agent-time increase, and complete valid controls/pipelines. Missing and invalid cases remain in the denominator and block promotion.
- Cross-platform fix: `prepare_eval_run.py` and the generated Kaggle setup now write public tasks as explicit UTF-8/LF bytes. This fixes a real local-to-Kaggle task hash mismatch without changing agent behavior. Earlier ignored preparation directories are retained; only `remote-ab-final` is the final handoff.
- Verification: 25 focused tests passed. Full suite: 39/46 passed; four fixed Linux/Python baseline ZIP-hash expectations and three POSIX-path assertions still fail on Windows. The task-hash test now gets past frozen-input verification and reaches its pre-existing ZIP-hash assertion. New checks cover ranker/gate fold isolation, shared commits, runtime/training parity, disabled-policy fallback, malformed activation values, and paired failure denominators. After staging, repository checks passed 33 Python, 10 shell, 16 configuration, 49 Markdown, and 2 agent-package checks; staged whitespace checks passed. Both staged and unstaged baseline diffs remained empty.
- Artifacts: final study and exact source/data hashes are in `runs/2026-10-08-localization-v1-portable/study-final/`; reproducible feature cache and earlier study are preserved alongside it. Portable experimental ZIP is 726,506 bytes (4,112,735 unpacked), SHA256 `e278af4648c25f95c264a1ca81cf592ca07012681b877aaeb01fa8ee1b514c48`. Final baseline arm SHA is `aa597ea39b66412eb53fcd3cd2fcc470c0650853a4ebe69f73d777f0ba7e822c`; BM25 arm SHA is `cb1aa5427f2ab42044bb1029371860da4476872774b9176aec3a11c3d3e746b7`. Recovery commands and all limitations are in `docs/experiments/2026-10-08-localization-v1-portable.md`.
- Changed files: portable runtime/policy, portable trainer, legacy trainer promotion guard, paired preparation/summary scripts, two new test modules, task-byte fixes in the existing evaluation/Kaggle preparers, skill instructions, experiment report/navigation, and this log.
- External state and remaining boundary: no GPU/model run, Kaggle upload/submission, credential change, push, PR, merge, baseline modification, or archived-checkpoint overwrite. The laptop lacks official evaluation packages and required model weights. GPU runs remain manual under AGENTS.md. Positive remote repair improvement is still unmeasured; launch the frozen integration pilot after quota review, then complete the predeclared comparison before promotion.

### 2026-10-08 - Expand Localization v1 gate learning and retain the safety fallback

- Objective: perform substantially more learning for Localization v1 and ensure that v1 is activated only when evidence supports a reliable improvement over full-tree BM25.
- Protocol expansion: changed `scripts/train_localization_gate.py` from fitting on 18 development tasks to fitting on 91 training tasks, with deterministic five-fold repository-stratified out-of-fold evaluation, leave-one-repository-out evaluation, frozen confirmation on 18 scorable development tasks, final veto on 19 holdout tasks, and 5,000 paired repository-stratified bootstrap samples on each confirmation split. Runtime features remain label-free, and holdout never selects a rule.
- Learned rule: the training-selected depth-two rule enables the fixed v1 hybrid when BM25 margin is at most `0.36349847748766084` and BM25 positive-score fraction is greater than `0.12810649472405317`. It activates on 86/91 training tasks, 18/18 development tasks, and 18/19 holdout tasks.
- Training evidence: BM25 versus fitted gate is Hit@1 23.1% versus 52.7%, Hit@5 54.9% versus 78.0%, and MRR 0.384 versus 0.640. Five-fold out-of-fold gate metrics are 47.3%, 73.6%, and 0.595; leave-one-repository-out metrics are 48.4%, 74.7%, and 0.603, all above the same BM25 training baseline.
- Frozen evidence: development improves Hit@1 from 38.9% to 50.0%, Hit@5 from 55.6% to 77.8%, and MRR from 0.466 to 0.612. Holdout reverses this pattern: Hit@1 falls from 31.6% to 15.8%, Hit@5 remains 68.4%, and MRR falls from 0.473 to 0.350. Development has 10 wins / 6 ties / 2 losses; holdout has 6 / 3 / 10.
- Uncertainty: development bootstrap deltas are Hit@1 +11.1 points with 95% interval [-11.1, +33.3], Hit@5 +22.2 [+5.6, +44.4], and MRR +0.147 [+0.012, +0.294]. Holdout deltas are Hit@1 -15.8 [-36.8, +5.3], Hit@5 0.0 [-26.3, +26.3], and MRR -0.123 [-0.285, +0.036]. The confidence and holdout promotion checks therefore fail.
- Safety decision: updated the versioned rule and evidence in `resources/gate.json` but kept `enabled: false`. Default adaptive runtime remains an exact BM25 alias and does not load the legacy checkpoint. More fitting on the same public labels cannot ensure v1 generalization; new repositories or untouched tasks are required.
- Verification: all 14 focused localization/gate tests passed outside the restrictive Windows temporary-directory sandbox. The full 35-test suite passed 28 tests; the same four Windows/Python-3.14 newline/ZIP-hash assumptions and three POSIX-path assertions failed. The new large-learning and bootstrap tests passed. After staging, `scripts/check_repository.py` passed 28 Python, 10 shell, 15 configuration, 48 Markdown, and 2 agent-package checks. Repeated packaging produced the same ZIP hash, and the baseline staged diff remained empty.
- Artifacts: `runs/2026-10-08-localization-v1-large-learning/gate-study.json`, generated gate JSON, and `submission-localization-v1-adaptive.zip`. The ZIP is 725,248 bytes (4,107,258 unpacked), SHA256 `b9750171abfa09139b9f5ba80fcebd6d5dc5746c30bd0f3af4243e0696efaf16`.
- Files changed: expanded gate trainer and tests, updated versioned gate and skill explanation, added `docs/experiments/2026-10-08-localization-v1-large-learning.md`, updated experiment/config/script navigation, marked the initial gate report as superseded for deployment evidence, and updated this work log.
- External state: no GPU/model run, Kaggle action, credential change, upload, competition submission, push, PR, merge, or baseline change was performed.
- Follow-up: obtain genuinely new tasks or repositories, train a full-tree portable ranker rather than the archived closed-world path ranker, and require untouched confirmation plus matched end-to-end repair A/B before enabling v1.

### 2026-10-08 - Learn and integrate a fail-safe Localization v1 gate

- Objective: write a learning program that studies development-versus-holdout differences and selectively enables or disables the v1 hybrid so it can complement the BM25 fallback.
- Learning program: added `scripts/train_localization_gate.py`, a dependency-free learner over 12 label-free runtime features: repository size, issue length, BM25 and legacy margins/positive fractions, exact evidence, and BM25/v1 ranking agreement. It fits at most two conjunctive thresholds on development labels, reports feature distribution shift and per-task outcomes, performs leave-one-task-out and leave-one-repository-out validation, freezes the rule before holdout scoring, and emits both a full study and a reviewable gate JSON.
- Leakage boundary: reference-patch paths are used only after ranking to fit the development rule and score evaluation. No gold path enters runtime features or the skill. The designated holdout had already been inspected at aggregate level in the previous experiment, so this gate study is explicitly retrospective rather than a new independent confirmation; holdout can veto deployment but never selects the rule.
- Learned rule: enable the fixed v1 hybrid only when BM25's positive-score fraction is at most `0.9879032258064516` and the strongest exact score is at most `11.75`. On 18 scorable development tasks it enabled 12, improving Hit@1 from 38.9% to 55.6%, Hit@5 from 55.6% to 77.8%, and MRR from 0.466 to 0.641. Leave-one-task-out and leave-one-repository-out also exceeded the development BM25 baseline.
- Holdout finding and safety decision: the same frozen rule enabled 10/19 tasks but regressed Hit@1 from 31.6% to 26.3%, Hit@5 from 68.4% to 63.2%, and MRR from 0.473 to 0.440. The candidate changes from 10 wins / 6 ties / 2 losses on development to 6 / 3 / 10 on holdout. FastAPI alone changes from 6 wins / 0 losses to 2 wins / 6 losses, showing within-repository task shift rather than only repository-mixture shift. The largest label-free shifts are BM25 margin (+0.501 standardized difference), legacy positive fraction (+0.488), and reduced BM25/v1 ranking agreement.
- Runtime integration: added validated `resources/gate.json` and an `adaptive` localizer mode. An enabled gate computes exactly the same features as training and chooses BM25 or v1 per issue. The current gate records the learned rule but sets `enabled: false`; adaptive mode therefore aliases BM25 and does not load the 4 MB checkpoint. A future independently validated gate can activate v1 without source changes.
- Verification: the final gate study is at `runs/2026-10-08-localization-v1-gate/gate-study-final.json`. All 12 focused localization/gate tests passed. With process-local UTF-8 and Git Bash on PATH, `python scripts/check_repository.py` passed 28 Python, 10 shell, 15 configuration, 47 Markdown, and 2 agent-package checks. The full 33-test suite passed 26 tests; the same four Windows/Python-3.14 newline/ZIP-hash assumptions and three POSIX-path assertions failed, with no new functional failure.
- Package: `runs/2026-10-08-localization-v1-gate/submission-localization-v1-adaptive.zip` is 725,149 bytes (4,106,980 unpacked), SHA256 `fbf74428030892b416fe23b2f56b6fa1dcbc892c4494c1ab4ac3728cef5b75c0`; deterministic build and CRC checks passed.
- Files changed: `scripts/train_localization_gate.py`, `tests/test_localization_gate.py`, adaptive-gate additions to the packaged localizer and its tests, `resources/gate.json`, the empty-repository-set fix in `scripts/benchmark_localization_v1.py`, `docs/experiments/2026-10-08-localization-v1-gate.md`, related navigation/report updates, and this work log.
- External state: no GPU/model run, Kaggle action, credential change, upload, competition submission, push, PR, or merge was performed.
- Follow-up: collect a larger grouped calibration set from additional repositories or untouched tasks. Do not continue optimizing against the same 19-task holdout and claim independence. Re-enable v1 only after a new gate passes grouped validation, an untouched confirmation set, official compilation, and matched end-to-end repair A/B.

### 2026-10-08 - Integrate and validate Localization v1 against complete source trees

- Objective: modify and complete Localization v1 so it can be inserted into the current remote agent framework and demonstrate a clear localization improvement without changing the measured baseline.
- Integration: added an isolated agent variant under `configs/experiments/localization-v1/`. The skill is attached directly to the root agent with the official root-relative `skills/localization-v1` reference. The prompt runs it once only when no target file is explicit, verifies the top three candidates from source, and preserves exact search, graph tools, and the read-only analyzer as fallbacks. The baseline model, analyzer configuration, sampling settings, and evaluation budgets remain unchanged.
- Runtime implementation: added a standard-library Python source scanner, BM25 file ranker, AST symbol hints, exact-score and archived-checkpoint diagnostic modes, bounded output, test-file exclusion, symlink exclusion, checkpoint validation, and signed-score-aware RRF. The packaged checkpoint is byte-identical to the protected archived model (SHA256 `56bec1d839b79ecaf9374a7f6da1e5cb496870e87c51750d2284d10f569cfdbc`); repository-specific memorized priors are disabled.
- Full-tree benchmark: evaluated 129 public tasks at their exact base commits with 21--789 non-test Python candidates per snapshot. The deterministic original split remained 91 train / 19 development / 19 test, seed 52, and reference patches were used only after ranking. Benchmark score caching reduced the final run to 26.828 seconds.
- Promotion decision: the development-selected hybrid overfit and was rejected (test Hit@1 15.8%, Hit@5 68.4%, MRR 0.350); BM25 plus exact with legacy disabled was also rejected (26.3%, 57.9%, 0.402). Full-tree BM25 is the safe default and clearly beats the archived legacy-only checkpoint on the held-out test: Hit@1 31.6% vs 21.1%, Hit@3 47.4% vs 36.8%, Hit@5 68.4% vs 42.1%, Hit@10 78.9% vs 52.6%, and MRR 0.473 vs 0.306. This is a localization improvement, not yet an end-to-end repair-resolution claim.
- Verification: all 6 focused Localization v1 tests passed outside the restrictive Windows temp-directory sandbox. With process-local UTF-8 and Git Bash on PATH, `python scripts/check_repository.py` passed 26 Python, 10 shell, 14 configuration, 46 Markdown, and 2 reproducibly packaged agent checks. The full 27-test suite passed 20 tests; the four Windows/Python-3.14 newline/ZIP-hash failures and three POSIX-path assertion failures are the same documented platform-sensitive assumptions, while all new tests passed. The official compiler script could not start in the laptop development environment because `litellm` is intentionally absent; no evaluation dependencies were mixed into it.
- Package: `runs/2026-10-08-localization-v1/submission-localization-v1.zip` is 722,599 bytes (4,097,771 unpacked), SHA256 `ed8a6df2da228a4d0b946027c555c45d6379d663e0c299a4a675f903b1989ed7`; local build, deterministic archive, and CRC checks passed.
- Files changed: new experiment bundle under `configs/experiments/localization-v1/`, `scripts/benchmark_localization_v1.py`, `tests/test_localization_v1.py`, `docs/experiments/2026-10-08-localization-v1.md`, navigation updates in `configs/README.md`, `scripts/README.md`, and `docs/experiments/README.md`, plus this work log. Local ignored benchmark/package artifacts are under `runs/2026-10-08-localization-v1/`.
- External state: fetched missing public GitHub blob objects needed by the local partial clones after explicit approval. No Kaggle run, GPU run, credential change, upload, competition submission, push, PR, or merge was performed.
- Follow-up: compile and run a matched baseline-versus-Localization-v1 repair A/B in the prepared Kaggle or H100 environment. Require valid controls, end-to-end resolution improvement, CPU CI, and one teammate approval before promoting the experiment into `configs/baseline/`.

### 2026-10-07 - Assess Localization v1 portability into the remote baseline

- Objective: determine whether the retained local Localization v1 can be moved into the fetched remote model/agent and whether the remote already contains a similar design.
- Local model finding: Localization v1 is a separate dependency-free hashed pairwise file ranker, not a Gemma adapter. Its protected checkpoint has dimension 262,144, 81,030 sparse nonzero weights, best epoch 1,850, and a 138-path candidate catalog covering only FastAPI, Rich, Requests, and HTTPX. It uses issue/path token pairs, lexical/path mentions, path depth, repository-prefixed path-token weights, and repository/path priors; the label-specific `issue_path` feature was removed in v3.
- Remote baseline finding: `configs/baseline/` uses the unchanged `gemma-4-31b-it-qat-w4a16-ct` base with no adapter. Localization is controlled by the prompt and the read-only `code_analyzer` sub-agent, using `run_command`, `read_file`, `search_similar_code`, `get_code_neighbors`, and `get_code_subgraph`. `search_similar_code` queries precomputed 256-dimensional symbol embeddings and is not a general learned issue-to-file ranker.
- Similar-design finding: the remote reference collection contains several unactivated third-party localization designs, including TF-IDF/BM25 and hashing rankers, GraphLoc BM25 plus graph PageRank as an ADK skill, GemmaLoc candidate reranking, graph propagation/RRF, and coder/localizer sub-agent layouts. These are research snapshots and reading notes, not part of the formal baseline. GraphLoc's packaged skill is the closest execution precedent; the active analyzer sub-agent is the closest architectural analogue.
- Portability conclusion: direct weight fusion into Gemma is not possible because the checkpoint is an independent linear scorer with a different parameter space. Tool-side integration is technically feasible: the official schema accepts packaged `skills`, `run_skill_script` can execute a standard-library localizer in the task repository, and the 3.75 MiB Git checkpoint is below the 50 MiB per-skill limit.
- Required adaptation before activation: enumerate candidate Python source files from the current task repository instead of the checkpoint's public-gold candidate catalog; define behavior for unseen repository-prefixed features; emit bounded top-k paths; preserve exact-search/BM25 fallback; and package it in `configs/experiments/localization-v1/` rather than altering `configs/baseline/`.
- Validation boundary: the current 47.4%/73.7%/78.9% held-out Hit@1/3/5 and 0.619 MRR came from a 19-task closed-world split whose candidate universe was built from public gold paths. It does not establish full-tree, unseen-repository, hidden-task, or end-to-end repair benefit. Require full-tree cross-repository localization followed by matched official baseline-versus-skill repair evaluation before promotion.
- Verification: inspected the protected trainer, unit tests, checkpoint metadata and metrics; inspected the remote agent/analyzer configuration, harness graph-tool implementation, skill schema/resolver and size limits, data guide, research report, and localization notebook reviews.
- Files changed: `WORKLOG.md` only. No source, checkpoint, Agent configuration, prompt, remote branch, Kaggle resource, or model weight was changed.
- Follow-up: if approved, create an isolated Localization v1 skill experiment with full-tree candidate enumeration and tests, keeping the remote baseline byte-for-byte unchanged.

### 2026-10-07 - Protect local work and adopt the remote baseline

- Objective: follow the agreed migration plan while ensuring every pre-existing local source file, experiment, model artifact, ignored dataset, result, and archive remains recoverable.
- Filesystem protection: copied the complete working directory except `.git` to `E:\Desktop\CodingAI-local-backup-20261007-1818`. A source/destination count-and-size check matched 106 files and 23,190,078 bytes. The copy includes the 53 Git-ignored files (3,187,773 bytes) that could not be protected by a normal commit.
- Git protection: created `archive/local-before-remote-2026-10-07` and committed all 43 changed/new versionable paths as `2ba0f75448f16bdd41281572fa0969641ccc44c7`; created annotated tag `local-before-remote-2026-10-07` at the same commit. No credential-like filename was staged. The original local root remains the snapshot commit's parent.
- Remote adoption: moved local `main` onto the remote history and configured `origin/main` as its upstream. Direct checkout exposed one remote path that is invalid on Windows because its timestamp contains colons. Local `main` therefore adds exactly one child commit, `88823c0f556874608cd993cd37ba32d94df7bbd8`, renaming that CSV timestamp from `03:50:43` to `03-50-43`; Git detected a 100% rename and the blob content is unchanged.
- Validation safeguard: an initial low-level tree construction introduced carriage returns into tree entry names. Diff validation caught the broad rename set before checkout, so that malformed commit was rejected, never used as a final branch/tag target, and never pushed. The replacement commit was accepted only after its diff contained exactly the intended single-file rename.
- Migration workspace: created `integration/local-work` from the Windows-compatible `main`. Restored the complete `WORKLOG.md`, combined the remote collaboration guide with the mandatory work-log instructions, and extended `.gitignore` for preserved local `artifacts/`, `data/public/`, `dist/`, and `results/` content.
- Migration boundary: no Programming v1, Localization, Graph Agent, trainer, checkpoint, A/B experiment, or old packaging implementation has been copied into the remote baseline yet. They remain protected and will be evaluated/migrated individually so the verified remote baseline is not overwritten.
- Verification: confirmed the archive branch and annotated tag resolve to the snapshot commit, local `main` has `origin/main` as its direct parent, the compatibility commit is a single 100%-similar rename, the external backup matches source count/size, and preserved generated data no longer appears as untracked after the ignore-rule merge. Installed the remote's locked `PyYAML==6.0.3` CI dependency. With UTF-8 mode and Git Bash on `PATH`, `python scripts/check_repository.py` passed all 23 Python, 10 shell, 8 configuration, 42 Markdown, and 1 agent-package checks.
- Test result: the 21-test CPU suite ran outside the restrictive sandbox and passed 14 tests. Four failures compare Linux/Python-3.12 archive or newline hashes with output produced by Windows/Python 3.14, and three assert POSIX `/` paths while the checker correctly reports Windows `\` paths. These are platform-sensitive test assumptions rather than observed baseline logic failures; tests were not weakened or edited. The first sandboxed attempts failed earlier because Python could not write temporary directories and are not code regressions.
- Files/state changed: external backup directory, local archive branch/tag, local `main` and integration branch refs, `.gitignore`, `AGENTS.md`, `WORKLOG.md`, and the local Python environment (`PyYAML==6.0.3`). No push, force-push, remote branch mutation, Kaggle action, model run, or competition submission occurred.
- Follow-up: confirm the full 21-test suite in the repository's Linux/Python-3.12 CI environment, then migrate one experimental area at a time. Prefer comparing the remote multi-agent baseline against archived Programming v1 before changing `configs/baseline/`; migrate localization source separately from generated checkpoints.

### 2026-10-07 - Fetch and audit the canonical GitHub remote

- Objective: bring `https://github.com/jwj1342/gemma-agent-lab.git` into the local Git repository as the canonical remote, compare it with the current local workspace, and stop before integrating anything so the user can choose the next action.
- Remote operation: added the URL as `origin` and ran a fetch with pruning. `origin/main` now points to `3054da842f794c854f438a5478ea9437b119e9bc` (`Fix Kaggle submission handoff and audit reproducibility gaps (#2)`). No merge, rebase, checkout, reset, branch-upstream change, push, or working-tree replacement was performed.
- History finding: local `main` is the independent root commit `561d85acf2b2fc1c03e2cc9030cb9fc62938634e`; remote `main` has five commits rooted at `61f7c7edae3e2986c4b88fb4f1104d9761089f76`. `git merge-base main origin/main` returns no merge base, and the two sides contain one and five unique commits respectively. A normal fast-forward pull is therefore impossible.
- Tree comparison: the fetched remote contains 897 files and 30,777,734 blob bytes; 815 files and 30,484,147 bytes are under `resource/`. Comparing remote paths with all local tracked and non-ignored untracked paths yields 3 common paths (`.gitignore`, `AGENTS.md`, and `README.md`), all content-different; 50 local-only paths; and 894 remote-only paths. Semantically related submission files also moved from local `submission/` to remote `configs/baseline/`.
- Remote additions: a multi-agent baseline with a read-only `code_analyzer` sub-agent, stricter 60-second/4-minute evaluation limits, reproducible packaging and Kaggle/HPC preparation scripts, CPU repository checks, GitHub CI/review templates, dependency locks, competition/reference snapshots, public notebook reviews, manifests, and expanded English documentation.
- Local-only work to preserve: Programming v1 under `submission/`, the package script and tests, Graph Agent and Kaggle A/B experiments, Localization v1/v2/v3/v3.1/RL/v4 trainers and 19 artifact files, extension notes, and this work log. The current local baseline uses one agent, a more detailed planning/retrieval prompt, 16,384 output tokens, 4,096 thinking tokens, and 300-second/30-minute evaluation limits; the remote baseline uses a main agent plus analyzer sub-agent, 8,192 output tokens, 2,048 thinking tokens, and 60-second/4-minute limits.
- Risk/decision: directly merging would require `--allow-unrelated-histories` and would collide on the three common files while also mixing two repository layouts. Keep the fetched remote isolated until the user chooses whether to replace the local tree, preserve local work on a branch and adopt the remote, or selectively transplant remote infrastructure into the current tree.
- Verification: confirmed both fetch and push URLs, remote default branch `origin/main`, exact commit/parent chain, absence of a merge base, tree/path counts, and the unchanged pre-existing dirty-worktree set before this log update. No code tests were needed because no application code or configuration was integrated.
- Files/state changed: `.git/config` gained the `origin` definition, Git downloaded remote objects and created `refs/remotes/origin/main`, and `WORKLOG.md` was updated. No fetched file was written into the working tree and no existing local source, prompt, experiment, model artifact, or result was modified.
- Follow-up: wait for the user's integration choice; do not run `git pull` on local `main` as currently structured.

### 2026-10-07 - Run Programming v1 versus retrieval-control A/B test

- Objective: test Programming v1 in the official Harness and compare it with the pre-change retrieval-v1 prompt without integrating other modules.
- Prior result recovered: downloaded the completed earlier baseline/retrieval smoke test. Both variants resolved 0/2 on `fastapi_15661` and `fastapi_15588`; this confirmed the two tasks were useful unresolved controls and exposed edit churn, temporary-file mistakes, and test-file modification in one earlier stochastic run.
- Experiment implementation:
  - Added an isolated private Kaggle script under `experiments/kaggle_programming_v1_ab/` with the exact retrieval-control and Programming v1 prompts, the same model/tools/sampling/budgets, and the same two explicit task IDs.
  - Added trace metrics for seven-field plan compliance before the first file mutation, edit/write calls, failed tool calls, changed-file count, test-file modification, duration, and official resolution.
  - Corrected the post-run plan parser to accept the numbered Markdown labels actually produced by the model and to treat both `write_file` and `edit_file` as first mutations. The remote CSV's zero-compliance value was a parser undercount; corrected trace audit is 1/2. Inference results are unaffected.
- Official result: retrieval control resolved 0/2 and Programming v1 resolved 0/2. Both produced two non-empty final patches and neither final patch modified tests in this paired rerun.
- Efficiency/behavior comparison:
  - Programming v1 reduced median failed tool calls from 2.5 to 1.0 (60% fewer).
  - It increased median edit calls from 3.0 to 4.5, median tool calls from 24.5 to 25.5, and median duration from 350.4 to 400.6 seconds (14.3% slower).
  - Aggregate tokens were 557,846 versus 803,959 for the control, but this is confounded by the control entering a 22-call repeated-`submit_patch` loop on `fastapi_15661`.
- Task diagnosis:
  - For `fastapi_15661`, the gold source is `scripts/prepare_release.py`. The control chose that path but implemented an incomplete interface and exited 2; Programming v1 emitted no structured plan, created `scripts/release.py` plus a workflow, expanded scope, and exited 2.
  - For `fastapi_15588`, both chose the correct `fastapi/sse.py` but exited 1. Programming v1 emitted all seven fields, yet its plan over-scoped validation into `format_sse_event`; the reference patch only adds single-line validators to `ServerSentEvent`.
- Decision: do not promote Programming v1. The current seven-field gate is too weakly followed and too verbose, and an explicit plan can formalize a wrong or over-broad hypothesis. A future v2 should use three compact fields (target/evidence, minimal change plus negative scope, verification) and stop after one clean successful `submit_patch`.
- Verification: the experiment script compiles, embedded Programming v1 prompt exactly matches `submission/prompts/system.md`, credential scanning found no secrets, all 20 repository tests passed, and downloaded traces/results were manually cross-checked against public reference patches. Full analysis is in `results/programming_v1_ab_smoke/analysis.md`.
- Files changed: new `experiments/kaggle_programming_v1_ab/kernel-metadata.json` and `programming_v1_ab_test.py`, `WORKLOG.md`, and downloaded ignored results under `results/retrieval_ab_smoke/` and `results/programming_v1_ab_smoke/`; the production prompt and all retained model artifacts were not changed.
- External state: created and completed private Kaggle Notebook version 1 at `junzhejiao/codingai-programming-v1-a-b-smoke-test`; no competition submission, credential change, or other remote mutation was performed.

### 2026-10-07 - Implement prompt-only Programming v1

- Objective: without integrating Localization v1, Graph Agent, or any other module, add the proposed modification-planning workflow to the existing retrieval prompt.
- Prompt change:
  - Added a hard pre-edit gate after source-grounded localization and before the first edit.
  - Requires one compact `Change plan` with exactly seven fields: target, evidence, root cause, intended change, compatibility boundary, verification, and disproof condition.
  - If the disproof condition is unresolved, requires one bounded source read or targeted search and a plan update before editing.
  - After failed verification, requires classification as wrong location/root cause, implementation defect, or unrelated environment/test problem, followed by an evidence-based plan revision; unchanged patches and failing commands may not be repeated.
  - Stops revision after the diff matches the plan and targeted verification passes, preventing opportunistic cleanup.
- Isolation: compared the working submission byte-for-byte with `dist/submission-retrieval-v1.zip`; only `prompts/system.md` differs. Model, nine-tool list, single-agent topology, sampling configuration, evaluation limits, Localization v1, and Graph Agent are unchanged.
- Packaging: created `dist/submission-programming-v1.zip` with SHA-256 `3E9A58106BAEB78AB12335C0FC56734C71EF7B4C7CF5AB4CFF7394C7B7C20149`. The preserved baseline and retrieval-control hashes remain `F42BB63AA3D12545912CAB31BC6E4638FDFB634F099AC53888AC0CA005E39701` and `948DE71457C59581F545D766C2E07B9B1E56DAFB31E178C8A8DC3C86B00404D4`.
- Verification: added a package test for the planning gate and all seven required fields; all 20 repository tests passed, archive integrity passed, the archive contains the expected four files at its root, `agent_tool` is absent, and `git diff --check` plus explicit trailing-whitespace checks passed.
- Decision: keep Programming v1 as an isolated candidate pending paired end-to-end evaluation; do not replace the preserved baseline or retrieval-only control yet.
- Files changed: `submission/prompts/system.md`, `tests/test_package_submission.py`, `WORKLOG.md`; new ignored generated artifact `dist/submission-programming-v1.zip`.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Choose the starting point for programming improvements

- Objective: decide whether to begin with modification planning or broader prompt changes for the programming path.
- Recommendation: start with a compact modification-planning gate implemented as the only prompt change. This improves the current prompt-driven architecture without introducing another model, and it keeps localization, tools, sampling, and budgets fixed for causal comparison.
- Proposed pre-edit contract: after source verification and before the first edit, record the target file/symbol, observed evidence, root cause, intended behavior change, compatibility/impact boundary, smallest verification command, and one condition that would invalidate the plan. If evidence is insufficient, perform one bounded source read rather than edit.
- Proposed recovery rule: after a failed verification, revise the plan from the failure evidence before making another edit; do not repeat the same patch or command unchanged.
- Experiment order: (1) baseline current retrieval prompt, (2) planning-gate-only prompt, then only if it helps, (3) refine editing and recovery instructions. Do not simultaneously integrate Localization v1 or Graph Agent because that would confound attribution.
- Suggested measures: resolved rate first, then valid/non-empty patch rate, first-edited-file correctness, edit churn, targeted-test success, tool calls, and duration. Planning text should stay compact so it does not consume the 50-turn/tool budget.
- Files changed: `WORKLOG.md` only; no Agent prompt, model, configuration, localization artifact, or distribution archive was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Retain Localization v1 and audit the programming path

- Objective: retain the strongest v3 localization checkpoint under the product-level name Localization v1, then inspect how the current AI implements programming work.
- Naming decision: Localization v1 refers to the protected experiment-v3 checkpoint at `artifacts/localization_v3_3k/` (3,000 epochs, best epoch 1,850). The directory was not renamed because `artifacts/localization_v1/` already contains the older experiment-v1 artifact and preserving lineage avoids ambiguity.
- Architecture finding: there is no standalone programming model or deterministic coding engine in this repository. The official Harness serves `gemma-4-31b-it-qat-w4a16-ct`; `submission/agent.yaml` grants nine tools, and the system prompt instructs the model to parse the issue, localize source, form a hypothesis, edit minimally, run targeted verification, inspect the diff, and call `submit_patch`.
- Runtime finding: the Harness/`swegemma.Evaluator` supplies repository snapshots, graph data, embeddings, sandbox execution, task budgets, and final test evaluation. The model chooses each tool call; repository code only declares policy and limits.
- Integration gap: retained Localization v1 is an offline hashed pairwise ranker and is not referenced by `submission/`, the Notebook evaluator, or the Graph Agent experiment. Current runtime localization is prompt-directed exact search plus optional harness embedding/graph tools.
- Version finding: `submission/` exactly matches `dist/submission-retrieval-v1.zip`; the separately preserved `dist/submission.zip` contains the earlier baseline prompt. Graph Agent v1 remains an unactivated experiment.
- Control and validation finding: programming behavior is governed primarily by prompt rules, with no local end-to-end coding test. Package validation checks the required model, tool list, archive layout, and single-agent boundary, but not whether generated patches solve tasks.
- Verification: rebuilt a temporary audit package and ran the four package tests successfully; confirmed byte-for-byte agreement between `submission/` files and `dist/submission-retrieval-v1.zip`; `git diff --check` passed. The temporary audit archive was removed after exact-path verification; test-only archives were regenerated by the package tests.
- Files changed: `WORKLOG.md` only; no model, localization artifact, Agent configuration, prompt, retained distribution archive, or source code was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Confirm the strongest current localizer version

- Objective: identify the best currently available localization version after the v4 experiment.
- Finding: v3 remains the retained and strongest overall version. On the 19-task held-out test split it records Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, and MRR 0.619.
- Comparison: v3.1 selected its unchanged v3 parent checkpoint; RL v1 and v4 both selected epoch 0 and did not improve directly comparable prediction-quality metrics. V4 is therefore an ablation, not the promoted model.
- Scope caution: the benchmark is closed-world and small, so this is the best version under the current local evaluation rather than proof of best hidden Kaggle performance.
- Files changed: `WORKLOG.md` only; no model, source code, configuration, or artifact was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-07 - Train v4 with squared count and location matching

- Objective: change the RL localization reward to a weighted sum of squared position and count matching, expose independent proportions, run 300 training epochs as v4, and compare it with prior variants.
- Reward implementation:
  - V4 uses `count_weight * count_match^2 + location_weight * location_f1^2`, retaining the 0.50 positive-reward threshold and the guard that assigns zero match when no correct file is returned.
  - Added `--count-weight`, `--location-weight`, and `--reward-component-power`; weights must be non-negative and sum to one. Defaults are 0.65 count, 0.35 location, and power 2. The reward helper keeps power 1 as a backward-compatible linear control.
  - Changed the default output to the isolated `artifacts/localization_v4_300/` directory and records the full reward/training configuration in the model and metrics artifacts.
- Training: completed exactly 300 epochs with seed 52 on the same 91/19/19 split and frozen v3 checkpoint. Sampled training reward reached 0.218 at epoch 300, but deterministic development reward fell from -0.445 at epoch 0 to -0.656 and development exact-set rate fell from 26.3% to 15.8%; checkpoint protection selected epoch 0.
- Comparison:
  - Because both RL v1 and v4 selected epoch 0, their directly comparable selected-checkpoint metrics are identical: development exact count/set 63.2%/26.3%, and test exact count/set/location-F1 63.2%/31.6%/40.4%.
  - At epoch 300, v4 kept development exact count at 63.2% versus RL v1's 42.1%, but both had 15.8% exact-set accuracy and v4 had lower location-F1 (25.0% versus 27.6%).
  - Reward values are not directly comparable across v1 and v4 because the scoring formulas differ. The squared formula intentionally lowers partial-match scores and did not solve file-identity generalization or the one-file training imbalance.
  - Retained v3 remains preferable: its test ranking metrics are Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, and MRR 0.619, while its metrics are not directly equivalent to the RL complete-set measures.
- Decision: keep v4 as a reproducible reward ablation but do not promote it over v3. A comparison report is stored at `artifacts/localization_v4_300/comparison.md`.
- Verification: all 19 unit tests passed; `git diff --check` and explicit trailing-whitespace checks passed; v4 records 300 completed epochs, squared power 2, the requested 0.65/0.35 proportions, and best epoch 0. The protected v3 artifacts were only read and not overwritten.
- Files changed: `scripts/train_rl_localizer.py`, `tests/test_rl_localizer.py`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v4_300/`; prior v1, v2, v3, v3.1, and RL v1 artifacts were not modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Add count-emphasized reinforcement learning for localization

- Objective: reward or penalize the policy according to how closely the number and identity of returned files match the answer, with positive reward at 50% match and extra emphasis on count accuracy.
- Reward design:
  - `count_match = 1 - abs(predicted_count - gold_count) / max(predicted_count, gold_count)`.
  - `location_f1` is the set F1 between predicted and gold source-file paths.
  - Combined match is `0.65 * count_match + 0.35 * location_f1`; match at or above 0.50 receives positive reward equal to match, otherwise reward is `-(1 - match)`.
  - A zero-location guard forces match to zero when no correct file is returned, preventing a policy from earning positive reward by guessing only the correct count.
- Implementation:
  - Added `scripts/train_rl_localizer.py`, a dependency-free two-head contextual-bandit policy on frozen v3: a 1–26 file-count head and a trainable residual file-ranking head.
  - Added stochastic count selection, without-replacement file selection, REINFORCE-style advantage updates with per-repository moving baselines, global decoupled decay, deterministic evaluation, and best-development checkpoint protection.
  - Added five reward tests covering exact match, wrong locations with correct count, the 50% boundary, below-threshold proportional penalty, and stronger count weighting.
- Data finding: the full public gold-count distribution is extremely imbalanced: 91 of 129 tasks change one source file, while rare tasks change as many as 26. In the fixed training split, 67 of 91 tasks have count one.
- Pilot: ran 300 epochs with count weight 0.65. Sampled training reward rose from roughly -0.04 to +0.47, but deterministic development reward never beat the epoch-0 value of -0.350 and exact-set rate degraded from 26.3% to 15.8% during training.
- Decision: the saved best checkpoint is epoch 0, with zero residual/count learned weights and frozen v3 behavior plus the empirical count prior. Keep the code and curve as an RL prototype, but do not promote it over v3. Future work needs count-balanced sampling or more multi-file tasks before another RL run.
- Verification: fifteen tests passed, `git diff --check` passed, the 300-epoch artifact and reward configuration were validated, and retained v3 artifact hashes remained unchanged.
- Files changed: new `scripts/train_rl_localizer.py`, new `tests/test_rl_localizer.py`, `WORKLOG.md`, and generated artifacts under `artifacts/localization_rl_v1/`; no v3 artifact was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Show a representative public task structure

- Objective: give the user one concrete example of the localization-training task format.
- Example reviewed: public task `rich_3676` from `Textualize/rich` at base commit `2b5d648b2c62960deb6439f4204ed8fa6eb3cf7b`; its problem statement is `Notes\n\nAdd support for Exception.__notes__` and it has no hints.
- Structure explained: each JSONL record contains `instance_id`, `repo`, `base_commit`, `problem_statement`, `hints_text`, organizer `patch`, and organizer `test_patch`. Only the problem/hints are localization input; reference patches are post-run labels.
- Gold localization for the example: the reference source patch changes `rich/default_styles.py` and `rich/traceback.py`; the organizer test patch changes `tests/test_traceback.py`.
- Files changed: `WORKLOG.md` only; no model, training data, code, or artifact was modified.
- External state: no training run, Kaggle request, Notebook action, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Continue v3 for 3,000 epochs as v3.1

- Objective: preserve all v3 data, continue from its selected checkpoint for 3,000 additional epochs, and store the experiment separately as v3.1.
- Implementation:
  - Added validated sparse-checkpoint loading for the `no_issue_path_v3` feature set, including dimension and candidate-inventory checks.
  - Added `--init-model` and `--epoch-offset`; v3.1 restored v3's epoch-1,850 weights and continued the learning-rate schedule over global epoch indices 1,851 through 4,850.
  - Evaluated the parent checkpoint as local epoch 0 and retained it unless a continuation checkpoint improved the Hit@1/Hit@5 composite, preventing continuation regression.
  - Added non-empty output-directory refusal so retained artifacts cannot be accidentally overwritten.
  - Added a fixed read-only checkpoint fixture and unit test for sparse weight restoration. An ACL-restricted temporary test directory created during initial validation was explicitly removed after exact-path verification.
- Training: completed exactly 3,000 additional epochs and recorded 122 curve rows including the epoch-0 parent checkpoint. No continuation checkpoint exceeded the parent's development composite score of 0.6579; the best additional checkpoints reached at most 0.6316.
- Result: v3.1 selected local epoch 0/global epoch 1,850. Its saved weights are byte-for-byte equal as JSON weight values to the protected v3 weights, and its held-out metrics therefore remain Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, and MRR 0.619.
- Decision: keep `artifacts/localization_v3_1_additional_3k/` as the continuation record but do not promote it over v3; further identical-data epochs are not justified by this result.
- Verification: ten unit tests passed, `git diff --check` passed, v3.1 lineage/configuration and 3,000 additional epochs were validated, and all three retained v3 artifact hashes remained unchanged.
- Files changed: `scripts/train_localizer.py`, `tests/test_train_localizer.py`, new `tests/fixtures/v3_checkpoint.json`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v3_1_additional_3k/`; no v3 artifact was modified.
- External state: no Kaggle request, Notebook action, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Retain localizer v3

- Objective: preserve the current v3 implementation and trained checkpoint.
- Decision: treat v3 as the retained current localizer variant and keep `artifacts/localization_v3_3k/model.json`, `metrics.json`, and `training_curve.csv` unchanged. Keep v1 and v2 as comparison artifacts rather than deleting them.
- Verification: confirmed the v3 artifact still records 3,000 completed epochs, best epoch 1,850, feature set `no_issue_path_v3`, global decoupled weight decay, and the previously reported held-out metrics.
- Files changed: `WORKLOG.md` only; no source, model, metric, or training-curve file was modified.
- External state: no training run, Kaggle request, Notebook action, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Check remote L4 status without affecting local work

- Objective: inspect the existing Kaggle L4 Notebook while leaving the completed local v3 training and remote job unchanged.
- Result: a read-only Kaggle status query reports `junzhejiao/codingai-retrieval-a-b-smoke-test` as `KernelWorkerStatus.QUEUED`; it is still waiting for compute and has not started L4 execution.
- Files changed: `WORKLOG.md` only for this status check.
- External state: no Notebook stop, restart, rerun, push, competition submission, credential change, or artifact download was performed.

### 2026-10-06 - Remove issue-path memorization and train v3 for 3,000 epochs

- Objective: completely remove the label-specific `issue_path` feature, replace touched-coordinate L2 with true global decoupled weight decay, and train a separate v3 for 3,000 epochs.
- Implementation:
  - Removed `issue_path` from feature construction and removed its weight from the model and CLI configuration.
  - Implemented global decoupled decay as a lazy scalar applied to every weight on every update, including weights absent from the current positive/negative pair; materialization preserves exact weights for checkpoints and serialization.
  - Added tests proving the memorization feature is absent and an untouched coordinate is decayed.
  - Changed defaults to 3,000 epochs, evaluation every 25 epochs, output under `artifacts/localization_v3_3k/`, and weight decay `1e-4`.
- Training: completed exactly 3,000 epochs with seed 52 and recorded 121 evaluation points. The equal-weight Hit@1/Hit@5 development objective selected epoch 1,850 with composite score 0.6579.
- Held-out results on the same 19 test tasks:
  - v3: Hit@1 47.4%, Hit@3 73.7%, Hit@5 78.9%, MRR 0.619.
  - v1: Hit@1 47.4%, Hit@3 68.4%, Hit@5 73.7%, MRR 0.614.
  - v2: Hit@1 42.1%, Hit@3 63.2%, Hit@5 73.7%, MRR 0.556.
- Interpretation: v3 matches v1 Hit@1 and improves Hit@3, Hit@5, and MRR by roughly one held-out task or less. This is directionally positive but not conclusive because the test split contains only 19 tasks and the benchmark remains closed-world.
- Verification: nine unit tests passed, `git diff --check` passed, and the saved artifact records the requested 3,000 epochs, best epoch 1,850, the no-`issue_path` feature set, and global-decoupled decay mode.
- Files changed: `scripts/train_localizer.py`, `tests/test_train_localizer.py`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v3_3k/`; v1 and v2 artifacts were not overwritten.
- External state during training: no Kaggle request, Notebook push, competition submission, credential change, or remote-process change was performed.

### 2026-10-06 - Diagnose why localizer v2 underperformed

- Objective: explain why the downweighted, more strongly regularized 10,000-epoch v2 did not outperform v1.
- Evidence:
  - V2 delayed perfect training Hit@1 from epoch 75 to epoch 450 but still reached 100%, so the changes slowed memorization rather than preventing it.
  - Despite nominally stronger L2, the selected v2 model has 127,877 nonzero weights, L1 norm 3,198.31, and L2 norm 26.93 versus v1's 102,911, 1,104.67, and 11.05. The longer run allowed weights to compensate for the 0.25 feature scaling; the current update applies shrinkage only to touched coordinates and scales it by the decaying learning rate.
  - V1 and v2 reached the same maximum development composite score of 0.6053. V1 reached it once at epoch 50, while v2 reached it at 68 recorded evaluation points between epochs 450 and 6,100, leaving the new objective with weak checkpoint discrimination.
  - With only 19 development and 19 test tasks, each Hit@k task changes the metric by 5.26 percentage points. V2's Hit@1 and Hit@3 decline relative to v1 each corresponds to one test task, so the apparent difference is not statistically strong.
  - Equal weighting of Hit@1 and Hit@5 does not directly optimize a listwise top-five objective; pairwise training still compares one selected positive against at most one hard and one random negative. The frequency-plus-lexical baseline therefore retains substantially better Hit@5.
- Conclusion: v2 combined an underpowered, highly discrete selection signal with excessive repeated optimization on 91 training tasks. Feature rescaling and nominal L2 were insufficient structural controls, and 10,000 epochs amplified rather than solved the small-data problem.
- Files changed: `WORKLOG.md` only; no source code, model, or artifact was modified.
- External state: no training run, Kaggle request, Notebook push, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Regularize the localizer and retrain for 10,000 epochs

- Objective: reduce `issue_path` memorization, strengthen regularization, balance Hit@1 and Hit@5 during checkpoint selection, and complete 10,000 new training epochs without overwriting v1.
- Implementation:
  - Changed `PairwiseLocalizer` to expose `issue_path_weight`, with a new default of 0.25 instead of the prior implicit 1.0.
  - Increased default L2 regularization from `1e-6` to `1e-4`.
  - Added an equal-weight checkpoint objective, `0.5 * Hit@1 + 0.5 * Hit@5`, using development MRR only to break equal composite scores.
  - Added CLI controls for the issue-path weight and checkpoint Hit@1 weight, and recorded all training/selection settings in the saved metrics.
  - Added tests for issue-path downweighting and composite checkpoint scoring.
- Training: ran exactly 10,000 epochs with seed 52, evaluation every 50 epochs plus epoch 1, and wrote a separate v2 artifact directory. The run produced 201 recorded evaluation points and selected epoch 450 with a development composite score of 0.6053.
- Held-out results on the same 19 test tasks:
  - v2: Hit@1 42.1%, Hit@3 63.2%, Hit@5 73.7%, MRR 0.556.
  - v1: Hit@1 47.4%, Hit@3 68.4%, Hit@5 73.7%, MRR 0.614.
  - Frequency-plus-lexical baseline: Hit@1 36.8%, Hit@3 63.2%, Hit@5 94.7%, MRR 0.568.
- Decision: retain v2 as an ablation but do not replace v1. The changes preserved v1 Hit@5 but reduced Hit@1 and MRR; the 10,000-epoch trajectory also confirms that useful development performance peaks early and later epochs mainly sustain training-set memorization.
- Verification: eight unit tests passed, `git diff --check` passed, and artifacts record 10,000 completed epochs, best epoch 450, 201 curve rows, and 127,877 nonzero weights.
- Files changed: `scripts/train_localizer.py`, `tests/test_train_localizer.py`, `WORKLOG.md`, and new generated artifacts under `artifacts/localization_v2_10k/`; v1 artifacts were not modified.
- External state: no Kaggle request, Notebook push, competition submission, credential change, or remote L4 process change was performed.
- Follow-up: test a hybrid score that preserves the frequency-plus-lexical candidate recall, replace gold-derived candidates with full repository file inventories, and use early stopping instead of additional epochs.

### 2026-10-06 - Expose localization code for manual tuning

- Objective: show the user the trained file-localization implementation and identify the sections and parameters suitable for manual tuning.
- Reviewed components: public-patch label extraction, deterministic repository split, issue/path tokenization, hashed feature construction, pairwise logistic update, hard-negative selection, development-checkpoint selection, evaluation metrics, and CLI hyperparameters.
- Important tuning targets: token limit and stop words, feature values in `PairwiseLocalizer.features`, learning-rate decay and L2 regularization, negative sampling, development checkpoint objective, and closed-world candidate construction.
- Important caution: `issue_path=<repo:path>:<token>` is highly label-specific and explains the rapid training-set memorization; `build_candidates` uses the union of public gold paths and must be replaced by a full repository file inventory for a realistic benchmark.
- Files changed: `WORKLOG.md` only; localization source and trained artifacts were not modified.
- External state: no training run, Kaggle request, Notebook push, submission, credential change, or remote-process change was performed.

### 2026-10-06 - Train a lightweight file localizer for 1000 epochs

- Objective: start no-queue training of code-location ability and complete 1000 training passes without changing the existing embedding model or interfering with the remote L4 run.
- Data preparation:
  - Downloaded the public `tasks.jsonl` locally and excluded it from Git; it contains 129 tasks across FastAPI, Requests, Rich, and HTTPX.
  - Corrected reference-patch parsing to use unified-diff `+++ b/<path>` headers; all 129 tasks yielded at least one non-test Python target and 38 tasks changed multiple source files.
  - Used a deterministic repository-stratified split of 91 train, 19 development, and 19 test tasks. Reference patches supplied labels only and were not included in issue text.
- Implementation:
  - Added `scripts/train_localizer.py`, a dependency-free hashed pairwise logistic ranker using issue tokens, path priors, issue/path associations, lexical overlap, and hard-negative training.
  - Added parser and deterministic-split tests in `tests/test_train_localizer.py`.
  - Ran exactly 1000 full epochs with seed 52, evaluated the development set every 25 epochs (plus epoch 1), and saved the epoch-50 best-development checkpoint instead of the overfit final weights.
  - Saved `model.json`, `metrics.json`, and `training_curve.csv` under `artifacts/localization_v1/`.
- Results:
  - Training Hit@1 at the selected checkpoint: 97.8%; development Hit@1/MRR: 47.4%/0.598.
  - On the untouched 19-task test split, the selected model reached Hit@1 47.4%, Hit@3 68.4%, Hit@5 73.7%, and MRR 0.614.
  - The frequency-plus-lexical test baseline reached Hit@1 36.8%, Hit@3 63.2%, Hit@5 94.7%, and MRR 0.568. The model improved top-rank precision and MRR, but degraded Hit@5, so it should be treated as an experimental reranker rather than a replacement retrieval system.
- Verification: six unit tests passed; the artifact records 1000 completed epochs, 41 evaluation points, best epoch 50, and 102,911 nonzero hashed weights; final `git diff --check` passed.
- Limitations: the candidate inventory is built from public changed-file labels and omits never-changed repository files, only file-level localization is trained, and the test split is small. Full-tree candidates and symbol-level labels are required before agent integration.
- Files changed: `.gitignore`, `scripts/train_localizer.py`, `tests/test_train_localizer.py`, `WORKLOG.md`, and generated artifacts under `artifacts/localization_v1/`; downloaded ignored input under `data/public/`.
- External state: no Kaggle Notebook push, competition submission, remote-run change, or credential-file modification was performed.
- Follow-up: construct complete per-commit file inventories, add symbol-level targets, and compare a conservative hybrid that falls back to the frequency/lexical ranker so Hit@5 is not lost.

### 2026-10-06 - Assess no-queue training options

- Objective: determine which training work can begin immediately from the public data without waiting for Kaggle L4 capacity.
- Finding: data preparation, reference-patch parsing, gold file/symbol extraction, retrieval-policy supervision, lightweight ranking/classification, graph-policy tuning, prompt/skill development, and CPU/static evaluation can start immediately after downloading `tasks.jsonl` and selected graph/embedding files.
- Hardware boundary: the local NVIDIA RTX 5060 Laptop GPU has 8,151 MiB VRAM, which is insufficient for practical fine-tuning or rollout-based RL of the required `gemma-4-31b-it-qat-w4a16-ct`; the official runtime uses four L4 GPUs. LoRA/RL for the required 31B model therefore still needs substantially larger local/external compute or queued Kaggle capacity.
- Recommended immediate training target: build a symbol-localization/router dataset from public issue text and post-run reference patches, then train or tune a lightweight policy that predicts exact symbols and whether Graph Agent delegation is needed. Keep a fixed repository/task holdout to measure generalization.
- Files changed: `WORKLOG.md` only.
- External state: no dataset download, credential change, Notebook push, rerun, or competition submission was performed.

### 2026-10-06 - Clarify when localization accuracy can improve

- Objective: clarify whether localization accuracy can improve only after a formal competition submission.
- Finding: localization can be measured and improved entirely through private evaluation on the 129 public development tasks. Reference patches are used only after each run to score files/symbols and never exposed to the agent. A fixed development/validation/holdout split prevents tuning directly to every public answer.
- Runtime feedback: even without reference patches, the agent can improve confidence during a task by checking exact symbol resolution, source behavior, graph relationships, call sites, and targeted tests.
- Boundary: hidden-test localization cannot be observed before formal scoring; public holdout performance is only an estimate of generalization. Formal submissions should be used for confirmation rather than the main tuning loop.
- Current opportunity: after the private smoke test completes, download its trajectory artifacts and compute localization metrics offline without pushing a new Notebook version or submitting to the competition.
- Files changed: `WORKLOG.md` only.
- External state: no credential changes, Notebook pushes, reruns, or competition submissions.

### 2026-10-06 - Check and explain remote L4 results access

- Objective: show how to inspect the private Kaggle L4 smoke-test status, logs, and output artifacts.
- Current status: `junzhejiao/codingai-retrieval-a-b-smoke-test` remains `KernelWorkerStatus.QUEUED`; no runtime logs or A/B result files exist yet.
- Result locations after completion: the private Notebook version page exposes Logs and Output; `ab_results.csv` contains per-task details and `ab_summary.csv` contains the variant comparison.
- Files changed: `WORKLOG.md` only.
- External state: performed one read-only status query; no credential changes, Notebook pushes, reruns, or competition submissions.

### 2026-10-06 - Define code-localization accuracy metrics

- Objective: define how to measure whether the agent accurately finds the bug or target edit location.
- Measurement design: derive gold changed files, hunks, and enclosing symbols from each public reference patch after the run; parse the ATIF trajectory/tool trace to compute file Hit@k, symbol Hit@k, MRR, tool calls/time to first correct source read, irrelevant-read rate, and whether the first edited file is a gold file. Keep official test resolution as the ultimate outcome because alternative valid fixes may not match the reference location.
- Leakage guardrail: reference patches and test patches may be used only for post-run scoring, never included in the agent prompt or retrieval input.
- Important harness finding: official documentation states that offline `search_similar_code` resolves queries against stored node keys/suffixes and recommends class/function/module symbols rather than free-form natural-language sentences. The current retrieval-v1 query instructions may therefore underperform and should be revised in a later version after preserving the current smoke-test result.
- Current limitation: the queued smoke test records resolution, tool calls, duration, and patch size, but does not yet extract localization metrics from trajectory traces.
- Files changed: `WORKLOG.md` only.
- External state: no credential changes, Notebook pushes, reruns, or competition submissions.

### 2026-10-06 - Document the end-to-end agent runtime logic

- Objective: explain how the project agent understands an issue, writes code, verifies correctness, and is finally scored.
- Summary: the harness supplies the problem statement, hints, clean repository snapshot, graph, and embeddings; the Gemma root agent extracts requirements, localizes source with exact search and conditional retrieval, forms a source-grounded hypothesis, makes a minimal source patch, runs targeted verification, inspects status/diff, and calls the internal `submit_patch` tool. The harness then applies the patch to a clean verification sandbox, restores protected tests/configuration, applies organizer tests, and runs pytest for pass/fail scoring.
- Architecture note: the current retrieval-v1 candidate is single-agent; Graph Agent v1 is a separate, not-yet-activated experiment that delegates uncertain localization to a read-only specialist while retaining all edits and verification on the root.
- Files changed: `WORKLOG.md` only.
- External state: no credential changes, Notebook pushes, or competition submissions were made for this explanation.

### 2026-10-06 - Implement Graph Agent v1 while monitoring retrieval smoke test

- Objective: optimize the current retrieval-v1 design with a read-only Graph Agent without modifying credentials or pushing another version of the active Kaggle Notebook.
- Work performed:
  - Downloaded and reviewed the official `HARNESS_README.md` and `sample_submission` AgentTool configuration.
  - Added an isolated multi-agent experiment under `experiments/graph_agent/submission/` rather than modifying the controlled `submission/` directory.
  - Added `code_graph_analyzer`, restricted to `read_file`, `search_similar_code`, `get_code_neighbors`, and `get_code_subgraph`, with `include_contents: none`, disabled transfers, and `skip_summarization: true` at the root tool boundary.
  - Kept editing, shell commands, tests, status inspection, and `submit_patch` exclusively on the root agent.
  - Added bounded delegation and structured evidence-packet prompts, including ranked candidates, dependency evidence, root-cause hypothesis, verification advice, and uncertainty.
  - Extended the packaging script with an explicit `--allow-multi-agent` mode while preserving single-agent validation by default.
  - Added two package tests for the Graph Agent directory and archive layout; updated the extension documentation.
  - Generated `dist/submission-graph-agent-v1.zip` without replacing any baseline or retrieval archive.
- Verification:
  - `python -m unittest discover -s tests -v`: 4 tests passed.
  - `git diff --check`: passed, with only Git line-ending notices.
  - Archive integrity and expected seven-file layout verified.
  - Graph Agent archive SHA-256: `1131E65894755726A5A41B3EFC59B72580B16BEE038C46D4F7002B09C214D5E5`.
- Files changed: `scripts/package_submission.py`, `tests/test_package_submission.py`, `extensions/graph_agent/README.md`, `WORKLOG.md`, and new files under `experiments/graph_agent/`; generated ignored artifact: `dist/submission-graph-agent-v1.zip`.
- External-state guardrails: did not modify the local credential file and did not push Graph Agent or another version of the active Notebook.
- Follow-up: after the retrieval-v1 smoke test finishes, inspect its results; then create a separate Graph Agent A/B Notebook only with explicit approval or after the current run is complete.

### 2026-10-06 - Authenticate and launch private retrieval A/B smoke test

- Objective: validate retrieval-v1 against the original baseline in the official Kaggle environment without creating a competition submission.
- Work performed:
  - Verified the local credential through Kaggle's read-only token introspection endpoint without printing or recording it.
  - Installed Kaggle CLI 2.2.4 into a temporary directory and confirmed the account has joined `gemma-4-developer-agent`.
  - Pulled the official Starter Notebook and used its offline wheel, vLLM, model-registry, and `Evaluator` setup as the basis for a private test script.
  - Added `experiments/kaggle_retrieval_ab/` with a private, internet-disabled L4 script comparing the original baseline and retrieval-v1 on the same two public tasks lacking explicit location hints.
  - Scanned the upload directory for credential patterns, then pushed private Kaggle Notebook version 1 at `junzhejiao/codingai-retrieval-a-b-smoke-test`.
- Verification: local script syntax, private kernel metadata, credential scan, two repository tests, and `git diff --check` passed before upload; Kaggle accepted version 1 and placed it in the L4 queue.
- Files changed: `WORKLOG.md` and new files under `experiments/kaggle_retrieval_ab/`; no credential file changes.
- External state: one private Kaggle Notebook/version was created and started; no `Submit to Competition` action was used.
- Follow-up: monitor completion, download `ab_results.csv` and `ab_summary.csv`, and diagnose any runtime failure before expanding beyond two tasks.

### 2026-10-06 - Reject an exposed Kaggle credential

- Objective: assess a newly generated Kaggle API token supplied for testing.
- Finding: the value had the expected Kaggle API-token prefix, but it was pasted into chat and must therefore be treated as compromised.
- Action: did not validate, use, persist, or reproduce the credential; instructed the user to revoke it, generate a replacement, and store the replacement only in the local `~/.kaggle/access_token` file.
- Follow-up decision: even after the user accepted the risk, the chat credential was not embedded in a command or request because doing so would copy it into additional execution logs. A local credential file remains the minimum safe handoff.
- Credential check: `C:\Users\32747\.kaggle\access_token` was created as a directory rather than a text file, so no token was read and no Kaggle authentication request was attempted.
- User guidance: provided Windows Notepad steps to remove the mistaken directory and save a UTF-8, extensionless `access_token` file without placing the credential in chat or project files.
- Second credential check: the mistaken directory was removed, but `C:\Users\32747\.kaggle` was empty and no credential file was present; authentication was therefore not attempted.
- Files changed: `WORKLOG.md` only; no credential data was recorded.
- Follow-up: after the replacement token is stored locally and the exposed token is revoked, verify authentication without printing the token and resume the private Kaggle test.

### 2026-10-06 - Attempt to start the official A/B test

- Objective: run the approved retrieval-policy experiment without creating a Kaggle competition submission.
- Work performed:
  - Checked available external integrations; no Kaggle plugin/connector was available.
  - Audited the local environment for Kaggle authentication, CLI/runtime packages, official data, Docker, and GPU capacity.
  - Confirmed the cached official starter-notebook response is readable and describes a two-task public evaluation flow, but the competition-data download is a 40-byte `401 Unauthenticated` response.
  - Attempted to install the official Kaggle CLI into a temporary directory; dependency download failed because the PyPI connection terminated during TLS, and a cached-package-only fallback was unavailable.
- Environment findings:
  - No Kaggle CLI, `swegemma`, Google ADK, vLLM, PyTorch, or Docker is installed.
  - No `KAGGLE_API_TOKEN`, legacy Kaggle environment credentials, `~/.kaggle/access_token`, or `~/.kaggle/kaggle.json` is configured.
  - The local GPU is an NVIDIA GeForce RTX 5060 Laptop GPU with 8,151 MiB, insufficient for the official four-L4 model-serving setup.
- Result: the real agent A/B run cannot start until Kaggle authentication is configured and the competition rules have been accepted; no notebook was launched and no website submission was created. The CLI is also not yet installed, but authentication is the first required user action and direct API access can be used if CLI installation remains unavailable.
- Files changed: `WORKLOG.md` only.
- Follow-up: configure Kaggle authentication locally (prefer `~/.kaggle/access_token` or `kaggle auth login` after CLI installation), then download/push the required assets and launch a private Notebook run without selecting `Submit to Competition`.

### 2026-10-06 - Clarify Kaggle evaluation versus submission

- Objective: determine whether running the experiment in the official Kaggle environment changes the website competition submissions.
- Finding: running a notebook, evaluating against public tasks, or generating `/kaggle/working/submission.parquet` does not by itself create a competition submission. A submission is created only through an explicit `Submit to Competition` action or the Notebook Editor shortcut that combines saving and submitting.
- Operational caution: notebook runs consume Kaggle compute/GPU quota, and saving a version creates notebook output/version history, but neither should change the leaderboard submission history unless the submit action is selected.
- Files changed: `WORKLOG.md` only.
- Verification: checked the current official starter-notebook metadata and Kaggle competition/notebook submission documentation on 2026-10-06.
- Follow-up: run the 24-task public-data A/B as a private notebook evaluation and stop after inspecting saved outputs; do not select `Submit to Competition` until the variant is intentionally ready for hidden scoring.

### 2026-10-06 - Implement embedding-retrieval policy v1

- Objective: implement the approved prompt-only strategy for using the existing `search_similar_code` tool more effectively.
- Work performed:
  - Updated `submission/prompts/system.md` with a localization ladder that prioritizes direct file/symbol inspection and exact text search.
  - Added explicit semantic-search triggers, compact query guidance, `k=5`, inspection of at most three initial candidates, source-based confidence checks, and one materially reformulated retry.
  - Added conditional escalation to `get_code_neighbors` and small, known-node uses of `get_code_subgraph`.
  - Added retrieval budgets and stop conditions so localization does not consume the edit, verification, diff, and submission budget.
  - Generated `dist/submission-retrieval-v1.zip` without overwriting the existing baseline archive.
- Verification:
  - `python -m unittest discover -s tests -v`: 2 tests passed.
  - Confirmed the experiment archive contains the retrieval-policy heading and `search_similar_code(query, k=5)` instruction.
  - Baseline SHA-256 remained `F42BB63AA3D12545912CAB31BC6E4638FDFB634F099AC53888AC0CA005E39701`.
  - Experiment SHA-256 is `948DE71457C59581F545D766C2E07B9B1E56DAFB31E178C8A8DC3C86B00404D4`.
- Files changed: `submission/prompts/system.md`, `WORKLOG.md`; generated ignored artifact: `dist/submission-retrieval-v1.zip`.
- Limitation: no end-to-end task was executed because the official Kaggle harness and task data are not present in this repository.
- Follow-up: run the planned paired 24-task baseline-vs-policy pilot in the official harness before promoting the experiment archive to the default submission.

### 2026-10-06 - Design the embedding-retrieval policy experiment

- Objective: specify how to improve the existing `search_similar_code` strategy before deciding whether to implement it.
- Proposed implementation:
  - Make a prompt-only change in `submission/prompts/system.md`; keep the model, agent topology, tools, sampling settings, and embeddings unchanged.
  - Add a localization ladder: use direct file/symbol reads and exact text search first; call semantic retrieval only when location remains uncertain or exact search is ambiguous.
  - Build compact queries from exact errors, API/symbol names, actual behavior, and expected behavior; use `k=5`, inspect at most the top three candidates, and allow at most one materially reformulated retry before editing.
  - Treat retrieval results as candidates, require source verification, use graph neighbors only after identifying a credible symbol, and stop retrieval once one concrete edit hypothesis is supported.
  - Reserve `get_code_subgraph` for a small set of already identified symbols involved in a cross-file flow.
- Proposed evaluation: paired baseline-vs-policy runs on a fixed public-task pilot, followed by a held-out confirmation only if resolution improves without excessive tool-call/time cost or new invalid patches.
- Decision gate: no functional Agent files were changed; implementation awaits explicit user approval.
- Files changed: `WORKLOG.md` only.
- Follow-up: if approved, patch the prompt, package a separately named variant archive, run static validation, and prepare/run the controlled A/B evaluation where the official harness is available.

### 2026-10-06 - Prioritize the extension roadmap

- Objective: rank Multi-Agent, Graph Agent, RAG, RL, LoRA, and embedding retrieval for the current baseline.
- Recommendation: start with embedding retrieval by improving query construction, small-`k` candidate selection, and source verification using the existing `search_similar_code` tool.
- Rationale:
  - The competition already supplies per-task graph-node embeddings and an offline semantic-search tool, so this route has the lowest implementation cost and directly targets code localization.
  - Use graph neighbors/subgraphs as a second-stage expansion after finding a credible symbol; create a separate read-only Graph Agent only if controlled evaluation shows delegation adds value beyond direct tool use.
  - Custom RAG overlaps with the supplied retrieval system; generic Multi-Agent adds coordination cost; LoRA and RL require a stable evaluation/training pipeline and are later-stage experiments.
- Verification: reviewed all extension design notes and checked the current Kaggle competition overview and official starter-notebook metadata on 2026-10-06.
- Files changed: `WORKLOG.md` only.
- Follow-up: run a baseline-vs-retrieval-policy ablation on the public tasks and record resolution rate, localization success, tool calls, wall time, context use, and patch validity.

### 2026-10-06 - Add persistent project work log

- Objective: create a durable Markdown record that future work sessions can use to understand prior activity quickly.
- Work performed:
  - Added this `WORKLOG.md` with a concise current-state summary and reverse-chronological history.
  - Added root-level `AGENTS.md` instructions requiring future agents to read and update the log on every task.
- Files changed: `AGENTS.md`, `WORKLOG.md`.
- Verification: confirmed the repository previously had no `AGENTS.md` or existing work-log Markdown file.
- Follow-up: maintain both the current-state summary and a new history entry during every subsequent task.

### 2026-10-06 - Assess basic programming capability

- Objective: determine whether the current project can perform simple programming tasks.
- Findings:
  - The Kaggle submission configuration exposes tools for reading, searching, editing, writing, running commands, checking status, graph/code retrieval, and submitting patches.
  - The system prompt defines a complete minimal code-repair workflow.
  - The project can serve as a basic coding agent inside the official Kaggle harness.
  - It cannot yet run as a standalone local interactive coding assistant because the local runtime and model integration are absent.
- Verification:
  - Ran `python -m unittest discover -s tests -v` successfully: 2 tests passed.
  - Inspected `dist/submission.zip`; it contains `agent.yaml`, `configs/sampling.yaml`, `eval_config.yaml`, and `prompts/system.md` at the expected paths.
- Files changed: none.
- Follow-up: run an end-to-end sample issue through the official notebook/harness when available.
