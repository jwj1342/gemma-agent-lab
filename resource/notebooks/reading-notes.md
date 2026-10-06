# Public notebook reading notes

Snapshot: 112 public entries, 112 individual reading records. Every entry has a reading record; the empty-source entry documents its review limitation.
These notes distinguish submitted payloads, surrounding notebook experiments, and author claims. No third-party notebook was executed; this is not a line-by-line security audit.
Original downloads remain unchanged. Static decoding of embedded configuration does not establish that its scripts ran or its model was trained.

See the [complete index](README.md) for sources and [competition research](../../docs/competition-research.md) for conclusions.

## 1. Chronos: Completion-Authority-Separated Agent Harn

[Kaggle](https://www.kaggle.com/code/segamega/chronos-completion-authority-separated-agent-harn) · [Local source](public/segamega/chronos-completion-authority-separated-agent-harn/readable.txt)

**Category: Local evaluation and log audit.** Copies the official sample and evaluates the first five tasks, recording claim/resolved status, event chains, and divergence. The final ZIP contains only the agent directory.

Reusable ideas: Retain actual patches, test verdicts, and exceptions; count patch production separately from issue resolution.

Caveats: GATE changes notebook predictions but is absent from the submitted ZIP, as is the hash-chain audit. Surrounding analysis is not automatically executed by scoring.

Source locations: L180–L230, L260–L349.

## 2. Agentic | Counterexample Lab

[Kaggle](https://www.kaggle.com/code/lucifer19/agentic-counterexample-lab) · [Local source](public/lucifer19/agentic-counterexample-lab/readable.txt)

**Category: Counterexample-probe skill packaging.** Builds one coder plus probe.py, with at most three probes. Saves an assertion-failure baseline, reruns the same hashed probe, and separates ERROR/TIMEOUT/PASS.

Reusable ideas: Establish that a probe reproduces the issue before checking a repair. Keep assertions and time bounds; continue normal repair if the helper fails.

Caveats: Primarily CPU packaging; GPU pilot and score were not independently reproduced. include_thoughts=false disables thinking in this harness. Frozen files share the agent sandbox and are not a tamper-proof scoring boundary.

Source locations: L1–L28, L41–L127.

## 3. Gemma 4 Agent: Run the Real Evaluator Locally

[Kaggle](https://www.kaggle.com/code/hitarthjain0/gemma-4-agent-run-the-real-evaluator-locally) · [Local source](public/hitarthjain0/gemma-4-agent-run-the-real-evaluator-locally/readable.txt)

**Category: Multi-task official evaluator entry.** Evaluates tasks[::13][:10], recording patches, test status, command counts, and repeated commands. The author reports variation across repeated runs of the same configuration.

Reusable ideas: Reuse per-task diagnostics and repeat-call counts. Repeat configurations rather than drawing conclusions from one or two task differences.

Caveats: Ten minutes per task cannot fit roughly 120 tasks into 12 hours. The sample is not held out by repository. Truncating commands to 200 characters may merge distinct calls; results such as 3/10 are author-reported.

Source locations: L1–L22, L87–L94, L228–L243, L250–L310.

## 4. Gemma Developer Agent: 0.13 Submission

[Kaggle](https://www.kaggle.com/code/hsiaosuan/gemma-developer-agent-0-13-submission) · [Local source](public/hsiaosuan/gemma-developer-agent-0-13-submission/readable.txt)

**Category: Published packaged configuration.** The base64 ZIP contains a SequentialAgent: read-only exploration outputs repair_plan, then a coder edits and submits. No LoRA; five minutes, 60 tool calls, 100 turns.

Reusable ideas: Short planning handoffs, bounded graph queries, and fixed bundle bytes/SHA256 provide a useful controlled comparison.

Caveats: The claimed 0.13 was not independently reproduced. Embedded YAML was decoded statically; include_thoughts=false means a nonzero thinking_budget does not establish enabled thinking.

Source locations: L1–L26.

## 5. Gemma 4 Developer Agent SWE Architecture

[Kaggle](https://www.kaggle.com/code/aryangupta85/gemma-4-developer-agent-swe-architecture) · [Local source](public/aryangupta85/gemma-4-developer-agent-swe-architecture/readable.txt)

**Category: Empty source entry.** Download succeeded, but public source content is empty.

Reusable ideas: Retain the entry so an empty file is not confused with a missed download or implementation evidence.

Caveats: Architecture, execution paths, and results cannot be reviewed; the title cannot fill in missing implementation.

Source locations: Downloaded source is empty.

## 6. The Gemma 4 Developer Agent Complete EDA

[Kaggle](https://www.kaggle.com/code/abelmatas/the-gemma-4-developer-agent-complete-eda) · [Local source](public/abelmatas/the-gemma-4-developer-agent-complete-eda/readable.txt)

**Category: Dataset exploration.** Inspects inventory, task/schema, diff sizes, graphs, NPZs, snapshots, wheels, and sample configuration; adds heuristic task categories using regular expressions.

Reusable ideas: Useful onboarding and distribution analysis with explicit fields, sizes, and label provenance.

Caveats: Category features mix issues with gold/test patches and belong to offline label analysis, not answer-free inputs. Token counts use character estimates, not a tokenizer. No agent-run or score evidence.

Source locations: L363–L374, L475–L515, L564–L650.

## 7. gemma4-agent-baseline-v2

[Kaggle](https://www.kaggle.com/code/juna1013/gemma4-agent-baseline-v2) · [Local source](public/juna1013/gemma4-agent-baseline-v2/readable.txt)

**Category: Optional single-agent baseline and evaluation.** Writes four main configuration files with four minutes, 60 tool calls, and 80 turns. An optional seeded five-task evaluator is disabled by default through RUN_LOCAL_EVAL=false.

Reusable ideas: Explicit budgets, fixed sampling, and a clear distinction between packaging and GPU experiments.

Caveats: Compilation uses dummy tool signatures, not full official-tool behavior. Completing the default notebook does not establish successful GPU evaluation.

Source locations: L1–L31, L266–L300, L350–L419.

## 8. Gemma 4-V0 | Walkthrough & First Submission

[Kaggle](https://www.kaggle.com/code/liujiahua2358/gemma-4-v0-walkthrough-first-submission) · [Local source](public/liujiahua2358/gemma-4-v0-walkthrough-first-submission/readable.txt)

**Category: Version walkthrough and baseline construction.** Inspects sample agents, example LoRA, graphs, and embeddings, then builds successive analyzer/coder versions and packages the result.

Reusable ideas: Track version changes and their sources; check graph output sizes, test-log tails, and wheel hashes.

Caveats: Older claims and later revisions coexist. Current wheel source controls thinking/seed forwarding; quoted leaderboard scores are not independent ablations.

Source locations: L1–L15, L275–L318, L380–L410, L630–L653.

## 9. Gemma 4 Where Is the Bug - Hop Distance Analysis

[Kaggle](https://www.kaggle.com/code/hajimekumami/gemma-4-where-is-the-bug-hop-distance-analysis) · [Local source](public/hajimekumami/gemma-4-where-is-the-bug-hop-distance-analysis/readable.txt)

**Category: Offline graph-localization analysis.** Embedded gzip/base64 analysis.py maps gold edit sites, issue anchors, shortest hops, retrieval, and leave-one-repository-out evaluation; also repairs commit-based filename lookup.

Reusable ideas: Choose localization settings with repository holdouts and separately measure missing anchors, missing gold nodes, and graph distance.

Caveats: Script text was decoded but not executed. The original uses clone/git show; offline adaptation should use supplied snapshots. Gold-derived labels must not enter the solving agent.

Source locations: L26–L45.

## 10. gemma4-agent-run1

[Kaggle](https://www.kaggle.com/code/sujanmajhisuzan/gemma4-agent-run1) · [Local source](public/sujanmajhisuzan/gemma4-agent-run1/readable.txt)

**Category: GPU adaptation of an embedded baseline.** Decodes an existing bundle and evaluates eight tasks. Chooses dtype by GPU capability and sets disable-custom-all-reduce for the T4 path.

Reusable ideas: Diagnose model startup, record per-task outcomes, and check bundle identity.

Caveats: Skips some official binary wheels and targets T4 behavior; this does not establish equivalence to official L4 scoring. Throughput and accuracy were not independently reproduced.

Source locations: L17–L30, L114–L169, L202–L244.

## 11. Gemma4-Agent（0.12points）

[Kaggle](https://www.kaggle.com/code/liuuu114514/gemma4-agent-0-12points) · [Local source](public/liuuu114514/gemma4-agent-0-12points/readable.txt)

**Category: EDA, lexical localization, and prompt construction.** Runs TF-IDF localization analysis, then writes coder/analyzer prompts and a ZIP. Prompts add reproduction, post-edit compilation, tests, and bounded retries.

Reusable ideas: Short localization reports and failure categories can support prompt ablations; keep EDA separate from real evaluation.

Caveats: The prompt applies git apply --check to a forward diff on an already edited tree and may checkout on failure, potentially undoing valid changes. Missing-data paths use synthetic demos. The title score of 0.12 is not our measurement.

Source locations: L187–L191, L1253–L1274, L1280–L1315.

## 12. Gemma 4 Agent Research Artifact

[Kaggle](https://www.kaggle.com/code/jek1wantaufik/gemma-4-agent-research-artifact) · [Local source](public/jek1wantaufik/gemma-4-agent-research-artifact/readable.txt)

**Category: Research artifact archive.** Reads experiment_card/submission/bundle artifacts attached to the buddy model entry, parses includes, hashes files, and copies them into paper_artifact.

Reusable ideas: Per-file sources, configuration, and hashes improve paper and reproduction handoffs.

Caveats: Inputs are previously saved third-party artifacts. Copying and manifest checks do not prove training/evaluation, nor authorize the associated Models-page entry as a replacement base.

Source locations: L18–L38, L535–L583, L765–L813.

## 13. gemma4-data-eda

[Kaggle](https://www.kaggle.com/code/cylik93/gemma4-data-eda) · [Local source](public/cylik93/gemma4-data-eda/readable.txt)

**Category: Retrieval and graph-coverage exploration.** Uses public gold diffs to locate changed lines, measure grep coverage, file lengths, missing graph nodes, and truncated tool outputs; writes per_task.csv.

Reusable ideas: Measure the gap between a target appearing in retrieval and remaining visible after output truncation.

Caveats: Verify tool limits such as 5000 characters/150 lines against the current harness. Empty hints in the public set do not guarantee hidden behavior. Gold is used as labels; no complete SWE agent is run.

Source locations: L64–L68, L92–L118, L202–L221, L469–L487.

## 14. Gemma 4 Developer Agent - Lean

[Kaggle](https://www.kaggle.com/code/jek1wantaufik/gemma-4-developer-agent-lean) · [Local source](public/jek1wantaufik/gemma-4-developer-agent-lean/readable.txt)

**Category: Existing model-page artifact inspection and repackaging.** Selects submission or submission_bundle from buddy inputs, reads experiment_card, YAML, and ZIP hashes, then copies and repackages them.

Reusable ideas: Track configuration provenance, includes, and archive consistency.

Caveats: No new training/inference entry; depends on separately attached third-party artifacts. A model-page name is not an approved-base check.

Source locations: L12–L38, L255–L278, L378–L425.

## 15. Gemma4 Prompt Improvement Agent

[Kaggle](https://www.kaggle.com/code/kevlarzanderchi/gemma4-prompt-improvement-agent) · [Local source](public/kevlarzanderchi/gemma4-prompt-improvement-agent/readable.txt)

**Category: Prompt variants and fallback bundles.** Produces v12, original v10, and v11 ZIPs. v12 adds an inline AST helper and environment notes to an existing system prompt, with explicit differences elsewhere.

Reusable ideas: Keep exact anchors and isolated changes, verifying comparisons with file hashes.

Caveats: All three omit eval_config and use scoring defaults. A custom validator does not replace official compilation or real tasks. Nonzero thinking_budget coexists with include_thoughts=false.

Source locations: L1–L12, L15–L45, L69–L128.

## 16. Gemma + Gemini - The Ultimate Duo

[Kaggle](https://www.kaggle.com/code/matterhorn3838/gemma-gemini-the-ultimate-duo) · [Local source](public/matterhorn3838/gemma-gemini-the-ultimate-duo/readable.txt)

**Category: Baseline profiles and exploration.** Defaults to top_bundle. The custom branch toggles a single agent, graph tools, prompt rules, and sampling; packages each profile with diffs/hashes.

Reusable ideas: Explicit switches and byte comparisons against a known anchor support controlled ablations.

Caveats: Despite the Gemini title, the inspected submission builds the required Gemma configuration. Its interpretation of include_thoughts=false as display-only does not match this harness. Default eval_config is absent; quoted scores are not causal evidence.

Source locations: L80–L113, L1880–L1923, L2144–L2156.

## 17. pass2

[Kaggle](https://www.kaggle.com/code/abelog/pass2) · [Local source](public/abelog/pass2/readable.txt)

**Category: Graph-expansion research prototype.** Compares centroid retrieval, fixed hops, and adaptive expansion, measuring reference/test-patch file recall and context size.

Reusable ideas: Separate retrieval hits from context cost before connecting real issue queries and SWE tests.

Caveats: The query is vectors.mean(axis=0), without issue text. This diagnoses infrastructure rather than issue-to-code semantic localization or end-to-end resolution.

Source locations: L1003–L1044, L1355–L1437.

## 18. gemma4-graph-coverage-notebook

[Kaggle](https://www.kaggle.com/code/fawad07/gemma4-graph-coverage-notebook) · [Local source](public/fawad07/gemma4-graph-coverage-notebook/readable.txt)

**Category: CPU replay of paper results.** Copies scripts and rebuilt graphs from an attached reproduction dataset, recomputes coverage/retrieval tables, and reads saved summaries for agent comparisons.

Reusable ideas: Label recomputed CPU metrics separately from cited GPU-run results.

Caveats: Key implementation and caches come from another dataset. This notebook does not rerun the L4 agent, so report replay is not independent patch/test reproduction.

Source locations: L1–L10, L25–L42, L65–L81.

## 19. repair-manifold-ablation

[Kaggle](https://www.kaggle.com/code/dalezhong/repair-manifold-ablation) · [Local source](public/dalezhong/repair-manifold-ablation/readable.txt)

**Category: Repair-prompt ablation with known locations.** Compares free generation with operator-first prompts on few-hunk tasks, obtains target files from gold patches, and asks the model for whole-file replacements.

Reusable ideas: A controlled local-repair study given the correct file; retain model identity, iteration counts, and test status.

Caveats: Uses gold file paths and applies test_patch in advance, so it is not answer-free end-to-end localization. A failed load falls back to Qwen, outside the required-model result. Network installs and file truncation need separate treatment.

Source locations: L117–L196, L260–L288, L300–L381.

## 20. PatchPilot Gemma 4

[Kaggle](https://www.kaggle.com/code/shorooghahmadi/patchpilot-gemma-4) · [Local source](public/shorooghahmadi/patchpilot-gemma-4/readable.txt)

**Category: Configuration packaging with optional CLI evaluation.** A coder calls an analyzer and uses a repo_navigation skill. Writes YAML and custom checks, builds a ZIP, and supplies an optional swegemma eval command.

Reusable ideas: Inspect submission layout/includes and distinguish uploaded ZIPs from intermediate scoring parquet.

Caveats: max_time_minutes=30 would far exceed 12 hours for 120 fully budgeted tasks. Tighten and measure it. Old path detection only scans one input level. Packaging checks do not establish official scoring success.

Source locations: L1–L22, L58–L87.

## 21. Gemma 4: A 16384 Output Cap Refuses Long Prompts

[Kaggle](https://www.kaggle.com/code/dariushafshar/gemma-4-a-16384-output-cap-refuses-long-prompts) · [Local source](public/dariushafshar/gemma-4-a-16384-output-cap-refuses-long-prompts/readable.txt)

**Category: Context-budget and saved-trace analysis.** Uses embedded call counts and author records to analyze a 32768-token context, 16384-token output cap, compaction delay, and time headroom; exports budget tables.

Reusable ideas: Record actual prompt/output tokens and include compaction cost in timing budgets.

Caveats: Reanalyzes author traces rather than running inference every time. Estimates of 58 public/116 total tasks are not official guarantees. Verify title claims against current vLLM requests and versions.

Source locations: L1–L43, L304–L354, L619–L649, L690–L729.

## 22. Code graph audit

[Kaggle](https://www.kaggle.com/code/akhilchinta1505/code-graph-audit) · [Local source](public/akhilchinta1505/code-graph-audit/readable.txt)

**Category: Source-graph reconstruction and coverage audit.** An AST builder adds async/nested definitions, import resolution, and contains edges. Gold edit ranges provide labels; recorded calls and harness graph lookup are replayed.

Reusable ideas: Compare graph coverage, actual-call recall, tool output, and retrieval benefit separately; retain source/data licenses.

Caveats: Recorded calls come from an attached dataset. Graph metrics do not equal SWE resolution; rebuilding cost and permitted runtime integration still need validation.

Source locations: L1–L14, L53–L88, L294–L368, L661–L679, L778–L800.

## 23. GemmaLoc: edit-site labels and a Gemma 4 rerank

[Kaggle](https://www.kaggle.com/code/giuseppecastelluccio/gemmaloc-edit-site-labels-and-a-gemma-4-rerank) · [Local source](public/giuseppecastelluccio/gemmaloc-edit-site-labels-and-a-gemma-4-rerank/readable.txt)

**Category: Localization labels and cached LLM-reranking replay.** Labels nodes from gold diffs and scores eight retrievers. Defaults to exact-prompt-hash Gemma caches, with a LIVE_URL option and an external SWE-bench subset.

Reusable ideas: Check label alignment, fail on cache misses, use paired comparisons, and test cross-repository generalization.

Caveats: Default execution is cached, requiring additional inputs; live clone operations cannot be copied directly into offline L4 sessions. Doubled top-1 localization is not doubled repair success.

Source locations: L1–L10, L542–L601, L823–L859, L1628–L1636.

## 24. Vibranium-Bench: Deterministic SWE Benchmarking fo

[Kaggle](https://www.kaggle.com/code/thurshwinc/vibranium-bench-deterministic-swe-benchmarking-fo) · [Local source](public/thurshwinc/vibranium-bench-deterministic-swe-benchmarking-fo/readable.txt)

**Category: Graph-pruning and evaluation-interface demonstration.** Provides AST-graph and budgeted-traversal examples. Context-reduction tables use fixed formulas; the agent returns synthetic_patch and sleeps.

Reusable ideas: Useful teaching interfaces/data structures; replace them with real graphs, inference, and verification for experiments.

Caveats: Tests cover simple arithmetic, timing comes from sleep, and token reduction is not measured. Outputs cannot support speed, VRAM, or SWE-performance claims.

Source locations: L355–L411, L458–L526.

## 25. Dhia-Gemma4AA

[Kaggle](https://www.kaggle.com/code/dhiaalhemdani/dhia-gemma4aa) · [Local source](public/dhiaalhemdani/dhia-gemma4aa/readable.txt)

**Category: Small derivative of the official starter.** Changes the sample source to the author dataset and max_output_tokens from 16384 to 8192; the remaining execution structure matches the archived starter.

Reusable ideas: Compare one setting against a fixed starter.

Caveats: Actual agent content comes from another sample-submission dataset. Similar notebook code does not establish identical submissions; inspect input hashes and results.

Source locations: L70–L88.

## 26. notebook46419a07d8

[Kaggle](https://www.kaggle.com/code/ayushmldixit1/notebook46419a07d8) · [Local source](public/ayushmldixit1/notebook46419a07d8/readable.txt)

**Category: Model-loading debugging draft.** Attempts Transformers multimodal Gemma4 AutoModel loading, device maps, and CPU offload; also includes a vllm serve command.

Reusable ideas: Class/processor selection, device maps, and VRAM diagnostics can inform debugging.

Caveats: Contains a SIGKILL cell terminating the kernel and repeated loading fragments; unsuitable for Run All. No complete agent ZIP or official-task validation path.

Source locations: L156–L182, L323–L368.

## 27. SurgiGraph-Paper-Experiments

[Kaggle](https://www.kaggle.com/code/yukailiu/surgigraph-paper-experiments) · [Local source](public/yukailiu/surgigraph-paper-experiments/readable.txt)

**Category: Paper figures and statistical-model analysis.** Includes data summaries, Dirichlet hotspot posteriors, embedded localization results, and patch-simplicity comparisons for the paper track.

Reusable ideas: Turn hypotheses such as minimal edits and context localization into independent ablations.

Caveats: Some token costs are modeled, and repair/regression tables use hazard-model derivations rather than actual agent tests in this notebook. Do not report those estimates as measured gains.

Source locations: L1288–L1318, L1380–L1391, L1425–L1450, L1719–L1728.

## 28. notebook703cae6987

[Kaggle](https://www.kaggle.com/code/phmtin1306/notebook703cae6987) · [Local source](public/phmtin1306/notebook703cae6987/readable.txt)

**Category: Manual data inspection.** Finds the competition directory and inspects a snapshot, graph/embedding files, and wheel internals.

Reusable ideas: Quick introduction to layouts and archive types.

Caveats: Data browsing only: no training, agent configuration, or real-task evaluation. Example task names differ from this competition and cannot be used directly.

Source locations: L1–L25, L160–L194.

## 29. Gemma4 Candidate v7 Compile

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-candidate-v7-compile) · [Local source](public/eeshsaxena/gemma4-candidate-v7-compile/readable.txt)

**Category: Official compilation check for a candidate bundle.** Decodes an embedded ZIP, installs the wheelhouse, builds a model registry and signature-compatible stub tools, and calls compile_submission.

Reusable ideas: Distinguish official compilation from path/extension checks and retain the ZIP SHA256.

Caveats: Tools are ellipsis stubs and the endpoint is a placeholder; no inference occurs. The decoded configuration omits eval_config and uses default budgets, with include_thoughts=false.

Source locations: L5–L12, L17–L44.

## 30. Gemma4 Baseline Agent v2 Compile Fixed

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-baseline-agent-v2-compile-fixed) · [Local source](public/eeshsaxena/gemma4-baseline-agent-v2-compile-fixed/readable.txt)

**Category: Compilation repair for a v2 configuration.** Decodes files individually, invokes the official compiler, and packages coder/analyzer configurations with five minutes, 48 tools, and 100 turns.

Reusable ideas: Signature-aware compilation prechecks help diagnose includes and schema errors.

Caveats: Endpoint and tools remain placeholders; compilation is not issue resolution. ZIP timestamps are not fixed.

Source locations: L4–L15, L19–L40.

## 31. Gemma4 Baseline Agent v2 Packaging

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-baseline-agent-v2-packaging) · [Local source](public/eeshsaxena/gemma4-baseline-agent-v2-packaging/readable.txt)

**Category: Packaging-only v2 configuration.** Decodes coder/analyzer files, checks root agent.yaml, and compresses them; thinking_budget=0, five minutes, 48 tools, 100 turns.

Reusable ideas: A minimal CPU path for isolating upload-format problems.

Caveats: No official compiler, model, or tests. Distinguish it from the compile-fixed version. exist_ok may retain stale files; use a clean directory and deterministic archive checks.

Source locations: L1–L19.

## 32. Google-Gemma4: Multi-Agent + Graph-RAG  Base

[Kaggle](https://www.kaggle.com/code/dedquoc/google-gemma4-multi-agent-graph-rag-base) · [Local source](public/dedquoc/google-gemma4-multi-agent-graph-rag-base/readable.txt)

**Category: Successive drafts ending in sample-adapter packaging.** Repeatedly overwrites submission.zip. The last block copies sample main_lora/tool_lora, writes root/analyzer agents, skills, and configuration, then packages them.

Reusable ideas: Inspect the final payload actually written instead of relying on descriptions of earlier drafts.

Caveats: Copying sample adapters is not training. Final timeout_seconds=900 limits one command, not a 15-minute task. Some displays/retrieval use mock fallbacks; skills and final configuration need official compilation and runtime checks.

Source locations: L340–L376, L705–L825.

## 33. Gemma4 Baseline Agent

[Kaggle](https://www.kaggle.com/code/eeshsaxena/gemma4-baseline-agent) · [Local source](public/eeshsaxena/gemma4-baseline-agent/readable.txt)

**Category: CPU-only two-agent packaging.** Writes coder, read-only analyzer, prompts, and eval_config with five minutes, 32 tools, and 80 turns, then checks ZIP root and size.

Reusable ideas: Small readable starting configuration with explicit task budgets and patch submission.

Caveats: No official compiler or Evaluator call. Layout/size checks cannot establish tool-signature or runtime compatibility. thinking_budget=0 and include_thoughts=false.

Source locations: L11–L46, L48–L79.

## 34. notebook-agent.ipynb

[Kaggle](https://www.kaggle.com/code/harigov63/notebook-agent-ipynb) · [Local source](public/harigov63/notebook-agent-ipynb/readable.txt)

**Category: Teaching simulation with dataset-upload code.** Looks for test.csv, creates dummy rows when absent, and writes a fixed mock_fix into submission.csv. Another function simulates inference through prints and sleep.

Reusable ideas: Use as a negative example when checking real input schemas, output formats, and model calls.

Caveats: This competition requires an agent ZIP. Fixed patches/CSV are not substitutes. The final dataset_upload has external side effects; do not Run All. The model name also differs from the required base.

Source locations: L30–L72, L84–L104, L116–L146.

## 35. graphforge-digest

[Kaggle](https://www.kaggle.com/code/vinay0005/graphforge-digest) · [Local source](public/vinay0005/graphforge-digest/readable.txt)

**Category: Compatible graph extraction and localization benchmark.** Implements GraphForge Lite, then compares issue-query BM25, hashing, BGE, RRF, and graph propagation against gold-diff labels.

Reusable ideas: Compare graph/no-graph and lexical/embedding retrieval, retaining exclusion reasons.

Caveats: Excludes graphs without gold nodes or with fewer than 30 nodes, so results cover a selected subset. Compatibility mode omits async. ALLOW_PIP=true and BGE downloads need staging. Localization hits are not repair success.

Source locations: L1–L16, L1103–L1135, L1400–L1485.

## 36. Gemma 4 SWE: Task Map & 12-Hour Budget

[Kaggle](https://www.kaggle.com/code/officialabhinavsingh/gemma-4-swe-task-map-12-hour-budget) · [Local source](public/officialabhinavsingh/gemma-4-swe-task-map-12-hour-budget/readable.txt)

**Category: Task distribution and total-budget scenarios.** Measures reference-patch sizes and computes scenarios with 720 total minutes, a 90-minute setup reserve, and several task caps.

Reusable ideas: Budget setup and task execution together; a cap is not a target runtime.

Caveats: evaluation_tasks=len(df) is explicitly a public-set scenario, not the hidden count. Patch size and issue length do not directly predict solvability.

Source locations: L75–L118.

## 37. Gemma 4 LB: 58 Tasks, Bronze Is a 124-Team Tie

[Kaggle](https://www.kaggle.com/code/dariushafshar/gemma-4-lb-58-tasks-bronze-is-a-124-team-tie) · [Local source](public/dariushafshar/gemma-4-lb-58-tasks-bronze-is-a-124-team-tie/readable.txt)

**Category: Historical leaderboard quantization and private-board simulation.** Infers public task count from rounded scores, fits an empirical-Bayes prior, and simulates private scores/ties under different correlations.

Reusable ideas: Understand quantization, ties, and best-submission selection bias; retain assumptions and sensitivity analysis.

Caveats: The inferred 58 tasks are unconfirmed. Uses an old day-six snapshot of 863 teams. Private results are unobserved, and independence, equal split sizes, and correlations are modeling assumptions.

Source locations: L697–L723, L862–L881.

## 38. Hidden Interfaces – audit and execution.

[Kaggle](https://www.kaggle.com/code/murtadhaabdabbas/hidden-interfaces-audit-and-execution) · [Local source](public/murtadhaabdabbas/hidden-interfaces-audit-and-execution/readable.txt)

**Category: Hidden-interface and evaluation-soundness audit.** Reconstructs historical worktrees and compares empty, gold, and interface-renamed patches while recording import paths, dependencies, tests, and failure types.

Reusable ideas: Use controls to separate repair capability from environment defects and confirm tests import the intended worktree.

Caveats: Merge-time/PR-number reconstruction needs comparison with actual snapshots. Online mode installs dependencies. No repair model runs; gold variants are audit controls, not predictions.

Source locations: L242–L271, L337–L358, L434–L466.

## 39. Gemma 4: Measure Before You Tune

[Kaggle](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-measure-before-you-tune) · [Local source](public/dmitriigluzdov/gemma-4-measure-before-you-tune/readable.txt)

**Category: Existing-result audit and LoRA decision record.** Checks external CSVs/manifests and presents 114/129 sound tasks, repository holdouts, prompt-rendering checks, and a LoRA NO-GO decision; packages an existing public baseline.

Reusable ideas: Check adapter effect, rendering consistency, and resolution before training conclusions; compare lower loss against unresolved trajectories.

Caveats: Charts replay historical results without retraining. Of 936 checks, only five compare token identity; the rest compare lengths. Author results show adapter 2/32 versus base 4/32. NLL reduction is not score improvement, and soundness depends on that historical environment.

Source locations: L24–L66, L107–L118, L165–L171, L220–L243.

## 40. GTUA - Stability-Augmented Graph-Tool Utility Audi

[Kaggle](https://www.kaggle.com/code/norikazuhaji/gtua-stability-augmented-graph-tool-utility-audi) · [Local source](public/norikazuhaji/gtua-stability-augmented-graph-tool-utility-audi/readable.txt)

**Category: Hyperparameter-stability audit of graph tools.** Compares TF-IDF plus one-hop propagation across 30 settings, classifying gold-file localization changes as Help/Hurt/Neutral and stability at 80% agreement.

Reusable ideas: Test graph benefit across parameters and repositories instead of relying on one favorable setting.

Caveats: Localization is a proxy and no repair agent runs. The 80% and gain thresholds are methodological choices, not official scoring criteria.

Source locations: L209–L245, L265–L310.

## 41. Gemma 4 Triad LB score 0.03

[Kaggle](https://www.kaggle.com/code/kozykappa/gemma-4-triad-lb-score-0-03) · [Local source](public/kozykappa/gemma-4-triad-lb-score-0-03/readable.txt)

**Category: Coder/localizer/critic configuration packaging.** Root calls localization and review agents. All three disable thinking; budget is eight minutes, 100 tools, 140 turns, with custom static checks and fixed ZIP metadata.

Reusable ideas: Bound extra calls to one localization and one review pass; deterministic packaging aids version comparisons.

Caveats: The 0.03 title is author-reported. Custom checks do not validate official execution. Eight minutes times about 120 tasks is 16 hours. Read-only duties are prompt instructions; run_command itself can write.

Source locations: L318–L395, L396–L420, L503–L535.

## 42. Google-TheGemma4DeveloperAgentCompetition

[Kaggle](https://www.kaggle.com/code/rifat2022/google-thegemma4developeragentcompetition) · [Local source](public/rifat2022/google-thegemma4developeragentcompetition/readable.txt)

**Category: Historical graph repair and retrieval audit tools.** Contains alias/empty-file repairs, source-symbol enumeration including async, graph rebuilding, and retrieval-coverage reports.

Reusable ideas: Distinguish missing publication files from missing graph structure; perform a census before repairs.

Caveats: The 256-file/hardlink narrative targets an older release. Our 127 nonempty graph/embedding pairs map all 129 tasks. Do not copy old conclusions or modify original inputs blindly. Coverage is not agent repair benefit.

Source locations: L1–L24, L355–L400, L506–L542, L984–L1015.

## 43. Eval errors - Gemma 4 Developer Agent

[Kaggle](https://www.kaggle.com/code/bharat0/eval-errors-gemma-4-developer-agent) · [Local source](public/bharat0/eval-errors-gemma-4-developer-agent/readable.txt)

**Category: Gold-patch isolation of the verification phase.** Disables model startup and replaces Evaluator._run_agent_sandbox with a function returning task.patch and an empty trace to diagnose phase two.

Reusable ideas: Test whether a reference solution passes in the chosen environment and locate installation/verifier defects.

Caveats: No agent runs. Gold pass rate is not model performance. Keep this monkeypatch in isolated control experiments, outside formal prediction.

Source locations: L161–L194, L218–L240.

## 44. Gemma 4 Dev Agent: task anatomy + survival kit

[Kaggle](https://www.kaggle.com/code/osama133/gemma-4-dev-agent-task-anatomy-survival-kit) · [Local source](public/osama133/gemma-4-dev-agent-task-anatomy-survival-kit/readable.txt)

**Category: Public-task anatomy and onboarding.** Estimates source-file, hunk, changed-line, function, and async features from gold diffs, plots issue cues, and supplies path/export helpers.

Reusable ideas: Useful team onboarding and change-scope analysis, emphasizing matched tasks and repeated runs.

Caveats: Function attribution uses hunk context and may misassign or miss functions. Gold is offline analysis only. No inference or scoring evidence; small patches do not imply easy issues.

Source locations: L156–L203, L440–L449, L455–L489.

## 45. Gemma 4 Baseline Reproduction

[Kaggle](https://www.kaggle.com/code/ahmadagbaria/gemma-4-baseline-reproduction) · [Local source](public/ahmadagbaria/gemma-4-baseline-reproduction/readable.txt)

**Category: EDA, lexical localization, and optional two-agent packaging.** Parses reference diffs, runs TF-IDF localization, and constructs single-agent or analyzer/coder configurations with custom validation, deterministic ZIP, and experiment_card.

Reusable ideas: Archive source, configuration, and hashes together; separate localization metrics from submitted behavior.

Caveats: Byte-identical to the romanrozen entry, not an independent reproduction. No official GPU evaluation. Natural-language search_similar_code prompts conflict with the indexed-symbol lookup implementation. Token counts are character estimates.

Source locations: L1218–L1304, L1449–L1506, L1573–L1635, L1664–L1674.

## 46. Gemma 4 Developer Agent Top 1 Master Baseline

[Kaggle](https://www.kaggle.com/code/tharunkumar369/gemma-4-developer-agent-top-1-master-baseline) · [Local source](public/tharunkumar369/gemma-4-developer-agent-top-1-master-baseline/readable.txt)

**Category: Single-agent prompts and static packaging.** Gives nine tools directly to a coder with 3.5 minutes, 32 tools, and 40 turns, then packages after custom structural checks.

Reusable ideas: Short budgets, precise small edits, syntax checks, and final submit_patch.

Caveats: Top-1/SOTA and sweet-spot claims lack measured ablations. temperature=0.15 is not deterministic; include_thoughts=false disables thinking. The checker does not validate tool arguments; verify prompt examples against official signatures.

Source locations: L139–L178, L190–L240, L263–L307, L344–L351.

## 47. Gemma4 Day1 Baseline

[Kaggle](https://www.kaggle.com/code/tomaszszyrowski/gemma4-day1-baseline) · [Local source](public/tomaszszyrowski/gemma4-day1-baseline/readable.txt)

**Category: Minimal packaging with optional model Q&A smoke tests.** Creates a two-file ZIP with optional graph tools, plus local Ollama Q&A and HF loading/generation for the specified checkpoint.

Reusable ideas: Separate ZIP structure, official validation, model Q&A, and complete repair verification.

Caveats: RUN_HF_SMOKE is actually True despite a disabled comment, installing/downloading and requiring 32 GiB free on one GPU. Ollama is a different runtime. No official tool agent or repository tests run.

Source locations: L47–L83, L91–L130, L133–L180, L181–L206.

## 48. Gemma 4 Dev Agent: Zero to First Valid Submission

[Kaggle](https://www.kaggle.com/code/cocoyachi/gemma-4-dev-agent-zero-to-first-valid-submission) · [Local source](public/cocoyachi/gemma-4-dev-agent-zero-to-first-valid-submission/readable.txt)

**Category: First-submission guide using official validation helpers.** Installs the wheelhouse and builds one agent with five minutes, 40 tools, and 100 turns. Calls model-declaration/directory validation and computes budgets before writing a ZIP.

Reusable ideas: Directory conventions and official validation functions provide stronger checks than ZIP existence alone.

Caveats: No actual GPU agent execution. The budget discussion includes phase-two verification in the 12 hours, contrary to current rules. Use our verified file-upload route for submission details.

Source locations: L147–L176, L214–L271.

## 49. GTUA - Graph-Tool Utility Audit

[Kaggle](https://www.kaggle.com/code/norikazuhaji/gtua-graph-tool-utility-audit) · [Local source](public/norikazuhaji/gtua-graph-tool-utility-audit/readable.txt)

**Category: Graph-propagation localization benchmark.** Adds one-hop propagation to TF-IDF; excludes same-file or already-gold-hit seeds before measuring cross-file rescue across 30 settings and repository interactions.

Reusable ideas: Avoid crediting graph propagation for existing seed hits; freeze the grid and retain negative results.

Caveats: Task-level p-values do not establish cross-repository generalization. No agent/leaderboard run. This is the predecessor of the stability audit, not an independent replication.

Source locations: L180–L243, L294–L322.

## 50. notebookllms.io

[Kaggle](https://www.kaggle.com/code/harigov63/notebookllms-io) · [Local source](public/harigov63/notebookllms-io/readable.txt)

**Category: Markdown-only simulation snippets.** Defines mock_swegemma_inference as displayed text: prints stages, sleeps, and returns a fixed patch. All three cells are Markdown.

Reusable ideas: Distinguish displayed code, executable cells, and genuine inference calls.

Caveats: No execution cells, model loading, evaluation, or submission ZIP. The title and Framework online print string are not runtime evidence.

Source locations: L1–L37.

## 51. Gemma 4 SuperAgent

[Kaggle](https://www.kaggle.com/code/matterhorn3838/gemma-4-superagent) · [Local source](public/matterhorn3838/gemma-4-superagent/readable.txt)

**Category: Public configuration comparison and repackaging.** Defaults to the unchanged top_bundle; custom enables architecture, issue prompts, graph tools, and sampling changes. Retains version hashes and a custom validator.

Reusable ideas: Assert changes explicitly and distinguish unchanged defaults from custom experiments.

Caveats: Scores such as 0.12/0.15 are quoted author claims. include_thoughts=false is incorrectly described as display-only; it disables thinking here. Missing eval_config uses harness defaults.

Source locations: L40–L100, L1851–L1895.

## 52. Google - The Gemma 4 Developer 5

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer-5) · [Local source](public/chaitanyajamble/google-the-gemma-4-developer-5/readable.txt)

**Category: Budget calculation and navigation-skill packaging.** Builds coder/analyzer plus a source-localization skill, optionally copies existing adapters, and attempts CLI evaluation without making failure block packaging.

Reusable ideas: Avoid fabricated adapters; distinguish copied weights from training and retain budgets.

Caveats: MIN_TASK_MINUTES=5 can exceed the total: default 300 tasks requires at least 25 hours. Public 129 is not the hidden count. CLI presence does not imply a ready endpoint; check=False swallows errors, so packaging success is not a passed smoke test.

Source locations: L45–L66, L469–L505, L539–L566, L615–L644, L711–L766.

## 53. Gemma 4 Dev Agent: What Moves the Leaderboard

[Kaggle](https://www.kaggle.com/code/hitarthjain0/gemma-4-dev-agent-what-moves-the-leaderboard) · [Local source](public/hitarthjain0/gemma-4-dev-agent-what-moves-the-leaderboard/readable.txt)

**Category: Competition walkthrough and two-agent packaging.** Summarizes tools, budgets, and failure modes; builds analyzer/coder with a lightweight custom validator, 8192 output tokens, and thinking disabled.

Reusable ideas: Explain the submission contract and separate experiment logs from payload.

Caveats: Graph-first benefit and public scores are not paired ablations. Claims that LoRA still fails are historical and need current-wheel/organizer checks. Structural checks are neither official compilation nor scoring.

Source locations: L106–L123, L150–L184, L233–L286, L316–L332.

## 54. Gemma 4 AST Graph-Augmented SWE Multi-Agent System

[Kaggle](https://www.kaggle.com/code/avikdas567/gemma-4-ast-graph-augmented-swe-multi-agent-system) · [Local source](public/avikdas567/gemma-4-ast-graph-augmented-swe-multi-agent-system/readable.txt)

**Category: EDA, two-agent configuration, and optional evaluation.** Copies sample main/tool adapters and sets a 20-minute task budget. If serving starts, evaluates two tasks; otherwise writes empty predictions.

Reusable ideas: Inspect genuine Evaluator calls and output fields together with failure branches.

Caveats: Missing adapters trigger 512-byte zero-filled fake safetensors, which must not be used. Context curves/transition matrices are simulated constants. T4×2 prose is not official L4×4 hardware, and 20-minute task caps do not fit the global budget.

Source locations: L783–L803, L868–L888, L945–L987, L1011–L1052, L1113–L1188.

## 55. GraphLoc end-to-end: failure analysis + resource

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-failure-analysis-resource) · [Local source](public/zzgtylors/graphloc-end-to-end-failure-analysis-resource/readable.txt)

**Category: Failure attribution and variance analysis of saved GraphLoc runs.** Reads multiple base/skill artifacts, matches patches to gold files/functions, parses test failures, and performs paired task bootstrap with summary exports.

Reusable ideas: Retain traces, errors, actual patches, and paired outcomes to distinguish localization from repair failures.

Caveats: Does not rerun inference and requires complete attached outputs. Regex/time-threshold failure labels are heuristic. Task bootstrap does not establish repository generalization, and gold-location hits are not resolution.

Source locations: L1–L9, L92–L137, L167–L245, L318–L349.

## 56. GraphLoc end-to-end: repeat run 3

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-repeat-run-3) · [Local source](public/zzgtylors/graphloc-end-to-end-repeat-run-3/readable.txt)

**Category: Paired official evaluation of a GraphLoc skill.** Implements offline BM25 plus rebuilt-graph PageRank localization. Matched base/skill settings evaluate gold-passing tasks with 12 workers sharing a TP4 model.

Reusable ideas: Real end-to-end comparisons, matched-task statistics, repeats, and full traces are closer to the competition objective than retrieval alone.

Caveats: Gold-only filtering does not establish empty-patch failure. Added dependencies and Pygments changes alter the environment. Twelve-worker timing differs from sequential scoring. Three identical source entries are not three independently verified runs.

Source locations: L294–L344, L968–L1021, L1088–L1158, L1194–L1209, L1399–L1452.

## 57. GraphLoc end-to-end: repeat run 2

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-repeat-run-2) · [Local source](public/zzgtylors/graphloc-end-to-end-repeat-run-2/readable.txt)

**Category: Paired official evaluation of a GraphLoc skill.** Source SHA256 matches entry 56, zzgtylors/graphloc-end-to-end-repeat-run-3. Uses offline BM25/rebuilt-graph PageRank, matched budgets, gold-passing tasks, and 12 workers sharing TP4.

Reusable ideas: End-to-end comparisons, matched-task statistics, repeated runs, and full traces are more informative than retrieval alone.

Caveats: Identical source is not independent execution/score evidence. Gold-only filtering does not prove empty failure; dependency/Pygments changes affect the environment. Concurrent timing is not sequential scoring, and three identical entries do not validate three independent runs.

Source locations: L294–L344, L968–L1021, L1088–L1158, L1194–L1209, L1399–L1452.

## 58. Gemma 4 | Budget-Aware Coding Agent

[Kaggle](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent) · [Local source](public/flexonafft/gemma-4-budget-aware-coding-agent/readable.txt)

**Category: Two-agent baseline with explicit budgets.** Retains public prompt/sampling anchors, adds coder navigation tools, and sets 4.5 minutes, 40 tools, and 80 turns; checks local includes, Apache attribution, and ZIP layout.

Reusable ideas: Preserve upstream licensing and explicit budgets; optionally check official ADK schemas when installed.

Caveats: The author explicitly reports no measured score for this candidate. Optional schema checks skip when packages are absent. Thinking is disabled, and L4 timing still needs measurement.

Source locations: L1–L44, L397–L447.

## 59. Gemma 4 Developer Agent CPU Audit Tables

[Kaggle](https://www.kaggle.com/code/nitish5236goel/gemma-4-developer-agent-cpu-audit-tables) · [Local source](public/nitish5236goel/gemma-4-developer-agent-cpu-audit-tables/readable.txt)

**Category: Historical release audit and power-table regeneration.** Embeds original audit records/code, recomputes gold-symbol/graph/power tables, compares the old release with the September 25 reissue, and checks source-commit hashes.

Reusable ideas: Distinguish newly recomputed tables from historical verifier/tool probes; establish real test execution and required-case success.

Caveats: The 1210+32 recorded verifications are not rerun. Historical empty-graph conclusions do not describe current data. Power estimates assume models such as independent tasks, not measured agent scores or guaranteed future gains.

Source locations: L1–L34, L869–L924, L1093–L1171, L3495–L3510, L3561–L3591, L3657–L3668.

## 60. G4CA_Basv1.0

[Kaggle](https://www.kaggle.com/code/ckskaggle/g4ca-basv1-0) · [Local source](public/ckskaggle/g4ca-basv1-0/readable.txt)

**Category: Restricted two-agent baseline and custom rule checker.** Coder retains navigation and calls an analyzer at most once. Prompts emphasize early edits and short checks; custom validation precedes packaging, with dynamic official-helper discovery if installed.

Reusable ideas: Separate schema availability, structural checks, and runtime evaluation; record prompts and tool permissions.

Caveats: Official validation exceptions are printed and ignored, so this is not a passing gate. No real GPU evaluation or eval_config. Claims about tests always being discarded require current-harness checks.

Source locations: L37–L128, L249–L327.

## 61. TFD-Agent: Autonomous SWE-bench Resolution with Ge

[Kaggle](https://www.kaggle.com/code/rajrajak99/tfd-agent-autonomous-swe-bench-resolution-with-ge) · [Local source](public/rajrajak99/tfd-agent-autonomous-swe-bench-resolution-with-ge/readable.txt)

**Category: Paper-style architecture and simulated objects.** CodebaseEnvironment manipulates strings in a dictionary. Four hard-coded scores and a two-agent diagram appear without Gemma or official evaluation.

Reusable ideas: Check numerical provenance and distinguish reported historical scores from projections.

Caveats: 0.13/rank are author claims. M4 is explicitly projected but plotted as 0.18, not a measured result. No complete runnable agent or submission ZIP.

Source locations: L1–L16, L59–L112.

## 62. Gemma 4 Developer Agent: End-to-End ADK Starter

[Kaggle](https://www.kaggle.com/code/koushikkumardinda/gemma-4-developer-agent-end-to-end-adk-starter) · [Local source](public/koushikkumardinda/gemma-4-developer-agent-end-to-end-adk-starter/readable.txt)

**Category: Incomplete ADK teaching and packaging draft.** Empty-graph handling increments restored_count while actual copying is commented out; vLLM startup is also commented out. Writes inline sub_agents/as_tool configuration and a ZIP.

Reusable ideas: A useful example for checking whether described steps actually execute.

Caveats: Restored messages do not prove repairs. eval_config is placed in configs/ rather than the root and may be ignored. The subagent schema was not checked with official validation; end-to-end success is unestablished.

Source locations: L62–L85, L95–L115, L126–L172.

## 63. Can Gemma 4 Fix an Unseen Repo? | 31B QLoRA Study

[Kaggle](https://www.kaggle.com/code/denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study) · [Local source](public/denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study/readable.txt)

**Category: QLoRA training study with oracle localization.** Uses Unsloth 31B bnb4bit, rank 8, 1024 tokens, one SFT epoch, a Rich repository holdout, and prompt masking. Saves PEFT weights and generates paired base/adapter diffs.

Reusable ideas: Pin revisions, dependencies, and splits; report training loss, draft metrics, artifact checks, and official resolution separately.

Caveats: The training base differs from official QAT. Gold diffs select context, making this oracle-conditioned patch generation rather than navigation. git apply --numstat only parses. Official loading/resolution is unverified. trainer.train exists, but we did not execute it.

Source locations: L149–L170, L283–L299, L308–L388, L392–L415, L451–L518, L638–L708, L751–L769.

## 64. Pathfinder  | Gemma 4 Agent: EDA + Baseline+

[Kaggle](https://www.kaggle.com/code/mizeroluckygall/pathfinder-gemma-4-agent-eda-baseline) · [Local source](public/mizeroluckygall/pathfinder-gemma-4-agent-eda-baseline/readable.txt)

**Category: Pathfinder task/retrieval analysis and two-agent packaging.** Compares issue-symbol lookup, neighboring embeddings, and lexical search. v2 packages coder/analyzer without eval_config by default and records hashes.

Reusable ideas: Study retrieval around concrete symbols and distinguish local prompt changes, reported scores, and untested versions.

Caveats: Only gold-mappable tasks enter graph analysis, capped at 60 commits/300 seconds. v1 0.08 is author-reported; v2 is pending. include_thoughts=false disables thinking despite display-only comments.

Source locations: L57–L71, L416–L472, L641–L697, L837–L849.

## 65. The Gemma 4 Developer Agent Competition

[Kaggle](https://www.kaggle.com/code/alfira17/the-gemma-4-developer-agent-competition) · [Local source](public/alfira17/the-gemma-4-developer-agent-competition/readable.txt)

**Category: Successive drafts ending in a V12 two-agent bundle.** Early cells build a single agent, then repeatedly overwrite submission.zip. Final V12 copies the official sample, removes adapters/eval_config, and replaces coder/analyzer prompts and 8192-token sampling.

Reusable ideas: Inspect the final ZIP; fixed timestamps and a five-file check identify its actual contents.

Caveats: The earlier four-file bundle is not the final payload. V12 has no custom budget, official compilation, or GPU evaluation. YAML compose/string assertions establish only local structure.

Source locations: L207–L244, L303–L392.

## 66. GraphLoc end-to-end: Gemma 4 agent + skill

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-end-to-end-gemma-4-agent-skill) · [Local source](public/zzgtylors/graphloc-end-to-end-gemma-4-agent-skill/readable.txt)

**Category: Paired official evaluation of a GraphLoc skill.** Source SHA256 matches entry 56, zzgtylors/graphloc-end-to-end-repeat-run-3. Implements offline BM25/rebuilt-graph PageRank, matched sampling, gold filtering, and 12 workers sharing TP4.

Reusable ideas: Retain end-to-end paired comparisons, repeats, and complete traces.

Caveats: Duplicate source is not independent score evidence. Gold-only filtering omits the empty-failure requirement, environment modifications affect tests, and concurrent runtime is not sequential scoring. Three identical entries do not verify three independent runs.

Source locations: L294–L344, L968–L1021, L1088–L1158, L1194–L1209, L1399–L1452.

## 67. 🔬 Gemma 4 Starter | Inside the Harness + E 63dd0e

[Kaggle](https://www.kaggle.com/code/ahmadalalony/gemma-4-starter-inside-the-harness-e-63dd0e) · [Local source](public/ahmadalalony/gemma-4-starter-inside-the-harness-e-63dd0e/readable.txt)

**Category: Prompt iteration on a common starter.** Compared with its baseline, lowers max_output_tokens to 4096 and replaces coder/analyzer prompts to emphasize tools, symbol queries, early edits, and avoiding failed-search loops.

Reusable ideas: Use source diffs to identify changes and correct assumptions about natural-language graph queries or /tmp access.

Caveats: Fixed pytest -p no:anyio may change async-test behavior and requires task-specific checks. Tool availability varies across runtimes. Static packaging supplies no new score evidence.

Source locations: L54–L60, L1306–L1371.

## 68. Gemma 4 Developer Agent: Direct Code Verifier

[Kaggle](https://www.kaggle.com/code/nihilisticneuralnet/gemma-4-developer-agent-direct-code-verifier) · [Local source](public/nihilisticneuralnet/gemma-4-developer-agent-direct-code-verifier/readable.txt)

**Category: Paired edit-tool candidates under official evaluation.** rapid and guard share a QAT service. Two eight-task batches compare candidates; guard retains shell/read but removes edit_file. Selection requires at least two more resolved tasks across 16, with zero test edits/infrastructure errors.

Reusable ideas: Keep per-task stage failures and progress. Only the selected agent enters the ZIP; evaluation records remain external.

Caveats: Tools, prompts, and output/thinking budgets all change, preventing single-variable attribution. Both batches affect selection, so the second is not a final holdout. Empty/gold controls are absent. Source calls are not our independent reproduction.

Source locations: L269–L310, L317–L357, L433–L455, L497–L578, L622–L640, L677–L718.

## 69. Gemma 4 DevAgent - KerasHub LoRA Training on TPU

[Kaggle](https://www.kaggle.com/code/nursrijan/gemma-4-devagent-kerashub-lora-training-on-tpu) · [Local source](public/nursrijan/gemma-4-devagent-kerashub-lora-training-on-tpu/readable.txt)

**Category: KerasHub TPU tutorial with incompatible model/adapter assumptions.** Loads gemma2_instruct_2b_en and rank-16 LoRA, then labels exports as Gemma4 31B. On loading failure, constructs safetensors from random 64-dimensional matrices.

Reusable ideas: General TPU/Keras interfaces are educational; verify architecture and conversion before reuse.

Caveats: Gemma 2 2B is not the required Gemma 4 31B. Changing base_model_name or simple weight-name substitution does not convert architecture, PEFT names, or layouts. Random fallback tensors are not trained competition adapters.

Source locations: L168–L203, L212–L245, L249–L301, L355–L369.

## 70. Black Cat SWE Agent

[Kaggle](https://www.kaggle.com/code/harishyadav0506/black-cat-swe-agent) · [Local source](public/harishyadav0506/black-cat-swe-agent/readable.txt)

**Category: Black Cat main and fallback single-agent bundles.** Builds a nine-tool main bundle and a simpler fallback from strings, checks paths/extensions recursively, and outputs two ZIPs.

Reusable ideas: Clearly name anchors/fallbacks; symbol-query and minimal-repair prompts can be compared.

Caveats: Only custom structural validation. thinking_level/budget coexist with include_thoughts=false, so MEDIUM does not mean thinking is enabled. Source explicitly marks the score pending.

Source locations: L126–L128, L138–L198, L203–L214.

## 71. 🐈‍⬛ Black Cat SWE Agent | Second Strike

[Kaggle](https://www.kaggle.com/code/azouzibillel/black-cat-swe-agent-second-strike) · [Local source](public/azouzibillel/black-cat-swe-agent-second-strike/readable.txt)

**Category: Minimal configuration with an unused model variable.** Packages one agent.yaml. A later MODEL_ID points to an older Gemma 2B but is neither loaded nor used.

Reusable ideas: Distinguish actual YAML model declarations from unused notebook variables.

Caveats: No compilation, evaluation, or training. The final check expects nonexistent submission.json instead of the generated ZIP. A readiness print is not execution evidence.

Source locations: L1–L46.

## 72. 📘Gemma-4 | The Complete Guide. Expanded

[Kaggle](https://www.kaggle.com/code/leoprovorov/gemma-4-the-complete-guide-expanded) · [Local source](public/leoprovorov/gemma-4-the-complete-guide-expanded/readable.txt)

**Category: Unchanged official-sample packaging control.** Copies all sample_submission files, checks per-file hashes, archives them, and reads each archived file back for comparison.

Reusable ideas: A strong packaging-content control with a recorded sample version.

Caveats: No GPU, official compilation, or evaluation. Included adapters depend on the mounted sample revision. Equal file contents do not establish deterministic ZIP bytes or successful model execution.

Source locations: L1–L4, L24–L71.

## 73. notebook67dada2eb5

[Kaggle](https://www.kaggle.com/code/bang1850/notebook67dada2eb5) · [Local source](public/bang1850/notebook67dada2eb5/readable.txt)

**Category: Dependency-installation placeholder.** Contains one pip command installing TRL, PEFT, bitsandbytes, and related packages.

Reusable ideas: Identify a notebook whose implementation has not begun.

Caveats: No pinned versions, training, model, evaluation, or submission construction. This does not implement an RL/SFT tutorial.

Source locations: L1–L2.

## 74. GemmaForge Agent

[Kaggle](https://www.kaggle.com/code/salvapiol/gemmaforge-agent) · [Local source](public/salvapiol/gemmaforge-agent/readable.txt)

**Category: Agent prototype, trajectory templates, and final configuration.** Builds environment/trace classes and converts SWE-bench Lite into template steps; ModelPolicy.generate remains unimplemented. Finally copies sample adapters and packages coder/analyzer/navigation skill.

Reusable ideas: Separate tool environments, template trajectories, model policies, and actual ADK payload; reuse trace fields where useful.

Caveats: Template steps lack real tool observations and are not successful trajectories. The LoRA section only creates LoraConfig. Submitted weights are copied examples; missing files merely print warnings. An intermediate project ZIP is not the final submission.

Source locations: L477–L554, L4799–L4836, L5161–L5173, L5196–L5218, L5861–L5888, L6030–L6104.

## 75. Google - The Gemma 4 Developer -11

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer-11) · [Local source](public/chaitanyajamble/google-the-gemma-4-developer-11/readable.txt)

**Category: Single-agent packaging with explicit limits.** Uses nine tools, 8192 output tokens, eight minutes, 30 tools, and 40 turns; custom validation precedes ZIP creation and output-directory checks.

Reusable ideas: Short configurations, include readback, and explicit packaging errors.

Caveats: Eight minutes across about 120 tasks gives a 16-hour upper bound, not verified 12-hour safety. The author rule that only a ZIP may remain does not justify deleting our logs. No actual inference/score.

Source locations: L124–L154, L212–L233, L348–L359.

## 76. GEMMA 4 developer agent EDA

[Kaggle](https://www.kaggle.com/code/msama01/gemma-4-developer-agent-eda) · [Local source](public/msama01/gemma-4-developer-agent-eda/readable.txt)

**Category: Configuration-adapted EDA starter.** Parses gold diffs and lexical localization, discovers sample root/model settings, then builds one agent or coder/analyzer and packages them.

Reusable ideas: Track final variable values: later sample discovery overrides the initial swe_coder.yaml assignment.

Caveats: EDA and a custom validator, without official GPU evaluation. Synthetic fallback data and character-based token estimates are not measurements. Thinking is disabled.

Source locations: L7–L40, L387–L421, L1366–L1394, L1408–L1489.

## 77. Gemma 4 Developer Agent

[Kaggle](https://www.kaggle.com/code/harishyadav0506/gemma-4-developer-agent) · [Local source](public/harishyadav0506/gemma-4-developer-agent/readable.txt)

**Category: Early walkthrough and minimal single-agent bundle.** Builds one agent with six core tools, temperature 0/top_k 1, and thinking enabled; prompts plan around about ten tool calls and short tasks before structural packaging checks.

Reusable ideas: Readable single-agent baseline with an executable configuration as the final artifact.

Caveats: Descriptions of 256 graph pairs and very short defaults are historical; current data has 127 pairs. pytest exit 0 alone is insufficient without actual required tests and verifier conditions. No measured score.

Source locations: L26–L44, L229–L280, L350–L428, L484–L508.

## 78. 🐈‍⬛ Gemma 4 Agent: EDA & ADK Starter

[Kaggle](https://www.kaggle.com/code/nilanjansaha123/gemma-4-agent-eda-adk-starter) · [Local source](public/nilanjansaha123/gemma-4-agent-eda-adk-starter/readable.txt)

**Category: Simplified EDA and two-agent packaging.** Parses diffs, attempts graph lookup by instance_id, creates coder/analyzer, and writes a ZIP.

Reusable ideas: Short configuration illustrates agent_tool structure and independent prompts.

Caveats: Task-name-only graph lookup misses current commit-based names and may silently omit graphs. Claimed schema checks only inspect extensions/paths. No official compilation; HIGH plus include_thoughts=false does not establish enabled thinking.

Source locations: L89–L111, L117–L171, L177–L190.

## 79. GraphLoc: auditing Gemma 4 agent code graphs

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-auditing-gemma-4-agent-code-graphs) · [Local source](public/zzgtylors/graphloc-auditing-gemma-4-agent-code-graphs/readable.txt)

**Category: GraphLoc CPU graph/embedding/localization research.** Derives function labels from snapshot ASTs and reference diffs, compares G0/G+, BM25, propagation, embeddings, and external SWE-bench transfer, then exports a standard-library skill.

Reusable ideas: Match candidate-inspection budgets, record label provenance, and separate graph coverage from localization effectiveness.

Caveats: No repair agent: labels/ranks are not resolution. The final concurrent batch jobs are unsuitable for a login-node Run All. External transfer needs staged repositories/network preparation.

Source locations: L1–L12, L1130–L1187, L1760–L1778, L1781–L1809, L1998–L2019, L2104–L2133.

## 80. Google - The Gemma 4 Developer/

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer) · [Local source](public/chaitanyajamble/google-the-gemma-4-developer/readable.txt)

**Category: Optional SFT with three submission fallbacks.** RUN_LORA_TRAINING=false by default; enabling it trains issue-to-diff LoRA. Packaging tries sample, then built-in agent, then an emergency single-file bundle.

Reusable ideas: Make the default no-adapter behavior explicit and record final branch labels/hashes.

Caveats: Training failures are swallowed and continue without adapters; the emergency result lacks full checks. Always producing a ZIP does not prove the intended training/configuration succeeded. Random task splits do not establish new-repository generalization; official execution remains necessary.

Source locations: L383–L435, L518–L550, L1055–L1085, L1187–L1202.

## 81. notebook13918923cc

[Kaggle](https://www.kaggle.com/code/manuelgagofernandez/notebook13918923cc) · [Local source](public/manuelgagofernandez/notebook13918923cc/readable.txt)

**Category: Default Kaggle script template.** Imports numpy, pandas, and kagglehub, then prints mounted file paths.

Reusable ideas: Retain coverage for blank development entries.

Caveats: No competition implementation, model, evaluation, or artifact; nearly the same as another template from this author.

Source locations: L1–L24.

## 82. 🧬 Anatomy of a Fix

[Kaggle](https://www.kaggle.com/code/proamari/anatomy-of-a-fix) · [Local source](public/proamari/anatomy-of-a-fix/readable.txt)

**Category: Issue/reference-repair/test exploration and prompt bundle.** Explains the 129 issues, gold patches, and test patches, analyzes cues/changes, then writes coder/analyzer configuration and hashes.

Reusable ideas: Useful onboarding and evidence-based hypotheses; verify imports resolve to the actual worktree.

Caveats: No agent run or evidence of prompt improvement. Globally disabling anyio or never repeating commands can prevent needed verification. Claims that every valid submission shares certain settings are not established.

Source locations: L14–L25, L169–L183, L302–L323, L329–L342, L376–L421.

## 83. Gemma 4 Agent Working Baseline

[Kaggle](https://www.kaggle.com/code/leoprovorov/gemma-4-agent-working-baseline) · [Local source](public/leoprovorov/gemma-4-agent-working-baseline/readable.txt)

**Category: Another prompt variant of a common starter.** Diffs against the inside-the-harness entry show an 8192-token output cap, import-path checks, and changed reproduction/verification/repetition instructions.

Reusable ideas: Identify variables from actual diffs; import-path checks help detect false repairs.

Caveats: No new GPU result. Never repeating commands conflicts with rerunning tests after repair; disabling anyio needs task-specific judgment. This is not an independent run of identical source.

Source locations: L54–L60, L1308–L1371.

## 84. Google - The Gemma 4 Developer Agent Competition

[Kaggle](https://www.kaggle.com/code/manuelgagofernandez/google-the-gemma-4-developer-agent-competition) · [Local source](public/manuelgagofernandez/google-the-gemma-4-developer-agent-competition/readable.txt)

**Category: Default Kaggle Python template.** Imports packages and lists input files, without competition-specific logic.

Reusable ideas: Record complete entry coverage and identify unimplemented content.

Caveats: No agent, training, serving, or ZIP. A competition title does not establish implementation progress.

Source locations: L1–L23.

## 85. GraphLoc: can Gemma 4 pick the code to change?

[Kaggle](https://www.kaggle.com/code/zzgtylors/graphloc-can-gemma-4-pick-the-code-to-change) · [Local source](public/zzgtylors/graphloc-can-gemma-4-pick-the-code-to-change/readable.txt)

**Category: Candidate-code reranking with the official base.** Builds top10, hop1, and length-matched candidate pools from BM25/one-hop graphs. Gemma TP4 selects top three; compares pool coverage, selection, thinking, and paired bootstrap.

Reusable ideas: Control candidate count/order and separate missing candidates from wrong model choices; record prompt/output tokens.

Caveats: Only tasks with mappable gold functions are included. Localization is not repair. A chat-parameter TypeError falls back without an explicit thinking flag; record rendered behavior. We did not independently run GPU evaluation.

Source locations: L1–L12, L80–L127, L148–L191, L213–L240.

## 86. Gemma 4 Paper Track Async Codegraph Resource

[Kaggle](https://www.kaggle.com/code/sayandeepghosh/gemma-4-paper-track-async-codegraph-resource) · [Local source](public/sayandeepghosh/gemma-4-paper-track-async-codegraph-resource/readable.txt)

**Category: AsyncPolyGraph prototype with biased localization labels.** Defines async/call/decorator/dependency builders, but evaluation reads existing graphs and lexical path scores. The candidate-file catalog comes from every task gold patch.

Reusable ideas: Async and dependency-injection representations are research ideas requiring answer-independent candidate construction.

Caveats: The builder is unused in the reported comparison. Gold files in the candidate universe leak oracle information. instance_id lookup misses current commit filenames; reported gains do not establish async-graph benefit.

Source locations: L41–L58, L102–L126, L151–L189, L197–L258, L286–L311.

## 87. Gemma4 RQ4 Query Encoding

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-rq4-query-encoding) · [Local source](public/woldywei/gemma4-rq4-query-encoding/readable.txt)

**Category: Query-text encoding versus published embedding channels.** Falls back from Jina-code to BGE-small, encodes issues/node text, and compares TF-IDF, indexed embeddings, and RRF with file/function recall and MRR.

Reusable ideas: Identify the missing query-text encoder in current search_similar_code and measure query representation separately from candidate coverage.

Caveats: emb_rank averages token-overlap seeds instead of reproducing official exact/suffix lookup. New-diff line numbers may misalign with old ASTs. Encoder fallback changes the experiment. No repair agent runs, and third-party encoders are not automatically approved for submissions.

Source locations: L1–L13, L183–L211, L267–L290, L399–L454.

## 88. Gemma | Super Basic

[Kaggle](https://www.kaggle.com/code/twangygarlic449/gemma-super-basic) · [Local source](public/twangygarlic449/gemma-super-basic/readable.txt)

**Category: Minimal five-tool single-agent scaffold.** Writes root, sampling, and prompt files with five tools, 8192 output tokens, and thinking disabled; packages after required-file checks.

Reusable ideas: Isolate basic prompt and package-structure changes.

Caveats: No official compilation or inference. No eval_config means current defaults apply. File/root checks do not prove successful tool execution.

Source locations: L17–L31, L66–L85, L102–L157.

## 89. Getting Started - Gemma 4 Developer Agent

[Kaggle](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent) · [Local source](public/ryanholbrook/getting-started-gemma-4-developer-agent/readable.txt)

**Category: Official end-to-end starter.** Installs patched wheels offline, copies the sample with thinking enabled, and starts vLLM on v2 with 32768 context. Official Evaluator generates/verifies the first two tasks sequentially, then packages the agent.

Reusable ideas: Authoritative reference for model registration, adapter discovery, two-stage evaluation, and ZIP-root structure.

Caveats: Two examples are not full-set performance, and copied adapters are not newly trained. Forced reinstalls and cutlass .pth deletion belong in isolated environments. Add empty/gold controls and actual resource checks.

Source locations: L26–L98, L167–L203, L239–L320.

## 90. Gemma4 Localization Repaired

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-localization-repaired) · [Local source](public/woldywei/gemma4-localization-repaired/readable.txt)

**Category: Localization after graph reconstruction.** Rebuilds calls graphs including async definitions from snapshot ASTs; compares TF-IDF, means of existing symbol embeddings, PPR, and RRF for file/function recall.

Reusable ideas: Separate coverage from ranking and check async definitions, name resolution, and graph/embedding alignment.

Caveats: Embeddings are not rebuilt alongside graphs. New-file hunk lines may mislabel old ASTs. Seed averaging is not exact official retrieval; localization is not resolution. Bulk tar extraction is unsuitable for login nodes.

Source locations: L149–L170, L195–L264, L322–L362, L477–L575.

## 91. Gemma4 Repaired Graphs Regen

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-repaired-graphs-regen) · [Local source](public/woldywei/gemma4-repaired-graphs-regen/readable.txt)

**Category: Async graph-generation utility.** Extracts each snapshot, collects synchronous/asynchronous functions and classes, links calls with name/module heuristics, and writes instance_id.json plus regen_summary.

Reusable ideas: Source-based async coverage prototype and graph-statistics reference.

Caveats: No training, inference, repair, or embedding updates. Heuristic edges are not an exact dynamic call graph. Map task-name outputs to official commit names and isolate/path-check extraction.

Source locations: L69–L90, L138–L184, L193–L222.

## 92. Gemma4 Graph Localization Bench

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-graph-localization-bench) · [Local source](public/woldywei/gemma4-graph-localization-bench/readable.txt)

**Category: Original-graph localization benchmark.** Compares TF-IDF file/function search, symbol-seed embeddings, PPR, and RRF using recall/MRR, including the fraction of gold targets absent from graph nodes.

Reusable ideas: Measure coverage ceilings before ranking over matched candidate spaces; stratify by repository and async status.

Caveats: Function tables include only reachable gold targets. New-hunk/old-source alignment may bias labels. Multi-symbol embedding seeds neither encode natural language nor exactly reproduce the official lookup. No repair inference.

Source locations: L92–L128, L229–L275, L389–L445, L480–L504.

## 93. Gemma4 Graph Coverage Audit

[Kaggle](https://www.kaggle.com/code/woldywei/gemma4-graph-coverage-audit) · [Local source](public/woldywei/gemma4-graph-coverage-audit/readable.txt)

**Category: Static graph-coverage audit.** Finds graphs through task, commit, and shared-commit names, scans definitions in gold-touched files, and compares graph coverage and async categories.

Reusable ideas: Data mapping, empty-file checks, and AST-coverage diagnostics.

Caveats: Counts all definitions in touched files, not precisely edited functions. Same-name matches may overstate coverage. Historical hardlink issues do not establish current corruption; recheck snapshot prefixes.

Source locations: L111–L138, L143–L182, L185–L227.

## 94. Gemma and the Shape of Doubt

[Kaggle](https://www.kaggle.com/code/prvsiyan/gemma-and-the-shape-of-doubt) · [Local source](public/prvsiyan/gemma-and-the-shape-of-doubt/readable.txt)

**Category: Evidence-organized agent development and evaluation.** Preserves a submitted V10 ZIP, invokes official compilation, and builds a bounded-search candidate. Includes fixed-patch CPU replay and a 16-task vLLM/official-verifier screen. RUN_COMMAND_SCREEN=True enables GPU execution.

Reusable ideas: Preserve bundle hashes and separate candidates; check endpoint health, strict JUnit results, and environment versus agent contributions.

Caveats: 0.08, 2/4, and 3/12 are retained author records, not our reproduction or current top scores. Development repeatedly uses several sets. Extra V14 passes coincide with environment fixes. Default is not CPU-only; additional wheels/test inventories are required.

Source locations: L9–L33, L845–L904, L955–L985, L2700–L2745, L2849–L2875, L3773–L3791.

## 95. mainmainmain

[Kaggle](https://www.kaggle.com/code/tuyennguyen21pt/mainmainmain) · [Local source](public/tuyennguyen21pt/mainmainmain/readable.txt)

**Category: Unrelated sentiment-analysis training.** Uses Vietnamese APC data, mBERT/mT5, and the external KLTN-Token-Merging project for ATE/APC training and span-error analysis.

Reusable ideas: General experiment organization, error analysis, and multiple-seed patterns only.

Caveats: Association with the competition Code page does not approve this task/model. It is not a Gemma agent, ADK bundle, or competition evaluation.

Source locations: L56–L84, L449–L480, L525–L550.

## 96. The Gemma 4 Developer Agent- Google

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/the-gemma-4-developer-agent-google) · [Local source](public/chaitanyajamble/the-gemma-4-developer-agent-google/readable.txt)

**Category: Optional SFT with three packaging fallbacks.** Defaults RUN_LORA_TRAINING=False. Falls back from sample rewriting to a built-in agent and then an emergency file. Enabled training can generate adapter files.

Reusable ideas: Record the actual package_label, archive readback, and warnings about unreferenced adapters.

Caveats: Custom YAML/file checks are not official compilation. Trained adapters are only added as files and explicitly not connected to agent configuration. ZIP success may mean emergency fallback rather than the intended agent.

Source locations: L377–L387, L504–L531, L793–L814, L842–L866.

## 97. notebook057a67b5a8

[Kaggle](https://www.kaggle.com/code/tuynnguynee/notebook057a67b5a8) · [Local source](public/tuynnguynee/notebook057a67b5a8/readable.txt)

**Category: Variant of sentiment training/error analysis.** Uses the same APC and mBERT/mT5 setup as mainmainmain, with ATE artifacts and span-error statistics.

Reusable ideas: Shows that unrelated notebooks can appear in the competition listing; exclude it from strategy comparisons.

Caveats: Unrelated model names are not permission to replace the required Gemma base. This is a variant rather than a byte-identical duplicate of that training notebook.

Source locations: L56–L84, L449–L480.

## 98. Gemma 4 DevAgent - Graph Reasoning and SWE Engine

[Kaggle](https://www.kaggle.com/code/nursrijan/gemma-4-devagent-graph-reasoning-and-swe-engine) · [Local source](public/nursrijan/gemma-4-devagent-graph-reasoning-and-swe-engine/readable.txt)

**Category: EDA, synthetic graphs, and navigator packaging.** Analyzes patches and editing examples. Failed task-name graph lookup creates a synthetic FastAPI graph; writes navigator/coder YAML with file-level checks.

Reusable ideas: Before/after checks, symbol queries, local tests, and role separation can inspire prompt comparisons.

Caveats: Current graphs use commit names, so lookup may silently use a mock. The function labeled Formal adk-submission checks only root/model count/extensions/size. No real evaluator; include_thoughts=False disables thinking.

Source locations: L250–L286, L438–L510, L544–L584.

## 99. 119 of 129 sound: the Gemma 4 grader, rebuilt

[Kaggle](https://www.kaggle.com/code/busyaprime/119-of-129-sound-the-gemma-4-grader-rebuilt) · [Local source](public/busyaprime/119-of-129-sound-the-gemma-4-grader-rebuilt/readable.txt)

**Category: Data/verifier audit and lexical-localization research.** Inspects live data/task fields, creates repository/time dev-heldout splits, and scans snapshots for BM25 localization. Sound-task counts come from historical tables/attached CSVs and are summarized locally.

Reusable ideas: Distinguish live versus cited evidence, require empty failure/gold success and required-node/JUnit checks, and fix splits before localization comparisons.

Caveats: The title 119 is not a new 129-task verification. The raw required-node surrogate is 117; 119 includes the author interpretation of skips. Historical no-harness checks only inspected competition data, not the later separate wheelhouse, and do not prove current offline evaluation impossible.

Source locations: L1–L21, L677–L710, L767–L807, L835–L877, L1018–L1032, L1765–L1799.

## 100. Google - The Gemma 4 Developer Agent 2

[Kaggle](https://www.kaggle.com/code/chaitanyajamble/google-the-gemma-4-developer-agent-2) · [Local source](public/chaitanyajamble/google-the-gemma-4-developer-agent-2/readable.txt)

**Category: EDA, repository priors, and optional LoRA packaging.** Defaults to injecting repository-level gold-file frequencies into prompts. Optional issue-to-diff SFT masks prompts, drops long samples, and accumulates gradients; adapters require explicit arguments and training failure falls back to none.

Reusable ideas: Modular CLI, real label masking, and final-bundle readback.

Caveats: Gold-file priors leak into evaluation on the same public set; hidden repositories differ. SFT lacks source context/independent split, and leftover accumulated gradients are not explicitly flushed. Quantized-base and official-serving compatibility remain unverified.

Source locations: L358–L388, L797–L857, L865–L883, L935–L967.

## 101. Gemma4

[Kaggle](https://www.kaggle.com/code/rishavae/gemma4) · [Local source](public/rishavae/gemma4/readable.txt)

**Category: Three-agent YAML draft.** Writes coder, analyzer, and repro_verifier YAML/prompts/sampling and compresses submission.zip.

Reusable ideas: Illustrates separation of localization, modification, and verification roles.

Caveats: Although the title claims GENSL/EC-PPR and official verification, all 78 lines lack data loading, solvers, compiler, or evaluator calls. Nested AgentTool and top-level thinking_budget are unvalidated; natural-language search_similar_code assumptions need correction.

Source locations: L1–L78.

## 102. Dual-Space Spectral Graph SWE Agent for Gemma 4

[Kaggle](https://www.kaggle.com/code/avikdas567/dual-space-spectral-graph-swe-agent-for-gemma-4) · [Local source](public/avikdas567/dual-space-spectral-graph-swe-agent-for-gemma-4/readable.txt)

**Category: Dual-space retrieval demonstration and placeholder adapters.** Compares cosine/graph-weighted retrieval using target embeddings plus noise; hybrid uses the target symbol as a seed. Packaging generates random LoRA A and zero B.

Reusable ideas: Teaching structure for graph/vector combinations and artifact inspection only.

Caveats: Retrieval knows the target in advance and may use different noisy queries across modes. Adapters are untrained with hand-written shapes/layer names. Safetensors validity proves neither compatibility nor benefit. No official repair evaluation.

Source locations: L1153–L1223, L1736–L1767, L1772–L1800.

## 103. Gemma 4 | Walkthrough & First Submission

[Kaggle](https://www.kaggle.com/code/zhukovoleksiy/gemma-4-walkthrough-first-submission) · [Local source](public/zhukovoleksiy/gemma-4-walkthrough-first-submission/readable.txt)

**Category: Version walkthrough and baseline construction.** Source SHA256 matches entry 8, liujiahua2358/gemma-4-v0-walkthrough-first-submission. Inspects samples, LoRA, graphs, and embeddings, builds analyzer/coder revisions, and packages them.

Reusable ideas: Track version provenance, graph-output size, test tails, and wheel hashes.

Caveats: Duplicate source is not independent execution/score evidence. Old and revised claims coexist; current wheels determine thinking/seed forwarding. Quoted leaderboard scores are not ablations.

Source locations: L1–L15, L275–L318, L380–L410, L630–L653.

## 104. Gemma4_DevAgent_GraphTools_Submission_v2

[Kaggle](https://www.kaggle.com/code/yuki16/gemma4-devagent-graphtools-submission-v2) · [Local source](public/yuki16/gemma4-devagent-graphtools-submission-v2/readable.txt)

**Category: Local graph tools, short navigation, and packaging checks.** Implements graph tools with budgets/error schemas and optional hashed embeddings. Attempts up to two vLLM/Transformers navigation rounds, skipping failures; checks a ZIP and removes references to missing adapters.

Reusable ideas: Explicit skipped/failed states, temporary-archive validation before release, and omission of unavailable weights.

Caveats: Two navigation rounds do not generate patches or run official verification. Tool outputs enter as user text with heuristic parsing rather than ADK execution. Hashed embeddings differ from official vectors, and normal notebook completion permits skipped models.

Source locations: L300–L323, L1014–L1047, L1072–L1118, L1723–L1737, L1740–L1790.

## 105. Gemma 4 Agent 🌟EDA + Agent + Validator

[Kaggle](https://www.kaggle.com/code/mizeroluckygall/gemma-4-agent-eda-agent-validator) · [Local source](public/mizeroluckygall/gemma-4-agent-eda-agent-validator/readable.txt)

**Category: Pathfinder EDA/localization with analyzer and coder.** Defaults to analyzer/coder with graph analysis capped at 60 commits/300 seconds. Prompts suggest a four-minute coder stop and eight analyzer calls; packaging uses a custom validator.

Reusable ideas: Start from exact symbols/grep, repeat the reproducer after edits, and retain file/line locations.

Caveats: soft_limit_seconds only enters prompts; ship_eval_config=False means no four-minute hard cap. Eight analyzer calls are also a prompt limit. include_thoughts=False disables thinking despite the comment. No end-to-end GPU repair evidence.

Source locations: L56–L71, L575–L627, L717–L770.

## 106. GEMMA: EDA, Baseline for a start | LB TOP 1

[Kaggle](https://www.kaggle.com/code/romanrozen/gemma-eda-baseline-for-a-start-lb-top-1) · [Local source](public/romanrozen/gemma-eda-baseline-for-a-start-lb-top-1/readable.txt)

**Category: EDA, lexical localization, and optional two-agent packaging.** Source SHA256 matches entry 45, ahmadagbaria/gemma-4-baseline-reproduction, including TF-IDF, single-agent or analyzer/coder options, custom validation, deterministic ZIP, and experiment_card.

Reusable ideas: Archive source/configuration/hashes and distinguish localization metrics from submitted behavior.

Caveats: Duplicate source is not an independent reproduction, including the matching romanrozen entry. No official GPU evaluation. Natural-language search_similar_code conflicts with indexed-symbol lookup. Token counts use character estimates.

Source locations: L1218–L1304, L1449–L1506, L1573–L1635, L1664–L1674.

## 107. [0.10]: Gemma 4 Developer Agent Submission

[Kaggle](https://www.kaggle.com/code/nihilisticneuralnet/0-10-gemma-4-developer-agent-submission) · [Local source](public/nihilisticneuralnet/0-10-gemma-4-developer-agent-submission/readable.txt)

**Category: Lean/reviewer/full prompt-ablation scaffold.** Defaults to lean without two subagents, with 4.5 minutes, 40 tools, and 100 turns; creates three ZIPs. External candidate search, ablation, and local evaluation default off.

Reusable ideas: Compare lean and optional-reviewer agents on matched tasks and inspect repeated-call traces.

Caveats: The title 0.10 is author-reported; default execution does not evaluate. pytest -p no:anyio -o timeout=0 changes tests and needs repository-specific checks. include_thoughts=False disables thinking.

Source locations: L294–L300, L353–L390, L454–L500, L595–L614, L672–L687.

## 108. Google - The Gemma 4

[Kaggle](https://www.kaggle.com/code/evgendvorkin/google-the-gemma-4) · [Local source](public/evgendvorkin/google-the-gemma-4/readable.txt)

**Category: EDA and agent starter with Russian source explanations.** Defaults to analyzer/coder and 8192 output tokens. Analyzes tasks, graphs, embeddings, and lexical localization; builds from the sample schema, reads back the ZIP, and writes experiment_card.

Reusable ideas: Configuration/hash/archive records and explanatory visualizations.

Caveats: time_budget_s=240 bounds analysis loops, not the agent. Validation is custom with no official inference score. Thinking is disabled; file/character estimates do not measure actual context occupancy.

Source locations: L285–L308, L1721–L1759, L1884–L1900, L1976–L1998.

## 109. 🐈‍⬛ Black Cat SWE Agent | Pack Instinct

[Kaggle](https://www.kaggle.com/code/lucifer19/black-cat-swe-agent-pack-instinct) · [Local source](public/lucifer19/black-cat-swe-agent-pack-instinct/readable.txt)

**Category: Lexical-search challenger preserving a scored anchor.** Defaults to v13 challenger. Decodes and hashes three historical ZIPs. A skill ranks identifiers, error text, behavior, and full-issue views with BM25/RRF, bounded to 6000 files, 12000 chunks, and 32 MiB; failures fall back to the anchor.

Reusable ideas: Preserve scored/recovery ZIPs and separately name candidates; bounded source retrieval is a useful ablation hypothesis.

Caveats: 0.12 belongs to historical v10, not default v13. Embedded pilot records show a 600-second server-start timeout and zero completed pairs. This notebook reruns neither official compilation nor GPU evaluation. Analyzer natural-language graph-query guidance still mismatches the tool.

Source locations: L1–L25, L128–L160, L176–L219, L225–L239.

## 110. Gemma4 Agent Advanced EDA

[Kaggle](https://www.kaggle.com/code/foysalemonshanto/gemma4-agent-advanced-eda) · [Local source](public/foysalemonshanto/gemma4-agent-advanced-eda/readable.txt)

**Category: Detailed data and context-cost exploration.** Checks tasks/reference/test patches, both graph/embedding naming schemes, and vector layouts. Scans snapshots to estimate full-file versus targeted reading using gold-touched files/lines.

Reusable ideas: Current commit-name mapping, allow_pickle=False, repository splits, and tool-output-length analysis.

Caveats: Gold-based reading counts are oracle-conditioned cost references, not agent performance. No training or repair evaluation. ProcessPoolExecutor(os.cpu_count()) can trigger heavy parallel scans and is unsuitable for login-node Run All.

Source locations: L414–L440, L556–L581, L659–L704, L711–L741.

## 111. Gemma 4 SWE Agent: Complete EDA & ADK Starter Kit

[Kaggle](https://www.kaggle.com/code/nursrijan/gemma-4-swe-agent-complete-eda-adk-starter-kit) · [Local source](public/nursrijan/gemma-4-swe-agent-complete-eda-adk-starter-kit/readable.txt)

**Category: EDA and two-agent packaging tutorial.** Reads tasks with a two-row synthetic fallback, simulates edit_file, writes analyzer/coder YAML, checks sizes/model/extensions, and outputs a ZIP.

Reusable ideas: Basic layout and concise local-edit/verification prompts only.

Caveats: The final placeholder submission.parquet directly copies row.patch, exporting gold answers rather than predictions. No real evaluator; mock data/graphs are not measured statistics. Thinking is disabled.

Source locations: L96–L135, L382–L413, L446–L478, L502–L539, L556–L598.

## 112. Gemma develop first try

[Kaggle](https://www.kaggle.com/code/haideptry/gemma-develop-first-try) · [Local source](public/haideptry/gemma-develop-first-try/readable.txt)

**Category: Sample-adapter two-agent packaging with explicit budgets.** Packages sample or embedded base64 adapters for coder/analyzer with five minutes, 15 tools, 25 turns, and 60-second commands. Uses fixed ZIP timestamps and readback. Embedded headers/data were decoded statically.

Reusable ideas: Reproducible archives, current language_model paths, and actual nonsquare q/o tensor shapes.

Caveats: Both embedded weights are identical and LoRA B is all zero. No training, benefit, or universal compatibility is established. Five minutes times 120 gives ten hours before setup, not a complete deadline guarantee. Omitting skills is an author choice, not an official prohibition.

Source locations: L5–L11, L35–L65, L98–L124, L163–L189.
