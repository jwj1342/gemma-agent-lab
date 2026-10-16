# Selective graph retrieval: preregistration

Status on October 16, 2026: candidate and comparison registered on CPU; inference
not started. This experiment addresses issue #12 and starts from
`experiment/prompt-icl-route` at `3a7ef37`. The protected holdout remains sealed.

## Question and comparison

- Hypothesis: after repository search identifies a relevant symbol, one bounded
  graph-neighbor inspection can improve source localization without increasing
  malformed tool use, timeouts, or repair regressions.
- Baseline: unchanged `configs/baseline` from `3a7ef37`.
- Candidate: `configs/prompt-selective-graph`, with only the main system prompt
  changed. Model, tools, analyzer, sampling, seed, and time/tool/turn budgets match
  the baseline. LoRA is disabled in both packages.
- Sequence: the fixed `smoke` task, followed only after a clean integration run by
  the unchanged `fast` cohort in `configs/evaluation/public-v1.json`.
- Success evidence: actual graph use plus paired repair outcomes. More graph calls
  alone are not success.

## Tool contract

The official wrappers are implemented in `swegemma/tools/graph.py`.

| Tool | Valid input | Interpretation and fallback |
| --- | --- | --- |
| `get_code_neighbors` | Exact function, method, or class node; optional edge type; integer result limit | The candidate uses one exact symbol, no edge filter, and eight results. Node names are navigation hints and must be confirmed in source. Empty/error results fall back immediately to repository search. |
| `search_similar_code` | Exact known symbol or code snippet plus integer `k` | It is not a general issue-text search. Use it only for a specific unresolved dependency after source and neighbor inspection. Empty/error results fall back to source search. |
| `get_code_subgraph` | A list of exact node names already obtained from source or graph output | This returns an induced subgraph and does not discover nodes. Use it only when the relationship among at least two known nodes is necessary. |

The audited public graphs contain `calls` edges. Documentation examples mentioning
other relationship types do not establish that those edges exist in this dataset.
Graphs may omit source symbols, including async functions, so missing graph output
does not establish missing code.

## Registered execution

`configs/evaluation/selective-graph-v1.json` freezes the source revision, benchmark,
agent paths, cohorts, runtime settings, and reporting fields. Before inference:

1. Build both agent archives and confirm their hashes match the registration.
2. Check current Kaggle quota and existing reservations.
3. Coordinate mounted inputs with `@jwj1342`.
4. Prepare distinct baseline and candidate output paths; do not overwrite or select
   among retries.
5. Run valid empty/reference controls and official compilation before inference.

The smoke must report actual hardware, graph calls and arguments, empty or failing
queries, repository fallback, inspected source locations, token/time cost, controls,
clean completion, and independent patch verification. Run the unchanged fast cohort
only if smoke compilation, controls, trace capture, and graph reporting are complete.

## Result template

- Agent ZIP hashes: registered; recheck before execution.
- Platform and actual allocation: not measured.
- Kaggle quota: checked at `2026-10-16T15:11:22+08:00` with Kaggle CLI
  2.2.4. The CLI reported GPU `0.00h` used, `30.00h` remaining, and `30.00h`
  total, refreshing at `2026-10-17T00:00:00`. This was a read-only check; no
  notebook version or GPU session was started. The displayed table did not expose
  a separate reservation field.
- Smoke controls, compilation, graph invocation, and repair: not measured.
- Fast eligible coverage and paired outcomes: not measured.
- Extra model tokens and agent time versus baseline: not measured.
- Empty/failing graph queries and useful inspected locations: not measured.
- Conclusion: pending measured smoke and fast evidence.

Representative traces may be reported only with task, run, agent, source, and trace
hash provenance. Task solutions and held-out traces must not become reusable prompt
examples. Competition submission remains a separate coordinated decision.
