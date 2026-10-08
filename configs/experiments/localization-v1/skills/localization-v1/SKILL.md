---
name: localization-v1
description: Rank the repository files and Python symbols most likely to require changes for an issue. Use once near the start when the target file is not already explicit.
---

# Localization v1

This is an offline, standard-library-only source localizer. It enumerates the Python source tree mounted
in the sandbox and uses a versioned adaptive gate to choose full-tree BM25 or the v1 hybrid ranking for
each issue. It does not modify the repository. The current gate is safety-disabled because a rule fit
on 91 tasks still regressed the frozen full-tree holdout, so adaptive mode falls back to BM25 without
loading the checkpoint. A future gate may enable v1 only after the same promotion checks pass.

An additional portable v1 refinement learns seven content/field weights and a confidence gate.
Its frozen policy is in `resources/portable-policy.json`. Explicit `mode: portable` evaluates this
candidate without the archived model; its production switch remains disabled pending remote repair
A/B. The current default continues to use the BM25 fallback. An enabled portable policy takes
precedence in adaptive mode.

## Run

Call `run_skill_script` with:

- `skill_name`: `localization-v1`
- `file_path`: `scripts/localize.py`
- `args`: `{"issue": "<problem statement with identifiers, paths, and errors preserved>", "k": 10}`

The script discovers `/workspace` automatically. Pass `root` only outside the evaluation sandbox.

## Interpret the result

The header reports `adaptive(v1=on)` or `adaptive(v1=off)`. Each row is
`rank score path:line symbol [bm25 exact legacy]`.

1. Read the top three source files, beginning near the suggested symbols. Skip any result that conflicts
   with an explicit path in the issue.
2. Confirm every candidate against the source before forming a diagnosis. Rankings are evidence, not proof.
3. If the first three are irrelevant, inspect the remainder and then use exact search and code-graph tools.
4. Run the localizer once per issue unless new evidence materially changes the suspected component.

Adaptive mode reads `resources/gate.json`; a disabled gate does not load model data. Explicit `bm25`,
`legacy`, `hybrid`, and `portable` modes remain available for diagnostics. Portable mode reports
whether its confidence gate accepted the learned ranking and marks unpromoted execution experimental.
In modes that use the archived model,
repository-specific path priors are deliberately disabled because they do not generalize beyond its
four public training repositories; missing or invalid model data is then an error.
