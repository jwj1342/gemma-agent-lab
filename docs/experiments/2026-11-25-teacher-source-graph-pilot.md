# Teacher source-graph mechanism pilot (2026-11-25, issue #39)

## Question and scope

Following #34's infrastructure closeout, begin #39 with a concrete source-only teacher exercise. Can a
small structural query provide useful evidence beyond literal search, and can the teacher produce a
verified source change without reading the reference patch?

This is one current-Codex-session mechanism trajectory, not an independent graph-on/graph-off model
comparison. The user is unsure whether a separate teacher API is configured; none is required or charged
by this prototype. Exact teacher provider/model revision is not exposed in the recorded interface.
The graph is a new partial AST tool, not the official ADK graph or semantic-search implementation. This
exercise does not settle the official graph's utility or show a Gemma improvement.

## Frozen development set and boundaries

`resource/manifests/teacher-dev-20261125.json` records 12 tasks, three each from more-itertools, tenacity,
w3lib and mistune. Selection is deterministic, seed `teacher-repository-disjoint-2026-11-25`: prefer terse
statements, exclude design statements and leak-flagged rows, then rank by SHA-256 within each repository.
All four repositories are disjoint from the seven mined evaluation repositories and from the historical
29-task competition cohort. Upstream licenses are retained per task: MIT, Apache-2.0 or BSD-3-Clause.
The source dataset's labels/code are Apache-2.0; original task/repository material retains upstream terms.

Only the first preselected task is exercised here: `more_itertools_757` at base
`468755cbb6e40cce1d015d57129ae6137ae1f19e`. Its short public statement requests a stop argument for
iter_index to match the itertools documentation. The administrative CPU worker receives reference/tests
for controls, but the teacher opens only the public task packet and unpatched source after controls pass.
No reference patch or injected test patch was read by the teacher before freezing its own patch.

Public packet, source snapshot, graph implementation and experiment protocol hashes were frozen in
`runs/2026-11-25-issue39-teacher/mechanism-protocol.json` in the original workspace. Reference/test patches,
repository snapshots, generated patches and raw test outputs remain outside Git.

## Implemented tool

`src/source_graph.py` locates top-level Python functions and direct class methods, returning explicit
path/symbol IDs, source spans, same-file syntactic callers/callees, unresolved calls and source hashes.
It deliberately leaves imports, dynamic receivers and nested function bodies unresolved. Parameter/local
bindings are excluded from same-name global links. These links are source-navigation candidates, not a
sound runtime call graph; missing edges are not proof of missing calls.

```bash
python src/source_graph.py --repo /path/to/base-source --symbol iter_index --trace /path/to/queries.jsonl
python src/source_graph.py --repo /path/to/base-source \
  --node more_itertools/recipes.py::iter_index --trace /path/to/queries.jsonl
```

The CLI is callable by an assistant through its shell. It is not registered or packaged as a Gemma
competition skill, and no submission compatibility or skill invocation result is claimed.

## Executed observations

The literal search, before the graph-neighbor query, already found both the named target function and a
call from sieve. The graph indexed 823 functions/methods and 64 syntactic edges, found that same caller,
and left the important calls to islice, operator.indexOf and the instance's index method unresolved.

Two graph queries cost 0.6035 seconds of measured local process time and returned 1,025 UTF-8 bytes.
These are process/retrieval measurements, not model-token usage or an end-to-end teacher speedup. No
additional diagnostic fact was attributable to these graph responses in the observed trajectory.
The text-search comparison is not a second independent solve, so no paired effect is estimated.

The useful repair evidence came from the public target, reading its two execution branches, and the
[Python 3.12 itertools recipe contract](https://docs.python.org/3.12/library/itertools.html#itertools-recipes).
The teacher added the exclusive stop bound in both the general-iterator and sequence paths while
preserving default behavior and the existing sieve caller. Its patch was frozen before grading feedback.
Five existing IterIndex/Sieve tests and 108 independent start/stop/container combinations pass locally.

## Independent verification and recovery

The first CPU job, `junzhejiao/teacher-controls-20261125` version 1, passes official v31 empty/gold controls
for the selected task before the teacher opens it. The second job,
`junzhejiao/teacher-patch-verification-20261125`, repeats those controls and grades the frozen teacher patch
with the same unmodified official verifier in a fresh isolated sandbox. The generator's optional
`--prediction` accepts only a nonempty patch for a selected task and checks valid controls before grading.
Teacher patch hashes are added to the runtime receipt. No model runs inside either CPU job.

Local recovery paths, under the original workspace's ignored `runs/2026-11-25-issue39-teacher/`:

- `tasks.public.jsonl`: teacher-visible input; `tasks.verification.jsonl`: administrative controls only.
- `source-base.tgz`, `source-base/`, `source-candidate/`: frozen base and isolated modified copy.
- `graph-queries.jsonl`, `mechanism-protocol.json`, `mechanism-result.json`: requests and interpretation.
- `teacher.patch`, `teacher-prediction.json`, `patch-provenance.json`: generated patch and its hashes.
- `controls/` and `teacher-verification/`: self-contained CPU notebooks, metadata and captured log streams.

Tracked `teacher-mechanism-20261125.json` contains the compact metadata/result, not the task solution.
Final official grading and publication evidence are recorded below after collection.

Final result: `teacher-patch-verification-20261125` version 1 passes both controls and resolves the teacher
patch with exit code zero and no verifier error. The result is recorded in
`resource/manifests/teacher-verifier-20261125.json`, with notebook/log/patch and per-output hashes.
The verifier reports zero model calls for every side. This is a successful independently checked teacher
repair on one task; it is not evidence that the graph helped or that Gemma would follow the same procedure.
All 97 project CPU tests pass, including source-graph ambiguity/shadowing and prediction-task isolation.
Infrastructure PR #42 at `5d234fd` also has a successful hosted Repository checks run.

## Cases 2 and 3 (2026-11-25, same session)

**Case 2, `more_itertools_1116` (deferred, not graded).** The public statement is "Add random ordered
combinatoric functions". Official controls are valid (empty fails, gold passes; kernel
`junzhejiao/teacher-controls-20261125-case2`), but the statement does not name the new API, and the four
plausible names (`random_product`, `random_permutation`, `random_combination`,
`random_combination_with_replacement`) already exist in `more_itertools/recipes.py` as single-draw recipes that
shadow anything new in `more.py`. A patch written from the statement alone would have to guess both the exported
names and the iteration semantics, so it was not frozen. This is a concrete limit of a terse statement on a
new-API task: the teacher can read the workspace but cannot see the required interface.

**Case 3, `more_itertools_1157` (executed, resolved).** The statement asks to "add generator support to
serialize()", make the attributes private and support `throw`, `send` and `close`. On the unpatched source the
graph located `serialize.__init__/__iter__/__next__` and the existing `TestSerialize.test_concurrent_calls`
reference, which is the first case where the structural query surfaced something the teacher had not already
looked at. The frozen patch moved state to `_iterator`/`_lock` and forwarded `send`, `throw` and `close` under
the same lock; local smoke checks exercised next/send/close, and the official verifier
(`junzhejiao/teacher-verification-20261125-case3`) confirmed valid controls and `teacher_resolved: true` with
zero model calls.

**Protocol deviation, recorded.** For case 3 the controls were run after the teacher opened the source rather
than before; no reference patch or injected test was read at any point.

**Receipts.** `resource/manifests/teacher-mechanism-cases-20261125.json` (all three cases) and
`resource/manifests/teacher-verifier-case3-20261125.json` (hashes, kernel ids, result).

## Case 4 (2026-11-26, same session)

**`tenacity_317` — executed, resolved.** The statement is "Fix after_log logger format" and the defect is
inside one function: `sec = sec_format % _utils.get_callback_name(retry_state.fn)` applied a float format to
the callback name, then the message printed the unformatted `seconds_since_start`. The frozen patch applies
`sec_format` to the elapsed seconds and puts the callback name back into the message.

- Controls: `junzhejiao/teacher-controls-20261126-case4` — valid (empty fails, gold passes).
- Grading: `junzhejiao/teacher-verification-20261126-case4` — valid controls and `teacher_resolved: true`,
  zero model calls in the job.
- Graph: the query returned the single `after_log` definition and no callers; for a defect inside one function
  body the structural tool adds nothing, which is consistent with case 1.
- Recorded deviation: the teacher opened the source before the controls job finished; no reference or test
  patch was read.

## Case 5 (2026-11-26, same session)

**`tenacity_645` — environment-blocked, patch frozen but not graded.** The statement asks that
`Retrying(enabled=False)(fn)` and `AsyncRetrying(enabled=False)(fn)` call the function once instead of running
the retry loop. The frozen patch adds the short-circuit to both `__call__` implementations and local smoke
confirms `ValueError` propagates after a single call.

The official sandbox could not grade it, for two packaging reasons rather than a teacher error: the repository's
`pyproject.toml` uses `hatch-vcs` with `source = "vcs"`, and the sandbox checkout has no `.git`, so editable
metadata generation fails; on retry the sandbox venv could not build (`ensurepip` unavailable), leaving no pip
for the setup hook (`ContainerSetupError`). Kernels: `junzhejiao/teacher-verification-20261126-case5` v1-v3;
logs in the ignored `runs/2026-11-26-issue39-tenacity_645/`.

This is a reproducibility limit of the CPU verifier for VCS-versioned repositories, worth fixing in the harness
(setuptools_scm/hatch-vcs pretend version plus a pip-less venv fallback) before using tenacity-family tasks.

## Case 6 (2026-11-26, same session)

**`mistune_486` — executed, resolved.** "Preserve literal hashes at the end of headings." The markdown
renderer rewrites a Setext heading as ATX text, so a heading whose text ends in a hash run (`foo #` from
`foo #\n---`) is emitted as `## foo #`, and the trailing hash is consumed as an ATX closing sequence on
reparse. The frozen patch escapes the run when a space or the start of the text precedes it.

- Controls: `junzhejiao/teacher-controls-20261126-case6` — valid (empty fails, gold passes).
- Grading: `junzhejiao/teacher-verification-20261126-case6` — valid controls and `teacher_resolved: true`,
  zero model calls in the job.
- Local round trips: Setext trailing hash, escaped ATX hash, hash-only heading, closing-sequence heading and
  attached hashes all keep the same HTML after reformatting.
- `w3lib_338` was attempted first in this window and deferred: "scan the undecoded bytes of a document before
  decoding it" names no function or symptom in the unpatched tree, and this checkout already contains an
  `EncodingDecision`/`read_bom` layer, so a patch would be guesswork.

## Case 7 (2026-11-26, same session)

**`mistune_395` — executed, resolved.** "Certain characters in markdown code incorrectly parsed (e.g., &)."
`escape_url` returned `escape(quote(unescape(link), safe=safe))`, so a URL containing `&metric=` rendered as
`&amp;metric=` inside the link, and the result depended on how the `html` module resolved. The frozen patch
drops the `escape()` wrapper, which is exactly what the public issue's comment proposes.

- Controls: `junzhejiao/teacher-controls-20261126-case7` — valid (empty fails, gold passes).
- Grading: `junzhejiao/teacher-verification-20261126-case7` — valid controls and `teacher_resolved: true`,
  zero model calls in the job.
- Local smoke: the badge URL now keeps its literal ampersand.

## Case 8 (2026-11-26, same session)

**`mistune_502` — resolved on the second attempt.** "Fix CommonMark 0.31.2 inline HTML comment recognition."
The first frozen pattern (`<!--(?:>|[\s\S]*?--)`) handled `<!-->` and bodies with internal `--`, but for
`foo <!---> foo -->` it swallowed everything up to the later terminator, so the expected `<p>foo <!---> foo
--&gt;</p>` came back as a live comment. The visible failing case named the missing `->` terminator; the second
attempt added it and the official verifier accepted the patch.

- Controls: `junzhejiao/teacher-controls-20261126-case8` — valid (empty fails, gold passes).
- Grading: `junzhejiao/teacher-verification-20261126-case8` v3 — valid controls and `teacher_resolved: true`,
  zero model calls in the job.
- Both 0.31.2 examples now render exactly as the corpus expects.

This case is a repair-with-feedback, not a frozen-prior prediction; the distinction is recorded in the receipt.

## Case 9 (2026-11-27, same session)

**`w3lib_317` — deferred, not coded.** "Handle UTF-{16,32} the same for all 3 sources of encoding." The
checkout confirms the discrepancy the title refers to: header and meta labels resolve to bare `utf-16`/`utf-32`
(decoding then depends on a BOM), while `read_bom` returns `utf-16-le`/`utf-16-be`. The title does not say which
normalisation is wanted — prefer the BOM, reject the bare label, or pick an endianness — and the choice needs the
bytes, not the label. A patch would be a guess about the expected contract, so the case is deferred like
`more_itertools_1116` and `w3lib_338`.

## Case 10 (2026-11-27, same session)

**`w3lib_167` — resolved on the second attempt.** "Expose the separator parameter added upstream to parse_qs and
parse_qsl." Attempt 1 threaded a separator through `parse_qsl_to_bytes`, `url_query_cleaner` and the internal
helpers; the official tests then failed with `TypeError`, naming the surface they expect: `separator=` on
`url_query_parameter` and `add_or_replace_parameter`, and `query_separator=` on `canonicalize_url`. Attempt 2
added those three and the internal plumbing, and the verifier accepted it.

- Controls: `junzhejiao/teacher-controls-20261127-case10` — valid (empty fails, gold passes).
- Grading: `junzhejiao/teacher-verification-20261127-case10` v2 — valid controls and `teacher_resolved: true`,
  zero model calls in the job.
- The retry used the visible failure, so it is recorded as repair-with-feedback, like case 8.

## Decision and next gate

Do not derive a graph-forcing student instruction from this case. Literal search already supplied the
observed caller, while the contract and two execution paths drove the repair. This is a null information
gain for this partial tool on one explicitly named API task, not a rejection of #39 or the official graph.

Keep the 12-task development manifest frozen. The next predeclared cases can test whether structural
relationships help when a repair crosses functions. Any procedure must recur across tasks, help an
independent teacher under a controlled tool interface, then survive a separate Gemma A/B on held-out
repositories. No student prompt, adapter, baseline agent or final-submission selection is changed.
