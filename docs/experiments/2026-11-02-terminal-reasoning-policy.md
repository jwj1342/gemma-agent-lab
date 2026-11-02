# Terminal reasoning policy and verification

Policy A: under `--preserve-reasoning`, keep native tool-call reasoning and omit
terminal reasoning. Retain the terminal answer unchanged. Do not move thoughts
into visible content or modify the official inference template. This matches
what Kaggle v2 actually serializes, rather than exporting untrainable fields or
changing the model's final-answer contract. Default actions-only conversion is
unchanged.

The export policy is `native-tool-reasoning-terminal-omitted-v1`. Both per-selected
trajectory/example metadata and the export manifest count
`terminal_reasoning_dropped`: terminal messages whose nonempty reasoning was
omitted, not discarded answers or tokens. Counts cover selected exported rows,
not rejected or unselected rows. Null/empty reasoning contributes zero. Dedicated
reasoning fields still take priority over content for native tool calls.

The CPU integration test `tests/test_real_tokenizer_reasoning.py` verifies all
three Kaggle v2 file hashes before loading locally. It first demonstrates that
the unmodified template omits terminal thoughts but emits tool-call thoughts.
The policy regression failed before implementation for both string and content-
block terminal answers. Round-trip checks include reasoning present, absent and
empty, both flag states, tool-call thoughts and loss masking. Model weights and
GPU execution are unnecessary.

Run with transformers 5.13.1, tokenizers 0.22.2 and Jinja2 3.1.6:

```bash
PR29_TOKENIZER_DIR=/path/to/hash-matched/tokenizer python -m unittest discover -s tests -p test_real_tokenizer_reasoning.py -v
```

These integration tests are opt-in and skip without the explicit local directory;
ordinary CI does not download gated artifacts. An explicitly configured but
incorrect directory fails rather than skips. Real-tokenizer success establishes
serialization compatibility only, not training readiness or a repair gain.
Independent review and complete action-format coverage remain separate gates.

## Pinned public shard coverage

CPU audit of `SWE-bench/SWE-smith-trajectories` at revision
`08e109b4a59eaeebf80e4675cd125d42e7ac99a4`, shard
`tool-00004-of-00008.parquet`, SHA-256
`e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715`:

- 3,012 rows; 1,255 resolved trajectories; 21,424 resolved assistant messages.
- 143/1,255 (11.39%) fully convertible by the current native-only converter:
  accepted without omitted XML actions. This is structural coverage, not replay.
- 15,696/21,424 (73.26%) XML-only assistant actions, across 962 trajectories.
  There are 15,715 messages without native calls; the remaining 19 are final prose.
- 19 source terminal messages; 0/19 have nonempty explicit reasoning fields (0%).
  Terminal content is not inferred to be reasoning. Three terminal answers are
  emitted in accepted trajectories, with zero terminal reasoning omissions.
- 1,112 resolved rows are rejected: 1,111 UnsupportedAction and one KeyError.
  XML support alone will not necessarily address all unsupported actions.

Counts cover all resolved rows before deduplication, reserved-evaluation filters
or training caps. Do not confuse these with a selected training dataset. The
sanitized fixture in `tests/test_sft_coverage.py` mirrors native-call, observation,
XML-only and terminal row shapes using invented content only. The aggregate
counter never writes upstream identifiers or text; it verifies the shard hash.

```bash
python -m scripts.audit_sft_coverage --parquet /path/to/tool-00004-of-00008.parquet --output runs/coverage/new-receipt.json
```

Coverage is insufficient for a LoRA claim spanning this resolved corpus. XML
conversion is a prerequisite, tracked in [issue #30](https://github.com/jwj1342/gemma-agent-lab/issues/30),
which links the owner's private `convert_teacher.py` draft reference. No training
readiness or representative repair coverage is claimed. Independent review,
remaining action support and runtime validation are still required.
