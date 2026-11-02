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
