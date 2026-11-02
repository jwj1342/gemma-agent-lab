# Proposed converter extraction: approval required before execution

Do not create a branch or PR, move commits, or rewrite PR #29 until the user
confirms this plan. This is a review boundary proposal only.

## Small converter PR

Start from the current approved main revision and port only the minimal SFT
changes, preserving main's intervening behavior:

- `src/sft_data.py`: opt-in explicit/native-content tool reasoning, explicit-field
  priority, terminal omission with export counters, default actions-only behavior.
- `scripts/prepare_sft_data.py`: flag and versioned metadata, per-selected-row and
  aggregate terminal omission counts.
- Relevant additions in `tests/test_sft_data.py`, the export-receipt tests and
  `tests/test_real_tokenizer_reasoning.py`. Keep actual upstream tokenizer files
  local; integration tests verify hashes and run explicitly on CPU.
- A concise policy document and validation commands. Include the sanitized native
  fixture needed by these tests, never task data or private trace annotations.

Do not move complete historical experiment reports or audit infrastructure into
this PR. Keep `audit_sft_coverage.py`, its test, public coverage discussion and
XML follow-up in the research PR unless reviewers explicitly request the small
counter alongside the converter. Link issue #30 as a full-corpus training gate.

## Remaining PR #29

Retain failure auditing, candidate packaging, research reports and checkpoint
protocol. Link the converter PR as a dependency where relevant. Rebase/reconcile
only after user approval; do not discard unrelated local work or force-push.
Rerun both full CPU suites and hash-pinned tokenizer checks after extraction.

## Review and delivery gates

The extracted PR claims native-format serialization compatibility, not an
end-to-end training pipeline. Request independent review of precedence, terminal
policy, metadata semantics and loss boundaries. Report opt-in tests separately
from ordinary CI skips. Confirm scope and chosen main base with the user before
creating either new branch or PR. No split has been performed.
