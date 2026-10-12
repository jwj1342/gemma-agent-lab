# Prompt and context-learning candidate

The default package combines a concise workflow, illustrative decision examples,
task-local execution feedback and an optional repair playbook. It uses the official
base model with no adapter. These are research candidates, not a promoted
submission policy. The fast comparison did not establish a reliable benefit,
and ICL regressed in the larger development comparison. See the
[execution report](../../docs/experiments/2026-10-09-prompt-context.md) for results.

See [the research and experiment plan](../../docs/prompt-route.md) for sources,
competition constraints, ablations and cost accounting. Build `workflow`, `protocol`
and `icl` controls with `scripts/build_prompt_candidate.py`; the default is `reflect`.

The playbook and examples are team-authored generic guidance. They contain no
benchmark task solutions. Generated prompt variants remain in ignored run directories.
