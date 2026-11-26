# Team implementations

Add agent tools, data transformations, training, and evaluation code here.
Keep upstream references in `resource/`. Record attribution and changes when adapting code,
and preserve the original notebooks. Write outputs to a dedicated `runs/<run-id>/` directory.

[`sft_data.py`](sft_data.py) implements the first trajectory conversion, snapshot grouping, JSONL
reader, and Gemma assistant-action loss masks. Entry points live in `scripts/`; the
[first LoRA guide](../docs/first-lora.md) explains what the pilot does and how it is evaluated.

[source_graph.py](source_graph.py) is a read-only partial AST navigation prototype for the
[teacher mechanism pilot](../docs/experiments/2026-11-25-teacher-source-graph-pilot.md). It reports
syntactic links and unresolved calls; it is not a sound runtime graph or a registered competition tool.
