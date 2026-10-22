# Team implementations

Add agent tools, data transformations, training, and evaluation code here.
Keep upstream references in `resource/`. Record attribution and changes when adapting code,
and preserve the original notebooks. Write outputs to a dedicated `runs/<run-id>/` directory.

[`sft_data.py`](sft_data.py) implements the first trajectory conversion, snapshot grouping, JSONL
reader, and Gemma assistant-action loss masks. Entry points live in `scripts/`; the
[first LoRA guide](../docs/first-lora.md) explains what the pilot does and how it is evaluated.
