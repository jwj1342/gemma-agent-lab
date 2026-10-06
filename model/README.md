# Models and adapters

- `base/`: pinned base weights, tokenizer, configuration, and model manifest.
- `adapters/`: exported LoRA adapters validated against the official evaluation base.
- `runs/<run-id>/`: intermediate training checkpoints, separate from final adapters.

The required v2 base was downloaded and checked for file sizes, SHA256 hashes,
and safetensors offset bounds. Offline loading and one-task patch generation passed;
no adapter training has been completed. See [runtime evidence](../resource/manifests/hpc-smoke.json),
[asset verification](../resource/manifests/eval-assets-verification.json), and
[model references](../resource/models/README.md).

Weights stay outside Git. Record source, revision, hashes, license, and export environment.
Reuse a read-only base across experiments to avoid duplicate copies.
