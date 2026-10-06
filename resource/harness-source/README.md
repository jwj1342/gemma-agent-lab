# Official harness source snapshot

Source: the competition's official [wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse).
Only Python source and wheel `METADATA` were extracted for inspection; these snapshots were not imported,
installed, or executed during collection. Versions, sizes, and SHA256 hashes of the three wheels are in the
[manifest](../manifests/harness-wheels.json).

The review focused on:

- `swegemma/config.py`: 3 GiB limit, allowed extensions, and output/thinking token ranges.
- `adk_submission/server.py`: vLLM LoRA settings derived from actual adapter count and rank.
- `adk_submission/resolvers/generation.py`: thinking-budget and seed forwarding to LiteLLM/vLLM.
- `swegemma/tools/graph.py`: numeric argument conversion and result truncation.
- `swegemma/graph/`: graph/embedding retrieval paths; arbitrary natural-language embedding support must be verified.

Static inspection does not establish dependency compatibility, model startup, or behavior in the scoring environment.
