# Lightweight preparation checks

Run from the repository root with Python 3.10+:

```bash
python3 -m unittest discover -s tests -v
```

`test_kaggle_preparation.py` generates real notebooks through the command-line entry point,
checks the embedded shared scripts and baseline ZIP, and executes only the input-preparation
cell against tiny temporary mount fixtures. It checks label removal, frozen-task tampering,
input mismatch, and protection of existing output. It never installs the wheelhouse, contacts
Kaggle, loads a model, runs repository tests, or consumes GPU quota.

Changes to inference, hardware settings, dependencies, or agent behavior also need a platform
experiment with valid controls. Historical runtime evidence is indexed in
[resource/manifests/](../resource/manifests/README.md).
