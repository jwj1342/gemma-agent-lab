# Lightweight preparation checks

Run from the repository root with Python 3.10+, Git, Bash, and the development dependencies:

```bash
python3 -m pip install -r configs/ci.requirements.txt
python3 -m unittest discover -s tests -v
```

`test_kaggle_preparation.py` generates real notebooks through the command-line entry point,
checks the embedded shared scripts and baseline ZIP, and executes only the input-preparation
cell against tiny temporary mount fixtures. It checks label removal, frozen-input tampering,
input mismatch, execution provenance, the downloaded-run submission dry-run, and output preservation.
Synthetic result JSON files exercise the submission handoff without claiming model performance.
The wheel-inventory test also executes the start of the installation cell with pip mocked, requiring
unexpected wheels to fail before any installation or deletion. It never installs the wheelhouse, contacts
Kaggle, loads a model, runs repository tests, or consumes GPU quota.

`test_repository_checks.py` verifies the CI gates using tiny temporary Git repositories. It checks
that missing/untracked local links, escaping configuration references, ignored tracked artifacts,
broken YAML, and team/agent Python syntax errors fail. Valid nested references and complete agent
packages pass. Original upstream material is excluded from these gates.

The notebook-preparation tests alone still use only the standard library:
`python3 -m unittest discover -s tests -p test_kaggle_preparation.py -v`.
See [CI and review](../docs/ci-and-review.md) for the hosted workflow and local static-check command.

Changes to inference, hardware settings, dependencies, or agent behavior also need a platform
experiment with valid controls. Historical runtime evidence is indexed in
[resource/manifests/](../resource/manifests/README.md).

`test_wheelhouse_runtime.py` covers old/new snapshot selection, same-name byte changes,
explicit record mode, package metadata, malformed registries, and mutations between
setup and installation. The notebook tests also check policy embedding and receipts.
