# Lightweight preparation checks

Run from the repository root with Python 3.10+, Git, Bash, and the development dependencies:

```bash
python3 -m pip install -r configs/ci.requirements.txt
python3 -m unittest discover -s tests -v
```

On Windows PowerShell with Git for Windows in its default location, expose Bash and
enable UTF-8 for Python before running the same checks:

```powershell
$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH
$env:PYTHONUTF8 = '1'
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

Adjust the Git path if installed elsewhere. Notebook tests compare the embedded ZIP with
a direct build and verify each member's exact source bytes; a historical compressed ZIP
hash is not a portable test fixture across Python versions and checkout line endings.
Frozen experiment hashes remain exact and are checked separately by `verify_eval_run.py`.

`test_kaggle_preparation.py` generates real notebooks through the command-line entry point,
checks the embedded shared scripts and baseline ZIP, and executes only the input-preparation
cell against tiny temporary mount fixtures. It checks label removal, frozen-input tampering,
input mismatch, execution provenance, the downloaded-run submission dry-run, and output preservation.
It also checks compressed inline payloads, rejection of source over Kaggle's 1 MB limit,
both observed input mount layouts, and the hash of an explicitly selected wheel inventory.
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
