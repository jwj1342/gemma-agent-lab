# Configuration

Version agent YAML, data split rules, training parameters, environment versions, and evaluation settings.
Keep account credentials in HOME and out of configuration files.

## Agent files

| File | What it controls |
| --- | --- |
| [baseline/agent.yaml](baseline/agent.yaml) | Required model, main agent identity, allowed tools, and included prompts/sub-agent |
| [baseline/prompts/system.md](baseline/prompts/system.md) | Main agent's instructions for investigating and repairing issues |
| [baseline/sub_agents/code_analyzer.yaml](baseline/sub_agents/code_analyzer.yaml) | Analyzer agent configuration and tools |
| [baseline/prompts/analyzer.md](baseline/prompts/analyzer.md) | Analyzer's repository-reading instructions |
| [baseline/configs/sampling.yaml](baseline/configs/sampling.yaml) | Generation settings, output budget, thinking settings, and seed |
| [baseline/eval_config.yaml](baseline/eval_config.yaml) | Per-task time, tool-call, turn, and command-timeout budgets |

Copy `baseline/` to `experiments/<name>/` for a variant. Keep reports outside the agent directory:
the packaging script includes every allowed file underneath it. Existing smoke evidence applies
to the unchanged baseline ZIP, not automatically to a new variant.

## Environment records

| File | Scope |
| --- | --- |
| [kaggle-cli.requirements.txt](kaggle-cli.requirements.txt) | Maintainer's Kaggle collection/upload environment snapshot |
| [compiler-precheck.requirements.txt](compiler-precheck.requirements.txt) | Preliminary native compiler checks on Trillium; not the complete runtime |
| [eval-container.requirements.txt](eval-container.requirements.txt) | Prepared Trillium container environment package freeze |
| [eval-container.requirements.official-constraints.txt](eval-container.requirements.official-constraints.txt) | Official wheel versions preserved during installation |
| [eval-container.requirements.versions.json](eval-container.requirements.versions.json) | Installed package versions and official-source identity |
| [eval-container.requirements.dependency-audit.json](eval-container.requirements.dependency-audit.json) | Known dependency metadata conflicts and preparation checks |

Kaggle uses the pinned official starter image and mounted wheelhouse selected by
`scripts/prepare_kaggle_runtime.py`. Its default tensor-parallel size is 4 and memory fraction is
0.90; the existing H100 runner uses 1 and 0.80. See [runtime choices](../docs/kaggle-runtime.md).
These settings belong to the execution platform, not the submitted agent bundle.

`kaggle-cli.requirements.txt` captures the maintainer's download-tool environment, not training dependencies.
Lock evaluation and training dependencies separately; see the [training plan](../docs/training-plan.md).
