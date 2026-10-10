#!/bin/bash
#SBATCH --job-name=swe-lora-preflight-validate
#SBATCH --account=def-zhouyang
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
set -eo pipefail
project_root=${1:?project asset root}
train_run=${2:?partial training run}
code_root=${3:?frozen validation source}
probe_steps=${4:-100}
probe_name=${5:-preflight-validation}
[[ "$probe_steps" =~ ^[1-9][0-9]*$ ]] || { echo 'probe_steps must be a positive integer' >&2; exit 2; }
[[ "$probe_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'probe_name must be a simple directory name' >&2; exit 2; }
cd "$code_root"
module load StdEnv/2023 python/3.12.4 apptainer/1.4.5
source /home/jwj/venvs/gemma4-compiler/bin/activate
export GEMMA_PROJECT_ROOT="$project_root" GEMMA_CODE_ROOT="$code_root"
unset KAGGLE_API_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
python scripts/check_repository.py
python -m unittest discover -s tests -v

# This is an explicitly partial training probe. Never manufacture completed.json
# or modify the resumable training directory to make a release gate pass.
python - "$train_run" "$code_root" "$probe_steps" "$probe_name" <<'PY'
from pathlib import Path
import hashlib
import json
import shutil
import sys

run, code = map(Path, sys.argv[1:3])
probe_steps = int(sys.argv[3])
latest = json.loads((run / 'train/latest.json').read_text())
recovery = json.loads((run / 'train/resume-check.json').read_text())
if latest['step'] != probe_steps or not recovery['contract_matches'] or not recovery['optimizer_and_rng_restored']:
    raise ValueError('Require the registered partial probe and successful recovery')
source = run / 'train' / latest['checkpoint']
probe = run / sys.argv[4]
probe.mkdir()
adapter = probe / 'train/adapter'
adapter.mkdir(parents=True)
hashes = {}
for name in ('adapter_config.json', 'adapter_model.safetensors'):
    shutil.copy2(source / name, adapter / name)
    with (adapter / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
complete = run / 'train/completed.json'
training_completed = complete.exists() and json.loads(complete.read_text())['steps'] == probe_steps
status = (f'Completed {probe_steps}-step training; autonomous acceptance and release checks pending'
          if training_completed else
          f'Partial {probe_steps}-step protocol probe; not completed training or a release candidate')
record = {'status': status, 'training_completed': training_completed,
          'source_checkpoint': str(source), 'step': latest['step'], 'adapter_sha256': hashes,
          'recovery': recovery}
(probe / 'train/training-result.json').write_text(json.dumps(record, indent=2) + '\n')
(probe / 'terminal-data').symlink_to((run / 'terminal-data').resolve(), target_is_directory=True)
agent = probe / 'agent'
shutil.copytree(code / 'configs/baseline', agent)
shutil.copytree(adapter, agent / 'adapters/swe_lora_v1')
definition = agent / 'agent.yaml'
if 'adapter:' in definition.read_text():
    raise ValueError('Baseline unexpectedly already specifies an adapter')
definition.write_text(definition.read_text() + 'adapter: swe_lora_v1\n')
print(json.dumps(record, indent=2))
PY

probe="$train_run/$probe_name"
bash scripts/slurm_check_lora.sh "$project_root" "$probe" "$train_run/swe-data" "$train_run/graph-data"
smoke="$probe/smoke"
python scripts/prepare_eval_run.py --run-dir "$smoke" --agent-dir "$probe/agent" \
  --task-id rich_3454 --tasks "$project_root/data/raw/gemma-4-developer-agent/tasks.jsonl"
runner="$smoke/inputs/runtime/run_eval_container.sh"
bash "$runner" compile "$smoke"
mkdir -p "$smoke/infer"
bash "$runner" verify "$smoke" --task-id rich_3454 --patch-source empty --results /work/run/empty
bash "$runner" verify "$smoke" --task-id rich_3454 --patch-source gold --results /work/run/gold
python "$smoke/inputs/runtime/summarize_eval_smoke.py" --run-dir "$smoke" --controls-only
bash "$runner" infer "$smoke" --task-id rich_3454
bash "$runner" verify "$smoke" --task-id rich_3454 --patch-source agent --results /work/run/agent
python "$smoke/inputs/runtime/summarize_eval_smoke.py" --run-dir "$smoke"
# No submission from this partial probe. A full training continuation and its
# separate release validation remain required even if this smoke succeeds.
