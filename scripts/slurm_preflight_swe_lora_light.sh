#!/bin/bash
#SBATCH --job-name=swe-lora-light-check
#SBATCH --account=def-zhouyang
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
set -eo pipefail
project_root=${1:?project root}
train_run=${2:?partial training run}
code_root=${3:?frozen source}
probe_name=${4:-preflight-validation}
[[ "$probe_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'probe_name must be a simple directory name' >&2; exit 2; }
bash "$code_root/scripts/slurm_preflight_swe_lora.sh" "$project_root" "$train_run" "$code_root" 32 "$probe_name"

# Register these exposed development tasks before training; do not replace
# failed tasks or tune against the protected holdout.
cd "$code_root"
module load StdEnv/2023 python/3.12.4 apptainer/1.4.5
source /home/jwj/venvs/gemma4-compiler/bin/activate
export GEMMA_PROJECT_ROOT="$project_root" GEMMA_CODE_ROOT="$code_root"
unset KAGGLE_API_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
for task in fastapi_14455 requests_7502; do
  smoke="$train_run/$probe_name/smoke-$task"
  python scripts/prepare_eval_run.py --run-dir "$smoke" --agent-dir "$train_run/$probe_name/agent" \
    --task-id "$task" --tasks "$project_root/data/raw/gemma-4-developer-agent/tasks.jsonl"
  runner="$smoke/inputs/runtime/run_eval_container.sh"
  bash "$runner" compile "$smoke"
  mkdir -p "$smoke/infer"
  bash "$runner" verify "$smoke" --task-id "$task" --patch-source empty --results /work/run/empty
  bash "$runner" verify "$smoke" --task-id "$task" --patch-source gold --results /work/run/gold
  python "$smoke/inputs/runtime/summarize_eval_smoke.py" --run-dir "$smoke" --controls-only
  bash "$runner" infer "$smoke" --task-id "$task"
  bash "$runner" verify "$smoke" --task-id "$task" --patch-source agent --results /work/run/agent
  python "$smoke/inputs/runtime/summarize_eval_smoke.py" --run-dir "$smoke"
done
