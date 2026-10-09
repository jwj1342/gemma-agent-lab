#!/bin/bash
#SBATCH --job-name=swe-lora-validate
#SBATCH --account=def-zhouyang
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:30:00
set -eo pipefail
project_root=${1:?project asset root}
train_run=${2:?completed training run}
code_root=${3:?frozen validation source worktree}
baseline_run=${4:?frozen matching baseline suite}
cd "$code_root"
module load StdEnv/2023 python/3.12.4 apptainer/1.4.5
source /home/jwj/venvs/gemma4-compiler/bin/activate
export GEMMA_PROJECT_ROOT="$project_root" GEMMA_CODE_ROOT="$code_root"
unset KAGGLE_API_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
test -f "$train_run/train/completed.json"
python scripts/check_repository.py
python -m unittest discover -s tests -v

# Verify both adapter loading and a probability change on the required W4A16 base.
bash scripts/slurm_check_lora.sh "$project_root" "$train_run" "$train_run/swe-data" "$train_run/graph-data"
python scripts/export_swe_lora_candidate.py --training "$train_run/train" \
  --baseline configs/baseline --output "$train_run/agent"

smoke="$train_run/submission-smoke"
python scripts/prepare_eval_run.py --run-dir "$smoke" --agent-dir "$train_run/agent" \
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

fast="$train_run/fast"
python scripts/prepare_eval_suite.py --run-dir "$fast" --agent-dir "$train_run/agent" \
  --tasks "$project_root/data/raw/gemma-4-developer-agent/tasks.jsonl" --cohort fast --runtime-from "$baseline_run"
bash "$fast/inputs/runtime/run_eval_container.sh" compile "$fast"
python "$fast/inputs/runtime/run_eval_suite.py" --project-root "$project_root" --run-dir "$fast"
python scripts/submit_competition.py --run-dir "$smoke"
# This is an artifact check only. Upload from the login/datamover node after review.
