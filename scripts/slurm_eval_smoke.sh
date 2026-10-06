#!/bin/bash
#SBATCH --job-name=gemma4-smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
set -eo pipefail
project_root=${1:?project root}; run_dir=${2:?run directory}; task_id=${3:?task id}
cd "$project_root"
module load StdEnv/2023 python/3.12.4 apptainer/1.4.5
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bash scripts/run_eval_container.sh compile "$run_dir"
mkdir -p "$run_dir/infer"
bash scripts/run_eval_container.sh verify "$run_dir" --task-id "$task_id" --patch-source empty --results /work/run/empty
bash scripts/run_eval_container.sh verify "$run_dir" --task-id "$task_id" --patch-source gold --results /work/run/gold
python scripts/summarize_eval_smoke.py --run-dir "$run_dir" --controls-only
bash scripts/run_eval_container.sh infer "$run_dir" --task-id "$task_id"
bash scripts/run_eval_container.sh verify "$run_dir" --task-id "$task_id" --patch-source agent --results /work/run/agent
python scripts/summarize_eval_smoke.py --run-dir "$run_dir"
