#!/bin/bash
#SBATCH --job-name=gemma4-eval-suite
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
set -eo pipefail
project_root=${1:?project root}
run_dir=${2:?prepared suite directory}
shift 2
module load StdEnv/2023 python/3.12.4 apptainer/1.4.5
cd "$project_root"
python "$run_dir/inputs/runtime/run_eval_suite.py" --project-root "$project_root" --run-dir "$run_dir" "$@"
