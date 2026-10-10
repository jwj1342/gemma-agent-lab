#!/bin/bash
#SBATCH --job-name=swe-lora-light
#SBATCH --account=def-zhouyang
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
set -eo pipefail
project_root=${1:?project root}
run_dir=${2:?frozen run directory}
code_root=${3:?frozen source}
stop_after=${4:-}
configuration=${5:-configs/training/swe-lora-light.json}
bash "$code_root/scripts/slurm_swe_lora_v1.sh" "$project_root" "$run_dir" "$code_root" \
  "$stop_after" "$configuration" 1
