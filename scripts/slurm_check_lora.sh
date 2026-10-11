#!/bin/bash
#SBATCH --job-name=gemma4-lora-serving
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
set -eo pipefail
project_root=${1:?project root}
code_root=${GEMMA_CODE_ROOT:-$project_root}
train_run=${2:?completed training run}
swe_dataset=${3:-"$project_root/data/processed/first-swe-graph-lora"}
graph_dataset=${4:-"$project_root/data/processed/first-graph-lora"}
if [[ "$swe_dataset" != /* ]]; then swe_dataset="$project_root/data/processed/$swe_dataset"; fi
module load StdEnv/2023 apptainer/1.4.5
test -f "$train_run/train/training-result.json" || test -f "$train_run/train/completed.json"
mkdir -p "$train_run/serving-work/tmp" "$train_run/serving-work/cache"
unset APPTAINER_BIND APPTAINER_BINDPATH APPTAINER_MOUNT SINGULARITY_BIND SINGULARITY_BINDPATH
terminal_mounts=()
terminal_args=()
if [[ -d "$train_run/terminal-data" ]]; then
  terminal_mounts=(--bind "$train_run/terminal-data:/terminal-data:ro")
  terminal_args=(--terminal-data /terminal-data)
fi
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs,bind-paths,cwd --pwd /app \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --env "LORA_SERVING_PORT=$((20000 + SLURM_JOB_ID % 10000))" \
  --env HF_HUB_OFFLINE=1 --env LITELLM_LOCAL_MODEL_COST_MAP=True --env OTEL_SDK_DISABLED=true \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn --env VLLM_NO_USAGE_STATS=1 --env OMP_NUM_THREADS=4 \
  --env XDG_CACHE_HOME=/output/cache/xdg --env HF_HOME=/output/cache/hf --env VLLM_CACHE_ROOT=/output/cache/vllm \
  --env TRITON_CACHE_DIR=/output/cache/triton --env TORCH_EXTENSIONS_DIR=/output/cache/torch-extensions \
  --env FLASHINFER_WORKSPACE_BASE=/output/cache/flashinfer \
  --bind "$code_root/src:/app/src:ro" --bind "$code_root/scripts:/app/scripts:ro" \
  --bind /home/jwj/venvs/gemma4-eval-v29:/home/jwj/venvs/gemma4-eval-v29:ro \
  --bind "$project_root/model/base/gemma-4-31b-it-qat-w4a16-ct-v2:/model:ro" \
  --bind "$train_run/train/adapter:/adapter:ro" --bind "$swe_dataset:/swe-data:ro" \
  --bind "$graph_dataset:/graph-data:ro" \
  --bind "$train_run/serving-work:/output" --bind "$train_run/serving-work/tmp:/tmp" \
  "${terminal_mounts[@]}" \
  "$project_root/resource/downloads/images/vllm-0.19.1.sif" \
  /home/jwj/venvs/gemma4-eval-v29/bin/python -m scripts.check_lora_serving \
  --base /model --adapter /adapter --data /swe-data --graph-data /graph-data \
  --output /output/check "${terminal_args[@]}"
