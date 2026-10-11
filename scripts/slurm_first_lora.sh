#!/bin/bash
#SBATCH --job-name=gemma4-first-lora
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
# A bounded debug pilot uses --partition=debug at submission; omit it for regular training.
set -eo pipefail
project_root=${1:?project root}
run_dir=${2:?new run directory}
swe_data=${3:-"$project_root/data/processed/first-swe-graph-lora"}
init_adapter=${4:-}
cd "$project_root"
module load StdEnv/2023 python/3.12.4 cuda/12.9 apptainer/1.4.5
mkdir -p "$run_dir/cache" "$run_dir/tmp"
export HF_HOME="$run_dir/cache/hf" XDG_CACHE_HOME="$run_dir/cache/xdg"
export TORCH_HOME="$run_dir/cache/torch" TRITON_CACHE_DIR="$run_dir/cache/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
unset KAGGLE_API_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
unset APPTAINER_BIND APPTAINER_BINDPATH APPTAINER_MOUNT SINGULARITY_BIND SINGULARITY_BINDPATH
if [[ ! -d "$project_root/data/processed/first-graph-lora" ]]; then
  apptainer exec --cleanenv --containall --no-home --no-mount hostfs,bind-paths,cwd --pwd /app \
    --env LITELLM_LOCAL_MODEL_COST_MAP=True --env OTEL_SDK_DISABLED=true --env HF_HUB_OFFLINE=1 \
    --bind "$project_root/src:/app/src:ro" --bind "$project_root/scripts:/app/scripts:ro" \
    --bind "/home/jwj/venvs/gemma4-eval:/home/jwj/venvs/gemma4-eval:ro" \
    --bind "$project_root/data/raw/gemma-4-developer-agent:/data:ro" \
    --bind "$project_root/data/processed:/output" --bind "$run_dir/tmp:/tmp" \
    "$project_root/resource/downloads/images/vllm-0.19.1.sif" \
    /home/jwj/venvs/gemma4-eval/bin/python -m scripts.prepare_graph_sft \
    --tasks /data/tasks.jsonl --graphs /data/graphs --embeddings /data/embeddings --output /output/first-graph-lora
fi
if [[ -n "$init_adapter" && ! -d "$run_dir/terminal-data" ]]; then
  apptainer exec --cleanenv --containall --no-home --no-mount hostfs,bind-paths,cwd --pwd /app \
    --env LITELLM_LOCAL_MODEL_COST_MAP=True --env OTEL_SDK_DISABLED=true --env HF_HUB_OFFLINE=1 \
    --bind "$run_dir/inputs/training/src:/app/src:ro" --bind "$run_dir/inputs/training/scripts:/app/scripts:ro" \
    --bind "/home/jwj/venvs/gemma4-eval:/home/jwj/venvs/gemma4-eval:ro" \
    --bind "$run_dir:/output" --bind "$run_dir/tmp:/tmp" \
    "$project_root/resource/downloads/images/vllm-0.19.1.sif" \
    /bin/bash -c 'source /home/jwj/venvs/gemma4-eval/bin/activate; exec python -m scripts.prepare_terminal_sft --output /output/terminal-data'
fi
source /home/jwj/venvs/gemma4-train/bin/activate
base="$project_root/model/base/gemma-4-31b-it-qat-w4a16-ct-v2"
data="$swe_data"
graph_data="$project_root/data/processed/first-graph-lora"
test -f "$run_dir/source-manifest.json"
configuration=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["configuration"])' "$run_dir/source-manifest.json")
cd "$run_dir/inputs/training"
training_args=(--base "$base" --data "$data" --graph-data "$graph_data" --output "$run_dir/train" --config "$configuration")
if [[ -n "$init_adapter" ]]; then
  training_args+=(--terminal-data "$run_dir/terminal-data" --init-adapter "$init_adapter")
fi
# Two separate processes exercise loading the saved adapter, optimizer, and RNG state.
python -u -m scripts.train_first_lora "${training_args[@]}" --stop-after 8
python -u -m scripts.train_first_lora "${training_args[@]}" --resume "$run_dir/train/checkpoint-0008"
