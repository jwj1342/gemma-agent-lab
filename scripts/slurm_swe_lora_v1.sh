#!/bin/bash
#SBATCH --job-name=swe-lora-v1
#SBATCH --account=def-zhouyang
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
set -eo pipefail
project_root=${1:?project root}
run_dir=${2:?frozen run directory}
cd "$run_dir/inputs/training"
module load StdEnv/2023 python/3.12.4 cuda/12.9 arrow/25.0.0 apptainer/1.4.5
mkdir -p "$run_dir/cache" "$run_dir/tmp"
export HF_HOME="$run_dir/cache/hf" XDG_CACHE_HOME="$run_dir/cache/xdg"
export TORCH_HOME="$run_dir/cache/torch" TRITON_CACHE_DIR="$run_dir/cache/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset KAGGLE_API_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
unset APPTAINER_BIND APPTAINER_BINDPATH APPTAINER_MOUNT SINGULARITY_BIND SINGULARITY_BINDPATH
source /home/jwj/venvs/gemma4-train/bin/activate
python scripts/check_repository.py
python -m unittest discover -s tests -v
configuration=configs/training/swe-lora-v1.json
base="$project_root/model/base/gemma-4-31b-it-qat-w4a16-ct-v2"
if [[ ! -d "$run_dir/swe-data" ]]; then
  python -m scripts.prepare_sft_data --parquet "$project_root/data/raw/swesmith/tool-00004-of-00008.parquet" \
    --shard-manifest resource/manifests/swesmith-tool-shards.json \
    --output "$run_dir/swe-data" --max-trajectories 768 --reject-setup-actions \
    --tasks "$project_root/data/raw/gemma-4-developer-agent/tasks.jsonl"
fi
eval_env=/home/jwj/venvs/gemma4-eval-v29
container=(apptainer exec --cleanenv --containall --no-home --no-mount hostfs,bind-paths,cwd --pwd /app
  --env LITELLM_LOCAL_MODEL_COST_MAP=True --env OTEL_SDK_DISABLED=true --env HF_HUB_OFFLINE=1
  --bind "$PWD:/app:ro" --bind "$eval_env:$eval_env:ro"
  --bind "$project_root/data/raw/gemma-4-developer-agent:/data:ro"
  --bind "$run_dir:/output" --bind "$run_dir/tmp:/tmp"
  "$project_root/resource/downloads/images/vllm-0.19.1.sif")
if [[ ! -d "$run_dir/graph-data" ]]; then
  "${container[@]}" "$eval_env/bin/python" -m scripts.prepare_graph_sft \
    --tasks /data/tasks.jsonl --graphs /data/graphs --embeddings /data/embeddings --output /output/graph-data
fi
if [[ ! -d "$run_dir/terminal-data" ]]; then
  "${container[@]}" /bin/bash -c 'source /home/jwj/venvs/gemma4-eval-v29/bin/activate; exec python -m scripts.prepare_terminal_sft --output /output/terminal-data'
fi
if [[ ! -d "$run_dir/encoded-data" ]]; then
  python -m scripts.encode_sft_dataset --base "$base" --swe "$run_dir/swe-data" \
    --graph "$run_dir/graph-data" --terminal "$run_dir/terminal-data" --config "$configuration" \
    --output "$run_dir/encoded-data"
fi
train_args=(--base "$base" --data "$run_dir/encoded-data" --config "$configuration" --output "$run_dir/train")
if [[ ! -f "$run_dir/train/latest.json" ]]; then
  torchrun --standalone --nproc_per_node=4 -m scripts.train_swe_lora "${train_args[@]}" --probe --stop-after 8
fi
checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run_dir/train/latest.json")
torchrun --standalone --nproc_per_node=4 -m scripts.train_swe_lora "${train_args[@]}" --resume "$run_dir/train/$checkpoint"
