#!/bin/bash
# Invoke from a compute allocation for infer/verify. compile is a light import check.
set -eo pipefail
mode=${1:?mode}; run_dir=${2:?run directory}; shift 2
project_root=${GEMMA_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
run_dir=$(realpath "$run_dir")
module load apptainer/1.4.5
# Alliance exports /project,/scratch as user bind mounts. --cleanenv and
# --no-mount bind-paths do not disable those client-side environment settings.
# This runner supplies its complete mount list explicitly below.
unset APPTAINER_BIND APPTAINER_BINDPATH APPTAINER_MOUNT
unset SINGULARITY_BIND SINGULARITY_BINDPATH SINGULARITY_MOUNT
image="$project_root/resource/downloads/images/vllm-0.19.1.sif"
eval_venv=/home/jwj/venvs/gemma4-eval
data_dir="$project_root/data/raw/gemma-4-developer-agent"
model_dir="$project_root/model/base/gemma-4-31b-it-qat-w4a16-ct-v2"
agent_dir="$run_dir/inputs/agent"
test -f "$agent_dir/agent.yaml"
case "$mode" in infer|verify|compile) ;; *) echo 'Mode must be infer, verify or compile' >&2; exit 2;; esac
if [[ "$mode" != compile && -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Inference and repository tests must run within a Slurm allocation.' >&2; exit 2
fi
runtime_dir="$project_root/scripts"
if [[ -d "$run_dir/inputs/runtime" ]]; then runtime_dir="$run_dir/inputs/runtime"; fi
python3 "$runtime_dir/verify_eval_run.py" --run-dir "$run_dir"
work="$run_dir/$mode"
mkdir -p "$work/tmp" "$work/home"
args=(exec --cleanenv --containall --no-home --no-mount hostfs,bind-paths,cwd
  --pwd /work/run
  --bind "$runtime_dir:/app:ro" --bind "$agent_dir:/agent:ro"
  --bind "$eval_venv:$eval_venv:ro" --bind "$work:/work/run" --bind "$work/tmp:/tmp")
if [[ -f "$runtime_dir/local_eval_setup.py" && -d "$run_dir/inputs/runtime" ]]; then
  args+=(--bind "$project_root/resource/downloads/eval-test-wheels-v3:/eval-test-wheels:ro")
fi
if [[ "$mode" == compile ]]; then
  exec apptainer "${args[@]}" "$image" /bin/bash /app/container_entry.sh \
    /app/validate_agent.py --agent-dir /agent --output /work/run/validation.json "$@"
fi
args+=(--bind "$data_dir/snapshots:/data/snapshots:ro" --bind "$data_dir/graphs:/data/graphs:ro"
  --bind "$data_dir/embeddings:/data/embeddings:ro" --bind "$data_dir/wheels:/data/wheels:ro")
if [[ "$mode" == infer ]]; then
  args+=(--nv --bind "$model_dir:/model:ro" --bind "$run_dir/inputs/tasks.jsonl:/data/tasks.jsonl:ro")
  args+=(--env "HPC_VLLM_PORT=$((20000 + SLURM_JOB_ID % 10000))")
  if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    args+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  fi
else
  args+=(--bind "$data_dir/tasks.jsonl:/data/tasks.jsonl:ro" --bind "$run_dir/infer:/predictions:ro")
fi
exec apptainer "${args[@]}" "$image" /bin/bash /app/container_entry.sh /app/evaluate_agent.py "$mode" "$@"
