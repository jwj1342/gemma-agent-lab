#!/bin/bash
set -eo pipefail
case "$(hostname -s)" in *-dm*) ;; *) echo 'Run on a datamover node.' >&2; exit 2;; esac
module load apptainer/1.4.5
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image="$project_root/resource/downloads/images/vllm-0.19.1.sif"
eval_venv=/home/jwj/venvs/gemma4-eval-v31
export APPTAINER_CACHEDIR="$project_root/cache/apptainer"
export APPTAINER_TMPDIR="$project_root/cache/apptainer-tmp"
test -f "$image"
if [[ ! -f "$eval_venv/pyvenv.cfg" ]]; then
  apptainer exec --cleanenv "$image" /usr/bin/python3.12 -m venv --system-site-packages "$eval_venv"
fi
apptainer exec --cleanenv "$image" "$eval_venv/bin/python" \
  "$project_root/scripts/install_eval_environment.py" \
  --wheelhouse "$project_root/resource/downloads/eval-wheelhouse-v31" \
  --normalized-wheelhouse "$project_root/cache/eval-wheels-v31-normalized" \
  --test-wheelhouse "$project_root/data/raw/gemma-4-developer-agent/wheels" \
  --lock "$project_root/configs/eval-container.requirements.txt"
