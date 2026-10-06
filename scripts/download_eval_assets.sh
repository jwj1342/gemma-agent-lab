#!/bin/bash
# Run on a data transfer node; never perform this large transfer on a login node.
set -eo pipefail
case "$(hostname -s)" in *-dm*) ;; *) echo 'Run this script on a datamover node.' >&2; exit 2;; esac
module load StdEnv/2023 python/3.12.4
source /home/jwj/venvs/kaggle-cli/bin/activate
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_dir="$project_root/model/base/gemma-4-31b-it-qat-w4a16-ct-v2"
wheel_dir="$project_root/resource/downloads/eval-wheelhouse"
mkdir -p "$model_dir" "$wheel_dir"
kaggle models instances versions download \
  google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2 \
  --path "$model_dir" --untar --quiet
kaggle datasets download metric/gemma-4-developer-agent-wheelhouse \
  --path "$wheel_dir" --unzip --quiet
echo 'Model and official evaluation wheelhouse downloaded.'
