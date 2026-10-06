#!/bin/bash
# Large transfers and extraction belong on a data transfer node.
set -eo pipefail
module load StdEnv/2023 python/3.12.4
source /home/jwj/venvs/kaggle-cli/bin/activate
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
download_dir="$project_root/resource/downloads"
mkdir -p "$download_dir"
kaggle competitions download -c gemma-4-developer-agent -p "$download_dir"
python "$project_root/scripts/extract_competition.py"
