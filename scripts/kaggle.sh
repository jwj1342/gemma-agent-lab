#!/bin/bash
# Run with: bash -l scripts/kaggle.sh <Kaggle CLI arguments>
set -eo pipefail
module load StdEnv/2023 python/3.12.4
source /home/jwj/venvs/kaggle-cli/bin/activate
exec kaggle "$@"
