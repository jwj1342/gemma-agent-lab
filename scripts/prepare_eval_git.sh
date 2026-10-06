#!/bin/bash
# Install Ubuntu's Git executable into the HOME evaluation environment, without root.
set -eo pipefail
case "$(hostname -s)" in *-dm*) ;; *) echo 'Run on a datamover node.' >&2; exit 2;; esac
module load apptainer/1.4.5
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image="$project_root/resource/downloads/images/vllm-0.19.1.sif"
apptainer exec --cleanenv "$image" /bin/bash "$project_root/scripts/install_eval_git.sh" "$project_root"
