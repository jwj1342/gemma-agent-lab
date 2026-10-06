#!/bin/bash
set -eo pipefail
case "$(hostname -s)" in *-dm*) ;; *) echo 'Run on a datamover node.' >&2; exit 2;; esac
module load apptainer/1.4.5
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export APPTAINER_CACHEDIR="$project_root/cache/apptainer"
export APPTAINER_TMPDIR="$project_root/cache/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$project_root/resource/downloads/images"
image="$project_root/resource/downloads/images/vllm-0.19.1.sif"
if [[ ! -f "$image" ]]; then
  apptainer build --mksquashfs-args '-processors 2' "$image" \
    docker://vllm/vllm-openai@sha256:89c1d0629d377daa3f7f369cbea6167a7b48ea89aaacd12555e2b0b2f7f740d3
fi
apptainer exec --cleanenv "$image" /usr/bin/python3.12 -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda)'
