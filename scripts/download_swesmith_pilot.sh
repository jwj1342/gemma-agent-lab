#!/bin/bash
# Download the pinned public SWE-smith pilot shard; on Trillium use a datamover.
set -eo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output="$project_root/data/raw/swesmith/tool-00004-of-00008.parquet"
expected=e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715
if [[ -f "$output" ]]; then
  printf '%s  %s\n' "$expected" "$output" | sha256sum --check
  exit
fi
case "$(hostname -s)" in
  tri-login*|trig-login*|trig[0-9]*|tri[0-9]*|robot[0-9]*|trig-robot*)
    echo 'Use a datamover for this transfer on Trillium.' >&2
    exit 2;;
esac
mkdir -p "$(dirname -- "$output")"
curl --fail --location --retry 3 \
  https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/resolve/08e109b4a59eaeebf80e4675cd125d42e7ac99a4/data/tool-00004-of-00008.parquet \
  --output "$output.partial"
printf '%s  %s\n' "$expected" "$output.partial" | sha256sum --check
mv -- "$output.partial" "$output"
