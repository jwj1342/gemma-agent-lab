#!/bin/bash
set -eo pipefail
source /home/jwj/venvs/gemma4-eval-v29/bin/activate
export HOME=/work/run/home
export TMPDIR=/tmp
export XDG_CACHE_HOME=/work/run/cache/xdg
export HF_HOME=/work/run/cache/huggingface
export TORCH_HOME=/work/run/cache/torch
export TORCH_EXTENSIONS_DIR=/work/run/cache/torch-extensions
export TRITON_CACHE_DIR=/work/run/cache/triton
export VLLM_CACHE_ROOT=/work/run/cache/vllm
export FLASHINFER_WORKSPACE_BASE=/work/run/cache/flashinfer
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export LITELLM_LOCAL_MODEL_COST_MAP=True OTEL_SDK_DISABLED=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled PYTHONNOUSERSITE=1
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
python - <<'PY'
import json, os, shutil, subprocess, tempfile
from pathlib import Path
readonly={p:bool(os.statvfs(p).f_flag & os.ST_RDONLY)
          for p in ['/agent','/app','/home/jwj/venvs/gemma4-eval-v29']}
checks={'required_mounts_readonly':all(readonly.values()),
        'kaggle_credentials_not_mounted':not Path('/home/jwj/.kaggle').exists(),
        'full_project_not_mounted':not Path('/scratch/jwj/kaggle_gemma4_agent').exists(),
        'run_writable':os.access('/work/run',os.W_OK),
        'tmp_writable':os.access('/tmp',os.W_OK)}
commands={name:shutil.which(name) for name in ['git','patch','tar','find','bash','python3']}
checks['required_commands_available']=all(commands.values())
if commands['git']:
    with tempfile.TemporaryDirectory(dir='/work/run',prefix='git-preflight-') as tmp:
        def git(*args):
            return subprocess.check_output([commands['git'],'-C',tmp,*args],text=True,stderr=subprocess.STDOUT)
        git('init','-q')
        (Path(tmp)/'sample.txt').write_text('before\n')
        git('add','sample.txt')
        git('-c','user.name=Runtime check','-c','user.email=runtime-check@invalid',
            'commit','-qm','Check patch generation')
        (Path(tmp)/'sample.txt').write_text('after\n')
        diff=git('diff','HEAD')
        checks['git_patch_generation']=('-before' in diff and '+after' in diff)
Path('/work/run/runtime-filesystem.json').write_text(json.dumps(
    {'checks':checks,'readonly_mounts':readonly,'commands':commands,'home':os.environ['HOME'],
     'tmpdir':os.environ['TMPDIR'],'scope':'Filesystem checks inside the local Apptainer runner'},indent=2)+'\n')
if not all(checks.values()):raise SystemExit('Container filesystem checks failed')
PY
exec python "$@"
