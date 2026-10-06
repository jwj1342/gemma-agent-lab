"""Generate a private, offline L4x4 notebook for one real official-harness smoke task."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = (args.run_dir / 'inputs/submission.zip').read_bytes()
    if len(bundle) > 5 * 1024**2:
        raise ValueError('Inline runtime notebook only supports small baseline bundles')
    run = json.loads((args.run_dir / 'run-manifest.json').read_text())
    if hashlib.sha256(bundle).hexdigest() != run['agent_archive_sha256']:
        raise ValueError('Run bundle hash changed')
    official = json.loads((root / 'resource/notebooks/official/kernel-metadata.json').read_text())
    assets = json.loads((root / 'resource/manifests/eval-assets-verification.json').read_text())
    payload = {'bundle': base64.b64encode(bundle).decode(), 'bundle_sha256': run['agent_archive_sha256'],
               'public_task': (args.run_dir / 'inputs/tasks.jsonl').read_text(), 'task_id': run['task_id'],
               'scripts': {name: (root / 'scripts' / name).read_text() for name in
                           ['validate_agent.py', 'evaluate_agent.py', 'summarize_eval_smoke.py']},
               'wheels': {item['path']: item['sha256'] for item in assets['wheelhouse_files']}}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    setup = '''import base64, hashlib, json, os, subprocess, sys, zipfile, io
from pathlib import Path
PAYLOAD = json.loads(base64.b64decode(PAYLOAD_B64))
WORK = Path('/kaggle/working/runtime-smoke')
DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WHEELS = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
WORK.mkdir(parents=True, exist_ok=True)
(WORK/'scripts').mkdir(exist_ok=True)
(WORK/'inputs').mkdir(exist_ok=True)
AGENT = WORK/'inputs/agent'
raw = base64.b64decode(PAYLOAD['bundle'])
assert hashlib.sha256(raw).hexdigest() == PAYLOAD['bundle_sha256']
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts
    archive.extractall(AGENT)
(WORK/'inputs/tasks.jsonl').write_text(PAYLOAD['public_task'])
for name, source in PAYLOAD['scripts'].items():
    (WORK/'scripts'/name).write_text(source)
assert DATA.is_dir() and WHEELS.is_dir() and MODEL.is_dir()
for name, expected in PAYLOAD['wheels'].items():
    h=hashlib.sha256()
    with (WHEELS/name).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):h.update(chunk)
    assert h.hexdigest() == expected, 'Official wheelhouse changed: '+name
print('Exact local bundle and 41 official wheel hashes match.')
'''
    install = '''# Same installation scope as the reviewed official starter, in this disposable notebook.
import glob
for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pattern):
        Path(pth).unlink(missing_ok=True)
normalized = Path('/tmp/gemma4-official-wheels')
normalized.mkdir(exist_ok=True)
packages=[]
for source in sorted(WHEELS.glob('*.whl')):
    if 'cutlass' in source.name.lower():continue
    with zipfile.ZipFile(source) as z:
        meta=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
        version=next(x[9:] for x in z.read(meta).decode().splitlines() if x.startswith('Version: '))
    parts=source.name.split('-');parts[1]=version
    target=normalized/'-'.join(parts)
    if not target.exists():target.symlink_to(source)
    packages.append(str(target))
with (WORK/'installation.log').open('w') as log:
    completed=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
                              '--force-reinstall',*packages],stdout=log,stderr=subprocess.STDOUT)
if completed.returncode:
    print((WORK/'installation.log').read_text()[-8000:])
    completed.check_returncode()
print('Official patched wheels installed; helper scripts use fresh Python processes.')
'''
    evaluate = '''def run_script(script, arguments, phase, log_name):
    stage=WORK/phase;stage.mkdir(parents=True,exist_ok=True)
    for name in ('home','tmp','cache'):(stage/name).mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(HOME=str(stage/'home'),TMPDIR=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'cache'),
        HF_HOME=str(stage/'cache/hf'),TORCH_HOME=str(stage/'cache/torch'),
        TORCH_EXTENSIONS_DIR=str(stage/'cache/torch-extensions'),TRITON_CACHE_DIR=str(stage/'cache/triton'),
        VLLM_CACHE_ROOT=str(stage/'cache/vllm'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',LITELLM_LOCAL_MODEL_COST_MAP='True',
        OTEL_SDK_DISABLED='true',TRANSFORMERS_NO_TF='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_NO_USAGE_STATS='1',VLLM_ENGINE_READY_TIMEOUT_S='1200',WANDB_MODE='disabled',
        VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    path=stage/log_name
    print('Starting', phase, log_name, flush=True)
    with path.open('w') as log:
        completed=subprocess.run([sys.executable,str(WORK/'scripts'/script),*map(str,arguments)],
                                 env=env,stdout=log,stderr=subprocess.STDOUT)
    print(path.read_text(errors='replace')[-8000:],flush=True)
    completed.check_returncode()

hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True)
(WORK/'hardware.csv').write_text(hardware);print(hardware)
run_script('validate_agent.py',['--agent-dir',AGENT,'--output',WORK/'compile/validation.json'],'compile','compile.log')
common=['--task-id',PAYLOAD['task_id'],'--data-dir',DATA,'--agent-dir',AGENT]
for source in ['empty','gold']:
    run_script('evaluate_agent.py',['verify',*common,'--tasks',DATA/'tasks.jsonl','--patch-source',source,
                                   '--results',WORK/'verify'/source],'verify',source+'.log')
run_script('summarize_eval_smoke.py',['--run-dir',WORK,'--controls-only'],'checks','controls.log')
run_script('evaluate_agent.py',['infer',*common,'--tasks',WORK/'inputs/tasks.jsonl',
                               '--model',MODEL,'--tensor-parallel-size','4','--gpu-memory-utilization','0.90',
                               '--results',WORK/'infer/results'],'infer','inference.log')
run_script('evaluate_agent.py',['verify',*common,'--tasks',DATA/'tasks.jsonl','--patch-source','agent',
                               '--prediction',WORK/'infer/prediction.json','--results',WORK/'verify/agent'],
                               'verify','agent.log')
run_script('summarize_eval_smoke.py',['--run-dir',WORK],'checks','summary.log')
Path('/kaggle/working/submission.zip').write_bytes(base64.b64decode(PAYLOAD['bundle']))
print('One-task smoke complete; this notebook has NOT submitted a competition entry.')
'''
    nb = {'nbformat': 4, 'nbformat_minor': 4,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'cells': [cell('markdown', '# Official L4 × 4 runtime smoke\nPrivate, offline, one public task. Not a competition submission or benchmark.\nThe official subprocess backend is used; Kaggle inputs remain mounted by the platform.'),
                    cell('code', 'PAYLOAD_B64 = '+repr(encoded)+'\n'+setup),
                    cell('code', install), cell('code', evaluate)]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'runtime.ipynb').write_text(json.dumps(nb, indent=2)+'\n')
    metadata = {'id': args.owner+'/gemma4-official-runtime-smoke', 'title': 'Gemma4 Official Runtime Smoke',
                'code_file': 'runtime.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'machine_shape': 'NvidiaL4', 'docker_image': official['dockerImage'],
                'docker_image_pinning_type': 'original',
                'competition_sources': ['gemma-4-developer-agent'],
                'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'], 'kernel_sources': [],
                'model_sources': ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2']}
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for c in nb['cells']:
        if c['cell_type'] == 'code':compile(c['source'], '<notebook-cell>', 'exec')
    print('Prepared', metadata['id'], 'task', run['task_id'], 'bundle', run['agent_archive_sha256'])


if __name__ == '__main__':
    main()
