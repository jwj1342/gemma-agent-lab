"""Generate a private, offline L4x4 notebook for an official-harness smoke task or a frozen task suite."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from build_submission import build, collect_files
from prepare_eval_task import PUBLIC_FIELDS
from verify_eval_run import verify_inputs
from wheelhouse_runtime import load_registry


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path, help='Existing frozen evaluation run')
    source.add_argument('--agent-dir', type=Path, help='Agent configuration; no local competition data needed')
    parser.add_argument('--task-id', help='Public task to select from Kaggle inputs, with --agent-dir')
    parser.add_argument('--slug', default='gemma4-official-runtime-smoke')
    parser.add_argument('--title', default='Gemma4 Official Runtime Smoke')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheelhouse-mode', choices=['verified', 'record'], default='verified',
                        help='Match a registered snapshot, or record unreviewed inputs for exploration')
    parser.add_argument('--wheelhouse-snapshot', default='auto',
                        help='Registered snapshot ID; auto accepts any registered matching snapshot')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    snapshots = load_registry(root / 'resource/manifests/kaggle-wheelhouse-snapshots.json')
    if args.wheelhouse_snapshot != 'auto':
        if args.wheelhouse_mode == 'record':
            parser.error('--wheelhouse-snapshot requires --wheelhouse-mode verified')
        if args.wheelhouse_snapshot not in {s['id'] for s in snapshots}:
            parser.error('Unknown wheelhouse snapshot: ' + args.wheelhouse_snapshot)
    if args.agent_dir and not args.task_id:
        parser.error('--agent-dir requires --task-id')
    if args.run_dir and args.task_id:
        parser.error('--run-dir already records its task; omit --task-id')
    if args.output.exists():
        raise ValueError('Output directory exists; choose a new path to preserve previous notebook evidence')
    try:
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        git_revision = revision.stdout.strip() if revision.returncode == 0 else None
    except FileNotFoundError:
        git_revision = None  # Source archives can be used without Git installed.
    if args.run_dir:
        bundle_path = args.run_dir / 'inputs/submission.zip'
        if bundle_path.stat().st_size > 5 * 1024**2:
            raise ValueError('Inline runtime notebook only supports small baseline bundles')
        run = verify_inputs(args.run_dir)
        bundle = bundle_path.read_bytes()
        archive_manifest = json.loads(bundle_path.with_suffix('.manifest.json').read_text())
        public_task = (args.run_dir / 'inputs/tasks.jsonl').read_text()
    else:
        _, files = collect_files(args.agent_dir)
        if sum(path.stat().st_size for path in files) > 5 * 1024**2:
            raise ValueError('Inline notebooks support small configurations; use versioned inputs for adapters')
        with tempfile.TemporaryDirectory(prefix='gemma-notebook-') as temporary:
            bundle_path = Path(temporary) / 'submission.zip'
            archive_manifest = build(args.agent_dir, bundle_path)
            bundle = bundle_path.read_bytes()
        run = {'task_id': args.task_id, 'agent_archive_sha256': archive_manifest['sha256'],
               'git_revision': git_revision}
        public_task = None
    if len(bundle) > 5 * 1024**2:
        raise ValueError('Inline runtime notebook only supports small baseline bundles')
    task_ids = run.get('task_ids') or [run['task_id']]
    if hashlib.sha256(bundle).hexdigest() != run['agent_archive_sha256']:
        raise ValueError('Run bundle hash changed')
    official = json.loads((root / 'resource/notebooks/official/kernel-metadata.json').read_text())
    payload = {'bundle': base64.b64encode(bundle).decode(), 'bundle_sha256': run['agent_archive_sha256'],
               'archive_manifest': archive_manifest,
               'public_task': public_task, 'public_fields': sorted(PUBLIC_FIELDS), 'task_ids': task_ids,
               'git_revision': git_revision,
               'source_run_git_revision': run.get('git_revision') if args.run_dir else None,
               'scripts': {name: (root / 'scripts' / name).read_text() for name in
                           ['validate_agent.py', 'evaluate_agent.py', 'summarize_eval_smoke.py',
                            'verify_eval_run.py', 'build_submission.py', 'prepare_eval_task.py',
                            'wheelhouse_runtime.py']},
               'wheelhouse_snapshots': snapshots, 'wheelhouse_mode': args.wheelhouse_mode,
               'wheelhouse_snapshot': args.wheelhouse_snapshot}
    payload['script_sha256'] = {name: hashlib.sha256(source.encode()).hexdigest()
                                for name, source in payload['scripts'].items()}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    setup = '''import base64, hashlib, json, os, subprocess, sys, zipfile, io
from pathlib import Path
PAYLOAD = json.loads(base64.b64decode(PAYLOAD_B64))
WORK = Path('/kaggle/working/runtime-smoke')
DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WHEELS = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
assert DATA.is_dir() and WHEELS.is_dir() and MODEL.is_dir(), 'Attach the competition, official wheelhouse, and model v2 inputs.'
IDS = PAYLOAD['task_ids']
with (DATA/'tasks.jsonl').open() as stream:
    mounted = {task['instance_id']: task for line in stream if line.strip() for task in [json.loads(line)]}
assert all(task_id in mounted for task_id in IDS), 'Expected exactly one matching public task for each selected ID in Kaggle inputs.'
public_tasks = [{key: value for key, value in mounted[task_id].items() if key in PAYLOAD['public_fields']}
                for task_id in IDS]
if PAYLOAD['public_task'] is not None:
    frozen = [json.loads(line) for line in PAYLOAD['public_task'].splitlines() if line.strip()]
    assert frozen == public_tasks, 'Frozen task differs from mounted competition data.'
task_text = ''.join(json.dumps(task) + '\\n' for task in public_tasks)
del mounted
assert not WORK.exists(), 'Runtime output already exists. Export it and start a fresh session before rerunning.'
wheelhouse_module = {}
exec(compile(PAYLOAD['scripts']['wheelhouse_runtime.py'], '<wheelhouse-runtime>', 'exec'), wheelhouse_module)
wheelhouse_record = wheelhouse_module['capture'](
    WHEELS, PAYLOAD['wheelhouse_snapshots'], PAYLOAD['wheelhouse_mode'], PAYLOAD['wheelhouse_snapshot'])
ACTIVE_WHEELS = dict(wheelhouse_record['wheels'])
WORK.mkdir(parents=True)
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
(WORK/'inputs/submission.zip').write_bytes(raw)
archive_manifest = dict(PAYLOAD['archive_manifest'], archive=str(WORK/'inputs/submission.zip'))
(WORK/'inputs/submission.manifest.json').write_text(json.dumps(archive_manifest, indent=2) + '\\n')
(WORK/'inputs/tasks.jsonl').write_text(task_text)
(WORK/'run-manifest.json').write_text(json.dumps({
    **({'task_id': IDS[0]} if len(IDS) == 1 else {'task_ids': IDS}), 'git_revision': PAYLOAD['git_revision'],
    'source_run_git_revision': PAYLOAD['source_run_git_revision'],
    'script_sha256': PAYLOAD['script_sha256'],
    'agent_archive_sha256': PAYLOAD['bundle_sha256'],
    'public_task_sha256': hashlib.sha256(task_text.encode()).hexdigest(),
    'public_fields': sorted(set().union(*public_tasks)),
    'model_handle': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
    'tensor_parallel_size': 4, 'gpu_memory_utilization': 0.90,
    'wheelhouse': wheelhouse_record,
    'status': 'inputs prepared; see smoke-summary.json for executed results'
}, indent=2) + '\\n')
for name, source in PAYLOAD['scripts'].items():
    (WORK/'scripts'/name).write_text(source)
(WORK/'inputs/wheelhouse.json').write_text(json.dumps(wheelhouse_record, indent=2) + '\\n')
def verify_wheels():
    wheelhouse_module['recheck'](WHEELS, ACTIVE_WHEELS)
print('Wheelhouse:', wheelhouse_record['status'], wheelhouse_record['matched_snapshot'])
print('Versions:', wheelhouse_record['versions'])
print('Input hashes captured. Installation will recheck the same bytes.')
'''
    install = '''# Same installation scope as the reviewed official starter, in this disposable notebook.
import glob
verify_wheels()
for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pattern):
        Path(pth).unlink(missing_ok=True)
normalized = Path('/tmp/gemma4-official-wheels')
normalized.mkdir(exist_ok=True)
packages=[]
for name in sorted(ACTIVE_WHEELS):
    source = WHEELS/name
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
if len(IDS) == 1:
    run_script('verify_eval_run.py',['--run-dir',WORK],'checks','inputs.log')
run_script('validate_agent.py',['--agent-dir',AGENT,'--output',WORK/'compile/validation.json'],'compile','compile.log')
if len(IDS) == 1:
    common=['--task-id',IDS[0],'--data-dir',DATA,'--agent-dir',AGENT]
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
else:
    # The setup cell already checked the bundle hash and every frozen task against the mounted inputs.
    selection=['--task-ids',*IDS,'--data-dir',DATA,'--agent-dir',AGENT]
    for source in ['empty','gold']:
        run_script('evaluate_agent.py',['verify',*selection,'--tasks',DATA/'tasks.jsonl','--patch-source',source,
                                       '--results',WORK/'verify'/source],'verify',source+'.log')
    run_script('evaluate_agent.py',['infer',*selection,'--tasks',WORK/'inputs/tasks.jsonl',
                                   '--model',MODEL,'--tensor-parallel-size','4','--gpu-memory-utilization','0.90',
                                   '--results',WORK/'infer/results'],'infer','inference.log')
    run_script('evaluate_agent.py',['verify',*selection,'--tasks',DATA/'tasks.jsonl','--patch-source','agent',
                                   '--prediction',WORK/'infer/prediction.json','--results',WORK/'verify/agent'],
                                   'verify','agent.log')
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    rows=[]
    for task_id in IDS:
        empty,gold,agent=(read(WORK/'verify'/phase/task_id/'result.json') for phase in ('empty','gold','agent'))
        inference=read(WORK/'infer'/task_id/'results/result.json')
        valid=empty.get('resolved') is False and gold.get('resolved') is True
        rows.append({'task_id':task_id,'controls_valid':valid,'resolved':bool(agent.get('resolved')) if valid else None,
                     'agent_seconds':inference.get('agent_seconds'),'tool_calls':inference.get('tool_calls'),
                     'error':inference.get('error')})
    valid=[row for row in rows if row['controls_valid']]
    summary={'task_count':len(IDS),'valid_controls':len(valid),'resolved':sum(row['resolved'] for row in valid),
             'tasks':rows,'scope':'Private L4x4 run of public tasks; not a competition score'}
    (WORK/'cohort-summary.json').write_text(json.dumps(summary,indent=2)+'\\n')
    print(json.dumps({key:summary[key] for key in ('task_count','valid_controls','resolved')}))
Path('/kaggle/working/submission.zip').write_bytes(base64.b64decode(PAYLOAD['bundle']))
print('Run complete; this notebook has NOT submitted a competition entry.')
'''
    nb = {'nbformat': 4, 'nbformat_minor': 4,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'cells': [cell('markdown', '# Official L4 × 4 runtime run\nPrivate, offline, '+(str(len(task_ids))+' public tasks' if len(task_ids) > 1 else 'one public task')+'. Not a competition submission or benchmark.\nThe official subprocess backend is used; Kaggle inputs remain mounted by the platform.\n\nWheelhouse mode: **'+args.wheelhouse_mode+'**; snapshot: **'+args.wheelhouse_snapshot+'**. Record mode permits unreviewed versions and fingerprints this run; it does not establish compatibility.'),
                    cell('code', 'PAYLOAD_B64 = '+repr(encoded)+'\n'+setup),
                    cell('code', install), cell('code', evaluate)]}
    args.output.mkdir(parents=True)
    (args.output / 'runtime.ipynb').write_text(json.dumps(nb, indent=2)+'\n')
    metadata = {'id': args.owner+'/'+args.slug, 'title': args.title,
                'code_file': 'runtime.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'machine_shape': 'NvidiaL4', 'docker_image': official['dockerImage'],
                'docker_image_pinning_type': 'original',
                'competition_sources': ['gemma-4-developer-agent'],
                'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'], 'kernel_sources': [],
                'model_sources': ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2']}
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.output / 'preparation.json').write_text(json.dumps({
        'task_ids': task_ids, 'agent_archive_sha256': run['agent_archive_sha256'],
        'git_revision': git_revision, 'source_run_git_revision': payload['source_run_git_revision'],
        'script_sha256': payload['script_sha256'], 'kernel': metadata['id'],
        'wheelhouse_mode': args.wheelhouse_mode, 'wheelhouse_snapshot': args.wheelhouse_snapshot,
        'task_source': 'frozen run checked against Kaggle inputs' if args.run_dir else 'Kaggle inputs at runtime',
        'status': 'notebook generated locally; no GPU execution or upload'
    }, indent=2) + '\n')
    for c in nb['cells']:
        if c['cell_type'] == 'code':compile(c['source'], '<notebook-cell>', 'exec')
    print('Prepared', metadata['id'], 'tasks', ','.join(task_ids), 'bundle', run['agent_archive_sha256'])


if __name__ == '__main__':
    main()
