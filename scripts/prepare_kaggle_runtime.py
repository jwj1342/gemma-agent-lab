"""Generate a private, offline L4x4 notebook for one real official-harness smoke task."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from build_submission import build, collect_files
from prepare_eval_task import PUBLIC_FIELDS
from verify_eval_run import verify_inputs

# Test-only dependencies installed offline before the harness runs. The subprocess sandbox used
# on Kaggle never installs wheels (see resource/discussions/743973-full.json and the harness
# source), so repository test imports must exist in the notebook interpreter. Packages the image
# already provides at a task-relevant version (pytest, rich, pydantic, starlette, flask, ...) are
# deliberately excluded.
TEST_DEPENDENCY_WHEELS = (
    'asttokens-3.0.2-py3-none-any.whl',
    'executing-2.2.1-py2.py3-none-any.whl',
    'dirty_equals-0.11-py3-none-any.whl',
    'inline_snapshot-0.36.0-py3-none-any.whl',
    'pytest_httpbin-2.1.0-py3-none-any.whl',
    'httpbin-0.10.0-py2.py3-none-any.whl',
    'python_multipart-0.0.32-py3-none-any.whl',
    'ujson-6.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl',
    'orjson-3.13.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
)


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build_test_dependency_blob(deps_dir):
    """Pack the reviewed test-only wheels and return (base85 tarball, manifest record)."""
    manifest_path = deps_dir / 'test-deps-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    by_name = {item['path']: item for item in manifest['wheels']}
    install = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name in TEST_DEPENDENCY_WHEELS:
            if name not in by_name:
                raise ValueError(f'{name} missing from {deps_dir}; rebuild the dependency set')
            wheel = deps_dir / name
            payload = wheel.read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
            install.append({'name': name, 'sha256': by_name[name]['sha256'],
                            'bytes': by_name[name]['bytes']})
    blob = base64.b85encode(gzip.compress(buffer.getvalue(), mtime=0)).decode()
    record = {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'install': install}
    return blob, record


def test_dependency_cell(blob, record):
    """Install the reviewed test-only wheels offline, never upgrading image packages."""
    return cell('code', '# Install reviewed test-only dependencies; nothing else changes.\n'
                        'TEST_DEPS_B85 = ' + repr(blob) + '\n'
                        'TEST_DEPS = ' + json.dumps(record) + '\n'
                        'import base64 as _b64, gzip as _gzip, hashlib as _hash, io as _io, tarfile as _tar\n'
                        'import subprocess as _sp, sys as _sys\n'
                        "DEPOT = Path('/tmp/reviewed-test-deps'); DEPOT.mkdir(parents=True, exist_ok=True)\n"
                        'with _tar.open(fileobj=_io.BytesIO(_gzip.decompress(_b64.b85decode(TEST_DEPS_B85)))) as _a:\n'
                        '    _a.extractall(DEPOT)\n'
                        '_selected = []\n'
                        "for _item in TEST_DEPS['install']:\n"
                        '    _wheel = DEPOT / _item[\'name\']\n'
                        '    assert _hash.sha256(_wheel.read_bytes()).hexdigest() == _item[\'sha256\'], \'test dependency changed\'\n'
                        '    _selected.append(str(_wheel))\n'
                        '_install = _sp.run([_sys.executable, \'-m\', \'pip\', \'install\', \'--no-index\',\n'
                        '                    \'--no-deps\', \'--force-reinstall\', *_selected],\n'
                        '                   capture_output=True, text=True)\n'
                        'print(_install.stdout[-3000:]); print(_install.stderr[-2000:])\n'
                        '_install.check_returncode()\n'
                        'print(\'reviewed test-only dependencies installed.\')\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path, help='Existing frozen evaluation run')
    source.add_argument('--agent-dir', type=Path, help='Agent configuration; no local competition data needed')
    parser.add_argument('--task-id', help='Public task to select from Kaggle inputs, with --agent-dir')
    parser.add_argument('--slug', default='gemma4-official-runtime-smoke')
    parser.add_argument('--title', help='Display title; defaults to the slug with spaces')
    parser.add_argument('--wheel-manifest', type=Path,
                        help='Reviewed wheel inventory JSON; defaults to the October 9 snapshot')
    parser.add_argument('--test-deps-dir', type=Path,
                        help='Directory from scripts/build_test_dependency_set.py; installs the '
                             'reviewed test-only wheels offline before the harness runs')
    parser.add_argument('--adapter-dir', type=Path,
                        help='Local PEFT adapter directory (adapter_config.json plus '
                             'adapter_model.safetensors). Adapters exceed the inline bundle limit, so '
                             'this is mounted and staged into the run instead of embedded')
    parser.add_argument('--adapter-mount', default=None,
                        help='Kaggle mount path of the adapter, e.g. '
                             '/kaggle/input/gemma4-lora-adapter/main_lora; needs --adapter-dir')
    parser.add_argument('--adapter-name', default='main_lora',
                        help='Adapter directory name the agent YAML references')
    parser.add_argument('--adapter-dataset-source', default=None,
                        help='Kaggle dataset handle holding the adapter, recorded in kernel metadata')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.agent_dir and not args.task_id:
        parser.error('--agent-dir requires --task-id')
    if args.run_dir and args.task_id:
        parser.error('--run-dir already records its task; omit --task-id')
    if bool(args.adapter_dir) != bool(args.adapter_mount):
        parser.error('--adapter-dir and --adapter-mount must be given together')
    if args.adapter_mount and not args.adapter_dataset_source:
        parser.error('--adapter-mount needs --adapter-dataset-source for the kernel metadata')
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
    if hashlib.sha256(bundle).hexdigest() != run['agent_archive_sha256']:
        raise ValueError('Run bundle hash changed')
    official = json.loads((root / 'resource/notebooks/official/kernel-metadata.json').read_text())
    wheel_manifest = args.wheel_manifest or root / 'resource/manifests/eval-wheelhouse-2026-10-09.json'
    assets = json.loads(wheel_manifest.read_text(encoding='utf-8'))
    wheel_manifest_sha256 = hashlib.sha256(wheel_manifest.read_bytes()).hexdigest()
    test_deps_blob = None
    test_deps_record = None
    if args.test_deps_dir:
        test_deps_blob, test_deps_record = build_test_dependency_blob(args.test_deps_dir)
    if args.adapter_dir:
        # An adapter is far too large to embed; it is mounted and staged at run time. Fail here rather
        # than inside a paid GPU session.
        for required in ('adapter_config.json', 'adapter_model.safetensors'):
            if not (args.adapter_dir / required).is_file():
                raise ValueError(f'adapter directory is missing {required}: {args.adapter_dir}')
        adapter_ranks = json.loads((args.adapter_dir / 'adapter_config.json').read_text())
        if not isinstance(adapter_ranks.get('r'), int) or adapter_ranks['r'] <= 0:
            raise ValueError('adapter_config.json must carry a positive integer r')
        if not args.agent_dir:
            raise ValueError('--adapter-dir requires --agent-dir so the adapter can be referenced')
    payload = {'bundle': base64.b64encode(bundle).decode(), 'bundle_sha256': run['agent_archive_sha256'],
               'archive_manifest': archive_manifest,
               'adapter': ({'name': args.adapter_name,
                            'mount': args.adapter_mount,
                            'config': json.loads((args.adapter_dir / 'adapter_config.json').read_text()),
                            'config_sha256': hashlib.sha256(
                                (args.adapter_dir / 'adapter_config.json').read_bytes()).hexdigest(),
                            'weights_sha256': hashlib.sha256(
                                (args.adapter_dir / 'adapter_model.safetensors').read_bytes()).hexdigest(),
                            'weights_bytes': (args.adapter_dir / 'adapter_model.safetensors').stat().st_size}
                           if args.adapter_dir else None),
               'public_task': public_task, 'public_fields': sorted(PUBLIC_FIELDS), 'task_id': run['task_id'],
               'git_revision': git_revision,
               'source_run_git_revision': run.get('git_revision') if args.run_dir else None,
               'scripts': {name: (root / 'scripts' / name).read_text() for name in
                           ['validate_agent.py', 'evaluate_agent.py', 'summarize_eval_smoke.py',
                            'verify_eval_run.py', 'build_submission.py', 'prepare_eval_task.py']},
               'wheels': {item['path']: item['sha256'] for item in assets['wheelhouse_files']},
               'wheel_manifest_sha256': wheel_manifest_sha256}
    payload['script_sha256'] = {name: hashlib.sha256(source.encode()).hexdigest()
                                for name, source in payload['scripts'].items()}
    encoded = base64.b85encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    setup = '''import base64, gzip, hashlib, json, os, subprocess, sys, zipfile, io
from pathlib import Path
PAYLOAD = json.loads(gzip.decompress(base64.b85decode(PAYLOAD_B85)))
WORK = Path('/kaggle/working/runtime-smoke')
DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WHEELS = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
MODEL = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
if not DATA.is_dir():
    DATA = Path('/kaggle/input/gemma-4-developer-agent')
if not WHEELS.is_dir():
    WHEELS = Path('/kaggle/input/gemma-4-developer-agent-wheelhouse')
mounts = {str(path): path.is_dir() for path in (DATA, WHEELS, MODEL)}
assert all(mounts.values()), 'Required Kaggle inputs missing: ' + repr(mounts)
print('Input mounts:', mounts, flush=True)
with (DATA/'tasks.jsonl').open() as stream:
    matches = [task for line in stream if line.strip()
               for task in [json.loads(line)] if task['instance_id'] == PAYLOAD['task_id']]
assert len(matches) == 1, 'Expected exactly one matching public task in Kaggle inputs.'
public_task = {key: value for key, value in matches[0].items() if key in PAYLOAD['public_fields']}
if PAYLOAD['public_task'] is not None:
    assert json.loads(PAYLOAD['public_task']) == public_task, 'Frozen task differs from mounted competition data.'
task_text = json.dumps(public_task) + '\\n'
del matches
assert not WORK.exists(), 'Runtime output already exists. Export it and start a fresh session before rerunning.'
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
# Stage a mounted PEFT adapter into the run's agent directory. Adapters are far too large to embed
# in the notebook, so they arrive as a versioned dataset input and are copied in before validation.
ADAPTER = PAYLOAD.get('adapter')
if ADAPTER:
    import shutil
    source = Path(ADAPTER['mount'])/ADAPTER['name']
    assert source.is_dir(), f'Adapter mount missing: {source}'
    destination = AGENT/'adapters'/ADAPTER['name']
    destination.mkdir(parents=True, exist_ok=True)
    for item in ('adapter_config.json', 'adapter_model.safetensors'):
        candidate = source/item
        assert candidate.is_file(), f'Adapter mount is incomplete, missing {item}: {candidate}'
        shutil.copy(candidate, destination/item)
    actual_config = hashlib.sha256((destination/'adapter_config.json').read_bytes()).hexdigest()
    actual_weights = hashlib.sha256((destination/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert actual_config == ADAPTER['config_sha256'], 'Mounted adapter_config.json differs from the reviewed one'
    assert actual_weights == ADAPTER['weights_sha256'], 'Mounted adapter_model.safetensors differs from the reviewed one'
    print('Staged adapter', ADAPTER['name'], 'rank', ADAPTER['config'].get('r'),
          'bytes', ADAPTER['weights_bytes'], flush=True)
(WORK/'inputs/submission.zip').write_bytes(raw)
archive_manifest = dict(PAYLOAD['archive_manifest'], archive=str(WORK/'inputs/submission.zip'))
(WORK/'inputs/submission.manifest.json').write_text(json.dumps(archive_manifest, indent=2) + '\\n')
(WORK/'inputs/tasks.jsonl').write_bytes(task_text.encode('utf-8'))
(WORK/'run-manifest.json').write_text(json.dumps({
    'task_id': PAYLOAD['task_id'], 'git_revision': PAYLOAD['git_revision'],
    'source_run_git_revision': PAYLOAD['source_run_git_revision'],
    'script_sha256': PAYLOAD['script_sha256'],
    'wheel_manifest_sha256': PAYLOAD['wheel_manifest_sha256'],
    'agent_archive_sha256': PAYLOAD['bundle_sha256'],
    'public_task_sha256': hashlib.sha256(task_text.encode()).hexdigest(),
    'public_fields': sorted(public_task),
    'adapter': (dict(PAYLOAD['adapter'], mount=str(PAYLOAD['adapter']['mount']))
                if PAYLOAD.get('adapter') else None),
    'model_handle': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
    'tensor_parallel_size': 4, 'gpu_memory_utilization': 0.90,
    'input_paths': {'data': str(DATA), 'wheels': str(WHEELS), 'model': str(MODEL)},
    'status': 'inputs prepared; see smoke-summary.json for executed results'
}, indent=2) + '\\n')
for name, source in PAYLOAD['scripts'].items():
    (WORK/'scripts'/name).write_text(source)
def verify_wheels():
    actual = {path.name for path in WHEELS.glob('*.whl')}
    assert actual == set(PAYLOAD['wheels']), 'Official wheelhouse inventory changed.'
    for name, expected in PAYLOAD['wheels'].items():
        h=hashlib.sha256()
        with (WHEELS/name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024**2), b''):h.update(chunk)
        assert h.hexdigest() == expected, 'Official wheelhouse changed: '+name
verify_wheels()
print('Exact local bundle and official wheel inventory/hashes match.')
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
for name in sorted(PAYLOAD['wheels']):
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
run_script('verify_eval_run.py',['--run-dir',WORK],'checks','inputs.log')
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
                    cell('code', 'PAYLOAD_B85 = '+repr(encoded)+'\n'+setup),
                    cell('code', install)]
                    + ([test_dependency_cell(test_deps_blob, test_deps_record)] if test_deps_blob else [])
                    + [cell('code', evaluate)]}
    # Kaggle rejects notebook source over 1 MB, including inline payloads.
    if len(json.dumps(nb).encode('utf-8')) >= 1_000_000:
        raise ValueError('Notebook source exceeds Kaggle 1 MB limit; use smaller agent assets')
    args.output.mkdir(parents=True)
    (args.output / 'runtime.ipynb').write_text(json.dumps(nb, indent=2)+'\n')
    metadata = {'id': args.owner+'/'+args.slug, 'title': args.title or args.slug.replace('-', ' '),
                'code_file': 'runtime.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'machine_shape': 'NvidiaL4', 'docker_image': official['dockerImage'],
                'docker_image_pinning_type': 'original',
                'competition_sources': ['gemma-4-developer-agent'],
                'dataset_sources': (['metric/gemma-4-developer-agent-wheelhouse']
                                    + ([args.adapter_dataset_source] if args.adapter_dataset_source else [])),
                'kernel_sources': [],
                'model_sources': ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2']}
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.output / 'preparation.json').write_text(json.dumps({
        'task_id': run['task_id'], 'agent_archive_sha256': run['agent_archive_sha256'],
        'git_revision': git_revision, 'source_run_git_revision': payload['source_run_git_revision'],
        'script_sha256': payload['script_sha256'], 'kernel': metadata['id'],
        'wheel_manifest_sha256': wheel_manifest_sha256,
        'test_deps_manifest_sha256': test_deps_record['manifest_sha256'] if test_deps_record else None,
        'test_deps_installed': [item['name'] for item in test_deps_record['install']] if test_deps_record else [],
        'task_source': 'frozen run checked against Kaggle inputs' if args.run_dir else 'Kaggle inputs at runtime',
        'status': 'notebook generated locally; no GPU execution or upload'
    }, indent=2) + '\n')
    for c in nb['cells']:
        if c['cell_type'] == 'code':compile(c['source'], '<notebook-cell>', 'exec')
    print('Prepared', metadata['id'], 'task', run['task_id'], 'bundle', run['agent_archive_sha256'])


if __name__ == '__main__':
    main()
