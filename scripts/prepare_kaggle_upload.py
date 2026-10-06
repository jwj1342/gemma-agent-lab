"""Generate a private CPU notebook that emits an already reviewed submission ZIP."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', default='gemma4-pipeline-baseline')
    parser.add_argument('--title', default='Gemma4 Pipeline Baseline')
    args = parser.parse_args()
    # Embedded source is convenient for the small YAML baseline. Large adapters
    # should use a private versioned Kaggle model/dataset input instead.
    if args.submission.stat().st_size > 5 * 1024**2:
        raise ValueError('Use a private versioned input for bundles larger than 5 MiB')
    payload = args.submission.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode()
    source = f'''import base64, hashlib, io, json, stat, zipfile
from pathlib import Path, PurePosixPath

EXPECTED_SHA256 = {sha!r}
payload = base64.b64decode({encoded!r}, validate=True)
assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    members = archive.infolist()
    assert "agent.yaml" in archive.namelist()
    assert len(archive.namelist()) == len(set(archive.namelist()))
    assert sum(m.file_size for m in members) < 3 * 1024**3
    for member in members:
        name = PurePosixPath(member.filename)
        assert not name.is_absolute() and ".." not in name.parts
        assert not stat.S_ISLNK(member.external_attr >> 16)
        assert name.suffix.lower() in {{'.yaml', '.yml', '.md', '.txt', '.py', '.json', '.safetensors'}}
    assert archive.testzip() is None
output = Path('/kaggle/working')
output.mkdir(parents=True, exist_ok=True)
(output / 'submission.zip').write_bytes(payload)
(output / 'bundle-manifest.json').write_text(json.dumps({{
    'sha256': EXPECTED_SHA256, 'bytes': len(payload),
    'validation': 'Reproducible upload artifact only; this notebook does not perform model evaluation'
}}, indent=2))
print('submission.zip', len(payload), 'bytes', EXPECTED_SHA256)
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [
                    {'id':'description','cell_type':'markdown','metadata':{},'source':[
                        '# Gemma 4 pipeline baseline\n',
                        'Private CPU packaging notebook. The reviewed YAML baseline has no LoRA.\n',
                        'This preserves the locally built ZIP byte for byte; inference runs in competition scoring.\n']},
                    {'id':'package','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                     'source':source.splitlines(keepends=True)}]}
    metadata = {'id': f'{args.owner}/{args.slug}', 'title': args.title, 'code_file':'build.ipynb',
                'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,
                'enable_internet':False,'competition_sources':['gemma-4-developer-agent'],
                'dataset_sources':[],'kernel_sources':[],'model_sources':[]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'build.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    (args.output_dir/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'kernel':metadata['id'],'sha256':sha,'output_dir':str(args.output_dir)},indent=2))


if __name__ == '__main__':
    main()
