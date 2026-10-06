"""Build a reproducible submission ZIP; official compilation is a separate gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ALLOWED = {'.yaml', '.yml', '.md', '.txt', '.py', '.json', '.safetensors'}
MAX_BYTES = 3 * 1024**3


def collect_files(agent_dir):
    root = Path(agent_dir).resolve(strict=True)
    if not (root / 'agent.yaml').is_file():
        raise ValueError('agent.yaml must be at the submission root')
    files = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink not permitted: {p.relative_to(root)}')
        if p.is_dir():
            continue
        if not p.is_file() or p.suffix.lower() not in ALLOWED:
            raise ValueError(f'Unsupported member: {p.relative_to(root)}')
        if any(part.startswith('.') for part in p.relative_to(root).parts):
            raise ValueError('Hidden files/directories do not belong in an agent bundle')
        files.append(p)
    if len(files) > 10000 or sum(p.stat().st_size for p in files) >= MAX_BYTES:
        raise ValueError('Submission exceeds file count or unpacked size limit')
    return root, files


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(agent_dir, output):
    root, files = collect_files(agent_dir)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Write the archive outside the source agent directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix('.zip.partial')
    records = []
    with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open('rb') as source, archive.open(info, 'w') as target:
                for chunk in iter(lambda: source.read(1024**2), b''):
                    target.write(chunk)
            records.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    with zipfile.ZipFile(pending) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failed: {bad}')
        assert archive.namelist() == [x['path'] for x in records]
    pending.replace(output)
    result = {'built_at': datetime.now(timezone.utc).isoformat(), 'archive': str(output),
              'sha256': sha256(output), 'bytes': output.stat().st_size,
              'unpacked_bytes': sum(x['bytes'] for x in records), 'files': records,
              'validation': 'Local packaging checks and ZIP CRC only; official compiler and runtime are separate gates'}
    output.with_suffix('.manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.agent_dir, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))
