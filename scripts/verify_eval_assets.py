"""Hash downloaded evaluation assets and compare official size manifests; run on datamover."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if '-dm' not in socket.gethostname():
        raise RuntimeError('Read large model weights on a datamover node')
    root = Path(__file__).resolve().parents[1]
    wheels = root / 'resource/downloads/eval-wheelhouse'
    expected = json.loads((root / 'resource/manifests/wheelhouse-files.json').read_text())['datasetFiles']
    wheel_records = []
    for item in expected:
        path = wheels / item['name']
        if path.stat().st_size != item['totalBytes']:
            raise ValueError(f'Size mismatch: {path.name}')
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f'Wheel CRC error: {path.name}: {bad}')
        wheel_records.append({'path': item['name'], 'bytes': path.stat().st_size, 'sha256': digest(path)})
    model_dir = root / 'model/base/gemma-4-31b-it-qat-w4a16-ct-v2'
    files = sorted(p for p in model_dir.iterdir() if p.is_file())
    model_expected = json.loads((root / 'resource/models/competition-instance.json').read_text())
    total = sum(p.stat().st_size for p in files)
    if total != model_expected['totalUncompressedBytes']:
        raise ValueError('Downloaded model total size does not match official metadata')
    model_records = []
    for path in files:
        model_records.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        print('Verified', path.name, flush=True)
    safetensors = model_dir / 'model.safetensors'
    with safetensors.open('rb') as stream:
        header_bytes = int.from_bytes(stream.read(8), 'little')
        if not 0 < header_bytes < 100 * 1024**2:
            raise ValueError('Unexpected safetensors header length')
        header = json.loads(stream.read(header_bytes))
    payload_size = safetensors.stat().st_size - 8 - header_bytes
    for key, value in header.items():
        if key == '__metadata__':
            continue
        start, end = value['data_offsets']
        if not 0 <= start <= end <= payload_size:
            raise ValueError(f'Invalid tensor offsets: {key}')
    result = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'model_handle': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
              'model_bytes': total, 'model_files': model_records,
              'model_tensors': len(header) - ('__metadata__' in header),
              'wheelhouse_files': wheel_records,
              'scope': 'Size comparison, SHA256 fingerprints, wheel ZIP CRC, safetensors offset bounds. No model execution.'}
    target = root / 'resource/manifests/eval-assets-verification.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('Model files', len(files), 'official wheelhouse files', len(expected), flush=True)


if __name__ == '__main__':
    main()
