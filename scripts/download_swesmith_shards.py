"""Download and verify pinned public tool trajectories on a datamover."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    if '-dm' not in socket.gethostname():
        raise RuntimeError('Use a Trillium datamover for bulk dataset transfers')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'resource/manifests/swesmith-tool-shards.json').read_text())
    output = root / 'data/raw/swesmith'
    output.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        dest = output / Path(item['path']).name
        if not dest.exists():
            temporary = dest.with_suffix('.parquet.partial')
            url = f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/{manifest['revision']}/{item['path']}"
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
                            url, '--output', str(temporary)], check=True)
            candidate = temporary
        else:
            candidate = dest
        with candidate.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if candidate.stat().st_size != item['bytes'] or actual != item['sha256']:
            raise ValueError('Pinned dataset content mismatch: ' + dest.name)
        if candidate != dest:
            candidate.replace(dest)
        print('Verified', dest.name, flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, manifest['files']))


if __name__ == '__main__':
    main()
