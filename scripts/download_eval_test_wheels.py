"""Stage the pinned extra public-test dependencies for the isolated local runner."""
import argparse
import json
from pathlib import Path
import urllib.request

from build_submission import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('resource/manifests/local-eval-test-wheels.json'))
    parser.add_argument('--output', type=Path, default=Path('resource/downloads/eval-test-wheels-v3'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.manifest.read_text())['files']
    expected = {r['filename'] for r in records}
    if {p.name for p in args.output.glob('*.whl')} - expected:
        raise ValueError('Unexpected wheels in target directory; use a clean versioned directory')
    for record in records:
        target = args.output / record['filename']
        if target.exists():
            if sha256(target) != record['sha256']:
                raise ValueError('Existing wheel does not match source hash: ' + target.name)
            continue
        pending = target.with_suffix('.partial')
        with urllib.request.urlopen(record['url'], timeout=60) as response, pending.open('wb') as output:
            while chunk := response.read(1024**2):
                output.write(chunk)
        if pending.stat().st_size != record['bytes'] or sha256(pending) != record['sha256']:
            raise ValueError('Downloaded wheel failed size/hash verification: ' + target.name)
        pending.replace(target)
    print(f'Verified {len(records)} pinned wheels in {args.output}')


if __name__ == '__main__':
    main()
