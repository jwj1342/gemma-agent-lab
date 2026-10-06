"""Run inside the pinned vLLM container on a networked preparation node."""
import argparse
from importlib.metadata import version, distribution, PackageNotFoundError
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


# The competition starter installs its wheelhouse with --no-deps. These exact
# contradictions are present in that wheelhouse (plus one inherited OS package).
# Do not suppress any new pip-check failure or report this as a clean pip check.
KNOWN_METADATA_CONFLICTS = {
    'opentelemetry-exporter-otlp 1.44.0 has requirement opentelemetry-exporter-otlp-proto-grpc==1.44.0, but you have opentelemetry-exporter-otlp-proto-grpc 1.41.0.':
        'Official exporter metapackage is 1.44; google-adk caps the SDK at 1.41.1. Runtime tracing is disabled.',
    'opentelemetry-exporter-otlp 1.44.0 has requirement opentelemetry-exporter-otlp-proto-http==1.44.0, but you have opentelemetry-exporter-otlp-proto-http 1.41.0.':
        'Keep the base image OTLP runtime family at 1.41.0, matching google-adk.',
    'prometheus-fastapi-instrumentator 8.1.0 has requirement starlette<2.0.0,>=1.0.0, but you have starlette 0.52.1.':
        'Both packages come from the official wheelhouse/base image; google-adk requires starlette<1. Runtime validation required.',
    'xgrammar 0.2.5.post1 has requirement transformers<5,>=4.38.0, but you have transformers 5.13.1.':
        'Both exact versions are supplied together in the official competition wheelhouse. Runtime validation required.',
    'pygobject 3.42.1 requires pycairo, which is not installed.':
        'Inherited Ubuntu desktop package; not used by the headless competition harness.',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheelhouse',type=Path,required=True)
    p.add_argument('--normalized-wheelhouse',type=Path,required=True)
    p.add_argument('--test-wheelhouse',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    args=p.parse_args()
    if sys.version_info[:2] != (3,12):
        raise RuntimeError('Official binary wheels require the verified Python 3.12 container')
    args.normalized_wheelhouse.mkdir(parents=True,exist_ok=True)
    wheels=[]
    to_install=[]
    constraints=[]
    for source in sorted(args.wheelhouse.glob('*.whl')):
        if 'cutlass' in source.name.lower():
            continue
        with zipfile.ZipFile(source) as archive:
            metadata=next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            lines=archive.read(metadata).decode().splitlines()
            package_version=next(x[9:] for x in lines if x.startswith('Version: '))
            package_name=next(x[6:] for x in lines if x.startswith('Name: '))
            constraints.append(f'{package_name}=={package_version}')
        # Kaggle can strip '+' from CUDA wheel filenames. Use embedded metadata.
        fields=source.name.split('-')
        fields[1]=package_version
        target=args.normalized_wheelhouse/'-'.join(fields)
        if not target.exists():
            target.symlink_to(source.resolve())
        wheels.append(str(target))
        # Resume preparation without reinstalling hundreds of MB when pip's
        # installation provenance already identifies this exact official wheel.
        with source.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        try:
            installed=distribution(package_name)
            provenance=json.loads(installed.read_text('direct_url.json') or '{}')
            same=(installed.version==package_version and
                  provenance.get('archive_info',{}).get('hashes',{}).get('sha256')==digest)
        except (PackageNotFoundError,json.JSONDecodeError):
            same=False
        if not same:to_install.append(str(target))
    # A Kaggle wheel can contain patches despite matching a preinstalled version.
    # Install its exact bytes into this venv before resolving missing dependencies.
    if to_install:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-compile',
                        '--force-reinstall',*to_install],check=True)
    print(f'Official wheels: {len(wheels)}; exact installation provenance reused: {len(wheels)-len(to_install)}',flush=True)
    # Resolving all 41 wheels together is impossible: their declared requirements
    # conflict. Resolve the missing ADK/runtime dependencies while preserving the
    # exact official versions, then record narrowly enumerated metadata conflicts.
    args.lock.parent.mkdir(parents=True,exist_ok=True)
    constraints_path=args.lock.with_suffix('.official-constraints.txt')
    constraints_path.write_text('\n'.join(sorted(constraints))+ '\n')
    # The official subprocess backend inherits test tools and skips Docker's
    # per-repository dependency installer. The base vLLM image has no pytest.
    test_wheel=args.test_wheelhouse/'pytest-8.3.4-py3-none-any.whl'
    if not test_wheel.is_file():
        raise FileNotFoundError(test_wheel)
    subprocess.run([sys.executable,'-m','pip','install','--only-binary=:all:',
                    '--constraint',str(constraints_path),
                    'google-adk==1.36.1','google-genai==2.11.0',
                    'docker','litellm==1.83.14','pandas','httpx2','msal==1.36.0',
                    'nvidia-cutlass-dsl==4.6.2',str(test_wheel),'pytest-timeout==2.1.0'],check=True)
    check=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
    failures=[line.strip() for line in check.stdout.splitlines()
              if line.strip() and line.strip()!='No broken requirements found.']
    unexpected=[line for line in failures if line not in KNOWN_METADATA_CONFLICTS]
    report={'checked_at':datetime.now(timezone.utc).isoformat(),
            'status':'unexpected_conflicts' if unexpected else ('known_metadata_conflicts' if failures else 'clean'),
            'pip_check_returncode':check.returncode,'pip_check_stdout':check.stdout,
            'pip_check_stderr':check.stderr,'unexpected':unexpected,
            'known_conflicts':{line:KNOWN_METADATA_CONFLICTS[line] for line in failures if line in KNOWN_METADATA_CONFLICTS},
            'runtime_verified':False}
    args.lock.with_suffix('.dependency-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if unexpected or check.returncode not in (0,1) or (check.returncode and not failures):
        raise RuntimeError('Unexpected dependency failure; see dependency audit')
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    args.lock.write_text(freeze)
    versions={n:version(n) for n in ['torch','vllm','transformers','swegemma','adk-submission','google-adk','pytest','pytest-timeout']}
    args.lock.with_suffix('.versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    print(json.dumps(versions,indent=2))


if __name__=='__main__':main()
