"""Build the offline test-dependency wheel set that the public Kaggle image lacks.

Motivation (recorded evidence, not assumption):
  * `swegemma.harness.container_setup.install_test_dependencies` returns immediately for the
    subprocess backend, and the subprocess sandbox only uses `wheels_dir` for `/wheels` path
    translation - it never installs wheels. The sandbox venv is created with
    `system_site_packages=True` plus a `_host_env.pth` pointing at the host site-packages.
  * Therefore repository test dependencies must already exist in the notebook interpreter.
  * The public wheelhouse (41 files) contains no FastAPI/Requests test dependencies.

This script only downloads and hashes wheels for CPython 3.12 / manylinux x86_64.
It never installs anything and never modifies the competition wheelhouse.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PACKAGES = [
    # fastapi `dependency-groups.tests` direct imports missing from the public image
    "dirty-equals",
    "inline-snapshot",
    "typing-inspection",
    # requests test extras
    "pytest-httpbin",
    "pytest-mock",
    # tolerated direct imports in a few snapshots
    "ujson",
    "orjson",
    "python-multipart",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--abi", default=None,
                        help="Explicit ABI tag; defaults to cp<python-version without dots>")
    parser.add_argument("--platform", default="manylinux2014_x86_64")
    parser.add_argument("--packages", nargs="*", default=DEFAULT_PACKAGES)
    args = parser.parse_args()

    abi = args.abi or f"cp{args.python_version.replace('.', '')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in args.output_dir.glob("*.whl")}

    command = [
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:",
        "--no-cache-dir",
        "--python-version", args.python_version,
        "--platform", args.platform,
        "--implementation", "cp",
        "--abi", abi,
        "--dest", str(args.output_dir),
        *args.packages,
    ]
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    print(completed.stdout[-4000:])
    if completed.returncode:
        print(completed.stderr[-4000:], file=sys.stderr)
        return completed.returncode

    wheels = sorted(args.output_dir.glob("*.whl"))
    if not wheels:
        print("no wheels downloaded", file=sys.stderr)
        return 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Offline test dependencies for public Gemma 4 Developer Agent tasks whose snapshots "
            "import packages absent from the official public wheelhouse and the Kaggle image."
        ),
        "target": {
            "python_version": args.python_version,
            "platform": args.platform,
            "abi": abi,
        },
        "requested": args.packages,
        "wheels": [
            {
                "path": wheel.name,
                "bytes": wheel.stat().st_size,
                "sha256": sha256(wheel),
                "preexisting": wheel.name in existing,
            }
            for wheel in wheels
        ],
    }
    manifest_path = args.output_dir / "test-deps-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wheel count: {len(wheels)}")
    for item in manifest["wheels"]:
        print(f"  {item['path']}  {item['bytes']} bytes  {item['sha256'][:16]}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
