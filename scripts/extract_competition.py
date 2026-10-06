"""Extract the competition archive and verify every file against its API manifest."""

import json
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "resource/downloads/gemma-4-developer-agent.zip"
DESTINATION = ROOT / "data/raw/gemma-4-developer-agent"
MANIFEST = ROOT / "resource/manifests/competition-files.json"


def main():
    manifest = json.loads(MANIFEST.read_text())
    expected = {item["name"]: item["bytes"] for item in manifest["files"]}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        actual = {item.filename: item.file_size for item in files}
        if len(actual) != len(files) or actual != expected:
            raise RuntimeError("Archive names or sizes differ from the competition API manifest")
        for index, item in enumerate(files, 1):
            parts = PurePosixPath(item.filename)
            if parts.is_absolute() or ".." in parts.parts or "\\" in item.filename:
                raise RuntimeError("Unsafe archive member path")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError("Unexpected symlink in competition archive")
            target = (DESTINATION / item.filename).resolve()
            if not target.is_relative_to(DESTINATION.resolve()):
                raise RuntimeError("Archive member escapes the destination")
            # Reading to EOF during extraction also verifies each member's ZIP CRC.
            archive.extract(item, DESTINATION)
            if target.stat().st_size != expected[item.filename]:
                raise RuntimeError(f"Extracted size mismatch: {item.filename}")
            if index % 50 == 0 or index == len(files):
                print(f"Extracted and verified {index}/{len(files)} files", flush=True)
    result = {
        "competition": manifest["competition"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(expected),
        "uncompressed_bytes": sum(expected.values()),
        "archive_bytes": ARCHIVE.stat().st_size,
        "verification": "API manifest names and sizes; ZIP CRC checked during extraction",
    }
    (ROOT / "resource/manifests/download-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
