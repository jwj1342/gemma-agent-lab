"""Freeze pilot training source and configuration before submitting a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/first-lora.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configuration = args.config.resolve().relative_to(root)
    target = args.run_dir / "inputs/training"
    if target.exists():
        raise ValueError("Training snapshot already exists")
    files = [Path("src/sft_data.py"), Path("scripts/train_first_lora.py"),
             Path("scripts/prepare_terminal_sft.py"), configuration]
    hashes = {}
    for relative in files:
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, dest)
        hashes[str(relative)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    manifest = {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_sha256": hashes, "scope": "Frozen training source; weights and data are hashed by the allocated runner"}
    manifest["configuration"] = str(configuration)
    (args.run_dir / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
