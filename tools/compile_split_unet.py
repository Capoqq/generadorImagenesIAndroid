"""Compile partitions independently so peak compiler memory stays bounded."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source / "manifest.json").read_text())
    for part in manifest["partitions"]:
        source = args.source / part["file"]
        artifact = args.output / f"{source.stem}_MediaTek_MT6878_apply_plugin.tflite"
        if not artifact.exists() or artifact.stat().st_size == 0:
            subprocess.run([sys.executable, str(Path(__file__).with_name("compile_mediatek_aot.py")),
                            str(source), str(args.output)], check=True)
        if not artifact.exists() or artifact.stat().st_size == 0:
            raise RuntimeError(f"No compiled artifact for {source}")
        part["source_file"] = part["file"]
        part["file"] = artifact.name
        part["bytes"] = artifact.stat().st_size
        # Source hash must not be represented as the compiled artifact's hash.
        part["source_sha256"] = part.pop("sha256")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
