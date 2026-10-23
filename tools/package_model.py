"""Validate a converted MNN SD1.5 directory and create an importable ZIP.

No model weights are downloaded or redistributed by this tool.
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

REQUIRED = (
    "text_encoder.mnn", "text_encoder.mnn.weight",
    "unet.mnn", "unet.mnn.weight",
    "vae_decoder.mnn", "vae_decoder.mnn.weight", "vocab.json", "merges.txt",
)
OPTIONAL = ("alphas.txt", "tokenizer.mtok")
MAX_TOTAL = 4 * 1024**3


def package(directory: Path, output: Path, model_id: str, license_name: str, source: str, backend: str = "cpu") -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", model_id):
        raise ValueError("Invalid model ID")
    if not source.startswith("https://") or not license_name.strip():
        raise ValueError("License and HTTPS source are required")
    if backend not in ("cpu", "opencl"):
        raise ValueError("Backend must be cpu or opencl")
    files = [directory / name for name in REQUIRED]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty component: {path.name}")
    files += [directory / name for name in OPTIONAL if (directory / name).is_file()]
    if sum(path.stat().st_size for path in files) > MAX_TOTAL:
        raise ValueError("Model exceeds 4 GiB import limit")
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    manifest = {
        "schemaVersion": 1,
        "id": model_id,
        "architecture": "stable-diffusion-1.5",
        "runtime": "mnn",
        "license": license_name,
        "source": source,
        "supportedBackends": [backend],
        "sha256": hashes,
    }
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        for path in files:
            archive.write(path, path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--backend", choices=("cpu", "opencl"), default="cpu")
    args = parser.parse_args()
    package(args.directory, args.output, args.id, args.license, args.source, args.backend)
