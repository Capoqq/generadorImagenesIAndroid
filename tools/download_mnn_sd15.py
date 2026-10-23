"""Download the upstream MNN SD1.5 conversion into a local ignored directory.

The original Stable Diffusion 1.5 model license must be reviewed before redistribution.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DEFAULT_REPO = "taobao-mnn/stable-diffusion-v1-5-mnn-general"
FILES = (
    "text_encoder.mnn", "text_encoder.mnn.weight",
    "unet.mnn", "unet.mnn.weight",
    "vae_decoder.mnn", "vae_decoder.mnn.weight",
    "vocab.json", "merges.txt", "alphas.txt",
)


def fetch(destination: Path, repo: str = DEFAULT_REPO) -> None:
    if repo not in (
        "taobao-mnn/stable-diffusion-v1-5-mnn-general",
        "taobao-mnn/stable-diffusion-v1-5-mnn-opencl",
        "Zevas1993/nexus-sd15-inpaint-mnn-int8",
    ):
        raise ValueError("Unsupported model repository")
    destination.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        metadata = {item["path"]: item for item in json.load(response)}
    for name in FILES:
        expected = metadata[name]["size"]
        expected_hash = metadata[name].get("lfs", {}).get("oid")
        target = destination / name
        if target.is_file() and target.stat().st_size == expected:
            if expected_hash:
                digest = hashlib.sha256()
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise RuntimeError(f"SHA-256 mismatch: {name}")
            print(f"verified {name} ({expected} bytes)", flush=True)
            continue
        partial = destination / f"{name}.partial"
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{name}")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        print(f"download {name}: {offset}/{expected}", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and response.status != 206:
                offset = 0
            with partial.open("ab" if offset else "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if partial.stat().st_size != expected:
            raise RuntimeError(f"Wrong size for {name}: {partial.stat().st_size} != {expected}")
        if expected_hash:
            digest = hashlib.sha256()
            with partial.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch: {name}")
        partial.replace(target)
        print(f"saved {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()
    fetch(args.destination, args.repo)
