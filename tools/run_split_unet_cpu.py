"""Generate deterministic inputs and compute a CPU reference for split U-Net."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-parts", type=int, default=1000)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    rng = np.random.default_rng(42)
    values = {}
    for value in manifest["inputs"]:
        info = manifest["tensors"][str(value)]
        array = rng.normal(size=info["shape"]).astype(np.float32)
        values[value] = array
        array.tofile(args.output_dir / f"input_t{value}.bin")
    if args.inputs_only:
        return
    parts = manifest["partitions"][:args.max_parts]
    measurements = []
    for number, part in enumerate(parts):
        started = time.monotonic()
        interpreter = Interpreter(model_path=str(args.model_dir / part["file"]), num_threads=4)
        interpreter.allocate_tensors()
        for info in interpreter.get_input_details():
            value = int(info["name"][1:])
            interpreter.set_tensor(info["index"], values[value])
        run_started = time.monotonic()
        interpreter.invoke()
        invoke_ms = (time.monotonic() - run_started) * 1000
        for info in interpreter.get_output_details():
            value = int(info["name"][1:])
            data = interpreter.get_tensor(info["index"])
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite CPU output {value}")
            values[value] = data
        del interpreter
        gc.collect()
        result = {"partition": number, "invoke_ms": invoke_ms,
                  "total_ms": (time.monotonic() - started) * 1000}
        measurements.append(result)
        print(json.dumps(result), flush=True)
        needed = {v for future in parts[number + 1:] for v in future["inputs"]}
        needed.update(parts[-1]["outputs"])
        values = {v: data for v, data in values.items() if v in needed}
    for value in parts[-1]["outputs"]:
        values[value].tofile(args.output_dir / f"output_t{value}.bin")
    (args.output_dir / "measurements.json").write_text(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
