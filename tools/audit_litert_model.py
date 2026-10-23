"""Inspect a large LiteRT FlatBuffer without copying its weights into RAM."""

import argparse
from collections import Counter
import mmap
from pathlib import Path

from ai_edge_litert import schema_py_generated as tflite


def tensor_label(graph, index: int) -> str:
    tensor = graph.Tensors(index)
    shape = tensor.ShapeAsNumpy().tolist()
    return f"{tensor.Name().decode(errors='replace')}: type={tensor.Type()} shape={shape}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    with args.model.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        model = tflite.Model.GetRootAsModel(raw, 0)
        graph = model.Subgraphs(0)
        counts = Counter(
            max(model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode(),
                model.OperatorCodes(graph.Operators(i).OpcodeIndex()).DeprecatedBuiltinCode())
            for i in range(graph.OperatorsLength())
        )
        names = {value: name for name, value in vars(tflite.BuiltinOperator).items() if isinstance(value, int)}
        print(f"bytes={len(raw)} ops={graph.OperatorsLength()} buffers={model.BuffersLength()}")
        for i in range(graph.InputsLength()):
            print("input", tensor_label(graph, graph.Inputs(i)))
        for i in range(graph.OutputsLength()):
            print("output", tensor_label(graph, graph.Outputs(i)))
        for kind, count in counts.most_common():
            print(f"{names.get(kind, kind)}: {count}")


if __name__ == "__main__":
    main()
