"""Replace identity GATHER_ND nodes in experimental SD 1.5 LiteRT graphs.

The export uses GATHER_ND to turn a length-N vector into [batch, channels].
Every index is consecutive, so RESHAPE preserves the tensor values.
Requires: pip install tflite numpy.
"""

import argparse
import hashlib
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import tflite


def vector_start(table, field):
    return table._tab.Vector(table._tab.Offset(field))


def rewrite(source: Path, destination: Path) -> None:
    raw = bytearray(source.read_bytes())
    model = tflite.Model.GetRootAsModel(raw, 0)
    graph = model.Subgraphs(0)
    reshape_opcode = next(
        i for i in range(model.OperatorCodesLength())
        if model.OperatorCodes(i).BuiltinCode() == 22
    )
    gathers = [
        graph.Operators(i) for i in range(graph.OperatorsLength())
        if model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode() == 107
    ]
    assert gathers, "No GATHER_ND nodes to rewrite"
    index_tensors = Counter(op.Inputs(1) for op in gathers)
    all_consumers = Counter(
        graph.Operators(i).Inputs(j)
        for i in range(graph.OperatorsLength())
        for j in range(graph.Operators(i).InputsLength())
        if graph.Operators(i).Inputs(j) >= 0
    )
    original_reshapes = sum(
        model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode() == 22
        for i in range(graph.OperatorsLength())
    )

    for index in index_tensors:
        assert all_consumers[index] == index_tensors[index], "Index tensor has another consumer"
        tensor = graph.Tensors(index)
        assert tensor.Type() == 2  # INT32
        dimensions = tensor.ShapeAsNumpy().tolist()
        assert len(dimensions) == 3 and dimensions[2] == 1
        batch, channels, _ = dimensions
        size = batch * channels
        buffer = model.Buffers(tensor.Buffer())
        indices = np.frombuffer(buffer.DataAsNumpy(), dtype=np.int32)
        assert np.array_equal(indices, np.arange(size, dtype=np.int32))
        shape_start = vector_start(tensor, 4)
        struct.pack_into('<I', raw, shape_start - 4, 1)
        struct.pack_into('<i', raw, shape_start, 2)
        data_start = vector_start(buffer, 4)
        struct.pack_into('<I', raw, data_start - 4, 8)
        struct.pack_into('<ii', raw, data_start, batch, channels)

    for op in gathers:
        source_tensor = graph.Tensors(op.Inputs(0))
        result_tensor = graph.Tensors(op.Outputs(0))
        assert source_tensor.Type() == result_tensor.Type()
        assert source_tensor.ShapeAsNumpy().tolist() == [int(np.prod(result_tensor.ShapeAsNumpy()))]
        assert op.InputsLength() == 2 and op.BuiltinOptionsType() == 0
        struct.pack_into('<I', raw, op._tab.Pos + op._tab.Offset(4), reshape_opcode)

    updated = tflite.Model.GetRootAsModel(raw, 0)
    counts = Counter(
        updated.OperatorCodes(updated.Subgraphs(0).Operators(i).OpcodeIndex()).BuiltinCode()
        for i in range(updated.Subgraphs(0).OperatorsLength())
    )
    assert counts[107] == 0 and counts[22] == original_reshapes + len(gathers)
    destination.write_bytes(raw)
    print(f"{destination}: {len(raw)} bytes, {len(gathers)} identity GATHER_ND rewritten, "
          f"SHA-256 {hashlib.sha256(raw).hexdigest()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    rewrite(args.source, args.destination)
