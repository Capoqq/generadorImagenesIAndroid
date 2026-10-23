"""Split a static single-subgraph LiteRT graph at bounded activation frontiers.

Constants are copied only for the current partition. Tensor IDs in manifest.json
define the connections between partitions, including U-Net skip connections.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import mmap
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema


def split(source: Path, destination: Path, target_mib: int, frontier_mib: int,
          expand_constant_broadcast: bool = False):
    destination.mkdir(parents=True, exist_ok=True)
    stream = source.open("rb")
    raw = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAsModel(raw, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("Only single-subgraph static models are supported")
    graph = model.Subgraphs(0)
    count = graph.OperatorsLength()
    tensors = [graph.Tensors(i) for i in range(graph.TensorsLength())]
    def buffer_size(index):
        buffer = model.Buffers(index)
        return buffer.Size() if buffer.Offset() else buffer.DataLength()
    constant = {i for i, t in enumerate(tensors) if buffer_size(t.Buffer())}
    widths = {schema.TensorType.FLOAT32: 4, schema.TensorType.FLOAT16: 2,
              schema.TensorType.INT32: 4, schema.TensorType.INT64: 8,
              schema.TensorType.INT8: 1, schema.TensorType.BOOL: 1}
    tensor_info = {}
    for i, t in enumerate(tensors):
        shape = [int(v) for v in t.ShapeAsNumpy()]
        if any(v < 0 for v in shape) or t.IsVariable():
            raise ValueError("Dynamic or variable tensors are unsupported")
        tensor_info[i] = {"shape": shape, "type": int(t.Type()),
                          "bytes": math.prod(shape) * widths[t.Type()]}
    producer = {int(v): -1 for v in graph.InputsAsNumpy()}
    last_use = {}
    op_inputs, op_outputs = [], []
    for index in range(count):
        op = graph.Operators(index)
        inputs = [int(v) for v in op.InputsAsNumpy() if v >= 0]
        outputs = [int(v) for v in op.OutputsAsNumpy() if v >= 0]
        op_inputs.append(inputs)
        op_outputs.append(outputs)
        for value in inputs:
            if value not in constant and value not in producer:
                raise ValueError(f"Non-topological graph at op {index}: tensor {value}")
            last_use[value] = index
        for value in outputs:
            producer[value] = index
    for value in graph.OutputsAsNumpy():
        last_use[int(value)] = count

    # Exporters often schedule every weight DEQUANTIZE before the first layer.
    # These values are constants too; duplicate their small producer ops at the
    # consumers rather than passing gigabytes of expanded weights between cuts.
    static = set(constant)
    static_ops = set()
    for index in range(count):
        if all(v in static for v in op_inputs[index]):
            static_ops.add(index)
            static.update(op_outputs[index])
    def dependencies(values):
        ops, buffers = set(), set()
        def visit(value):
            if value in constant:
                buffers.add(tensors[value].Buffer())
            elif value in static:
                index = producer[value]
                if index not in ops:
                    ops.add(index)
                    for entry in op_inputs[index]:
                        visit(entry)
        for value in values:
            visit(value)
        return ops, buffers

    # Cut only when all live activations together fit the frontier budget.
    # This avoids materializing the large attention score matrices at boundaries.
    boundaries = [0]
    weight_buffers = set()
    weight_size = 0
    for index in range(count):
        if index in static_ops:
            continue
        _, required_buffers = dependencies(op_inputs[index])
        for buf in required_buffers:
            if buf not in weight_buffers:
                weight_size += buffer_size(buf)
                weight_buffers.add(buf)
        end = index + 1
        live_bytes = sum(tensor_info[t]["bytes"] for t, p in producer.items()
                         if t not in static and p < end and last_use.get(t, -1) >= end)
        if weight_size >= target_mib * 1024**2 and live_bytes <= frontier_mib * 1024**2:
            boundaries.append(end)
            weight_buffers.clear()
            weight_size = 0
    if boundaries[-1] != count:
        boundaries.append(count)

    manifest = {"source": source.name, "source_bytes": len(raw),
                "inputs": [int(v) for v in graph.InputsAsNumpy()],
                "outputs": [int(v) for v in graph.OutputsAsNumpy()],
                "tensors": tensor_info, "partitions": []}
    for number, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        selected_ops = set(range(start, end)) - static_ops
        static_dependencies, _ = dependencies(
            [v for index in selected_ops for v in op_inputs[index]])
        selected_ops.update(static_dependencies)
        selected_ops = sorted(selected_ops)
        used = {v for index in selected_ops for v in op_inputs[index] + op_outputs[index]}
        ordered = sorted(used)
        remap = {old: new for new, old in enumerate(ordered)}
        inputs = sorted(v for v in used if v not in static and producer[v] < start)
        outputs = sorted(v for v in used if v not in static and
                         start <= producer[v] < end and last_use.get(v, -1) >= end)
        buffer_ids = sorted({0} | {tensors[v].Buffer() for v in ordered})
        buffer_map = {old: new for new, old in enumerate(buffer_ids)}
        result = schema.ModelT()
        result.version = model.Version()
        result.description = f"RealVision U-Net partition {number}: ops [{start}, {end})".encode()
        result.operatorCodes = [schema.OperatorCodeT.InitFromObj(model.OperatorCodes(i))
                                for i in range(model.OperatorCodesLength())]
        result.buffers = []
        for i in buffer_ids:
            original = model.Buffers(i)
            buffer = schema.BufferT.InitFromObj(original)
            if original.Offset():
                if original.Offset() + original.Size() > len(raw):
                    raise ValueError(f"Buffer {i} extends beyond model file")
                buffer.data = np.frombuffer(raw, dtype=np.uint8,
                                            count=original.Size(), offset=original.Offset())
                buffer.offset = buffer.size = 0
            result.buffers.append(buffer)
        sub = schema.SubGraphT()
        sub.name = b"main"
        sub.tensors = []
        for old in ordered:
            t = schema.TensorT.InitFromObj(tensors[old])
            t.buffer = buffer_map[t.buffer]
            t.name = f"t{old}".encode()
            sub.tensors.append(t)
        sub.inputs = [remap[v] for v in inputs]
        sub.outputs = [remap[v] for v in outputs]
        sub.operators = []
        for index in selected_ops:
            op = schema.OperatorT.InitFromObj(graph.Operators(index))
            op.inputs = [remap[int(v)] if v >= 0 else -1 for v in op.inputs]
            op.outputs = [remap[int(v)] if v >= 0 else -1 for v in op.outputs]
            if op.intermediates is not None and len(op.intermediates):
                raise ValueError("Explicit operator intermediates are unsupported")
            sub.operators.append(op)
        result.subgraphs = [sub]
        expanded_count = 0
        if expand_constant_broadcast:
            for op in sub.operators:
                code = result.operatorCodes[op.opcodeIndex]
                if max(code.builtinCode, code.deprecatedBuiltinCode) != schema.BuiltinOperator.MUL:
                    continue
                output_shape = list(sub.tensors[op.outputs[0]].shape)
                shapes = [list(sub.tensors[v].shape) for v in op.inputs]
                if output_shape in shapes or math.prod(output_shape) > 262144:
                    continue
                for position, index in enumerate(op.inputs):
                    tensor = sub.tensors[index]
                    buffer = result.buffers[tensor.buffer]
                    if buffer.data is None or len(buffer.data) == 0:
                        continue
                    dtype = {schema.TensorType.FLOAT32: np.float32,
                             schema.TensorType.FLOAT16: np.float16}.get(tensor.type)
                    if dtype is None:
                        continue
                    original = np.frombuffer(buffer.data, dtype=dtype).reshape(shapes[position])
                    expanded = np.broadcast_to(original, output_shape).copy()
                    replacement_buffer = schema.BufferT()
                    replacement_buffer.data = expanded.view(np.uint8).reshape(-1)
                    replacement = copy.deepcopy(tensor)
                    replacement.name = tensor.name + b"_expanded"
                    replacement.shape = output_shape
                    replacement.shapeSignature = output_shape
                    replacement.buffer = len(result.buffers)
                    result.buffers.append(replacement_buffer)
                    op.inputs[position] = len(sub.tensors)
                    sub.tensors.append(replacement)
                    expanded_count += 1
                    break
        signature = schema.SignatureDefT()
        signature.signatureKey = b"serving_default"
        signature.subgraphIndex = 0
        def maps(ids):
            entries = []
            for value in ids:
                entry = schema.TensorMapT()
                entry.name = f"t{value}".encode()
                entry.tensorIndex = remap[value]
                entries.append(entry)
            return entries
        signature.inputs = maps(inputs)
        signature.outputs = maps(outputs)
        result.signatureDefs = [signature]
        builder = flatbuffers.Builder(1024)
        builder.Finish(result.Pack(builder), file_identifier=b"TFL3")
        data = builder.Output()
        filename = f"unet_{number:02d}.tflite"
        (destination / filename).write_bytes(data)
        item = {"file": filename, "op_start": start, "op_end": end,
                "inputs": inputs, "outputs": outputs, "bytes": len(data),
                "expanded_broadcast_constants": expanded_count,
                "sha256": hashlib.sha256(data).hexdigest()}
        manifest["partitions"].append(item)
        print(json.dumps(item), flush=True)
        del data, builder, result, sub
        gc.collect()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-mib", type=int, default=128)
    parser.add_argument("--frontier-mib", type=int, default=128)
    parser.add_argument("--expand-constant-broadcast", action="store_true")
    args = parser.parse_args()
    split(args.model, args.output, args.target_mib, args.frontier_mib,
          args.expand_constant_broadcast)
