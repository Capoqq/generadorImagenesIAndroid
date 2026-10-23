"""Print MNN FlatBuffer operation counts for a conversion feasibility check.

Generate Python bindings first with `flatc --python -o <dir> -I schema/default
schema/default/MNN.fbs`, then put <dir> on PYTHONPATH.
"""

import argparse
from collections import Counter
from pathlib import Path

from MNN.Net import Net
from MNN.OpType import OpType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    graph = Net.GetRootAs(args.model.read_bytes())
    names = {value: name for name, value in vars(OpType).items() if isinstance(value, int)}
    counts = Counter(graph.Oplists(i).Type() for i in range(graph.OplistsLength()))
    print(f"{args.model}: {graph.OplistsLength()} ops")
    for kind, count in counts.most_common():
        print(f"{names.get(kind, kind)}: {count}")


if __name__ == "__main__":
    main()
