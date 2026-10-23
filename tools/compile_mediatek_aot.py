"""Attempt offline LiteRT MediaTek compilation and print partition results."""

import argparse
import resource
import time

from ai_edge_litert.aot.aot_compile import aot_compile
from ai_edge_litert.aot.vendors.mediatek.target import SocModel, Target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    started = time.monotonic()
    try:
        result = aot_compile(
            args.model,
            output_dir=args.output_dir,
            target=Target(SocModel.MT6878),
            keep_going=True,
        )
        print(result, flush=True)
    finally:
        print(
            f"Elapsed {time.monotonic() - started:.1f}s; "
            f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
