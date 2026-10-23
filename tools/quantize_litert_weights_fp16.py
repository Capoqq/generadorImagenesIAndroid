"""Quantize LiteRT constant weights, retaining FP32 activations."""

import argparse
import resource
import time

from ai_edge_quantizer import algorithm_manager, qtyping, quantizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--bits", type=int, choices=(8, 16), default=16)
    args = parser.parse_args()

    started = time.monotonic()
    model_quantizer = quantizer.Quantizer(args.input)
    model_quantizer.add_weight_only_config(
        regex=".*",
        operation_name="*",
        num_bits=args.bits,
        granularity=(
            qtyping.QuantGranularity.CHANNELWISE
            if args.bits == 16
            else qtyping.QuantGranularity.TENSORWISE
        ),
        algorithm_key=(
            algorithm_manager.AlgorithmName.FLOAT_CASTING
            if args.bits == 16
            else algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT
        ),
    )
    model_quantizer.quantize(
        serialize_to_path=args.output,
        enable_progress_bar=True,
        enable_progress_report=True,
    )
    print(
        f"Completed in {time.monotonic() - started:.1f}s; "
        f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
