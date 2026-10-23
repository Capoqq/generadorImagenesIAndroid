"""Export the installed model's Realistic Vision v5.1 U-Net to LiteRT.

Run inside a Linux environment with litert-torch installed. The checkpoint
must be the original SG161222 FP16 no-EMA Stable Diffusion 1.5 safetensors
checkpoint, not the already-quantized MNN weight file.
"""

import argparse
import gc
import os
import resource
import time
import types

import litert_torch
import torch
from torch.nn import functional as torch_functional
from litert_torch.generative.examples.stable_diffusion import diffusion
from litert_torch.generative.examples.stable_diffusion import util
from litert_torch.generative.layers import normalization
from litert_torch.generative.utilities import stable_diffusion_loader


def report(stage: str, started: float) -> None:
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"{stage}: elapsed={time.monotonic() - started:.1f}s peak_rss={peak_mb:.0f}MiB", flush=True)


def fp16_group_norm(self, value):
    result = torch_functional.group_norm(
        value.float(), self.group_num, self.weight.float(), self.bias.float(), self.eps
    )
    return result.to(value.dtype)


def fp16_layer_norm(self, value):
    bias = self.bias.float() if self.bias is not None else None
    result = torch_functional.layer_norm(
        value.float(), self.normalized_shape, self.weight.float(), bias, self.eps
    )
    return result.to(value.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()

    model = diffusion.Diffusion(diffusion.get_model_config(batch_size=2, device_type="cpu"))
    if args.dtype == "fp16":
        model.half()
        for module in model.modules():
            if isinstance(module, normalization.GroupNorm):
                module.forward = types.MethodType(fp16_group_norm, module)
            elif isinstance(module, normalization.LayerNorm):
                module.forward = types.MethodType(fp16_layer_norm, module)
    model.eval()
    report("model instantiated", started)

    loader = stable_diffusion_loader.DiffusionModelLoader(args.checkpoint, diffusion.TENSOR_NAMES)
    missing_keys, unexpected_keys = loader.load(model, strict=False)
    print(
        f"weight mapping: missing={len(missing_keys)} unexpected={len(unexpected_keys)}",
        flush=True,
    )
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"Incomplete checkpoint mapping: missing={missing_keys[:10]}, "
            f"unexpected={unexpected_keys[:10]}"
        )
    del loader
    gc.collect()
    report("checkpoint loaded", started)
    if args.verify_only:
        return

    latents = torch.zeros((2, 4, 64, 64), dtype=torch.float32)
    context = torch.zeros((2, 77, 768), dtype=torch.float32)
    timestep = util.get_time_embedding(0)
    if args.dtype == "fp16":
        latents = latents.half()
        context = context.half()
        timestep = timestep.half()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.inference_mode():
        litert_torch.signature("diffusion", model, (latents, context, timestep)).convert(
            quant_config=None
        ).export(args.output)
    report("LiteRT export finished", started)
    print(f"output={args.output} bytes={os.path.getsize(args.output)}", flush=True)


if __name__ == "__main__":
    main()
