# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-shape cuDNN convolution audit for the Wan VAE (VaeBlaze).

Records every conv2d/conv3d call made by one decode chunk and one encode chunk
(steady state and first chunk), then times each distinct shape standalone under
CUPTI (CUDA-graph replay, L2 flush) and reports achieved TFLOP/s, the kernel names
cuDNN dispatched to, and any layout-conversion kernels launched around them.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_wan_vae import COSMOS_VAE_CONFIG, _reference_state_dict, _time_cupti  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.wan.vae_blaze import VaeBlaze, VaeBlazeOptions  # noqa: E402

BLAZE_OPTS = {"fused_norm": True, "fused_residual": True, "fused_upsample": True, "fused_bias": True}
# GB200 dense BF16 tensor-core peak (no sparsity); override with --peak-tflops.
DEFAULT_PEAK_TFLOPS = 2250.0


class ConvRecorder(TorchFunctionMode):
    """Collects (shape signature -> count) for every conv2d/conv3d call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: "OrderedDict[tuple, dict]" = OrderedDict()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (F.conv2d, F.conv3d, torch.conv2d, torch.conv3d):
            x, w = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
            padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
            nd = w.dim() - 2
            stride = tuple(stride) if isinstance(stride, (tuple, list)) else (stride,) * nd
            padding = tuple(padding) if isinstance(padding, (tuple, list)) else (padding,) * nd
            layout = "channels_last" if (
                x.is_contiguous(memory_format=torch.channels_last_3d if nd == 3 else torch.channels_last)
            ) else ("contiguous" if x.is_contiguous() else "strided")
            key = (nd, tuple(x.shape), tuple(w.shape), stride, padding, bias is not None, layout, str(x.dtype))
            e = self.calls.setdefault(key, {"count": 0})
            e["count"] += 1
        return func(*args, **kwargs)


def _macs(key) -> float:
    nd, xs, ws, stride, padding, _, _, _ = key
    cout, cin_g = ws[0], ws[1]
    kernel = ws[2:]
    spatial_in = xs[2:]
    out = []
    for n, k, s, p in zip(spatial_in, kernel, stride, padding):
        out.append((n + 2 * p - k) // s + 1)
    positions = xs[0]
    for o in out:
        positions *= o
    taps = 1
    for k in kernel:
        taps *= k
    return float(positions) * cout * cin_g * taps


def _out_bytes(key) -> int:
    nd, xs, ws, stride, padding, _, _, _ = key
    out = [(n + 2 * p - k) // s + 1 for n, k, s, p in zip(xs[2:], ws[2:], stride, padding)]
    n = xs[0] * ws[0]
    for o in out:
        n *= o
    return n * 2


def _make_tensors(key):
    nd, xs, ws, stride, padding, has_bias, layout, dtype = key
    dt = getattr(torch, dtype.split(".")[-1])
    x = torch.randn(*xs, device="cuda").to(dt)
    w = (0.05 * torch.randn(*ws, device="cuda")).to(dt)
    if layout == "channels_last":
        mf = torch.channels_last_3d if nd == 3 else torch.channels_last
        x = x.contiguous(memory_format=mf)
        w = w.contiguous(memory_format=mf)
    b = torch.randn(ws[0], device="cuda").to(dt) if has_bias else None
    return x, w, b, stride, padding


def record_model_convs(model: VaeBlaze, shape_name: str, latent, pixels) -> "OrderedDict[tuple, dict]":
    rec = ConvRecorder()
    with torch.inference_mode(), rec:
        model.decode(latent, return_dict=False, temporal_chunk_size=1)
        model.encode(pixels, return_dict=False)[0].mode()
    for e in rec.calls.values():
        e["shape_set"] = shape_name
    return rec.calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    parser.add_argument("--latent-frames", type=int, default=3, help="latent frames for the decode pass (chunk size 1)")
    parser.add_argument("--benchmark-mode", action="store_true", help="also time with torch.backends.cudnn.benchmark=True")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  cuDNN {torch.backends.cudnn.version()}  torch {torch.__version__}")
    sd = _reference_state_dict(1234)
    model = VaeBlaze(COSMOS_VAE_CONFIG, VaeBlazeOptions(**BLAZE_OPTS)).cuda()
    model.load_state_dict(sd, strict=True)
    model = model.to(torch.bfloat16).eval().requires_grad_(False)

    # 720p: latent 45x80 -> 720x1280; pixel chunk 1 + 4 frames.
    lat = torch.randn(1, 48, args.latent_frames, 45, 80, device="cuda", dtype=torch.bfloat16)
    pix = torch.randn(1, 3, 1 + 4, 720, 1280, device="cuda", dtype=torch.bfloat16)
    calls = record_model_convs(model, "720p", lat, pix)
    print(f"distinct conv shapes: {len(calls)}  (calls: {sum(e['count'] for e in calls.values())})")

    results = []
    hdr = f"{'#':>3s} {'nd':>2s} {'input':>26s} {'weight':>20s} {'str':>5s} {'lay':>13s} {'calls':>5s} | {'ms':>8s} {'TFLOP/s':>8s} {'%peak':>6s} {'out GB/s':>8s} | kernels"
    print(hdr)
    for i, (key, e) in enumerate(calls.items()):
        nd, xs, ws, stride, padding, has_bias, layout, dtype = key
        x, w, b, s_, p_ = _make_tensors(key)
        conv = F.conv3d if nd == 3 else F.conv2d

        def run():
            return conv(x, w, b, s_, p_)

        t = _time_cupti(run, args.warmup, args.iters, use_graph=True)
        ms = t["span"]["median_us"] / 1e3
        flops = 2 * _macs(key)
        tflops = flops / (ms / 1e3) / 1e12
        top = t["top_kernels_us_per_iter"]
        names = [k["name"] for k in top]
        kernel_us = {k["name"]: k["us"] for k in top}
        short = [f"{n[:60]} ({kernel_us[n]:.0f}us)" for n in names[:3]]
        extra = [n for n in names if any(tok in n.lower() for tok in ("nchw", "nhwc", "transpose", "copy", "elementwise"))]
        row = {
            "nd": nd, "input": list(xs), "weight": list(ws), "stride": list(stride), "padding": list(padding),
            "bias": has_bias, "layout": layout, "dtype": dtype, "calls_per_pass": e["count"], "ms": ms,
            "tflops": tflops, "pct_peak": 100 * tflops / args.peak_tflops,
            "out_gbs": _out_bytes(key) / (ms / 1e3) / 1e9, "kernels": top[:6], "side_kernels": extra, "kernels_per_iter": t["kernels_per_iter"],
        }
        results.append(row)
        print(f"{i:3d} {nd:2d} {str(list(xs)):>26s} {str(list(ws)):>20s} {str(stride):>5s} {layout:>13s} {e['count']:5d} | {ms:8.3f} {tflops:8.1f} {row['pct_peak']:5.1f}% {row['out_gbs']:8.0f} | {'; '.join(short)}" + (f"  !! side: {len(extra)}" if extra else ""))

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True
        print("\n--- cudnn.benchmark=True ---")
        for i, (key, e) in enumerate(calls.items()):
            nd = key[0]
            x, w, b, s_, p_ = _make_tensors(key)
            conv = F.conv3d if nd == 3 else F.conv2d
            t = _time_cupti(lambda: conv(x, w, b, s_, p_), args.warmup, args.iters, use_graph=True)
            ms = t["span"]["median_us"] / 1e3
            results[i]["ms_benchmark"] = ms
            print(f"{i:3d} {ms:8.3f} ms  ({results[i]['ms'] / ms:.2f}x vs default)")

    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": gpu, "cudnn": torch.backends.cudnn.version(), "peak_tflops": args.peak_tflops, "results": results}, indent=1))


if __name__ == "__main__":
    main()
