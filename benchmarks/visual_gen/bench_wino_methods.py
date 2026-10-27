# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-conv cost of the ways a decoder 3x3 conv can run, on the decoder's own layer shapes.

For each layer shape the temporal Winograd slabs of one chunk (``order + 2`` component frames)
are convolved by their component filters with every method that applies: cuDNN direct (one
conv2d per component), spatial Winograd F(2x2, 3x3) and F(4x4, 3x3) on cuBLAS batched GEMMs,
the Kernel Factory F(4x4, 3x3) pipeline and the fused 1D-Winograd kernel. Each method is timed
with CUPTI as the sum of all its kernels (transforms included), CUDA-graph replay, L2 flushed
per iteration, and checked against an fp32 conv2d. Writes JSON for ``plot_wino_methods.py``.

    python benchmarks/visual_gen/bench_wino_methods.py --out wino_methods.json
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_wan_vae import _time_cupti  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (  # noqa: E402
    cuda_wino_spatial_available,
    kf_fused_eligible,
    kf_fused_filters,
    kf_spatial_eligible,
    kf_spatial_filters,
    kf_wino_available,
    wino_spatial_conv_cuda,
    wino_spatial_conv_fused,
    wino_spatial_conv_kf,
    winograd_spatial_weights,
    winograd_weights,
)

# (name, C_in, C_out, H, W, temporal order): the residual-block convs of a 720p decode, one chunk
# of one latent frame. mid/up0 run one frame per chunk in production (spatial Winograd alone);
# here every layer takes the pair path so the methods compare on equal footing.
LAYERS = [
    ("mid / up0\n1024 ch, 45×80", 1024, 1024, 45, 80, 2),
    ("up1\n1024 ch, 90×160", 1024, 1024, 90, 160, 2),
    ("up2\n512 ch, 180×320", 512, 512, 180, 320, 4),
    ("up3\n256 ch, 360×640", 256, 256, 360, 640, 4),
]
# What the decoder ships at each layer.
SHIPPED = {0: "F(2×2) cuBLAS", 1: "F(4×4) KF", 2: "F(4×4) KF", 3: "fused 1D F(2,3)"}


def _as_rows(out, comps: int, pairs: int, h: int, w: int, cout: int) -> torch.Tensor:
    """Any method's output as ``[comps, pairs, H*W, Cout]``."""
    if isinstance(out, (list, tuple)):  # conv2d per component: [pairs, Cout, H, W]
        return torch.stack([o.permute(0, 2, 3, 1).reshape(pairs, h * w, cout) for o in out])
    return out.reshape(comps, pairs, h * w, cout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument(
        "--sweep",
        choices=["layers", "res", "channels"],
        default="layers",
        help="layers: the decoder's shapes; res: fixed channels over resolutions; "
        "channels: fixed resolution over channel counts",
    )
    parser.add_argument("--channels", type=int, default=512, help="C_in = C_out for --sweep res")
    parser.add_argument("--res", default="180x320", help="HxW for --sweep channels")
    parser.add_argument("--sizes", default=None, help="comma-separated HxW list for --sweep res")
    parser.add_argument(
        "--append",
        action="store_true",
        help="extend an existing --out file (one shape per process)",
    )
    args = parser.parse_args()
    if not cuda_wino_spatial_available():
        raise RuntimeError("the spatial Winograd CUDA ops are not built")
    from cupti import cupti

    atexit.register(cupti.finalize)
    torch.manual_seed(0)
    device = torch.device("cuda")
    results = []
    if args.append and Path(args.out).exists():
        results = json.loads(Path(args.out).read_text())["layers"]
    if args.sweep == "layers":
        cases = [(LAYERS[i], SHIPPED[i]) for i in (int(s) for s in args.layers.split(","))]
    elif args.sweep == "res":
        c = args.channels
        sizes = (
            (45, 80),
            (90, 160),
            (180, 320),
            (360, 640),
            (720, 1280),
            (1080, 1920),
            (1440, 2560),
            (2160, 3840),
        )
        if args.sizes:
            sizes = tuple(tuple(int(v) for v in sz.split("x")) for sz in args.sizes.split(","))
        cases = [((f"{h}×{w}", c, c, h, w, 2), "") for h, w in sizes]
    else:
        h, w = (int(v) for v in args.res.split("x"))
        cases = [((f"{c} ch", c, c, h, w, 2), "") for c in (128, 256, 512, 1024)]
    for (name, cin, cout, h, w, order), shipped in cases:
        comps, pairs = order + 2, 1
        d = torch.randn(comps, pairs, h * w, cin, device=device).to(torch.float16)
        weight = (torch.randn(cout, cin, 3, 3, 3, device=device) / (27 * cin) ** 0.5).to(
            torch.bfloat16
        )
        g16 = winograd_weights(weight, order, torch.float16)
        g32 = winograd_weights(weight, order, torch.float32)
        imgs16 = [d[k].view(pairs, h, w, cin).permute(0, 3, 1, 2) for k in range(comps)]

        def rel_l2(out) -> float:
            """Relative L2 error against an fp32 conv2d.

            One component frame at a time: the whole fp32 reference would not fit at 4K.
            """
            rows = _as_rows(out, comps, pairs, h, w, cout)
            num = den = 0.0
            for k in range(comps):
                ref = (
                    F.conv2d(imgs16[k].float(), g32[k].float(), padding=1)
                    .permute(0, 2, 3, 1)
                    .reshape(pairs, h * w, cout)
                )
                num += (rows[k].float() - ref).square().sum().item()
                den += ref.square().sum().item()
                del ref
            return (num / den) ** 0.5

        methods: dict[str, callable] = {}
        methods["direct cuDNN"] = lambda: [
            F.conv2d(imgs16[k], g16[k], padding=1) for k in range(comps)
        ]
        u2 = winograd_spatial_weights(weight, 2, order)
        methods["F(2×2) cuBLAS"] = lambda: wino_spatial_conv_cuda(d, u2, h, w)
        u4 = winograd_spatial_weights(weight, 4, order)
        methods["F(4×4) cuBLAS"] = lambda: wino_spatial_conv_cuda(d, u4, h, w)
        if kf_wino_available() and kf_spatial_eligible(cin, cout, w):
            ukf = kf_spatial_filters(g16)
            methods["F(4×4) KF"] = lambda: wino_spatial_conv_kf(d, ukf, h, w)
        if kf_wino_available() and kf_fused_eligible(cin, cout, h, w):
            fused = kf_fused_filters(g16)
            methods["fused 1D F(2,3)"] = lambda: wino_spatial_conv_fused(d, fused, h, w)

        row = {
            "layer": name,
            "cin": cin,
            "cout": cout,
            "h": h,
            "w": w,
            "order": order,
            "shipped": shipped,
            "methods": {},
        }
        results.append(row)

        def write() -> None:
            meta = {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "iters": args.iters,
                "sweep": args.sweep,
            }
            Path(args.out).write_text(json.dumps({"meta": meta, "layers": results}, indent=1))

        for mname, fn in methods.items():
            try:
                with torch.inference_mode():
                    err = rel_l2(fn())
            except RuntimeError as exc:  # a kernel that does not cover this shape
                msg = str(exc).splitlines()[0][:200]
                print(f"{name.replace(chr(10), ' '):28s} {mname:18s} failed: {msg}", flush=True)
                row["methods"][mname] = {"failed": msg}
                write()
                continue
            t = _time_cupti(fn, args.warmup, args.iters, use_graph=True)
            row["methods"][mname] = {
                "span_us": t["span"]["median_us"],
                "busy_us": t["busy"]["median_us"],
                "kernels": t["kernels_per_iter"],
                "rel_l2": err,
                "top_kernels": t["top_kernels_us_per_iter"][:6],
            }
            write()
            print(
                f"{name.replace(chr(10), ' '):28s} {mname:18s} {t['span']['median_us']:9.1f} us"
                f"  busy {t['busy']['median_us']:9.1f}  kernels {t['kernels_per_iter']:4.0f}  rel_l2 {err:.3e}",
                flush=True,
            )
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
