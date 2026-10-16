# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporal Winograd F(2,3) for the Wan VAE's causal 3x3x3 convolutions: kernel-level bench.

One steady-state chunk (2 cached + 4 new input frames -> 4 output frames). Compares
cuDNN conv3d against the Winograd form: per output-frame pair, four 3x3 conv2d calls
on +-1 frame combinations with transformed weights, then an output combine.
Reports time (CUPTI, graph replay, L2 flush) and accuracy against an fp32 conv3d.
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

# (cin, cout, H, W): decoder steady-state shapes at 720p, then encoder ones.
CASES = [
    (1024, 1024, 90, 160),
    (1024, 512, 180, 320),
    (512, 512, 180, 320),
    (512, 256, 360, 640),
    (256, 256, 360, 640),
    (640, 640, 90, 160),
    (320, 320, 180, 320),
    (160, 160, 360, 640),
]
CL2 = torch.channels_last
CL3 = torch.channels_last_3d


def winograd_weights(w: torch.Tensor) -> torch.Tensor:
    """[Cout, Cin, 3, kh, kw] -> [4, Cout, Cin, kh, kw]: G0 = g0, G1 = (g0+g1+g2)/2, G2 = (g0-g1+g2)/2, G3 = g2."""
    g0, g1, g2 = (w[:, :, i].float() for i in range(3))
    return torch.stack([g0, (g0 + g1 + g2) / 2, (g0 - g1 + g2) / 2, g2]).to(w.dtype)


def winograd_inputs(x: torch.Tensor) -> torch.Tensor:
    """x [1, C, 6, H, W] -> D [4, 2, C, H, W]: per pair p (frames 2p..2p+3), D0 = d0-d2, D1 = d1+d2, D2 = d2-d1, D3 = d1-d3."""
    d = x[0]  # [C, 6, H, W]
    pairs = []
    for p in range(2):
        d0, d1, d2, d3 = (d[:, 2 * p + i] for i in range(4))
        pairs.append(torch.stack([d0 - d2, d1 + d2, d2 - d1, d1 - d3]))  # [4, C, H, W]
    return torch.stack(pairs, dim=1)  # [4, 2, C, H, W]


def winograd_combine(m: list[torch.Tensor]) -> torch.Tensor:
    """m[k] [2, Cout, H, W] -> y [1, Cout, 4, H, W]: y_even = m0+m1+m2, y_odd = m1-m2-m3."""
    even = m[0] + m[1] + m[2]
    odd = m[1] - m[2] - m[3]
    y = torch.stack([even, odd], dim=1)  # [2 pairs, 2, Cout, H, W]
    return y.reshape(4, *even.shape[1:]).permute(1, 0, 2, 3).unsqueeze(0)


def winograd_full(x: torch.Tensor, gw: torch.Tensor) -> torch.Tensor:
    D = winograd_inputs(x)
    m = [F.conv2d(D[k].contiguous(memory_format=CL2), gw[k], None, 1, 1) for k in range(4)]
    return winograd_combine(m)


def winograd_grouped(x: torch.Tensor, gw_grouped: torch.Tensor) -> torch.Tensor:
    """One grouped conv2d: input [2, 4*Cin, H, W] (D_k stacked on channels), weight [4*Cout, Cin, 3, 3], groups=4."""
    D = winograd_inputs(x)  # [4, 2, C, H, W]
    Dg = D.permute(1, 0, 2, 3, 4).reshape(2, -1, *D.shape[-2:]).contiguous(memory_format=CL2)
    mg = F.conv2d(Dg, gw_grouped, None, 1, 1, 1, 4)  # [2, 4*Cout, H, W]
    cout = gw_grouped.shape[0] // 4
    m = [mg[:, k * cout : (k + 1) * cout] for k in range(4)]
    return winograd_combine(m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'cin->cout @ HxW':>22s} | {'conv3d':>8s} {'wino 4x2d':>9s} {'wino core':>9s} {'grouped':>8s} {'xform':>6s} | {'speedup':>7s} {'core':>5s} | {'relL2 3d':>9s} {'relL2 wino':>10s}")
    rows = []
    for cin, cout, h, w in CASES:
        gen = torch.Generator(device="cuda").manual_seed(cin + h)
        x32 = torch.randn(1, cin, 6, h, w, generator=gen, device="cuda")
        w32 = 0.05 * torch.randn(cout, cin, 3, 3, 3, generator=gen, device="cuda")
        x = x32.to(torch.bfloat16).contiguous(memory_format=CL3)
        wt = w32.to(torch.bfloat16).contiguous(memory_format=CL3)
        gw = winograd_weights(wt).contiguous()
        gw = torch.stack([gw[k].contiguous(memory_format=CL2) for k in range(4)])
        gw_grouped = gw.reshape(4 * cout, cin, 3, 3).contiguous(memory_format=CL2)
        D = winograd_inputs(x)
        Dk = [D[k].contiguous(memory_format=CL2) for k in range(4)]

        def conv3d():
            return F.conv3d(x, wt, None, 1, (0, 1, 1))

        def wino_core():
            return [F.conv2d(Dk[k], gw[k], None, 1, 1) for k in range(4)]

        def wino_full():
            return winograd_full(x, gw)

        def grouped():
            return winograd_grouped(x, gw_grouped)

        def xform_only():
            return winograd_combine([Dk[k][:, :cout] if cin >= cout else Dk[k].repeat(1, cout // cin, 1, 1) for k in range(4)]), winograd_inputs(x)

        t3 = _time_cupti(conv3d, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        tc = _time_cupti(wino_core, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        tf = _time_cupti(wino_full, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        try:
            tg = _time_cupti(grouped, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        except Exception as exc:  # cuDNN may reject the grouped layout
            print(f"grouped failed: {exc}", file=sys.stderr)
            tg = float("nan")
        tx = _time_cupti(xform_only, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3

        # accuracy vs fp32 conv3d (exact algebra of the Winograd form checked in fp32 too)
        ref = F.conv3d(x32.contiguous(memory_format=CL3), w32.contiguous(memory_format=CL3), None, 1, (0, 1, 1))
        y3 = conv3d().float()
        yw = wino_full().float()
        yw32 = winograd_full(x32.contiguous(memory_format=CL3), winograd_weights(w32).contiguous()).float()
        rel = lambda a: ((a - ref).norm() / ref.norm()).item()
        exact = rel(yw32)
        print(f"{cin:5d}->{cout:4d} @ {h:3d}x{w:3d} | {t3:8.3f} {tf:9.3f} {tc:9.3f} {tg:8.3f} {tx:6.3f} | {t3 / tf:6.2f}x {t3 / tc:4.2f}x | {rel(y3):9.2e} {rel(yw):10.2e}   (fp32 algebra {exact:.1e})")
        rows.append({"cin": cin, "cout": cout, "h": h, "w": w, "conv3d_ms": t3, "wino_full_ms": tf, "wino_core_ms": tc, "grouped_ms": tg, "xform_ms": tx, "rel_l2_conv3d_bf16": rel(y3), "rel_l2_wino_bf16": rel(yw), "rel_l2_wino_fp32": exact})
    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "results": rows}, indent=1))


if __name__ == "__main__":
    main()
