# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd F(2x2, 3x3) feasibility bench for the Wan VAE's 3x3 convs.

Kernel-level, no model wiring. The current path (after temporal Winograd) runs a
residual-block conv as four 2D 3x3 conv2d calls per frame pair, one per temporal
component, each with its own weights. The portable spatial-Winograd form replaces
each with: input transform (4x4 tiles, stride 2) -> 64 batched GEMMs (4 components x
16 tile coordinates) through cuBLAS -> output transform (2x2 outputs per tile).

Reported per layer: the conv2d path, the GEMMs alone (the tensor-core work), the two
transforms done eagerly (an upper bound), the full eager path, the transformed-data
bytes and the time they would take at copy bandwidth (the floor for any fused
transform), and the accuracy of both paths against an fp32 conv2d.
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

import tensorrt_llm  # noqa: E402, F401  (loads the trtllm:: custom ops)

# (cin, cout, H, W): the 720p decoder levels, then the two 1024-channel levels of the
# 480x832 bimanual clip.
CASES = [
    (1024, 1024, 45, 80),
    (1024, 1024, 90, 160),
    (512, 512, 180, 320),
    (256, 256, 360, 640),
    (1024, 1024, 30, 52),
    (1024, 1024, 60, 104),
]
COMPONENTS = 4  # temporal Winograd components per frame pair
PAIRS = 2  # frame pairs per steady chunk (4 output frames)

# F(2x2, 3x3) transform matrices (Lavin & Gray).
BT = torch.tensor([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=torch.float32)
G = torch.tensor([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=torch.float32)
AT = torch.tensor([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=torch.float32)


_DEV: dict[str, torch.Tensor] = {}


def _const(name: str, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    key = f"{name}:{device}"
    if key not in _DEV:
        _DEV[key] = t.to(device)
    return _DEV[key]


def transform_weights(w: torch.Tensor) -> torch.Tensor:
    """W [comp, Cout, Cin, 3, 3] -> U [comp*16, Cin, Cout] (fp32 arithmetic, caller casts)."""
    g = _const("G", G, w.device)
    u = torch.einsum(
        "xa,noiab,yb->nxyoi", g, w.float(), g
    )  # U[x, y] = G g G^T per (cout, cin): [comp, 4, 4, Cout, Cin]
    comp, _, _, cout, cin = u.shape
    return u.reshape(comp * 16, cout, cin).transpose(1, 2).contiguous()


def transform_input(d: torch.Tensor) -> torch.Tensor:
    """D [comp*pairs, Cin, H, W] -> V [comp*16, pairs*tiles, Cin] (fp32 arithmetic, caller casts).

    Tiles are 4x4 windows at stride 2 of the 1-padded input; tile (i, j) feeds output
    pixels (2i, 2j) .. (2i+1, 2j+1).
    """
    n, cin, h, w = d.shape
    bt = _const("BT", BT, d.device)
    padded = F.pad(d.float(), (1, 1 + w % 2, 1, 1 + h % 2))  # odd sizes: a dropped last row/col
    tiles = F.unfold(padded, kernel_size=4, stride=2)  # [n, cin*16, L]
    L = tiles.shape[-1]
    tiles = tiles.view(n, cin, 4, 4, L)
    v = torch.einsum(
        "xa,ncabl,yb->nxylc", bt, tiles, bt
    )  # V[x, y] = B^T d B per tile: [n, 4, 4, L, cin]
    comp = COMPONENTS
    pairs = n // comp
    v = v.reshape(comp, pairs, 16, L, cin).permute(0, 2, 1, 3, 4).reshape(comp * 16, pairs * L, cin)
    return v


def transform_output(m: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """M [comp*16, pairs*tiles, Cout] -> y [comp*pairs, Cout, H, W] (fp32 arithmetic, caller casts)."""
    at = _const("AT", AT, m.device)
    th, tw = (h + 1) // 2, (w + 1) // 2
    L = th * tw
    comp16, pl, cout = m.shape
    comp = comp16 // 16
    pairs = pl // L
    mm = m.float().view(comp, 4, 4, pairs, L, cout)
    y = torch.einsum(
        "rx,nxyplo,cy->nplorc", at, mm, at
    )  # Y = A^T M A per tile: [comp, pairs, L, cout, 2, 2]
    y = (
        y.reshape(comp * pairs, th, tw, cout, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(comp * pairs, cout, 2 * th, 2 * tw)
    )
    return y[..., :h, :w]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--copy-tbs",
        type=float,
        default=6.2,
        help="measured large-copy bandwidth for the traffic floor",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--case", type=int, default=None, help="run only CASES[i]")
    parser.add_argument(
        "--probe",
        type=int,
        default=None,
        help="no timing, for ncu: 0 runs the CUDA path once whole-frame, N > 0 once with N tile rows per strip",
    )
    args = parser.parse_args()
    cases = CASES if args.case is None else [CASES[args.case]]
    if args.probe is None:
        atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    header = [
        f"{'cin->cout @ HxW':>20s} |",
        f" {'conv2d x4':>9s} {'GEMMs x64':>9s} {'xform in':>8s} {'xform out':>9s} {'full':>8s} |",
        f" {'GEMM TF/s':>9s} {'ratio':>6s} | {'xfm GB':>6s} {'floor ms':>8s} {'net':>7s} |",
        f" {'relL2 conv':>10s} {'relL2 wino':>10s}",
    ]
    print("".join(header))
    rows = []
    for cin, cout, h, w in cases:
        gen = torch.Generator(device="cuda").manual_seed(cin + h)
        n = COMPONENTS * PAIRS
        d32 = torch.randn(n, cin, h, w, generator=gen, device="cuda")
        w32 = 0.05 * torch.randn(COMPONENTS, cout, cin, 3, 3, generator=gen, device="cuda")
        d = d32.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
        wts = [
            w32[k].to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
            for k in range(COMPONENTS)
        ]
        d_comp = [d[k * PAIRS : (k + 1) * PAIRS] for k in range(COMPONENTS)]

        def conv_path():
            return [F.conv2d(d_comp[k], wts[k], None, 1, 1) for k in range(COMPONENTS)]

        U = transform_weights(w32.to(torch.bfloat16)).to(
            torch.bfloat16
        )  # [64, Cin, Cout], rounded once
        V = transform_input(d).to(torch.bfloat16).contiguous()  # [64, pairs*L, Cin]

        def gemms():
            return torch.bmm(V, U)  # [64, pairs*L, Cout], fp32 accumulate inside cuBLAS

        def xform_in():
            return transform_input(d).to(torch.bfloat16)

        M = gemms()

        def xform_out():
            return transform_output(M, h, w).to(torch.bfloat16)

        def full():
            return transform_output(torch.bmm(transform_input(d).to(torch.bfloat16), U), h, w).to(
                torch.bfloat16
            )

        # CUDA transform kernels (trtllm::vae_wino_spatial_in/out), whole frame and strip-blocked.
        have_cuda_xf = hasattr(torch.ops, "trtllm") and hasattr(
            torch.ops.trtllm, "vae_wino_spatial_in"
        )
        frames_nhwc = d.permute(0, 2, 3, 1).contiguous()  # [N, H, W, Cin]
        th = (h + 1) // 2
        y_buf = torch.empty(n, h, w, cout, dtype=torch.bfloat16, device="cuda")

        def cuda_full():
            v = torch.ops.trtllm.vae_wino_spatial_in(frames_nhwc, COMPONENTS, 0, th)
            mm = torch.bmm(v, U)
            return torch.ops.trtllm.vae_wino_spatial_out(mm, y_buf, COMPONENTS, 0, th)

        def make_strips(rows_per_strip: int):
            def run():
                for r0 in range(0, th, rows_per_strip):
                    r1 = min(th, r0 + rows_per_strip)
                    v = torch.ops.trtllm.vae_wino_spatial_in(frames_nhwc, COMPONENTS, r0, r1)
                    mm = torch.bmm(v, U)
                    torch.ops.trtllm.vae_wino_spatial_out(mm, y_buf, COMPONENTS, r0, r1)
                return y_buf

            return run

        if args.probe is not None:
            if args.probe == 0:
                cuda_full()
            else:
                make_strips(args.probe)()
            torch.cuda.synchronize()
            continue

        t_cuda_full = float("nan")
        strip_results: dict[int, float] = {}
        if have_cuda_xf:
            t_cuda_full = (
                _time_cupti(cuda_full, args.warmup, args.iters, use_graph=True)["span"]["median_us"]
                / 1e3
            )
            for rows_per in (2, 4, 8, 16):
                if rows_per < th:
                    strip_results[rows_per] = (
                        _time_cupti(make_strips(rows_per), args.warmup, args.iters, use_graph=True)[
                            "span"
                        ]["median_us"]
                        / 1e3
                    )

        t_conv = (
            _time_cupti(conv_path, args.warmup, args.iters, use_graph=True)["span"]["median_us"]
            / 1e3
        )
        t_gemm = (
            _time_cupti(gemms, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        )
        t_xin = (
            _time_cupti(xform_in, args.warmup, args.iters, use_graph=True)["span"]["median_us"]
            / 1e3
        )
        t_xout = (
            _time_cupti(xform_out, args.warmup, args.iters, use_graph=True)["span"]["median_us"]
            / 1e3
        )
        t_full = (
            _time_cupti(full, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        )

        L = ((h + 1) // 2) * ((w + 1) // 2)
        gemm_flops = 2.0 * 64 * (PAIRS * L) * cin * cout
        gemm_tflops = gemm_flops / (t_gemm / 1e3) / 1e12
        xfm_bytes = 2 * (V.numel() + M.numel()) * 2  # V and M each written once and read once, bf16
        floor_ms = xfm_bytes / 1e9 / args.copy_tbs
        net_ms = t_conv - (t_gemm + floor_ms)  # what a perfectly fused transform path would save

        # accuracy vs fp32 conv2d
        ref = torch.cat(
            [
                F.conv2d(d32[k * PAIRS : (k + 1) * PAIRS], w32[k], None, 1, 1)
                for k in range(COMPONENTS)
            ]
        )
        y_conv = torch.cat(conv_path()).float()
        y_wino = full().float()
        y_wino32 = transform_output(torch.bmm(transform_input(d32), transform_weights(w32)), h, w)
        rel = lambda a: ((a - ref).norm() / ref.norm()).item()  # noqa: E731
        rel_cuda = float("nan")
        if have_cuda_xf:
            y_cuda = cuda_full().permute(0, 3, 1, 2).float()
            rel_cuda = rel(y_cuda)
            y_strip = make_strips(4)().permute(0, 3, 1, 2).float()
            assert torch.equal(y_strip, y_cuda), (
                "strip and whole-frame CUDA paths must agree bitwise"
            )
        print(
            f"{cin:5d}->{cout:4d} @ {h:3d}x{w:3d} |"
            f" {t_conv:9.3f} {t_gemm:9.3f} {t_xin:8.3f} {t_xout:9.3f} {t_full:8.3f} |"
            f" {gemm_tflops:9.0f} {t_conv / t_gemm:5.2f}x | {xfm_bytes / 1e9:6.2f} {floor_ms:8.3f} {net_ms:+7.3f} |"
            f" {rel(y_conv):10.2e} {rel(y_wino):10.2e}   (fp32 algebra {rel(y_wino32):.1e})"
        )
        if have_cuda_xf:
            strips = ", ".join(f"{k} rows: {v:.3f}" for k, v in strip_results.items())
            print(
                f"{'':>20s} | CUDA transforms: whole frame {t_cuda_full:.3f} ms"
                f" (net vs conv {t_conv - t_cuda_full:+.3f}); strips [{strips}] ms; relL2 {rel_cuda:.2e}"
            )
        rows.append(
            {
                "cin": cin,
                "cout": cout,
                "h": h,
                "w": w,
                "conv2d_ms": t_conv,
                "gemm_ms": t_gemm,
                "xform_in_ms": t_xin,
                "xform_out_ms": t_xout,
                "full_eager_ms": t_full,
                "gemm_tflops": gemm_tflops,
                "xform_gb": xfm_bytes / 1e9,
                "floor_ms": floor_ms,
                "net_fused_ms": net_ms,
                "rel_l2_conv_bf16": rel(y_conv),
                "rel_l2_wino_bf16": rel(y_wino),
                "rel_l2_wino_fp32": rel(y_wino32),
                "cuda_full_ms": t_cuda_full,
                "strip_ms": strip_results,
                "rel_l2_cuda": rel_cuda,
            }
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"gpu": torch.cuda.get_device_name(0), "copy_tbs": args.copy_tbs, "results": rows},
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
