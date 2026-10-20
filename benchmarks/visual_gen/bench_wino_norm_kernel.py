# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-only bandwidth benchmark for the temporal-Winograd norm kernel (jit_norm.wino_norm_silu).

Shapes are the residual-block norms of one steady 720p chunk. Two modes per shape:
"producer" (input x, no residual: a block's norm2) and "consumer" (input m0..m3 of the
previous Winograd conv, residual and deferred bias: a block's norm1). Bytes moved are
the unique reads and writes of the call; the kernel is judged against a torch.add of
the same byte count.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_wan_vae import _time_cupti  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (
    wino_norm_silu_cuda,  # noqa: E402
)

# (C, H, W, T): decoder up1 (1 pair), up2/up3 (2 pairs), encoder down0/down1.
CASES = [
    (256, 360, 640, 4),
    (512, 180, 320, 4),
    (1024, 90, 160, 2),
    (160, 360, 640, 4),
    (320, 180, 320, 2),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    def achievable_tbs(num_bytes: int) -> float:
        n = num_bytes // 4
        cal = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        gb = n * 4 / 1e9
        t_add = (
            _time_cupti(lambda: torch.add(cal, 1), args.warmup, args.iters, use_graph=True)["span"][
                "median_us"
            ]
            / 1e3
        )
        t_copy = (
            _time_cupti(lambda: cal.clone(), args.warmup, args.iters, use_graph=True)["span"][
                "median_us"
            ]
            / 1e3
        )
        return max(gb / t_add, gb / t_copy)

    print(
        f"{'case':>22s} {'mode':>8s} {'GB moved':>8s} | {'ms':>7s} {'TB/s':>6s} {'achv':>6s} {'%achv':>6s}"
    )
    rows = []
    for C, H, W, T in CASES:
        P = H * W
        gen = torch.Generator(device="cuda").manual_seed(C)
        bf = torch.bfloat16
        rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
        x = rnd(T, P, C)
        m = [rnd(T // 2, P, C) for _ in range(4)]
        residual = rnd(T, P, C)
        cache = rnd(2, P, C)
        gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
        in_bias = (0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
        frame_b = P * C * 2
        for mode in ("producer", "consumer"):
            if mode == "producer":
                fn = lambda: wino_norm_silu_cuda(x, None, None, None, cache, gamma, None, C**0.5, T)  # noqa: E731
                nbytes = frame_b * (T + 2 + 2 * T + 2)  # x, cache | D (2 per frame), y_tail
            else:
                fn = lambda: wino_norm_silu_cuda(
                    None, m, residual, in_bias, cache, gamma, None, C**0.5, T
                )  # noqa: E731
                nbytes = frame_b * (
                    2 * T + T + 2 + 2 * T + 2 + T
                )  # 4 m (= 2 frames per frame), residual, cache | D, y_tail, sum
            ms = _time_cupti(fn, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
            achv = achievable_tbs(nbytes)
            tbs = nbytes / 1e9 / ms * 1e3 / 1e3
            print(
                f"{f'C={C} {H}x{W} T={T}':>22s} {mode:>8s} {nbytes / 1e9:8.3f} | {ms:7.3f} {tbs:6.2f} {achv:6.2f} {100 * tbs / achv:5.1f}%"
            )
            rows.append(
                {
                    "C": C,
                    "H": H,
                    "W": W,
                    "T": T,
                    "mode": mode,
                    "gb": nbytes / 1e9,
                    "ms": ms,
                    "tbs": tbs,
                    "achievable_tbs": achv,
                }
            )
    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": gpu, "results": rows}, indent=1))


if __name__ == "__main__":
    main()
