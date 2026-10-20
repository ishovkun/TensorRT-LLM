# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-only bandwidth benchmark for the upsample phase gather (+ DupUp3D shortcut add).

Shapes are the three decoder up blocks at 720p for one steady-state latent chunk
(temporal_chunk_size 1). Bytes moved = temp read + shortcut read + output write; the
kernel is judged against a `torch.add` moving the same number of bytes.
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

from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (  # noqa: E402
    upsample_gather_cuda,
    upsample_gather_reference,
)

# name, cout, cin (shortcut channels), H, W (low-res grid), output frames, factor_t
# repeats = cout * factor_t * 4 // cin  (DupUp3D)
CASES = [
    ("up0 720p", 1024, 1024, 45, 80, 2, 2),
    ("up1 720p", 1024, 1024, 90, 160, 4, 2),
    ("up2 720p", 512, 1024, 180, 320, 4, 1),
    ("up2 480p", 512, 1024, 120, 208, 4, 1),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    def achievable_tbs(num_bytes: int) -> float:
        n = num_bytes // 4  # torch.add(x, 1) reads n and writes n bf16 elements
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

    hdr = f"{'case':>9s} {'GB moved':>9s} | {'eager ms':>9s} {'cuda ms':>9s} {'cuda+sc ms':>10s} | {'achv TB/s':>9s} {'cuda TB/s':>9s} {'%achv':>6s} {'+sc TB/s':>9s} {'%achv':>6s}"
    print(hdr)
    results = []
    for name, cout, cin, h, w, frames, factor_t in CASES:
        repeats = cout * factor_t * 4 // cin
        gen = torch.Generator(device="cuda").manual_seed(cout + h)
        temp = torch.randn(frames, 4 * cout, h + 1, w + 1, generator=gen, device="cuda").to(
            torch.bfloat16
        )
        temp = temp.contiguous(memory_format=torch.channels_last)
        shortcut = torch.randn(1, cin, frames // factor_t, h, w, generator=gen, device="cuda").to(
            torch.bfloat16
        )
        shortcut = shortcut.contiguous(memory_format=torch.channels_last_3d)

        temp_b = temp.numel() * 2
        sc_b = shortcut.numel() * 2
        out_b = frames * cout * 4 * h * w * 2
        gb_plain = (temp_b + out_b) / 1e9
        gb_sc = (temp_b + sc_b + out_b) / 1e9

        def eager():
            return upsample_gather_reference(temp, 1, frames)

        def cuda_plain():
            return upsample_gather_cuda(temp, None, 1, frames)

        def cuda_sc():
            return upsample_gather_cuda(temp, shortcut, 1, frames, factor_t, repeats, 0)

        t_e = _time_cupti(eager, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        t_c = (
            _time_cupti(cuda_plain, args.warmup, args.iters, use_graph=True)["span"]["median_us"]
            / 1e3
        )
        t_s = (
            _time_cupti(cuda_sc, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        )
        achv_plain = achievable_tbs(temp_b + out_b)
        achv_sc = achievable_tbs(temp_b + sc_b + out_b)
        bw_c = gb_plain / t_c
        bw_s = gb_sc / t_s
        print(
            f"{name:>9s} {gb_sc:9.3f} | {t_e:9.3f} {t_c:9.3f} {t_s:10.3f} | {achv_sc:9.2f} {bw_c:9.2f} {100 * bw_c / achv_plain:5.1f}% {bw_s:9.2f} {100 * bw_s / achv_sc:5.1f}%"
        )
        results.append(
            {
                "case": name,
                "cout": cout,
                "cin": cin,
                "h": h,
                "w": w,
                "frames": frames,
                "factor_t": factor_t,
                "repeats": repeats,
                "gb_plain": gb_plain,
                "gb_shortcut": gb_sc,
                "eager_ms": t_e,
                "cuda_ms": t_c,
                "cuda_shortcut_ms": t_s,
                "achievable_plain_tbs": achv_plain,
                "achievable_shortcut_tbs": achv_sc,
                "cuda_tbs": bw_c,
                "cuda_shortcut_tbs": bw_s,
            }
        )
    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": gpu, "results": results}, indent=1))


if __name__ == "__main__":
    main()
