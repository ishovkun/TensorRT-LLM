# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-only bandwidth benchmark for the fused RMSNorm+SiLU (Triton vs CUDA op vs eager).

Large channels-last tensors at the Wan VAE channel counts; CUPTI-timed graph replays
with an L2 flush per replay (same helper as bench_wan_vae.py). Reports achieved
GB/s = (read + write bytes) / kernel span.
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

from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (  # noqa: E402
    cuda_op_available,
    rms_norm_silu,
    rms_norm_silu_cuda,
    rms_norm_silu_reference,
)

# (channels, positions): decoder level 3 at 720p (4 frames of 360x640), level 2, level 1,
# encoder level 0/1/2 and the 1024-wide low-res blocks.
CASES = [
    (256, 4 * 360 * 640),
    (512, 4 * 180 * 320),
    (1024, 2 * 90 * 160),
    (160, 4 * 360 * 640),
    (320, 4 * 180 * 320),
    (640, 2 * 90 * 160),
    (48, 4 * 360 * 640),
    (96, 4 * 360 * 640),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--peak-tbs", type=float, default=8.0, help="HBM peak for the utilization column")
    parser.add_argument("--out", default=None, help="optional JSON path")
    parser.add_argument("--quick", action="store_true", help="only the two heavy decoder shapes, fewer iterations")
    args = parser.parse_args()
    cases = CASES[:2] if args.quick else CASES
    if args.quick:
        args.iters = min(args.iters, 10)

    atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  cuda op: {cuda_op_available()}")
    # Calibration per case: the read+write bandwidth PyTorch's own elementwise kernel
    # reaches on a tensor of exactly that case's size. Small tensors cannot reach the
    # large-tensor number (launch and tail are a bigger share), so each row is judged
    # against a copy of its own size.
    def achievable_tbs(num_elements: int) -> float:
        cal = torch.empty(num_elements, device="cuda", dtype=torch.bfloat16)
        gb = num_elements * 2 * 2 / 1e9
        t_add = _time_cupti(lambda: torch.add(cal, 1), args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        t_copy = _time_cupti(lambda: cal.clone(), args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        return max(gb / t_add, gb / t_copy)

    big = achievable_tbs(CASES[0][0] * CASES[0][1])
    print(f"calibration (largest case, {CASES[0][0] * CASES[0][1] * 4 / 1e9:.3f} GB r+w): {big:.2f} TB/s  (nominal peak {args.peak_tbs} TB/s)")
    print(f"{'C':>5s} {'rows':>9s} {'GB moved':>9s} | {'eager ms':>9s} {'triton ms':>9s} {'cuda ms':>9s} | {'achv TB/s':>9s} {'triton TB/s':>11s} {'cuda TB/s':>9s} {'cuda %achv':>10s}")
    results = []
    for C, rows in cases:
        gen = torch.Generator(device="cuda").manual_seed(C)
        x = torch.randn(1, C, 4, rows // 4 // 640 if rows % (4 * 640) == 0 else 1, 640 if rows % (4 * 640) == 0 else rows // 4, generator=gen, device="cuda")
        x = x.to(torch.bfloat16).to(memory_format=torch.channels_last_3d)
        assert x.shape[2] * x.shape[3] * x.shape[4] == rows
        gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(torch.bfloat16)
        scale = C**0.5
        gb = rows * C * 2 * 2 / 1e9  # bf16 in + bf16 out

        def eager():
            return rms_norm_silu_reference(x, gamma, 0.0, scale)

        def triton():
            return rms_norm_silu(x, gamma, None, scale)

        def cuda():
            return rms_norm_silu_cuda(x, gamma, None, scale)

        t_e = _time_cupti(eager, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        t_t = _time_cupti(triton, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3
        t_c = _time_cupti(cuda, args.warmup, args.iters, use_graph=True)["span"]["median_us"] / 1e3 if cuda_op_available() else float("nan")
        bw_t = gb / t_t
        bw_c = gb / t_c
        achievable = achievable_tbs(rows * C)
        print(f"{C:5d} {rows:9d} {gb:9.3f} | {t_e:9.3f} {t_t:9.3f} {t_c:9.3f} | {achievable:9.2f} {bw_t:11.2f} {bw_c:9.2f} {100 * bw_c / achievable:9.1f}%")
        results.append({"C": C, "rows": rows, "gb": gb, "eager_ms": t_e, "triton_ms": t_t, "cuda_ms": t_c, "achievable_tbs": achievable, "triton_tbs": bw_t, "cuda_tbs": bw_c})
    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": gpu, "achievable_tbs_largest": big, "results": results}, indent=1))


if __name__ == "__main__":
    main()
