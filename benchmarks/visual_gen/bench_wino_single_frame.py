# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-frame causal 3x3x3 conv: cuDNN versus spatial Winograd alone.

Two calls a streaming decoder makes at the low-resolution levels: the first frame of a clip
(zero history; cuDNN conv2d on the current-frame taps) and a steady-state frame with its
two cached predecessors (cuDNN conv3d on the 3-frame window). Spatial Winograd sums the
taps inside the GEMM (K = taps * Cin). CUPTI kernel time, L2 flushed per iteration.

    python bench_wino_single_frame.py --cases 1024,1024,90,160 1024,1024,180,320
"""

import argparse
import atexit
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_wan_vae import _time_cupti  # noqa: E402

import tensorrt_llm  # noqa: E402,F401  (trtllm::vae_wino_spatial_in)
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (  # noqa: E402
    wino_spatial_conv_frames,
    winograd_spatial_taps_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["1024,1024,90,160", "1024,1024,180,320", "512,512,180,320", "1024,1024,64,64"],
        help="cin,cout,h,w per case",
    )
    parser.add_argument("--tiles", default="2,4")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device="cuda").manual_seed(0)
    for case in args.cases:
        cin, cout, h, w = (int(x) for x in case.split(","))
        weight = (0.05 * torch.randn(cout, cin, 3, 3, 3, generator=gen, device="cuda")).to(
            torch.bfloat16
        )
        frames = torch.randn(3, h * w, cin, generator=gen, device="cuda").to(torch.bfloat16)
        x3 = frames.view(3, h, w, cin).permute(3, 0, 1, 2).unsqueeze(0)  # [1, Cin, 3, H, W]
        x3 = x3.contiguous(memory_format=torch.channels_last_3d)
        x1 = frames[2:].view(1, h, w, cin).permute(0, 3, 1, 2)
        x1 = x1.contiguous(memory_format=torch.channels_last)
        w2d = weight[:, :, 2].contiguous(memory_format=torch.channels_last)
        print(f"\n=== {cin}->{cout} @ {h}x{w} ===")
        t_conv3d = _time_cupti(
            lambda: F.conv3d(x3, weight, None, 1, (0, 1, 1)), 3, args.iters, use_graph=True
        )["span"]["median_us"]
        t_conv2d = _time_cupti(
            lambda: F.conv2d(x1, w2d, None, 1, 1), 3, args.iters, use_graph=True
        )["span"]["median_us"]
        ref3 = F.conv3d(x3.float(), weight.float(), None, 1, (0, 1, 1))[0, :, 0]
        ref3 = ref3.permute(1, 2, 0).reshape(h * w, cout)
        print(f"  cached frame (3 taps): cuDNN conv3d {t_conv3d / 1e3:8.3f} ms")
        print(f"  first frame  (1 tap) : cuDNN conv2d {t_conv2d / 1e3:8.3f} ms")
        for tile in (int(t) for t in args.tiles.split(",")):
            u3 = winograd_spatial_taps_weights(weight, tile, 3)
            u1 = winograd_spatial_taps_weights(weight, tile, 1)
            got = wino_spatial_conv_frames(frames, u3, h, w).float()
            rel = ((got - ref3).norm() / ref3.norm()).item()
            t3 = _time_cupti(
                lambda: wino_spatial_conv_frames(frames, u3, h, w), 3, args.iters, use_graph=True
            )["span"]["median_us"]
            t1 = _time_cupti(
                lambda: wino_spatial_conv_frames(frames[2:], u1, h, w),
                3,
                args.iters,
                use_graph=True,
            )["span"]["median_us"]
            print(
                f"  F({tile}x{tile},3x3) spatial-only: cached frame {t3 / 1e3:8.3f} ms"
                f" ({t_conv3d / t3:.2f}x), first frame {t1 / 1e3:8.3f} ms ({t_conv2d / t1:.2f}x);"
                f" rel-L2 vs fp32 {rel:.2e}"
            )


if __name__ == "__main__":
    main()
