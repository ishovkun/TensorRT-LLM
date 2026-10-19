# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel Factory output (campaign wan-vae-spatial-conv, solution
# wan_conv3x3_hybrid_wino_f4_direct_cutedsl, harvested Mon 19 Oct 2026). Kept byte-identical to
# the harvest apart from this header and package-relative imports; wino_kf.py holds our glue.

# Wan2.2 VAE spatial 3x3 conv (pad 1, stride 1), channels-last bf16, CuTe DSL on sm_100a.
# Per-shape dispatch between two pipelines:
#   wa: Winograd F(4x4,3x3), fp16 transformed data, M = tiles GEMM (2-CTA 256x256), 2-chunk PDL
#       overlap of transforms with the GEMM -- best when the tile count fills 256-row M tiles.
#   wb: Winograd F(4x4,3x3) with M = Cout and an N tile (256/224/192) chosen to divide the tile
#       count -- best for few tiles (30x52, 60x104 x4) and Cin != Cout.
#   direct (Cin <= 256): CuTe filter permute + im2col-TMA implicit GEMM, 4x1 preferred clusters
#       (filter tile multicast across two CTA pairs) with 2x1 fallback.
import os
import sys
import torch
import cutlass
import cuda.bindings.driver as cuda
from . import wa
from . import wb

SMALL_TILES = 1900  # below this many 4x4 tiles, M = tiles pads too much -> wb


def _direct(x, weight, y):
    N, H, W, Ci = x.shape
    Co = weight.shape[0]
    wp = torch.empty((Co, 3, 3, Ci), dtype=torch.bfloat16, device=x.device)
    wa._get(("pw", Co, Ci // 2, None))(weight.view(-1).view(torch.int32),
                                       wp.view(-1).view(torch.int32))
    a = wb._to_cute(x.view(N, 1, H, W, Ci))
    b = wb._to_cute(wp.view(Co, 1, 3, 3, Ci))
    c = wb._to_cute(y.view(N, 1, H, W, Co))
    fn = wb._get_kernel(wb._config(N, H, W, Ci, Co), a, b, c)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(a, b, c, *[cutlass.Int32(v) for v in (0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1)], stream)


@torch.no_grad()
def run(x, weight, y):
    N, H, W, Ci = x.shape
    Co = weight.shape[0]
    if Ci <= 256:
        _direct(x, weight, y)
        return
    tiles = N * ((H + 3) // 4) * ((W + 3) // 4)
    if Ci != Co or tiles < SMALL_TILES or W % 4:
        wb.run(x, weight, y)
    else:
        wa.run(x, weight, y)
