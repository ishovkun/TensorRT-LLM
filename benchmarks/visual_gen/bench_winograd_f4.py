# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd F(4x4, 3x3) versus F(2x2, 3x3) on the VAE's 3x3x3 convs.

Accuracy with the real checkpoint filters at the deployed precisions (bf16 storage of the transformed data, fp32
transform arithmetic, bf16 GEMM operands with fp32 accumulation), and the cuBLAS time of the
GEMM shapes each tile size produces. No kernels: torch only, for a go/no-go.

    python bench_winograd_f4.py --vae $MODELS/Cosmos3-Nano/vae
"""

import argparse
import atexit
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_wan_vae import _time_cupti  # noqa: E402

# Lavin & Gray F(2x2, 3x3) and F(4x4, 3x3) transforms (interpolation points 0, +-1, +-2).
BT2 = torch.tensor([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=torch.float64)
G2 = torch.tensor([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=torch.float64)
AT2 = torch.tensor([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=torch.float64)
BT4 = torch.tensor(
    [
        [4, 0, -5, 0, 1, 0],
        [0, -4, -4, 1, 1, 0],
        [0, 4, -4, -1, 1, 0],
        [0, -2, -1, 2, 1, 0],
        [0, 2, -1, -2, 1, 0],
        [0, 4, 0, -5, 0, 1],
    ],
    dtype=torch.float64,
)
G4 = torch.tensor(
    [
        [1 / 4, 0, 0],
        [-1 / 6, -1 / 6, -1 / 6],
        [-1 / 6, 1 / 6, -1 / 6],
        [1 / 24, 1 / 12, 1 / 6],
        [1 / 24, -1 / 12, 1 / 6],
        [0, 0, 1],
    ],
    dtype=torch.float64,
)
AT4 = torch.tensor(
    [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]],
    dtype=torch.float64,
)

LAYERS = {
    # name: (weight key, norm gamma key, H, W) at 720p decode
    "1024@90x160": (
        "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.mid_block.resnets.0.norm1.gamma",
        90,
        160,
    ),
    "1024->512@180x320": (
        "decoder.up_blocks.2.resnets.0.conv1.weight",
        "decoder.up_blocks.2.resnets.0.norm1.gamma",
        180,
        320,
    ),
    "512@180x320": (
        "decoder.up_blocks.2.resnets.1.conv1.weight",
        "decoder.up_blocks.2.resnets.1.norm1.gamma",
        180,
        320,
    ),
    "256@360x640": (
        "decoder.up_blocks.3.resnets.1.conv1.weight",
        "decoder.up_blocks.3.resnets.1.norm1.gamma",
        360,
        640,
    ),
}


def component_filters(weight: torch.Tensor) -> torch.Tensor:
    """Temporal F(2,3) components of a [Cout, Cin, 3, 3, 3] filter -> [4, Cout, Cin, 3, 3] fp64."""
    g0, g1, g2 = (weight[:, :, i].double() for i in range(3))
    return torch.stack([g0, (g0 + g1 + g2) / 2, (g0 - g1 + g2) / 2, g2])


def winograd_conv(
    x: torch.Tensor,
    g: torch.Tensor,
    m_out: int,
    BT: torch.Tensor,
    G: torch.Tensor,
    AT: torch.Tensor,
    sdt=torch.bfloat16,
):
    """3x3 conv (padding 1) of x [N, Cin, H, W] (bf16) by g [Cout, Cin, 3, 3] (fp64).

    Spatial Winograd F(m x m, 3x3): transforms in fp32, V/U/M stored as bf16, GEMM in bf16 with fp32
    accumulation (torch.bmm), output transform in fp32, result bf16. Returns [N, Cout, H, W].
    """
    n, cin, h, w = x.shape
    cout = g.shape[0]
    a = m_out + 2  # tile side
    th, tw = -(-h // m_out), -(-w // m_out)
    hp, wp = th * m_out + 2, tw * m_out + 2
    xp = F.pad(x.float(), (1, wp - w - 1, 1, hp - h - 1))
    tiles = F.unfold(xp, kernel_size=a, stride=m_out)  # [N, Cin*a*a, T]
    t_count = tiles.shape[-1]
    d = tiles.view(n, cin, a, a, t_count)
    bt = BT.float().to(x.device)
    v = torch.einsum("xa,ncabt,yb->xyntc", bt, d, bt)  # [a, a, N, T, Cin] fp32
    v = v.reshape(a * a, n * t_count, cin).to(sdt)
    gm = G.to(x.device)
    u = torch.einsum("xa,oiab,yb->xyoi", gm, g, gm).float()  # [a, a, Cout, Cin]
    u = u.reshape(a * a, cout, cin).transpose(1, 2).contiguous().to(sdt)  # [a*a, Cin, Cout]
    m = torch.bmm(v, u)  # [a*a, N*T, Cout] in the storage dtype (fp32 accumulate)
    winograd_conv.ranges = (
        v.float().abs().max().item(),
        u.float().abs().max().item(),
        m.float().abs().max().item(),
    )
    at = AT.float().to(x.device)
    mm = m.float().view(a, a, n, t_count, cout)
    y = torch.einsum("ra,abntc,sb->ntcrs", at, mm, at)  # [N, T, Cout, m, m]
    y = y.permute(0, 2, 3, 4, 1).reshape(n, cout * m_out * m_out, t_count)
    y = F.fold(y, output_size=(th * m_out, tw * m_out), kernel_size=m_out, stride=m_out)
    return y[:, :, :h, :w].to(torch.bfloat16), v, u


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vae", required=True, help="directory with diffusion_pytorch_model.safetensors"
    )
    parser.add_argument("--frames", type=int, default=2, help="frames per component slab (pairs)")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--layers", default=",".join(LAYERS))
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()
    atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    torch.manual_seed(0)
    dev = "cuda"
    with safe_open(str(Path(args.vae) / "diffusion_pytorch_model.safetensors"), "pt") as sf:
        for name in args.layers.split(","):
            wkey, gkey, h, w = LAYERS[name]
            weight = sf.get_tensor(wkey).to(dev)
            gamma = sf.get_tensor(gkey).to(dev).float()
            cout, cin = weight.shape[:2]
            comps = component_filters(weight)  # [4, Cout, Cin, 3, 3] fp64
            # Activations: the conv input is SiLU(RMSNorm(x) * gamma + bias) in deployment; emulate
            # with Gaussian x, the checkpoint's gamma, bias 0, and bf16 storage.
            x = torch.randn(args.frames, cin, h, w, device=dev)
            x = F.silu(F.normalize(x, dim=1) * math.sqrt(cin) * gamma.view(1, -1, 1, 1)).to(
                torch.bfloat16
            )
            print(
                f"\n=== {name}: filter {tuple(weight.shape)}, slab {tuple(x.shape)}, "
                f"activation rms {x.float().pow(2).mean().sqrt().item():.3f}"
            )
            err2 = err4 = err2h = err4h = errd = 0.0
            refn = 0.0
            for k in range(4):
                g = comps[k]
                ref = F.conv2d(x.float(), g.float(), None, 1, 1)
                refn += ref.pow(2).sum().item()
                y2, v2, u2 = winograd_conv(x, g, 2, BT2, G2, AT2)
                y4, v4, u4 = winograd_conv(x, g, 4, BT4, G4, AT4)
                y2h, _, _ = winograd_conv(x, g, 2, BT2, G2, AT2, torch.float16)
                y4h, _, _ = winograd_conv(x, g, 4, BT4, G4, AT4, torch.float16)
                if k == 0:
                    rv, ru, rm = winograd_conv.ranges
                    xh = (x.float() * 8).to(torch.bfloat16)  # heavy-tailed check: activations x8
                    winograd_conv(xh, g, 4, BT4, G4, AT4, torch.float16)
                    rv8, _, rm8 = winograd_conv.ranges
                    print(
                        f"  F4 fp16 ranges: max|V| {rv:.1f}  max|U| {ru:.3f}  max|M| {rm:.1f};"
                        f"  with activations x8: max|V| {rv8:.1f} max|M| {rm8:.1f} (fp16 max 65504)"
                    )
                yd = F.conv2d(x, g.to(torch.bfloat16), None, 1, 1)
                err2 += (y2.float() - ref).pow(2).sum().item()
                err4 += (y4.float() - ref).pow(2).sum().item()
                err2h += (y2h.float() - ref).pow(2).sum().item()
                err4h += (y4h.float() - ref).pow(2).sum().item()
                errd += (yd.float() - ref).pow(2).sum().item()
            print(
                f"  rel-L2 vs fp32 conv (4 temporal components): direct bf16 cuDNN {math.sqrt(errd / refn):.3e}"
                f"   F(2x2,3x3) bf16 {math.sqrt(err2 / refn):.3e}  fp16 {math.sqrt(err2h / refn):.3e}"
                f"   F(4x4,3x3) bf16 {math.sqrt(err4 / refn):.3e}  fp16 {math.sqrt(err4h / refn):.3e}"
            )
            if args.skip_speed:
                continue
            # GEMM shapes and cuBLAS times for the four components batched together
            for label, (v, u) in (("F2", (v2, u2)), ("F4", (v4, u4))):
                vb = v.repeat(4, 1, 1)
                ub = u.repeat(4, 1, 1)
                t_us = _time_cupti(
                    lambda vb=vb, ub=ub: torch.bmm(vb, ub), 3, args.iters, use_graph=True
                )["span"]["median_us"]
                flops = 2.0 * vb.shape[0] * vb.shape[1] * vb.shape[2] * ub.shape[2]
                v_mb = vb.numel() * 2 / 1e6
                m_mb = vb.shape[0] * vb.shape[1] * ub.shape[2] * 2 / 1e6
                print(
                    f"  {label}: GEMMs {tuple(vb.shape)} x {tuple(ub.shape)}: cuBLAS {t_us / 1000:7.3f} ms"
                    f"  {flops / t_us / 1e6:6.0f} TF/s;  V {v_mb:7.1f} MB, M {m_mb:7.1f} MB,"
                    f" U {ub.numel() * 2 / 1e6:6.1f} MB"
                )
            # direct cuDNN conv2d x4 for reference
            gs = [
                comps[k].to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
                for k in range(4)
            ]
            xc = x.contiguous(memory_format=torch.channels_last)
            t_conv = _time_cupti(
                lambda: [F.conv2d(xc, gs[k], None, 1, 1) for k in range(4)],
                3,
                args.iters,
                use_graph=True,
            )["span"]["median_us"]
            print(f"  direct cuDNN conv2d x4: {t_conv / 1000:7.3f} ms")


if __name__ == "__main__":
    main()
