# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused kernels for VaeBlaze. Triton prototypes; each has an exact PyTorch fallback."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_NORM_EPS = 1e-12  # F.normalize default
_MAX_TILE_ELEMENTS = 4096


def cuda_op_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_rms_norm_silu")


# (lanes per row, 16-byte vectors per lane) layouts instantiated in vaeRmsNormSiluKernel.cu.
_CUDA_LAYOUTS = {
    (32, 1),
    (32, 2),
    (32, 4),
    (4, 5),
    (8, 5),
    (16, 5),
    (2, 3),
    (4, 3),
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (1, 3),
    (32, 3),
}


def _cuda_layout(channels: int) -> tuple[int, int] | None:
    if channels % 8:
        return None
    nvec = channels // 8
    lanes = 32
    while nvec % lanes:
        lanes //= 2
    layout = (lanes, nvec // lanes)
    return layout if layout in _CUDA_LAYOUTS else None


def can_use_cuda_rms_norm_silu(x: torch.Tensor) -> bool:
    return (
        cuda_op_available()
        and x.is_cuda
        and x.dtype is torch.bfloat16
        and _cuda_layout(x.shape[1]) is not None
        and _channel_rows(x) is not None
    )


def _channel_rows(x: torch.Tensor) -> torch.Tensor | None:
    """View a channels-last NC(T)HW tensor as [positions, C] without copying."""
    if x.dim() == 5:
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            return None
        return x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
    if x.dim() == 4:
        if not x.is_contiguous(memory_format=torch.channels_last):
            return None
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return None


def can_fuse_rms_norm_silu(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dim() in (4, 5)
        and x.shape[1] <= _MAX_TILE_ELEMENTS
        and _channel_rows(x) is not None
    )


@triton.jit
def _rms_norm_silu_kernel(
    x_ptr,
    y_ptr,
    gamma_ptr,
    bias_ptr,
    n_rows,
    scale,
    eps,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    REFERENCE_ROUNDING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program normalizes BLOCK_M positions; channels are contiguous per position.
    row0 = tl.program_id(0).to(tl.int64) * BLOCK_M
    rows = row0 + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_ok = rows < n_rows
    col_ok = cols < C
    mask = row_ok[:, None] & col_ok[None, :]
    offs = rows[:, None] * C + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    sumsq = tl.sum(xf * xf, axis=1)
    denom = tl.maximum(tl.sqrt(sumsq), eps)
    n = xf / denom[:, None]
    gamma = tl.load(gamma_ptr + cols, mask=col_ok, other=0.0)
    if REFERENCE_ROUNDING:
        # Today's eager sequence: round to the activation dtype after normalize,
        # after each multiply and after the bias add; SiLU in fp32 on that value.
        n = n.to(x.dtype).to(tl.float32)
        r = (n * scale).to(x.dtype).to(tl.float32)
        r = (r * gamma.to(tl.float32)).to(x.dtype).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + cols, mask=col_ok, other=0.0)
            r = (r + b.to(tl.float32)).to(x.dtype).to(tl.float32)
    else:
        r = n * scale * gamma.to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + cols, mask=col_ok, other=0.0)
            r = r + b.to(tl.float32)
    y = r * tl.sigmoid(r)
    tl.store(y_ptr + offs, y.to(x.dtype), mask=mask)


def rms_norm_silu_cuda(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    in_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    silu: bool = True,
) -> torch.Tensor:
    """The CUDA op (``trtllm::vae_rms_norm_silu``); same contract as ``rms_norm_silu``.
    ``silu=False`` returns the RMSNorm alone (plain form only: no bias, in_bias or out).

    ``in_bias`` is a deferred convolution bias, applied to ``x`` first as a bf16 add.
    ``out`` is an optional channels-last destination of ``x``'s shape (a frame-buffer
    window); the result is written there and returned.
    """
    rows = _channel_rows(x)
    if rows is None:
        raise ValueError("rms_norm_silu_cuda needs a channels-last CUDA tensor")
    dst = None
    if out is not None:
        dst = _channel_rows(out)
        if dst is None:
            raise ValueError("out must be a channels-last CUDA tensor")
    out_rows = torch.ops.trtllm.vae_rms_norm_silu(
        rows,
        gamma.reshape(-1),
        bias.reshape(-1) if bias is not None else None,
        float(scale),
        reference_rounding,
        in_bias.reshape(-1) if in_bias is not None else None,
        dst,
        silu,
    )
    if out is not None:
        return out
    # The op returns [rows, C]; view it back as the channels-last 4D/5D tensor.
    if x.dim() == 5:
        n, c, t, h, w = x.shape
        return out_rows.view(n, t, h, w, c).permute(0, 4, 1, 2, 3)
    n, c, h, w = x.shape
    return out_rows.view(n, h, w, c).permute(0, 3, 1, 2)


def _rows_to_like(rows: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """View a [rows, C] result back as the channels-last 4D/5D tensor it came from."""
    if like.dim() == 5:
        n, c, t, h, w = like.shape
        return rows.view(n, t, h, w, c).permute(0, 4, 1, 2, 3)
    n, c, h, w = like.shape
    return rows.view(n, h, w, c).permute(0, 3, 1, 2)


def add_rms_norm_silu_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    in_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = (x + in_bias) + residual`` (bf16-rounded) and ``SiLU(RMSNorm(s))``; returns ``(s, act)``.
    ``out``: optional channels-last destination for the activation."""
    rows = _channel_rows(x)
    rrows = _channel_rows(residual)
    if rows is None or rrows is None:
        raise ValueError("add_rms_norm_silu_cuda needs channels-last CUDA tensors")
    dst = _channel_rows(out) if out is not None else None
    if out is not None and dst is None:
        raise ValueError("out must be a channels-last CUDA tensor")
    s_rows, a_rows = torch.ops.trtllm.vae_add_rms_norm_silu(
        rows,
        rrows,
        gamma.reshape(-1),
        bias.reshape(-1) if bias is not None else None,
        float(scale),
        reference_rounding,
        in_bias.reshape(-1) if in_bias is not None else None,
        dst,
    )
    return _rows_to_like(s_rows, x), (out if out is not None else _rows_to_like(a_rows, x))


def apply_in_bias(x: torch.Tensor, in_bias: torch.Tensor | None) -> torch.Tensor:
    """Eager form of the deferred convolution bias: ``x + bias`` broadcast over channels."""
    if in_bias is None:
        return x
    return x + in_bias.reshape(1, -1, *([1] * (x.dim() - 2)))


def add_rms_norm_silu_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    in_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = apply_in_bias(x, in_bias) + residual
    return s, rms_norm_silu_reference(s, gamma, bias, scale)


def rms_norm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    block_m: int | None = None,
    num_warps: int | None = None,
    enable_fp_fusion: bool | None = None,
) -> torch.Tensor:
    """SiLU(RMSNorm(x)) for a channels-last tensor, one memory pass (Triton prototype).

    ``gamma``/``bias`` are the module parameters in their stored dtype with any
    broadcast dims; ``scale`` is ``sqrt(C)``. ``reference_rounding`` reproduces
    the eager path's intermediate roundings; otherwise everything stays in fp32
    until the single store.
    """
    rows = _channel_rows(x)
    if rows is None:
        raise ValueError("rms_norm_silu needs a channels-last CUDA tensor")
    n_rows, C = rows.shape
    out = torch.empty_like(x)
    out_rows = _channel_rows(out)
    gamma_flat = gamma.reshape(-1)
    bias_flat = bias.reshape(-1) if bias is not None else gamma_flat  # dummy pointer when unused
    block_c = triton.next_power_of_2(C)
    block_m = block_m or max(1, min(64, _MAX_TILE_ELEMENTS // block_c))
    grid = (triton.cdiv(n_rows, block_m),)
    _rms_norm_silu_kernel[grid](
        rows,
        out_rows,
        gamma_flat,
        bias_flat,
        n_rows,
        float(scale),
        _NORM_EPS,
        C=C,
        HAS_BIAS=bias is not None,
        REFERENCE_ROUNDING=reference_rounding,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=num_warps or (4 if block_c <= 512 else 8),
        # FMA contraction would merge `r * gamma + b` across the intermediate rounding
        # that reference mode exists to reproduce.
        enable_fp_fusion=(not reference_rounding) if enable_fp_fusion is None else enable_fp_fusion,
    )
    return out


def rms_norm_reference(x: torch.Tensor, gamma: torch.Tensor, scale: float) -> torch.Tensor:
    """The eager WanRMSNorm.forward alone (no activation, no bias)."""
    needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    normalized = F.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
    return normalized * scale * gamma


def rms_norm_silu_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    in_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The eager path, verbatim (WanRMSNorm.forward followed by F.silu)."""
    x = apply_in_bias(x, in_bias)
    needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    normalized = F.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
    return F.silu(normalized * scale * gamma + bias)


# ---------------------------------------------------------------------------
# Temporal Winograd F(2,3): norm kernels that emit / consume the transformed frames
# ---------------------------------------------------------------------------


def cuda_wino_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_wino_norm_silu")


def winograd_weights(weight: torch.Tensor) -> list[torch.Tensor]:
    """``[Cout, Cin, 3, kh, kw]`` -> four ``[Cout, Cin, kh, kw]`` channels-last filters
    ``G0 = g0, G1 = (g0 + g1 + g2) / 2, G2 = (g0 - g1 + g2) / 2, G3 = g2`` (summed in fp32)."""
    g0, g1, g2 = (weight[:, :, i].detach().float() for i in range(3))
    gs = [g0, (g0 + g1 + g2) / 2, (g0 - g1 + g2) / 2, g2]
    return [g.to(weight.dtype).contiguous(memory_format=torch.channels_last) for g in gs]


# F(2x2, 3x3) spatial transform constants (Lavin & Gray); fp32, moved to the device once.
_WINO_G = torch.tensor(
    [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=torch.float32
)


def winograd_spatial_weights(weight: torch.Tensor) -> torch.Tensor:
    """``[Cout, Cin, 3, 3, 3]`` -> ``U [64, Cin, Cout]``: the four temporal components of
    :func:`winograd_weights`, each transformed as ``G g G^T`` (fp32, rounded once), ordered
    ``component * 16 + row * 4 + col`` to match ``trtllm::vae_wino_spatial_in``."""
    g0, g1, g2 = (weight[:, :, i].detach().float() for i in range(3))
    comps = torch.stack([g0, (g0 + g1 + g2) / 2, (g0 - g1 + g2) / 2, g2])  # [4, Cout, Cin, 3, 3]
    gm = _WINO_G.to(weight.device)
    u = torch.einsum("xa,noiab,yb->nxyoi", gm, comps, gm)  # [4, 4, 4, Cout, Cin]
    cout, cin = weight.shape[:2]
    return u.reshape(64, cout, cin).transpose(1, 2).contiguous().to(weight.dtype)


def cuda_wino_spatial_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_wino_spatial_in")


@functools.lru_cache(maxsize=None)
def cute_wino_available() -> bool:
    """The CuTe DSL fused spatial-Winograd kernel: Blackwell (sm100) and ``nvidia-cutlass-dsl`` installed."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from tensorrt_llm._torch.visual_gen.models.wan import wino_cute  # noqa: F401
    except ImportError:
        return False
    return True


def cute_spatial_filters(u_nk: torch.Tensor) -> torch.Tensor:
    """Filters for :func:`wino_spatial_conv_cute` from ``u_nk [64, Cout, Cin]``: the 256-wide
    two-pass kernel wants the y = 3 coordinate negated; the 128-wide kernel takes them as is."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_cute import winograd_pass_filters

    return winograd_pass_filters(u_nk) if u_nk.shape[1] % 256 == 0 else u_nk


def wino_spatial_conv_cute(
    d: torch.Tensor, u_cute: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Same contract as :func:`wino_spatial_conv_cuda` with ``u_cute`` from
    :func:`cute_spatial_filters`: input transform by the CUDA kernel, then the CuTe DSL kernels.
    Cout % 256 == 0: the 256x256-tile two-pass kernel (output transform finished by the last
    arriving CTA of each tile block); otherwise the 128-wide persistent kernel."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_cute import (
        wino_spatial_conv_pass2,
        wino_spatial_conv_persistent,
    )

    comps, pairs, positions, cin = d.shape
    cout = u_cute.shape[1]
    frames = d.view(comps * pairs, height, width, cin)
    tile_rows = (height + 1) // 2
    v = torch.ops.trtllm.vae_wino_spatial_in(frames, comps, 0, tile_rows)
    out = torch.empty((comps * pairs, height, width, cout), dtype=d.dtype, device=d.device)
    if cout % 256 == 0:
        wino_spatial_conv_pass2(v, u_cute, out, comps, pairs, height, width)
    else:
        wino_spatial_conv_persistent(v, u_cute, out, comps, pairs, height, width)
    return out.view(comps, pairs, positions, cout)


def wino_spatial_conv_cuda(
    d: torch.Tensor, u: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """3x3 conv (padding 1) of the four slabs by their four filters as spatial Winograd:
    ``D [4, pairs, H*W, Cin]`` -> ``M [4, pairs, H*W, Cout]`` (bias-free, fp32 accumulate).
    Input transform and output transform are CUDA kernels; the 64 GEMMs are one
    cuBLAS batched call."""
    comps, pairs, positions, cin = d.shape
    frames = d.view(comps * pairs, height, width, cin)
    tile_rows = (height + 1) // 2
    v = torch.ops.trtllm.vae_wino_spatial_in(frames, comps, 0, tile_rows)
    m = torch.bmm(v, u)
    out = torch.empty((comps * pairs, height, width, u.shape[2]), dtype=d.dtype, device=d.device)
    torch.ops.trtllm.vae_wino_spatial_out(m, out, comps, 0, tile_rows)
    return out.view(comps, pairs, positions, u.shape[2])


def frames_view(x: torch.Tensor) -> torch.Tensor:
    """Batch-1 channels-last ``[1, C, T, H, W]`` -> ``[T, H*W, C]`` view (no copy)."""
    _, c, t, h, w = x.shape
    rows = _channel_rows(x)
    if rows is None or x.shape[0] != 1:
        raise ValueError("frames_view needs a batch-1 channels-last CUDA tensor")
    return rows.view(t, h * w, c)


def frames_to_video(f: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``[T, H*W, C]`` -> channels-last ``[1, C, T, H, W]`` view."""
    t, _, c = f.shape
    return f.view(1, t, h, w, c).permute(0, 4, 1, 2, 3)


def pairs_to_images(d: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``[pairs, H*W, C]`` -> channels-last ``[pairs, C, H, W]`` view for conv2d."""
    p, _, c = d.shape
    return d.view(p, h, w, c).permute(0, 3, 1, 2)


def images_to_pairs(m: torch.Tensor) -> torch.Tensor:
    """channels-last ``[pairs, C, H, W]`` -> ``[pairs, H*W, C]`` view."""
    rows = _channel_rows(m)
    if rows is None:
        raise ValueError("images_to_pairs needs a channels-last CUDA tensor")
    return rows.view(m.shape[0], m.shape[2] * m.shape[3], m.shape[1])


def cuda_frame_ops_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_wino_combine")


def wino_combine_cuda(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    frames: int,
    height: int,
    width: int,
    pad_h: int = 0,
    pad_w: int = 0,
) -> torch.Tensor:
    """``(combine(m) | x) + bias + residual`` in frame layout ``[T, P, C]``, eager rounding order.

    With ``pad_h``/``pad_w`` the result is ``[T, (H+pad_h)*(W+pad_w), C]`` with zero rows/columns
    appended (the downsampler's ``ZeroPad2d((0, 1, 0, 1))``)."""
    return torch.ops.trtllm.vae_wino_combine(
        x,
        m if m is not None else [],
        bias.reshape(-1).contiguous() if bias is not None else None,
        residual,
        frames,
        height,
        width,
        pad_h,
        pad_w,
    )


def avg_down_add_cuda(
    x: torch.Tensor,
    r: torch.Tensor,
    out_height: int,
    out_width: int,
    in_width: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
    pad_t: int,
) -> torch.Tensor:
    """``x + AvgDown3D(r)`` in frame layout: ``x [T', P', Cout]``, ``r [T, P, Cin]``."""
    return torch.ops.trtllm.vae_avg_down_add(
        x, r, out_height, out_width, in_width, factor_t, factor_s, group_size, pad_t
    )


def interleave_bias_cuda(u: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``u [T, P, 2C]`` -> ``[2T, P, C]``: even frames from channels ``0:C``, odd from ``C:2C``, plus the bias."""
    return torch.ops.trtllm.vae_interleave_bias(
        u, bias.reshape(-1).contiguous() if bias is not None else None
    )


def interleave_bias_reference(u: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    t, p, c2 = u.shape
    c = c2 // 2
    ub = u + bias.reshape(1, 1, -1) if bias is not None else u
    return torch.stack([ub[:, :, :c], ub[:, :, c:]], dim=1).reshape(2 * t, p, c)


def wino_norm_silu_cuda(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    residual: torch.Tensor | None,
    in_bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    frames: int,
    *,
    reference_rounding: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
    """All tensors in frame layout ``[T, P, C]`` (``m``/``D`` as ``[T/2, P, C]``, cache ``[0..2, P, C]``).
    Returns ``(sum or None, y_tail [2, P, C], D [4, T/2, P, C])``; ``D[k]`` is slab k."""
    s, y_tail, d = torch.ops.trtllm.vae_wino_norm_silu(
        x,
        m,
        residual,
        in_bias.reshape(-1) if in_bias is not None else None,
        cache,
        gamma.reshape(-1),
        bias.reshape(-1) if bias is not None else None,
        float(scale),
        reference_rounding,
        frames,
    )
    return (s if s.numel() else None), y_tail, d


def wino_combine_reference(m: list[torch.Tensor]) -> torch.Tensor:
    """``m[k] [pairs, P, C]`` -> frames ``[2*pairs, P, C]``: even = (m0 + m1) + m2, odd = (m1 - m2) - m3."""
    even = (m[0] + m[1]) + m[2]
    odd = (m[1] - m[2]) - m[3]
    return torch.stack([even, odd], dim=1).reshape(2 * m[0].shape[0], *m[0].shape[1:])


def wino_norm_silu_reference(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    residual: torch.Tensor | None,
    in_bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
) -> tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
    """Eager form of the kernel, same rounding points (bf16 after every add)."""
    if x is None:
        assert m is not None
        x = wino_combine_reference(m)
    s = None
    if in_bias is not None:
        x = s = x + in_bias.reshape(1, 1, -1)
    if residual is not None:
        x = s = x + residual
    if m is not None and s is None:
        s = x
    t, p, c = x.shape
    y = rms_norm_silu_reference(x.permute(0, 2, 1), gamma.reshape(1, -1, 1), bias, scale).permute(
        0, 2, 1
    )
    cache_frames = cache.shape[0] if cache is not None else 0
    pad = torch.zeros(2 - cache_frames, p, c, dtype=x.dtype, device=x.device)
    hist = torch.cat([pad, cache], 0) if cache is not None and cache_frames else pad
    f = torch.cat([hist, y], 0)  # frames -2 .. T-1
    d = [[], [], [], []]
    for pr in range(t // 2):
        f0, f1, f2, f3 = f[2 * pr], f[2 * pr + 1], f[2 * pr + 2], f[2 * pr + 3]
        for k, v in enumerate((f0 - f2, f1 + f2, f2 - f1, f1 - f3)):
            d[k].append(v)
    return s, y[-2:].contiguous(), [torch.stack(dk) for dk in d]


# ---------------------------------------------------------------------------
# Nearest-2x upsample + 3x3 conv as one 2x2 conv on the low-res grid + phase gather
# ---------------------------------------------------------------------------


def cuda_upsample_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_upsample_gather")


def can_use_cuda_upsample_gather(cout: int, factor_t: int, repeats: int) -> bool:
    """Mirror of the kernel's instantiation table."""
    return (
        cuda_upsample_available()
        and cout % 8 == 0
        and factor_t in (1, 2)
        and 1 <= repeats <= 32
        and repeats & (repeats - 1) == 0
    )


def pack_upsample_weights(
    weight: torch.Tensor, bias: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold nearest-2x upsampling into a 3x3 conv's weights.

    ``weight`` is ``[Cout, Cin, 3, 3]``. Returns ``[4*Cout, Cin, 2, 2]`` with output
    phase ``2p + q`` in channel block ``p*2+q``: with the conv run at ``padding=1``,
    temp position ``j`` covers low-res rows ``(j-1, j)`` and phase ``p`` of output ``i``
    reads temp row ``i + p``, so original tap ``dy`` lands on ``ky = (p+dy)//2 - p + 1``.
    Composed in fp32, cast once; the bias is replicated per phase.
    """
    cout, cin = weight.shape[:2]
    w = weight.detach().float()
    packed = w.new_zeros(4 * cout, cin, 2, 2)
    for p in (0, 1):
        for q in (0, 1):
            block = packed[(2 * p + q) * cout : (2 * p + q + 1) * cout]
            for dy in (-1, 0, 1):
                ky = (p + dy) // 2 - p + 1
                for dx in (-1, 0, 1):
                    kx = (q + dx) // 2 - q + 1
                    block[:, :, ky, kx] += w[:, :, dy + 1, dx + 1]
    packed = packed.to(weight.dtype).contiguous(memory_format=torch.channels_last)
    packed_bias = bias.detach().repeat(4).contiguous() if bias is not None else None
    return packed, packed_bias


def upsample_gather_reference(
    temp: torch.Tensor,
    batch: int,
    frames: int,
    shortcut_expanded: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager phase gather: ``out[.., 2i+p, 2j+q] = temp[.., phase, i+p, j+q]`` (+ bias, + shortcut)."""
    bt, c4, hp1, wp1 = temp.shape
    cout, h, w = c4 // 4, hp1 - 1, wp1 - 1
    t4 = temp.view(bt, 4, cout, hp1, wp1)
    out = temp.new_empty(bt, cout, 2 * h, 2 * w)
    for p in (0, 1):
        for q in (0, 1):
            out[:, :, p::2, q::2] = t4[:, 2 * p + q, :, p : p + h, q : q + w]
    out = out.view(batch, frames, cout, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
    out = apply_in_bias(out, bias)
    if shortcut_expanded is not None:
        out = out + shortcut_expanded
    return out.contiguous(memory_format=torch.channels_last_3d)


def upsample_gather_cuda(
    temp: torch.Tensor,
    shortcut: torch.Tensor | None,
    batch: int,
    frames: int,
    factor_t: int = 1,
    repeats: int = 1,
    crop: int = 0,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA phase gather with the conv bias and the DupUp3D shortcut add folded in."""
    return torch.ops.trtllm.vae_upsample_gather(
        temp,
        shortcut,
        batch,
        frames,
        factor_t,
        repeats,
        crop,
        bias.contiguous() if bias is not None else None,
    )
