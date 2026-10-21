# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd F(4x4, 3x3) for the Winograd pair path on Blackwell (sm100), on the Kernel
Factory CuTe DSL kernels in ``kf_wino``: input transform, 36 batched tcgen05 GEMMs (fp16 in,
fp32 accumulate, fp16 out), output transform, the transforms co-running with the GEMMs through
programmatic dependent launch (PDL). Interpolation points (0, 1, -1, 2, -1/2, inf).

A conv's four temporal components share one call: the frames of all components are one batch
for the transforms, and the GEMMs run as four chunks (one per component filter) so chunk k+1's
input transform and chunk k-1's output transform overlap GEMM k. Filters are transformed once
at load time by :func:`kf_spatial_filters`.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def kf_wino_available() -> bool:
    """Blackwell (sm100) and ``nvidia-cutlass-dsl`` installed."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from tensorrt_llm._torch.visual_gen.models.wan.kf_wino import wa, wb  # noqa: F401
    except ImportError:
        return False
    return True


def kf_spatial_filters(g: list[torch.Tensor]) -> torch.Tensor:
    """``U [36, components, Cout, Cin]`` fp16: ``G g G^T`` per component filter ``g[k]
    [Cout, Cin, 3, 3]`` (fp32 math, rounded once), batch index ``coord * components + k``."""
    from tensorrt_llm._torch.visual_gen.models.wan.kf_wino import wa

    comps = len(g)
    cout, cin = g[0].shape[:2]
    gmat = torch.tensor(wa.G, dtype=torch.float32, device=g[0].device)
    u = torch.einsum("ia,kocab,jb->ijkoc", gmat, torch.stack(g).float(), gmat)
    return u.reshape(36, comps, cout, cin).to(torch.float16).contiguous()


def kf_eligible(cin: int, cout: int) -> bool:
    """Channel counts the GEMMs and transforms accept (K tiles of 64, 256-row Cout tiles)."""
    return cin % 64 == 0 and cout % 256 == 0


def _pick_mode(pairs: int, height: int, width: int, cin: int) -> str:
    """One GEMM over all components, no PDL. Measured on GB200 (pair path, four components):
    the PDL-pipelined chunks are 10-20% faster per conv in isolation, but the decoder runs at
    the board's power cap, where the overlap only lowers the clock (720p decode 2066 ms
    pipelined vs 2033 ms single GEMM vs 2045 ms cuBLAS); the single GEMM also launches the
    fewest kernels. The chunked forms stay selectable through ``mode``."""
    return "wb1"


def wino_spatial_conv_kf(
    d: torch.Tensor, u: torch.Tensor, height: int, width: int, mode: str | None = None
) -> torch.Tensor:
    """3x3 conv (padding 1) of the component slabs by their filters: ``D [components, pairs,
    H*W, Cin]`` (bf16) -> ``M [components, pairs, H*W, Cout]`` (bias-free), ``u`` from
    :func:`kf_spatial_filters`. ``mode``: "wa" = GEMM rows are tiles (256x256 2-CTA tiles), "wb" =
    GEMM rows are output channels with the tile count split into 256/224/192-wide column
    tiles, both as four per-component chunks pipelined under PDL; "wa0" / "wb0" = the same
    chunks as plain serialized launches; "wb1" = "wb" layout as one GEMM over all components
    without PDL (the default); "wb1d" = the same on the CLC dynamic schedule (not reproducible
    under contention, measurements only); every other form runs the static persistent GEMM;
    ``None`` picks by shape."""
    comps, pairs, positions, cin = d.shape
    cout = u.shape[2]
    x = d.view(comps * pairs, height, width, cin)
    y = torch.empty((comps * pairs, height, width, cout), dtype=d.dtype, device=d.device)
    mode = mode or _pick_mode(pairs, height, width, cin)
    if mode in ("wa", "wa0"):
        if d.dtype != torch.bfloat16:
            raise ValueError("the tile-row layouts ('wa', 'wa0') read bf16 frames only")
        _run_wa(x, u, y, comps, pairs, pdl=mode == "wa")
    elif mode in ("wb", "wb0"):
        _run_wb(x, u, y, comps, pairs, chunked=True, pdl=mode == "wb")
    elif mode == "wb1":
        _run_wb(x, u, y, comps, pairs, chunked=False, pdl=False)
    elif mode == "wb1d":
        _run_wb(x, u, y, comps, pairs, chunked=False, pdl=False, dynamic=True)
    else:
        raise ValueError(f"unknown spatial Winograd mode {mode!r}")
    return y.view(comps, pairs, positions, cout)


def _run_wa(
    x: torch.Tensor, u: torch.Tensor, y: torch.Tensor, comps: int, pairs: int, pdl: bool
) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.kf_wino import wa

    _, height, width, cin = x.shape
    cout = u.shape[2]
    th, tw = -(-height // 4), -(-width // 4)
    tiles = pairs * th * tw
    ci2, co2 = cin // 2, cout // 2
    if (tiles * ci2) % 128 or (tiles * co2) % 128:
        raise ValueError("the tile-row transforms need tiles * channels / 2 divisible by 128")
    dev = x.device
    v = [torch.empty((36, tiles, cin), dtype=torch.float16, device=dev) for _ in range(comps)]
    m = [torch.empty((36, tiles, cout), dtype=torch.float16, device=dev) for _ in range(comps)]
    xs = [x[k * pairs : (k + 1) * pairs].reshape(-1).view(torch.int32) for k in range(comps)]
    ys = [y[k * pairs : (k + 1) * pairs].reshape(-1).view(torch.int32) for k in range(comps)]
    cluster = (
        wa.GEMM_CLUSTER_SMALL_U if cin * cout <= 512 * 512 else wa.GEMM_CFG["cluster_shape_mn"]
    )
    gemm = wa._get(("gemm", 256, pdl, False, cluster))
    first, mid, last = (wa._FIRST, wa._OVERLAP, wa._LAST) if pdl else (None, None, None)
    in_first = wa._get(("it", height, width, ci2, th, tw, tiles, first))
    in_mid = wa._get(("it", height, width, ci2, th, tw, tiles, mid))
    out_mid = wa._get(("ot", height, width, co2, th, tw, tiles, mid))  # codespell:ignore ot
    out_last = wa._get(("ot", height, width, co2, th, tw, tiles, last))  # codespell:ignore ot
    # it0, G0, it1, G1, ot0, it2, G2, ot1, ..., ot_{n-1}: the first kernel waits for the
    # producer of x, every other transform overlaps the GEMM launched before it.
    for k in range(comps):
        (in_first if k == 0 else in_mid)(xs[k], v[k].view(-1).view(torch.int32))
        gemm(v[k], u[:, k], m[k])
        if k > 0:
            out_mid(m[k - 1].view(-1).view(torch.int32), ys[k - 1])
    out_last(m[comps - 1].view(-1).view(torch.int32), ys[comps - 1])


_xform_compiled: dict = {}


def _xform_kernel(name: str, jit_fn, tensors: list[torch.Tensor], consts: tuple):
    """Compile-once launcher for the static-layout transform kernels (the layouts and the
    element types are baked into the code, so the key carries shapes, strides and dtypes)."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    key = (name, *((tuple(t.shape), t.stride(), t.dtype) for t in tensors), *consts)
    args = [from_dlpack(t, assumed_align=16) for t in tensors]
    fn = _xform_compiled.get(key)
    if fn is None:
        fn = cute.compile(jit_fn, *args, *consts, make_fake_stream())
        _xform_compiled[key] = fn
    return lambda stream: fn(*args, stream)


def _run_wb(
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    comps: int,
    pairs: int,
    chunked: bool,
    pdl: bool,
    dynamic: bool = False,
) -> None:
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch.visual_gen.models.wan.kf_wino import wb, wino
    from tensorrt_llm._torch.visual_gen.models.wan.kf_wino.wino import (
        PDL_TRIGGER,
        PDL_WAIT_END,
        PDL_WAIT_START,
    )

    frames, height, width, cin = x.shape
    cout = u.shape[2]
    tiles = pairs * -(-height // 4) * (width // 4)
    dev = x.device
    v = torch.empty((36, comps * tiles, cin), dtype=torch.float16, device=dev)
    m = torch.empty((36, comps * tiles, cout), dtype=torch.float16, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    n_tile = wb._gemm_n_tile(tiles)

    def xform_in(k: int, flags: int) -> None:
        x_, v_ = x[k * pairs : (k + 1) * pairs], v[:, k * tiles : (k + 1) * tiles]
        _xform_kernel("in", wino.in_xform, [x_, v_], (pairs, height, width, cin, 4, 4, flags))(
            stream
        )

    def xform_out(k: int, flags: int) -> None:
        m_, y_ = m[:, k * tiles : (k + 1) * tiles], y[k * pairs : (k + 1) * pairs]
        _xform_kernel("out", wino.out_xform, [m_, y_], (pairs, height, width, cout, 4, 4, flags))(
            stream
        )

    def gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, pdl: bool, static=True) -> None:
        # M = Cout, N = tiles, K = Cin, batched over the 36 coordinates (x components)
        a = from_dlpack(a.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        b = from_dlpack(b.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        c = from_dlpack(c.permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0)
        wb._gemm_kernel(a, b, c, n_tile, pdl, static)(a, b, c, stream)

    if not chunked:
        # One GEMM over all components; the dynamic schedule only on request (see
        # wb._gemm_kernel).
        _xform_kernel("in", wino.in_xform, [x, v], (frames, height, width, cin, 4, 4, 0))(stream)
        gemm(
            u.view(36 * comps, cout, cin),
            v.view(36 * comps, tiles, cin),
            m.view(36 * comps, tiles, cout),
            False,
            static=not dynamic,
        )
        _xform_kernel("out", wino.out_xform, [m, y], (frames, height, width, cout, 4, 4, 0))(stream)
        return
    if not pdl:
        for k in range(comps):
            xform_in(k, 0)
            gemm(
                u[:, k], v[:, k * tiles : (k + 1) * tiles], m[:, k * tiles : (k + 1) * tiles], False
            )
            xform_out(k, 0)
        return
    # in0, G0, in1, G1, out0, in2, G2, out1, ..., out_{n-1} under PDL (see _run_wa).
    for k in range(comps):
        xform_in(k, (PDL_WAIT_START | PDL_TRIGGER) if k == 0 else (PDL_TRIGGER | PDL_WAIT_END))
        gemm(u[:, k], v[:, k * tiles : (k + 1) * tiles], m[:, k * tiles : (k + 1) * tiles], True)
        if k > 0:
            xform_out(k - 1, PDL_TRIGGER | PDL_WAIT_END)
    xform_out(comps - 1, PDL_WAIT_START)


def fused_eligible(cin: int, cout: int, height: int, width: int) -> bool:
    """Shapes the fused 1D-Winograd conv (``kf_wino.fused256``) covers: 256 output channels,
    input channels in multiples of 32, 8-row and 32-pixel tiles."""
    return cout == 256 and cin % 32 == 0 and height % 8 == 0 and width % 32 == 0


def fused_filters(g: list[torch.Tensor]) -> list[torch.Tensor]:
    """Transformed filters for :func:`wino_spatial_conv_fused`, one per component filter
    ``g[k] [Cout, Cin, 3, 3]``."""
    from tensorrt_llm._torch.visual_gen.models.wan.kf_wino import fused256

    return [fused256.prepare(gk.contiguous().to(torch.bfloat16)) for gk in g]


def wino_spatial_conv_fused(
    d: torch.Tensor, filters: list[torch.Tensor], height: int, width: int
) -> torch.Tensor:
    """The fused 1D-Winograd 3x3 conv per component (Kernel Factory campaign 2): ``D [components,
    pairs, H*W, Cin]`` (bf16 or fp16) -> ``M [components, pairs, H*W, 256]`` in ``D``'s dtype,
    one stacked buffer (the consumer norm kernel reads the components at one stride)."""
    from tensorrt_llm._torch.visual_gen.models.wan.kf_wino import fused256

    comps, pairs, positions, cin = d.shape
    cout = filters[0].shape[0]
    y = torch.empty((comps, pairs, height, width, cout), dtype=d.dtype, device=d.device)
    x = d.view(comps, pairs, height, width, cin)
    for k in range(comps):
        fused256.run_prepared(x[k], filters[k], y[k])
    return y.view(comps, pairs, positions, cout)
