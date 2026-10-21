# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel Factory output (campaign wan-vae-spatial-conv, solution
# wan_conv3x3_hybrid_wino_f4_direct_cutedsl, harvested Mon 19 Oct 2026). Kept byte-identical to
# the harvest apart from this header and package-relative imports; wino_kf.py holds our glue.

import os
import sys


import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from .conv_fprop import PersistentConvKernel
from .gemm_static import PersistentDenseGemmKernel as StaticPersistentGemm
from .gemm_pref import PersistentDenseGemmKernelDynamicPreferredCluster as ClcGemm
import cutlass.utils as utils
from . import wino
from .wino import PDL_TRIGGER, PDL_WAIT_START, PDL_WAIT_END

_compiled = {}


def _config(N, H, W, Cin, Cout):
    # (mma_tiler_mn, preferred_cluster_mn, fallback_cluster_mn, use_2cta)
    return ((256, 256), (4, 1), (2, 1), True)


def _to_cute(t):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=4)


def _get_kernel(cfg, a, b, c):
    fn = _compiled.get(cfg)
    if fn is None:
        tiler, pref, fall, use_2cta = cfg
        op = PersistentConvKernel(
            cutlass.Float32,
            use_2cta,
            tiler,
            pref,
            fall,
            (1, 3, 3),
            (0, 1, 1),
            (0, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        )
        fn = cute.compile(
            op, a, b, c,
            *[cutlass.Int32(v) for v in (0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1)],
            make_fake_stream(),
        )
        _compiled[cfg] = fn
    return fn


def _wino_tile(N, H, W, Cin, Cout):
    """Winograd output tile (MH, MW), or None for the direct implicit-GEMM path."""
    if Cin >= 512 and W % 4 == 0:
        return (4, 4)
    return None


def _static(t):
    return from_dlpack(t, assumed_align=16)


def _gemm_n_tile(T):
    """GEMM N tile over Winograd tiles: least padding, ties to the larger tile."""
    return min((-(-T // n) * n, -n, n) for n in (256, 224, 192))[2]


def _wino_plan(N, H, W, TPF):
    """(nchunk, pdl) for the Winograd pipeline, from measurements. On the large shapes, frame
    chunks overlap transforms with GEMMs via PDL on a static persistent GEMM; elsewhere the
    unchunked CLC-balanced GEMM without PDL is faster."""
    return {(8, 45, 80): (2, True), (8, 90, 160): (4, True), (8, 180, 320): (4, True),
            (2, 30, 52): (1, True)}.get((N, H, W), (1, False))


def _gemm_kernel(a, b, c, n_tile, pdl, static=True):
    """The batched GEMM on the static persistent schedule (with or without the PDL release in
    the prologue), or, with ``static=False``, on the CLC dynamic persistent schedule. The
    dynamic one is opt-in for measurements only: under GPU contention it leaves tiles of a run
    unwritten (garbage in the output) in 1-10% of the launches, in every form that has a CTA
    process more than one tile (measured on GB200, 150-300 repeats per shape), while the
    static schedule stayed bit-reproducible on every shape."""
    key = ("gemm", n_tile, pdl, static)
    fn = _compiled.get(key)
    if fn is None:
        if static:
            max_clusters = utils.HardwareInfo().get_max_active_clusters(2)
            op = StaticPersistentGemm(cutlass.Float32, True, (256, n_tile), (2, 1), True,
                                      use_pdl=pdl)
            fn = cute.compile(op, a, b, c, max_clusters, make_fake_stream())
        else:
            op = ClcGemm(cutlass.Float32, True, (256, n_tile), (2, 1), (2, 1), True)
            fn = cute.compile(op, a, b, c, make_fake_stream())
        _compiled[key] = fn
    return fn


def _xform_kernel(name, jit_fn, tensors, consts):
    """Compile-once launcher for the static-layout transform kernels; the layouts are baked
    into the code, so the cache key carries the full shapes and strides."""
    key = (name,) + tuple((tuple(t.shape), t.stride()) for t in tensors) + tuple(consts)
    fn = _compiled.get(key)
    args = [_static(t) for t in tensors]
    if fn is None:
        fn = cute.compile(jit_fn, *args, *consts, make_fake_stream())
        _compiled[key] = fn
    return lambda stream: fn(*args, stream)


def _run_wino(x, weight, y, MH, MW):
    N, H, W, Cin = x.shape
    Cout = weight.shape[0]
    NC = (MH + 2) * (MW + 2)
    TPF = ((H + MH - 1) // MH) * (W // MW)  # Winograd tiles per frame
    T = N * TPF
    nchunk, pdl = _wino_plan(N, H, W, TPF)
    Nc = N // nchunk
    Tc = Nc * TPF
    dev = x.device
    U = torch.empty((NC, Cout, Cin), dtype=torch.float16, device=dev)
    V = torch.empty((NC, T, Cin), dtype=torch.float16, device=dev)
    M = torch.empty((NC, T, Cout), dtype=torch.float16, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    n_tile = _gemm_n_tile(Tc)

    def xform_in(k, flags):
        x_, v_ = x[k * Nc:(k + 1) * Nc], V[:, k * Tc:(k + 1) * Tc]
        if k == 0:  # weight transform rides along in the first input-transform grid
            _xform_kernel("wi", wino.wi_xform, [weight.view(Cout, Cin * 9), U, x_, v_],
                          (Cout, Nc, H, W, Cin, MH, MW, flags & PDL_TRIGGER))(stream)
        else:
            _xform_kernel("in", wino.in_xform, [x_, v_], (Nc, H, W, Cin, MH, MW, flags))(stream)

    def gemm(k):
        # GEMM M = Cout (multiple of 256), N = tiles (N tile picked to divide T), K = Cin
        b = from_dlpack(V[:, k * Tc:(k + 1) * Tc].permute(1, 2, 0), assumed_align=16)
        c = from_dlpack(M[:, k * Tc:(k + 1) * Tc].permute(2, 1, 0), assumed_align=16)
        b = b.mark_layout_dynamic(leading_dim=1)
        c = c.mark_layout_dynamic(leading_dim=0)
        _gemm_kernel(a, b, c, n_tile, pdl)(a, b, c, stream)

    def xform_out(k, flags):
        m_, y_ = M[:, k * Tc:(k + 1) * Tc], y[k * Nc:(k + 1) * Nc]
        _xform_kernel("out", wino.out_xform, [m_, y_], (Nc, H, W, Cout, MH, MW, flags))(stream)

    a = from_dlpack(U.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)

    # Programmatic-dependent-launch pipeline on one stream:
    #   w+in0, G0, in1, G1, out0, in2, G2, out1, ..., out_{n-1}
    # Each GEMM waits for its predecessor, then releases the next transform, which co-runs with
    # it; every transform waits on its predecessor before exiting so completion stays transitive.
    if not pdl:
        xform_in(0, 0)
        gemm(0)
        xform_out(0, 0)
        return
    for k in range(nchunk):
        xform_in(k, PDL_TRIGGER | PDL_WAIT_END)
        gemm(k)
        if k > 0:
            xform_out(k - 1, PDL_TRIGGER | PDL_WAIT_END)
    xform_out(nchunk - 1, PDL_WAIT_START)


@torch.no_grad()
def run(x, weight, y):
    N, H, W, Cin = x.shape
    Cout = weight.shape[0]
    tile = _wino_tile(N, H, W, Cin, Cout)
    if tile is not None:
        _run_wino(x, weight, y, *tile)
        return
    # filter KCRS -> KRSC (C contiguous) so TMA can stream K = (C, S, R)
    w_krsc = weight.permute(0, 2, 3, 1).contiguous().view(Cout, 1, 3, 3, Cin)
    a = _to_cute(x.view(N, 1, H, W, Cin))
    b = _to_cute(w_krsc)
    c = _to_cute(y.view(N, 1, H, W, Cout))
    cfg = _config(N, H, W, Cin, Cout)
    fn = _get_kernel(cfg, a, b, c)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(a, b, c, *[cutlass.Int32(v) for v in (0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1)], stream)
