# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel Factory output (campaign wan-vae-spatial-conv, solution
# wan_conv3x3_hybrid_wino_f4_direct_cutedsl, harvested Mon 19 Oct 2026). Kept byte-identical to
# the harvest apart from this header and package-relative imports; wino_kf.py holds our glue.

# Wan2.2 VAE spatial 3x3 conv (pad 1, stride 1): Winograd F(4x4, 3x3) in CuTe DSL on sm_100a.
#   1. weight transform  U[c][co][ci] = (G g G^T)[c]               (fp16)
#   2. input transform   V[c][p][ci]  = (B^T d B)[c] per 4x4 tile p (fp16)
#   3. 36 batched GEMMs  M[c][p][co]  = sum_ci V[c][p][ci] U[c][co][ci]  (tcgen05, fp32 acc, fp16 out)
#   4. output transform  y[n,h,w,co]  = (A^T M A) (bf16)
# Interpolation points (0, +-3/2, +-5/8, inf): the real point set with the lowest fp16 rounding
# sensitivity (1.7e-3 per conv; (0, +-1, 2, -1/2, inf) gives 2.5e-3, the canonical (0, +-1, +-2) 3.8e-3).
from typing import Optional, Tuple, Type, Union
import math
import os

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass import Int32, Float32, Float16, BFloat16
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

BT = [[225/256, 0, -169/64, 0, 1, 0], [0, -75/128, -25/64, 3/2, 1, 0], [0, 75/128, -25/64, -3/2, 1, 0],
      [0, -45/32, -9/4, 5/8, 1, 0], [0, 45/32, -9/4, -5/8, 1, 0], [0, 225/256, 0, -169/64, 0, 1]]
G = [[256/225, 0, 0], [128/1071, 64/357, 32/119], [128/1071, -64/357, 32/119],
     [-2048/2975, -256/595, -32/119], [-2048/2975, 256/595, -32/119], [0, 0, 1]]
AT = [[1, 1, 1, 1, 1, 0], [0, 3/2, -3/2, 5/8, -5/8, 0], [0, 9/4, 9/4, 25/64, 25/64, 0],
      [0, 27/8, -27/8, 125/512, -125/512, 1]]

TPB = 128  # threads per block for the transform kernels (all workload sizes divide evenly)


# ---------------------------------------------------------------------------------------------
# Trace-time helpers (plain Python: loops unroll while tracing). Each transform thread owns two
# adjacent channels, carried as an (f32, f32) pair so every op is one packed fma.rn.f32x2.
# ---------------------------------------------------------------------------------------------
@dsl_user_op
def _cvt_pack(lo: Float32, hi: Float32, op: cutlass.Constexpr, *, loc=None, ip=None) -> Int32:
    r = llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        f"cvt.rn.{op}.f32 $0, $1, $2;", "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(r)


def _bf16x2_to_f32x2(v):
    return (v << 16).bitcast(Float32), (v & Int32(-65536)).bitcast(Float32)


def _f16x2_to_f32x2(v):
    f = cutlass.Vector.from_elements((v,), dtype=Int32).bitcast(Float16).to(Float32)
    return f[0], f[1]


def _lin2(coefs, vals):
    acc = None
    for c, v in zip(coefs, vals):
        if c == 0:
            continue
        cc = (Float32(c), Float32(c))
        if acc is None:
            acc = v if c == 1 else cute.arch.mul_packed_f32x2(v, cc)
        elif c == 1:
            acc = cute.arch.add_packed_f32x2(acc, v)
        elif c == -1:
            acc = cute.arch.sub_packed_f32x2(acc, v)
        else:
            acc = cute.arch.fma_packed_f32x2(v, cc, acc)
    return acc


def _sandwich(L, d, R):
    # L (a x k) . d (k x l) . R^T (b x l)  ->  a x b, entries are f32 pairs
    t = [[_lin2(R[b], d[k]) for b in range(len(R))] for k in range(len(d))]
    return [[_lin2(L[a], [t[k][b] for k in range(len(d))]) for b in range(len(R))]
            for a in range(len(L))]


# Factorized F(4x4) transforms on (f32, f32) pairs: B^T in 14 ops (vs 30 for the dense rows),
# A^T in 13 (vs 26). The +-p rows share an even part (d4 - p^2 d2) and an odd part; plain fp32
# re-association of the same sums.
def _padd(a, b):
    return cute.arch.add_packed_f32x2(a, b)


def _psub(a, b):
    return cute.arch.sub_packed_f32x2(a, b)


def _pfma(a, c, b):  # a * c + b
    return cute.arch.fma_packed_f32x2(a, (Float32(c), Float32(c)), b)


def _bt_vec(d):
    d0, d1, d2, d3, d4, d5 = d
    e1 = _pfma(d2, -0.390625, d4)   # d4 - (5/8)^2 d2
    e2 = _pfma(d2, -2.25, d4)       # d4 - (3/2)^2 d2
    return [_pfma(d0, 0.87890625, _pfma(d2, -2.640625, d4)),
            _pfma(d3, 1.5, _pfma(d1, -0.5859375, e1)),
            _pfma(d3, -1.5, _pfma(d1, 0.5859375, e1)),
            _pfma(d3, 0.625, _pfma(d1, -1.40625, e2)),
            _pfma(d3, -0.625, _pfma(d1, 1.40625, e2)),
            _pfma(d1, 0.87890625, _pfma(d3, -2.640625, d5))]


def _at_vec(m):
    m0, m1, m2, m3, m4, m5 = m
    p, q, r, s = _padd(m1, m2), _psub(m1, m2), _padd(m3, m4), _psub(m3, m4)
    z = _psub(m0, m0)
    return [_padd(_padd(m0, p), r),
            _pfma(q, 1.5, _pfma(s, 0.625, z)),
            _pfma(p, 2.25, _pfma(r, 0.390625, z)),
            _pfma(q, 3.375, _pfma(s, 0.244140625, m5))]


def _sandwich_f(fn, d):
    # fn applied along rows, then along columns: out[a][b] = (T d T^T)[a][b]
    t = [fn(row) for row in d]
    cols = [fn([t[k][b] for k in range(len(d))]) for b in range(len(t[0]))]
    return [[cols[b][a] for b in range(len(cols))] for a in range(len(cols[0]))]


# PDL protocol, pdl = None (plain launch) or (wait_at_start, release_at_start, wait_at_end).
# A kernel that reads its predecessor's output waits at start; one that does not may skip that
# wait to overlap the predecessor, but then waits at its end so completion order stays transitive.
def _pdl_start(pdl):
    if pdl is not None:
        if pdl[0]:
            cute.arch.griddepcontrol_wait()
        if pdl[1]:
            cute.arch.griddepcontrol_launch_dependents()


def _wait_end(pdl):
    return pdl is not None and pdl[2]


def _valid(cond_needed, cond):
    # 1 (python const) when the bound can never be violated, else a runtime 0/1
    return Int32(cond) if cond_needed else 1


# Kernel bodies live in plain helpers so Python loops unroll at trace time (a `for` written in a
# @cute.kernel body would be staged as a runtime loop). Shapes are trace-time constants.
def _wt_body(wi, ui, gid, Co, C2):
    base = gid * 9
    vals = []
    for j in range(9):
        lo, hi = _bf16x2_to_f32x2(wi[base + j])
        vals += [lo, hi]
    # pair the two input channels: element (ci0, k) with (ci0 + 1, k)
    g = [[(vals[i * 3 + j], vals[9 + i * 3 + j]) for j in range(3)] for i in range(3)]
    u = _sandwich(G, g, G)
    for a in range(6):
        for b in range(6):
            ui[(a * 6 + b) * (Co * C2) + gid] = _cvt_pack(u[a][b][0], u[a][b][1], "f16x2")


def _it_body(xi, vi, gid, H, W, C2, TH, TW, P):
    c2 = gid % C2
    p = gid // C2
    tw = p % TW
    t2 = p // TW
    th = t2 % TH
    n = t2 // TH
    h0 = th * 4 - 1
    w0 = tw * 4 - 1
    okh = [Int32(th > 0)] + [_valid(4 * TH - 5 + y >= H, h0 + y < H) for y in range(1, 6)]
    okw = [Int32(tw > 0)] + [_valid(4 * TW - 5 + x >= W, w0 + x < W) for x in range(1, 6)]
    rowoff = [(n * H + (h0 + y) * okh[y]) * (W * C2) + c2 for y in range(6)]
    coloff = [(w0 + x) * okw[x] * C2 for x in range(6)]
    rm = [None if isinstance(o, int) else Int32(0) - o for o in okh]
    cm = [None if isinstance(o, int) else Int32(0) - o for o in okw]
    d = []
    for y in range(6):
        row = []
        for x in range(6):
            v = xi[rowoff[y] + coloff[x]]
            if rm[y] is not None:
                v = v & rm[y]
            if cm[x] is not None:
                v = v & cm[x]
            row.append(_bf16x2_to_f32x2(v))
        d.append(row)
    vv = _sandwich_f(_bt_vec, d)
    for a in range(6):
        for b in range(6):
            vi[(a * 6 + b) * (P * C2) + gid] = _cvt_pack(vv[a][b][0], vv[a][b][1], "f16x2")


def _ot_body(mi, gid, P, C2):
    m = [[_f16x2_to_f32x2(mi[(a * 6 + b) * (P * C2) + gid]) for b in range(6)] for a in range(6)]
    yy = _sandwich_f(_at_vec, m)
    return [_cvt_pack(yy[i][j][0], yy[i][j][1], "bf16x2") for i in range(4) for j in range(4)]


def _make_wt(Co, C2, pdl):
    wait_end = _wait_end(pdl)

    @cute.kernel
    def wt_kernel(wi: cute.Tensor, ui: cute.Tensor):
        _pdl_start(pdl)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        _wt_body(wi, ui, bidx * TPB + tidx, Co, C2)
        if cutlass.const_expr(wait_end):
            if bidx == 0:  # one waiting CTA orders grid completion; all would stall later waves
                cute.arch.griddepcontrol_wait()

    @cute.jit
    def wt_launch(wi: cute.Tensor, ui: cute.Tensor, stream):
        wt_kernel(wi, ui).launch(grid=[Co * C2 // TPB, 1, 1], block=[TPB, 1, 1], stream=stream,
                  use_pdl=pdl is not None)
    return wt_launch


def _make_it(H, W, C2, TH, TW, P, pdl):
    wait_end = _wait_end(pdl)

    @cute.kernel
    def it_kernel(xi: cute.Tensor, vi: cute.Tensor):
        _pdl_start(pdl)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        _it_body(xi, vi, bidx * TPB + tidx, H, W, C2, TH, TW, P)
        if cutlass.const_expr(wait_end):
            if bidx == 0:  # one waiting CTA orders grid completion; all would stall later waves
                cute.arch.griddepcontrol_wait()

    @cute.jit
    def it_launch(xi: cute.Tensor, vi: cute.Tensor, stream):
        it_kernel(xi, vi).launch(grid=[P * C2 // TPB, 1, 1], block=[TPB, 1, 1], stream=stream,
                  use_pdl=pdl is not None)
    return it_launch


def _make_ot(H, W, C2, TH, TW, P, pdl):
    wait_end = _wait_end(pdl)

    @cute.kernel
    def ot_kernel(mi: cute.Tensor, yi: cute.Tensor):
        _pdl_start(pdl)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * TPB + tidx          # = p * C2 + c2
        c2 = gid % C2
        p = gid // C2
        tw = p % TW
        t2 = p // TW
        th = t2 % TH
        n = t2 // TH
        vals = _ot_body(mi, gid, P, C2)
        h0 = th * 4
        w0 = tw * 4
        base = ((n * H + h0) * W + w0) * C2 + c2
        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(4):
                if h0 + i < H:
                    if w0 + j < W:
                        yi[base + (i * W + j) * C2] = vals[i * 4 + j]
        if cutlass.const_expr(wait_end):
            if bidx == 0:  # one waiting CTA orders grid completion; all would stall later waves
                cute.arch.griddepcontrol_wait()

    @cute.jit
    def ot_launch(mi: cute.Tensor, yi: cute.Tensor, stream):
        ot_kernel(mi, yi).launch(grid=[P * C2 // TPB, 1, 1], block=[TPB, 1, 1], stream=stream,
                  use_pdl=pdl is not None)
    return ot_launch



# ---------------------------------------------------------------------------------------------
# direct-conv filter permute: w [Co, Ci, 3, 3] -> wp [Co, 3, 3, Ci] (both int32 views)
# ---------------------------------------------------------------------------------------------
def _make_pw(Co, C2, pdl):
    wait_end = _wait_end(pdl)

    @cute.kernel
    def pw_kernel(wi: cute.Tensor, wo: cute.Tensor):
        _pdl_start(pdl)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * TPB + tidx          # = co * C2 + c2
        _pw_body(wi, wo, gid, C2)
        if cutlass.const_expr(wait_end):
            if bidx == 0:  # one waiting CTA orders grid completion; all would stall later waves
                cute.arch.griddepcontrol_wait()

    @cute.jit
    def pw_launch(wi: cute.Tensor, wo: cute.Tensor, stream):
        pw_kernel(wi, wo).launch(grid=[Co * C2 // TPB, 1, 1], block=[TPB, 1, 1], stream=stream,
                  use_pdl=pdl is not None)
    return pw_launch


def _pw_body(wi, wo, gid, C2):
    base = gid * 9
    raw = [wi[base + j] for j in range(9)]
    h16 = []
    for j in range(9):
        h16 += [raw[j] & Int32(0xFFFF), (raw[j] >> 16) & Int32(0xFFFF)]
    co = gid // C2
    c2 = gid % C2
    for k in range(9):
        wo[(co * 9 + k) * C2 + c2] = h16[k] | (h16[9 + k] << 16)


# ---------------------------------------------------------------------------------------------
# 3. batched GEMM (adapted from the CUTLASS CuTe DSL Blackwell persistent dense GEMM example)
# ---------------------------------------------------------------------------------------------
def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    use_tma_store: bool,
    c_smem_layout: Union[cute.Layout, None],
) -> Tuple[int, int, int]:
    # Default ACC stages
    num_acc_stage = 2

    # Default C stages
    num_c_stage = 2 if use_tma_store else 0

    # Calculate smem layout and size for one stage of A, B, and C with 1-stage
    a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a_dtype, 1
    )
    b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b_dtype, 1
    )

    ab_bytes_per_stage = cute.size_in_bytes(
        a_dtype, a_smem_layout_stage_one
    ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    # Calculate A/B stages:
    # Start with total smem per CTA (capacity / occupancy)
    # Subtract reserved bytes and initial C stages bytes
    # Divide remaining by bytes needed per A/B stage
    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    # Refine epilogue stages:
    # Calculate remaining smem after allocating for A/B stages and reserved bytes
    # Add remaining unused smem to epilogue
    if use_tma_store:
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
    return num_acc_stage, num_ab_stage, num_c_stage


class PersistentDenseGemmKernel:

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
    ):

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.arch = "sm_100"
        self.k_tuple = False
        self.use_pdl = False

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        # Configure tiled mma
        tiled_mma = self._create_tiled_mma()

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            (mma_inst_shape_k * mma_inst_tile_k,) if self.k_tuple else mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        c_smem_layout = None
        if cutlass.const_expr(self.use_tma_store):
            c_smem_layout = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, 1
            )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,
        )

        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        self.c_smem_layout_staged = None
        if self.use_tma_store:
            self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # Setup TMA load for B
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
            )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # PDL: everything above (barrier init, descriptor prefetch) overlaps the previous kernel;
        # wait for it to fully complete before touching global memory, then release the next one.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        #
        # Construct the scheduler
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
                # Walk the (possibly hierarchical) K mode with an incremental coordinate so the
                # conv path does not pay a runtime idx2crd per k-tile.
                k_shape = cute.shape(tAgA_slice, mode=1)
                k_coord = cute.repeat_like(0, k_shape)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_coord)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_coord)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                    k_coord = cute.increment_coord(k_coord, k_shape)

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblk_crd],
                                tCrB[kblk_crd],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            if cutlass.const_expr(self.use_tma_store):
                assert tma_atom_c is not None and sC is not None
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # Pre-advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.use_tma_store):
                    acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        tCgC,
                        epi_tile,
                        num_tiles_executed,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                else:
                    acc_consumer_state = utils.gemm.sm100.epilogue(
                        self,
                        tidx,
                        tCtAcc_base,
                        tCgC,
                        epi_tile,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                    )

            if cutlass.const_expr(self.use_tma_store):
                # Wait for C store complete
                c_pipeline.producer_tail()
            else:
                # Synchronize before TMEM dealloc (done by the caller)
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
        arch: str,
    ) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch=arch)

        return num_tmem_alloc_cols


class ConvKernel(PersistentDenseGemmKernel):
    """Direct 3x3 conv (pad 1) as implicit GEMM: A = im2col TMA view of x, B = (Co, (C, S, R, T))
    view of the tap-major filter, C = y as a plain (N*H*W, Co) matrix."""

    def __init__(self, *args):
        super().__init__(*args)
        self.k_tuple = True

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
                 max_active_clusters: cutlass.Constexpr, stream,
                 epilogue_op: cutlass.Constexpr = lambda x: x):
        # a: (N, 1, H, W, C), b: (Co, 1, 3, 3, C), c: (L=1, N*H*W, Co)
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_major_mode = cute.nvgpu.OperandMajorMode.K
        self.b_major_mode = cute.nvgpu.OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        mA = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[3, 2, 1, 0, 4]))
        mA = cute.group_modes(mA, begin=0, end=4)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        a_op = (cpasync.CopyBulkTensorIm2ColG2SMulticastOp(cta_group=self.cta_group)
                if self.is_a_mcast else cpasync.CopyBulkTensorIm2ColG2SOp(cta_group=self.cta_group))
        pad = (Int32(0), Int32(1), Int32(1))
        one = (Int32(1), Int32(1), Int32(1))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_im2col_tma_atom_A(
            a_op, mA, a_smem_layout, self.mma_tiler, tiled_mma,
            (b.shape[1], b.shape[2], b.shape[3]), pad, pad, one, one,
            self.cluster_layout_vmnk.shape, internal_type=None)

        mB = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[0, 4, 3, 2, 1]))
        mB = cute.group_modes(mB, begin=1, end=5)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
            internal_type=None)

        tma_tensor_a = cute.make_tensor(tma_tensor_a.iterator,
                                        cute.append(tma_tensor_a.layout, cute.make_layout(1)))
        tma_tensor_b = cute.make_tensor(tma_tensor_b.iterator,
                                        cute.append(tma_tensor_b.layout, cute.make_layout(1)))

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile)

        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters)

        self.kernel(
            tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_c,
            tma_tensor_c, self.cluster_layout_vmnk, self.a_smem_layout_staged,
            self.b_smem_layout_staged, self.c_smem_layout_staged, self.epi_tile,
            self.tile_sched_params, epilogue_op,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                 cluster=(*self.cluster_shape_mn, 1), stream=stream, use_pdl=self.use_pdl)


@cute.jit
def _bmm_noswap(gemm_op: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
                max_active_clusters: cutlass.Constexpr, stream):
    # M = tiles (A = V), N = Cout (B = U), C = M_ row-major
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, max_active_clusters, stream)


@cute.jit
def _bmm(gemm_op: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
         max_active_clusters: cutlass.Constexpr, stream):
    # Operand-swapped Winograd GEMM: M = Cout (A = U), N = tiles (B = V), C = M_ stored Cout-major.
    # torch U [L, Co, K], V [L, P, K], M_ [L, P, Co]  ->  cute (Co, K, L), (P, K, L), (Co, P, L)
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[2, 1, 0]))
    gemm_op(a, b, c, max_active_clusters, stream)


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------
GEMM_CFG = dict(cluster_shape_mn=(2, 1), use_2cta_instrs=True)
# Cin*Cout <= 512*512: a 2x2 cluster multicasts the V tile to both Cout halves (512@180x320:
# 1.012 -> 0.981 ms); at Cin = Cout = 1024 it measured 3% slower, so those keep 2x1.
GEMM_CLUSTER_SMALL_U = (2, 2)
NUM_CLUSTERS = 74  # 148 SMs / 2-CTA clusters


def _pick_ntile(P, Co):
    # tiles along P: minimise (padded waves x tile width); near-ties go to the wider tile
    costs = {}
    for nt in (256, 224, 192, 160, 128):
        tiles = 36 * (Co // 256) * (-(-P // nt))
        costs[nt] = -(-tiles // NUM_CLUSTERS) * nt
    best = min(costs.values())
    return max(nt for nt, c in costs.items() if c <= best * 1.03)
CONV_CFG = dict(mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1), use_2cta_instrs=True)
_cache = {}
CHUNK_MIN_WORK = 1_500_000_000
CHUNKS_SMALL_U = 2  # 4 chunks measured 3% slower on 512@180x320 (transforms co-running with GEMMs share HBM)


def _num_chunks(N, TPF, Ci, Co):
    if N % 2 or N * TPF * Ci * Co <= CHUNK_MIN_WORK:
        return 1
    if Ci * Co <= 512 * 512 and N % CHUNKS_SMALL_U == 0:
        return CHUNKS_SMALL_U
    return 2
GEMM_SWAP = os.environ.get("WINO_SWAP", "0") == "1"
DIRECT_MAX_CIN = 256  # Winograd's transform traffic does not pay off at Cin=256


def _max_clusters(cl):
    return utils.HardwareInfo().get_max_active_clusters(cl)


def _fake(dtype, shape, div=8):
    stride = []
    for i in range(len(shape)):
        stride.append(1 if i == len(shape) - 1 else cute.sym_int64(divisibility=div))
    return cute.runtime.make_fake_tensor(dtype, tuple(shape), stride=tuple(stride),
                                         assumed_align=16)


def _fake1d():
    return cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(1,), assumed_align=16)


def _get(key):
    if key in _cache:
        return _cache[key]
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    opts = "--enable-tvm-ffi"
    name, shp = key[0], key[1:]
    if name == "gemm":
        cfg = GEMM_CFG
        ntile, pdl, swap, cluster = shp
        gemm = PersistentDenseGemmKernel(Float32, cfg["use_2cta_instrs"], (256, ntile),
                                         cluster, True)
        gemm.use_pdl = pdl
        cl = cluster[0] * cluster[1]
        mac = _max_clusters(cl)
        sp, sk, sn = cute.sym_int(), cute.sym_int(divisibility=64), cute.sym_int(divisibility=64)
        if swap:
            a, b = _fake(Float16, (36, sn, sk)), _fake(Float16, (36, sp, sk))
        else:
            a, b = _fake(Float16, (36, sp, sk)), _fake(Float16, (36, sn, sk))
        c = _fake(Float16, (36, sp, sn))
        fn = cute.compile(_bmm if swap else _bmm_noswap, gemm, a, b, c, mac, stream, options=opts)
    elif name == "conv":
        cfg = CONV_CFG
        conv = ConvKernel(Float32, cfg["use_2cta_instrs"], cfg["mma_tiler_mn"],
                          cfg["cluster_shape_mn"], True)
        conv.use_pdl = shp[0]
        mac = _max_clusters(cfg["cluster_shape_mn"][0] * cfg["cluster_shape_mn"][1])
        sn, sh, sw = cute.sym_int(), cute.sym_int(), cute.sym_int()
        sc, sk = DIRECT_MAX_CIN, DIRECT_MAX_CIN
        a = _fake(BFloat16, (sn, 1, sh, sw, sc))
        b = _fake(BFloat16, (sk, 1, 3, 3, sc))
        c = _fake(BFloat16, (1, cute.sym_int(), sk))
        fn = cute.compile(conv, a, b, c, mac, stream, options=opts)
    else:
        maker = {"wt": _make_wt, "it": _make_it, "ot": _make_ot, "pw": _make_pw}[name]
        fn = cute.compile(maker(*shp), _fake1d(), _fake1d(), stream, options=opts)
    _cache[key] = fn
    return fn


# kernel-level PDL modes (wait_at_start, release_at_start, wait_at_end)
_FIRST = (True, True, False)     # first kernel of a call: previous call fully done before writing
_OVERLAP = (False, True, True)   # independent of predecessor: run alongside it, finish after it
_CONSUME = (True, True, False)   # reads predecessor output
_LAST = (True, False, False)     # never release the next call early (its buffers may alias ours)


@torch.no_grad()
def run(x, weight, y):
    N, H, W, Ci = x.shape
    Co = weight.shape[0]
    dev = x.device
    if Ci <= DIRECT_MAX_CIN:
        assert Ci == DIRECT_MAX_CIN and Co == DIRECT_MAX_CIN
        wp = torch.empty((Co, 3, 3, Ci), dtype=torch.bfloat16, device=dev)
        _get(("pw", Co, Ci // 2, None))(weight.view(-1).view(torch.int32), wp.view(-1).view(torch.int32))
        _get(("conv", False))(x.view(N, 1, H, W, Ci), wp.view(Co, 1, 3, 3, Ci), y.view(1, N * H * W, Co))
        return
    TH = (H + 3) // 4
    TW = (W + 3) // 4
    ci2, co2 = Ci // 2, Co // 2
    # Batch chunks pipeline the transforms against the GEMM under PDL:
    #   wt, it0, G0, it1, G1, ot0, it2, G2, ot1, ..., ot_{n-1}
    # so chunk k+1's input transform and chunk k-1's output transform co-run with GEMM k.
    nch = _num_chunks(N, TH * TW, Ci, Co)
    Nc = N // nch
    Pc = Nc * TH * TW
    U = torch.empty((36, Co, Ci), dtype=torch.float16, device=dev)
    V = [torch.empty((36, Pc, Ci), dtype=torch.float16, device=dev) for _ in range(nch)]
    M = [torch.empty((36, Pc, Co), dtype=torch.float16, device=dev) for _ in range(nch)]
    xs = [x[i * Nc:(i + 1) * Nc].reshape(-1).view(torch.int32) for i in range(nch)]
    ys = [y[i * Nc:(i + 1) * Nc].reshape(-1).view(torch.int32) for i in range(nch)]
    cluster = GEMM_CLUSTER_SMALL_U if Ci * Co <= 512 * 512 else GEMM_CFG["cluster_shape_mn"]
    gemm = _get(("gemm", 256, True, False, cluster))
    it = _get(("it", H, W, ci2, TH, TW, Pc, _OVERLAP))
    _get(("wt", Co, ci2, _FIRST))(weight.view(-1).view(torch.int32), U.view(-1).view(torch.int32))
    for k in range(nch):
        it(xs[k], V[k].view(-1).view(torch.int32))
        gemm(V[k], U, M[k])
        if k > 0:
            _get(("ot", H, W, co2, TH, TW, Pc, _OVERLAP))(M[k - 1].view(-1).view(torch.int32), ys[k - 1])
    _get(("ot", H, W, co2, TH, TW, Pc, _LAST))(M[nch - 1].view(-1).view(torch.int32), ys[nch - 1])
