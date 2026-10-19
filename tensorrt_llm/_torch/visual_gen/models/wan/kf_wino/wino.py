# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel Factory output (campaign wan-vae-spatial-conv, solution
# wan_conv3x3_hybrid_wino_f4_direct_cutedsl, harvested Mon 19 Oct 2026). Kept byte-identical to
# the harvest apart from this header and package-relative imports; wino_kf.py holds our glue.

# Separable Winograd F(MH x MW, 3x3) building blocks for the channels-last 3x3 conv.
#
#   U[xi, co, ci] = (G_h g G_w^T)[xi]       fp16, xi = i*AW + j   (weight transform)
#   V[xi, t,  ci] = (B_h^T d B_w)[xi]       fp16, t = MHxMW output tile (input transform)
#   M[xi, t,  co] = sum_ci V[xi,t,ci] U[xi,co,ci]                   (batched tcgen05 GEMMs)
#   y[tile t]     = A_h^T M[:, :, t] A_w                            (output transform)
#
# F(4x4) needs the well-conditioned F(4,3) point set below to keep fp16 error inside tolerance.
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float16, Int32
import cuda.bindings.driver as cuda

NT = 128  # threads per CTA for the transform kernels

# Programmatic-dependent-launch flags for the transform kernels (bit set):
PDL_TRIGGER = 1    # release the next grid at kernel start (it only needs data we do not write)
PDL_WAIT_START = 2  # wait for the previous grid before reading its output
PDL_WAIT_END = 4    # CTA 0 waits for the previous grid before exiting (completion stays transitive)


def _pdl_start(pdl):
    if cutlass.const_expr(pdl & PDL_TRIGGER):
        cute.arch.griddepcontrol_launch_dependents()
    if cutlass.const_expr(pdl & PDL_WAIT_START):
        cute.arch.griddepcontrol_wait()


WINO = {
    2: dict(
        BT=[[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
        G=[[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]],
        AT=[[1, 1, 1, 0], [0, 1, -1, -1]],
    ),
    # F(4,3) on points (0, 1, -1, 2, -1/2, inf): ~1.5x lower fp16 error than (0, +-1, +-2, inf)
    4: dict(
        BT=[[1, 3 / 2, -2, -3 / 2, 1, 0], [0, -1, -5 / 2, -1 / 2, 1, 0], [0, 1, 1 / 2, -5 / 2, 1, 0],
            [0, -1 / 2, -1, 1 / 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 1, 3 / 2, -2, -3 / 2, 1]],
        G=[[1, 0, 0], [-1 / 3, -1 / 3, -1 / 3], [1 / 3, -1 / 3, 1 / 3], [1 / 15, 2 / 15, 4 / 15],
           [-16 / 15, 8 / 15, -4 / 15], [0, 0, 1]],
        AT=[[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -1 / 2, 0], [0, 1, 1, 4, 1 / 4, 0], [0, 1, -1, 8, -1 / 8, 1]],
    ),
}


def _bt4(d):
    # factorized F(4,3) B^T for points (0, 1, -1, 2, -1/2, inf): 16 ops instead of 26
    d0, d1, d2, d3, d4, d5 = d
    e, f, s = d1 - d3, d4 - d2, d1 + d3
    h = f + s * -0.75
    return [(f - d2) + e * 1.5 + d0, h + d2 * -1.5 + e * -0.25, h + d2 * 1.5 + e * 1.75,
            f + e * -0.5, f + e * 2.0, (d5 + f * -1.5) + (e - d3)]


def _at4(m):
    # factorized F(4,3) A^T: 12 ops instead of 18
    m0, m1, m2, m3, m4, m5 = m
    p, q = m1 + m2, m1 - m2
    return [m0 + p + m3 + m4, q + m3 * 2.0 + m4 * -0.5, p + m3 * 4.0 + m4 * 0.25,
            q + m3 * 8.0 + m4 * -0.125 + m5]


def _lin(mat, v):
    if mat is WINO[4]["BT"]:
        return _bt4(v)
    if mat is WINO[4]["AT"]:
        return _at4(v)
    """out[i] = sum_k mat[i][k] * v[k] with trace-time constant coefficients."""
    out = []
    for row in mat:
        acc = None
        for c, x in zip(row, v):
            if c == 0:
                continue
            if acc is None:
                acc = x if c == 1 else (-x if c == -1 else x * c)
            elif c == 1:
                acc = acc + x
            elif c == -1:
                acc = acc - x
            else:
                acc = acc + x * c
        out.append(acc)
    return out


def _vec(mT, tile, coord):
    """Contiguous rank-1 view (last tile mode) of mT at vector coordinate `coord`."""
    v = cute.local_tile(mT, tile, coord)
    return v[tuple(0 for _ in range(len(tile) - 1)) + (None,)]


# ---------------------------------------------------------------- weight transform
@cute.jit
def _w_body(mWf: cute.Tensor, mU: cute.Tensor, L: Int32, K: cutlass.Constexpr,
            C: cutlass.Constexpr, MH: cutlass.Constexpr, MW: cutlass.Constexpr,
            VEC: cutlass.Constexpr):
    Gh, Gw = WINO[MH]["G"], WINO[MW]["G"]
    AW = MW + 2
    CG = C // VEC
    if L < K * CG:
        cg = L % CG
        co = L // CG
        frag = cute.make_rmem_tensor((VEC * 9,), mWf.element_type)
        cute.autovec_copy(_vec(mWf, (1, VEC * 9), (co, cg)), frag)
        fv = cute.make_tensor(frag.iterator, cute.make_layout((VEC, 3, 3), stride=(9, 3, 1)))
        g = [[fv[None, r, s].load().to(Float32) for s in range(3)] for r in range(3)]
        gh = [_lin(Gh, [g[0][s], g[1][s], g[2][s]]) for s in range(3)]  # gh[s][i]
        for i in cutlass.range_constexpr(MH + 2):
            u = _lin(Gw, [gh[0][i], gh[1][i], gh[2][i]])
            for j in cutlass.range_constexpr(AW):
                of = cute.make_rmem_tensor((VEC,), Float16)
                of.store(u[j].to(Float16))
                cute.autovec_copy(of, _vec(mU, (1, 1, VEC), (i * AW + j, co, cg)))


def _w_vec(MH, MW):
    return 8 if (MH + 2) * (MW + 2) <= 16 else 4


# ---------------------------------------------------------------- input transform
@cute.jit
def _in_body(mX: cute.Tensor, mV: cute.Tensor, L: Int32, N: cutlass.Constexpr,
             H: cutlass.Constexpr, W: cutlass.Constexpr, C: cutlass.Constexpr,
             MH: cutlass.Constexpr, MW: cutlass.Constexpr, VEC: cutlass.Constexpr):
    BTh, BTw = WINO[MH]["BT"], WINO[MW]["BT"]
    AH, AW = MH + 2, MW + 2
    TH = (H + MH - 1) // MH
    TW = W // MW
    CG = C // VEC
    T = N * TH * TW
    if L < T * CG:
        g = L % CG
        t = L // CG
        tw = t % TW
        r = t // TW
        th = r % TH
        n = r // TH
        # column (W) transform of each input row as it is loaded
        rows = []
        for a in cutlass.range_constexpr(AH):
            h = MH * th - 1 + a
            hok = (h >= 0) and (h < H)
            hc = h if hok else Int32(0)
            d = []
            for b in cutlass.range_constexpr(AW):
                w = MW * tw - 1 + b
                wok = (w >= 0) and (w < W)
                wc = w if wok else Int32(0)
                mk = Float32(1.0) if (hok and wok) else Float32(0.0)
                frag = cute.make_rmem_tensor((VEC,), mX.element_type)
                cute.autovec_copy(_vec(mX, (1, 1, 1, VEC), (n, hc, wc, g)), frag)
                d.append(frag.load().to(Float32) * mk)
            rows.append(_lin(BTw, d))  # rows[a][j]
        for j in cutlass.range_constexpr(AW):
            v = _lin(BTh, [rows[a][j] for a in range(AH)])
            for i in cutlass.range_constexpr(AH):
                of = cute.make_rmem_tensor((VEC,), Float16)
                of.store(v[i].to(Float16))
                cute.autovec_copy(of, _vec(mV, (1, 1, VEC), (i * AW + j, t, g)))


def _in_vec(MH, MW):
    return {16: 8, 24: 4, 36: 2}[(MH + 2) * (MW + 2)]


@cute.kernel
def _in_xform_kernel(mX: cute.Tensor, mV: cute.Tensor, N: cutlass.Constexpr, H: cutlass.Constexpr,
                     W: cutlass.Constexpr, C: cutlass.Constexpr, MH: cutlass.Constexpr,
                     MW: cutlass.Constexpr, PDL: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    _pdl_start(PDL)
    _in_body(mX, mV, bidx * NT + tidx, N, H, W, C, MH, MW, _in_vec(MH, MW))
    # A grid completes only when all of its CTAs exit, so one waiting CTA is enough to make our
    # completion imply the predecessor's; waiting in every CTA would park finished CTAs on the
    # SMs and stall the rest of the grid behind the co-running GEMM.
    if cutlass.const_expr(PDL & PDL_WAIT_END):
        if bidx == 0:
            cute.arch.griddepcontrol_wait()


@cute.jit
def in_xform(mX: cute.Tensor, mV: cute.Tensor, N: cutlass.Constexpr, H: cutlass.Constexpr,
             W: cutlass.Constexpr, C: cutlass.Constexpr, MH: cutlass.Constexpr,
             MW: cutlass.Constexpr, PDL: cutlass.Constexpr, stream: cuda.CUstream):
    T = N * ((H + MH - 1) // MH) * (W // MW)
    nblk = (T * (C // _in_vec(MH, MW)) + NT - 1) // NT
    _in_xform_kernel(mX, mV, N, H, W, C, MH, MW, PDL).launch(
        grid=[nblk, 1, 1], block=[NT, 1, 1], stream=stream, use_pdl=PDL != 0)


# ---------------------------------------------------------------- weight + first input chunk
@cute.kernel
def _wi_xform_kernel(mWf: cute.Tensor, mU: cute.Tensor, mX: cute.Tensor, mV: cute.Tensor,
                     K: cutlass.Constexpr, N: cutlass.Constexpr, H: cutlass.Constexpr,
                     W: cutlass.Constexpr, C: cutlass.Constexpr, MH: cutlass.Constexpr,
                     MW: cutlass.Constexpr, NBW: cutlass.Constexpr, PDL: cutlass.Constexpr):
    # The weight and input transforms are independent: CTAs [0, NBW) do weights, the rest do
    # input tiles, so both streams of HBM traffic run concurrently in one grid.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    _pdl_start(PDL)
    if bidx < NBW:
        _w_body(mWf, mU, bidx * NT + tidx, K, C, MH, MW, _w_vec(MH, MW))
    else:
        _in_body(mX, mV, (bidx - NBW) * NT + tidx, N, H, W, C, MH, MW, _in_vec(MH, MW))


@cute.jit
def wi_xform(mWf: cute.Tensor, mU: cute.Tensor, mX: cute.Tensor, mV: cute.Tensor,
             K: cutlass.Constexpr, N: cutlass.Constexpr, H: cutlass.Constexpr,
             W: cutlass.Constexpr, C: cutlass.Constexpr, MH: cutlass.Constexpr,
             MW: cutlass.Constexpr, PDL: cutlass.Constexpr, stream: cuda.CUstream):
    T = N * ((H + MH - 1) // MH) * (W // MW)
    nbw = (K * (C // _w_vec(MH, MW)) + NT - 1) // NT
    nbi = (T * (C // _in_vec(MH, MW)) + NT - 1) // NT
    _wi_xform_kernel(mWf, mU, mX, mV, K, N, H, W, C, MH, MW, nbw, PDL).launch(
        grid=[nbw + nbi, 1, 1], block=[NT, 1, 1], stream=stream, use_pdl=PDL != 0)


# ---------------------------------------------------------------- output transform
@cute.kernel
def _out_xform_kernel(mM: cute.Tensor, mY: cute.Tensor, N: cutlass.Constexpr, H: cutlass.Constexpr,
                      W: cutlass.Constexpr, K: cutlass.Constexpr, MH: cutlass.Constexpr,
                      MW: cutlass.Constexpr, VEC: cutlass.Constexpr,
        PDL: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    _pdl_start(PDL)
    ATh, ATw = WINO[MH]["AT"], WINO[MW]["AT"]
    AH, AW = MH + 2, MW + 2
    TH = (H + MH - 1) // MH
    TW = W // MW
    KG = K // VEC
    T = N * TH * TW
    L = bidx * NT + tidx
    if L < T * KG:
        g = L % KG
        t = L // KG
        tw = t % TW
        r = t // TW
        th = r % TH
        n = r // TH
        z = []
        for i in cutlass.range_constexpr(AH):
            m = []
            for j in cutlass.range_constexpr(AW):
                frag = cute.make_rmem_tensor((VEC,), mM.element_type)
                cute.autovec_copy(_vec(mM, (1, 1, VEC), (i * AW + j, t, g)), frag)
                m.append(frag.load().to(Float32))
            z.append(_lin(ATw, m))  # z[i][q]
        for q in cutlass.range_constexpr(MW):
            yq = _lin(ATh, [z[i][q] for i in range(AH)])
            for p in cutlass.range_constexpr(MH):
                of = cute.make_rmem_tensor((VEC,), mY.element_type)
                of.store(yq[p].to(mY.element_type))
                h = MH * th + p
                if cutlass.const_expr(H % MH == 0 or p == 0):
                    cute.autovec_copy(of, _vec(mY, (1, 1, 1, VEC), (n, h, MW * tw + q, g)))
                else:
                    if h < H:
                        cute.autovec_copy(of, _vec(mY, (1, 1, 1, VEC), (n, h, MW * tw + q, g)))
    # A grid completes only when all of its CTAs exit, so one waiting CTA is enough to make our
    # completion imply the predecessor's; waiting in every CTA would park finished CTAs on the
    # SMs and stall the rest of the grid behind the co-running GEMM.
    if cutlass.const_expr(PDL & PDL_WAIT_END):
        if bidx == 0:
            cute.arch.griddepcontrol_wait()


@cute.jit
def out_xform(mM: cute.Tensor, mY: cute.Tensor, N: cutlass.Constexpr, H: cutlass.Constexpr,
              W: cutlass.Constexpr, K: cutlass.Constexpr, MH: cutlass.Constexpr,
              MW: cutlass.Constexpr, PDL: cutlass.Constexpr, stream: cuda.CUstream):
    VEC = 8 if (MH + 2) * (MW + 2) <= 24 else 4
    T = N * ((H + MH - 1) // MH) * (W // MW)
    nblk = (T * (K // VEC) + NT - 1) // NT
    _out_xform_kernel(mM, mY, N, H, W, K, MH, MW, VEC, PDL).launch(
        grid=[nblk, 1, 1], block=[NT, 1, 1], stream=stream, use_pdl=PDL != 0)
