# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Winograd F(4x4, 3x3) input transform with the pixel box staged through shared memory.

Same contract as :func:`wino.in_xform`: ``x [N, H, W, C]`` fp16 or bf16 channels-last -> ``V [36, T, C]``
fp16 with ``T = N * ceil(H/4) * (W/4)`` tiles, ``V[i*6+j, t, c] = (B^T d B)[i, j]`` of tile t's
6x6 input patch (padding 1) in fp32, rounded once.  The values are those of ``wino.in_xform``
up to fp32 multiply-add contraction (the compiler fuses the two kernels' multiply-add pairs
differently; in practice a few elements per 1e9 differ by one fp16 ulp) and the sign of zero at
padding pixels (zero-filled here instead of multiplied by a zero mask).

One CTA walks a persistent sequence of work items (frame, tile row, group of TTW tiles along W,
chunk of KC channels).  A TMA warp streams each item's raw box (6 rows x (4*TTW+2) pixels x KC
channels, out-of-frame pixels zero-filled) into a shared-memory stage; TTW * KC/64 transform
warps, one per (tile, 64-channel slice), read their 36 pixels from the stage (a warp reads one
128-B piece of a pixel row per load, one channel pair per lane), run the fp32 transform of
``wino`` and store V.  Bytes in flight per SM are set by the stage count, not by the registers
of the transform; KC sets the contiguous run length of the box reads and the V writes.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float16, Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda

from .wino import PDL_WAIT_END, WINO, _lin, _pdl_start, _vec

CS = 64        # channels per transform warp: one 128-B piece of a pixel row, a channel pair per lane
TMA_C = 256    # channels per TMA copy (box dimensions are at most 256 elements)
BOXH = 6


@dsl_user_op
def _lds32(addr: Int32, *, loc=None, ip=None) -> Int32:
    # side effects declared so the read stays between the stage's full-barrier wait and its
    # release (the compiler cannot see the TMA writes behind the barriers)
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def _pair_to_f32(v: Int32, dtype):
    """The two 16-bit halves (``dtype`` = fp16 or bf16) of a shared-memory word as the fp32 pair
    the transform works on."""
    frag = cute.make_rmem_tensor((2,), dtype)
    cute.recast_tensor(frag, Int32)[0] = v
    return frag.load().to(Float32)


class InXformSmem:
    """Compiled per shape: ``InXformSmem(frames, h, w, c, ttw, kc, stages, pdl)(mX, mV, grid,
    stream)``, ``grid`` = number of persistent CTAs (at most the number of work items)."""

    def __init__(self, frames, h, w, c, ttw, kc, stages, pdl):
        assert w % 4 == 0 and kc % CS == 0 and c % kc == 0 and ttw >= 1 and stages >= 1
        self.frames, self.h, self.w, self.c = frames, h, w, c
        self.ttw, self.kc, self.stages, self.pdl = ttw, kc, stages, pdl
        self.th = -(-h // 4)
        self.tw = w // 4
        self.ng = -(-self.tw // ttw)
        self.nkc = c // kc
        self.ncs = kc // CS
        self.boxw = 4 * ttw + 2
        self.items = frames * self.th * self.ng * self.nkc
        self.warps = ttw * self.ncs
        self.threads = 32 * (self.warps + 1)
        self.stage_bytes = BOXH * self.boxw * kc * 2
        self.tma_c = min(kc, TMA_C)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mV: cute.Tensor, grid: Int32, stream: cuda.CUstream):
        kc, boxw, tma_c = self.kc, self.boxw, self.tma_c
        stage_layout = cute.make_layout((1, BOXH, boxw, kc), stride=(BOXH * boxw * kc, boxw * kc, kc, 1))
        # one TMA copy covers tma_c channels of the box: the smem image of a copy is the stage
        # layout restricted to those channels
        copy_layout = cute.make_layout((1, BOXH, boxw, tma_c), stride=(BOXH * boxw * kc, boxw * kc, kc, 1))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX, copy_layout, (1, BOXH, boxw, tma_c))
        self.kernel(tma_atom, tma_tensor, mV, stage_layout, copy_layout, mX.element_type).launch(
            grid=(grid, 1, 1), block=(self.threads, 1, 1), stream=stream, use_pdl=self.pdl != 0)

    @cute.kernel
    def kernel(self, tma_atom: cute.CopyAtom, mX: cute.Tensor, mV: cute.Tensor,
               stage_layout: cute.Layout, copy_layout: cute.Layout, dtype: cutlass.Constexpr):
        ttw, kc, stages, nkc, ng = self.ttw, self.kc, self.stages, self.nkc, self.ng
        th_n, tw_n, ncs, nwarps, tma_c = self.th, self.tw, self.ncs, self.warps, self.tma_c
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % 32
        _pdl_start(self.pdl)
        if warp_idx == nwarps:
            cpasync.prefetch_descriptor(tma_atom)

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, stages * 2]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, nwarps),
            tx_count=self.stage_bytes,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        sX = smem.allocate_tensor(
            element_type=dtype,
            layout=cute.append(stage_layout, cute.make_layout(stages, stride=cute.size(stage_layout))),
            byte_alignment=1024)
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        nitems = Int32(self.items)
        count = (nitems - bidx + gdim - 1) // gdim

        if warp_idx == nwarps:
            # the TMA copies see the stage as (channel sub-box, stage) with the sub-box layout
            sC = cute.make_tensor(sX.iterator, cute.make_layout(
                (1, BOXH, self.boxw, tma_c, kc // tma_c, stages),
                stride=(BOXH * self.boxw * kc, self.boxw * kc, kc, 1, tma_c, cute.size(stage_layout))))
            st = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, stages)
            for k in cutlass.range(count, unroll=1):
                it = bidx + k * gdim
                c0 = it % nkc
                r = it // nkc
                g = r % ng
                r = r // ng
                th = r % th_n
                n = r // th_n
                gX = cute.local_tile(cute.domain_offset((n, 4 * th - 1, 4 * ttw * g - 1, c0 * kc), mX),
                                     (1, BOXH, self.boxw, tma_c), (0, 0, 0, None))
                tXsX, tXgX = cpasync.tma_partition(
                    tma_atom, 0, cute.make_layout(1),
                    cute.group_modes(sC, 0, 4), cute.group_modes(gX, 0, 4))
                pipe.producer_acquire(st)
                for j in cutlass.range_constexpr(kc // tma_c):
                    cute.copy(tma_atom, tXgX[(None, j)], tXsX[(None, j, st.index)],
                              tma_bar_ptr=pipe.producer_get_barrier(st))
                st.advance()
            pipe.producer_tail(st)
        else:
            BT = WINO[4]["BT"]
            tx = warp_idx // ncs
            cs = warp_idx % ncs
            x_base = Int32(sX.iterator.toint()) + tx * (4 * kc * 2) + cs * (CS * 2) + lane * 4
            cst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)
            for k in cutlass.range(count, unroll=1):
                it = bidx + k * gdim
                c0 = it % nkc
                r = it // nkc
                g = r % ng
                r = r // ng
                th = r % th_n
                n = r // th_n
                tw = ttw * g + tx
                valid = tw < tw_n
                t = Int64((n * th_n + th) * tw_n + tw)  # tile coordinate in 64 bits: V above 2^31 elements
                cpair = (c0 * kc + cs * CS) // 2 + lane
                pipe.consumer_wait(cst)
                base = x_base + cst.index * self.stage_bytes
                rows = []
                for a in cutlass.range_constexpr(BOXH):
                    d = []
                    for b in cutlass.range_constexpr(6):
                        d.append(_pair_to_f32(_lds32(base + (a * self.boxw + b) * (kc * 2)), dtype))
                    rows.append(_lin(BT, d))  # rows[a][j]
                for j in cutlass.range_constexpr(6):
                    v = _lin(BT, [rows[a][j] for a in range(BOXH)])
                    for i in cutlass.range_constexpr(6):
                        of = cute.make_rmem_tensor((2,), Float16)
                        of.store(v[i].to(Float16))
                        if valid:
                            cute.autovec_copy(of, _vec(mV, (1, 1, 2), (i * 6 + j, t, cpair)))
                pipe.consumer_release(cst)
                cst.advance()
        if cutlass.const_expr(self.pdl & PDL_WAIT_END):
            if bidx == 0:
                cute.arch.griddepcontrol_wait()


_compiled: dict = {}


def in_xform_smem(x, v, pdl: int, ttw: int = 2, kc: int = 256, stages: int = 4, ctas_per_sm: int = 0) -> None:
    """``v [36, T, C]`` fp16 = the F(4x4, 3x3) input transform of ``x [N, H, W, C]`` fp16 or bf16 (see the
    module docstring), launched on the current torch stream; ``ttw`` tiles per work item along W,
    ``kc`` channels per work item (lowered to the largest multiple of 64 dividing C), ``stages``
    shared-memory stages (``stages * 6 * (4 * ttw + 2) * kc * 2`` bytes); ``ctas_per_sm`` overrides the persistent grid size (0: as many CTAs as
    registers and shared memory fit).  The defaults (2 tiles x 256 channels per item, 4 stages
    of 30 KB) were the fastest or within noise of the fastest on the six decoder shapes measured
    on GB200 (CUPTI, L2 flushed); 64-channel items lose on the 180x320 shapes, 1024-channel items
    gain nothing."""
    import torch
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    frames, h, w, c = x.shape
    kc = min(kc, c)
    while c % kc:
        kc -= CS
    key = (frames, h, w, c, x.dtype, tuple(x.stride()), tuple(v.stride()), pdl, ttw, kc, stages, ctas_per_sm)
    fn = _compiled.get(key)
    if fn is None:
        op = InXformSmem(frames, h, w, c, ttw, kc, stages, pdl)
        props = torch.cuda.get_device_properties(x.device)
        smem_per_cta = op.stages * op.stage_bytes + 1024 + 16 * stages
        per_sm = ctas_per_sm or min(props.shared_memory_per_multiprocessor // smem_per_cta,
                                    65536 // (104 * op.threads))
        grid = min(op.items, props.multi_processor_count * max(per_sm, 1))
        fn = (cute.compile(op, from_dlpack(x, assumed_align=16), from_dlpack(v, assumed_align=16),
                           Int32(grid), make_fake_stream()), grid)
        _compiled[key] = fn
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn[0](from_dlpack(x, assumed_align=16), from_dlpack(v, assumed_align=16), Int32(fn[1]), stream)
