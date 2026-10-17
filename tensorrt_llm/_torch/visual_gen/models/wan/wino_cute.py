# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd F(2x2, 3x3) GEMMs for Blackwell (sm100) in CuTe DSL.

Stage 0: the 64 batched GEMMs ``M[l] = V[l] @ U[l]`` (l = temporal component x tile
coordinate) as one tcgen05 kernel: TMA loads into a multi-stage shared-memory ring, one
128x128x64 MMA tile per CTA, fp32 accumulation in tensor memory, bf16 output.

Tensors (torch, bf16, CUDA):
    v: [L, M, K]   transformed input tiles, K contiguous
    u: [L, N, K]   transformed filters, K contiguous (``winograd_spatial_weights`` gives
                   [L, K, N]; transpose once at load time)
    m: [L, M, N]   output, N contiguous
M must be a multiple of 128 and N of 128, K of 64 (pad M on the host; padded rows are
independent garbage and are never read).
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

IO_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32
MMA_INST_MNK = (128, 128, 16)
MMA_TILER_MNK = (128, 128, 64)
THREADS = 128
AB_STAGES = 6
ACC_STAGES = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, bidz = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier
    )
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    num_tma_copy_bytes = cute.size_in_bytes(
        IO_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(IO_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # (bM, bK, RestM, RestK, RestL) etc.
    gA = cute.local_tile(mA_mkl, cute.slice_(MMA_TILER_MNK, (None, 0, None)), (None, None, None))
    gB = cute.local_tile(mB_nkl, cute.slice_(MMA_TILER_MNK, (0, None, None)), (None, None, None))
    gC = cute.local_tile(mC_mnl, cute.slice_(MMA_TILER_MNK, (None, None, 0)), (None, None, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)  # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgB = thr_mma.partition_B(gB)  # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgC = thr_mma.partition_C(gC)  # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # Epilogue partition: every thread owns one accumulator row (one M row), 32 columns at a time.
    subtile_cnt = MMA_TILER_MNK[1] // 32
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_this = tCgC[(None, None, None, bidx, bidy, bidz)]
    gC_epi = cute.zipped_divide(gC_this, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, ACC_DTYPE)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, IO_DTYPE)

    num_k_tiles = cute.size(gA, mode=[3])
    tAgA_this = tAgA[(None, bidx, None, bidz)]
    tBgB_this = tBgB[(None, bidy, None, bidz)]
    prefetch_k_tiles = cutlass.min(AB_STAGES - 2, num_k_tiles)
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile in cutlass.range(prefetch_k_tiles, unroll=1):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA_this[(None, k_tile)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b,
                tBgB_this[(None, k_tile)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
        peek_full = ab_consumer.try_wait()
        peek_empty = ab_producer.try_acquire()
        for k_tile in cutlass.range(num_k_tiles):
            if k_tile < num_k_tiles - prefetch_k_tiles:
                ab_empty = ab_producer.acquire_and_advance(peek_empty)
                cute.copy(
                    tma_atom_a,
                    tAgA_this[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_this[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
            ab_full = ab_consumer.wait_and_advance(peek_full)
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block in cutlass.range(num_k_blocks, unroll_full=True):
                coord = (None, None, k_block, ab_full.index)
                cute.gemm(tiled_mma, tCtAcc, tCrA[coord], tCrB[coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_full.release()
            if k_tile + 1 < num_k_tiles - prefetch_k_tiles:
                peek_empty = ab_producer.try_acquire()
            if k_tile + 1 < num_k_tiles:
                peek_full = ab_consumer.try_wait()
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(IO_DTYPE))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)
    if warp_idx == 0:
        ab_producer.tail()


@cute.jit
def gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
    op = tcgen05.MmaF16BF16Op(
        IO_DTYPE,
        ACC_DTYPE,
        MMA_INST_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, MMA_TILER_MNK, IO_DTYPE, AB_STAGES)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, MMA_TILER_MNK, IO_DTYPE, AB_STAGES)
    a_one = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_one = cute.select(b_smem_layout, mode=[0, 1, 2])
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, a_one, MMA_TILER_MNK, tiled_mma
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, b_one, MMA_TILER_MNK, tiled_mma
    )
    m, n, l = c.layout.shape
    grid = (cute.ceil_div(m, MMA_TILER_MNK[0]), cute.ceil_div(n, MMA_TILER_MNK[1]), l)
    gemm_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)


# ---------------------------------------------------------------------------------------
# Stage 1: fused output transform. A cluster of four CTAs handles one block of 128 tiles
# and 128 output channels for one temporal component; CTA x accumulates the four
# coordinates (x, 0..3) into four tensor-memory accumulators, forms the two partial row
# sums t_x[r2] = sum_y AT[r2][y] M[x][y] in fp32, and the cluster combines them through
# distributed shared memory: CTA r writes output pixel (r1, r2) = (r >> 1, r & 1) of each
# tile as y[r1][r2] = sum_x AT[r1][x] t_x[r2].
# ---------------------------------------------------------------------------------------

F_TILER_MNK = (128, 128, 32)
F_INST_MNK = (128, 128, 16)
F_STAGES = 3
F_COORDS = 4  # coordinates per CTA (one row of the 4x4 grid)
F_AB_ELEMS = F_STAGES * F_COORDS * (F_TILER_MNK[0] + F_TILER_MNK[1]) * F_TILER_MNK[2]
F_X_BYTES = 128 * 128 * 4  # one exchange slot
assert 3 * F_X_BYTES <= F_AB_ELEMS * 2, "exchange buffers must fit in the operand ring"


@cute.struct
class FusedSharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, F_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    recv_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32
    smem_ab: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, F_AB_ELEMS], 1024]


@cute.kernel
def wino_fused_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mY: cute.Tensor,  # (pixels, Cout), Cout contiguous
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    m_valid: cutlass.Int32,
    ntiles: cutlass.Int32,
    tiles_w: cutlass.Int32,
    height: cutlass.Int32,
    width: cutlass.Int32,
    pairs: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, bidz = cute.arch.block_idx()
    x_row = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    m_tile = bidx // F_COORDS
    comp = bidz

    smem = utils.SmemAllocator()
    storage = smem.allocate(FusedSharedStorage)
    a_elems = F_STAGES * F_COORDS * F_TILER_MNK[0] * F_TILER_MNK[2]
    sA = storage.smem_ab.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = cute.make_tensor(
        cute.recast_ptr(storage.smem_ab.data_ptr() + a_elems, swizzle_=b_smem_layout.inner),
        b_smem_layout.outer,
    )
    # Epilogue exchange buffers alias the operand ring (used only after every CTA's mainloop):
    # slot 0 = this CTA's own partial sum, slots 1 and 2 = the two peers' partial sums, each
    # [128 rows][128 cols] fp32.
    x_layout = cute.make_layout((3, 128, 128), stride=(128 * 128, 128, 1))
    sX = storage.smem_ab.get_tensor(x_layout, dtype=cutlass.Float32)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier
    )
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    tma_bytes = F_COORDS * (
        cute.size_in_bytes(IO_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(IO_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=F_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=tma_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    recv_bar = storage.recv_mbar.ptr
    if tidx == 0:
        cute.arch.mbarrier_init(recv_bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()

    gA = cute.local_tile(mA_mkl, cute.slice_(F_TILER_MNK, (None, 0, None)), (None, None, None))
    gB = cute.local_tile(mB_nkl, cute.slice_(F_TILER_MNK, (0, None, None)), (None, None, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    # Operand rings: (MMA, MMA_M, MMA_K, STAGE * COORDS) with coordinate fastest within a stage.
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(F_TILER_MNK[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, F_COORDS))

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

    num_k_tiles = cute.size(gA, mode=[3])
    l_base = comp * 16 + x_row * F_COORDS
    prefetch_k_tiles = cutlass.min(F_STAGES - 2, num_k_tiles)
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile in cutlass.range(prefetch_k_tiles, unroll=1):
            ab_empty = ab_producer.acquire_and_advance()
            for y in cutlass.range_constexpr(F_COORDS):
                slot = ab_empty.index * F_COORDS + y
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, m_tile, k_tile, l_base + y)],
                    tAsA[(None, slot)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, bidy, k_tile, l_base + y)],
                    tBsB[(None, slot)],
                    tma_bar_ptr=ab_empty.barrier,
                )
        peek_full = ab_consumer.try_wait()
        peek_empty = ab_producer.try_acquire()
        for k_tile in cutlass.range(num_k_tiles):
            if k_tile < num_k_tiles - prefetch_k_tiles:
                ab_empty = ab_producer.acquire_and_advance(peek_empty)
                for y in cutlass.range_constexpr(F_COORDS):
                    slot = ab_empty.index * F_COORDS + y
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, m_tile, ab_empty.count, l_base + y)],
                        tAsA[(None, slot)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, bidy, ab_empty.count, l_base + y)],
                        tBsB[(None, slot)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
            ab_full = ab_consumer.wait_and_advance(peek_full)
            num_k_blocks = cute.size(tCrA, mode=[2])
            for y in cutlass.range_constexpr(F_COORDS):
                slot = ab_full.index * F_COORDS + y
                acc_y = tCtAcc[(None, None, None, y)]
                if k_tile == 0:
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(num_k_blocks, unroll_full=True):
                    coord = (None, None, k_block, slot)
                    cute.gemm(tiled_mma, acc_y, tCrA[coord], tCrB[coord], acc_y)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_full.release()
            if k_tile + 1 < num_k_tiles - prefetch_k_tiles:
                peek_empty = ab_producer.try_acquire()
            if k_tile + 1 < num_k_tiles:
                peek_full = ab_consumer.try_wait()
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    # Every CTA of the cluster must be done with its operand ring before peers push into it.
    cute.arch.cluster_wait()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    if tidx == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(recv_bar, 2 * F_X_BYTES)

    # Partial sums over the four coordinates of this CTA's row x, 32 columns at a time:
    # t0 = m0 + m1 + m2 and t1 = m1 - m2 - m3 (the column transform). CTA x keeps t[x & 1] for
    # its own output pixel and pushes the others to the CTAs that own the pixels needing them.
    subtiles = F_TILER_MNK[1] // 32
    epi_tiler = ((128, 32),)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    acc0_epi = cute.zipped_divide(tCtAcc[(None, None, None, 0)], epi_tiler)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, acc0_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    sX0_epi = cute.zipped_divide(thr_mma.partition_C(sX[(0, None, None)]), epi_tiler)
    frag_shape = tmem_thr_copy.partition_D(sX0_epi)[None, None, 0].shape
    r0 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
    r1 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
    r2 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
    r3 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
    t0 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
    t1 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
    tDtAcc = [
        tmem_thr_copy.partition_S(cute.zipped_divide(tCtAcc[(None, None, None, y)], epi_tiler))
        for y in range(F_COORDS)
    ]
    row_m = tidx
    keep = x_row % 2
    seg = cute.make_layout(32)
    quad = cute.make_layout(4)
    for i in cutlass.range_constexpr(subtiles):
        cute.copy(tmem_tiled_copy, tDtAcc[0][None, None, i], r0)
        cute.copy(tmem_tiled_copy, tDtAcc[1][None, None, i], r1)
        cute.copy(tmem_tiled_copy, tDtAcc[2][None, None, i], r2)
        cute.copy(tmem_tiled_copy, tDtAcc[3][None, None, i], r3)
        t0.store((r0.load() + r1.load()) + r2.load())
        t1.store((r1.load() - r2.load()) - r3.load())
        chunk = row_m * 128 + i * 32
        # Own partial sum into slot 0.
        own = cute.make_tensor((sX.iterator + chunk).align(16), seg)
        if keep == 0:
            cute.autovec_copy(t0, own)
        else:
            cute.autovec_copy(t1, own)
        # Pushes: (t, peer, slot) by row x (see the sends table in the design note).
        for g in cutlass.range_constexpr(8):
            q0 = cute.make_tensor(t0.iterator + 4 * g, quad).load().bitcast(cutlass.Int32)
            q1 = cute.make_tensor(t1.iterator + 4 * g, quad).load().bitcast(cutlass.Int32)
            v0 = (q0[0], q0[1], q0[2], q0[3])
            v1 = (q1[0], q1[1], q1[2], q1[3])
            dst1 = (sX.iterator + (1 * 128 * 128 + chunk + 4 * g)).align(16)
            dst2 = (sX.iterator + (2 * 128 * 128 + chunk + 4 * g)).align(16)
            if x_row == 0:
                cute.arch.store_async_dsmem(dst1, v1, recv_bar, 1)
            elif x_row == 1:
                cute.arch.store_async_dsmem(dst1, v0, recv_bar, 0)
                cute.arch.store_async_dsmem(dst1, v0, recv_bar, 2)
                cute.arch.store_async_dsmem(dst1, v1, recv_bar, 3)
            elif x_row == 2:
                cute.arch.store_async_dsmem(dst2, v0, recv_bar, 0)
                cute.arch.store_async_dsmem(dst2, v1, recv_bar, 1)
                cute.arch.store_async_dsmem(dst2, v1, recv_bar, 3)
            else:
                cute.arch.store_async_dsmem(dst2, v0, recv_bar, 2)
    acc_full.release()
    cute.arch.mbarrier_wait(recv_bar, 0)

    # Combine: rank r owns pixel (r1, r2) = (r >> 1, r & 1). r1 = 0: t_0 + t_1 + t_2 = own + slot1 + slot2;
    # r1 = 1: t_1 - t_2 - t_3 = slot1 - own - slot2.
    r1_sel = x_row // 2
    r2_sel = x_row % 2
    m_global = m_tile * 128 + row_m
    if m_global < m_valid:
        pair = m_global // ntiles
        t_in_frame = m_global - pair * ntiles
        ti = t_in_frame // tiles_w
        tj = t_in_frame - ti * tiles_w
        prow = 2 * ti + r1_sel
        pcol = 2 * tj + r2_sel
        if prow < height and pcol < width:
            frame = comp * pairs + pair
            pixel = (frame * height + prow) * width + pcol
            n0 = bidy * F_TILER_MNK[1]
            for i in cutlass.range_constexpr(subtiles):
                chunk = row_m * 128 + i * 32
                va = cute.make_tensor((sX.iterator + chunk).align(16), seg).load()
                vb = cute.make_tensor((sX.iterator + (128 * 128 + chunk)).align(16), seg).load()
                vc = cute.make_tensor((sX.iterator + (2 * 128 * 128 + chunk)).align(16), seg).load()
                acc = (va + vb) + vc
                if r1_sel == 1:
                    acc = (vb - va) - vc
                out = cute.make_rmem_tensor(seg.shape, IO_DTYPE)
                out.store(acc.to(IO_DTYPE))
                dst = cute.make_tensor(
                    (mY.iterator + (pixel * cute.size(mY, mode=[1]) + n0 + i * 32)).align(16), seg
                )
                cute.autovec_copy(out, dst)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)
    if warp_idx == 0:
        ab_producer.tail()


@cute.jit
def wino_fused_host(
    a: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    m_valid: cutlass.Int32,
    ntiles: cutlass.Int32,
    tiles_w: cutlass.Int32,
    height: cutlass.Int32,
    width: cutlass.Int32,
    pairs: cutlass.Int32,
    components: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    op = tcgen05.MmaF16BF16Op(
        IO_DTYPE,
        ACC_DTYPE,
        F_INST_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, F_TILER_MNK, IO_DTYPE, F_STAGES * F_COORDS
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, F_TILER_MNK, IO_DTYPE, F_STAGES * F_COORDS
    )
    a_one = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_one = cute.select(b_smem_layout, mode=[0, 1, 2])
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, a_one, F_TILER_MNK, tiled_mma
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, b_one, F_TILER_MNK, tiled_mma
    )
    m_pad, _, _ = a.layout.shape
    n, _, _ = b.layout.shape
    grid = (
        F_COORDS * cute.ceil_div(m_pad, F_TILER_MNK[0]),
        cute.ceil_div(n, F_TILER_MNK[1]),
        components,
    )
    wino_fused_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        y,
        a_smem_layout,
        b_smem_layout,
        m_valid,
        ntiles,
        tiles_w,
        height,
        width,
        pairs,
    ).launch(grid=grid, block=(THREADS, 1, 1), cluster=(F_COORDS, 1, 1), stream=stream)


@functools.lru_cache(maxsize=None)
def _compiled_fused(key: tuple[int, ...]):
    l, m_pad, n, k, pixels = key
    v = torch.empty(l, m_pad, k, dtype=torch.bfloat16, device="cuda")
    u = torch.empty(l, n, k, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(pixels, n, dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        wino_fused_host,
        _as_mkl(v, 1),
        _as_mkl(u, 1),
        _as_rows(y),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        l // 16,
        stream,
    )


def _as_rows(t: torch.Tensor) -> cute.Tensor:
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=64)
    )


def wino_spatial_conv_fused(
    v: torch.Tensor,
    u_nk: torch.Tensor,
    out: torch.Tensor,
    components: int,
    pairs: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Fused-output spatial Winograd: ``v [components*16, M_pad, Cin]`` (M = pairs * ceil(H/2) * ceil(W/2)
    rows, padded to a multiple of 128), ``u_nk [components*16, Cout, Cin]``, ``out [components*pairs, H, W, Cout]``."""
    l, m_pad, k = v.shape
    n = u_nk.shape[1]
    tiles_w = (width + 1) // 2
    ntiles = ((height + 1) // 2) * tiles_w
    m_valid = pairs * ntiles
    assert (
        l == components * 16
        and m_pad % 128 == 0
        and m_pad >= m_valid
        and n % 128 == 0
        and k % 32 == 0
    )
    pixels = components * pairs * height * width
    out_rows = out.view(pixels, n)
    fn = _compiled_fused((l, m_pad, n, k, pixels))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        _as_mkl(v, 1),
        _as_mkl(u_nk, 1),
        _as_rows(out_rows),
        cutlass.Int32(m_valid),
        cutlass.Int32(ntiles),
        cutlass.Int32(tiles_w),
        cutlass.Int32(height),
        cutlass.Int32(width),
        cutlass.Int32(pairs),
        stream,
    )
    return out


def _as_mkl(t: torch.Tensor, leading_dim: int) -> cute.Tensor:
    """torch ``[L, R, C]`` (C contiguous) -> cute tensor ``(R, C, L)`` with the C mode contiguous."""
    return (
        from_dlpack(t.permute(1, 2, 0), assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, divisibility=64)
    )


@functools.lru_cache(maxsize=None)
def _compiled(shape_key: tuple[int, ...]):
    l, m, n, k = shape_key
    v = torch.empty(l, m, k, dtype=torch.bfloat16, device="cuda")
    u = torch.empty(l, n, k, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(l, m, n, dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(gemm_host, _as_mkl(v, 1), _as_mkl(u, 1), _as_mkl(out, 1), stream)


def wino_gemm_cute(
    v: torch.Tensor, u_nk: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``out[l] = v[l] @ u_nk[l].T``: v [L, M, K], u_nk [L, N, K], out [L, M, N] (bf16)."""
    l, m, k = v.shape
    n = u_nk.shape[1]
    assert m % 128 == 0 and n % 128 == 0 and k % 64 == 0, (m, n, k)
    if out is None:
        out = torch.empty(l, m, n, dtype=v.dtype, device=v.device)
    fn = _compiled((l, m, n, k))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(_as_mkl(v, 1), _as_mkl(u_nk, 1), _as_mkl(out, 1), stream)
    return out


def _main_fused(args) -> None:
    import atexit
    import sys
    from pathlib import Path

    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "visual_gen"))
    from bench_wan_vae import _time_cupti  # noqa: E402

    import tensorrt_llm  # noqa: F401  (trtllm::vae_wino_spatial_in)
    from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (
        winograd_spatial_weights,
        winograd_weights,
    )

    if not args.probe:
        atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    cin, cout, h, w = (int(x) for x in args.case.split(","))
    components, pairs = 4, 2
    gen = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.randn(components * pairs, h, w, cin, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn(cout, cin, 3, 3, 3, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    th, tw = (h + 1) // 2, (w + 1) // 2
    m_valid = pairs * th * tw
    m_pad = (m_valid + 127) // 128 * 128
    v_exact = torch.ops.trtllm.vae_wino_spatial_in(frames, components, 0, th)  # [64, m_valid, cin]
    v = torch.zeros(components * 16, m_pad, cin, dtype=torch.bfloat16, device="cuda")
    v[:, :m_valid] = v_exact
    u_nk = winograd_spatial_weights(weight).transpose(1, 2).contiguous()  # [64, cout, cin]
    out = torch.empty(components * pairs, h, w, cout, dtype=torch.bfloat16, device="cuda")
    wino_spatial_conv_fused(v, u_nk, out, components, pairs, h, w)
    torch.cuda.synchronize()
    g = winograd_weights(weight)
    ref = torch.cat(
        [
            F.conv2d(
                frames[k * pairs : (k + 1) * pairs].permute(0, 3, 1, 2).float(),
                g[k].float(),
                None,
                1,
                1,
            )
            for k in range(components)
        ]
    ).permute(0, 2, 3, 1)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"fused rel-L2 vs fp32 conv2d of the component filters: {rel:.3e}")
    if args.probe:
        for _ in range(3):
            wino_spatial_conv_fused(v, u_nk, out, components, pairs, h, w)
        torch.cuda.synchronize()
        return
    conv_flops = 2.0 * components * pairs * h * w * cin * cout * 9
    wino_flops = 2.0 * 64 * m_valid * cin * cout
    t_fused = _time_cupti(
        lambda: wino_spatial_conv_fused(v, u_nk, out, components, pairs, h, w),
        3,
        args.iters,
        use_graph=True,
    )["span"]["median_us"]
    wts = [g[k].contiguous(memory_format=torch.channels_last) for k in range(components)]
    d_comp = [frames[k * pairs : (k + 1) * pairs].permute(0, 3, 1, 2) for k in range(components)]

    def conv_path():
        return [F.conv2d(d_comp[k], wts[k], None, 1, 1) for k in range(components)]

    t_conv = _time_cupti(conv_path, 3, args.iters, use_graph=True)["span"]["median_us"]
    u_kn = u_nk.transpose(1, 2).contiguous()
    t_bmm = _time_cupti(lambda: torch.bmm(v, u_kn), 3, args.iters, use_graph=True)["span"][
        "median_us"
    ]
    print(
        f"fused kernel : {t_fused / 1e3:8.3f} ms  {wino_flops / (t_fused * 1e-6) / 1e12:6.0f} TF/s on Winograd FLOPs,"
        f" {conv_flops / (t_fused * 1e-6) / 1e12:6.0f} TF/s conv-equivalent"
    )
    print(f"cuDNN conv2d : {t_conv / 1e3:8.3f} ms  {conv_flops / (t_conv * 1e-6) / 1e12:6.0f} TF/s")
    print(
        f"cuBLAS bmm   : {t_bmm / 1e3:8.3f} ms  {wino_flops / (t_bmm * 1e-6) / 1e12:6.0f} TF/s (GEMMs alone)"
    )


def _main() -> None:
    import argparse
    import atexit
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "visual_gen"))
    from bench_wan_vae import _time_cupti  # noqa: E402

    parser = argparse.ArgumentParser()
    parser.add_argument("--lmnk", default="64,7296,1024,1024")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--probe", action="store_true", help="no CUPTI: run a few times (for ncu), event timing"
    )
    parser.add_argument(
        "--fused", action="store_true", help="test/bench the fused-output kernel instead"
    )
    parser.add_argument("--case", default="1024,1024,90,160", help="cin,cout,H,W for --fused")
    args = parser.parse_args()
    if args.fused:
        return _main_fused(args)
    if not args.probe:
        atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    l, m, n, k = (int(x) for x in args.lmnk.split(","))
    gen = torch.Generator(device="cuda").manual_seed(0)
    v = torch.randn(l, m, k, generator=gen, device="cuda").to(torch.bfloat16)
    u = (0.05 * torch.randn(l, n, k, generator=gen, device="cuda")).to(torch.bfloat16)
    out = wino_gemm_cute(v, u)
    torch.cuda.synchronize()
    if args.probe:
        u_kn = u.transpose(1, 2).contiguous()
        for name, fn in (
            ("cute", lambda: wino_gemm_cute(v, u, out)),
            ("cuBLAS", lambda: torch.bmm(v, u_kn)),
        ):
            fn()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                fn()
            end.record()
            torch.cuda.synchronize()
            print(f"{name:7s} eager, events: {start.elapsed_time(end) / 3:.3f} ms per call")
        return
    ref = torch.bmm(v, u.transpose(1, 2))
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    print(f"rel-L2 vs torch.bmm: {rel:.3e}")
    flops = 2.0 * l * m * n * k
    t_cute = _time_cupti(lambda: wino_gemm_cute(v, u, out), 3, args.iters, use_graph=True)["span"][
        "median_us"
    ]
    u_kn = u.transpose(1, 2).contiguous()
    t_bmm = _time_cupti(lambda: torch.bmm(v, u_kn), 3, args.iters, use_graph=True)["span"][
        "median_us"
    ]
    print(f"cute  : {t_cute / 1e3:8.3f} ms  {flops / (t_cute * 1e-6) / 1e12:7.0f} TF/s")
    print(f"cuBLAS: {t_bmm / 1e3:8.3f} ms  {flops / (t_bmm * 1e-6) / 1e12:7.0f} TF/s")


if __name__ == "__main__":
    _main()
