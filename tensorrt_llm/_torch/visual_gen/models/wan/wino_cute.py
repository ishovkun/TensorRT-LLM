# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd F(2x2, 3x3) GEMMs for Blackwell (sm100) in CuTe DSL.

Kernels, oldest first (the earlier ones are kept as the measured steps towards the last):

- stage 0, ``wino_gemm_cute``: the 64 batched GEMMs ``M[l] = V[l] @ U[l]`` (l = temporal
  component x tile coordinate) as one plain tcgen05 kernel (calibration).
- stage 1, ``wino_spatial_conv_fused``: GEMMs + output transform in one kernel, a cluster of
  four CTAs (one per coordinate row) exchanging partial sums through shared memory.
- stage 2, ``wino_spatial_conv_persistent``: the persistent, warp-specialized form of stage 1
  with 2-CTA MMA (256x128 tiles). The path for Cout % 256 != 0.
- stage 4, ``wino_spatial_conv_pass2``: 256x256 tiles, cluster of 2, the row combination
  folded into two accumulator passes, the column combination done by combine warps pulling
  finished tile blocks from a global job queue. The path for Cout % 256 == 0.

Tensors (torch, bf16, CUDA): ``v [L, M, K]`` transformed input tiles (K contiguous), ``u
[L, N, K]`` transformed filters (K contiguous; ``winograd_spatial_weights`` gives [L, K, N],
transpose once at load time), outputs N contiguous. K must be a multiple of 64.
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

IO_DTYPE = cutlass.Float16  # transformed data (V, U, partials): fp16 end to end
OUT_DTYPE = cutlass.BFloat16  # output frames
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
    m, n, nl = c.layout.shape
    grid = (cute.ceil_div(m, MMA_TILER_MNK[0]), cute.ceil_div(n, MMA_TILER_MNK[1]), nl)
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
    n_block = bidx // F_COORDS
    m_tile = bidy
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
    prefetch_k_tiles = cutlass.min(F_STAGES - 1, num_k_tiles)
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
                    tBgB[(None, n_block, k_tile, l_base + y)],
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
                        tBgB[(None, n_block, ab_empty.count, l_base + y)],
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
    # Every CTA of the cluster must be done with its operand ring before peers copy into it.
    cute.arch.cluster_wait()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    if tidx == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(recv_bar, 2 * F_X_BYTES)

    # Column transform over this CTA's four coordinates (row x), 32 columns at a time:
    # t0 = m0 + m1 + m2 and t1 = m1 - m2 - m3. CTA x keeps t[x & 1] for its own output pixel; the
    # other partial sum goes to the CTAs owning the pixels that need it. Both are staged in slot 0
    # and pushed with coalesced st.async stores (slot 1 / 2 of the receiver, lower / higher source row).
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
    quad = cute.make_layout(4)
    # Slot addressing: the 16-byte units of a row are XOR-swizzled by (row & 7), so a warp writing
    # one unit of 32 consecutive rows and a warp reading 32 units of one row are both conflict-free.
    swz = row_m % 8
    rowbase = row_m * 128

    for which in cutlass.range_constexpr(2):  # 0: the partial sum for the peers, 1: the own one
        for i in cutlass.range_constexpr(subtiles):
            cute.copy(tmem_tiled_copy, tDtAcc[0][None, None, i], r0)
            cute.copy(tmem_tiled_copy, tDtAcc[1][None, None, i], r1)
            cute.copy(tmem_tiled_copy, tDtAcc[2][None, None, i], r2)
            cute.copy(tmem_tiled_copy, tDtAcc[3][None, None, i], r3)
            t0.store((r0.load() + r1.load()) + r2.load())
            t1.store((r1.load() - r2.load()) - r3.load())
            for g in cutlass.range_constexpr(8):
                unit = i * 8 + g
                dst = cute.make_tensor(
                    (sX.iterator + (rowbase + ((unit ^ swz) * 4))).align(16), quad
                )
                g0 = cute.make_tensor(t0.iterator + 4 * g, quad)
                g1 = cute.make_tensor(t1.iterator + 4 * g, quad)
                if (keep == 0) == (which == 1):
                    cute.autovec_copy(g0, dst)
                else:
                    cute.autovec_copy(g1, dst)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        # Push the staged slot: consecutive threads store consecutive 16-byte units (coalesced on the
        # cluster fabric), completing on the receiver's barrier.
        for i in cutlass.range(tidx * 4, 128 * 128, 512):
            v = cute.make_tensor((sX.iterator + i).align(16), quad).load().bitcast(cutlass.Int32)
            vv = (v[0], v[1], v[2], v[3])
            d1 = (sX.iterator + (128 * 128 + i)).align(16)
            d2 = (sX.iterator + (2 * 128 * 128 + i)).align(16)
            if which == 0:
                if x_row == 0:
                    cute.arch.store_async_dsmem(d1, vv, recv_bar, 1)
                elif x_row == 1:
                    cute.arch.store_async_dsmem(d1, vv, recv_bar, 0)
                    cute.arch.store_async_dsmem(d1, vv, recv_bar, 2)
                elif x_row == 2:
                    cute.arch.store_async_dsmem(d2, vv, recv_bar, 1)
                    cute.arch.store_async_dsmem(d2, vv, recv_bar, 3)
                else:
                    cute.arch.store_async_dsmem(d2, vv, recv_bar, 2)
            else:
                if x_row == 1:
                    cute.arch.store_async_dsmem(d1, vv, recv_bar, 3)
                elif x_row == 2:
                    cute.arch.store_async_dsmem(d2, vv, recv_bar, 0)
        cute.arch.sync_threads()  # slot 0 has been read by every pushing thread before it is rewritten
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
        if (prow < height) & (pcol < width):
            frame = comp * pairs + pair
            pixel = (frame * height + prow) * width + pcol
            n0 = n_block * F_TILER_MNK[1]
            ybase = pixel * cute.size(mY, mode=[1]) + n0
            for unit in cutlass.range_constexpr(32):
                off = rowbase + ((unit ^ swz) * 4)
                va = cute.make_tensor((sX.iterator + off).align(16), quad).load()
                vb = cute.make_tensor((sX.iterator + (128 * 128 + off)).align(16), quad).load()
                vc = cute.make_tensor((sX.iterator + (2 * 128 * 128 + off)).align(16), quad).load()
                acc = (va + vb) + vc
                if r1_sel == 1:
                    acc = (vb - va) - vc
                out = cute.make_rmem_tensor(quad.shape, OUT_DTYPE)
                out.store(acc.to(OUT_DTYPE))
                gdst = cute.make_tensor((mY.iterator + (ybase + unit * 4)).align(8), quad)
                cute.autovec_copy(out, gdst)

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
    # Coordinate row fastest (the cluster), then output-channel blocks of the same tile block so
    # their shared activation tiles hit in L2, then tile blocks, then temporal components.
    grid = (
        F_COORDS * cute.ceil_div(n, F_TILER_MNK[1]),
        cute.ceil_div(m_pad, F_TILER_MNK[0]),
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
    nl, m_pad, n, k, pixels = key
    v = torch.empty(nl, m_pad, k, dtype=torch.float16, device="cuda")
    u = torch.empty(nl, n, k, dtype=torch.float16, device="cuda")
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
        nl // 16,
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
    nl, m_pad, k = v.shape
    n = u_nk.shape[1]
    tiles_w = (width + 1) // 2
    ntiles = ((height + 1) // 2) * tiles_w
    m_valid = pairs * ntiles
    assert (
        nl == components * 16
        and m_pad % 128 == 0
        and m_pad >= m_valid
        and n % 128 == 0
        and k % 32 == 0
    )
    pixels = components * pairs * height * width
    out_rows = out.view(pixels, n)
    fn = _compiled_fused((nl, m_pad, n, k, pixels))
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


# ---------------------------------------------------------------------------------------
# Stage 2: persistent, warp-specialized version of the fused-output kernel.
#
# Warps: 0 = TMA producer, 1 = MMA issuer, 2..9 = epilogue (two groups of four warps; group g
# owns output columns 64g .. 64g+63 of the 128 rows). Each cluster of four CTAs (the four
# coordinate rows) walks the tile blocks (n_block fastest, then m_tile, then component) with a
# stride of the cluster count. The operand ring (3 stages x 32 K) is never aliased, so the TMA
# warp keeps loading the next block while the epilogue runs; the accumulators are released as
# soon as their contents are in registers, so the MMA warp starts the next block during the
# exchange.
#
# Exchange, per 32-column chunk: the partial sum a peer needs is pushed as bf16 straight from
# registers with st.async into the receiver's buffer for the group, two 8 KB slots (lower /
# higher source row), completing on the receiver's barrier; the receiver combines it with its
# own fp32 partial sum from registers and signals "buffer free" to its two senders by a
# remote mbarrier arrive, which gates their next push. Only the kernel start and end use
# cluster-wide barriers.
# ---------------------------------------------------------------------------------------

P_PAIR = 2  # 2: 2-CTA MMA across a CTA pair (256-row tile blocks); 1: single-CTA MMA
P_N = 128  # output channels per tile block; 256 needs 16-bit accumulators (tensor memory)
P_IO = IO_DTYPE  # operand type
P_ACC = ACC_DTYPE
P_TILER_MNK = (128 * P_PAIR, P_N, 64)  # 128 rows per CTA
P_INST_MNK = (128 * P_PAIR, P_N, 16)
P_STAGES = (4 if P_N == 128 else 3) if P_PAIR == 2 else 3
P_CPG = P_N // 64  # 32-column chunks per epilogue group (two groups split the columns)
# 32 accumulator columns per thread: 32 x 32-bit words for fp32; 16 words holding two fp16 each.
P_TMEM_LD_OP = (
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x32)
    if P_ACC is cutlass.Float32
    else tcgen05.Ld32x32bOp(tcgen05.Repetition.x16, tcgen05.Pack.PACK_16b_IN_32b)
)

P_CTA_GROUP = tcgen05.CtaGroup.TWO if P_PAIR == 2 else tcgen05.CtaGroup.ONE
P_CPS = 2  # coordinates per pipeline stage (a stage = 2 coordinates x 64 K)
P_COORDS = 4
P_EPI_WARPS = 8
P_THREADS = 32 * (2 + P_EPI_WARPS)
P_TMA_WARP = 0
P_MMA_WARP = 1
P_AB_ELEMS = (
    P_STAGES * P_CPS * (P_TILER_MNK[0] // P_PAIR + P_TILER_MNK[1] // P_PAIR) * P_TILER_MNK[2]
)
P_SLOT_BYTES = 128 * 32 * 2  # one received chunk: 128 rows x 32 cols bf16
P_SLOT_ELEMS = 128 * 32
P_NBUF = 2  # receive buffers: one per epilogue group (its two chunks are sequenced)


@cute.struct
class PersistentSharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, P_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    recv_mbar: cute.struct.MemRange[cutlass.Int64, P_NBUF]
    free_mbar: cute.struct.MemRange[cutlass.Int64, P_NBUF]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32
    recv: cute.struct.Align[cute.struct.MemRange[P_IO, P_NBUF * 2 * P_SLOT_ELEMS], 128]
    smem_ab: cute.struct.Align[cute.struct.MemRange[P_IO, P_AB_ELEMS], 1024]


@cute.kernel
def wino_persistent_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mY: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    cluster_layout_vmnk: cute.Layout,
    m_valid: cutlass.Int32,
    ntiles: cutlass.Int32,
    tiles_w: cutlass.Int32,
    height: cutlass.Int32,
    width: cutlass.Int32,
    pairs: cutlass.Int32,
    n_blocks: cutlass.Int32,
    m_tiles: cutlass.Int32,
    num_blocks: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    # Cluster (2, 4): x = the MMA pair half (v), y = the coordinate row. Rank = v + 2 * row.
    num_clusters = gdim // P_PAIR
    cluster_id = bidx // P_PAIR
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    x_row = rank // P_PAIR
    v_half = rank % P_PAIR
    is_leader = v_half == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(PersistentSharedStorage)
    a_elems = P_STAGES * P_CPS * (P_TILER_MNK[0] // P_PAIR) * P_TILER_MNK[2]
    sA = storage.smem_ab.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = cute.make_tensor(
        cute.recast_ptr(storage.smem_ab.data_ptr() + a_elems, swizzle_=b_smem_layout.inner),
        b_smem_layout.outer,
    )
    recv_ptr = storage.recv.data_ptr()
    recv_bars = storage.recv_mbar.data_ptr()
    free_bars = storage.free_mbar.data_ptr()

    if warp_idx == P_TMA_WARP:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Both CTAs of the pair load into the leader's full barrier (hence x2); the leader's MMA
    # warp releases both CTAs' empty barriers. Signaling is confined to the pair (B-mode
    # multicast of size 1), never across coordinate rows, whose operands differ.
    tma_bytes = (
        P_PAIR
        * P_CPS
        * (
            cute.size_in_bytes(P_IO, cute.select(a_smem_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(P_IO, cute.select(b_smem_layout, mode=[0, 1, 2]))
        )
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=P_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=tma_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
        mcast_mode_mn=(0, 1),
        defer_sync=True,
    ).make_participants()
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, P_PAIR * 32 * P_EPI_WARPS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )
    # Receivers this CTA pushes to per round: rows 1 and 2 feed three owners, rows 0 and 3 one.
    push_own = (x_row == 1) | (x_row == 2)
    n_receivers = cutlass.Int32(1)
    if push_own:
        n_receivers = cutlass.Int32(3)
    if tidx < P_NBUF:
        cute.arch.mbarrier_init(recv_bars + tidx, 1)
        cute.arch.mbarrier_init(free_bars + tidx, n_receivers)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (1 + P_EPI_WARPS))
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=2,
        is_two_cta=P_PAIR == 2,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    gA = cute.local_tile(mA_mkl, cute.slice_(P_TILER_MNK, (None, 0, None)), (None, None, None))
    gB = cute.local_tile(mB_nkl, cute.slice_(P_TILER_MNK, (0, None, None)), (None, None, None))
    thr_mma = tiled_mma.get_slice(v_half)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(P_TILER_MNK[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, P_COORDS))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
    )
    num_k_tiles = cute.size(gA, mode=[3])

    cute.arch.cluster_wait()

    # ---------------- TMA warp ----------------
    if warp_idx == P_TMA_WARP:
        blk = cluster_id
        while blk < num_blocks:
            n_block = blk % n_blocks
            m_tile = (blk // n_blocks) % m_tiles
            comp = blk // (n_blocks * m_tiles)
            l_base = comp * 16 + x_row * P_COORDS
            for k_tile in cutlass.range(num_k_tiles):
                for cp in cutlass.range_constexpr(P_COORDS // P_CPS):
                    ab_empty = ab_producer.acquire_and_advance()
                    for yy in cutlass.range_constexpr(P_CPS):
                        y = cp * P_CPS + yy
                        slot = ab_empty.index * P_CPS + yy
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, m_tile, k_tile, l_base + y)],
                            tAsA[(None, slot)],
                            tma_bar_ptr=ab_empty.barrier,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, n_block, k_tile, l_base + y)],
                            tBsB[(None, slot)],
                            tma_bar_ptr=ab_empty.barrier,
                        )
            blk += num_clusters
        ab_producer.tail()

    # ---------------- MMA warp ----------------
    if warp_idx == P_MMA_WARP:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(P_ACC)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        if is_leader:
            blk = cluster_id
            while blk < num_blocks:
                acc_pipeline.producer_acquire(acc_state)
                for k_tile in cutlass.range(num_k_tiles):
                    for cp in cutlass.range_constexpr(P_COORDS // P_CPS):
                        ab_full = ab_consumer.wait_and_advance()
                        for yy in cutlass.range_constexpr(P_CPS):
                            y = cp * P_CPS + yy
                            slot = ab_full.index * P_CPS + yy
                            acc_y = tCtAcc[(None, None, None, y)]
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                            for k_block in cutlass.range(
                                cute.size(tCrA, mode=[2]), unroll_full=True
                            ):
                                coord = (None, None, k_block, slot)
                                cute.gemm(tiled_mma, acc_y, tCrA[coord], tCrB[coord], acc_y)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_full.release()
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                blk += num_clusters
            acc_pipeline.producer_tail(acc_state)

    # ---------------- epilogue warps ----------------
    if warp_idx >= 2:
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(P_ACC)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        epi_warp = warp_idx - 2
        group = epi_warp // 4  # output columns 64*group .. +63
        quarter = warp_idx % 4  # tensor-memory lane quarter this warp may access
        lane = tidx % 32
        row_m = quarter * 32 + lane
        acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        group_bar = pipeline.NamedBarrier(barrier_id=2 + group, num_threads=128)

        epi_tiler = ((128, 32),)
        tmem_atom = cute.make_copy_atom(P_TMEM_LD_OP, P_ACC)
        acc0_epi = cute.zipped_divide(tCtAcc[(None, None, None, 0)], epi_tiler)
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, acc0_epi[None, 0])
        tmem_thr_copy = tmem_tiled_copy.get_slice(row_m)
        tDtAcc = [
            tmem_thr_copy.partition_S(cute.zipped_divide(tCtAcc[(None, None, None, y)], epi_tiler))
            for y in range(P_COORDS)
        ]
        # Register fragment shape: what one thread gets from a 128x32 sub-tile (a C-shaped view is
        # partitioned only for its shape; the memory is never touched through it).
        c_view = cute.make_tensor(
            cute.recast_ptr(recv_ptr, dtype=P_ACC), cute.make_layout((128, P_N), stride=(P_N, 1))
        )
        c_epi = cute.zipped_divide(thr_mma.partition_C(c_view), epi_tiler)
        frag_shape = tmem_thr_copy.partition_D(c_epi)[None, None, 0].shape
        fl = cute.make_layout(frag_shape)
        r0 = cute.make_rmem_tensor(frag_shape, P_ACC)
        r1 = cute.make_rmem_tensor(frag_shape, P_ACC)
        r2 = cute.make_rmem_tensor(frag_shape, P_ACC)
        r3 = cute.make_rmem_tensor(frag_shape, P_ACC)
        seg32 = fl
        oct_ = cute.make_layout(8)
        # Own partial sums (fp32 with fp32 accumulators, else the operand type) and the other ones
        # (operand type, ready to push) for this group's chunks.
        own_dtype = cutlass.Float32 if P_ACC is cutlass.Float32 else P_IO
        own = [cute.make_rmem_tensor(seg32, own_dtype) for _ in range(P_CPG)]
        oth = [cute.make_rmem_tensor(seg32, P_IO) for _ in range(P_CPG)]
        keep = x_row % 2
        r1_sel = x_row // 2
        r2_sel = x_row % 2
        leader = (lane == 0) & (quarter == 0)
        blk = cluster_id
        while blk < num_blocks:
            n_block = blk % n_blocks
            m_tile = (blk // n_blocks) % m_tiles
            comp = blk // (n_blocks * m_tiles)
            acc_pipeline.consumer_wait(acc_state)
            for cc in cutlass.range_constexpr(P_CPG):
                chunk = P_CPG * group + cc
                cute.copy(tmem_tiled_copy, tDtAcc[0][None, None, chunk], r0)
                cute.copy(tmem_tiled_copy, tDtAcc[1][None, None, chunk], r1)
                cute.copy(tmem_tiled_copy, tDtAcc[2][None, None, chunk], r2)
                cute.copy(tmem_tiled_copy, tDtAcc[3][None, None, chunk], r3)
                a0 = r0.load().to(cutlass.Float32)
                a1 = r1.load().to(cutlass.Float32)
                a2 = r2.load().to(cutlass.Float32)
                a3 = r3.load().to(cutlass.Float32)
                t0 = (a0 + a1) + a2
                t1 = (a1 - a2) - a3
                tk = t0
                to = t1
                if keep == 1:
                    tk = t1
                    to = t0
                own[cc].store(tk.to(own_dtype))
                oth[cc].store(to.to(P_IO))
            acc_pipeline.consumer_release(acc_state)
            acc_state.advance()

            # Pixel coordinates of this thread's tile row.
            m_global = m_tile * P_TILER_MNK[0] + v_half * 128 + row_m
            pair = m_global // ntiles
            t_in_frame = m_global - pair * ntiles
            ti = t_in_frame // tiles_w
            tj = t_in_frame - ti * tiles_w
            prow = 2 * ti + r1_sel
            pcol = 2 * tj + r2_sel
            valid = (m_global < m_valid) & (prow < height) & (pcol < width)
            frame = comp * pairs + pair
            pixel = (frame * height + prow) * width + pcol
            ybase = pixel * cute.size(mY, mode=[1]) + n_block * P_N

            # Per chunk of this group (buffer = group, rounds alternate parity = chunk): push the
            # partial sums the peers need, receive theirs, combine with the own fp32 one, store,
            # and free the buffer for the peers' next round.
            buf = group
            recv_b = recv_bars + buf
            free_b = free_bars + buf
            rbase = buf * (2 * P_SLOT_ELEMS)
            for cc in cutlass.range_constexpr(P_CPG):
                cute.arch.mbarrier_wait(free_b, (cc % 2) ^ 1)
                if leader:
                    cute.arch.mbarrier_arrive_and_expect_tx(recv_b, 2 * P_SLOT_BYTES)
                for pas in cutlass.range_constexpr(2):
                    do_pass = cutlass.Boolean(True) if pas == 0 else push_own
                    if do_pass:
                        tmp = cute.make_rmem_tensor(seg32, P_IO)
                        if pas == 0:
                            cute.autovec_copy(oth[cc], tmp)
                        else:
                            tmp.store(own[cc].load().to(P_IO))
                        # Register transposition: lane L stores row 8k + L // 4, 16-byte unit L % 4, so
                        # each warp store covers eight full 64-byte rows (coalesced on the cluster fabric).
                        words = tmp.load().bitcast(cutlass.Int32)
                        unit = lane % 4
                        for k in cutlass.range_constexpr(4):
                            src_lane = 8 * k + lane // 4
                            vals = []
                            for w in cutlass.range_constexpr(4):
                                c0 = cute.arch.shuffle_sync_op(words[0 * 4 + w], src_lane)
                                c1 = cute.arch.shuffle_sync_op(words[1 * 4 + w], src_lane)
                                c2 = cute.arch.shuffle_sync_op(words[2 * 4 + w], src_lane)
                                c3 = cute.arch.shuffle_sync_op(words[3 * 4 + w], src_lane)
                                sel = c0
                                if unit == 1:
                                    sel = c1
                                elif unit == 2:
                                    sel = c2
                                elif unit == 3:
                                    sel = c3
                                vals.append(sel)
                            vv = (vals[0], vals[1], vals[2], vals[3])
                            grow = quarter * 32 + src_lane
                            sw = unit ^ (
                                grow % 4
                            )  # 16-byte units swizzled by row: conflict-free reads below
                            d0 = (recv_ptr + (rbase + grow * 32 + sw * 8)).align(16)
                            d1 = (recv_ptr + (rbase + P_SLOT_ELEMS + grow * 32 + sw * 8)).align(16)
                            if pas == 0:
                                if x_row == 0:
                                    cute.arch.store_async_dsmem(d0, vv, recv_b, 1 * P_PAIR + v_half)
                                elif x_row == 1:
                                    cute.arch.store_async_dsmem(d0, vv, recv_b, 0 * P_PAIR + v_half)
                                    cute.arch.store_async_dsmem(d0, vv, recv_b, 2 * P_PAIR + v_half)
                                elif x_row == 2:
                                    cute.arch.store_async_dsmem(d1, vv, recv_b, 1 * P_PAIR + v_half)
                                    cute.arch.store_async_dsmem(d1, vv, recv_b, 3 * P_PAIR + v_half)
                                else:
                                    cute.arch.store_async_dsmem(d1, vv, recv_b, 2 * P_PAIR + v_half)
                            else:
                                if x_row == 1:
                                    cute.arch.store_async_dsmem(d0, vv, recv_b, 3 * P_PAIR + v_half)
                                else:
                                    cute.arch.store_async_dsmem(d1, vv, recv_b, 0 * P_PAIR + v_half)
                cute.arch.mbarrier_wait(recv_b, cc % 2)
                ra = cute.make_rmem_tensor(seg32, P_IO)
                rb = cute.make_rmem_tensor(seg32, P_IO)
                for u in cutlass.range_constexpr(4):
                    off = rbase + row_m * 32 + ((u ^ (row_m % 4)) * 8)
                    cute.autovec_copy(
                        cute.make_tensor((recv_ptr + off).align(16), oct_),
                        cute.make_tensor(ra.iterator + 8 * u, oct_),
                    )
                    cute.autovec_copy(
                        cute.make_tensor((recv_ptr + (off + P_SLOT_ELEMS)).align(16), oct_),
                        cute.make_tensor(rb.iterator + 8 * u, oct_),
                    )
                sa = ra.load().to(cutlass.Float32)
                sb = rb.load().to(cutlass.Float32)
                mine = own[cc].load().to(cutlass.Float32)
                res = (mine + sa) + sb  # r1 = 0: t0 + t1 + t2 (own + lower + higher)
                if r1_sel == 1:
                    res = (sa - mine) - sb  # r1 = 1: t1 - t2 - t3 (lower - own - higher)
                out = cute.make_rmem_tensor(seg32, OUT_DTYPE)
                out.store(res.to(OUT_DTYPE))
                if valid:
                    gdst = cute.make_tensor(
                        (mY.iterator + (ybase + (P_CPG * group + cc) * 32)).align(16), seg32
                    )
                    cute.autovec_copy(out, gdst)
                # All 128 threads of the group are done reading the buffer: tell the two senders.
                group_bar.arrive_and_wait()
                if leader:
                    if x_row == 0:
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=1 * P_PAIR + v_half
                        )
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=2 * P_PAIR + v_half
                        )
                    elif x_row == 1:
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=0 * P_PAIR + v_half
                        )
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=2 * P_PAIR + v_half
                        )
                    elif x_row == 2:
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=1 * P_PAIR + v_half
                        )
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=3 * P_PAIR + v_half
                        )
                    else:
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=1 * P_PAIR + v_half
                        )
                        cute.arch.mbarrier_arrive(
                            free_b, peer_cta_rank_in_cluster=2 * P_PAIR + v_half
                        )
            blk += num_clusters
        # Dealloc tensor memory once every epilogue warp is done.
        pipeline.NamedBarrier(barrier_id=4, num_threads=32 * P_EPI_WARPS).arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    # No CTA may leave while peers can still arrive on its barriers.
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


@cute.jit
def wino_persistent_host(
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
    num_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    op = tcgen05.MmaF16BF16Op(
        P_IO,
        P_ACC,
        P_INST_MNK,
        P_CTA_GROUP,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, P_TILER_MNK, P_IO, P_STAGES * P_CPS)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, P_TILER_MNK, P_IO, P_STAGES * P_CPS)
    a_one = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_one = cute.select(b_smem_layout, mode=[0, 1, 2])
    # The TMA atoms see only the pair (no multicast across coordinate rows); the pipelines see
    # the whole (2, 4) cluster so ranks map to the right coordinates.
    pair_layout_vmnk = cute.tiled_divide(cute.make_layout((P_PAIR, 1, 1)), (tiled_mma.thr_id,))
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((P_PAIR, P_COORDS, 1)), (tiled_mma.thr_id,)
    )
    tma_op = cpasync.CopyBulkTensorTileG2SOp(P_CTA_GROUP)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, a_one, P_TILER_MNK, tiled_mma, pair_layout_vmnk.shape
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, b_one, P_TILER_MNK, tiled_mma, pair_layout_vmnk.shape
    )
    m_pad, _, _ = a.layout.shape
    n, _, _ = b.layout.shape
    n_blocks = cute.ceil_div(n, P_N)
    m_tiles = cute.ceil_div(m_pad, P_TILER_MNK[0])
    num_blocks = n_blocks * m_tiles * components
    wino_persistent_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        y,
        a_smem_layout,
        b_smem_layout,
        cluster_layout_vmnk,
        m_valid,
        ntiles,
        tiles_w,
        height,
        width,
        pairs,
        n_blocks,
        m_tiles,
        num_blocks,
    ).launch(
        grid=(P_PAIR * num_clusters, P_COORDS, 1),
        block=(P_THREADS, 1, 1),
        cluster=(P_PAIR, P_COORDS, 1),
        stream=stream,
    )


@functools.lru_cache(maxsize=None)
def _max_active_clusters() -> int:
    return int(utils.HardwareInfo().get_max_active_clusters(P_PAIR * P_COORDS))


@functools.lru_cache(maxsize=None)
def _compiled_persistent(key: tuple[int, ...]):
    nl, m_pad, n, k, pixels, num_clusters = key
    tdt = torch.float16 if P_IO is cutlass.Float16 else torch.bfloat16
    v = torch.empty(nl, m_pad, k, dtype=tdt, device="cuda")
    u = torch.empty(nl, n, k, dtype=tdt, device="cuda")
    y = torch.empty(pixels, n, dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    one = cutlass.Int32(1)
    return cute.compile(
        wino_persistent_host,
        _as_mkl(v, 1),
        _as_mkl(u, 1),
        _as_rows(y),
        one,
        one,
        one,
        one,
        one,
        one,
        nl // 16,
        num_clusters,
        stream,
    )


def wino_spatial_conv_persistent(
    v: torch.Tensor,
    u_nk: torch.Tensor,
    out: torch.Tensor,
    components: int,
    pairs: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Persistent variant of :func:`wino_spatial_conv_fused`; any number of rows (the TMA loads
    zero-fill past the end and the stores are guarded)."""
    nl, m_pad, k = v.shape
    n = u_nk.shape[1]
    tiles_w = (width + 1) // 2
    ntiles = ((height + 1) // 2) * tiles_w
    m_valid = pairs * ntiles
    assert nl == components * 16 and m_pad >= m_valid and n % P_N == 0 and k % 64 == 0
    pixels = components * pairs * height * width
    num_blocks = (n // P_N) * -(-m_pad // P_TILER_MNK[0]) * components
    num_clusters = min(_max_active_clusters(), num_blocks)
    out_rows = out.view(pixels, n)
    fn = _compiled_persistent((nl, m_pad, n, k, pixels, num_clusters))
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


# ---------------------------------------------------------------------------------------
# Stage 4: 256x256 tiles, two passes per block, no cluster exchange.
#
# The per-SM limiter of the 256x128 kernel is shared-memory traffic per MMA cycle (TMA
# writes plus the MMA's operand reads); a 256x256 tile halves it per FLOP, which is the
# shape cuDNN and the CuTe DSL reference GEMM reach ~1400 TF/s with. Tensor memory holds
# two 128x256 fp32 accumulators, so a block (one coordinate row x of 4 column coordinates)
# runs its four coordinates one at a time, each over all of K, in the order 1, 2, 0, 3:
# y = 1 and 2 fill the two accumulators, the epilogue rewrites them in place as a1 + a2
# and a1 - a2 (fp32 read-modify-write in tensor memory), then y = 0 accumulates onto the
# first and y = 3, with its filter pre-negated by the caller, onto the second, which
# leaves exactly t0 = a0 + a1 + a2 and t1 = a1 - a2 - a3: the row combinations, written
# as bf16 partials, two per (tile, x), with no extra GEMMs and fp32 accumulation
# throughout. The accumulator handshake is per accumulator, so draining one overlaps the
# MMA on the other. The four coordinate rows of a tile block run on neighbouring CTA pairs
# (row fastest in the block order); each CTA writes its two partials, then bumps a
# per-(tile block, pair half) counter (block order: output-channel block fastest, then tile
# block, then coordinate row, so the filters of one row stay hot in L2); the CTA that
# arrives last pushes the finished tile
# block onto a global job queue. Every CTA has a combine warpgroup that pulls jobs from that
# queue whenever it is free: it brings all four rows' partials into shared memory with bulk
# async copies, 4 tile rows x 256 columns at a time through a 4-deep ring, and writes the
# output pixels y[a][b] = sum_x A^T[a][x] t[x][b], a warp per pixel so the stores are 512
# contiguous bytes. The epilogue does the same work in every CTA and the combine load spreads
# over all SMs, so the pairs stay in step and the operand reuse in L2 holds. The partials
# live as [(comp, x, b)][output-channel block][row][256], so a slab is one contiguous copy.
# Nobody waits on anyone (a combine warpgroup may briefly poll for a job that is being
# pushed). The cluster is just the MMA pair.
# ---------------------------------------------------------------------------------------

Q_N = 256
Q_K = 64
Q_TILER_MNK = (128 * P_PAIR, Q_N, Q_K)
Q_INST_MNK = (128 * P_PAIR, Q_N, 16)
Q_STAGES = 5  # 5 x 32 KB; the sixth stage's space holds the partial staging buffers
Q_CPS = 1  # one coordinate per stage
Q_ACCS = 2  # accumulators in tensor memory (2 x 128 x 256 fp32 = all 512 columns)
Q_ORDER = (1, 2, 0, 3)  # coordinate order; coordinate slot s uses accumulator s % 2
Q_EPI_WARPS = 8  # two warps per tensor-memory lane quarter, each owning half the columns
Q_TMA_WARP = 8
Q_MMA_WARP = 9
Q_COMBINE_WARP0 = 12  # warps 12-15: the combine warpgroup
Q_COMBINE_THREADS = 128
Q_THREADS = (
    32 * 16
)  # epilogue warpgroups 0-1, producer warpgroup 2 (TMA, MMA, 2 idle), combine warpgroup 3
Q_AB_ELEMS = Q_STAGES * Q_CPS * (Q_TILER_MNK[0] // P_PAIR + Q_TILER_MNK[1] // P_PAIR) * Q_K
Q_CHUNKS = Q_N // 32
Q_CROWS = 4  # tile rows per combine chunk: a slab (4 rows x 256 cols bf16) is one 2 KB bulk copy
Q_NCHUNK = 128 // Q_CROWS
Q_SLAB_ELEMS = Q_CROWS * Q_N
Q_STAGE_ELEMS = 2 * P_COORDS * Q_SLAB_ELEMS  # 8 slabs: 4 coordinate rows x 2 partials (16 KB)
Q_RING = 4  # staging buffers in flight (64 KB)
Q_JOB_SPLIT = 4  # combine jobs per finished (tile block, pair half): 32 rows each
Q_JOB_CHUNKS = Q_NCHUNK // Q_JOB_SPLIT
Q_CHUNKS_PER_THREAD = Q_CHUNKS // (Q_EPI_WARPS // 4)


@cute.struct
class PassSharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, Q_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, Q_ACCS * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32
    join_mbar: cute.struct.MemRange[cutlass.Int64, Q_RING]
    job_slot: cutlass.Int32  # the job claimed by the combine warpgroup, for all its threads
    stage: cute.struct.Align[cute.struct.MemRange[IO_DTYPE, Q_RING * Q_STAGE_ELEMS], 128]
    smem_ab: cute.struct.Align[cute.struct.MemRange[IO_DTYPE, Q_AB_ELEMS], 1024]


@cute.jit
def _issue_slabs(
    k, buf, tidx, join_bars, bulk_atom, slab_base, n_blocks, m_alloc, stage_ptr, slab_layout
):
    """Warp 0 pulls the 8 slabs (4 coordinate rows x 2 partials) of row chunk ``k`` into staging
    buffer ``buf``; ``slab_base`` points at (x = 0, b = 0, first row of this CTA's tile half)."""
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(join_bars + buf, Q_STAGE_ELEMS * 2)
        for sidx in cutlass.range_constexpr(2 * P_COORDS):
            goff = (sidx * n_blocks * m_alloc + k * Q_CROWS) * Q_N
            gsrc = cute.make_tensor((slab_base + goff).align(16), slab_layout)
            sdst = cute.make_tensor(
                (stage_ptr + (buf * Q_STAGE_ELEMS + sidx * Q_SLAB_ELEMS)).align(128), slab_layout
            )
            cute.copy(bulk_atom, gsrc, sdst, mbar_ptr=join_bars + buf)


@cute.kernel
def wino_pass_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mT: cute.Tensor,
    mY: cute.Tensor,
    mCount: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    cluster_layout_vmnk: cute.Layout,
    m_valid: cutlass.Int32,
    m_pad: cutlass.Int32,
    m_alloc: cutlass.Int32,
    n_blocks: cutlass.Int32,
    m_tiles: cutlass.Int32,
    num_blocks: cutlass.Int32,
    ntiles: cutlass.Int32,
    tiles_w: cutlass.Int32,
    height: cutlass.Int32,
    width: cutlass.Int32,
    pairs: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    num_clusters = gdim // P_PAIR
    cluster_id = bidx // P_PAIR
    v_half = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader = v_half == 0

    smem = utils.SmemAllocator()
    storage = smem.allocate(PassSharedStorage)
    a_elems = Q_STAGES * Q_CPS * (Q_TILER_MNK[0] // P_PAIR) * Q_K
    sA = storage.smem_ab.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = cute.make_tensor(
        cute.recast_ptr(storage.smem_ab.data_ptr() + a_elems, swizzle_=b_smem_layout.inner),
        b_smem_layout.outer,
    )

    if warp_idx == Q_TMA_WARP:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    tma_bytes = (
        P_PAIR
        * Q_CPS
        * (
            cute.size_in_bytes(IO_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(IO_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))
        )
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=Q_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=tma_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    ).make_participants()
    # The epilogue warps synchronize on a named barrier and one thread per CTA releases.
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        num_stages=Q_ACCS,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, P_PAIR),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )
    # Global job queue layout in mCount: [arrival counters | head | tail | jobs...].
    n_counters = num_blocks // 2  # tile blocks x pair halves
    n_jobs = n_counters * Q_JOB_SPLIT
    q_head = mCount.iterator + n_counters
    q_tail = q_head + 1
    q_jobs = q_head + 2
    job_slot_t = cute.make_tensor(storage.job_slot.ptr, cute.make_layout(1))
    if tidx < Q_RING:
        cute.arch.mbarrier_init(storage.join_mbar.data_ptr() + tidx, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (1 + Q_EPI_WARPS))
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=0,
        is_two_cta=P_PAIR == 2,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    gA = cute.local_tile(mA_mkl, cute.slice_(Q_TILER_MNK, (None, 0, None)), (None, None, None))
    gB = cute.local_tile(mB_nkl, cute.slice_(Q_TILER_MNK, (0, None, None)), (None, None, None))
    thr_mma = tiled_mma.get_slice(v_half)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(Q_TILER_MNK[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, Q_ACCS))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
    )
    num_k_tiles = cute.size(gA, mode=[3])
    blocks_per_x = n_blocks * m_tiles

    cute.arch.cluster_wait()

    # ---------------- TMA warp ----------------
    if warp_idx == Q_TMA_WARP:
        blk = cluster_id
        while blk < num_blocks:
            n_block = blk % n_blocks
            m_tile = (blk // n_blocks) % m_tiles
            x_row = (blk // blocks_per_x) % P_COORDS
            comp = blk // (blocks_per_x * P_COORDS)
            l_base = comp * 16 + x_row * P_COORDS
            for sl in cutlass.range_constexpr(P_COORDS):
                l_coord = l_base + Q_ORDER[sl]
                for k_tile in cutlass.range(num_k_tiles):
                    ab_empty = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, m_tile, k_tile, l_coord)],
                        tAsA[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, n_block, k_tile, l_coord)],
                        tBsB[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
            blk += num_clusters
        ab_producer.tail()

    # ---------------- MMA warp ----------------
    if warp_idx == Q_MMA_WARP:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, Q_ACCS)
        if is_leader:
            blk = cluster_id
            while blk < num_blocks:
                for sl in cutlass.range_constexpr(P_COORDS):
                    acc_pipeline.producer_acquire(acc_state)
                    acc_y = tCtAcc[(None, None, None, sl % Q_ACCS)]
                    for k_tile in cutlass.range(num_k_tiles):
                        ab_full = ab_consumer.wait_and_advance()
                        if cutlass.const_expr(sl < Q_ACCS):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        else:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        for k_block in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                            coord = (None, None, k_block, ab_full.index)
                            cute.gemm(tiled_mma, acc_y, tCrA[coord], tCrB[coord], acc_y)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_full.release()
                    acc_pipeline.producer_commit(acc_state)
                    acc_state.advance()
                blk += num_clusters
            acc_pipeline.producer_tail(acc_state)

    # ---------------- epilogue warps ----------------
    if warp_idx < Q_EPI_WARPS:
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        quarter = warp_idx % 4
        col_half = warp_idx // 4
        lane = tidx % 32
        row_m = quarter * 32 + lane
        acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, Q_ACCS)
        epi_tiler = ((128, 32),)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), ACC_DTYPE)
        acc0_epi = cute.zipped_divide(tCtAcc[(None, None, None, 0)], epi_tiler)
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, acc0_epi[None, 0])
        tmem_thr_copy = tmem_tiled_copy.get_slice(row_m)
        tDtAcc = [
            tmem_thr_copy.partition_S(cute.zipped_divide(tCtAcc[(None, None, None, yy)], epi_tiler))
            for yy in range(Q_ACCS)
        ]
        st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), ACC_DTYPE)
        tmem_st_copy = tcgen05.make_tmem_copy(st_atom, acc0_epi[None, 0])
        tmem_st_thr = tmem_st_copy.get_slice(row_m)
        tStAcc = [
            tmem_st_thr.partition_D(cute.zipped_divide(tCtAcc[(None, None, None, yy)], epi_tiler))
            for yy in range(Q_ACCS)
        ]
        c_view = cute.make_tensor(
            cute.recast_ptr(storage.smem_ab.data_ptr(), dtype=ACC_DTYPE),
            cute.make_layout((128, Q_N), stride=(Q_N, 1)),
        )
        c_epi = cute.zipped_divide(thr_mma.partition_C(c_view), epi_tiler)
        frag_shape = tmem_thr_copy.partition_D(c_epi)[None, None, 0].shape
        seg32 = cute.make_layout(frag_shape)
        r0 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
        r1 = cute.make_rmem_tensor(frag_shape, ACC_DTYPE)
        epi_bar = pipeline.NamedBarrier(barrier_id=5, num_threads=32 * Q_EPI_WARPS)
        epi_leader = tidx == 0
        n_cols = cute.size(mT, mode=[1])
        blk = cluster_id
        while blk < num_blocks:
            n_block = blk % n_blocks
            m_tile = (blk // n_blocks) % m_tiles
            x_row = (blk // blocks_per_x) % P_COORDS
            comp = blk // (blocks_per_x * P_COORDS)
            m_global = m_tile * Q_TILER_MNK[0] + v_half * 128 + row_m
            valid = m_global < m_valid

            # a1 ready: prefetch its first chunk while the MMA still works on a2.
            acc_pipeline.consumer_wait(acc_state)
            st_a1 = acc_state.clone()
            acc_state.advance()
            cute.copy(tmem_tiled_copy, tDtAcc[0][None, None, col_half * Q_CHUNKS_PER_THREAD], r0)
            # a2 ready: rewrite both accumulators as a1 + a2 and a1 - a2, one chunk at a time
            # (two fp32 fragments live: the kernel runs 512 threads at 128 registers).
            acc_pipeline.consumer_wait(acc_state)
            st_a2 = acc_state.clone()
            acc_state.advance()
            for c in cutlass.range_constexpr(Q_CHUNKS_PER_THREAD):
                cc = col_half * Q_CHUNKS_PER_THREAD + c
                if cutlass.const_expr(c > 0):
                    cute.copy(tmem_tiled_copy, tDtAcc[0][None, None, cc], r0)
                cute.copy(tmem_tiled_copy, tDtAcc[1][None, None, cc], r1)
                a1 = r0.load()
                a2 = r1.load()
                r0.store(a1 + a2)
                r1.store(a1 - a2)
                cute.copy(tmem_st_copy, r0, tStAcc[0][None, None, cc])
                cute.copy(tmem_st_copy, r1, tStAcc[1][None, None, cc])
            cute.arch.fence_view_async_tmem_store()
            epi_bar.arrive_and_wait()
            if epi_leader:
                acc_pipeline.consumer_release(st_a1)
                acc_pipeline.consumer_release(st_a2)

            # t0 = a0 + a1 + a2 ready (y = 0 accumulated on top), then t1 = a1 - a2 - a3.
            # Each is converted to bf16 in registers and released before its stores, so the
            # drain overlaps the MMA on the other accumulator.
            for acc_i in cutlass.range_constexpr(Q_ACCS):
                acc_pipeline.consumer_wait(acc_state)
                outs = [None] * Q_CHUNKS_PER_THREAD
                for c in cutlass.range_constexpr(Q_CHUNKS_PER_THREAD):
                    cc = col_half * Q_CHUNKS_PER_THREAD + c
                    cute.copy(tmem_tiled_copy, tDtAcc[acc_i][None, None, cc], r0)
                    o = cute.make_rmem_tensor(seg32, IO_DTYPE)
                    o.store(r0.load().to(IO_DTYPE))
                    outs[c] = o
                epi_bar.arrive_and_wait()
                if epi_leader:
                    acc_pipeline.consumer_release(acc_state)
                acc_state.advance()
                cxb = (comp * P_COORDS + x_row) * 2 + acc_i
                trow = ((cxb * n_blocks + n_block) * m_alloc + m_global) * Q_N + col_half * (
                    Q_CHUNKS_PER_THREAD * 32
                )
                for c in cutlass.range_constexpr(Q_CHUNKS_PER_THREAD):
                    if valid:
                        g = cute.make_tensor((mT.iterator + (trow + c * 32)).align(16), seg32)
                        cute.autovec_copy(outs[c], g)

            # Join of the four coordinate rows of this tile block and pair half: publish the
            # partials, count the arrival; the last arrival pushes the job onto the queue. The
            # named barrier makes every thread's stores cumulative with the leader's fence.
            epi_bar.arrive_and_wait()
            if epi_leader:
                cute.arch.fence_acq_rel_gpu()
                cidx = ((comp * m_tiles + m_tile) * n_blocks + n_block) * P_PAIR + v_half
                seen = cute.arch.atomic_add(
                    mCount.iterator + cidx, cutlass.Int32(1), sem="acq_rel", scope="gpu"
                )
                if seen == P_COORDS - 1:
                    mCount[cidx] = cutlass.Int32(0)  # self-resetting counter
                    slot = cute.arch.atomic_add(
                        q_tail, cutlass.Int32(Q_JOB_SPLIT), sem="relaxed", scope="gpu"
                    )
                    for part in cutlass.range_constexpr(Q_JOB_SPLIT):
                        cute.arch.atomic_exch(
                            q_jobs + (slot + part),
                            cidx * Q_JOB_SPLIT + part + 1,
                            sem="release",
                            scope="gpu",
                        )
            blk += num_clusters
        pipeline.NamedBarrier(barrier_id=4, num_threads=32 * Q_EPI_WARPS).arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    # ---------------- combine warps ----------------
    if warp_idx >= Q_COMBINE_WARP0:
        ctid = tidx - 32 * Q_COMBINE_WARP0
        n_cols = cute.size(mT, mode=[1])
        seg8 = cute.make_layout(8)
        slab_layout = cute.make_layout(Q_SLAB_ELEMS)
        bulk_atom = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(), IO_DTYPE, num_bits_per_copy=Q_SLAB_ELEMS * 16
        )
        join_bars = storage.join_mbar.data_ptr()
        stage_ptr = storage.stage.data_ptr()
        cbar = pipeline.NamedBarrier(barrier_id=6, num_threads=Q_COMBINE_THREADS)
        # Thread -> (row within the 4-row chunk, 8-column group): a warp covers one row's 256
        # columns, so its output stores are one pixel's 512 contiguous bytes.
        crow = ctid // 32
        cgrp = (ctid % 32) * 8
        n_done = cutlass.Int32(0)  # jobs this warpgroup has run; the ring parity follows it
        more = cutlass.Boolean(True)
        while more:
            # Claim the next job; poll until its producer has published it.
            if ctid == 0:
                idx = cute.arch.atomic_add(q_head, cutlass.Int32(1), sem="relaxed", scope="gpu")
                job = cutlass.Int32(0)
                if idx < n_jobs:
                    job = cute.arch.atomic_add(
                        q_jobs + idx, cutlass.Int32(0), sem="acquire", scope="gpu"
                    )
                    while job == 0:
                        job = cute.arch.atomic_add(
                            q_jobs + idx, cutlass.Int32(0), sem="acquire", scope="gpu"
                        )
                job_slot_t[0] = job
            cbar.arrive_and_wait()
            job = job_slot_t[0]
            cbar.arrive_and_wait()
            if job == 0:
                more = cutlass.Boolean(False)
            else:
                cute.arch.fence_acq_rel_gpu()
                cute.arch.fence_proxy("async.global")
                part = (job - 1) % Q_JOB_SPLIT
                cidx = (job - 1) // Q_JOB_SPLIT
                jv = cidx % P_PAIR
                tb = cidx // P_PAIR
                n_block = tb % n_blocks
                m_tile = (tb // n_blocks) % m_tiles
                comp = tb // (n_blocks * m_tiles)
                slab_row0 = m_tile * Q_TILER_MNK[0] + jv * 128 + part * (Q_JOB_CHUNKS * Q_CROWS)
                cxb0 = comp * P_COORDS * 2
                slab_base = mT.iterator + ((cxb0 * n_blocks + n_block) * m_alloc + slab_row0) * Q_N
                if warp_idx == Q_COMBINE_WARP0:
                    for k0 in cutlass.range_constexpr(Q_RING):
                        _issue_slabs(
                            k0,
                            k0,
                            tidx,
                            join_bars,
                            bulk_atom,
                            slab_base,
                            n_blocks,
                            m_alloc,
                            stage_ptr,
                            slab_layout,
                        )
                for k in cutlass.range_constexpr(Q_JOB_CHUNKS):
                    buf = k % Q_RING
                    m_row = slab_row0 + k * Q_CROWS + crow
                    r_valid = m_row < m_valid
                    pair = m_row // ntiles
                    t_in_frame = m_row - pair * ntiles
                    ti = t_in_frame // tiles_w
                    tj = t_in_frame - ti * tiles_w
                    frame = comp * pairs + pair
                    prow0 = 2 * ti
                    pcol0 = 2 * tj
                    col = n_block * Q_N + cgrp
                    cute.arch.mbarrier_wait(join_bars + buf, (k // Q_RING) % 2)
                    for b in cutlass.range_constexpr(2):
                        tx = [None] * P_COORDS
                        for x in cutlass.range_constexpr(P_COORDS):
                            src = cute.make_tensor(
                                (
                                    stage_ptr
                                    + (
                                        buf * Q_STAGE_ELEMS
                                        + (x * 2 + b) * Q_SLAB_ELEMS
                                        + crow * Q_N
                                        + cgrp
                                    )
                                ).align(16),
                                seg8,
                            )
                            tmp = cute.make_rmem_tensor(seg8, IO_DTYPE)
                            cute.autovec_copy(src, tmp)
                            tx[x] = tmp.load().to(ACC_DTYPE)
                        y0 = cute.make_rmem_tensor(seg8, OUT_DTYPE)
                        y1 = cute.make_rmem_tensor(seg8, OUT_DTYPE)
                        y0.store(((tx[0] + tx[1]) + tx[2]).to(OUT_DTYPE))
                        y1.store(((tx[1] - tx[2]) - tx[3]).to(OUT_DTYPE))
                        pcol = pcol0 + b
                        if r_valid & (pcol < width):
                            if prow0 < height:
                                pix = (frame * height + prow0) * width + pcol
                                g = cute.make_tensor(
                                    (mY.iterator + (pix * n_cols + col)).align(16), seg8
                                )
                                cute.autovec_copy(y0, g)
                            if prow0 + 1 < height:
                                pix = (frame * height + prow0 + 1) * width + pcol
                                g = cute.make_tensor(
                                    (mY.iterator + (pix * n_cols + col)).align(16), seg8
                                )
                                cute.autovec_copy(y1, g)
                    cbar.arrive_and_wait()  # everyone is done reading this buffer before it is refilled
                    if cutlass.const_expr(k + Q_RING < Q_JOB_CHUNKS):
                        if warp_idx == Q_COMBINE_WARP0:
                            cute.arch.fence_proxy("async.shared", space="cta")
                            _issue_slabs(
                                k + Q_RING,
                                buf,
                                tidx,
                                join_bars,
                                bulk_atom,
                                slab_base,
                                n_blocks,
                                m_alloc,
                                stage_ptr,
                                slab_layout,
                            )
                n_done = n_done + 1

    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


@cute.jit
def wino_pass_host(
    a: cute.Tensor,
    b: cute.Tensor,
    t: cute.Tensor,
    y: cute.Tensor,
    counters: cute.Tensor,
    m_valid: cutlass.Int32,
    m_alloc: cutlass.Int32,
    ntiles: cutlass.Int32,
    tiles_w: cutlass.Int32,
    height: cutlass.Int32,
    width: cutlass.Int32,
    pairs: cutlass.Int32,
    components: cutlass.Constexpr[int],
    num_clusters: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    op = tcgen05.MmaF16BF16Op(
        IO_DTYPE,
        ACC_DTYPE,
        Q_INST_MNK,
        P_CTA_GROUP,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, Q_TILER_MNK, IO_DTYPE, Q_STAGES * Q_CPS
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, Q_TILER_MNK, IO_DTYPE, Q_STAGES * Q_CPS
    )
    a_one = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_one = cute.select(b_smem_layout, mode=[0, 1, 2])
    cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((P_PAIR, 1, 1)), (tiled_mma.thr_id,))
    tma_op = cpasync.CopyBulkTensorTileG2SOp(P_CTA_GROUP)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, a_one, Q_TILER_MNK, tiled_mma, cluster_layout_vmnk.shape
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, b_one, Q_TILER_MNK, tiled_mma, cluster_layout_vmnk.shape
    )
    m_pad, _, _ = a.layout.shape
    n, _, _ = b.layout.shape
    n_blocks = cute.ceil_div(n, Q_N)
    m_tiles = cute.ceil_div(m_pad, Q_TILER_MNK[0])
    num_blocks = n_blocks * m_tiles * P_COORDS * components
    wino_pass_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        t,
        y,
        counters,
        a_smem_layout,
        b_smem_layout,
        cluster_layout_vmnk,
        m_valid,
        m_pad,
        m_alloc,
        n_blocks,
        m_tiles,
        num_blocks,
        ntiles,
        tiles_w,
        height,
        width,
        pairs,
    ).launch(
        grid=(P_PAIR * num_clusters, 1, 1),
        block=(Q_THREADS, 1, 1),
        cluster=(P_PAIR, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
    )


@functools.lru_cache(maxsize=None)
def _max_active_pairs() -> int:
    return int(utils.HardwareInfo().get_max_active_clusters(P_PAIR))


@functools.lru_cache(maxsize=None)
def _compiled_pass(key: tuple[int, ...]):
    nl, m_pad, n, k, pixels, num_clusters = key
    v = torch.empty(nl, m_pad, k, dtype=torch.float16, device="cuda")
    u = torch.empty(nl, n, k, dtype=torch.float16, device="cuda")
    t = torch.empty(nl // 2 * m_pad, n, dtype=torch.float16, device="cuda")
    y = torch.empty(pixels, n, dtype=torch.bfloat16, device="cuda")
    counters = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    one = cutlass.Int32(1)
    return cute.compile(
        wino_pass_host,
        _as_mkl(v, 1),
        _as_mkl(u, 1),
        _as_rows(t),
        _as_rows(y),
        from_dlpack(counters, assumed_align=16),
        one,
        one,
        one,
        one,
        one,
        one,
        one,
        nl // 16,
        num_clusters,
        stream,
    )


@functools.lru_cache(maxsize=None)
def _pass_counters(n: int, device_index: int) -> torch.Tensor:
    """Arrival counters (one per tile block and pair half, self-resetting), then the job
    queue: head, tail and one slot per job (zeroed by the caller before each launch)."""
    return torch.zeros(n, dtype=torch.int32, device=torch.device("cuda", device_index))


def pass2_workspace_numel(components: int, rows: int, cout: int) -> int:
    """Elements of the bf16 partials workspace of :func:`wino_spatial_conv_pass2` (rows padded
    to the 256-row tile)."""
    return components * 8 * (-(-rows // Q_TILER_MNK[0]) * Q_TILER_MNK[0]) * cout


def wino_spatial_conv_pass2(
    v: torch.Tensor,
    u_nk: torch.Tensor,
    out: torch.Tensor,
    components: int,
    pairs: int,
    height: int,
    width: int,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Spatial Winograd conv from the transformed tiles: ``v`` [components*16, rows, Cin]
    (coordinate l = comp*16 + x*4 + y, rows >= pairs * ntiles), ``u_nk`` [components*16, Cout,
    Cin] **with the y = 3 coordinate negated** (see :func:`winograd_pass_filters`); writes the
    output frames ``out`` [components*pairs, H, W, Cout]. ``workspace`` bf16 of
    ``pass2_workspace_numel(components, rows, Cout)`` elements holds the row partials in flight
    (allocated here if not given). Cout % 256 == 0, Cin % 64 == 0."""
    nl, m_pad, k = v.shape
    n = u_nk.shape[1]
    tiles_w = (width + 1) // 2
    ntiles = ((height + 1) // 2) * tiles_w
    m_valid = pairs * ntiles
    assert nl == components * 16 and n % Q_N == 0 and k % Q_K == 0
    assert m_valid <= m_pad and tuple(out.shape) == (components * pairs, height, width, n)
    m_tiles = -(-m_pad // Q_TILER_MNK[0])
    m_alloc = m_tiles * Q_TILER_MNK[0]
    numel = pass2_workspace_numel(components, m_pad, n)
    t = workspace if workspace is not None else torch.empty(numel, dtype=v.dtype, device=v.device)
    assert t.numel() == numel
    num_blocks = (n // Q_N) * m_tiles * P_COORDS * components
    num_clusters = min(_max_active_pairs(), num_blocks)
    n_counters = components * m_tiles * (n // Q_N) * P_PAIR
    counters = _pass_counters(n_counters + 2 + n_counters * Q_JOB_SPLIT, v.device.index)
    counters[n_counters:].zero_()  # queue head, tail and job slots start empty every call
    pixels = components * pairs * height * width
    fn = _compiled_pass((nl, m_pad, n, k, pixels, num_clusters))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        _as_mkl(v, 1),
        _as_mkl(u_nk, 1),
        _as_rows(t.view(nl // 2 * m_alloc, n)),
        _as_rows(out.view(pixels, n)),
        from_dlpack(counters, assumed_align=16),
        cutlass.Int32(m_valid),
        cutlass.Int32(m_alloc),
        cutlass.Int32(ntiles),
        cutlass.Int32(tiles_w),
        cutlass.Int32(height),
        cutlass.Int32(width),
        cutlass.Int32(pairs),
        stream,
    )
    return out


def winograd_pass_filters(u_nk: torch.Tensor) -> torch.Tensor:
    """Filters for :func:`wino_spatial_gemm_pass2`: ``u_nk`` [components*16, Cout, Cin] with the
    y = 3 coordinate of every row negated, so the second pass accumulates -a3."""
    u = u_nk.clone()
    u.view(-1, 4, 4, *u.shape[1:])[:, :, 3].neg_()
    return u


def _as_mkl(t: torch.Tensor, leading_dim: int) -> cute.Tensor:
    """torch ``[L, R, C]`` (C contiguous) -> cute tensor ``(R, C, L)`` with the C mode contiguous."""
    return (
        from_dlpack(t.permute(1, 2, 0), assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, divisibility=64)
    )


@functools.lru_cache(maxsize=None)
def _compiled(shape_key: tuple[int, ...]):
    nl, m, n, k = shape_key
    v = torch.empty(nl, m, k, dtype=torch.float16, device="cuda")
    u = torch.empty(nl, n, k, dtype=torch.float16, device="cuda")
    out = torch.empty(nl, m, n, dtype=torch.float16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(gemm_host, _as_mkl(v, 1), _as_mkl(u, 1), _as_mkl(out, 1), stream)


def wino_gemm_cute(
    v: torch.Tensor, u_nk: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``out[l] = v[l] @ u_nk[l].T``: v [L, M, K], u_nk [L, N, K], out [L, M, N] (bf16)."""
    nl, m, k = v.shape
    n = u_nk.shape[1]
    assert m % 128 == 0 and n % 128 == 0 and k % 64 == 0, (m, n, k)
    if out is None:
        out = torch.empty(nl, m, n, dtype=v.dtype, device=v.device)
    fn = _compiled((nl, m, n, k))
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
    m_pad = m_valid if args.no_pad else (m_valid + 255) // 256 * 256
    v_exact = torch.ops.trtllm.vae_wino_spatial_in(frames, components, 0, th)  # [64, m_valid, cin]
    v = torch.zeros(components * 16, m_pad, cin, dtype=v_exact.dtype, device="cuda")
    v[:, :m_valid] = v_exact
    u_nk = winograd_spatial_weights(weight).transpose(1, 2).contiguous()  # [64, cout, cin]
    out = torch.empty(components * pairs, h, w, cout, dtype=torch.bfloat16, device="cuda")
    run = wino_spatial_conv_persistent if args.persistent else wino_spatial_conv_fused
    if args.pass2:
        t_part = torch.empty(
            pass2_workspace_numel(components, m_pad, cout), dtype=v.dtype, device="cuda"
        )
        u_pass = winograd_pass_filters(u_nk)

        def run(v, u_nk, out, components, pairs, h, w):
            return wino_spatial_conv_pass2(v, u_pass, out, components, pairs, h, w, t_part)

    run(v, u_nk, out, components, pairs, h, w)
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
    if args.pass2:
        run(v, u_nk, out, components, pairs, h, w)
        torch.cuda.synchronize()
        rel2 = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"second call rel-L2 (counters self-reset): {rel2:.3e}")
    if args.diag:
        for pr in range(2):
            for pc in range(2):
                o, r = out[:, pr::2, pc::2].float(), ref[:, pr::2, pc::2]
                print(f"  parity ({pr},{pc}) rel-L2 {((o - r).norm() / r.norm()).item():.3e}")
        for nb in range(min(4, cout // 64)):
            o, r = out[..., nb * 64 : (nb + 1) * 64].float(), ref[..., nb * 64 : (nb + 1) * 64]
            print(
                f"  channels {nb * 64}..{(nb + 1) * 64 - 1} rel-L2 {((o - r).norm() / r.norm()).item():.3e}"
            )
        # Identity filters: the GEMMs must return V itself, so the kernel output is the plain
        # inverse transform of V. Any channel/row permutation of A shows up as a permutation here.
        u_eye = (
            torch.eye(cout, cin, dtype=torch.bfloat16, device="cuda")
            .expand(components * 16, -1, -1)
            .contiguous()
        )
        ref_eye = torch.empty_like(out)
        torch.ops.trtllm.vae_wino_spatial_out(v_exact, ref_eye, components, 0, th)
        run(v, u_eye, out, components, pairs, h, w)
        torch.cuda.synchronize()
        o2, r2 = out.float().reshape(-1, cout), ref_eye.float().reshape(-1, cout)
        print(f"  identity-filter rel-L2 {((o2 - r2).norm() / r2.norm()).item():.3e}")
        on = o2 / (o2.norm(dim=0, keepdim=True) + 1e-6)
        rn = r2 / (r2.norm(dim=0, keepdim=True) + 1e-6)
        corr = on.t() @ rn  # [cout_out, cout_ref]
        best = corr.argmax(dim=1)
        print(f"  channel match: out ch -> ref ch (first 40): {best[:40].tolist()}")
        print(
            f"  best corr (first 8): {[round(x, 3) for x in corr.max(dim=1).values[:8].tolist()]}"
        )
        on_r = o2 / (o2.norm(dim=1, keepdim=True) + 1e-6)
        rn_r = r2 / (r2.norm(dim=1, keepdim=True) + 1e-6)
        rows = min(512, o2.shape[0])
        corr_r = on_r[:rows] @ rn_r[:rows].t()
        print(f"  row match (first 24 of pixels): {corr_r.argmax(dim=1)[:24].tolist()}")
        print(
            f"  per-pixel-row cosine (first 8): {[round(x, 3) for x in (on_r * rn_r).sum(dim=1)[:8].tolist()]}"
        )
        for pr in range(2):
            for pc in range(2):
                o, r = out[:, pr::2, pc::2].float(), ref_eye[:, pr::2, pc::2].float()
                print(
                    f"  identity parity ({pr},{pc}) rel-L2 {((o - r).norm() / r.norm()).item():.3e}"
                )
        return
    if args.probe:
        for _ in range(3):
            run(v, u_nk, out, components, pairs, h, w)
        torch.cuda.synchronize()
        return
    conv_flops = 2.0 * components * pairs * h * w * cin * cout * 9
    wino_flops = 2.0 * 64 * m_valid * cin * cout
    t_fused = _time_cupti(
        lambda: run(v, u_nk, out, components, pairs, h, w),
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
    if args.pass2:
        from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (
            cute_spatial_filters,
            wino_spatial_conv_cuda,
            wino_spatial_conv_cute,
        )

        d4 = frames.view(components, pairs, h * w, cin)
        u_portable = winograd_spatial_weights(weight)
        u_cute = cute_spatial_filters(u_portable.transpose(1, 2).contiguous())
        t_port = _time_cupti(
            lambda: wino_spatial_conv_cuda(d4, u_portable, h, w), 3, args.iters, use_graph=True
        )["span"]["median_us"]
        u_f4 = winograd_spatial_weights(weight, 4)
        y_f4 = wino_spatial_conv_cuda(d4, u_f4, h, w).view(components * pairs, h, w, cout)
        rel_f4 = ((y_f4.float() - ref).norm() / ref.norm()).item()
        t_f4 = _time_cupti(
            lambda: wino_spatial_conv_cuda(d4, u_f4, h, w), 3, args.iters, use_graph=True
        )["span"]["median_us"]
        t_cute = _time_cupti(
            lambda: wino_spatial_conv_cute(d4, u_cute, h, w), 3, args.iters, use_graph=True
        )["span"]["median_us"]
        print(
            f"portable path: {t_port / 1000:8.3f} ms  (in-transform + cuBLAS bmm + out-transform)"
        )
        print(f"portable F4  : {t_f4 / 1000:8.3f} ms  (F(4x4,3x3): 36 GEMMs; rel-L2 {rel_f4:.3e})")
        print(
            f"cute path    : {t_cute / 1000:8.3f} ms  (in-transform + two-pass GEMM, combine inside)"
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
    parser.add_argument(
        "--persistent", action="store_true", help="with --fused: the persistent kernel"
    )
    parser.add_argument(
        "--diag", action="store_true", help="with --fused: correctness diagnostics only"
    )
    parser.add_argument(
        "--pass2",
        action="store_true",
        help="with --fused: the 256x256 two-pass kernel + torch combine",
    )
    parser.add_argument(
        "--no-pad", action="store_true", help="with --fused: exact row count, no padding to 256"
    )
    args = parser.parse_args()
    if args.fused:
        return _main_fused(args)
    if not args.probe:
        atexit.register(lambda: __import__("cupti.cupti", fromlist=["cupti"]).finalize())
    nl, m, n, k = (int(x) for x in args.lmnk.split(","))
    gen = torch.Generator(device="cuda").manual_seed(0)
    v = torch.randn(nl, m, k, generator=gen, device="cuda").to(torch.bfloat16)
    u = (0.05 * torch.randn(nl, n, k, generator=gen, device="cuda")).to(torch.bfloat16)
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
    flops = 2.0 * nl * m * n * k
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
