#include "conversion.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
// #include <cuda/barrier>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm::kernels
{

__forceinline__ __device__ float softplus(float x)
{
    return __logf(1.f + __expf(x));
}

constexpr float SOFTPLUS_THRESHOLD = 20.f;

template <typename input_t, typename weight_t, int DSTATE, int CHANNELS_PER_BLOCK = 128>
__global__ void selective_state_update_kernel(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
    auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;
    int const dim = params.dim;

    auto const idx_dim = blockIdx.x * CHANNELS_PER_BLOCK + threadIdx.x;
    auto const batch = blockIdx.y;
    auto const head = blockIdx.z;
    auto const group = head / (nheads / ngroups);

    if (idx_dim >= dim)
        return;

    // adjust state pointer within the cache
    if (state_batch_indices)
    {
        // state_batch_indices: (batch,)
        auto batch_idx = state_batch_indices[batch];
        if (batch_idx == params.pad_slot_id)
        {
            return;
        }
        state += batch_idx * nheads * dim * DSTATE;
    }
    else
    {
        state += batch * nheads * dim * DSTATE;
    }

    // int const state_loops = (params.dstate + STATE_UNROLL - 1) / STATE_UNROLL;

    // x: (batch, nheads, dim)
    auto const x_offset = (batch * nheads + head) * dim + idx_dim;
    auto x_value = toFloat(x[x_offset]);

    // z: (batch, nheads, dim)
    auto z_value = z ? toFloat(z[batch * nheads * dim + head * dim + idx_dim]) : 0.f;

    // dt: (batch, nheads, dim)
    // auto const dt_offset = (batch * nheads + head) * dim + idx_dim;
    auto const dt_offset = x_offset;
    auto dt_value = toFloat(dt[dt_offset]);

    if (dt_bias)
    {
        dt_value += toFloat(dt_bias[head * dim + idx_dim]);
    }
    if (dt_softplus)
    {
        dt_value = (dt_value <= SOFTPLUS_THRESHOLD) ? softplus(dt_value) : dt_value;
    }

    // Load matrices
    float rA[DSTATE];
    float rB[DSTATE];
    float rC[DSTATE];
    float rState[DSTATE];
    for (int i = 0; i < DSTATE; i++)
    {
        // A: (nheads, dim, dstate)
        rA[i] = toFloat(A[head * dim * DSTATE + idx_dim * DSTATE + i]);
        // B: (batch, ngroups, dstate)
        rB[i] = toFloat(B[batch * ngroups * DSTATE + group * DSTATE + i]);
        // C: (batch, ngroups, dstate)
        rC[i] = toFloat(C[batch * ngroups * DSTATE + group * DSTATE + i]);
        // state: (batch, nheads, dim, dstate)
        rState[i] = toFloat(state[head * dim * DSTATE + idx_dim * DSTATE + i]);
    }

    // Update sate and compute output
    // D: (nheads, dim)
    float out_value = D ? toFloat(D[head * dim + idx_dim]) * x_value : 0.f;
    for (int i = 0; i < DSTATE; i++)
    {
        auto const dA = __expf(rA[i] * dt_value);
        auto const dB = rB[i] * dt_value;
        auto const new_state = rState[i] * dA + dB * x_value;
        convertAndStore(&state[head * dim * DSTATE + idx_dim * DSTATE + i], new_state);
        out_value += new_state * rC[i];
    }

    if (z)
    {
        float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
    }

    // output: (batch, nheads, dim)
    convertAndStore(&output[x_offset], out_value);
}

template <int numBytes>
__device__ __forceinline__ void cp_async_ldgsts(void* smem_dst, void const* gmem_src)
{
    static_assert(numBytes == 4 || numBytes == 8 || numBytes == 16, "cp.async only supports 4, 8, or 16 byte copies");

    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    // asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
    //              :: "r"(smem_addr), "l"(gmem_src), "n"(numBytes));
    if constexpr (numBytes == 16)
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
    }
    else
    {
        // For 4 and 8 byte copies, use cp.async.ca
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(smem_addr), "l"(gmem_src), "n"(numBytes));
    }
}

__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ inline auto swizzle_func(int row, int col, int width, int factor) -> int
{
    return (col + factor * row) % width;
}

template <typename input_t, typename weight_t, int DSTATE, int blockSize, int CHANNELS_PER_BLOCK>
__global__ void selective_state_update_kernel_opt(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
    auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;
    int const dim = params.dim;

    auto const idx_dim = blockIdx.x * CHANNELS_PER_BLOCK + threadIdx.x;
    auto const batch = blockIdx.y;
    auto const head = blockIdx.z;
    auto const group = head / (nheads / ngroups);

    // adjust state pointer within the cache
    if (state_batch_indices)
    {
        // state_batch_indices: (batch,)
        auto batch_idx = state_batch_indices[batch];
        if (batch_idx == params.pad_slot_id)
        {
            return;
        }
        state += batch_idx * nheads * dim * DSTATE;
    }
    else
    {
        state += batch * nheads * dim * DSTATE;
    }

    // x: (batch, nheads, dim)
    auto const x_offset = (batch * nheads + head) * dim + idx_dim;
    auto x_value = (idx_dim < dim) ? toFloat(x[x_offset]) : 0.f;

    // z: (batch, nheads, dim)
    auto z_value = (z && idx_dim < dim) ? toFloat(z[batch * nheads * dim + head * dim + idx_dim]) : 0.f;

    // dt: (batch, nheads, dim)
    auto const dt_offset = x_offset;
    auto dt_value = (idx_dim < dim) ? toFloat(dt[dt_offset]) : 0.f;

    if (dt_bias)
    {
        dt_value += toFloat(dt_bias[head * dim + idx_dim]);
    }
    if (dt_softplus)
    {
        dt_value = (dt_value <= SOFTPLUS_THRESHOLD) ? softplus(dt_value) : dt_value;
    }

    static constexpr auto warpSize = 32;
    // using load_t = float4;
    using load_t = float2;
    static constexpr auto kVecSizeWeight = sizeof(load_t) / sizeof(weight_t);
    static constexpr auto kVecSizeInput = sizeof(load_t) / sizeof(weight_t);
    // static constexpr auto kInputElementsPerBank = sizeof(float) / sizeof(input_t);
    // static constexpr auto kWeightElementsPerBank = sizeof(float) / sizeof(input_t);

    __shared__ weight_t sA[CHANNELS_PER_BLOCK][DSTATE];
    __shared__ weight_t sB[DSTATE];
    __shared__ weight_t sC[DSTATE];
    __shared__ weight_t sState[CHANNELS_PER_BLOCK][DSTATE];

    auto warp = threadIdx.x / warpSize;
    auto lane = threadIdx.x % warpSize;

    if (warp == 0)
    {
        for (int i = lane * kVecSizeInput; i < DSTATE; i += warpSize * kVecSizeInput)
        {
            auto const* src = reinterpret_cast<load_t const*>(&B[batch * ngroups * DSTATE + group * DSTATE + i]);
            auto* dst = reinterpret_cast<load_t*>(&sB[i]);
            *dst = *src;
        }
    }
    else if (warp == 1)
    {
        for (int i = lane * kVecSizeInput; i < DSTATE; i += warpSize * kVecSizeInput)
        {
            auto const* src = reinterpret_cast<load_t const*>(&C[batch * ngroups * DSTATE + group * DSTATE + i]);
            auto* dst = reinterpret_cast<load_t*>(&sC[i]);
            *dst = *src;
        }
    }

    // Copy A
    constexpr auto numIterationsA = (CHANNELS_PER_BLOCK * DSTATE) / (blockSize * kVecSizeInput);
#pragma unroll
    for (int iter = 0; iter < numIterationsA; iter++)
    // for (int offset = 0; offset < CHANNELS_PER_BLOCK * DSTATE; offset += blockSize * kVecSizeWeight)
    {
        auto const pos = (iter * blockSize + threadIdx.x) * kVecSizeWeight;
        // auto const pos = offset + threadIdx.x * kVecSizeWeight;
        auto const state_idx = pos % DSTATE;
        auto const channel = pos / DSTATE;
        auto const inBounds = (blockIdx.x * CHANNELS_PER_BLOCK + channel) < dim;
        if (inBounds)
        {
            auto const* src = reinterpret_cast<load_t const*>(
                &A[head * dim * DSTATE + (blockIdx.x * CHANNELS_PER_BLOCK + channel) * DSTATE + state_idx]);
            auto state_idx_sw = swizzle_func(channel, state_idx, DSTATE, kVecSizeWeight);
            auto* dst = reinterpret_cast<load_t*>(&sA[channel][state_idx_sw]);
            // *dst = *src;
            cp_async_ldgsts<sizeof(load_t)>(dst, src);
        }
    }

    static_assert((CHANNELS_PER_BLOCK * DSTATE) % (blockSize * kVecSizeInput) == 0, "For proper unrolling");
    constexpr auto numIterationsState = (CHANNELS_PER_BLOCK * DSTATE) / (blockSize * kVecSizeInput);
#pragma unroll
    // for (int iter = 0; iter < numIterationsState; iter++)
    for (int offset = 0; offset < CHANNELS_PER_BLOCK * DSTATE; offset += blockSize * kVecSizeInput)
    {
        auto const pos = offset + threadIdx.x * kVecSizeInput;
        // auto const pos = (iter * blockSize + threadIdx.x) * kVecSizeInput;
        auto const state_idx = pos % DSTATE;
        auto const channel = pos / DSTATE;
        auto const inBounds = (blockIdx.x * CHANNELS_PER_BLOCK + channel) < dim;
        if (inBounds)
        {
            auto const* src = reinterpret_cast<load_t const*>(
                &state[head * dim * DSTATE + (blockIdx.x * CHANNELS_PER_BLOCK + channel) * DSTATE + state_idx]);
            auto state_idx_sw = swizzle_func(channel, state_idx, DSTATE, kVecSizeInput);
            auto* dst = reinterpret_cast<load_t*>(&sState[channel][state_idx_sw]);
            // *dst = *src;
            cp_async_ldgsts<sizeof(load_t)>(dst, src);
            // cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(sizeof(load_t)), bar);
            // *dst = __ldg(src);
        }
    }
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();

    float out_value = (D && idx_dim < dim) ? toFloat(D[head * dim + idx_dim]) * x_value : 0.f;
    for (int i = 0; i < DSTATE; i++)
    {
        // load from smem + convert to float
        if (threadIdx.x >= CHANNELS_PER_BLOCK)
            continue;

        auto const i_sw_input = swizzle_func(threadIdx.x, i, DSTATE, kVecSizeInput);
        auto const i_sw_weight = swizzle_func(threadIdx.x, i, DSTATE, kVecSizeWeight);
        auto const A_value = toFloat(sA[threadIdx.x][i_sw_weight]);
        auto const B_value = toFloat(sB[i]);

        auto const C_value = toFloat(sC[i]);
        auto const state_value = toFloat(sState[threadIdx.x][i_sw_input]);
        // compute
        auto const dA = __expf(A_value * dt_value);
        auto const dB = B_value * dt_value;
        auto const new_state = state_value * dA + dB * x_value;

        convertAndStore(&sState[threadIdx.x][i_sw_input], new_state);
        out_value += new_state * C_value;
    }

    if (z)
    {
        float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
    }

    // output: (batch, nheads, dim)
    if (idx_dim < dim)
    {
        convertAndStore(&output[x_offset], out_value);
    }

    // Store state

    // NOTE: I do not use __syncthreads here because I make sure that
    // only warps that processed a particular range of tokens write this to
    // the global output from shmem
    auto firstWarpIdx = threadIdx.x - lane;
    for (int pos = firstWarpIdx * DSTATE + threadIdx.x * kVecSizeWeight; pos < (firstWarpIdx + 1) * DSTATE;
         pos += warpSize * kVecSizeWeight)
    {
        auto channel = pos / DSTATE;
        auto state_idx = pos % DSTATE;
        if (channel < dim)
        {
            auto state_idx_sw = swizzle_func(channel, state_idx, DSTATE, kVecSizeInput);
            auto const* src = reinterpret_cast<load_t const*>(&sState[channel][state_idx_sw]);
            auto* dst = reinterpret_cast<load_t*>(
                &state[head * dim * DSTATE + (blockIdx.x * CHANNELS_PER_BLOCK + channel) * DSTATE + state_idx]);
            *dst = *src;
        }
    }
}

template <typename input_t, typename weight_t>
void invokeSelectiveStateUpdate(
    SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType)
{
    if (kernelType == SelectiveStateUpdateKernelType::naive)
    {
        constexpr int channels_per_block = 128;
        int const blocks_per_dim = (params.dim + channels_per_block - 1) / channels_per_block;
        dim3 block(channels_per_block, 1);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);

        TLLM_CHECK(params.dstate % 16 == 0);

        constexpr int DSTATE = 128;
        TLLM_CHECK(params.dstate == DSTATE);
        selective_state_update_kernel<input_t, weight_t, DSTATE, channels_per_block>
            <<<grid, block, 0, stream>>>(params);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::optimized)
    {
        constexpr int channels_per_block = 64;
        // constexpr int channels_per_block = 128;
        constexpr int block_size = 128;
        int const blocks_per_dim = (params.dim + channels_per_block - 1) / channels_per_block;
        dim3 block(block_size, 1);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);

        TLLM_CHECK(params.dstate % 16 == 0);
        constexpr int DSTATE = 128;
        TLLM_CHECK(params.dstate == DSTATE);
        selective_state_update_kernel_opt<input_t, weight_t, DSTATE, block_size, channels_per_block>
            <<<grid, block, 0, stream>>>(params);
    }
    else
    {
        TLLM_CHECK(false);
    }
}

// we should focus on BF16, FP16 and even FP32 where the Mamba states are involved.
template void invokeSelectiveStateUpdate<half, half>(
    SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType);
template void invokeSelectiveStateUpdate<float, float>(
    SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType);

} // end namespace tensorrt_llm::kernels
