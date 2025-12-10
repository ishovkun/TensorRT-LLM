#include <iostream>
#include <cstdint>
#include <cstdint>
#include <cmath>
#include "tmaDescriptor.cuh"
#include "conversion.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda_runtime_api.h>

namespace tensorrt_llm::kernels
{

__forceinline__ __device__ float softplus(float x)
{
    return __logf(1.f + __expf(x));
}

// Pade approximation or minimax polynomial for exp
__device__ __forceinline__ float fast_exp_poly(float x)
{
    // For x in [-10, 0] range (typical for SSM)
    // Pade [3/3] approximation
    float x2 = x * x;
    float num = 1.0f + x * 0.5f + x2 * 0.08333333f;
    float den = 1.0f - x * 0.5f + x2 * 0.08333333f;
    float result = __fdividef(num, den);
    return result; // or adjust for better accuracy
}

__device__ __forceinline__ float fast_exp_pade44(float x)
{
    // Pade [4/4] approximation - extremely high accuracy
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;

    float num = 1.0f + 0.5f * x + 0.083333333333f * x2 + 0.0083333333333f * x3 + 0.00059523809524f * x4;
    float den = 1.0f - 0.5f * x + 0.083333333333f * x2 - 0.0083333333333f * x3 + 0.00059523809524f * x4;

    return __fdividef(num, den);
    // Coefficients: 1, 1/2, 1/12, 1/120, 1/1680
}

__device__ __forceinline__ float fast_exp(float x)
{
    constexpr float log2_E = 1.4426950408889634f;
    float y;
    asm("ex2.approx.f32 %0,%1;" : "=f"(y) : "f"(x * log2_E));
    return y;
}

__device__ __forceinline__ float thresholded_softplus(float dt_value)
{
    constexpr float threshold = 20.f;
    return (dt_value <= threshold) ? softplus(dt_value) : dt_value;
}

template <typename input_t, typename weight_t, int DSTATE, int CHANNELS_PER_BLOCK = 128>
__global__ void selective_state_update_kernel(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);   // A: (nheads, dim, dstate)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
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
    auto const dt_offset = x_offset;
    auto dt_value = toFloat(dt[dt_offset]);

    if (dt_bias)
    {
        dt_value += toFloat(dt_bias[head * dim + idx_dim]);
    }
    if (dt_softplus)
    {
        dt_value = thresholded_softplus(dt_value);
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
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);   // A: (nheads, dim, dstate)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
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
        dt_value = thresholded_softplus(dt_value);
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
    // constexpr auto numIterationsState = (CHANNELS_PER_BLOCK * DSTATE) / (blockSize * kVecSizeInput);
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

template <typename input_t, typename weight_t, int DSTATE>
__global__ void selective_state_update_kernel_simple(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);   // A: (nheads, dim, dstate)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;
    int const dim = params.dim;

    auto const idx_dim = blockIdx.x;
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

    /*
     *  Idea: each thread just
     * - separate: somehow load 1d arrays: D, C, x
     * - loads an element state[m, d], dA[m, d]
     * - update state[m, d] = state[m, d] * dA[m, d] + dB[M, d] * x[M]
     * - compute (state[m, d] * C[d]) + reduce along d dimension
     */

    // x: (batch, nheads, dim)
    auto const x_offset = (batch * nheads + head) * dim + idx_dim;
    auto x_value = toFloat(x[x_offset]);

    // z: (batch, nheads, dim)
    auto z_value = z ? toFloat(z[batch * nheads * dim + head * dim + idx_dim]) : 0.f;

    // dt: (batch, nheads, dim)
    auto const dt_offset = x_offset;
    auto dt_value = toFloat(dt[dt_offset]);
    if (dt_bias)
    {
        dt_value += toFloat(dt_bias[head * dim + idx_dim]);
    }
    if (dt_softplus)
    {
        dt_value = thresholded_softplus(dt_value);
    }

    constexpr auto warpSize = 32;
    auto lane = threadIdx.x % warpSize;
    float out_value = D ? toFloat(D[head * dim + idx_dim]) * x_value : 0.f;
    out_value *= int(threadIdx.x == 0);

    using load_t = float4;
    static_assert(sizeof(weight_t) == sizeof(input_t));
    static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);
    for (int i = threadIdx.x * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
    {
        load_t rA = *reinterpret_cast<load_t const*>(&A[head * dim * DSTATE + idx_dim * DSTATE + i]);
        load_t rState = *reinterpret_cast<load_t*>(&state[head * dim * DSTATE + idx_dim * DSTATE + i]);
        load_t rB = *reinterpret_cast<load_t const*>(&B[batch * ngroups * DSTATE + group * DSTATE + i]);
        load_t rC = *reinterpret_cast<load_t const*>(&C[batch * ngroups * DSTATE + group * DSTATE + i]);

        auto const* A_vals = reinterpret_cast<weight_t const*>(&rA);
        auto* state_vals = reinterpret_cast<input_t*>(&rState);
        auto const* B_vals = reinterpret_cast<weight_t const*>(&rB);
        auto const* C_vals = reinterpret_cast<weight_t const*>(&rC);

        for (int ii = 0; ii < vectorizedLoadSize; ii++)
        {
            auto A_value = toFloat(A_vals[ii]);
            auto state_value = toFloat(state_vals[ii]);
            auto B_value = toFloat(B_vals[ii]);
            auto C_value = toFloat(C_vals[ii]);

            auto const dA = __expf(A_value * dt_value);
            auto const dB = B_value * dt_value;
            auto const new_state = state_value * dA + dB * x_value;

            out_value += new_state * C_value;
        }
        *reinterpret_cast<load_t*>(&state[head * dim * DSTATE + idx_dim * DSTATE + i]) = rState;
    }

    // warpReduce the out_value
    for (int s = warpSize / 2; s > 0; s /= 2)
    {
        auto tmp = __shfl_down_sync(UINT32_MAX, out_value, s);
        if (s >= lane)
        {
            out_value += tmp;
        }
    }
    // now lane 0 holds the accumulated out_value
    if (threadIdx.x == 0)
    {
        if (z)
        {
            float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
            float silu_z = z_value * sig_z;
            out_value *= silu_z;
        }

        convertAndStore(&output[x_offset], out_value);
    }
}

template <typename input_t, typename weight_t, int DSTATE>
__global__ void selective_state_update_kernel_simple2(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);   // A: (nheads, dim, dstate)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;
    int const dim = params.dim;

    constexpr auto warpSize = 32;
    auto const dim_offset = blockIdx.x * warpSize;
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

    /*
     *  Idea: each thread just
     * - separate: somehow load 1d arrays: D, C, x
     * - loads an element state[m, d], dA[m, d]
     * - update state[m, d] = state[m, d] * dA[m, d] + dB[M, d] * x[M]
     * - compute (state[m, d] * C[d]) + reduce along d dimension
     */

    __shared__ input_t sx[warpSize];
    __shared__ float sdt[warpSize];
    __shared__ weight_t sz[warpSize];
    __shared__ input_t sD[warpSize];

    if (dim_offset + threadIdx.x < dim)
    {
        // x: (batch, nheads, dim)
        auto const x_offset = (batch * nheads + head) * dim + (dim_offset + threadIdx.x);
        sx[threadIdx.x] = x[x_offset];

        // dt: (batch, nheads, dim)
        auto const dt_offset = x_offset;
        auto dt_value = toFloat(dt[dt_offset]);
        if (dt_bias)
            dt_value += toFloat(dt_bias[head * dim + (dim_offset + threadIdx.x)]);
        if (dt_softplus)
            dt_value = thresholded_softplus(dt_value);
        sdt[threadIdx.x] = dt_value;

        sz[threadIdx.x] = z ? z[batch * nheads * dim + head * dim + (dim_offset + threadIdx.x)] : (weight_t) 0.f;
        sD[threadIdx.x] = D ? D[head * dim + (dim_offset + threadIdx.x)] : (input_t) 0.f;
    }

    auto lane = threadIdx.x % warpSize;

    using load_t = float2;
    // using load_t = float4;
    static_assert(sizeof(weight_t) == sizeof(input_t));
    static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);

    for (auto channel = 0; channel < warpSize; channel++)
    {

        float dt_value = sdt[channel];
        float x_value = toFloat(sx[channel]);
        float out_value = toFloat(sD[channel]) * x_value * int(threadIdx.x == 0); // first lane has the value

        for (int i = threadIdx.x * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
        {
            load_t rA = *reinterpret_cast<load_t const*>(&A[head * dim * DSTATE + (dim_offset + channel) * DSTATE + i]);
            load_t rState = *reinterpret_cast<load_t*>(&state[head * dim * DSTATE + (dim_offset + channel) * DSTATE + i]);
            load_t rB = *reinterpret_cast<load_t const*>(&B[batch * ngroups * DSTATE + group * DSTATE + i]);
            load_t rC = *reinterpret_cast<load_t const*>(&C[batch * ngroups * DSTATE + group * DSTATE + i]);

            auto const* A_vals = reinterpret_cast<weight_t const*>(&rA);
            auto* state_vals = reinterpret_cast<input_t*>(&rState);
            auto const* B_vals = reinterpret_cast<weight_t const*>(&rB);
            auto const* C_vals = reinterpret_cast<weight_t const*>(&rC);

            for (int ii = 0; ii < vectorizedLoadSize; ii++)
            {

                auto A_value = toFloat(A_vals[ii]);
                auto state_value = toFloat(state_vals[ii]);
                auto B_value = toFloat(B_vals[ii]);
                auto C_value = toFloat(C_vals[ii]);

                // auto const dA = __expf(A_value * dt_value);
                auto const dA = fast_exp(A_value * dt_value);
                // auto const dA = fast_exp_poly(A_value * dt_value);
                // auto const dA = fast_exp_pade44(A_value * dt_value);
                auto const dB = B_value * dt_value;
                auto const new_state = state_value * dA + dB * x_value;
                convertAndStore(&state_vals[ii], new_state);

                out_value += new_state * C_value;
            }
            *reinterpret_cast<load_t*>(&state[head * dim * DSTATE + (dim_offset + channel) * DSTATE + i]) = rState;
        }

        // warpReduce the out_value
        for (int s = warpSize / 2; s > 0; s /= 2)
        {
            auto tmp = __shfl_down_sync(UINT32_MAX, out_value, s);
            // out_value += tmp * int(s >= lane);
            if (s >= lane)
                out_value += tmp;
        }
        if (threadIdx.x == 0)
        {
            sdt[channel] = out_value;
        }
    }

    if (dim_offset + threadIdx.x < dim)
    {
        auto out_value = sdt[threadIdx.x];
        if (z)
        {
            float z_value = toFloat(sz[threadIdx.x]);
            float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
            float silu_z = z_value * sig_z;
            out_value *= silu_z;
        }
        auto const x_offset = (batch * nheads + head) * dim + (dim_offset + threadIdx.x);
        convertAndStore(&output[x_offset], out_value);
    }
}

template <typename input_t, typename weight_t, int DSTATE, int numWarps>
__global__ void selective_state_update_kernel_simple3(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);   // A: (nheads, dim, dstate)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;
    int const dim = params.dim;

    constexpr auto warpSize = 32;
    auto const dim_offset = blockIdx.x * warpSize * numWarps;
    auto const batch = blockIdx.y;
    auto const head = blockIdx.z;
    auto const group = head / (nheads / ngroups);
    auto lane = threadIdx.x % warpSize;
    auto warp = threadIdx.y;

    auto const batch_idx = (state_batch_indices) ? state_batch_indices[batch] : batch;
    state += batch_idx * nheads * dim * DSTATE;

    /*
     *  Idea: each thread just
     * - separate: somehow load 1d arrays: D, C, x
     * - loads an element state[m, d], dA[m, d]
     * - update state[m, d] = state[m, d] * dA[m, d] + dB[M, d] * x[M]
     * - compute (state[m, d] * C[d]) + reduce along d dimension
     */

    __shared__ input_t sx[numWarps*warpSize];
    __shared__ float sdt[numWarps*warpSize];
    __shared__ weight_t sz[numWarps*warpSize];
    __shared__ input_t sD[numWarps*warpSize];

    if (dim_offset + warp*warpSize + threadIdx.x < dim)
    {
        auto _d = warp*warpSize + threadIdx.x;
        auto d = dim_offset + _d;
        // x: (batch, nheads, dim)
        auto const x_offset = (batch * nheads + head) * dim + d;
        sx[_d] = x[x_offset];

        // dt: (batch, nheads, dim)
        auto const dt_offset = x_offset;
        auto dt_value = toFloat(dt[dt_offset]);
        if (dt_bias)
            dt_value += toFloat(dt_bias[head * dim + d]);
        if (dt_softplus)
            dt_value = thresholded_softplus(dt_value);
        sdt[_d] = dt_value;

        sz[_d] = z ? z[batch * nheads * dim + head * dim + d] : (weight_t) 0.f;
        sD[_d] = D ? D[head * dim + d] : (input_t) 0.f;
    }

    using load_t = float2;
    // using load_t = float4;
    static_assert(sizeof(weight_t) == sizeof(input_t));
    static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);

    for (auto _d = warp*warpSize; _d < (warp+1)*warpSize; _d++)
    {
        auto d = dim_offset + _d;
        float dt_value = sdt[_d];
        float x_value = toFloat(sx[_d]);
        float out_value = toFloat(sD[_d]) * x_value * int(lane == 0); // first lane has the value

        for (int i = threadIdx.x * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
        {
            load_t rA = *reinterpret_cast<load_t const*>(&A[head * dim * DSTATE + d * DSTATE + i]);
            load_t rState = *reinterpret_cast<load_t*>(&state[head * dim * DSTATE + d * DSTATE + i]);
            load_t rB = *reinterpret_cast<load_t const*>(&B[batch * ngroups * DSTATE + group * DSTATE + i]);
            load_t rC = *reinterpret_cast<load_t const*>(&C[batch * ngroups * DSTATE + group * DSTATE + i]);

            auto const* A_vals = reinterpret_cast<weight_t const*>(&rA);
            auto* state_vals = reinterpret_cast<input_t*>(&rState);
            auto const* B_vals = reinterpret_cast<weight_t const*>(&rB);
            auto const* C_vals = reinterpret_cast<weight_t const*>(&rC);

            for (int ii = 0; ii < vectorizedLoadSize; ii++)
            {
                auto A_value = toFloat(A_vals[ii]);
                auto state_value = toFloat(state_vals[ii]);
                auto B_value = toFloat(B_vals[ii]);
                auto C_value = toFloat(C_vals[ii]);

                // auto const dA = __expf(A_value * dt_value);
                auto const dA = fast_exp(A_value * dt_value);
                // auto const dA = fast_exp_poly(A_value * dt_value);
                // auto const dA = fast_exp_pade44(A_value * dt_value);
                auto const dB = B_value * dt_value;
                auto const new_state = state_value * dA + dB * x_value;
                convertAndStore(&state_vals[ii], new_state);

                // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && threadIdx.x == 0 && warp == 0 && d == 0 && i == 0 && ii == 0) {
                //     printf("d = %d i = %d old_state = %f new_state = %f \n", d, i, state_value, new_state);
                // }

                out_value += new_state * C_value;
            }
            *reinterpret_cast<load_t*>(&state[head * dim * DSTATE + d * DSTATE + i]) = rState;
        }

        // warpReduce the out_value
        for (int s = warpSize / 2; s > 0; s /= 2)
        {
            auto tmp = __shfl_down_sync(UINT32_MAX, out_value, s);
            // out_value += tmp * int(s >= lane);
            if (s >= lane)
                out_value += tmp;
        }
        if (lane == 0)
        {
            sdt[_d] = out_value;
        }
    }

    auto _d = threadIdx.y * warpSize + threadIdx.x;
    auto d = dim_offset + _d;
    if (d < dim)
    {
        auto out_value = sdt[_d];
        if (z)
        {
            float z_value = toFloat(sz[_d]);
            float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
            float silu_z = z_value * sig_z;
            out_value *= silu_z;
        }
        auto const x_offset = (batch * nheads + head) * dim + d;
        convertAndStore(&output[x_offset], out_value);
    }
}


template <typename input_t, typename weight_t, int rowsPerStage, int DSTATE, uint8_t numStages>
struct SharedStorage {
  alignas(128) weight_t A[numStages][rowsPerStage * DSTATE];
  alignas(128) input_t state[numStages][rowsPerStage * DSTATE];
  input_t x[64];
  float dt[64];
  weight_t z[64];
  input_t D[64];
  weight_t B[DSTATE];
  weight_t C[DSTATE];

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  barrier_t bar_empty[numStages];
  barrier_t bar_full[numStages];
  barrier_t bar_consumers;
};

template <typename input_t, typename weight_t, int DSTATE, int consumerWarps, int rowsPerStage, int numStages = 1>
__global__ void selective_state_update_kernel_producer_consumer(SelectiveStateUpdateParams params,
    __grid_constant__ CUtensorMap const tensorA,
    __grid_constant__ CUtensorMap const tensorState)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    // auto const* __restrict__ A = reinterpret_cast<weight_t const*>(params.A);   // A: (nheads, dim, dstate)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);
    bool const dt_softplus = params.dt_softplus;

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;
    int const dim = params.dim;

    constexpr auto warpSize = 32;
    constexpr auto numWarps = 1 + consumerWarps;

    auto const batch = blockIdx.x;
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto lane = threadIdx.x % warpSize;
    auto warp = threadIdx.y;

    auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;

    if (state_batch == params.pad_slot_id)
    {
        return;
    }

    using sram_t = SharedStorage<input_t, weight_t, rowsPerStage, DSTATE, numStages>;
    __shared__ sram_t sram;

    namespace cde = cuda::device::experimental;
    namespace cg = cooperative_groups;

    for (int stage = warp; stage < numStages; stage += numWarps) {
        if (lane > 0) continue;
        constexpr auto num_arrivals = 1 + consumerWarps * warpSize;
        init(&sram.bar_empty[stage], num_arrivals);
        init(&sram.bar_full[stage], num_arrivals);
        // signal to async proxy that barriers are initilized
        cde::fence_proxy_async_shared_cta();
    }
    if (lane == 0 && warp == 0) {
        init(&sram.bar_consumers, warpSize * consumerWarps);
    }
    __syncthreads();

    if (warp == consumerWarps) { // producer
        for (int d = 0, stage = 0; d < dim; d += rowsPerStage, stage = (stage + 1) % numStages) {
            if (lane == 0) {
                namespace cg = cooperative_groups;
                cg::invoke_one(cg::coalesced_threads(), [&]() {
                    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

                    auto tma_copy = cde::cp_async_bulk_tensor_2d_global_to_shared;
                    tma_copy(&sram.A[stage][0], &tensorA, /*x*/ 0, /*y*/ head*dim + d, sram.bar_full[stage]);
                    uint64_t const state_offset = (state_batch * nheads + head)*dim;
                    tma_copy(&sram.state[stage][0], &tensorState, /*x*/ 0, /*y*/ state_offset + d, sram.bar_full[stage]);

                    // Unblock the consumers
                    auto constexpr bytesState = rowsPerStage * DSTATE * sizeof(input_t);
                    auto constexpr bytesA = rowsPerStage * DSTATE * sizeof(weight_t);
                    auto constexpr bytesToArrive = bytesState + bytesA;
                    auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
                });
            }
        }
    }
    else { // consumers

        // Unblock the producer
        #pragma unroll
        for (uint8_t stage = 0; stage < numStages; ++stage) {
           auto const _ = sram.bar_empty[stage].arrive();
        }

        // Load x, dt, z, D
        {
            auto d = warp*warpSize + lane;
            if (d < dim)
            {
                sram.x[d] = x[(batch*nheads + head) * dim + d];
                auto dt_value = toFloat(dt[(batch*nheads + head) * dim + d]);
                if (dt_bias)
                    dt_value += toFloat(dt_bias[head * dim + d]);
                if (dt_softplus)
                    dt_value = thresholded_softplus(dt_value);
                sram.dt[d] = dt_value;
                sram.z[d] = z ? z[batch * nheads * dim + head * dim + d] : (weight_t) 0.f;
                sram.D[d] = D ? D[head * dim + d] : (input_t) 0.f;
            }
        }
        // Load B, C
        {
            for (auto i = warp * warpSize + lane; i < DSTATE; i += warpSize*consumerWarps) {
                sram.B[i] = B[batch * ngroups * DSTATE + group * DSTATE + i];
                sram.C[i] = C[batch * ngroups * DSTATE + group * DSTATE + i];
            }
        }

        sram.bar_consumers.wait(sram.bar_consumers.arrive());

        for (auto dBegin = 0, stage = 0; dBegin < dim; dBegin += rowsPerStage, stage = (stage+1) % numStages) {

            // wait for the producer
            sram.bar_full[stage].wait(sram.bar_full[stage].arrive());

            #pragma unroll
            for (auto ddBegin = 0; ddBegin < rowsPerStage; ddBegin += consumerWarps) {
                auto dd = ddBegin + warp;
                auto d = dBegin + dd;
                float dt_value = sram.dt[d];
                float x_value = toFloat(sram.x[d]);
                float out_value = toFloat(sram.D[d]) * x_value * int(lane == 0); // first lane has the value

                for (int i = lane; i < DSTATE; i += warpSize) {
                    auto A_value = toFloat(sram.A[stage][dd * DSTATE + i]);
                    auto state_value = toFloat(sram.state[stage][dd * DSTATE + i]);
                    auto B_value = toFloat(sram.B[i]);
                    auto C_value = toFloat(sram.C[i]);

                    auto const dA = fast_exp(A_value * dt_value);
                    auto const dB = B_value * dt_value;
                    auto const new_state = state_value * dA + dB * x_value;

                    convertAndStore(&sram.state[stage][dd*DSTATE + i], new_state);
                    out_value += new_state * C_value;
                }

                using load_t = float2;
                static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);
                for (int i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
                {
                    auto const * src = reinterpret_cast<load_t*>(&sram.state[stage][dd * DSTATE + i]);
                    auto * dst = reinterpret_cast<load_t*>(&state[state_batch * nheads * dim * DSTATE + head * dim * DSTATE + d * DSTATE + i]);
                    *dst = *src;
                }

                // warpReduce the out_value
                for (int s = warpSize / 2; s > 0; s /= 2)
                {
                    auto tmp = __shfl_down_sync(UINT32_MAX, out_value, s);
                    out_value += tmp * int(s >= lane);
                    // if (s >= lane)
                    //     out_value += tmp;
                }

                if (lane == 0)
                {
                    sram.dt[d] = out_value;
                }
            }

            // Unblock producer
            auto _ = sram.bar_empty[stage].arrive();
        }

        // Write output
        sram.bar_consumers.wait(sram.bar_consumers.arrive());
        auto d = warp * warpSize + lane;
        if (d < dim)
        {
            auto out_value = sram.dt[d];
            if (z)
            {
                float z_value = toFloat(sram.z[d]);
                float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
                float silu_z = z_value * sig_z;
                out_value *= silu_z;
            }
            auto const x_offset = (batch * nheads + head) * dim + d;
            convertAndStore(&output[x_offset], out_value);
        }

    }
}

template <typename input_t, typename weight_t>
void invokeSelectiveStateUpdate(
    SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType)
{
    TLLM_CHECK(params.dstate % 16 == 0);
    constexpr int DSTATE = 128;
    TLLM_CHECK(params.dstate == DSTATE);
    if (kernelType == SelectiveStateUpdateKernelType::naive)
    {
        constexpr int channels_per_block = 128;
        int const blocks_per_dim = (params.dim + channels_per_block - 1) / channels_per_block;
        dim3 block(channels_per_block, 1);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);

        TLLM_CHECK(params.dstate % 16 == 0);

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

        selective_state_update_kernel_opt<input_t, weight_t, DSTATE, block_size, channels_per_block>
            <<<grid, block, 0, stream>>>(params);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::simple)
    {
        int const blocks_per_dim = params.dim / 32;
        dim3 block(32, 1);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);
        selective_state_update_kernel_simple2<input_t, weight_t, DSTATE><<<grid, block, 0, stream>>>(params);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::simple3)
    {
        constexpr int numWarps = 2;
        int const blocks_per_dim = params.dim / (32 * numWarps);
        dim3 block(32, numWarps);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);
        selective_state_update_kernel_simple3<input_t, weight_t, DSTATE, numWarps><<<grid, block, 0, stream>>>(params);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::producer_consumer)
    {
        constexpr auto numConsumers = 2;
        constexpr auto numWarps = 1 + numConsumers;
        constexpr auto numStages = 2;
        constexpr auto rowsPerStage = 4 * numConsumers;
        auto scan_func = selective_state_update_kernel_producer_consumer<input_t, weight_t, DSTATE,
        numConsumers, rowsPerStage, numStages>;

        dim3 block(32, numWarps);
        dim3 grid(params.batch, params.nheads);

        // batchedGemm::trtllm::gen::Dtype A_dtype;
        // if constexpr (weight_t == half) {
        //     A_dtype = batchedGemm::trtllm::gen::Dtype::FP16;
        // } else if constexpr (weight_t == float) {
        //     A_dtype = batchedGemm::trtllm::gen::Dtype::FP32;
        // } else {
        //     TLLM_CHECK(false);
        // }

        auto nh = params.nheads;
        auto dim = params.dim;
        // auto b = params.batch;
        auto B = params.state_cache_size;
        TLLM_CHECK(reinterpret_cast<uintptr_t>(params.A) % 128 == 0);
        auto tensorA = tma::createTensorMap<weight_t>(params.A, nh*dim, DSTATE, rowsPerStage, DSTATE);
        auto tensorState = tma::createTensorMap<input_t>(params.state, B*nh*dim, DSTATE, rowsPerStage, DSTATE);
        TLLM_CHECK(params.dim % rowsPerStage == 0);

        // using namespace batchGemm::gemm;
        // namespace tg = trtllm::gen;
        // std::vector<uint64_t> A_shapes = {dstate, dim, nheads};
        // std::vector<uint64_t> A_strides = {1, dstate, dstate*dim};
        // std::vector<int32_t> A_tile_shapes = {dstate, rowsPerStage};
        // auto tensorA = batchedGemm::gemm::buildNdTmaDescriptor(A_dtype, MmaKind::Auto, A_shapes, A_strides, A_tile_shapes,
        //    params.A, /*doSwizzle */ false);

        scan_func<<<grid, block, 0, stream>>>(params, tensorA, tensorState);
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
