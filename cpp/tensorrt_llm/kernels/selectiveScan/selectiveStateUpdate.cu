#include "conversion.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm::kernels
{

__forceinline__ __device__ float softplus(float x)
{
    return __logf(1.f + __expf(x));
}

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
        constexpr float SOFTPLUS_THRESHOLD = 20.f;
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

template <typename input_t, typename weight_t, int DSTATE, int CHANNELS_PER_BLOCK = 128>
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
        constexpr float SOFTPLUS_THRESHOLD = 20.f;
        dt_value = (dt_value <= SOFTPLUS_THRESHOLD) ? softplus(dt_value) : dt_value;
    }

    __shared__ weight_t sA[CHANNELS_PER_BLOCK][DSTATE];
    __shared__ weight_t sB[DSTATE];
    __shared__ weight_t sC[DSTATE];
    __shared__ weight_t sState[CHANNELS_PER_BLOCK][DSTATE];

    // Copy A
    for (int channel = 0; channel < CHANNELS_PER_BLOCK; channel++)
        for (int i = threadIdx.x; i < DSTATE; i += blockDim.x)
        {
            sA[channel][i] = A[head * dim * DSTATE + (blockIdx.x * CHANNELS_PER_BLOCK + channel) * DSTATE + i];
        }

    for (int i = threadIdx.x; i < DSTATE; i += blockDim.x)
    {
        sB[i] = B[batch * ngroups * DSTATE + group * DSTATE + i];
    }
    for (int i = threadIdx.x; i < DSTATE; i += blockDim.x)
    {
        sC[i] = C[batch * ngroups * DSTATE + group * DSTATE + i];
    }
    for (int channel = 0; channel < CHANNELS_PER_BLOCK; channel++)
        for (int i = threadIdx.x; i < DSTATE; i += blockDim.x)
        {
            sState[channel][i] = state[head * dim * DSTATE + (blockIdx.x * CHANNELS_PER_BLOCK + channel) * DSTATE + i];
        }

    __syncthreads();

    float out_value = D ? toFloat(D[head * dim + idx_dim]) * x_value : 0.f;
    for (int i = 0; i < DSTATE; i++)
    {
        // load from smem + convert to float
        auto const A_value = toFloat(sA[threadIdx.x][i]);
        auto const B_value = toFloat(sB[i]);
        auto const C_value = toFloat(sC[i]);
        auto const state_value = toFloat(sState[threadIdx.x][i]);
        // compute
        auto const dA = __expf(A_value * dt_value);
        auto const dB = B_value * dt_value;
        auto const new_state = state_value * dA + dB * x_value;
        // store
        convertAndStore(&state[head * dim * DSTATE + idx_dim * DSTATE + i], new_state);
        out_value += new_state * C_value;
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

template <typename input_t, typename weight_t>
void invokeSelectiveStateUpdate(SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType)
{
    if (kernelType == SelectiveStateUpdateKernelType::naive) {
        constexpr int channels_per_block = 128;
        int const blocks_per_dim = (params.dim + channels_per_block - 1) / channels_per_block;
        dim3 block(channels_per_block, 1);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);

        TLLM_CHECK(params.dstate % 16 == 0);

        constexpr int DSTATE = 128;
        TLLM_CHECK(params.dstate == DSTATE);
        selective_state_update_kernel<input_t, weight_t, DSTATE, channels_per_block><<<grid, block, 0, stream>>>(params);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::optimized) {
        constexpr int channels_per_block = 128;
        int const blocks_per_dim = (params.dim + channels_per_block - 1) / channels_per_block;
        dim3 block(channels_per_block, 1);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);

        TLLM_CHECK(params.dstate % 16 == 0);
        constexpr int DSTATE = 128;
        TLLM_CHECK(params.dstate == DSTATE);
        selective_state_update_kernel_opt<input_t, weight_t, DSTATE, channels_per_block><<<grid, block, 0, stream>>>(params);
    }
    else {
        TLLM_CHECK(false);
    }
}

template void invokeSelectiveStateUpdate<half, half>(SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType);
template void invokeSelectiveStateUpdate<float, float>(SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType);

} // end namespace tensorrt_llm::kernels
