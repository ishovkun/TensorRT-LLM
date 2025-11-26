#include "conversion.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm::kernels
{

__forceinline__ __device__ float softplus(float x) {
    return __logf(1.f + __expf(x));
}

template <typename input_t, typename weight_t, int DSTATE, int CHANNELS_PER_BLOCK = 128, int STATE_UNROLL = 16>
__global__ void selective_state_update_kernel(SelectiveStateUpdateParams params)
{
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    // {
    //     printf("selective_state_update_kernel launched with dim=%d, batch=%d, nheads=%d, dstate=%d\n", params.dim,
    //         params.batch, params.nheads, params.dstate);
    // }
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

    if (idx_dim >= dim)
        return;

    int const group = head / (nheads / ngroups);

    // int const state_loops = (params.dstate + STATE_UNROLL - 1) / STATE_UNROLL;

    // x: (batch, nheads, dim)
    auto const x_offset = (batch * nheads + head) * dim + idx_dim;
    auto x_value = toFloat(x[x_offset]);

    // z: (batch, nheads, dim)
    auto z_value = z ? toFloat(z[batch * nheads * dim + head * dim + idx_dim]) : 0.f;

    // dt: (batch, nheads, dim)
    auto const dt_offset = (batch * nheads + head) * dim + idx_dim;
    auto dt_value = toFloat(dt[dt_offset]);

    if (dt_bias)
    {
        dt_value += toFloat(dt_bias[head * dim + idx_dim]);
    }
    if (dt_softplus)
    {
        dt_value = (dt_value <= 20.f) ? softplus(dt_value) : dt_value;
    }

    // adjust state pointer within the cache
    if (state_batch_indices)
    {
        // state_batch_indices: (batch,)
        state += state_batch_indices[batch] * nheads * dim * DSTATE;
    }
    else {
        state += batch * nheads * dim * DSTATE;
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
        // rState[i] = toFloat(state[batch * nheads * dim * DSTATE + head * dim * DSTATE + idx_dim * DSTATE + i]);
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
        convertAndStore(&state[batch * nheads * dim * DSTATE + head * dim * DSTATE + idx_dim * DSTATE + i], new_state);
        out_value += new_state * rC[i];
    }

    if (z) {
        float sig_z =__fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
    }

    // output: (batch, nheads, dim)
    convertAndStore(&output[x_offset], out_value);
}

template <typename input_t, typename weight_t>
void invokeSelectiveStateUpdate(SelectiveStateUpdateParams& params, cudaStream_t stream)
{
    // using tensorrt_llm::common::ceilDiv;

    // uint32_t block_size_m{0}, num_warps{0};
    // if (params.dstate <= 16) {
    //     block_size_m = 32;
    //     num_warps = 4;
    // }
    // else if (params.dstate <= 32) {
    //     block_size_m = 16;
    //     num_warps = 4;
    // }
    // else if (params.dstate <= 64) {
    //     block_size_m = 8;
    //     num_warps = 4;
    // }
    // else if (params.dstate <= 128) {
    //     block_size_m = 4;
    //     num_warps = 4;
    // }
    // else {
    //     block_size_m = 4;
    //     num_warps = 8;
    // }
    // dim3 block{32u, num_warps};
    // dim3 grid{ceilDiv(params.dim, block_size_m), params.batch, params.nheads};
    // std::cout << "launch grid " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;
    // selective_state_update_kernel<input_t, weight_t><<<grid, block, 0, stream>>>(params);
    // int samples = params.batch;
    // int channels = params.dim;
    // int nheads = params.nheads;
    // int ngroups = params.ngroups;

    constexpr int channels_per_block = 128;
    int const blocks_per_dim = (params.dim + channels_per_block - 1) / channels_per_block;
    dim3 block(channels_per_block, 1);
    dim3 grid(blocks_per_dim, params.batch, params.nheads);

    TLLM_CHECK(params.dstate % 16 == 0);

    constexpr int DSTATE = 128;
    TLLM_CHECK(params.dstate == DSTATE);
    selective_state_update_kernel<input_t, weight_t, DSTATE, channels_per_block><<<grid, block, 0, stream>>>(params);
}

template void invokeSelectiveStateUpdate<half, half>(SelectiveStateUpdateParams& params, cudaStream_t stream);
template void invokeSelectiveStateUpdate<float, float>(SelectiveStateUpdateParams& params, cudaStream_t stream);

} // end namespace tensorrt_llm::kernels
