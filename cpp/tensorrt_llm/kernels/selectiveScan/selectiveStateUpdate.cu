#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/selectiveScan/conversion.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveStateUpdate.h"
// #include "tensorrt_llm/kernels/trtllmGenKernels/fmha/kernelParams.h"
#include "tmaDescriptor.cuh"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cuda/barrier>
#include <cuda_runtime_api.h>
#include <iostream>

// #define IKET_BUILD_ENABLED 0
// #include "iket/iket_device_apis.cuh"
// CREATE_IKET_START_END_RANGE(INITIALIZE);
// CREATE_IKET_START_END_RANGE(TMA_WRITE);
// CREATE_IKET_START_END_RANGE(TMA_READ);
// CREATE_IKET_START_END_RANGE(CONSUMER_LOAD);
// CREATE_IKET_START_END_RANGE(CONSUMER_WAIT);
// CREATE_IKET_START_END_RANGE(CONSUMER_COMPUTE);
// CREATE_IKET_START_END_RANGE(CONSUMER_FINALIZE);
// CREATE_IKET_MARKER(CONSUMER_CYCLE);

namespace tensorrt_llm::kernels
{

__forceinline__ __device__ float softplus(float x)
{
    return __logf(1.f + __expf(x));
    // return log1pf(exp(x));
}

__device__ __forceinline__ float thresholded_softplus(float dt_value)
{
    constexpr float threshold = 20.f;
    return (dt_value <= threshold) ? softplus(dt_value) : dt_value;
}

template <typename T>
__device__ inline auto make_zero() -> T;

template <>
__device__ inline auto make_zero<float2>() -> float2
{
    return make_float2(0.f, 0.f);
}

template <typename compute_t, typename load_t>
__device__ inline auto make_zeros() -> load_t
{
    load_t rValue;
#pragma unroll
    for (int i = 0; i < sizeof(load_t) / sizeof(compute_t); i++)
    {
        auto* dst = reinterpret_cast<compute_t*>(&rValue) + i;
        convertAndStore(dst, 0.f);
    }
    return rValue;
}

__device__ __forceinline__ float warpReduceSum(float val)
{
    constexpr auto warpSize = 32;
    for (int s = warpSize / 2; s > 0; s /= 2)
    {
        val += __shfl_down_sync(UINT32_MAX, val, s);
    }
    return val;
}

// Naive
// template <typename input_t, typename weight_t, int DSTATE, int CHANNELS_PER_BLOCK = 128>
// __global__ void selective_state_update_kernel(SelectiveStateUpdateParams params)
// {
//     auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
//     auto* __restrict__ state = reinterpret_cast<input_t*>(params.state);   // state: (batch, nheads, dim, dstate)

//     auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
//     auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
//     auto const* __restrict__ A = reinterpret_cast<weightA_t const*>(params.A);   // A: (nheads, dim, dstate)
//     auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
//     auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
//     auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
//     auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
//     auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
//     auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);
//     bool const dt_softplus = params.dt_softplus;

//     int const nheads = params.nheads;
//     int const ngroups = params.ngroups;
//     int const dim = params.dim;

//     auto const idx_dim = blockIdx.x * CHANNELS_PER_BLOCK + threadIdx.x;
//     auto const batch = blockIdx.y;
//     auto const head = blockIdx.z;
//     auto const group = head / (nheads / ngroups);

//     if (idx_dim >= dim)
//         return;

//     // adjust state pointer within the cache
//     if (state_batch_indices)
//     {
//         // state_batch_indices: (batch,)
//         auto batch_idx = state_batch_indices[batch];
//         if (batch_idx == params.pad_slot_id)
//         {
//             return;
//         }
//         state += batch_idx * nheads * dim * DSTATE;
//     }
//     else
//     {
//         state += batch * nheads * dim * DSTATE;
//     }

//     // int const state_loops = (params.dstate + STATE_UNROLL - 1) / STATE_UNROLL;

//     // x: (batch, nheads, dim)
//     auto const x_offset = (batch * nheads + head) * dim + idx_dim;
//     auto x_value = toFloat(x[x_offset]);

//     // z: (batch, nheads, dim)
//     auto z_value = z ? toFloat(z[batch * nheads * dim + head * dim + idx_dim]) : 0.f;

//     // dt: (batch, nheads, dim)
//     auto const dt_offset = x_offset;
//     auto dt_value = toFloat(dt[dt_offset]);

//     if (dt_bias)
//     {
//         dt_value += toFloat(dt_bias[head * dim + idx_dim]);
//     }
//     if (dt_softplus)
//     {
//         dt_value = thresholded_softplus(dt_value);
//     }

//     // Load matrices
//     float rA[DSTATE];
//     float rB[DSTATE];
//     float rC[DSTATE];
//     float rState[DSTATE];
//     for (int i = 0; i < DSTATE; i++)
//     {
//         // A: (nheads, dim, dstate)
//         rA[i] = toFloat(A[head * dim * DSTATE + idx_dim * DSTATE + i]);
//         // B: (batch, ngroups, dstate)
//         rB[i] = toFloat(B[batch * ngroups * DSTATE + group * DSTATE + i]);
//         // C: (batch, ngroups, dstate)
//         rC[i] = toFloat(C[batch * ngroups * DSTATE + group * DSTATE + i]);
//         // state: (batch, nheads, dim, dstate)
//         rState[i] = toFloat(state[head * dim * DSTATE + idx_dim * DSTATE + i]);
//     }

//     // Update sate and compute output
//     // D: (nheads, dim)
//     float out_value = D ? toFloat(D[head * dim + idx_dim]) * x_value : 0.f;
//     for (int i = 0; i < DSTATE; i++)
//     {
//         auto const dA = __expf(rA[i] * dt_value);
//         auto const dB = rB[i] * dt_value;
//         auto const new_state = rState[i] * dA + dB * x_value;
//         convertAndStore(&state[head * dim * DSTATE + idx_dim * DSTATE + i], new_state);
//         out_value += new_state * rC[i];
//     }

//     if (z)
//     {
//         float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
//         float silu_z = z_value * sig_z;
//         out_value *= silu_z;
//     }

//     // output: (batch, nheads, dim)
//     convertAndStore(&output[x_offset], out_value);
// }

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

template <typename input_t, typename weight_t, typename state_t>
struct VectorizedLoadTraits
{
};

template <>
struct VectorizedLoadTraits<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>
{
    using input = float2;
    using weight = float2;
    using state = float2;
    static constexpr auto chunk_size = sizeof(input) / sizeof(__nv_bfloat16);
};

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

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DSTATE, int numWarps>
__global__ void selective_state_update_kernel_simple(SelectiveStateUpdateParams params)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);              // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);           // dt: (batch, nheads)
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);            // A: (nheads)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);              // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);              // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);             // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias); // (nheads)
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

    auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;
    state += state_batch * nheads * dim * DSTATE + head * dim * DSTATE;

    __shared__ input_t sx[numWarps * warpSize];
    __shared__ float sdt[numWarps * warpSize];
    __shared__ weight_t sz[numWarps * warpSize];

    auto const A_value = toFloat(A[head]);

    auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
    if (dt_bias)
        dt_value += toFloat(dt_bias[head]);
    if (dt_softplus)
    {
        dt_value = thresholded_softplus(dt_value);
    }

    // auto const dA = fast_exp(A_value * dt_value);
    auto const dA = __expf(A_value * dt_value);
    // auto const dA = exp(A_value * dt_value);

    auto d_value = D ? toFloat(D[head]) : 0.f;

    auto _d = warp * warpSize + lane;
    auto d = dim_offset + _d;
    if (d < dim)
    {
        sx[_d] = x[batch * params.x_stride_batch + head * dim + d];
        if (z)
        {
            sz[_d] = z[batch * nheads * dim + head * dim + d];
        }
        else
        {
            convertAndStore(&sz[_d], 0.f);
        }
    }
    else
    {
        convertAndStore(&sx[_d], 0.f);
        convertAndStore(&sz[_d], 0.f);
    }

    using Load = VectorizedLoadTraits<input_t, weight_t, state_t>;

    for (auto _d = warp * warpSize; _d < (warp + 1) * warpSize; _d++)
    {
        auto d = dim_offset + _d;
        if (d >= dim)
            break;

        float x_value = toFloat(sx[_d]);
        float out_value = d_value * x_value * int(lane == 0); // first lane has the value

        for (int i = threadIdx.x * Load::chunk_size; i < DSTATE; i += warpSize * Load::chunk_size)
        {
            auto rState = make_zeros<state_t, Load::state>();
            if (state_batch != params.pad_slot_id)
                rState = *reinterpret_cast<typename Load::state*>(&state[d * DSTATE + i]);
            auto rB = *reinterpret_cast<typename Load::input const*>(
                &B[batch * params.B_stride_batch + group * DSTATE + i]);
            auto rC = *reinterpret_cast<typename Load::input const*>(
                &C[batch * params.C_stride_batch + group * DSTATE + i]);

            auto* state_vals = reinterpret_cast<state_t*>(&rState);
            auto const* B_vals = reinterpret_cast<input_t const*>(&rB);
            auto const* C_vals = reinterpret_cast<input_t const*>(&rC);

            for (int ii = 0; ii < Load::chunk_size; ii++)
            {
                auto state_value = toFloat(state_vals[ii]);
                auto B_value = toFloat(B_vals[ii]);
                auto C_value = toFloat(C_vals[ii]);

                auto const dB = B_value * dt_value;
                auto const new_state = state_value * dA + dB * x_value;
                convertAndStore(&state_vals[ii], new_state);

                out_value += new_state * C_value;
            }
            if (state_batch != params.pad_slot_id)
                *reinterpret_cast<typename Load::state*>(&state[d * DSTATE + i]) = rState;
        }

        // warpReduce the out_value
        out_value = warpReduceSum(out_value);
        if (lane == 0)
        {
            sdt[_d] = out_value;
        }
    }

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
        convertAndStore(&output[batch * params.out_stride_batch + head * dim + d], out_value);
    }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int rowsPerStage, int dim,
    int dstate, uint8_t numStages>
struct SharedStorage
{
    alignas(128) state_t state[numStages][rowsPerStage * dstate];
    input_t x[dim];
    float out[dim]; // dt is special cause we're gonna store input in there as well
    input_t z[dim];
    // input_t D[dim];
    input_t B[dstate];
    input_t C[dstate];

    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    barrier_t bar_empty[numStages];
    barrier_t bar_full[numStages];
    barrier_t bar_consumers;
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM, int DSTATE,
    int consumerWarps, int rowsPerStage, int numStages = 1>
__global__ void selective_state_update_kernel_producer_consumer(
    SelectiveStateUpdateParams params, __grid_constant__ CUtensorMap const tensorState)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)
    auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);   // state: (batch, nheads, dim, dstate)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);  // A: (nheads)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);

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

    using sram_t = SharedStorage<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM, DSTATE, numStages>;
#pragma nv_diag_suppress 20054
    __shared__ sram_t sram;
#pragma nv_diag_default 20054

    namespace cde = cuda::device::experimental;
    namespace cg = cooperative_groups;

    for (int stage = warp; stage < numStages; stage += numWarps)
    {
        if (lane > 0)
            continue;
        constexpr auto num_arrivals = 1 + consumerWarps * warpSize;
        init(&sram.bar_empty[stage], num_arrivals);
        init(&sram.bar_full[stage], num_arrivals);
        // signal to async proxy that barriers are initilized
        cde::fence_proxy_async_shared_cta();
    }
    if (lane == 0 && warp == 0)
    {
        init(&sram.bar_consumers, warpSize * consumerWarps);
    }
    __syncthreads();

    if (warp == consumerWarps)
    { // producer
        for (int d = 0, stage = 0; d < dim; d += rowsPerStage, stage = (stage + 1) % numStages)
        {
            if (lane == 0)
            {
                cg::invoke_one(cg::coalesced_threads(),
                    [&]()
                    {
                        sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

                        if (state_batch != params.pad_slot_id)
                        {
                            uint64_t const state_offset = (state_batch * nheads + head) * dim;
                            cde::cp_async_bulk_tensor_2d_global_to_shared(&sram.state[stage][0], &tensorState, /*x*/ 0,
                                /*y*/ state_offset + d, sram.bar_full[stage]);

                            // Unblock the consumers
                            auto constexpr bytesState = rowsPerStage * DSTATE * sizeof(state_t);
                            auto constexpr bytesToArrive = bytesState;
                            auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
                        }
                        else
                            auto const _ = sram.bar_full[stage].arrive();
                    });
            }
        }
    }
    else
    { // consumers

        using load_t = float2;
        static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);

#pragma unroll
        // Unblock the producer
        for (uint8_t stage = 0; stage < numStages; ++stage)
        {
            auto const _ = sram.bar_empty[stage].arrive();
        }

        // Load A
        auto const A_value = toFloat(A[head]);

        // Load D
        auto d_value = D ? toFloat(D[head]) : 0.f;

        // load dt_value
        auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
        if (dt_bias)
            dt_value += toFloat(dt_bias[head]);
        if (params.dt_softplus)
        {
            dt_value = thresholded_softplus(dt_value);
        }

        if (warp == 0)
        { // Load x, B
            for (auto d = lane * vectorizedLoadSize; d < dim; d += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.x[d]);
                *dst = *reinterpret_cast<load_t const*>(&x[batch * params.x_stride_batch + head * dim + d]);
            }
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.B[i]);
                *dst = *reinterpret_cast<load_t const*>(&B[batch * params.B_stride_batch + group * DSTATE + i]);
            }
        }
        else if (warp == 1)
        { // Load z, C
            for (auto d = lane * vectorizedLoadSize; d < dim; d += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.z[d]);
                *dst = z ? *reinterpret_cast<load_t const*>(&z[batch * nheads * dim + head * dim + d])
                         : make_zero<load_t>();
            }
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.C[i]);
                *dst = *reinterpret_cast<load_t const*>(&C[batch * params.C_stride_batch + group * DSTATE + i]);
            }
        }

        sram.bar_consumers.wait(sram.bar_consumers.arrive());

        for (auto dBegin = 0, stage = 0; dBegin < dim; dBegin += rowsPerStage, stage = (stage + 1) % numStages)
        {
            // wait for the producer
            sram.bar_full[stage].wait(sram.bar_full[stage].arrive());

#pragma unroll
            for (auto ddBegin = 0; ddBegin < rowsPerStage; ddBegin += consumerWarps)
            {
                auto dd = ddBegin + warp;
                auto d = dBegin + dd;
                float const x_value = toFloat(sram.x[d]);
                float out_value = toFloat(d_value) * x_value * int(lane == 0); // first lane has the value

                for (int i = lane; i < DSTATE; i += warpSize)
                {
                    auto const state_value
                        = (state_batch != params.pad_slot_id) ? toFloat(sram.state[stage][dd * DSTATE + i]) : 0.f;
                    auto const B_value = toFloat(sram.B[i]);
                    auto const C_value = toFloat(sram.C[i]);

                    // auto const dA = fast_exp(A_value * dt_value);
                    auto const dA = __expf(A_value * dt_value);
                    auto const dB = B_value * dt_value;
                    auto const new_state = state_value * dA + dB * x_value;

                    convertAndStore(&sram.state[stage][dd * DSTATE + i], new_state);
                    out_value += new_state * C_value;
                }

                constexpr auto state_chunk = sizeof(load_t) / sizeof(state_t);
                for (int i = lane * state_chunk; i < DSTATE; i += warpSize * state_chunk)
                {
                    auto const* src = reinterpret_cast<load_t*>(&sram.state[stage][dd * DSTATE + i]);
                    auto* dst = reinterpret_cast<load_t*>(
                        &state[state_batch * nheads * dim * DSTATE + head * dim * DSTATE + d * DSTATE + i]);
                    if (state_batch != params.pad_slot_id)
                        *dst = *src;
                }

                out_value = warpReduceSum(out_value);
                if (lane == 0)
                {
                    sram.out[d] = out_value;
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
            auto out_value = sram.out[d];
            if (z)
            {
                float z_value = toFloat(sram.z[d]);
                float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
                float silu_z = z_value * sig_z;
                out_value *= silu_z;
            }
            convertAndStore(&output[batch * params.out_stride_batch + head * dim + d], out_value);
        }
    }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM, int DSTATE,
    int consumerWarps, int rowsPerStage, int numStages = 1>
__global__ void selective_state_update_kernel_producer_consumer_writeback(
    SelectiveStateUpdateParams params, __grid_constant__ CUtensorMap const tensorState)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);

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

    using sram_t = SharedStorage<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM, DSTATE, numStages>;
#pragma nv_diag_suppress 20054
    __shared__ sram_t sram;
#pragma nv_diag_default 20054

    namespace cde = cuda::device::experimental;
    namespace cg = cooperative_groups;

    for (int stage = warp; stage < numStages; stage += numWarps)
    {
        if (lane > 0)
            continue;
        constexpr auto num_arrivals = 1 + consumerWarps * warpSize;
        init(&sram.bar_empty[stage], num_arrivals);
        init(&sram.bar_full[stage], num_arrivals);
        // signal to async proxy that barriers are initilized
        cde::fence_proxy_async_shared_cta();
    }
    if (lane == 0 && warp == 0)
    {
        init(&sram.bar_consumers, warpSize * consumerWarps);
    }
    __syncthreads();

    if (warp == consumerWarps)
    {
        auto const state_offset = (state_batch * nheads + head) * dim;

        for (int d = 0, stage = 0; d < dim + rowsPerStage * numStages;
             d += rowsPerStage, stage = (stage + 1) % numStages)
        {
            if (lane == 0)
            {
                cg::invoke_one(cg::coalesced_threads(),
                    [&]()
                    {
                        sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

                        if (state_batch != params.pad_slot_id)
                        {
                            // Writeback
                            if (d >= rowsPerStage * numStages)
                            {
                                cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorState,
                                    /*x*/ 0,
                                    /*y*/ state_offset + d - rowsPerStage * numStages, &sram.state[stage][0]);
                                cde::cp_async_bulk_commit_group();
                                cde::cp_async_bulk_wait_group_read<0>();
                            }

                            if (d < dim)
                            {
                                cde::cp_async_bulk_tensor_2d_global_to_shared(&sram.state[stage][0], &tensorState,
                                    /*x*/ 0, /*y*/ state_offset + d, sram.bar_full[stage]);

                                // Unblock the consumers
                                auto constexpr bytesState = rowsPerStage * DSTATE * sizeof(state_t);
                                auto constexpr bytesToArrive = bytesState;
                                auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
                            }
                        }
                        else
                        {
                            auto const _ = sram.bar_full[stage].arrive();
                        }
                    });
            }
        }
    }
    else
    { // consumers

        using load_t = float2;
        static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);

#pragma unroll
        // Unblock the producer
        for (uint8_t stage = 0; stage < numStages; ++stage)
        {
            auto const _ = sram.bar_empty[stage].arrive();
        }

        // Load A
        auto const A_value = toFloat(A[head]);

        // Load D
        auto const d_value = D ? toFloat(D[head]) : 0.f;

        // load dt_value
        auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
        if (dt_bias)
            dt_value += toFloat(dt_bias[head]);
        if (params.dt_softplus)
        {
            dt_value = thresholded_softplus(dt_value);
        }
        auto const dA = __expf(A_value * dt_value);

        if (warp == 0)
        { // Load x, B
            for (auto d = lane * vectorizedLoadSize; d < dim; d += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.x[d]);
                *dst = *reinterpret_cast<load_t const*>(&x[batch * params.x_stride_batch + head * dim + d]);
            }
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.B[i]);
                *dst = *reinterpret_cast<load_t const*>(&B[batch * params.B_stride_batch + group * DSTATE + i]);
            }
        }
        else if (warp == 1)
        { // Load z, C
            for (auto d = lane * vectorizedLoadSize; d < dim; d += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.z[d]);
                *dst = z ? *reinterpret_cast<load_t const*>(&z[batch * params.z_stride_batch + head * dim + d])
                         : make_zero<load_t>();
            }
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.C[i]);
                *dst = *reinterpret_cast<load_t const*>(&C[batch * params.C_stride_batch + group * DSTATE + i]);
            }
        }

        sram.bar_consumers.wait(sram.bar_consumers.arrive());

        for (auto dBegin = 0, stage = 0; dBegin < dim; dBegin += rowsPerStage, stage = (stage + 1) % numStages)
        {
            // wait for the producer
            sram.bar_full[stage].wait(sram.bar_full[stage].arrive());

#pragma unroll
            for (auto dd = warp; dd < rowsPerStage; dd += consumerWarps)
            {
                auto d = dBegin + dd;
                float const x_value = toFloat(sram.x[d]);
                float out_value = d_value * x_value * int(lane == 0); // first lane has the value

                for (int i = lane; i < DSTATE; i += warpSize)
                {
                    auto const state_value
                        = (state_batch != params.pad_slot_id) ? toFloat(sram.state[stage][dd * DSTATE + i]) : 0.f;
                    auto const B_value = toFloat(sram.B[i]);
                    auto const C_value = toFloat(sram.C[i]);

                    auto const dB = B_value * dt_value;
                    auto const new_state = state_value * dA + dB * x_value;

                    convertAndStore(&sram.state[stage][dd * DSTATE + i], new_state);
                    out_value += new_state * C_value;
                }

                out_value = warpReduceSum(out_value);
                if (lane == 0)
                {
                    sram.out[d] = out_value;
                }
            }

            // Unblock producer
            cde::fence_proxy_async_shared_cta();
            auto _ = sram.bar_empty[stage].arrive();
        }

        // Write output
        sram.bar_consumers.wait(sram.bar_consumers.arrive());
        auto d = warp * warpSize + lane;
        if (d < dim)
        {
            auto out_value = sram.out[d];
            if (z)
            {
                float z_value = toFloat(sram.z[d]);
                float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
                float silu_z = z_value * sig_z;
                out_value *= silu_z;
            }
            convertAndStore(&output[batch * params.out_stride_batch + head * dim + d], out_value);
        }
    }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, //
    int dim, int dstate, int stageCols, uint8_t numStages>
struct SharedStorageSuper
{
    alignas(128) state_t state[numStages][dim * stageCols];
    input_t B[dstate];
    input_t C[dstate];

    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    barrier_t bar_empty[numStages];
    barrier_t bar_full[numStages];
    barrier_t bar_consumers;
};

// SharedStorage for the horizontal_warps kernel variant
// In this variant:
// - Each thread processes exactly one dim value (one row)
// - Multiple warps process the same set of dims but different dstate columns
// - Warps that share the same dims reduce their out_value via shared memory atomics
// - Only one warp per dim group writes the final output
//
// Example: dim=64, consumerWarps=4
//   dimGroups = 64/32 = 2 groups
//   warpsPerDimGroup = 4/2 = 2
//   Warps 0,1 -> dims 0-31 (reduce together)
//   Warps 2,3 -> dims 32-63 (reduce together)
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int dim, int dstate, int stageCols,
    int numStages, int consumerWarps>
struct SharedStorageWarps
{
    static constexpr int warpSize = 32;
    static constexpr int dimGroups = dim / warpSize;
    static constexpr int warpsPerDimGroup = consumerWarps / dimGroups;
    static_assert(dimGroups * warpSize == dim, "dim must be a multiple of warpSize");
    static_assert(consumerWarps % dimGroups == 0, "consumerWarps must be divisible by dimGroups");

    alignas(128) state_t state[numStages][dim * stageCols];
    input_t B[dstate];
    input_t C[dstate];

    // For reduction across warps within the same dimGroup via atomics
    alignas(128) float out[dim];

    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    barrier_t bar_empty[numStages];
    barrier_t bar_full[numStages];
    barrier_t bar_consumers;
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM, int DSTATE,
    int consumerWarps, int colsPerStage, int numStages = 1>
__global__ void selective_state_update_kernel_producer_consumer_horizontal(
    SelectiveStateUpdateParams params, __grid_constant__ CUtensorMap const tensorState)
{
    // IKET_RANGE_START(INITIALIZE);
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);  // A: (nheads)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;

    constexpr auto warpSize = 32;
    constexpr auto numWarps = 1 + consumerWarps;

    auto const batch = blockIdx.x;
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto lane = threadIdx.x % warpSize;
    auto warp = threadIdx.y;

    auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;

    extern __shared__ uint8_t sbuffer[];
    using sram_t = SharedStorageSuper<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, colsPerStage, numStages>;
    auto& sram = *reinterpret_cast<sram_t*>(sbuffer);

    namespace cde = cuda::device::experimental;
    namespace cg = cooperative_groups;

    for (int stage = warp; stage < numStages; stage += numWarps)
    {
        if (lane > 0)
            continue;
        constexpr auto num_arrivals = 1 + consumerWarps * warpSize;
        init(&sram.bar_empty[stage], num_arrivals);
        init(&sram.bar_full[stage], num_arrivals);
        // signal to async proxy that barriers are initilized
        cde::fence_proxy_async_shared_cta();
    }
    if (lane == 0 && warp == 0)
    {
        init(&sram.bar_consumers, warpSize * consumerWarps);
    }
    __syncthreads();
    // IKET_RANGE_END(INITIALIZE);

    if (warp == consumerWarps)
    {
        auto const state_offset = (state_batch * nheads + head) * DIM;

        for (int i = 0, stage = 0; i < DSTATE + colsPerStage * numStages;
             i += colsPerStage, stage = (stage + 1) % numStages)
        {
            if (lane == 0)
            {
                cg::invoke_one(cg::coalesced_threads(),
                    [&]()
                    {
                        sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

                        if (state_batch != params.pad_slot_id)
                        {

                            // Writeback
                            if (i >= colsPerStage * numStages)
                            {
                                // IKET_RANGE_START(TMA_WRITE);
                                cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorState,
                                    /*x*/ i - colsPerStage * numStages,
                                    /*y*/ state_offset, &sram.state[stage][0]);
                                cde::cp_async_bulk_commit_group();
                                cde::cp_async_bulk_wait_group_read<0>();
                                // IKET_RANGE_END(TMA_WRITE);
                            }

                            if (i < DSTATE)
                            {
                                // IKET_RANGE_START(TMA_READ);
                                cde::cp_async_bulk_tensor_2d_global_to_shared(&sram.state[stage][0], &tensorState,
                                    /*x*/ i, /*y*/ state_offset, sram.bar_full[stage]);

                                // Unblock the consumers
                                auto constexpr bytesState = DIM * colsPerStage * sizeof(state_t);
                                auto constexpr bytesToArrive = bytesState;
                                // auto constexpr bytesToArrive = 0;
                                auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
                                // IKET_RANGE_END(TMA_READ);
                            }
                        }
                        else
                        {
                            auto const _ = sram.bar_full[stage].arrive();
                        }
                    });
            }
        }
    }
    else
    { // consumers

        // IKET_RANGE_START(CONSUMER_LOAD);
        using load_t = float2;
        static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);

        // Unblock the producer
#pragma unroll
        for (auto stage = 0; stage < numStages; ++stage)
        {
            auto const _ = sram.bar_empty[stage].arrive();
        }

        // Load A
        auto const A_value = toFloat(A[head]);

        // Load D
        auto const d_value = D ? toFloat(D[head]) : 0.f;

        // load dt_value
        auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
        if (dt_bias)
            dt_value += toFloat(dt_bias[head]);
        if (params.dt_softplus)
        {
            dt_value = thresholded_softplus(dt_value);
        }

        if (warp == 0)
        { // Load B
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.B[i]);
                *dst = *reinterpret_cast<load_t const*>(&B[batch * params.B_stride_batch + group * DSTATE + i]);
            }
        }
        else if (warp == 1)
        { // Load C
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.C[i]);
                *dst = *reinterpret_cast<load_t const*>(&C[batch * params.C_stride_batch + group * DSTATE + i]);
            }
        }

        constexpr auto lanesPerRow = (consumerWarps * warpSize) / DIM;
        static_assert(lanesPerRow >= 1);
        constexpr auto rowsPerWarp = warpSize / lanesPerRow;
        auto const group = lane % rowsPerWarp;
        auto const member = lane / rowsPerWarp;
        auto const d = warp * rowsPerWarp + group;
        constexpr auto itemsPerThread = colsPerStage / lanesPerRow;
        auto const x_value = toFloat(x[batch * params.x_stride_batch + head * DIM + d]);
        auto const z_value = z ? toFloat(z[batch * nheads * DIM + head * DIM + d]) : 0.f;

        sram.bar_consumers.wait(sram.bar_consumers.arrive());
        // IKET_RANGE_END(CONSUMER_LOAD);

        // Thread
        // float out_value = d_value * x_value;
        float out_value = 0.f;
#pragma unroll 1
        for (int iBegin = 0, stage = 0; iBegin < DSTATE; iBegin += colsPerStage, stage = (stage + 1) % numStages)
        {
            // IKET_RANGE_START(CONSUMER_WAIT);
            // wait for the producer
            sram.bar_full[stage].wait(sram.bar_full[stage].arrive());
            // IKET_RANGE_END(CONSUMER_WAIT);
            // IKET_RANGE_START(CONSUMER_COMPUTE);

            constexpr auto bankSize = sizeof(uint32_t);
            constexpr auto stateValuesPerBank = bankSize / sizeof(state_t);
#pragma unroll
            for (int item = 0; item < itemsPerThread; item += stateValuesPerBank)
            {
                auto const ii = (item + member * itemsPerThread + (group / 4) * 2) % colsPerStage;
                auto const i = iBegin + ii;

                auto* sState_ptr = reinterpret_cast<uint*>(&sram.state[stage][d * colsPerStage + ii]);
                uint32_t rState = *sState_ptr;
                auto* rState_ptr = reinterpret_cast<state_t*>(&rState);
                for (int e = 0; e < stateValuesPerBank; e++)
                {
                    auto const state_value = (state_batch != params.pad_slot_id) ? toFloat(rState_ptr[e]) : 0.f;

                    auto const B_value = toFloat(sram.B[i + e]);
                    auto const C_value = toFloat(sram.C[i + e]);

                    auto const dA = __expf(A_value * dt_value);
                    auto const dB = B_value * dt_value;
                    auto const new_state = state_value * dA + dB * x_value;

                    convertAndStore(&rState_ptr[e], new_state);
                    out_value += new_state * C_value;
                }
                *sState_ptr = rState;
            }

            // Unblock producer
            // IKET_MARK(CONSUMER_CYCLE);
            cde::fence_proxy_async_shared_cta();
            auto _ = sram.bar_empty[stage].arrive();
        }
        // IKET_RANGE_END(CONSUMER_COMPUTE);
        // IKET_RANGE_START(CONSUMER_FINALIZE);

        // for (int s = lanesPerRow / 2; s > 0; s /= 2) {
        //     out_value += __shfl_down_sync(UINT32_MAX, out_value, s);
        // }
        out_value += __shfl_down_sync(UINT32_MAX, out_value, 16);
        if constexpr (lanesPerRow == 4)
        {
            out_value += __shfl_down_sync(UINT32_MAX, out_value, 8);
        }

        if (member == 0)
        {
            out_value += d_value * x_value;

            // Write output
            if (z)
            {
                float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
                float silu_z = z_value * sig_z;
                out_value *= silu_z;
            }
            convertAndStore(&output[batch * params.out_stride_batch + head * DIM + d], out_value);
        }
        // IKET_RANGE_END(CONSUMER_FINALIZE);
    }
}

// New kernel variant where multiple warps share the same dims and reduce via shared memory
// Each thread processes exactly one dim value
// Multiple warps work on the same set of dims but different dstate columns
// After processing, they reduce via shared memory atomics and only one warp writes the output
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM, int DSTATE,
    int consumerWarps, int colsPerStage, int numStages = 1>
__global__ void selective_state_update_kernel_producer_consumer_horizontal_warps(
    SelectiveStateUpdateParams params, __grid_constant__ CUtensorMap const tensorState)
{
    auto* __restrict__ output = reinterpret_cast<input_t*>(params.output); // output: (batch, nheads, dim)

    auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);    // x: (batch, nheads, dim)
    auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt); // dt: (batch, nheads, dim)
    auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);  // A: (nheads)
    auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);    // B: (batch, ngroups, dstate)
    auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);    // C: (batch, ngroups, dstate)
    auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);   // D: (nheads, dim)
    auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
    auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
    auto const* __restrict__ state_batch_indices = reinterpret_cast<int const*>(params.state_batch_indices);

    int const nheads = params.nheads;
    int const ngroups = params.ngroups;

    constexpr auto warpSize = 32;
    constexpr auto numWarps = 1 + consumerWarps; // 1 producer + N consumers
    constexpr auto dimGroups = DIM / warpSize;
    constexpr auto warpsPerDimGroup = consumerWarps / dimGroups;
    static_assert(dimGroups * warpSize == DIM, "DIM must be a multiple of warpSize");
    static_assert(consumerWarps % dimGroups == 0, "consumerWarps must be divisible by dimGroups");

    auto const batch = blockIdx.x;
    auto const head = blockIdx.y;
    auto const group = head / (nheads / ngroups);
    auto lane = threadIdx.x % warpSize;
    auto warp = threadIdx.y;

    auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;

    extern __shared__ uint8_t sbuffer[];
    using sram_t = SharedStorageWarps<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, colsPerStage, numStages,
        consumerWarps>;
    auto& sram = *reinterpret_cast<sram_t*>(sbuffer);

    namespace cde = cuda::device::experimental;
    namespace cg = cooperative_groups;

    // Initialize barriers
    for (int stage = warp; stage < numStages; stage += numWarps)
    {
        if (lane > 0)
            continue;
        constexpr auto num_arrivals = 1 + consumerWarps * warpSize;
        init(&sram.bar_empty[stage], num_arrivals);
        init(&sram.bar_full[stage], num_arrivals);
        cde::fence_proxy_async_shared_cta();
    }
    if (lane == 0 && warp == 0)
    {
        init(&sram.bar_consumers, warpSize * consumerWarps);
    }
    __syncthreads();

    if (warp == consumerWarps)
    {
        // Producer warp - handles TMA reads and writes
        auto const state_offset = (state_batch * nheads + head) * DIM;

        for (int i = 0, stage = 0; i < DSTATE + colsPerStage * numStages;
             i += colsPerStage, stage = (stage + 1) % numStages)
        {
            if (lane == 0)
            {
                cg::invoke_one(cg::coalesced_threads(),
                    [&]()
                    {
                        sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

                        if (state_batch != params.pad_slot_id)
                        {
                            // Writeback
                            if (i >= colsPerStage * numStages)
                            {
                                cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorState,
                                    /*x*/ i - colsPerStage * numStages,
                                    /*y*/ state_offset, &sram.state[stage][0]);
                                cde::cp_async_bulk_commit_group();
                                cde::cp_async_bulk_wait_group_read<0>();
                            }

                            if (i < DSTATE)
                            {
                                cde::cp_async_bulk_tensor_2d_global_to_shared(&sram.state[stage][0], &tensorState,
                                    /*x*/ i, /*y*/ state_offset, sram.bar_full[stage]);

                                auto constexpr bytesState = DIM * colsPerStage * sizeof(state_t);
                                auto constexpr bytesToArrive = bytesState;
                                auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
                            }
                        }
                        else
                        {
                            auto const _ = sram.bar_full[stage].arrive();
                        }
                    });
            }
        }
    }
    else
    {
        // Consumer warps
        using load_t = float2;
        static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);

        // Unblock the producer for all stages
#pragma unroll
        for (auto stage = 0; stage < numStages; ++stage)
        {
            auto const _ = sram.bar_empty[stage].arrive();
        }

        // Load A (scalar per head)
        auto const A_value = toFloat(A[head]);

        // Load D (scalar per head)
        auto const d_value = D ? toFloat(D[head]) : 0.f;

        // Load dt_value (scalar per head)
        auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
        if (dt_bias)
            dt_value += toFloat(dt_bias[head]);
        if (params.dt_softplus)
        {
            dt_value = thresholded_softplus(dt_value);
        }

        // Cooperative load of B and C into shared memory
        // Use first two warps to load B and C respectively
        if (warp == 0)
        {
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.B[i]);
                *dst = *reinterpret_cast<load_t const*>(&B[batch * params.B_stride_batch + group * DSTATE + i]);
            }
        }
        else if (warp == 1)
        {
            for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize)
            {
                auto* dst = reinterpret_cast<load_t*>(&sram.C[i]);
                *dst = *reinterpret_cast<load_t const*>(&C[batch * params.C_stride_batch + group * DSTATE + i]);
            }
        }

        // Compute which dim this thread handles
        // dimGroup: which set of 32 dims (0 to dimGroups-1)
        // warpIdxInGroup: which warp within the dim group (0 to warpsPerDimGroup-1)
        auto const dimGroup = warp / warpsPerDimGroup;
        auto const warpIdxInGroup = warp % warpsPerDimGroup;
        auto const d = dimGroup * warpSize + lane; // The actual dim index this thread handles

        // Each warp in a group processes different columns of dstate
        // colsPerStage columns per stage, divided among warpsPerDimGroup warps
        constexpr auto colsPerWarp = colsPerStage / warpsPerDimGroup;
        static_assert(colsPerStage % warpsPerDimGroup == 0, "colsPerStage must be divisible by warpsPerDimGroup");

        auto const x_value = toFloat(x[batch * params.x_stride_batch + head * DIM + d]);
        auto const z_value = z ? toFloat(z[batch * nheads * DIM + head * DIM + d]) : 0.f;

        // Initialize out[] to zero - done by first warp in each dimGroup
        if (warpIdxInGroup == 0)
        {
            sram.out[d] = 0.f;
        }

        // Wait for B and C to be loaded (also ensures out[] is initialized)
        sram.bar_consumers.wait(sram.bar_consumers.arrive());

        // Accumulate output value
        float out_value = 0.f;

#pragma unroll 1
        for (int iBegin = 0, stage = 0; iBegin < DSTATE; iBegin += colsPerStage, stage = (stage + 1) % numStages)
        {
            // Wait for the producer to fill this stage
            sram.bar_full[stage].wait(sram.bar_full[stage].arrive());

            // Each warp in the group processes its subset of columns
            auto const colStart = warpIdxInGroup * colsPerWarp;

            constexpr auto bankSize = sizeof(uint32_t);
            constexpr auto stateValuesPerBank = bankSize / sizeof(state_t);
#pragma unroll
            for (int col = 0; col < colsPerWarp; col += stateValuesPerBank)
            {
                auto const ii = colStart + col;
                auto const i = iBegin + ii;

                auto* sState_ptr = reinterpret_cast<uint*>(&sram.state[stage][d * colsPerStage + ii]);
                uint32_t rState = *sState_ptr;
                auto* rState_ptr = reinterpret_cast<state_t*>(&rState);
                for (int e = 0; e < stateValuesPerBank; e++)
                {
                    auto const state_value = (state_batch != params.pad_slot_id) ? toFloat(rState_ptr[e]) : 0.f;

                    auto const B_value = toFloat(sram.B[i + e]);
                    auto const C_value = toFloat(sram.C[i + e]);

                    auto const dA = __expf(A_value * dt_value);
                    auto const dB = B_value * dt_value;
                    auto const new_state = state_value * dA + dB * x_value;

                    convertAndStore(&rState_ptr[e], new_state);
                    out_value += new_state * C_value;
                }
                *sState_ptr = rState;
            }
            // #pragma unroll
            //             for (int col = 0; col < colsPerWarp; ++col)
            //             {
            //                 auto const ii = colStart + col;
            //                 auto const i = iBegin + ii;

            //                 auto const state_value =
            //                     (state_batch != params.pad_slot_id) ? toFloat(sram.state[stage][d * colsPerStage +
            //                     ii]) : 0.f;

            //                 auto const B_value = toFloat(sram.B[i]);
            //                 auto const C_value = toFloat(sram.C[i]);

            //                 auto const dA = __expf(A_value * dt_value);
            //                 auto const dB = B_value * dt_value;
            //                 auto const new_state = state_value * dA + dB * x_value;

            //                 // All warps in the group write to the same state location
            //                 // This is safe because they process different columns
            //                 convertAndStore(&sram.state[stage][d * colsPerStage + ii], new_state);
            //                 out_value += new_state * C_value;
            //             }

            // Unblock producer
            cde::fence_proxy_async_shared_cta();
            auto _ = sram.bar_empty[stage].arrive();
        }

        // Atomic add partial out_value to shared memory for cross-warp reduction
        atomicAdd(&sram.out[d], out_value);

        // Synchronize all consumer warps
        __syncthreads();

        // Only the first warp in each dimGroup writes the final output
        if (warpIdxInGroup == 0)
        {
            float reduced_out = sram.out[d];
            reduced_out += d_value * x_value;

            // Apply z gating if present
            if (z)
            {
                float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
                float silu_z = z_value * sig_z;
                reduced_out *= silu_z;
            }

            convertAndStore(&output[batch * params.out_stride_batch + head * DIM + d], reduced_out);
        }
    }
}

template <typename KernelFunc>
void request_sram_if_needed(KernelFunc kernel, size_t sram_required)
{
    int max_sram;
    cudaDeviceGetAttribute(&max_sram, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_required > max_sram)
    {
        std::cerr << "Warning: Requested shared memory " << sram_required << " exceeds the default maximum " << max_sram
                  << std::endl;
        std::cerr << "Trying to bump up the maximum: " << std::endl;
        auto check = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sram_required);
        if (check != cudaSuccess)
        {
            std::cerr << "Failed to set dynamic shared memory size: " << sram_required << std::endl;
            throw std::runtime_error("Failed to set dynamic shared memory size");
        }
    }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t>
void invokeSelectiveStateUpdate(
    SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType)
{
    constexpr auto warpSize = 32;
    constexpr int DSTATE = 128;
    TLLM_CHECK(params.dstate == DSTATE);

    if (kernelType == SelectiveStateUpdateKernelType::simple)
    {
        constexpr int numWarps = 2;
        int const blocks_per_dim = (params.dim + warpSize * numWarps - 1) / (warpSize * numWarps);
        dim3 block(warpSize, numWarps);
        dim3 grid(blocks_per_dim, params.batch, params.nheads);
        selective_state_update_kernel_simple<input_t, weight_t, matrixA_t, state_t, DSTATE, numWarps>
            <<<grid, block, 0, stream>>>(params);
    }
    // assume that A is just one value per head
    else if (kernelType == SelectiveStateUpdateKernelType::producer_consumer)
    {
        constexpr auto numConsumers = 2;
        constexpr auto numWarps = 1 + numConsumers;
        constexpr auto numStages = 2;
        constexpr auto rowsPerStage = 4 * numConsumers;
        constexpr auto DIM = 64;
        TLLM_CHECK(params.dim == DIM);
        auto scan_func = selective_state_update_kernel_producer_consumer<input_t, weight_t, matrixA_t, state_t, DIM,
            DSTATE, numConsumers, rowsPerStage, numStages>;

        dim3 block(warpSize, numWarps);
        dim3 grid(params.batch, params.nheads);

        auto nh = params.nheads;
        auto dim = params.dim;
        auto B = params.state_cache_size;

        TLLM_CHECK(reinterpret_cast<uintptr_t>(params.state) % 128 == 0); // TMA requires 128B aligned
        auto tensorState = tma::createTensorMap<state_t>(params.state, B * nh * dim, DSTATE, rowsPerStage, DSTATE);
        TLLM_CHECK(params.dim % rowsPerStage == 0);

        scan_func<<<grid, block, 0, stream>>>(params, tensorState);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::producer_consumer_writeback)
    {
        constexpr auto numConsumers = 4;
        constexpr auto numWarps = 1 + numConsumers;
        constexpr auto numStages = 3;
        constexpr auto rowsPerStage = 4 * numConsumers;
        constexpr auto DIM = 64;
        TLLM_CHECK(params.dim == DIM);
        auto scan_func = selective_state_update_kernel_producer_consumer_writeback<input_t, weight_t, matrixA_t,
            state_t, DIM, DSTATE, numConsumers, rowsPerStage, numStages>;

        dim3 block(warpSize, numWarps);
        dim3 grid(params.batch, params.nheads);

        auto nh = params.nheads;
        auto dim = params.dim;
        auto B = params.state_cache_size;

        TLLM_CHECK(reinterpret_cast<uintptr_t>(params.state) % 128 == 0); // TMA requires 128B aligned
        auto tensorState = tma::createTensorMap<state_t>(params.state, B * nh * dim, DSTATE, rowsPerStage, DSTATE);
        TLLM_CHECK(params.dim % rowsPerStage == 0);

        scan_func<<<grid, block, 0, stream>>>(params, tensorState);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::producer_consumer_horizontal)
    {
        constexpr auto DIM = 64;
        TLLM_CHECK(params.dim == DIM);
        // constexpr auto numConsumers = (DIM + warpSize - 1) / warpSize;
        constexpr auto numConsumers = 4;
        constexpr auto numProducers = 1;
        constexpr auto numWarps = numProducers + numConsumers;

        // Compute the size (width) of the stage
        constexpr auto sectorSize = 32;                          // bytes
        constexpr auto stageCols = sectorSize / sizeof(state_t); // 32 / 2 = 16;
        constexpr auto numStages = 4;

        auto func = selective_state_update_kernel_producer_consumer_horizontal<input_t, weight_t, matrixA_t, state_t,
            DIM, DSTATE, numConsumers, stageCols, numStages>;

        dim3 block(warpSize, numWarps);
        dim3 grid(params.batch, params.nheads);

        auto nh = params.nheads;
        auto dim = params.dim;
        auto B = params.state_cache_size;

        TLLM_CHECK(reinterpret_cast<uintptr_t>(params.state) % 128 == 0); // TMA requires 128B aligned
        auto tensorState = tma::createTensorMap<state_t>(params.state, B * nh * dim, DSTATE, DIM, stageCols);
        TLLM_CHECK(DSTATE % stageCols == 0);

        using sram_t = SharedStorageSuper<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, stageCols, numStages>;
        constexpr auto sram_required = sizeof(sram_t);
        request_sram_if_needed(func, sram_required);
        func<<<grid, block, sram_required, stream>>>(params, tensorState);
    }
    else if (kernelType == SelectiveStateUpdateKernelType::producer_consumer_horizontal_warps)
    {
        constexpr auto DIM = 64;
        TLLM_CHECK(params.dim == DIM);
        constexpr auto numConsumers = 4;
        constexpr auto numProducers = 1;
        constexpr auto numWarps = numProducers + numConsumers;

        // Compute the size (width) of the stage
        constexpr auto sectorSize = 32;                          // bytes
        constexpr auto stageCols = sectorSize / sizeof(state_t); // 32 / 2 = 16;
        constexpr auto numStages = 4;

        auto func = selective_state_update_kernel_producer_consumer_horizontal_warps<input_t, weight_t, matrixA_t,
            state_t, DIM, DSTATE, numConsumers, stageCols, numStages>;

        dim3 block(warpSize, numWarps);
        dim3 grid(params.batch, params.nheads);

        auto nh = params.nheads;
        auto dim = params.dim;
        auto B = params.state_cache_size;

        TLLM_CHECK(reinterpret_cast<uintptr_t>(params.state) % 128 == 0); // TMA requires 128B aligned
        auto tensorState = tma::createTensorMap<state_t>(params.state, B * nh * dim, DSTATE, DIM, stageCols);
        TLLM_CHECK(DSTATE % stageCols == 0);

        using sram_t = SharedStorageWarps<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, stageCols, numStages,
            numConsumers>;
        constexpr auto sram_required = sizeof(sram_t);
        request_sram_if_needed(func, sram_required);
        func<<<grid, block, sram_required, stream>>>(params, tensorState);
    }
    else
    {
        TLLM_CHECK(false && "Unsupported SelectiveStateUpdateKernelType");
    }
}

// we should focus on BF16, FP16 and even FP32 where the Mamba states are involved.
// template void invokeSelectiveStateUpdate<half, half, float>(
//     SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType);

// template void invokeSelectiveStateUpdate<float, float>(
//     SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType);

template void invokeSelectiveStateUpdate<__nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16>(
    SelectiveStateUpdateParams& params, cudaStream_t stream, SelectiveStateUpdateKernelType kernelType);

} // end namespace tensorrt_llm::kernels
