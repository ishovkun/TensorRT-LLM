/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vaeRmsNormSiluKernel.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/packedVector.cuh"
#include "vaeNormCommon.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Streaming RMSNorm + SiLU over channels-last rows.
//
// Rows are read straight from global memory into registers: a row is owned by L
// lanes, each holding K vectors of V elements, and a warp keeps U rows in flight so
// the loads of U rows are issued before any math. L and K come from the channel
// count so every lane owns real data for every supported width. Per-channel
// constants sit in shared memory; a lane holds only its own vectors.

namespace
{

using common::PackedAligned;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlocksPerSm = 4;

using namespace vae_norm;

// Rows per warp per step: enough independent dependency chains to cover latency.
// With a residual input each row costs two vectors per lane, so half as many rows.
template <int K, bool HAS_RESIDUAL>
constexpr int rowsInFlight()
{
    constexpr int base = K >= 4 ? 1 : (K >= 2 ? 2 : 4);
    return HAS_RESIDUAL ? (base > 1 ? base / 2 : 1) : base;
}

// HAS_IN_BIAS: x is a convolution output whose per-channel bias was not applied; the
// kernel applies it first as round(x + inBias), exactly like the eager bias add.
// HAS_RESIDUAL: the normalized row is s = round(x + residual); s is also written to
// ``sumOut`` because the next block needs it as its own residual. The rounding
// matches an eager bf16 add, so the sum is bit-identical to ``x + residual``.
template <typename TIn, typename TOut, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm) vaeRmsNormSiluKernel(TIn const* __restrict__ x,
    TOut* __restrict__ y, TIn const* __restrict__ gamma, TIn const* __restrict__ bias,
    TIn const* __restrict__ inBias, TIn const* __restrict__ residual, TIn* __restrict__ sumOut, int64_t numRows,
    int channels, float scale, float eps)
{
    using VecIn = PackedAligned<TIn, V>;
    using VecOut = PackedAligned<TOut, V>;
    using VecF = PackedAligned<float, V>;
    constexpr int U = rowsInFlight<K, HAS_RESIDUAL>();
    constexpr int kRowsPerWarpIter = 32 / L;
    constexpr int kRowsPerWarp = kRowsPerWarpIter * U;
    constexpr int kRowsPerBlock = kRowsPerWarp * kWarps;

    extern __shared__ __align__(16) float smemF[];
    float* const sGamma = smemF;
    float* const sBias = smemF + channels;
    float* const sInBias = smemF + 2 * channels;

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const group = lane / L;
    int const laneInGroup = lane - group * L;

    for (int c = tid; c < channels; c += kThreads)
    {
        float const g = Elem<TIn>::toFloat(gamma[c]);
        sGamma[c] = REF ? g : g * (0.5f * scale);
        sBias[c] = HAS_BIAS ? Elem<TIn>::toFloat(bias[c]) : 0.0f;
        sInBias[c] = HAS_IN_BIAS ? Elem<TIn>::toFloat(inBias[c]) : 0.0f;
    }
    __syncthreads();

    // Block-relative row offsets of this lane's rows and its first chunk; constant across steps.
    int rowOff[U];
#pragma unroll
    for (int u = 0; u < U; ++u)
    {
        rowOff[u] = (warp * kRowsPerWarp + u * kRowsPerWarpIter + group) * channels;
    }
    int const laneChunkOff = laneInGroup * V;
    int64_t const gridRowsPerStep = static_cast<int64_t>(gridDim.x) * kRowsPerBlock;
    // Single-vector layouts keep their gamma/bias in registers (8 + 8 floats).
    constexpr bool kConstInRegs = (K == 1);
    VecF gReg = common::make_zeros<VecF>();
    VecF bReg = common::make_zeros<VecF>();
    VecF ibReg = common::make_zeros<VecF>();
    if constexpr (kConstInRegs)
    {
        gReg = *reinterpret_cast<VecF const*>(sGamma + laneChunkOff);
        if constexpr (HAS_BIAS)
        {
            bReg = *reinterpret_cast<VecF const*>(sBias + laneChunkOff);
        }
        if constexpr (HAS_IN_BIAS)
        {
            ibReg = *reinterpret_cast<VecF const*>(sInBias + laneChunkOff);
        }
    }

    // The full-step body has no row predicates at all; only the last, partial step
    // of a block clamps its loads and predicates its stores.
    auto step = [&](auto fullStepTag, int64_t blockRow0, int rowsLeft)
    {
        constexpr bool FULLSTEP = decltype(fullStepTag)::value;
        TIn const* const xLane = x + blockRow0 * channels + laneChunkOff;
        TOut* const yLane = y + blockRow0 * channels + laneChunkOff;
        TIn const* const rLane = HAS_RESIDUAL ? residual + blockRow0 * channels + laneChunkOff : nullptr;
        TIn* const sLane = HAS_RESIDUAL ? sumOut + blockRow0 * channels + laneChunkOff : nullptr;

        // Issue every load of this warp's rows before touching any of them; rows past the
        // end are clamped onto the last row and never stored.
        VecIn v[U][K];
        VecIn vr[HAS_RESIDUAL ? U : 1][HAS_RESIDUAL ? K : 1];
        int offs[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
        {
            int off = rowOff[u];
            if constexpr (!FULLSTEP)
            {
                int const row = warp * kRowsPerWarp + u * kRowsPerWarpIter + group;
                off = min(row, rowsLeft - 1) * channels;
            }
            offs[u] = off;
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                v[u][k] = *reinterpret_cast<VecIn const*>(xLane + off + k * L * V);
                if constexpr (HAS_RESIDUAL)
                {
                    vr[u][k] = *reinterpret_cast<VecIn const*>(rLane + off + k * L * V);
                }
            }
        }
        if constexpr (HAS_IN_BIAS)
        {
            // x = round(x + inBias): the deferred convolution bias, rounded like the eager add.
#pragma unroll
            for (int u = 0; u < U; ++u)
            {
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    VecF ib = ibReg;
                    if constexpr (!kConstInRegs)
                    {
                        ib = *reinterpret_cast<VecF const*>(sInBias + laneChunkOff + k * L * V);
                    }
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const a = pairAt(v[u][k], i);
                        setPair(v[u][k], i, make_float2(a.x + ib.val[2 * i], a.y + ib.val[2 * i + 1]));
                    }
                }
            }
        }
        if constexpr (HAS_RESIDUAL)
        {
            // s = round(x + residual), kept in v; the sum row is stored alongside the activation.
#pragma unroll
            for (int u = 0; u < U; ++u)
            {
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const a = pairAt(v[u][k], i);
                        float2 const b = pairAt(vr[u][k], i);
                        setPair(v[u][k], i, make_float2(a.x + b.x, a.y + b.y));
                    }
                }
            }
        }

        float ss0[U];
        float ss1[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
        {
            ss0[u] = 0.0f;
            ss1[u] = 0.0f;
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
#pragma unroll
                for (int i = 0; i < V / 2; ++i)
                {
                    float2 const f = pairAt(v[u][k], i);
                    ss0[u] = fmaf(f.x, f.x, ss0[u]);
                    ss1[u] = fmaf(f.y, f.y, ss1[u]);
                }
            }
        }
        float sumsq[U];
#pragma unroll
        for (int u = 0; u < U; ++u)
        {
            sumsq[u] = ss0[u] + ss1[u];
        }
        // Reduce across the L lanes that share a row; the U rows' shuffles interleave.
#pragma unroll
        for (int off = L >> 1; off > 0; off >>= 1)
        {
#pragma unroll
            for (int u = 0; u < U; ++u)
            {
                sumsq[u] += __shfl_xor_sync(0xffffffffu, sumsq[u], off);
            }
        }

#pragma unroll
        for (int u = 0; u < U; ++u)
        {
            float denom = 0.0f;
            float invDenom = 0.0f;
            if constexpr (REF)
            {
                denom = fmaxf(sqrtf(sumsq[u]), eps);
            }
            else
            {
                // 1 / max(sqrt(ss), eps) == rsqrt(max(ss, eps^2)); one MUFU, no slow path.
                invDenom = rsqrtf(fmaxf(sumsq[u], eps * eps));
            }
            int const row = warp * kRowsPerWarp + u * kRowsPerWarpIter + group;
            if constexpr (!FULLSTEP)
            {
                if (row >= rowsLeft)
                {
                    continue;
                }
            }
            TOut* const outRow = yLane + row * channels;
            if constexpr (HAS_RESIDUAL)
            {
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    *reinterpret_cast<VecIn*>(sLane + row * channels + k * L * V) = v[u][k];
                }
            }
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                int const c0 = laneChunkOff + k * L * V;
                VecF g = gReg;
                VecF b = bReg;
                if constexpr (!kConstInRegs)
                {
                    g = *reinterpret_cast<VecF const*>(sGamma + c0);
                    if constexpr (HAS_BIAS)
                    {
                        b = *reinterpret_cast<VecF const*>(sBias + c0);
                    }
                }
                VecOut o;
#pragma unroll
                for (int i = 0; i < V / 2; ++i)
                {
                    float2 const f = pairAt(v[u][k], i);
                    float2 r;
                    r.x = normElement<TIn, REF, HAS_BIAS>(f.x, denom, invDenom, scale, g.val[2 * i], b.val[2 * i]);
                    r.y = normElement<TIn, REF, HAS_BIAS>(
                        f.y, denom, invDenom, scale, g.val[2 * i + 1], b.val[2 * i + 1]);
                    setPair(o, i, r);
                }
                *reinterpret_cast<VecOut*>(outRow + k * L * V) = o;
            }
        }
    };

    for (int64_t blockRow0 = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock; blockRow0 < numRows;
         blockRow0 += gridRowsPerStep)
    {
        int64_t const remaining = numRows - blockRow0;
        if (remaining >= kRowsPerBlock)
        {
            step(std::true_type{}, blockRow0, kRowsPerBlock);
        }
        else
        {
            step(std::false_type{}, blockRow0, static_cast<int>(remaining));
        }
    }
}

template <typename TIn, typename TOut, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL>
void launchConfig(TIn const* x, TOut* y, TIn const* gamma, TIn const* bias, TIn const* inBias, TIn const* residual,
    TIn* sumOut, int64_t numRows, int channels, float scale, float eps, cudaStream_t stream)
{
    auto* kernel = vaeRmsNormSiluKernel<TIn, TOut, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL>;
    constexpr int kRowsPerBlock = (32 / L) * rowsInFlight<K, HAS_RESIDUAL>() * kWarps;
    int const smem = 3 * channels * static_cast<int>(sizeof(float));
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int64_t const steps = (numRows + kRowsPerBlock - 1) / kRowsPerBlock;
    int64_t const persistent = static_cast<int64_t>(sms) * kBlocksPerSm;
    unsigned const blocks = static_cast<unsigned>(steps < persistent ? steps : persistent);
    kernel<<<blocks, kThreads, smem, stream>>>(
        x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps);
}

template <typename TIn, typename TOut, int V, int L_, int K_, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS>
void launchResidual(TIn const* x, TOut* y, TIn const* gamma, TIn const* bias, TIn const* inBias,
    TIn const* residual, TIn* sumOut, int64_t numRows, int channels, float scale, float eps, cudaStream_t stream)
{
    if (residual != nullptr)
        launchConfig<TIn, TOut, V, L_, K_, REF, HAS_BIAS, HAS_IN_BIAS, true>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
    else
        launchConfig<TIn, TOut, V, L_, K_, REF, HAS_BIAS, HAS_IN_BIAS, false>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
}

template <typename TIn, typename TOut, int V, int L_, int K_, bool REF, bool HAS_BIAS>
void launchInBias(TIn const* x, TOut* y, TIn const* gamma, TIn const* bias, TIn const* inBias, TIn const* residual,
    TIn* sumOut, int64_t numRows, int channels, float scale, float eps, cudaStream_t stream)
{
    if (inBias != nullptr)
        launchResidual<TIn, TOut, V, L_, K_, REF, HAS_BIAS, true>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
    else
        launchResidual<TIn, TOut, V, L_, K_, REF, HAS_BIAS, false>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
}

template <typename TIn, typename TOut, int V, int L_, int K_>
void launchModes(TIn const* x, TOut* y, TIn const* gamma, TIn const* bias, TIn const* inBias, TIn const* residual,
    TIn* sumOut, int64_t numRows, int channels, float scale, float eps, bool ref, cudaStream_t stream)
{
    bool const hasBias = bias != nullptr;
    if (hasBias && ref)
        launchInBias<TIn, TOut, V, L_, K_, true, true>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
    else if (hasBias)
        launchInBias<TIn, TOut, V, L_, K_, false, true>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
    else if (ref)
        launchInBias<TIn, TOut, V, L_, K_, true, false>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
    else
        launchInBias<TIn, TOut, V, L_, K_, false, false>(
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, stream);
}

// Pick a compile-time lane layout for the (lanes, vectors) pair the width maps to;
// anything else runs the generic kernel with the layout read at runtime.
template <typename TIn, typename TOut, int V>
void launchLayout(TIn const* x, TOut* y, TIn const* gamma, TIn const* bias, TIn const* inBias, TIn const* residual,
    TIn* sumOut, int64_t numRows, int channels, float scale, float eps, bool ref, cudaStream_t stream)
{
    RowLayout const l = makeRowLayout<V>(channels);
#define TRY_LAYOUT(LL, KK)                                                                                            \
    if (l.lanesPerRow == (LL) && l.vecsPerLane == (KK))                                                                \
    {                                                                                                                  \
        launchModes<TIn, TOut, V, LL, KK>(                                                                             \
            x, y, gamma, bias, inBias, residual, sumOut, numRows, channels, scale, eps, ref, stream);                 \
        return;                                                                                                        \
    }
    VAE_NORM_FOR_EACH_LAYOUT(TRY_LAYOUT)
#undef TRY_LAYOUT
    TLLM_THROW("vaeRmsNormSilu: no lane layout for %d channels (lanes %d, vectors %d)", channels, l.lanesPerRow,
        l.vecsPerLane);
}

} // namespace

void launchVaeRmsNormSilu(void const* xv, void* yv, void const* gv, void const* bv, void const* ibv, void const* rv,
    void* sv, int64_t numRows, int channels, float scale, float eps, bool referenceRounding, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO((rv == nullptr) == (sv == nullptr), "vaeRmsNormSilu: residual and sum output go together");
    // One lane vector is a 16-byte load, so a row is a whole number of vectors.
    constexpr int kVecBf16 = static_cast<int>(sizeof(float4) / sizeof(__nv_bfloat16));
    TLLM_CHECK_WITH_INFO(channels > 0 && channels % kVecBf16 == 0 && channels <= kVaeRmsNormSiluMaxChannels,
        "vaeRmsNormSilu: channels must be a positive multiple of %d up to %d, got %d", kVecBf16,
        kVaeRmsNormSiluMaxChannels, channels);
    if (numRows == 0)
    {
        return;
    }
    launchLayout<__nv_bfloat16, __nv_bfloat16, kVecBf16>(reinterpret_cast<__nv_bfloat16 const*>(xv),
        reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16 const*>(gv),
        reinterpret_cast<__nv_bfloat16 const*>(bv), reinterpret_cast<__nv_bfloat16 const*>(ibv),
        reinterpret_cast<__nv_bfloat16 const*>(rv), reinterpret_cast<__nv_bfloat16*>(sv), numRows, channels, scale,
        eps, referenceRounding, stream);
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
