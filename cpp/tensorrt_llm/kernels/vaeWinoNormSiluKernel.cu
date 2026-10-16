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

#include "vaeWinoNormSiluKernel.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/packedVector.cuh"
#include "vaeNormCommon.cuh"
#include "vaeRmsNormSiluKernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using common::PackedAligned;
using namespace vae_norm;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

template <typename T>
struct Ptrs
{
    T const* x;
    T const* m[4];
    T const* residual;
    T const* inBias;
    T const* cache;
    T const* gamma;
    T const* bias;
    T* sum;
    T* yTail;
    T* d[4];
};

// a (+/-) b with the eager bf16 rounding in reference mode.
template <typename T, bool REF>
__device__ __forceinline__ float addR(float a, float b)
{
    float const s = a + b;
    return REF ? roundTo<T>(s) : s;
}

// Per-layout tuning. ``frames``: frames handled per group, always whole pairs.
// ``hoist``: issue the whole group's loads (up to four inputs per frame) before
// normalizing any frame, the kernel's memory-level parallelism; wider layouts would
// spill, so they load frame by frame. ``blocks``: resident blocks per SM the register
// budget is sized for (the kernel keeps four normalized frames plus the group's
// inputs in registers, so one block of 8 warps with up to 255 registers per thread
// beats two blocks that spill).
struct WinoTuning
{
    int frames;
    bool hoist;
    int blocks;
};

template <int K, bool IN_WINO, bool HAS_RESIDUAL>
constexpr WinoTuning winoTuning()
{
    // Measured on GB200 (bench_wino_norm_kernel.py): the one-vector layout wants
    // occupancy over bytes in flight, the two-vector producer gains from hoisting,
    // the five-vector layouts spill below 255 registers.
    if constexpr (K == 1)
    {
        return WinoTuning{2, false, 3}; // ~90 registers; occupancy is what this layout lacks
    }
    else if constexpr (K == 2)
    {
        return WinoTuning{2, !IN_WINO, 2};
    }
    else if constexpr (K >= 4)
    {
        return WinoTuning{2, false, 1}; // spills at 128 registers
    }
    else
    {
        return WinoTuning{2, false, 2};
    }
}

// One thread group (L lanes, K vectors each) owns one position and walks the frames
// in groups of G, keeping the two previous normalized frames in registers so the
// Winograd slabs of every pair can be written as soon as its second frame is done.
template <typename T, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS, bool HAS_RESIDUAL, bool IN_WINO>
__global__ void __launch_bounds__(kThreads, winoTuning<K, IN_WINO, HAS_RESIDUAL>().blocks) vaeWinoNormSiluKernel(
    Ptrs<T> ptrs, VaeWinoNormParams p)
{
    using Vec = PackedAligned<T, V>;
    using VecF = PackedAligned<float, V>;
    constexpr WinoTuning kTune = winoTuning<K, IN_WINO, HAS_RESIDUAL>();
    constexpr int G = kTune.frames;
    constexpr bool HOIST = kTune.hoist;
    constexpr int NIN = IN_WINO ? 3 : 1;
    constexpr int kPosPerWarp = 32 / L;
    constexpr int kPosPerBlock = kPosPerWarp * kWarps;

    extern __shared__ __align__(16) float smemF[];
    float* const sGamma = smemF;
    float* const sBias = smemF + p.channels;
    float* const sInBias = smemF + 2 * p.channels;

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const group = lane / L;
    int const laneInGroup = lane - group * L;
    int const C = p.channels;
    int64_t const P = p.positions;

    for (int c = tid; c < C; c += kThreads)
    {
        float const g = Elem<T>::toFloat(ptrs.gamma[c]);
        sGamma[c] = REF ? g : g * (0.5f * p.scale);
        sBias[c] = HAS_BIAS ? Elem<T>::toFloat(ptrs.bias[c]) : 0.0f;
        sInBias[c] = HAS_IN_BIAS ? Elem<T>::toFloat(ptrs.inBias[c]) : 0.0f;
    }
    __syncthreads();

    int const laneChunkOff = laneInGroup * V;
    int64_t const gridPos = static_cast<int64_t>(gridDim.x) * kPosPerBlock;
    int const T_ = p.frames;

    auto loadRow = [&](T const* base, int64_t row, Vec (&out)[K])
    {
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            out[k] = *reinterpret_cast<Vec const*>(base + row * C + laneChunkOff + k * L * V);
        }
    };
    auto storeRow = [&](T* base, int64_t row, Vec const (&v)[K])
    {
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            *reinterpret_cast<Vec*>(base + row * C + laneChunkOff + k * L * V) = v[k];
        }
    };
    auto zeroRow = [&](Vec (&out)[K])
    {
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            out[k] = common::make_zeros<Vec>();
        }
    };

    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * kPosPerBlock + warp * kPosPerWarp + group; pos < P;
         pos += gridPos)
    {
        // f0, f1: the two normalized frames before the current one (cache, then history).
        Vec f0[K], f1[K];
        if (p.cacheFrames >= 2)
        {
            loadRow(ptrs.cache, static_cast<int64_t>(p.cacheFrames - 2) * P + pos, f0);
        }
        else
        {
            zeroRow(f0);
        }
        if (p.cacheFrames >= 1)
        {
            loadRow(ptrs.cache, static_cast<int64_t>(p.cacheFrames - 1) * P + pos, f1);
        }
        else
        {
            zeroRow(f1);
        }

        for (int t0 = 0; t0 < T_; t0 += G)
        {
            int const nf = (T_ - t0) < G ? (T_ - t0) : G; // frames in this group (even, >= 2)

            Vec in[HOIST ? G : 1][NIN][K];
            Vec res[HAS_RESIDUAL ? (HOIST ? G : 1) : 1][K];
            auto loadFrame = [&](int g, int slot)
            {
                int const t = t0 + g;
                if constexpr (IN_WINO)
                {
                    int64_t const mrow = static_cast<int64_t>(t >> 1) * P + pos;
                    int const base = (t & 1) ? 1 : 0; // even frame: m0,m1,m2; odd frame: m1,m2,m3
                    loadRow(ptrs.m[base], mrow, in[slot][0]);
                    loadRow(ptrs.m[base + 1], mrow, in[slot][1]);
                    loadRow(ptrs.m[base + 2], mrow, in[slot][2]);
                }
                else
                {
                    loadRow(ptrs.x, static_cast<int64_t>(t) * P + pos, in[slot][0]);
                }
                if constexpr (HAS_RESIDUAL)
                {
                    loadRow(ptrs.residual, static_cast<int64_t>(t) * P + pos, res[slot]);
                }
            };
            if constexpr (HOIST)
            {
                // Issue every load of the group before touching any of it.
#pragma unroll
                for (int g = 0; g < G; ++g)
                {
                    if (g < nf)
                    {
                        loadFrame(g, g);
                    }
                }
            }

            // Normalize the group's frames; y[g] holds frame t0 + g.
            Vec y[G][K];
#pragma unroll
            for (int g = 0; g < G; ++g)
            {
                if (g >= nf)
                {
                    continue;
                }
                if constexpr (!HOIST)
                {
                    loadFrame(g, 0);
                }
                constexpr int S = 0; // register slot when not hoisting
                int const slot = HOIST ? g : S;
                int const t = t0 + g;
                float const sgn = (t & 1) ? -1.0f : 1.0f;
                VecF acc[K];
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const fa = pairAt(in[slot][0][k], i);
                        if constexpr (IN_WINO)
                        {
                            float2 const fb = pairAt(in[slot][1][k], i), fc = pairAt(in[slot][2][k], i);
                            acc[k].val[2 * i] = addR<T, REF>(addR<T, REF>(fa.x, sgn * fb.x), sgn * fc.x);
                            acc[k].val[2 * i + 1] = addR<T, REF>(addR<T, REF>(fa.y, sgn * fb.y), sgn * fc.y);
                        }
                        else
                        {
                            acc[k].val[2 * i] = fa.x;
                            acc[k].val[2 * i + 1] = fa.y;
                        }
                    }
                }
                if constexpr (HAS_IN_BIAS)
                {
#pragma unroll
                    for (int k = 0; k < K; ++k)
                    {
                        int const c0 = laneChunkOff + k * L * V;
#pragma unroll
                        for (int e = 0; e < V; ++e)
                        {
                            acc[k].val[e] += sInBias[c0 + e];
                        }
                    }
                }
                // Round to the activation dtype: the conv output as eager would hold it.
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        setPair(y[g][k], i, make_float2(acc[k].val[2 * i], acc[k].val[2 * i + 1]));
                    }
                }
                if constexpr (HAS_RESIDUAL)
                {
#pragma unroll
                    for (int k = 0; k < K; ++k)
                    {
#pragma unroll
                        for (int i = 0; i < V / 2; ++i)
                        {
                            float2 const a = pairAt(y[g][k], i), b = pairAt(res[slot][k], i);
                            setPair(y[g][k], i, make_float2(a.x + b.x, a.y + b.y));
                        }
                    }
                    storeRow(ptrs.sum, static_cast<int64_t>(t) * P + pos, y[g]);
                }
                float ss = 0.0f;
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const f = pairAt(y[g][k], i);
                        ss = fmaf(f.x, f.x, ss);
                        ss = fmaf(f.y, f.y, ss);
                    }
                }
#pragma unroll
                for (int off = L >> 1; off > 0; off >>= 1)
                {
                    ss += __shfl_xor_sync(0xffffffffu, ss, off);
                }
                float denom = 0.0f, invDenom = 0.0f;
                if constexpr (REF)
                {
                    denom = fmaxf(sqrtf(ss), p.eps);
                }
                else
                {
                    invDenom = rsqrtf(fmaxf(ss, p.eps * p.eps));
                }
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    int const c0 = laneChunkOff + k * L * V;
                    VecF const gm = *reinterpret_cast<VecF const*>(sGamma + c0);
                    VecF const bs = *reinterpret_cast<VecF const*>(sBias + c0);
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const f = pairAt(y[g][k], i);
                        float2 r;
                        r.x = normElement<T, REF, HAS_BIAS>(
                            f.x, denom, invDenom, p.scale, gm.val[2 * i], bs.val[2 * i]);
                        r.y = normElement<T, REF, HAS_BIAS>(
                            f.y, denom, invDenom, p.scale, gm.val[2 * i + 1], bs.val[2 * i + 1]);
                        setPair(y[g][k], i, r);
                    }
                }
            }

            // Winograd input transform for every pair completed in this group.
#pragma unroll
            for (int g = 0; g + 1 < G; g += 2)
            {
                if (g + 1 >= nf)
                {
                    continue;
                }
                Vec d0[K], d1[K], d2[K], d3[K];
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const a = pairAt(f0[k], i), b = pairAt(f1[k], i), c = pairAt(y[g][k], i),
                                     e = pairAt(y[g + 1][k], i);
                        setPair(d0[k], i, make_float2(a.x - c.x, a.y - c.y));
                        setPair(d1[k], i, make_float2(b.x + c.x, b.y + c.y));
                        setPair(d2[k], i, make_float2(c.x - b.x, c.y - b.y));
                        setPair(d3[k], i, make_float2(b.x - e.x, b.y - e.y));
                    }
                }
                int64_t const drow = static_cast<int64_t>((t0 + g) >> 1) * P + pos;
                storeRow(ptrs.d[0], drow, d0);
                storeRow(ptrs.d[1], drow, d1);
                storeRow(ptrs.d[2], drow, d2);
                storeRow(ptrs.d[3], drow, d3);
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    f0[k] = y[g][k];
                    f1[k] = y[g + 1][k];
                }
            }
        }
        storeRow(ptrs.yTail, pos, f0);
        storeRow(ptrs.yTail, P + pos, f1);
    }
}

template <typename T, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS, bool HAS_RESIDUAL, bool IN_WINO>
void launchConfig(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    auto* kernel = vaeWinoNormSiluKernel<T, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, IN_WINO>;
    constexpr int kPosPerBlock = (32 / L) * kWarps;
    int const smem = 3 * p.channels * static_cast<int>(sizeof(float));
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int64_t const steps = (p.positions + kPosPerBlock - 1) / kPosPerBlock;
    int64_t const persistent = static_cast<int64_t>(sms) * winoTuning<K, IN_WINO, HAS_RESIDUAL>().blocks;
    unsigned const blocks = static_cast<unsigned>(steps < persistent ? steps : persistent);
    kernel<<<blocks, kThreads, smem, stream>>>(ptrs, p);
}

template <typename T, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS, bool HAS_RESIDUAL>
void launchInWino(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    if (ptrs.x == nullptr)
        launchConfig<T, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, true>(ptrs, p, stream);
    else
        launchConfig<T, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, false>(ptrs, p, stream);
}

template <typename T, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS>
void launchResidual(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    if (ptrs.residual != nullptr)
        launchInWino<T, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, true>(ptrs, p, stream);
    else
        launchInWino<T, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, false>(ptrs, p, stream);
}

template <typename T, int V, int L, int K, bool REF, bool HAS_BIAS>
void launchInBias(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    if (ptrs.inBias != nullptr)
        launchResidual<T, V, L, K, REF, HAS_BIAS, true>(ptrs, p, stream);
    else
        launchResidual<T, V, L, K, REF, HAS_BIAS, false>(ptrs, p, stream);
}

template <typename T, int V, int L, int K>
void launchModes(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    bool const hasBias = ptrs.bias != nullptr;
    if (hasBias && p.referenceRounding)
        launchInBias<T, V, L, K, true, true>(ptrs, p, stream);
    else if (hasBias)
        launchInBias<T, V, L, K, false, true>(ptrs, p, stream);
    else if (p.referenceRounding)
        launchInBias<T, V, L, K, true, false>(ptrs, p, stream);
    else
        launchInBias<T, V, L, K, false, false>(ptrs, p, stream);
}

template <typename T, int V>
void launchLayout(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    RowLayout const l = makeRowLayout<V>(p.channels);
#define TRY_LAYOUT(LL, KK)                                                                                            \
    if (l.lanesPerRow == (LL) && l.vecsPerLane == (KK))                                                                \
    {                                                                                                                  \
        launchModes<T, V, LL, KK>(ptrs, p, stream);                                                                    \
        return;                                                                                                        \
    }
    VAE_NORM_FOR_EACH_LAYOUT(TRY_LAYOUT)
#undef TRY_LAYOUT
    TLLM_THROW("vaeWinoNormSilu: no lane layout for %d channels (lanes %d, vectors %d)", p.channels, l.lanesPerRow,
        l.vecsPerLane);
}

} // namespace

void launchVaeWinoNormSilu(VaeWinoNormPointers const& in, VaeWinoNormParams const& p, cudaStream_t stream)
{
    constexpr int kVecBf16 = static_cast<int>(sizeof(float4) / sizeof(__nv_bfloat16));
    TLLM_CHECK_WITH_INFO(p.channels > 0 && p.channels % kVecBf16 == 0 && p.channels <= kVaeRmsNormSiluMaxChannels,
        "vaeWinoNormSilu: channels must be a positive multiple of %d up to %d, got %d", kVecBf16,
        kVaeRmsNormSiluMaxChannels, p.channels);
    TLLM_CHECK_WITH_INFO(p.frames >= 2 && p.frames % 2 == 0, "vaeWinoNormSilu: frames must be even and >= 2");
    TLLM_CHECK_WITH_INFO(p.cacheFrames >= 0 && p.cacheFrames <= 2, "vaeWinoNormSilu: cacheFrames must be 0..2");
    TLLM_CHECK_WITH_INFO((in.x == nullptr) != (in.m[0] == nullptr), "vaeWinoNormSilu: exactly one of x and m");
    TLLM_CHECK_WITH_INFO((in.residual == nullptr) == (in.sum == nullptr), "vaeWinoNormSilu: residual and sum go together");
    if (p.positions == 0)
    {
        return;
    }
    using T = __nv_bfloat16;
    Ptrs<T> ptrs{};
    ptrs.x = static_cast<T const*>(in.x);
    for (int k = 0; k < 4; ++k)
    {
        ptrs.m[k] = static_cast<T const*>(in.m[k]);
        ptrs.d[k] = static_cast<T*>(in.d[k]);
    }
    ptrs.residual = static_cast<T const*>(in.residual);
    ptrs.inBias = static_cast<T const*>(in.inBias);
    ptrs.cache = static_cast<T const*>(in.cache);
    ptrs.gamma = static_cast<T const*>(in.gamma);
    ptrs.bias = static_cast<T const*>(in.bias);
    ptrs.sum = static_cast<T*>(in.sum);
    ptrs.yTail = static_cast<T*>(in.yTail);
    launchLayout<T, kVecBf16>(ptrs, p, stream);
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
