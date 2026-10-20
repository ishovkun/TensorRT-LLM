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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/packedVector.cuh"
#include "vaeFrameOpsKernel.h"
#include "vaeNormCommon.cuh"
#include "vaeWinoTables.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using common::PackedAligned;
using namespace vae_norm;
using namespace vae_wino;

constexpr int kThreads = 256;
using T = __nv_bfloat16;
constexpr int V = static_cast<int>(sizeof(float4) / sizeof(T));
using Vec = PackedAligned<T, V>;

__device__ __forceinline__ float2 roundPair(float2 v)
{
    return Elem<T>::toFloat2(Elem<T>::fromFloat2(v));
}

template <int I, typename TD>
__device__ __forceinline__ TD const* pick(
    TD const* m0, TD const* m1, TD const* m2, TD const* m3, TD const* m4, TD const* m5)
{
    if constexpr (I == 0)
        return m0;
    else if constexpr (I == 1)
        return m1;
    else if constexpr (I == 2)
        return m2;
    else if constexpr (I == 3)
        return m3;
    else if constexpr (I == 4)
        return m4;
    else
        return m5;
}

// Output frame j of a group: sum_k AT[j][k] m_k in index order, rounded to bf16 after every
// add (the products are exact: every AT entry is a power of two).
template <typename TD, int R, int J>
__device__ __forceinline__ Vec combineRow(
    TD const* m0, TD const* m1, TD const* m2, TD const* m3, TD const* m4, TD const* m5, int64_t mIdx)
{
    using W = Wino<R>;
    using VecD = PackedAligned<TD, V>;
    constexpr int n = numTerms<R>(J);
    VecD in[n];
    in[0] = *reinterpret_cast<VecD const*>(pick<termIndex<R>(J, 0)>(m0, m1, m2, m3, m4, m5) + mIdx * V);
    if constexpr (n > 1)
        in[1] = *reinterpret_cast<VecD const*>(pick<termIndex<R>(J, 1)>(m0, m1, m2, m3, m4, m5) + mIdx * V);
    if constexpr (n > 2)
        in[2] = *reinterpret_cast<VecD const*>(pick<termIndex<R>(J, 2)>(m0, m1, m2, m3, m4, m5) + mIdx * V);
    if constexpr (n > 3)
        in[3] = *reinterpret_cast<VecD const*>(pick<termIndex<R>(J, 3)>(m0, m1, m2, m3, m4, m5) + mIdx * V);
    if constexpr (n > 4)
        in[4] = *reinterpret_cast<VecD const*>(pick<termIndex<R>(J, 4)>(m0, m1, m2, m3, m4, m5) + mIdx * V);
    Vec out;
    constexpr float c0 = W::at(J, termIndex<R>(J, 0));
#pragma unroll
    for (int i = 0; i < V / 2; ++i)
    {
        float2 const a = pairAt(in[0], i);
        float2 v = make_float2(c0 * a.x, c0 * a.y);
#pragma unroll
        for (int q = 1; q < n; ++q)
        {
            float const cq = W::at(J, termIndex<R>(J, q)); // constant after unrolling
            float2 const mq = pairAt(in[q], i);
            v = roundPair(make_float2(v.x + cq * mq.x, v.y + cq * mq.y));
        }
        setPair(out, i, v);
    }
    return out;
}

// One thread per 16-byte output vector; grid-stride. With IN_WINO the frames are the
// temporal Winograd combine of the order + 2 products (TD: their dtype).
template <typename TD, int R, bool IN_WINO, bool HAS_BIAS, bool HAS_RESIDUAL>
__global__ void __launch_bounds__(kThreads)
    vaeWinoCombineKernel(T const* __restrict__ x, TD const* __restrict__ m0, TD const* __restrict__ m1,
        TD const* __restrict__ m2, TD const* __restrict__ m3, TD const* __restrict__ m4, TD const* __restrict__ m5,
        T const* __restrict__ bias, T const* __restrict__ residual, T* __restrict__ y, VaeWinoCombineParams p)
{
    int const C = p.channels;
    int const cvec = C / V;
    int64_t const rowVecs = p.positions * cvec; // source vectors per frame
    int const outW = p.width + p.padW;
    int64_t const outPositions = static_cast<int64_t>(p.height + p.padH) * outW;
    int64_t const outRowVecs = outPositions * cvec;
    int64_t const total = static_cast<int64_t>(p.frames) * outRowVecs;
    for (int64_t oidx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; oidx < total;
         oidx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const t = static_cast<int>(oidx / outRowVecs);
        int64_t const outInFrame = oidx - static_cast<int64_t>(t) * outRowVecs;
        int64_t const opos = outInFrame / cvec;
        int const c0 = static_cast<int>(outInFrame - opos * cvec) * V;
        int const oi = static_cast<int>(opos / outW), oj = static_cast<int>(opos - static_cast<int64_t>(oi) * outW);
        if (oi >= p.height || oj >= p.width)
        {
            *reinterpret_cast<Vec*>(y + oidx * V) = common::make_zeros<Vec>(); // the zero pad
            continue;
        }
        int64_t const inFrame = (static_cast<int64_t>(oi) * p.width + oj) * cvec + c0 / V; // source vector in frame
        int64_t const idx = static_cast<int64_t>(t) * rowVecs + inFrame;                   // source vector overall
        Vec out;
        if constexpr (IN_WINO)
        {
            int64_t const mIdx = static_cast<int64_t>(t / R) * rowVecs + inFrame;
            switch (t % R)
            {
            case 0: out = combineRow<TD, R, 0>(m0, m1, m2, m3, m4, m5, mIdx); break;
            case 1: out = combineRow<TD, R, 1>(m0, m1, m2, m3, m4, m5, mIdx); break;
            default:
                if constexpr (R == 4)
                {
                    if (t % R == 2)
                        out = combineRow<TD, R, 2>(m0, m1, m2, m3, m4, m5, mIdx);
                    else
                        out = combineRow<TD, R, 3>(m0, m1, m2, m3, m4, m5, mIdx);
                }
                else
                {
                    out = combineRow<TD, R, 1>(m0, m1, m2, m3, m4, m5, mIdx);
                }
                break;
            }
        }
        else
        {
            out = *reinterpret_cast<Vec const*>(x + idx * V);
        }
        if constexpr (HAS_BIAS)
        {
            Vec const bv = *reinterpret_cast<Vec const*>(bias + c0);
#pragma unroll
            for (int i = 0; i < V / 2; ++i)
            {
                float2 const o = pairAt(out, i), b = pairAt(bv, i);
                setPair(out, i, make_float2(o.x + b.x, o.y + b.y));
            }
        }
        if constexpr (HAS_RESIDUAL)
        {
            Vec const rv = *reinterpret_cast<Vec const*>(residual + idx * V);
#pragma unroll
            for (int i = 0; i < V / 2; ++i)
            {
                float2 const o = pairAt(out, i), r = pairAt(rv, i);
                setPair(out, i, make_float2(o.x + r.x, o.y + r.y));
            }
        }
        *reinterpret_cast<Vec*>(y + oidx * V) = out;
    }
}

// One thread per (output pixel, 8 output channels): y = x + mean over the group.
__global__ void __launch_bounds__(kThreads) vaeAvgDownAddKernel(
    T const* __restrict__ x, T const* __restrict__ r, T const* __restrict__ bias, T* __restrict__ y, VaeAvgDownParams p)
{
    int const cvec = p.cout / V;
    int64_t const total = static_cast<int64_t>(p.outFrames) * p.outPositions * cvec;
    int const factor = p.factorT * p.factorS * p.factorS;
    float const inv = 1.0f / static_cast<float>(p.groupSize);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const o0 = static_cast<int>(idx % cvec) * V;
        int64_t const pix = idx / cvec; // (t', i', j') flattened
        int const tOut = static_cast<int>(pix / p.outPositions);
        int64_t const sp = pix - static_cast<int64_t>(tOut) * p.outPositions;
        int const iOut = static_cast<int>(sp / p.outWidth);
        int const jOut = static_cast<int>(sp - static_cast<int64_t>(iOut) * p.outWidth);
        Vec xv = *reinterpret_cast<Vec const*>(x + idx * V);
        Vec const bv = bias != nullptr ? *reinterpret_cast<Vec const*>(bias + o0) : common::make_zeros<Vec>();
        Vec out;
#pragma unroll
        for (int k = 0; k < V; ++k)
        {
            float acc = 0.0f;
            for (int g = 0; g < p.groupSize; ++g)
            {
                int const f = (o0 + k) * p.groupSize + g;
                int const c = f / factor;
                int const rem = f - c * factor;
                int const a = rem / (p.factorS * p.factorS);
                int const rem2 = rem - a * p.factorS * p.factorS;
                int const b = rem2 / p.factorS;
                int const d = rem2 - b * p.factorS;
                int const tIn = tOut * p.factorT + a - p.padT;
                if (tIn >= 0)
                {
                    int64_t const row
                        = (static_cast<int64_t>(tIn) * (p.outHeight * p.factorS) + iOut * p.factorS + b) * p.inWidth
                        + jOut * p.factorS + d;
                    acc += Elem<T>::toFloat(r[row * p.cin + c]);
                }
            }
            // mean in fp32 (PyTorch's bf16 mean accumulates in fp32), then the eager add
            float const m = roundPair(make_float2(acc * inv, 0.0f)).x;
            float const xb = Elem<T>::toFloat(xv.val[k]) + Elem<T>::toFloat(bv.val[k]);
            out.val[k] = Elem<T>::fromFloat2(make_float2(xb + m, 0.0f)).x;
        }
        *reinterpret_cast<Vec*>(y + idx * V) = out;
    }
}

__global__ void __launch_bounds__(kThreads) vaeInterleaveBiasKernel(
    T const* __restrict__ u, T const* __restrict__ bias, T* __restrict__ y, VaeInterleaveParams p)
{
    int const C = p.channels;
    int const cvec = C / V;
    int64_t const rowVecs = p.positions * cvec; // output vectors per frame
    int64_t const total = static_cast<int64_t>(2 * p.frames) * rowVecs;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const tOut = static_cast<int>(idx / rowVecs);
        int64_t const inFrame = idx - static_cast<int64_t>(tOut) * rowVecs;
        int64_t const pos = inFrame / cvec;
        int const c0 = static_cast<int>(inFrame - pos * cvec) * V;
        int const half = tOut & 1;
        int const tIn = tOut >> 1;
        // u row (tIn, pos) has 2C channels; this output takes channels [half*C, half*C + C).
        int64_t const uOff = (static_cast<int64_t>(tIn) * p.positions + pos) * (2 * C) + half * C + c0;
        Vec v = *reinterpret_cast<Vec const*>(u + uOff);
        if (bias != nullptr)
        {
            Vec const bv = *reinterpret_cast<Vec const*>(bias + half * C + c0);
#pragma unroll
            for (int i = 0; i < V / 2; ++i)
            {
                float2 const o = pairAt(v, i), b = pairAt(bv, i);
                setPair(v, i, make_float2(o.x + b.x, o.y + b.y));
            }
        }
        *reinterpret_cast<Vec*>(y + idx * V) = v;
    }
}

// One thread per chunk pixel: its C*patch*patch channels scatter to patch x patch output
// pixels of C channels each (6-byte pixels for RGB; scalar stores).
__global__ void __launch_bounds__(kThreads)
    vaeUnpatchifyClampKernel(T const* __restrict__ chunk, T* __restrict__ out, VaeUnpatchifyParams p)
{
    int const P = p.patch, C = p.channels;
    int const cpp = C * P * P;
    int64_t const pixels = static_cast<int64_t>(p.frames) * p.height * p.width;
    int const outW = p.width * P, outH = p.height * P;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < pixels;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const w = static_cast<int>(idx % p.width);
        int64_t const th = idx / p.width;
        int const h = static_cast<int>(th % p.height);
        int const f = static_cast<int>(th / p.height);
        T const* src = chunk + idx * cpp;
        int64_t const rowBase = (static_cast<int64_t>(p.frame0 + f) * outH + static_cast<int64_t>(h) * P) * outW;
        for (int j = 0; j < P; ++j)
        {
            for (int i = 0; i < P; ++i)
            {
                T* dst = out + (rowBase + static_cast<int64_t>(j) * outW + static_cast<int64_t>(w) * P + i) * C;
                for (int c = 0; c < C; ++c)
                {
                    float const v = Elem<T>::toFloat(src[(c * P + i) * P + j]);
                    dst[c] = Elem<T>::fromFloat2(make_float2(fminf(fmaxf(v, -1.0f), 1.0f), 0.0f)).x;
                }
            }
        }
    }
}

unsigned gridFor(int64_t total)
{
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int64_t const blocks = (total + kThreads - 1) / kThreads;
    int64_t const cap = static_cast<int64_t>(sms) * 8;
    return static_cast<unsigned>(blocks < cap ? blocks : cap);
}

template <typename TD, int R, bool IN_WINO, bool HAS_BIAS>
void launchCombineRes(
    VaeWinoCombinePointers const& in, VaeWinoCombineParams const& p, unsigned blocks, cudaStream_t stream)
{
    auto const* x = static_cast<T const*>(in.x);
    TD const* m[6];
    for (int k = 0; k < 6; ++k)
    {
        m[k] = static_cast<TD const*>(in.m[k]);
    }
    auto const* b = static_cast<T const*>(in.bias);
    auto const* r = static_cast<T const*>(in.residual);
    auto* y = static_cast<T*>(in.y);
    if (in.residual != nullptr)
        vaeWinoCombineKernel<TD, R, IN_WINO, HAS_BIAS, true>
            <<<blocks, kThreads, 0, stream>>>(x, m[0], m[1], m[2], m[3], m[4], m[5], b, r, y, p);
    else
        vaeWinoCombineKernel<TD, R, IN_WINO, HAS_BIAS, false>
            <<<blocks, kThreads, 0, stream>>>(x, m[0], m[1], m[2], m[3], m[4], m[5], b, r, y, p);
}

template <typename TD, int R, bool IN_WINO>
void launchCombineBias(
    VaeWinoCombinePointers const& in, VaeWinoCombineParams const& p, unsigned blocks, cudaStream_t stream)
{
    if (in.bias != nullptr)
        launchCombineRes<TD, R, IN_WINO, true>(in, p, blocks, stream);
    else
        launchCombineRes<TD, R, IN_WINO, false>(in, p, blocks, stream);
}

} // namespace

void launchVaeWinoCombine(VaeWinoCombinePointers const& in, VaeWinoCombineParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.channels > 0 && p.channels % V == 0, "vaeWinoCombine: channels must be a multiple of %d", V);
    TLLM_CHECK_WITH_INFO((in.x == nullptr) != (in.m[0] == nullptr), "vaeWinoCombine: exactly one of x and m");
    bool const wino = in.x == nullptr;
    TLLM_CHECK_WITH_INFO(!wino || p.order == 2 || p.order == 4, "vaeWinoCombine: order must be 2 or 4");
    TLLM_CHECK_WITH_INFO(
        !wino || p.frames % p.order == 0, "vaeWinoCombine: Winograd input needs a frame count divisible by the order");
    TLLM_CHECK_WITH_INFO(
        static_cast<int64_t>(p.height) * p.width == p.positions, "vaeWinoCombine: height*width must equal positions");
    int64_t const total = static_cast<int64_t>(p.frames) * (p.height + p.padH) * (p.width + p.padW) * (p.channels / V);
    if (total == 0)
    {
        return;
    }
    unsigned const blocks = gridFor(total);
    if (!wino)
        launchCombineBias<T, 2, false>(in, p, blocks, stream);
    else if (p.order == 4)
        launchCombineBias<__half, 4, true>(in, p, blocks, stream); // order 4 products are fp16
    else if (p.halfProducts)
        launchCombineBias<__half, 2, true>(in, p, blocks, stream);
    else
        launchCombineBias<T, 2, true>(in, p, blocks, stream);
}

void launchVaeAvgDownAdd(
    void const* x, void const* r, void const* bias, void* y, VaeAvgDownParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.cout % V == 0, "vaeAvgDownAdd: cout must be a multiple of %d", V);
    TLLM_CHECK_WITH_INFO(p.groupSize >= 1 && p.factorT >= 1 && p.factorS >= 1, "vaeAvgDownAdd: bad factors");
    int64_t const total = static_cast<int64_t>(p.outFrames) * p.outPositions * (p.cout / V);
    if (total == 0)
    {
        return;
    }
    vaeAvgDownAddKernel<<<gridFor(total), kThreads, 0, stream>>>(
        static_cast<T const*>(x), static_cast<T const*>(r), static_cast<T const*>(bias), static_cast<T*>(y), p);
    sync_check_cuda_error(stream);
}

void launchVaeInterleaveBias(
    void const* u, void const* bias, void* y, VaeInterleaveParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        p.channels > 0 && p.channels % V == 0, "vaeInterleaveBias: channels must be a multiple of %d", V);
    int64_t const total = static_cast<int64_t>(2 * p.frames) * p.positions * (p.channels / V);
    if (total == 0)
    {
        return;
    }
    vaeInterleaveBiasKernel<<<gridFor(total), kThreads, 0, stream>>>(
        static_cast<T const*>(u), static_cast<T const*>(bias), static_cast<T*>(y), p);
    sync_check_cuda_error(stream);
}

void launchVaeUnpatchifyClamp(void const* chunk, void* out, VaeUnpatchifyParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.patch >= 1 && p.channels >= 1, "vaeUnpatchifyClamp: bad patch or channels");
    TLLM_CHECK_WITH_INFO(
        p.frame0 >= 0 && p.frame0 + p.frames <= p.outFrames, "vaeUnpatchifyClamp: frames out of range");
    int64_t const pixels = static_cast<int64_t>(p.frames) * p.height * p.width;
    vaeUnpatchifyClampKernel<<<gridFor(pixels), kThreads, 0, stream>>>(
        static_cast<T const*>(chunk), static_cast<T*>(out), p);
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
