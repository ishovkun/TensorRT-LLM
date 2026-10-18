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
#include "vaeNormCommon.cuh"
#include "vaeWinoSpatialKernel.h"
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
using T = __nv_bfloat16; // frames
using TS = __half;       // transformed data (V, M): fp16 keeps three more mantissa bits than bf16

// Channels per thread: 8 (16-byte vectors) for m = 2; 2 (one bf16x2 / fp16x2 pair) for m = 4,
// whose 6x6 window would otherwise not fit in registers (measured: 4 channels per thread
// costs 175 registers and is no faster).
template <int M>
struct Lanes
{
    static constexpr int V = M == 2 ? 8 : 2;
    using Vec = PackedAligned<T, V>;
    using VecS = PackedAligned<TS, V>;
};

struct Geometry
{
    int tw;                // tiles per row (ceil(W / m))
    int stripRows;         // row1 - row0
    int64_t tilesPerFrame; // stripRows * tw
    int64_t tilesPerComp;  // pairs * tilesPerFrame
    int pairs;
};

template <int M>
__device__ __forceinline__ Geometry geometry(VaeWinoSpatialParams const& p)
{
    Geometry g;
    g.tw = (p.width + M - 1) / M;
    g.stripRows = p.row1 - p.row0;
    g.tilesPerFrame = static_cast<int64_t>(g.stripRows) * g.tw;
    g.pairs = p.frames / p.components;
    g.tilesPerComp = g.pairs * g.tilesPerFrame;
    return g;
}

// Element offset of V/M[(comp, coord)][pair, tile][c0]: layout [comp * coords, pairs * tiles, C].
template <int COORDS>
__device__ __forceinline__ int64_t vmOffset(
    Geometry const& g, int comp, int coord, int pair, int64_t tile, int c0, int C)
{
    int64_t const row = (static_cast<int64_t>(comp) * COORDS + coord) * g.tilesPerComp + pair * g.tilesPerFrame + tile;
    return row * C + c0;
}

__device__ __forceinline__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 operator*(float k, float2 a)
{
    return make_float2(k * a.x, k * a.y);
}

// Lavin & Gray transforms, written out (interpolation points 0, +-1, +-2 for m = 4). Integer
// coefficients: with bf16 / fp16 inputs every product and sum is exact in fp32 except the
// final rounding, so the input transform is bitwise reproducible.
// bt<M>(x, d): row x of B^T applied to the (m+2) values d; at<M>(r, t): row r of A^T.
template <int M>
struct WinoMath;

template <>
struct WinoMath<2>
{
    static constexpr int A = 4;

    static __device__ __forceinline__ float2 bt(int x, float2 const (&d)[4])
    {
        switch (x)
        {
        case 0: return d[0] - d[2];
        case 1: return d[1] + d[2];
        case 2: return d[2] - d[1];
        default: return d[1] - d[3];
        }
    }

    static __device__ __forceinline__ float2 at(int r, float2 const (&t)[4])
    {
        return r == 0 ? t[0] + t[1] + t[2] : t[1] - t[2] - t[3];
    }
};

template <>
struct WinoMath<4>
{
    static constexpr int A = 6;

    static __device__ __forceinline__ float2 bt(int x, float2 const (&d)[6])
    {
        switch (x)
        {
        case 0: return 4.f * d[0] - 5.f * d[2] + d[4];
        case 1: return -4.f * (d[1] + d[2]) + d[3] + d[4];
        case 2: return 4.f * (d[1] - d[2]) - d[3] + d[4];
        case 3: return -2.f * d[1] - d[2] + 2.f * d[3] + d[4];
        case 4: return 2.f * d[1] - d[2] - 2.f * d[3] + d[4];
        default: return 4.f * d[1] - 5.f * d[3] + d[5];
        }
    }

    static __device__ __forceinline__ float2 at(int r, float2 const (&t)[6])
    {
        switch (r)
        {
        case 0: return t[0] + t[1] + t[2] + t[3] + t[4];
        case 1: return t[1] - t[2] + 2.f * (t[3] - t[4]);
        case 2: return t[1] + t[2] + 4.f * (t[3] + t[4]);
        default: return t[1] - t[2] + 8.f * (t[3] - t[4]) + t[5];
        }
    }
};

// V = B^T d B for one (m+2) x (m+2) window and V channels, written as (m+2)^2 vectors; one
// coordinate row at a time so the register footprint stays bounded.
template <int M>
__global__ void __launch_bounds__(kThreads)
    vaeWinoSpatialInputKernel(T const* __restrict__ frames, TS* __restrict__ v, VaeWinoSpatialParams p)
{
    using W = WinoMath<M>;
    constexpr int A = W::A;
    constexpr int V = Lanes<M>::V;
    using Vec = typename Lanes<M>::Vec;
    using VecS = typename Lanes<M>::VecS;
    Geometry const g = geometry<M>(p);
    int const C = p.channels;
    int const cvec = C / V;
    int64_t const total = static_cast<int64_t>(p.frames) * g.tilesPerFrame * cvec;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const c0 = static_cast<int>(idx % cvec) * V;
        int64_t const t = idx / cvec;
        int const n = static_cast<int>(t / g.tilesPerFrame);
        int64_t const tile = t - static_cast<int64_t>(n) * g.tilesPerFrame;
        int const i = p.row0 + static_cast<int>(tile / g.tw);
        int const j = static_cast<int>(tile - static_cast<int64_t>(i - p.row0) * g.tw);
        int const comp = n / g.pairs, pair = n - comp * g.pairs;

        Vec d[A][A];
        T const* frame0 = frames + (static_cast<int64_t>(n) * p.height * p.width) * C + c0;
#pragma unroll
        for (int a = 0; a < A; ++a)
        {
            int const row = M * i - 1 + a;
            T const* rowp = frame0 + (static_cast<int64_t>(row) * p.width + (M * j - 1)) * C;
#pragma unroll
            for (int b = 0; b < A; ++b)
            {
                int const col = M * j - 1 + b;
                if (row >= 0 && row < p.height && col >= 0 && col < p.width)
                {
                    d[a][b] = *reinterpret_cast<Vec const*>(rowp + b * C);
                }
                else
                {
                    d[a][b] = common::make_zeros<Vec>();
                }
            }
        }
        TS* vp = v + vmOffset<A * A>(g, comp, 0, pair, tile, c0, C);
        int64_t const coordStride = g.tilesPerComp * C;
#pragma unroll
        for (int x = 0; x < A; ++x)
        {
            VecS out[A];
#pragma unroll
            for (int e = 0; e < V / 2; ++e)
            {
                float2 tb[A];
#pragma unroll
                for (int b = 0; b < A; ++b)
                {
                    float2 col[A];
#pragma unroll
                    for (int a = 0; a < A; ++a)
                    {
                        col[a] = pairAt(d[a][b], e);
                    }
                    tb[b] = W::bt(x, col);
                }
#pragma unroll
                for (int y = 0; y < A; ++y)
                {
                    setPair(out[y], e, W::bt(y, tb));
                }
            }
#pragma unroll
            for (int y = 0; y < A; ++y)
            {
                *reinterpret_cast<VecS*>(vp) = out[y];
                vp += coordStride;
            }
        }
    }
}

// y = A^T M A for one tile and V channels: (m+2)^2 vectors in, m x m output pixels out.
template <int M>
__global__ void __launch_bounds__(kThreads)
    vaeWinoSpatialOutputKernel(TS const* __restrict__ m, T* __restrict__ frames, VaeWinoSpatialParams p)
{
    using W = WinoMath<M>;
    constexpr int A = W::A;
    constexpr int V = Lanes<M>::V;
    using Vec = typename Lanes<M>::Vec;
    using VecS = typename Lanes<M>::VecS;
    Geometry const g = geometry<M>(p);
    int const C = p.channels;
    int const cvec = C / V;
    int64_t const total = static_cast<int64_t>(p.frames) * g.tilesPerFrame * cvec;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const c0 = static_cast<int>(idx % cvec) * V;
        int64_t const t = idx / cvec;
        int const n = static_cast<int>(t / g.tilesPerFrame);
        int64_t const tile = t - static_cast<int64_t>(n) * g.tilesPerFrame;
        int const i = p.row0 + static_cast<int>(tile / g.tw);
        int const j = static_cast<int>(tile - static_cast<int64_t>(i - p.row0) * g.tw);
        int const comp = n / g.pairs, pair = n - comp * g.pairs;

        VecS mv[A * A];
        TS const* mp = m + vmOffset<A * A>(g, comp, 0, pair, tile, c0, C);
        int64_t const coordStride = g.tilesPerComp * C;
#pragma unroll
        for (int k = 0; k < A * A; ++k)
        {
            mv[k] = *reinterpret_cast<VecS const*>(mp);
            mp += coordStride;
        }
        T* frame0 = frames + (static_cast<int64_t>(n) * p.height * p.width) * C + c0;
#pragma unroll
        for (int r = 0; r < M; ++r)
        {
            Vec out[M];
#pragma unroll
            for (int e = 0; e < V / 2; ++e)
            {
                float2 tb[A];
#pragma unroll
                for (int b = 0; b < A; ++b)
                {
                    float2 col[A];
#pragma unroll
                    for (int x = 0; x < A; ++x)
                    {
                        col[x] = pairAt(mv[x * A + b], e);
                    }
                    tb[b] = W::at(r, col);
                }
#pragma unroll
                for (int sidx = 0; sidx < M; ++sidx)
                {
                    setPair(out[sidx], e, W::at(sidx, tb));
                }
            }
            T* rowp = frame0 + (static_cast<int64_t>(M * i + r) * p.width + M * j) * C;
#pragma unroll
            for (int sidx = 0; sidx < M; ++sidx)
            {
                if (M * i + r < p.height && M * j + sidx < p.width)
                {
                    *reinterpret_cast<Vec*>(rowp + sidx * C) = out[sidx];
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

void check(VaeWinoSpatialParams const& p)
{
    TLLM_CHECK_WITH_INFO(p.tile == 2 || p.tile == 4, "vaeWinoSpatial: tile must be 2 or 4");
    TLLM_CHECK_WITH_INFO(p.channels % 8 == 0, "vaeWinoSpatial: channels must be a multiple of 8");
    TLLM_CHECK_WITH_INFO(
        p.components >= 1 && p.frames % p.components == 0, "vaeWinoSpatial: frames must be a multiple of components");
    TLLM_CHECK_WITH_INFO(0 <= p.row0 && p.row0 < p.row1 && p.row1 <= (p.height + p.tile - 1) / p.tile,
        "vaeWinoSpatial: bad tile-row range");
}

int64_t stripVectors(VaeWinoSpatialParams const& p)
{
    int const lanes = p.tile == 2 ? Lanes<2>::V : Lanes<4>::V;
    return static_cast<int64_t>(p.frames) * (p.row1 - p.row0) * ((p.width + p.tile - 1) / p.tile)
        * (p.channels / lanes);
}

} // namespace

void launchVaeWinoSpatialInput(void const* frames, void* v, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    unsigned const grid = gridFor(stripVectors(p));
    if (p.tile == 2)
    {
        vaeWinoSpatialInputKernel<2>
            <<<grid, kThreads, 0, stream>>>(static_cast<T const*>(frames), static_cast<TS*>(v), p);
    }
    else
    {
        vaeWinoSpatialInputKernel<4>
            <<<grid, kThreads, 0, stream>>>(static_cast<T const*>(frames), static_cast<TS*>(v), p);
    }
    sync_check_cuda_error(stream);
}

void launchVaeWinoSpatialOutput(void const* m, void* frames, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    unsigned const grid = gridFor(stripVectors(p));
    if (p.tile == 2)
    {
        vaeWinoSpatialOutputKernel<2>
            <<<grid, kThreads, 0, stream>>>(static_cast<TS const*>(m), static_cast<T*>(frames), p);
    }
    else
    {
        vaeWinoSpatialOutputKernel<4>
            <<<grid, kThreads, 0, stream>>>(static_cast<TS const*>(m), static_cast<T*>(frames), p);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
