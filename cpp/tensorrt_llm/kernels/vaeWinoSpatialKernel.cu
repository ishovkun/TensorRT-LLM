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
using T = __nv_bfloat16;
constexpr int V = static_cast<int>(sizeof(float4) / sizeof(T));
using Vec = PackedAligned<T, V>;

struct Geometry
{
    int tw;                // tiles per row (W / 2)
    int stripRows;         // row1 - row0
    int64_t tilesPerFrame; // stripRows * tw
    int64_t tilesPerComp;  // pairs * tilesPerFrame
    int pairs;
};

__device__ __forceinline__ Geometry geometry(VaeWinoSpatialParams const& p)
{
    Geometry g;
    g.tw = (p.width + 1) / 2;
    g.stripRows = p.row1 - p.row0;
    g.tilesPerFrame = static_cast<int64_t>(g.stripRows) * g.tw;
    g.pairs = p.frames / p.components;
    g.tilesPerComp = g.pairs * g.tilesPerFrame;
    return g;
}

// Element offset of V/M[(comp, coord)][pair, tile][c0]: layout [comp*16, pairs*tiles, C].
__device__ __forceinline__ int64_t vmOffset(
    Geometry const& g, int comp, int coord, int pair, int64_t tile, int c0, int C)
{
    int64_t const row = (static_cast<int64_t>(comp) * 16 + coord) * g.tilesPerComp + pair * g.tilesPerFrame + tile;
    return row * C + c0;
}

// V = B^T d B for one 4x4 window and 8 channels, written as 16 vectors.
__global__ void __launch_bounds__(kThreads)
    vaeWinoSpatialInputKernel(T const* __restrict__ frames, T* __restrict__ v, VaeWinoSpatialParams p)
{
    Geometry const g = geometry(p);
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

        // Load the 4x4 window (zero outside the frame), 8 channels per element, kept as bf16.
        Vec d[4][4];
#pragma unroll
        for (int a = 0; a < 4; ++a)
        {
            int const row = 2 * i - 1 + a;
#pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                int const col = 2 * j - 1 + b;
                if (row >= 0 && row < p.height && col >= 0 && col < p.width)
                {
                    int64_t const off = ((static_cast<int64_t>(n) * p.height + row) * p.width + col) * C + c0;
                    d[a][b] = *reinterpret_cast<Vec const*>(frames + off);
                }
                else
                {
                    d[a][b] = common::make_zeros<Vec>();
                }
            }
        }
        // Rows: t[a] = BT[a] . d  (t0 = d0 - d2, t1 = d1 + d2, t2 = d2 - d1, t3 = d1 - d3), then the
        // same combination along columns; one 16-byte vector per coordinate.
        Vec out[16];
#pragma unroll
        for (int e = 0; e < V / 2; ++e)
        {
            float2 tmat[4][4];
#pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                float2 const d0 = pairAt(d[0][b], e), d1 = pairAt(d[1][b], e), d2 = pairAt(d[2][b], e),
                             d3 = pairAt(d[3][b], e);
                tmat[0][b] = make_float2(d0.x - d2.x, d0.y - d2.y);
                tmat[1][b] = make_float2(d1.x + d2.x, d1.y + d2.y);
                tmat[2][b] = make_float2(d2.x - d1.x, d2.y - d1.y);
                tmat[3][b] = make_float2(d1.x - d3.x, d1.y - d3.y);
            }
#pragma unroll
            for (int a = 0; a < 4; ++a)
            {
                float2 const t0 = tmat[a][0], t1 = tmat[a][1], t2 = tmat[a][2], t3 = tmat[a][3];
                setPair(out[a * 4 + 0], e, make_float2(t0.x - t2.x, t0.y - t2.y));
                setPair(out[a * 4 + 1], e, make_float2(t1.x + t2.x, t1.y + t2.y));
                setPair(out[a * 4 + 2], e, make_float2(t2.x - t1.x, t2.y - t1.y));
                setPair(out[a * 4 + 3], e, make_float2(t1.x - t3.x, t1.y - t3.y));
            }
        }
#pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            *reinterpret_cast<Vec*>(v + vmOffset(g, comp, k, pair, tile, c0, C)) = out[k];
        }
    }
}

// y = A^T M A for one tile and 8 channels: 16 vectors in, 4 output pixels out.
__global__ void __launch_bounds__(kThreads)
    vaeWinoSpatialOutputKernel(T const* __restrict__ m, T* __restrict__ frames, VaeWinoSpatialParams p)
{
    Geometry const g = geometry(p);
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

        Vec mv[16];
#pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            mv[k] = *reinterpret_cast<Vec const*>(m + vmOffset(g, comp, k, pair, tile, c0, C));
        }
        Vec out[2][2];
#pragma unroll
        for (int e = 0; e < V / 2; ++e)
        {
            // Rows: t[r] = AT[r] . m  (t0 = m0 + m1 + m2, t1 = m1 - m2 - m3), then along columns.
            float2 tmat[2][4];
#pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                float2 const m0 = pairAt(mv[0 * 4 + b], e), m1 = pairAt(mv[1 * 4 + b], e),
                             m2 = pairAt(mv[2 * 4 + b], e), m3 = pairAt(mv[3 * 4 + b], e);
                tmat[0][b] = make_float2(m0.x + m1.x + m2.x, m0.y + m1.y + m2.y);
                tmat[1][b] = make_float2(m1.x - m2.x - m3.x, m1.y - m2.y - m3.y);
            }
#pragma unroll
            for (int r = 0; r < 2; ++r)
            {
                float2 const t0 = tmat[r][0], t1 = tmat[r][1], t2 = tmat[r][2], t3 = tmat[r][3];
                setPair(out[r][0], e, make_float2(t0.x + t1.x + t2.x, t0.y + t1.y + t2.y));
                setPair(out[r][1], e, make_float2(t1.x - t2.x - t3.x, t1.y - t2.y - t3.y));
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
#pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                if (2 * i + r < p.height && 2 * j + c < p.width)
                {
                    int64_t const off
                        = ((static_cast<int64_t>(n) * p.height + 2 * i + r) * p.width + 2 * j + c) * C + c0;
                    *reinterpret_cast<Vec*>(frames + off) = out[r][c];
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
    TLLM_CHECK_WITH_INFO(p.channels % V == 0, "vaeWinoSpatial: channels must be a multiple of %d", V);
    TLLM_CHECK_WITH_INFO(
        p.components >= 1 && p.frames % p.components == 0, "vaeWinoSpatial: frames must be a multiple of components");
    TLLM_CHECK_WITH_INFO(
        0 <= p.row0 && p.row0 < p.row1 && p.row1 <= (p.height + 1) / 2, "vaeWinoSpatial: bad tile-row range");
}

} // namespace

void launchVaeWinoSpatialInput(void const* frames, void* v, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    int64_t const total = static_cast<int64_t>(p.frames) * (p.row1 - p.row0) * ((p.width + 1) / 2) * (p.channels / V);
    vaeWinoSpatialInputKernel<<<gridFor(total), kThreads, 0, stream>>>(
        static_cast<T const*>(frames), static_cast<T*>(v), p);
    sync_check_cuda_error(stream);
}

void launchVaeWinoSpatialOutput(void const* m, void* frames, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    int64_t const total = static_cast<int64_t>(p.frames) * (p.row1 - p.row0) * ((p.width + 1) / 2) * (p.channels / V);
    vaeWinoSpatialOutputKernel<<<gridFor(total), kThreads, 0, stream>>>(
        static_cast<T const*>(m), static_cast<T*>(frames), p);
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
