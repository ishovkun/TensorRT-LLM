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

// Channels per thread: 4 (8-byte vectors) for m = 2; 2 (one bf16x2 / fp16x2 pair) for m = 4,
// whose 6x6 window would otherwise not fit in registers (measured: 4 channels per thread
// costs 175 registers and is no faster; 8 channels for m = 2 cost 165 registers, one block
// per SM).
template <int M>
struct Lanes
{
    static constexpr int V = M == 2 ? 4 : 2;
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
    int64_t coordStride;   // elements between consecutive coordinates of V / M
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
    g.coordStride = p.stackFrames ? g.tilesPerFrame * p.frames * p.channels : g.tilesPerComp * p.channels;
    return g;
}

// Element offset of coordinate 0 of V / M for frame n (component comp, pair within it), a tile
// and channel c0. Layout [comp * coords, pairs * tiles, C], or with stacked frames
// [coords, tiles, N * C] (frame n at channel offset n * C).
template <int COORDS>
__device__ __forceinline__ int64_t vmOffset(
    Geometry const& g, VaeWinoSpatialParams const& p, int n, int comp, int pair, int64_t tile, int c0)
{
    if (p.stackFrames)
    {
        return tile * (static_cast<int64_t>(p.frames) * p.channels) + static_cast<int64_t>(n) * p.channels + c0;
    }
    int64_t const row = static_cast<int64_t>(comp) * COORDS * g.tilesPerComp + pair * g.tilesPerFrame + tile;
    return row * p.channels + c0;
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
    // Grid: x over channel vectors, y over the strip's tiles, z over frames; the per-block
    // tile and frame indices are uniform, so no thread does a 64-bit division.
    int const cv = blockIdx.x * kThreads + threadIdx.x;
    if (cv < cvec)
    {
        int const c0 = cv * V;
        int const tile = blockIdx.y;
        int const n = blockIdx.z;
        int const i = p.row0 + tile / g.tw;
        int const j = tile - (i - p.row0) * g.tw;
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
        // Keep every load of the window in flight before the arithmetic starts: without the
        // barrier the compiler sinks each load to its single use and the tile pays one
        // memory latency per window column instead of one per window.
        asm volatile("" ::: "memory");
        TS* vp = v + vmOffset<A * A>(g, p, n, comp, pair, tile, c0);
        int64_t const coordStride = g.coordStride;
        // Two separable passes per channel pair e: T[x][b] = row x of B^T applied to window
        // column b, then V[x][y] = row y of B^T applied to row x of T. Every input value is
        // converted once and every transform row evaluated once.
        VecS out[A * A];
#pragma unroll
        for (int e = 0; e < V / 2; ++e)
        {
            float2 tr[A][A];
#pragma unroll
            for (int b = 0; b < A; ++b)
            {
                float2 col[A];
#pragma unroll
                for (int a = 0; a < A; ++a)
                {
                    col[a] = pairAt(d[a][b], e);
                }
#pragma unroll
                for (int x = 0; x < A; ++x)
                {
                    tr[x][b] = W::bt(x, col);
                }
            }
#pragma unroll
            for (int x = 0; x < A; ++x)
            {
#pragma unroll
                for (int y = 0; y < A; ++y)
                {
                    setPair(out[x * A + y], e, W::bt(y, tr[x]));
                }
            }
        }
#pragma unroll
        for (int k = 0; k < A * A; ++k)
        {
            *reinterpret_cast<VecS*>(vp) = out[k];
            vp += coordStride;
        }
    }
}

// Staged input transform: a block covers one tile row x kStagedTiles consecutive tiles x
// kStagedChannels channels of one frame. The block first copies the (m+2) x (m * kStagedTiles + 2)
// window rows it needs into shared memory with 16-byte loads (zero outside the frame), then
// every thread transforms one (tile, channel pair) from shared memory, where all window
// addresses are compile-time offsets from the thread's base. Needs channels % kStagedChannels == 0.
constexpr int kStagedTiles = 8;
constexpr int kStagedChannels = 64;
static_assert(kStagedTiles * (kStagedChannels / 2) == kThreads, "one thread per (tile, channel pair)");

template <int M>
__global__ void __launch_bounds__(kThreads)
    vaeWinoSpatialInputStagedKernel(T const* __restrict__ frames, TS* __restrict__ v, VaeWinoSpatialParams p)
{
    using W = WinoMath<M>;
    constexpr int A = W::A;
    constexpr int Cols = M * kStagedTiles + 2;
    constexpr int CH = kStagedChannels;
    using PairS = PackedAligned<TS, 2>; // one channel pair per thread, whatever Lanes<M>::V is
    __shared__ __align__(16) T win[A][Cols][CH];

    Geometry const g = geometry<M>(p);
    int const C = p.channels;
    int const chunk = blockIdx.x; // channel chunk
    int const group = blockIdx.y; // tile row x tile-column group
    int const n = blockIdx.z;     // frame
    int const groupsPerRow = (g.tw + kStagedTiles - 1) / kStagedTiles;
    int const i = p.row0 + group / groupsPerRow;
    int const j0 = (group - (i - p.row0) * groupsPerRow) * kStagedTiles;
    int const comp = n / g.pairs, pair = n - comp * g.pairs;
    int const row0 = M * i - 1, col0 = M * j0 - 1;

    // Copy phase: 16-byte cp.async chunks over (window row, window column, 8-channel chunk),
    // all issued before the wait; chunks outside the frame are zero-filled by the copy unit.
    {
        constexpr int ChunksPerPixel = CH / 8;
        constexpr int Total = A * Cols * ChunksPerPixel;
        constexpr int Iters = (Total + kThreads - 1) / kThreads;
        T const* frame0 = frames + (static_cast<int64_t>(n) * p.height * p.width) * C + chunk * CH;
#pragma unroll
        for (int it = 0; it < Iters; ++it)
        {
            int const q = threadIdx.x + it * kThreads;
            if (q < Total)
            {
                int const cc = q % ChunksPerPixel;
                int const pix = q / ChunksPerPixel;
                int const a = pix / Cols, b = pix - a * Cols;
                int const row = row0 + a, col = col0 + b;
                bool const in = row >= 0 && row < p.height && col >= 0 && col < p.width;
                T const* src = in ? frame0 + (static_cast<int64_t>(row) * p.width + col) * C + cc * 8 : frame0;
                unsigned const dst = static_cast<unsigned>(__cvta_generic_to_shared(&win[a][b][cc * 8]));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(in ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();

    // Transform phase: thread -> (tile within the group, channel pair).
    int const tt = threadIdx.x / (CH / 2);
    int const pp = threadIdx.x - tt * (CH / 2);
    int const j = j0 + tt;
    if (j >= g.tw)
    {
        return;
    }
    float2 tr[A][A]; // T[x][b]
#pragma unroll
    for (int b = 0; b < A; ++b)
    {
        float2 col[A];
#pragma unroll
        for (int a = 0; a < A; ++a)
        {
            col[a] = Elem<T>::toFloat2(*reinterpret_cast<typename Elem<T>::Pair const*>(&win[a][M * tt + b][2 * pp]));
        }
#pragma unroll
        for (int x = 0; x < A; ++x)
        {
            tr[x][b] = W::bt(x, col);
        }
    }
    int const tile = (i - p.row0) * g.tw + j;
    TS* vp = v + vmOffset<A * A>(g, p, n, comp, pair, tile, chunk * CH + 2 * pp);
    int64_t const coordStride = g.coordStride;
#pragma unroll
    for (int x = 0; x < A; ++x)
    {
#pragma unroll
        for (int y = 0; y < A; ++y)
        {
            PairS out;
            setPair(out, 0, W::bt(y, tr[x]));
            *reinterpret_cast<PairS*>(vp) = out;
            vp += coordStride;
        }
    }
}

// y = A^T M A for one tile and V channels: (m+2)^2 vectors in, m x m output pixels out.
template <int M>
__global__ void __launch_bounds__(kThreads) vaeWinoSpatialOutputKernel(
    TS const* __restrict__ m, T* __restrict__ frames, T const* __restrict__ bias, VaeWinoSpatialParams p)
{
    using W = WinoMath<M>;
    constexpr int A = W::A;
    constexpr int V = Lanes<M>::V;
    using Vec = typename Lanes<M>::Vec;
    using VecS = typename Lanes<M>::VecS;
    Geometry const g = geometry<M>(p);
    int const C = p.channels;
    int const cvec = C / V;
    // Grid: x over channel vectors, y over the strip's tiles, z over frames; the per-block
    // tile and frame indices are uniform, so no thread does a 64-bit division.
    int const cv = blockIdx.x * kThreads + threadIdx.x;
    if (cv < cvec)
    {
        int const c0 = cv * V;
        int const tile = blockIdx.y;
        int const n = blockIdx.z;
        int const i = p.row0 + tile / g.tw;
        int const j = tile - (i - p.row0) * g.tw;
        int const comp = n / g.pairs, pair = n - comp * g.pairs;

        VecS mv[A * A];
        TS const* mp = m + vmOffset<A * A>(g, p, n, comp, pair, tile, c0);
        int64_t const coordStride = g.coordStride;
#pragma unroll
        for (int k = 0; k < A * A; ++k)
        {
            mv[k] = *reinterpret_cast<VecS const*>(mp);
            mp += coordStride;
        }
        Vec const bv = bias ? *reinterpret_cast<Vec const*>(bias + c0) : common::make_zeros<Vec>();
        asm volatile("" ::: "memory"); // all coordinate loads in flight before the arithmetic
        T* frame0 = frames + (static_cast<int64_t>(n) * p.height * p.width) * C + c0;
        // Two separable passes per channel pair e: T[r][b] = row r of A^T applied to
        // coordinate column b, then y[r][s] = row s of A^T applied to row r of T.
        Vec out[M * M];
#pragma unroll
        for (int e = 0; e < V / 2; ++e)
        {
            float2 tr[M][A];
#pragma unroll
            for (int b = 0; b < A; ++b)
            {
                float2 col[A];
#pragma unroll
                for (int x = 0; x < A; ++x)
                {
                    col[x] = pairAt(mv[x * A + b], e);
                }
#pragma unroll
                for (int r = 0; r < M; ++r)
                {
                    tr[r][b] = W::at(r, col);
                }
            }
            float2 const bp = pairAt(bv, e);
#pragma unroll
            for (int r = 0; r < M; ++r)
            {
#pragma unroll
                for (int sidx = 0; sidx < M; ++sidx)
                {
                    setPair(out[r * M + sidx], e, W::at(sidx, tr[r]) + bp);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < M; ++r)
        {
            T* rowp = frame0 + (static_cast<int64_t>(M * i + r) * p.width + M * j) * C;
#pragma unroll
            for (int sidx = 0; sidx < M; ++sidx)
            {
                if (M * i + r < p.height && M * j + sidx < p.width)
                {
                    *reinterpret_cast<Vec*>(rowp + sidx * C) = out[r * M + sidx];
                }
            }
        }
    }
}

dim3 gridFor(VaeWinoSpatialParams const& p)
{
    int const lanes = p.tile == 2 ? Lanes<2>::V : Lanes<4>::V;
    int const cvec = p.channels / lanes;
    int const tiles = (p.row1 - p.row0) * ((p.width + p.tile - 1) / p.tile);
    TLLM_CHECK_WITH_INFO(tiles <= 65535 && p.frames <= 65535, "vaeWinoSpatial: too many tiles or frames for the grid");
    return dim3((cvec + kThreads - 1) / kThreads, tiles, p.frames);
}

void check(VaeWinoSpatialParams const& p)
{
    TLLM_CHECK_WITH_INFO(p.tile == 2 || p.tile == 4, "vaeWinoSpatial: tile must be 2 or 4");
    TLLM_CHECK_WITH_INFO(p.channels % 8 == 0, "vaeWinoSpatial: channels must be a multiple of 8");
    TLLM_CHECK_WITH_INFO(
        p.components >= 1 && p.frames % p.components == 0, "vaeWinoSpatial: frames must be a multiple of components");
    TLLM_CHECK_WITH_INFO(!p.stackFrames || p.components == 1, "vaeWinoSpatial: stacked frames need components == 1");
    TLLM_CHECK_WITH_INFO(0 <= p.row0 && p.row0 < p.row1 && p.row1 <= (p.height + p.tile - 1) / p.tile,
        "vaeWinoSpatial: bad tile-row range");
}

} // namespace

template <int M>
void launchInputStaged(T const* frames, TS* v, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    int const tw = (p.width + M - 1) / M;
    int const groups = (p.row1 - p.row0) * ((tw + kStagedTiles - 1) / kStagedTiles);
    TLLM_CHECK_WITH_INFO(groups <= 65535 && p.frames <= 65535, "vaeWinoSpatial: too many tiles or frames for the grid");
    dim3 const grid(p.channels / kStagedChannels, groups, p.frames);
    vaeWinoSpatialInputStagedKernel<M><<<grid, kThreads, 0, stream>>>(frames, v, p);
}

void launchVaeWinoSpatialInput(void const* frames, void* v, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    T const* f = static_cast<T const*>(frames);
    TS* out = static_cast<TS*>(v);
    if (p.channels % kStagedChannels == 0)
    {
        p.tile == 2 ? launchInputStaged<2>(f, out, p, stream) : launchInputStaged<4>(f, out, p, stream);
    }
    else
    {
        dim3 const grid = gridFor(p);
        if (p.tile == 2)
        {
            vaeWinoSpatialInputKernel<2><<<grid, kThreads, 0, stream>>>(f, out, p);
        }
        else
        {
            vaeWinoSpatialInputKernel<4><<<grid, kThreads, 0, stream>>>(f, out, p);
        }
    }
    sync_check_cuda_error(stream);
}

void launchVaeWinoSpatialOutput(
    void const* m, void* frames, void const* bias, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    TLLM_CHECK_WITH_INFO(!p.stackFrames, "vaeWinoSpatial: the output transform has no stacked-frame layout");
    dim3 const grid = gridFor(p);
    if (p.tile == 2)
    {
        vaeWinoSpatialOutputKernel<2><<<grid, kThreads, 0, stream>>>(
            static_cast<TS const*>(m), static_cast<T*>(frames), static_cast<T const*>(bias), p);
    }
    else
    {
        vaeWinoSpatialOutputKernel<4><<<grid, kThreads, 0, stream>>>(
            static_cast<TS const*>(m), static_cast<T*>(frames), static_cast<T const*>(bias), p);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
