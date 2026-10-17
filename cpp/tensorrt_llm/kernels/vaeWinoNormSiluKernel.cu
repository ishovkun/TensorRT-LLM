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
#include "vaeRmsNormSiluKernel.h"
#include "vaeWinoNormSiluKernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

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

// Resident blocks per SM the register budget is sized for. Per-thread state is bounded
// independently of K: the row being normalized (K vectors) plus two vector slices of
// inputs in flight; the frame history lives in shared memory.
template <int K, bool IN_WINO>
constexpr int blocksPerSm()
{
    if (K >= 4)
    {
        return 2;                        // ~110 registers; bytes in flight come from the deeper slice pipeline
    }
    return (!IN_WINO && K <= 2) ? 4 : 3; // producers hold one input per slice and are latency-bound: more warps
}

// Vector slices of one frame kept in flight ahead of the one being normalized.
template <int K>
constexpr int sliceDepth()
{
    return K >= 4 ? 3 : 2;
}

// History ring slots. A pair's slabs need frames t-2, t-1 (history) and t, t+1; frame
// t-2 is dead once the even frame's slabs are out, so frame t+1 can take its slot: three
// slots suffice. slot(frame) = (frame + 2) mod 3, the cached frames -2, -1 start in 0, 1.
constexpr int kRingSlots = 3;

__device__ __forceinline__ int ringSlot(int frame)
{
    return (frame + 2) % kRingSlots;
}

// One thread group (L lanes, K vectors each) owns one position and walks its frames in
// order. The one-vector layout prefetches the next frame's inputs; wider layouts stream
// the row one vector slice at a time with the next slice's loads in flight, so register
// use does not grow with K. Normalized frames go to the shared-memory ring and the
// Winograd slabs are formed while normalizing (even frame: D0..D2 from the two history
// slots; odd frame: D3).
template <typename T, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS, bool HAS_RESIDUAL, bool IN_WINO>
__global__ void __launch_bounds__(kThreads, blocksPerSm<K, IN_WINO>())
    vaeWinoNormSiluKernel(Ptrs<T> ptrs, VaeWinoNormParams p)
{
    using Vec = PackedAligned<T, V>;
    using VecF = PackedAligned<float, V>;
    constexpr int NUM_IN = IN_WINO ? 3 : 1;
    constexpr bool FRAME_PF = (K == 1);   // prefetch whole frames
    constexpr bool CONST_REGS = (K == 1); // per-channel constants in registers
    constexpr int kPosPerWarp = 32 / L;
    constexpr int kPosPerBlock = kPosPerWarp * kWarps;

    extern __shared__ __align__(16) unsigned char smemRaw[];
    float* const sGamma = reinterpret_cast<float*>(smemRaw);
    float* const sBias = sGamma + p.channels;
    float* const sInBias = sGamma + 2 * p.channels;
    T* const sHist = reinterpret_cast<T*>(sGamma + 3 * p.channels); // [slots][positions in block][C]

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const group = lane / L;
    int const laneInGroup = lane - group * L;
    int const C = p.channels;
    int64_t const P = p.positions;
    int const posInBlock = warp * kPosPerWarp + group;

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

    VecF gReg, bReg, ibReg;
    if constexpr (CONST_REGS)
    {
        gReg = *reinterpret_cast<VecF const*>(sGamma + laneChunkOff);
        bReg = *reinterpret_cast<VecF const*>(sBias + laneChunkOff);
        ibReg = *reinterpret_cast<VecF const*>(sInBias + laneChunkOff);
    }
    auto constAt = [&](float const* table, VecF const& reg, int k) -> VecF
    {
        if constexpr (CONST_REGS)
        {
            return reg;
        }
        else
        {
            return *reinterpret_cast<VecF const*>(table + laneChunkOff + k * L * V);
        }
    };

    auto gRow = [&](T const* base, int64_t row) { return base + row * C + laneChunkOff; };
    auto gRowW = [&](T* base, int64_t row) { return base + row * C + laneChunkOff; };
    auto slotPtr = [&](int slot) { return sHist + (slot * kPosPerBlock + posInBlock) * C + laneChunkOff; };
    auto loadVec = [&](T const* rowBase, int k) { return *reinterpret_cast<Vec const*>(rowBase + k * L * V); };
    auto storeVec = [&](T* rowBase, int k, Vec const& v) { *reinterpret_cast<Vec*>(rowBase + k * L * V) = v; };

    // Row pointers of one frame's inputs. The m pointers are picked with a branch: the
    // parameter struct must never be indexed at runtime (that copies it to local memory).
    struct FrameSrc
    {
        T const* in[NUM_IN];
        T const* res;
    };

    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * kPosPerBlock + posInBlock; pos < P; pos += gridPos)
    {
        // Cached frames -2 and -1 into slots 0 and 1 (zeros where the cache is shorter).
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int const cacheIdx = p.cacheFrames - 2 + h;
            T* const dst = slotPtr(ringSlot(h - 2));
            if (cacheIdx >= 0)
            {
                T const* const src = gRow(ptrs.cache, static_cast<int64_t>(cacheIdx) * P + pos);
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    storeVec(dst, k, loadVec(src, k));
                }
            }
            else
            {
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    storeVec(dst, k, common::make_zeros<Vec>());
                }
            }
        }

        auto srcOf = [&](int t) -> FrameSrc
        {
            FrameSrc f{};
            int64_t const frow = static_cast<int64_t>(t) * P + pos;
            if constexpr (IN_WINO)
            {
                int64_t const mrow = static_cast<int64_t>(t >> 1) * P + pos;
                if (t & 1)
                {
                    f.in[0] = gRow(ptrs.m[1], mrow);
                    f.in[1] = gRow(ptrs.m[2], mrow);
                    f.in[2] = gRow(ptrs.m[3], mrow);
                }
                else
                {
                    f.in[0] = gRow(ptrs.m[0], mrow);
                    f.in[1] = gRow(ptrs.m[1], mrow);
                    f.in[2] = gRow(ptrs.m[2], mrow);
                }
            }
            else
            {
                f.in[0] = gRow(ptrs.x, frow);
            }
            if constexpr (HAS_RESIDUAL)
            {
                f.res = gRow(ptrs.residual, frow);
            }
            return f;
        };

        // Two input buffers (named, never indexed, so they stay in registers): two frames
        // for FRAME_PF, otherwise two vector slices of one frame.
        Vec inA[NUM_IN], inB[NUM_IN], inC[NUM_IN];
        Vec resA, resB, resC;
        auto loadSlice = [&](Vec(&dst)[NUM_IN], Vec& dres, FrameSrc const& f, int k)
        {
#pragma unroll
            for (int j = 0; j < NUM_IN; ++j)
            {
                dst[j] = loadVec(f.in[j], k);
            }
            if constexpr (HAS_RESIDUAL)
            {
                dres = loadVec(f.res, k);
            }
        };

        FrameSrc src = srcOf(0);
        if constexpr (FRAME_PF)
        {
            loadSlice(inA, resA, src, 0);
        }

        // One frame. ``bufTag``: for FRAME_PF the buffer holding this frame's inputs.
        auto frame = [&](auto bufTag, int t)
        {
            constexpr int fb = FRAME_PF ? decltype(bufTag)::value : 0;
            FrameSrc next{};
            if constexpr (FRAME_PF)
            {
                if (t + 1 < T_)
                {
                    next = srcOf(t + 1);
                    if constexpr (fb == 0)
                    {
                        loadSlice(inB, resB, next, 0);
                    }
                    else
                    {
                        loadSlice(inA, resA, next, 0);
                    }
                }
            }
            else
            {
                loadSlice(inA, resA, src, 0);
            }

            // Pass 1: combine, deferred bias, residual; the bf16 row stays in registers (y)
            // and goes to ``sum``; the sum of squares is accumulated on the way. The input
            // buffer is chosen with a constant branch (never a selected reference) so both
            // buffers stay in registers.
            T* const ySlot = slotPtr(ringSlot(t));
            float const sgn = (t & 1) ? -1.0f : 1.0f;
            float ss = 0.0f;
            Vec y[K];
            auto slice = [&](Vec const(&cin)[NUM_IN], Vec const& cres, Vec& yk, int k)
            {
                VecF const ib = constAt(sInBias, ibReg, k);
#pragma unroll
                for (int i = 0; i < V / 2; ++i)
                {
                    float2 v = pairAt(cin[0], i);
                    if constexpr (IN_WINO)
                    {
                        float2 const fb2 = pairAt(cin[1], i), fc = pairAt(cin[2], i);
                        v.x = addR<T, REF>(addR<T, REF>(v.x, sgn * fb2.x), sgn * fc.x);
                        v.y = addR<T, REF>(addR<T, REF>(v.y, sgn * fb2.y), sgn * fc.y);
                    }
                    if constexpr (HAS_IN_BIAS)
                    {
                        v.x += ib.val[2 * i];
                        v.y += ib.val[2 * i + 1];
                    }
                    v = Elem<T>::toFloat2(Elem<T>::fromFloat2(v)); // the conv output as eager holds it
                    if constexpr (HAS_RESIDUAL)
                    {
                        float2 const r = pairAt(cres, i);
                        v = Elem<T>::toFloat2(Elem<T>::fromFloat2(make_float2(v.x + r.x, v.y + r.y)));
                    }
                    ss = fmaf(v.x, v.x, ss);
                    ss = fmaf(v.y, v.y, ss);
                    setPair(yk, i, v);
                }
                if constexpr (HAS_RESIDUAL)
                {
                    storeVec(gRowW(ptrs.sum, static_cast<int64_t>(t) * P + pos), k, yk);
                }
            };
            if constexpr (FRAME_PF)
            {
                // K == 1: the frame's single slice sits in the buffer chosen by the frame parity.
                if constexpr (fb == 0)
                {
                    slice(inA, resA, y[0], 0);
                }
                else
                {
                    slice(inB, resB, y[0], 0);
                }
            }
            else if constexpr (sliceDepth<K>() == 2)
            {
                // Slices in pairs: load the next slice into the other buffer, then compute.
#pragma unroll
                for (int k = 0; k < K; k += 2)
                {
                    if (k + 1 < K)
                    {
                        loadSlice(inB, resB, src, k + 1);
                    }
                    slice(inA, resA, y[k], k);
                    if (k + 1 < K)
                    {
                        if (k + 2 < K)
                        {
                            loadSlice(inA, resA, src, k + 2);
                        }
                        slice(inB, resB, y[k + 1], k + 1);
                    }
                }
            }
            else
            {
                // Three rotating buffers: two slices in flight while one is normalized.
                if (1 < K)
                {
                    loadSlice(inB, resB, src, 1);
                }
#pragma unroll
                for (int k = 0; k < K; k += 3)
                {
                    if (k + 2 < K)
                    {
                        loadSlice(inC, resC, src, k + 2);
                    }
                    slice(inA, resA, y[k], k);
                    if (k + 1 < K)
                    {
                        if (k + 3 < K)
                        {
                            loadSlice(inA, resA, src, k + 3);
                        }
                        slice(inB, resB, y[k + 1], k + 1);
                    }
                    if (k + 2 < K)
                    {
                        if (k + 4 < K)
                        {
                            loadSlice(inB, resB, src, k + 4);
                        }
                        slice(inC, resC, y[k + 2], k + 2);
                    }
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

            // Pass 2: normalize from registers into the ring slot and emit this frame's
            // share of the pair's slabs.
            bool const odd = (t & 1) != 0;
            T const* const hist0 = slotPtr(ringSlot(t - 2));               // f0 (even frame only)
            T const* const hist1 = slotPtr(ringSlot(odd ? t - 2 : t - 1)); // the pair's f1
            int64_t const drow = static_cast<int64_t>(t >> 1) * P + pos;
            T* const dOut0 = gRowW(ptrs.d[0], drow);
            T* const dOut1 = gRowW(ptrs.d[1], drow);
            T* const dOut2 = gRowW(ptrs.d[2], drow);
            T* const dOut3 = gRowW(ptrs.d[3], drow);
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                VecF const gm = constAt(sGamma, gReg, k);
                VecF const bs = constAt(sBias, bReg, k);
                Vec const h1 = loadVec(hist1, k);
                Vec yk = y[k];
#pragma unroll
                for (int i = 0; i < V / 2; ++i)
                {
                    float2 const f = pairAt(yk, i);
                    float2 r;
                    r.x = normElement<T, REF, HAS_BIAS>(f.x, denom, invDenom, p.scale, gm.val[2 * i], bs.val[2 * i]);
                    r.y = normElement<T, REF, HAS_BIAS>(
                        f.y, denom, invDenom, p.scale, gm.val[2 * i + 1], bs.val[2 * i + 1]);
                    setPair(yk, i, r);
                }
                storeVec(ySlot, k, yk);
                if (!odd)
                {
                    Vec const h0 = loadVec(hist0, k);
                    Vec o0, o1, o2;
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const c = pairAt(yk, i), a = pairAt(h0, i), b = pairAt(h1, i);
                        setPair(o0, i, make_float2(a.x - c.x, a.y - c.y));
                        setPair(o1, i, make_float2(b.x + c.x, b.y + c.y));
                        setPair(o2, i, make_float2(c.x - b.x, c.y - b.y));
                    }
                    storeVec(dOut0, k, o0);
                    storeVec(dOut1, k, o1);
                    storeVec(dOut2, k, o2);
                }
                else
                {
                    Vec o3;
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const e = pairAt(yk, i), b = pairAt(h1, i);
                        setPair(o3, i, make_float2(b.x - e.x, b.y - e.y));
                    }
                    storeVec(dOut3, k, o3);
                }
            }
            if constexpr (FRAME_PF)
            {
                src = next;
            }
            else
            {
                if (t + 1 < T_)
                {
                    src = srcOf(t + 1);
                }
            }
        };

        for (int t = 0; t < T_; t += 2)
        {
            frame(std::integral_constant<int, 0>{}, t);
            frame(std::integral_constant<int, 1>{}, t + 1);
        }
        // The last two normalized frames are the next chunk's cache.
        {
            T const* const sa = slotPtr(ringSlot(T_ - 2));
            T const* const sb = slotPtr(ringSlot(T_ - 1));
            T* const ta = gRowW(ptrs.yTail, pos);
            T* const tb = gRowW(ptrs.yTail, P + pos);
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                storeVec(ta, k, loadVec(sa, k));
                storeVec(tb, k, loadVec(sb, k));
            }
        }
    }
}

template <int L>
constexpr int positionsPerBlock()
{
    return (32 / L) * kWarps;
}

template <int L>
inline int sharedBytes(int channels)
{
    return 3 * channels * static_cast<int>(sizeof(float)) + kRingSlots * positionsPerBlock<L>() * channels * 2;
}

template <typename T, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS, bool HAS_RESIDUAL, bool IN_WINO>
void launchConfig(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    auto* kernel = vaeWinoNormSiluKernel<T, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, IN_WINO>;
    constexpr int kPosPerBlock = positionsPerBlock<L>();
    int const smem = sharedBytes<L>(p.channels);
    static int configuredSmem = 0; // per instantiation; the attribute only needs raising once
    if (smem > configuredSmem)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configuredSmem = smem;
    }
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int64_t const steps = (p.positions + kPosPerBlock - 1) / kPosPerBlock;
    int64_t const persistent = static_cast<int64_t>(sms) * blocksPerSm<K, IN_WINO>();
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

// Lane layout for this kernel: like the plain norm kernel's, but a row is never split
// one vector per lane when two per lane is possible (the per-row overhead, shuffles and
// addressing, amortizes over too few elements at K = 1; measured 74% vs 78-90%).
template <int V>
RowLayout winoRowLayout(int channels)
{
    RowLayout l = makeRowLayout<V>(channels);
    if (l.vecsPerLane == 1 && l.lanesPerRow > 1)
    {
        l.lanesPerRow /= 2;
        l.vecsPerLane = 2;
    }
    return l;
}

template <typename T, int V>
void launchLayout(Ptrs<T> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    RowLayout const l = winoRowLayout<V>(p.channels);
#define TRY_LAYOUT(LL, KK)                                                                                             \
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
    TLLM_CHECK_WITH_INFO(
        (in.residual == nullptr) == (in.sum == nullptr), "vaeWinoNormSilu: residual and sum go together");
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
