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
#include "vaeWinoTables.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using common::PackedAligned;
using namespace vae_norm;
using namespace vae_wino;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// T: activations (x, residual, cache, sum, yTail, gamma, bias); TD: slabs d and products m.
template <typename T, typename TD>
struct Ptrs
{
    T const* x;
    TD const* m[kVaeWinoMaxComponents];
    T const* residual;
    T const* inBias;
    T const* cache;
    T const* gamma;
    T const* bias;
    T* sum;
    T* yTail;
    TD* d[kVaeWinoMaxComponents];
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

// History ring: a group needs frames t-2 .. t+R-1; frame t-2 is dead once the slabs that
// read it are out (while frame t+R-2 is normalized), so frame t+R-1 can take its slot:
// R + 1 slots suffice. slot(frame) = (frame + 2) mod (R + 1); the cached frames -2, -1
// start in slots 0 and 1.
template <int R>
constexpr int ringSlots()
{
    return R + 1;
}

template <int R>
__device__ __forceinline__ int ringSlot(int frame)
{
    return (frame + 2) % ringSlots<R>();
}

// One thread group (L lanes, K vectors each) owns one position and walks its frames in
// order. The one-vector layout prefetches the next frame's inputs; wider layouts stream
// the row one vector slice at a time with the next slice's loads in flight, so register
// use does not grow with K. Normalized frames go to the shared-memory ring and the
// Winograd slabs are formed while normalizing: slab k is emitted by the frame that
// completes its inputs (R = 2: D0..D2 by the even frame, D3 by the odd one; R = 4:
// D0..D4 by the group's third frame, D5 by its fourth).
template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL, bool IN_WINO>
__global__ void __launch_bounds__(kThreads, blocksPerSm<K, IN_WINO>())
    vaeWinoNormSiluKernel(Ptrs<T, TD> ptrs, VaeWinoNormParams p)
{
    using Vec = PackedAligned<T, V>;
    using VecD = PackedAligned<TD, V>;
    using VecIn = std::conditional_t<IN_WINO, VecD, Vec>;
    using VecF = PackedAligned<float, V>;
    using W = Wino<R>;
    constexpr int NUM_IN = IN_WINO ? W::kMaxTerms : 1;
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

    // Table layout: the V floats of vector vi are split into 16-byte quarters stored in
    // separate planes ((q * nvec + vi) * 4 + e), so a warp reads each quarter with a
    // 16-byte lane stride (bank-conflict free; the natural 32-byte stride is 2-way conflicted).
    constexpr int kQuarters = V / 4;
    int const nvec = C / V;
    for (int c = tid; c < C; c += kThreads)
    {
        float const g = Elem<T>::toFloat(ptrs.gamma[c]);
        int const idx = (((c % V) / 4) * nvec + c / V) * 4 + c % 4;
        sGamma[idx] = REF ? g : g * (0.5f * p.scale);
        sBias[idx] = HAS_BIAS ? Elem<T>::toFloat(ptrs.bias[c]) : 0.0f;
        sInBias[idx] = HAS_IN_BIAS ? Elem<T>::toFloat(ptrs.inBias[c]) : 0.0f;
    }
    __syncthreads();

    int const laneChunkOff = laneInGroup * V;
    int64_t const gridPos = static_cast<int64_t>(gridDim.x) * kPosPerBlock;
    int const T_ = p.frames;

    auto tableVec = [&](float const* table, int k) -> VecF
    {
        VecF r;
        int const vi = laneInGroup + k * L;
#pragma unroll
        for (int q = 0; q < kQuarters; ++q)
        {
            float4 const f = *reinterpret_cast<float4 const*>(table + (q * nvec + vi) * 4);
            r.val[4 * q] = f.x;
            r.val[4 * q + 1] = f.y;
            r.val[4 * q + 2] = f.z;
            r.val[4 * q + 3] = f.w;
        }
        return r;
    };
    VecF gReg, bReg, ibReg;
    if constexpr (CONST_REGS)
    {
        gReg = tableVec(sGamma, 0);
        bReg = tableVec(sBias, 0);
        ibReg = tableVec(sInBias, 0);
    }
    auto constAt = [&](float const* table, VecF const& reg, int k) -> VecF
    {
        if constexpr (CONST_REGS)
        {
            return reg;
        }
        else
        {
            return tableVec(table, k);
        }
    };

    auto gRow = [&](auto const* base, int64_t row) { return base + row * C + laneChunkOff; };
    auto gRowW = [&](auto* base, int64_t row) { return base + row * C + laneChunkOff; };
    auto slotPtr = [&](int slot) { return sHist + (slot * kPosPerBlock + posInBlock) * C + laneChunkOff; };
    auto loadVec = [&](auto const* rowBase, int k)
    {
        using E = std::remove_cv_t<std::remove_pointer_t<decltype(rowBase)>>;
        return *reinterpret_cast<PackedAligned<E, V> const*>(rowBase + k * L * V);
    };
    auto storeVec = [&](auto* rowBase, int k, auto const& v)
    { *reinterpret_cast<std::remove_cv_t<std::remove_reference_t<decltype(v)>>*>(rowBase + k * L * V) = v; };

    // Row pointers of one frame's inputs. The m pointers are picked with compile-time
    // indices: the parameter struct must never be indexed at runtime (that copies it to
    // local memory).
    struct FrameSrc
    {
        std::conditional_t<IN_WINO, TD, T> const* in[NUM_IN];
        T const* res;
    };

    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * kPosPerBlock + posInBlock; pos < P; pos += gridPos)
    {
        // Cached frames -2 and -1 into slots 0 and 1 (zeros where the cache is shorter).
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int const cacheIdx = p.cacheFrames - 2 + h;
            T* const dst = slotPtr(ringSlot<R>(h - 2));
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

        // Inputs of frame t, whose index within its group is J (compile time).
        auto srcOf = [&](auto jTag, int t) -> FrameSrc
        {
            constexpr int J = decltype(jTag)::value;
            FrameSrc f{};
            int64_t const frow = static_cast<int64_t>(t) * P + pos;
            if constexpr (IN_WINO)
            {
                // Rows with fewer than NUM_IN terms leave the spare entries unset (never loaded).
                int64_t const mrow = static_cast<int64_t>(t / R) * P + pos;
                f.in[0] = gRow(ptrs.m[termIndex<R>(J, 0)], mrow);
                if constexpr (numTerms<R>(J) > 1)
                {
                    f.in[1] = gRow(ptrs.m[termIndex<R>(J, 1)], mrow);
                }
                if constexpr (numTerms<R>(J) > 2)
                {
                    f.in[2] = gRow(ptrs.m[termIndex<R>(J, 2)], mrow);
                }
                if constexpr (numTerms<R>(J) > 3)
                {
                    f.in[3] = gRow(ptrs.m[termIndex<R>(J, 3)], mrow);
                }
                if constexpr (numTerms<R>(J) > 4)
                {
                    f.in[4] = gRow(ptrs.m[termIndex<R>(J, 4)], mrow);
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
        VecIn inA[NUM_IN], inB[NUM_IN], inC[NUM_IN];
        Vec resA, resB, resC;
        auto loadSlice = [&](auto nTag, VecIn(&dst)[NUM_IN], Vec& dres, FrameSrc const& f, int k)
        {
            constexpr int n = decltype(nTag)::value;
#pragma unroll
            for (int j = 0; j < n; ++j)
            {
                dst[j] = loadVec(f.in[j], k);
            }
            if constexpr (HAS_RESIDUAL)
            {
                dres = loadVec(f.res, k);
            }
        };

        FrameSrc src = srcOf(std::integral_constant<int, 0>{}, 0);
        if constexpr (FRAME_PF)
        {
            loadSlice(std::integral_constant < int, IN_WINO ? numTerms<R>(0) : 1 > {}, inA, resA, src, 0);
        }

        // One frame: J is its index within the group (compile time), t its frame number.
        // For FRAME_PF the inputs of frame J sit in buffer J % 2.
        auto frame = [&](auto jTag, int t)
        {
            constexpr int J = decltype(jTag)::value;
            constexpr int JN = (J + 1) % R;
            constexpr int fb = FRAME_PF ? (J % 2) : 0;
            constexpr auto termsIn = std::integral_constant < int, IN_WINO ? numTerms<R>(J) : 1 > {};
            constexpr auto termsNext = std::integral_constant < int, IN_WINO ? numTerms<R>(JN) : 1 > {};
            FrameSrc next{};
            if constexpr (FRAME_PF)
            {
                if (t + 1 < T_)
                {
                    next = srcOf(std::integral_constant<int, JN>{}, t + 1);
                    if constexpr (fb == 0)
                    {
                        loadSlice(termsNext, inB, resB, next, 0);
                    }
                    else
                    {
                        loadSlice(termsNext, inA, resA, next, 0);
                    }
                }
            }
            else
            {
                loadSlice(termsIn, inA, resA, src, 0);
            }

            // Pass 1: combine, deferred bias, residual; the bf16 row stays in registers (y)
            // and goes to ``sum``; the sum of squares is accumulated on the way. The input
            // buffer is chosen with a constant branch (never a selected reference) so both
            // buffers stay in registers.
            T* const ySlot = slotPtr(ringSlot<R>(t));
            float ss = 0.0f;
            Vec y[K];
            auto slice = [&](VecIn const(&cin)[NUM_IN], Vec const& cres, Vec& yk, int k)
            {
                VecF const ib = constAt(sInBias, ibReg, k);
#pragma unroll
                for (int i = 0; i < V / 2; ++i)
                {
                    float2 v = pairAt(cin[0], i);
                    if constexpr (IN_WINO)
                    {
                        // sum_q at(J, q) * m_q in index order; the products are exact (powers of
                        // two) so the reference rounding sequence is one rounding per add.
                        constexpr float c0 = W::at(J, termIndex<R>(J, 0));
                        v.x *= c0;
                        v.y *= c0;
#pragma unroll
                        for (int q = 1; q < numTerms<R>(J); ++q)
                        {
                            float const cq = W::at(J, termIndex<R>(J, q)); // constant after unrolling
                            float2 const mq = pairAt(cin[q], i);
                            v.x = addR<T, REF>(v.x, cq * mq.x);
                            v.y = addR<T, REF>(v.y, cq * mq.y);
                        }
                    }
                    if constexpr (HAS_IN_BIAS)
                    {
                        v.x += ib.val[2 * i];
                        v.y += ib.val[2 * i + 1];
                    }
                    // Reference mode reproduces eager's bf16 roundings of the conv output and of
                    // the residual sum; fp32 mode keeps the fp32 values (each round trip is a pack
                    // and an unpack per pair, and those conversions are this kernel's largest
                    // instruction class). The bf16 ``sum`` row is rounded once either way.
                    if constexpr (REF)
                    {
                        v = Elem<T>::toFloat2(Elem<T>::fromFloat2(v));
                    }
                    if constexpr (HAS_RESIDUAL)
                    {
                        float2 const r = pairAt(cres, i);
                        v = make_float2(v.x + r.x, v.y + r.y);
                        if constexpr (REF)
                        {
                            v = Elem<T>::toFloat2(Elem<T>::fromFloat2(v));
                        }
                    }
                    ss = fmaf(v.x, v.x, ss);
                    ss = fmaf(v.y, v.y, ss);
                    setPair(yk, i, v);
                }
                if (ptrs.sum != nullptr) // the pre-norm row: the next block's residual
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
                        loadSlice(termsIn, inB, resB, src, k + 1);
                    }
                    slice(inA, resA, y[k], k);
                    if (k + 1 < K)
                    {
                        if (k + 2 < K)
                        {
                            loadSlice(termsIn, inA, resA, src, k + 2);
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
                    loadSlice(termsIn, inB, resB, src, 1);
                }
#pragma unroll
                for (int k = 0; k < K; k += 3)
                {
                    if (k + 2 < K)
                    {
                        loadSlice(termsIn, inC, resC, src, k + 2);
                    }
                    slice(inA, resA, y[k], k);
                    if (k + 1 < K)
                    {
                        if (k + 3 < K)
                        {
                            loadSlice(termsIn, inA, resA, src, k + 3);
                        }
                        slice(inB, resB, y[k + 1], k + 1);
                    }
                    if (k + 2 < K)
                    {
                        if (k + 4 < K)
                        {
                            loadSlice(termsIn, inB, resB, src, k + 4);
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

            // Pass 2: normalize from registers into the ring slot and emit the slabs this
            // frame completes: slab k with lastFrameOf(k) == J + 2 reads the group's frames
            // f_0 .. f_{J+1} from the ring and f_{J+2} (this frame) from registers.
            int const t0 = t - J; // first output frame of the group; f_i is frame t0 - 2 + i
            int64_t const drow = static_cast<int64_t>(t / R) * P + pos;
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                VecF const gm = constAt(sGamma, gReg, k);
                VecF const bs = constAt(sBias, bReg, k);
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
                Vec hist[J + 2];
#pragma unroll
                for (int i = 0; i < J + 2; ++i)
                {
                    hist[i] = loadVec(slotPtr(ringSlot<R>(t0 - 2 + i)), k);
                }
                auto emit = [&](auto sTag)
                {
                    constexpr int s = decltype(sTag)::value;
                    if constexpr (s < W::kComps && lastFrameOf<R>(s) == J + 2)
                    {
                        VecD o;
#pragma unroll
                        for (int i = 0; i < V / 2; ++i)
                        {
                            // Products of 16-bit values and multiples of 1/2 are exact in fp32,
                            // so fma and separate multiply-add agree; the slab rounds once.
                            float2 acc = make_float2(0.0f, 0.0f);
                            bool first = true;
#pragma unroll
                            for (int fi = 0; fi < J + 3; ++fi)
                            {
                                float const coef = W::bt(s, fi); // constant after unrolling
                                if (coef != 0.0f)
                                {
                                    float2 const fv
                                        = (fi == J + 2) ? pairAt(yk, i) : pairAt(hist[fi < J + 2 ? fi : 0], i);
                                    if (first)
                                    {
                                        acc = make_float2(coef * fv.x, coef * fv.y);
                                        first = false;
                                    }
                                    else
                                    {
                                        acc.x = fmaf(coef, fv.x, acc.x);
                                        acc.y = fmaf(coef, fv.y, acc.y);
                                    }
                                }
                            }
                            setPair(o, i, acc);
                        }
                        storeVec(gRowW(ptrs.d[s], drow), k, o);
                    }
                };
                emit(std::integral_constant<int, 0>{});
                emit(std::integral_constant<int, 1>{});
                emit(std::integral_constant<int, 2>{});
                emit(std::integral_constant<int, 3>{});
                emit(std::integral_constant<int, 4>{});
                emit(std::integral_constant<int, 5>{});
            }
            if constexpr (FRAME_PF)
            {
                src = next;
            }
            else
            {
                if (t + 1 < T_)
                {
                    src = srcOf(std::integral_constant<int, JN>{}, t + 1);
                }
            }
        };

        for (int t = 0; t < T_; t += R)
        {
            frame(std::integral_constant<int, 0>{}, t);
            frame(std::integral_constant<int, 1>{}, t + 1);
            if constexpr (R == 4)
            {
                frame(std::integral_constant<int, 2>{}, t + 2);
                frame(std::integral_constant<int, 3>{}, t + 3);
            }
        }
        // The last two normalized frames are the next chunk's cache.
        {
            T const* const sa = slotPtr(ringSlot<R>(T_ - 2));
            T const* const sb = slotPtr(ringSlot<R>(T_ - 1));
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

template <int L, int R>
inline int sharedBytes(int channels)
{
    return 3 * channels * static_cast<int>(sizeof(float)) + ringSlots<R>() * positionsPerBlock<L>() * channels * 2;
}

template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL, bool IN_WINO>
void launchConfig(Ptrs<T, TD> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    auto* kernel = vaeWinoNormSiluKernel<T, TD, R, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, IN_WINO>;
    constexpr int kPosPerBlock = positionsPerBlock<L>();
    int const smem = sharedBytes<L, R>(p.channels);
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

template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL>
void launchInWino(Ptrs<T, TD> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    if (ptrs.x == nullptr)
        launchConfig<T, TD, R, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, true>(ptrs, p, stream);
    else
        launchConfig<T, TD, R, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, false>(ptrs, p, stream);
}

template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS>
void launchResidual(Ptrs<T, TD> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    if (ptrs.residual != nullptr)
        launchInWino<T, TD, R, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, true>(ptrs, p, stream);
    else
        launchInWino<T, TD, R, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, false>(ptrs, p, stream);
}

template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS>
void launchInBias(Ptrs<T, TD> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    if (ptrs.inBias != nullptr)
        launchResidual<T, TD, R, V, L, K, REF, HAS_BIAS, true>(ptrs, p, stream);
    else
        launchResidual<T, TD, R, V, L, K, REF, HAS_BIAS, false>(ptrs, p, stream);
}

template <typename T, typename TD, int R, int V, int L, int K>
void launchModes(Ptrs<T, TD> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    bool const hasBias = ptrs.bias != nullptr;
    if (hasBias && p.referenceRounding)
        launchInBias<T, TD, R, V, L, K, true, true>(ptrs, p, stream);
    else if (hasBias)
        launchInBias<T, TD, R, V, L, K, false, true>(ptrs, p, stream);
    else if (p.referenceRounding)
        launchInBias<T, TD, R, V, L, K, true, false>(ptrs, p, stream);
    else
        launchInBias<T, TD, R, V, L, K, false, false>(ptrs, p, stream);
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

template <typename T, typename TD, int R, int V>
void launchLayout(Ptrs<T, TD> const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream)
{
    RowLayout const l = winoRowLayout<V>(p.channels);
#define TRY_LAYOUT(LL, KK)                                                                                             \
    if (l.lanesPerRow == (LL) && l.vecsPerLane == (KK))                                                                \
    {                                                                                                                  \
        launchModes<T, TD, R, V, LL, KK>(ptrs, p, stream);                                                             \
        return;                                                                                                        \
    }
    VAE_NORM_FOR_EACH_LAYOUT(TRY_LAYOUT)
#undef TRY_LAYOUT
    TLLM_THROW("vaeWinoNormSilu: no lane layout for %d channels (lanes %d, vectors %d)", p.channels, l.lanesPerRow,
        l.vecsPerLane);
}

template <typename T, typename TD, int R>
void launchTyped(VaeWinoNormPointers const& in, VaeWinoNormParams const& p, cudaStream_t stream)
{
    constexpr int kVec = static_cast<int>(sizeof(float4) / sizeof(T));
    Ptrs<T, TD> ptrs{};
    ptrs.x = static_cast<T const*>(in.x);
    for (int k = 0; k < kVaeWinoMaxComponents; ++k)
    {
        ptrs.m[k] = static_cast<TD const*>(in.m[k]);
        ptrs.d[k] = static_cast<TD*>(in.d[k]);
    }
    ptrs.residual = static_cast<T const*>(in.residual);
    ptrs.inBias = static_cast<T const*>(in.inBias);
    ptrs.cache = static_cast<T const*>(in.cache);
    ptrs.gamma = static_cast<T const*>(in.gamma);
    ptrs.bias = static_cast<T const*>(in.bias);
    ptrs.sum = static_cast<T*>(in.sum);
    ptrs.yTail = static_cast<T*>(in.yTail);
    launchLayout<T, TD, R, kVec>(ptrs, p, stream);
}

} // namespace

void launchVaeWinoNormSilu(VaeWinoNormPointers const& in, VaeWinoNormParams const& p, cudaStream_t stream)
{
    constexpr int kVecBf16 = static_cast<int>(sizeof(float4) / sizeof(__nv_bfloat16));
    TLLM_CHECK_WITH_INFO(p.channels > 0 && p.channels % kVecBf16 == 0 && p.channels <= kVaeRmsNormSiluMaxChannels,
        "vaeWinoNormSilu: channels must be a positive multiple of %d up to %d, got %d", kVecBf16,
        kVaeRmsNormSiluMaxChannels, p.channels);
    TLLM_CHECK_WITH_INFO(p.order == 2 || p.order == 4, "vaeWinoNormSilu: order must be 2 or 4");
    TLLM_CHECK_WITH_INFO(p.order == 2 || p.halfSlabs, "vaeWinoNormSilu: order 4 needs fp16 slabs");
    TLLM_CHECK_WITH_INFO(
        p.frames >= p.order && p.frames % p.order == 0, "vaeWinoNormSilu: frames must be a positive multiple of order");
    TLLM_CHECK_WITH_INFO(p.cacheFrames >= 0 && p.cacheFrames <= 2, "vaeWinoNormSilu: cacheFrames must be 0..2");
    TLLM_CHECK_WITH_INFO((in.x == nullptr) != (in.m[0] == nullptr), "vaeWinoNormSilu: exactly one of x and m");
    TLLM_CHECK_WITH_INFO(in.residual == nullptr || in.sum != nullptr, "vaeWinoNormSilu: a residual needs a sum output");
    if (p.positions == 0)
    {
        return;
    }
    if (p.order == 4)
    {
        launchTyped<__nv_bfloat16, __half, 4>(in, p, stream);
    }
    else if (p.halfSlabs)
    {
        launchTyped<__nv_bfloat16, __half, 2>(in, p, stream);
    }
    else
    {
        launchTyped<__nv_bfloat16, __nv_bfloat16, 2>(in, p, stream);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
