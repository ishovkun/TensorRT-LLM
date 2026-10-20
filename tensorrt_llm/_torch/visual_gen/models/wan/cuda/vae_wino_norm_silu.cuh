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

// RMSNorm + SiLU with the temporal Winograd F(R,3) transforms: the device side of
// cpp/tensorrt_llm/kernels/vaeWinoNormSiluKernel.cu, compiled at run time by NVRTC for the
// one (dtype, order, layout, mode) combination a call needs. See that file's header for the
// contract; the host side (output allocation, layout choice, grid) lives in jit_norm.py.

#pragma once

#include "packed_vector.cuh"
#include "vae_norm_common.cuh"
#include "vae_wino_tables.cuh"
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace vaewino
{

using cuda::std::int64_t;
using common::PackedAligned;
using namespace vae_norm;
using namespace vae_wino;

struct VaeWinoNormParams
{
    int64_t positions;      // P
    int channels;           // C (multiple of 8, <= kVaeRmsNormSiluMaxChannels)
    int frames;             // T, a positive multiple of order
    int order;              // R: 2 or 4
    bool halfSlabs;         // d and m are fp16 (else bf16, order 2 only)
    int cacheFrames;        // 0, 1 or 2
    float scale;            // sqrt(C)
    float eps;              // 1e-12
    bool referenceRounding; // reproduce the eager rounding sequence
};

constexpr int kVaeWinoMaxComponents = 6;

struct VaeWinoNormPointers
{
    void const* x;                        // null when m is used
    void const* m[kVaeWinoMaxComponents]; // null when x is used
    void const* residual;                 // optional
    void const* inBias;                   // optional
    void const* cache;                    // optional when cacheFrames == 0
    void const* gamma;
    void const* bias;                     // optional
    void* sum;                            // optional; required with a residual
    void* yTail;
    void* d[kVaeWinoMaxComponents];
};

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// T: activations (x, residual, cache, sum, yTail, gamma, bias); TD: slabs d and products m,
// each one stacked buffer [components, groups, P, C] with compStride elements per component.
template <typename T, typename TD>
struct Ptrs
{
    T const* x;
    TD const* m;
    T const* residual;
    T const* inBias;
    T const* cache;
    T const* gamma;
    T const* bias;
    T* sum;
    T* yTail;
    TD* d;
    int64_t compStride;
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
// inputs in flight; the frame history lives in shared memory. Order 4 holds up to five
// products per slice and emits five slabs from one frame: at the order-2 budgets it spills
// (measured: 11-19 M local loads per launch, DRAM 44-62%), so it runs two blocks per SM.
template <int K, bool IN_WINO, int R>
__host__ __device__ constexpr int blocksPerSm()
{
    if (K >= 4 || R == 4)
    {
        return 2;                        // ~110 registers; bytes in flight come from the deeper slice pipeline
    }
    return (!IN_WINO && K <= 2) ? 4 : 3; // producers hold one input per slice and are latency-bound: more warps
}

// Vector slices of one frame kept in flight ahead of the one being normalized.
template <int K>
__host__ __device__ constexpr int sliceDepth()
{
    return K >= 4 ? 3 : 2;
}

// History ring: a group needs frames t-2 .. t+R-1; frame t-2 is dead once the slabs that
// read it are out (while frame t+R-2 is normalized), so frame t+R-1 can take its slot:
// R + 1 slots suffice. slot(frame) = (frame + 2) mod (R + 1); the cached frames -2, -1
// start in slots 0 and 1.
template <int R>
__host__ __device__ constexpr int ringSlots()
{
    return R + 1;
}

template <int R>
__host__ __device__ __forceinline__ int ringSlot(int frame)
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
__device__ __forceinline__ void vaeWinoNormSiluBody(Ptrs<T, TD> const& ptrs, VaeWinoNormParams const& p)
{
    using Vec = PackedAligned<T, V>;
    using VecD = PackedAligned<TD, V>;
    using VecIn = cuda::std::conditional_t<IN_WINO, VecD, Vec>;
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
        using E = cuda::std::remove_cv_t<cuda::std::remove_pointer_t<decltype(rowBase)>>;
        return *reinterpret_cast<PackedAligned<E, V> const*>(rowBase + k * L * V);
    };
    auto storeVec = [&](auto* rowBase, int k, auto const& v) {
        *reinterpret_cast<cuda::std::remove_cv_t<cuda::std::remove_reference_t<decltype(v)>>*>(rowBase + k * L * V) = v;
    };

    // Row pointers of one frame's inputs: the x row, or the row inside component 0 of the
    // products (component k of a term sits compStride elements further; the term indices
    // are compile-time constants, so each load is one base plus a constant multiple).
    struct FrameSrc
    {
        cuda::std::conditional_t<IN_WINO, TD, T> const* in;
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
            (void) jTag;
            FrameSrc f{};
            int64_t const frow = static_cast<int64_t>(t) * P + pos;
            if constexpr (IN_WINO)
            {
                f.in = gRow(ptrs.m, static_cast<int64_t>(t / R) * P + pos);
            }
            else
            {
                f.in = gRow(ptrs.x, frow);
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
        // Loads slice k of frame J's inputs: the x row, or the products of the terms of AT row J.
        auto loadSlice = [&](auto jTag, VecIn(&dst)[NUM_IN], Vec& dres, FrameSrc const& f, int k)
        {
            constexpr int J = decltype(jTag)::value;
            if constexpr (IN_WINO)
            {
#pragma unroll
                for (int j = 0; j < numTerms<R>(J); ++j)
                {
                    dst[j] = loadVec(f.in + termIndex<R>(J, j) * ptrs.compStride, k);
                }
            }
            else
            {
                dst[0] = loadVec(f.in, k);
            }
            if constexpr (HAS_RESIDUAL)
            {
                dres = loadVec(f.res, k);
            }
        };

        // Cross-frame prefetch (sliceDepth 2, i.e. K <= 3): slices 0 and 1 of the next frame
        // are loaded while this frame is normalized, so a frame's load latency hides behind
        // the previous frame's pass 2. The first frame of a position loads its own.
        constexpr bool XF_PF = !FRAME_PF && sliceDepth<K>() == 2;
        FrameSrc src = srcOf(cuda::std::integral_constant<int, 0>{}, 0);
        if constexpr (FRAME_PF || XF_PF)
        {
            loadSlice(cuda::std::integral_constant<int, 0>{}, inA, resA, src, 0);
        }
        if constexpr (XF_PF && K > 1)
        {
            loadSlice(cuda::std::integral_constant<int, 0>{}, inB, resB, src, 1);
        }

        // One frame: J is its index within the group (compile time), t its frame number.
        // For FRAME_PF the inputs of frame J sit in buffer J % 2.
        auto frame = [&](auto jTag, int t)
        {
            constexpr int J = decltype(jTag)::value;
            constexpr int JN = (J + 1) % R;
            constexpr int fb = FRAME_PF ? (J % 2) : 0;
            constexpr auto termsIn = cuda::std::integral_constant<int, J>{};
            constexpr auto termsNext = cuda::std::integral_constant<int, JN>{};
            FrameSrc next{};
            if constexpr (FRAME_PF)
            {
                if (t + 1 < T_)
                {
                    next = srcOf(cuda::std::integral_constant<int, JN>{}, t + 1);
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
            else if constexpr (!XF_PF)
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
                // Slices in pairs: load the next slice into the other buffer, then compute
                // (with the cross-frame prefetch, slices 0 and 1 are already in).
#pragma unroll
                for (int k = 0; k < K; k += 2)
                {
                    if (k + 1 < K && !(XF_PF && k == 0))
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
            if constexpr (XF_PF)
            {
                if (t + 1 < T_)
                {
                    next = srcOf(termsNext, t + 1);
                    loadSlice(termsNext, inA, resA, next, 0);
                    if constexpr (K > 1)
                    {
                        loadSlice(termsNext, inB, resB, next, 1);
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
                auto emit = [&](auto sTag)
                {
                    constexpr int s = decltype(sTag)::value;
                    if constexpr (s < W::kComps && lastFrameOf<R>(s) == J + 2)
                    {
                        // Each slab reloads the ring frames it reads (the barrier stops the
                        // compiler from hoisting them across slabs): a few extra LDS keep the
                        // live set small enough not to spill.
                        asm volatile("" ::: "memory");
                        Vec hs[J + 2];
#pragma unroll
                        for (int fi = 0; fi < J + 2; ++fi)
                        {
                            if (W::bt(s, fi) != 0.0f)
                            {
                                hs[fi] = loadVec(slotPtr(ringSlot<R>(t0 - 2 + fi)), k);
                            }
                        }
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
                                        = (fi == J + 2) ? pairAt(yk, i) : pairAt(hs[fi < J + 2 ? fi : 0], i);
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
                        storeVec(gRowW(ptrs.d + s * ptrs.compStride, drow), k, o);
                    }
                };
                emit(cuda::std::integral_constant<int, 0>{});
                emit(cuda::std::integral_constant<int, 1>{});
                emit(cuda::std::integral_constant<int, 2>{});
                emit(cuda::std::integral_constant<int, 3>{});
                emit(cuda::std::integral_constant<int, 4>{});
                emit(cuda::std::integral_constant<int, 5>{});
            }
            if constexpr (FRAME_PF || XF_PF)
            {
                src = next;
            }
            else
            {
                if (t + 1 < T_)
                {
                    src = srcOf(cuda::std::integral_constant<int, JN>{}, t + 1);
                }
            }
        };

        for (int t = 0; t < T_; t += R)
        {
            frame(cuda::std::integral_constant<int, 0>{}, t);
            frame(cuda::std::integral_constant<int, 1>{}, t + 1);
            if constexpr (R == 4)
            {
                frame(cuda::std::integral_constant<int, 2>{}, t + 2);
                frame(cuda::std::integral_constant<int, 3>{}, t + 3);
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

// Entry point for the runtime-compiled kernel: every argument a scalar (the launcher passes a
// typed argument list), the products and the slabs as one stacked buffer each with
// ``compStride`` elements between components.
template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL, bool IN_WINO>
__global__ void __launch_bounds__(kThreads, blocksPerSm<K, IN_WINO, R>()) vaeWinoNormSiluKernel(T const* x, TD const* m,
    TD* d, long long compStride, T const* residual, T const* inBias, T const* cache, T const* gamma, T const* bias,
    T* sum, T* yTail, long long positions, int channels, int frames, int cacheFrames, float scale, float eps)
{
    Ptrs<T, TD> ptrs{};
    ptrs.x = x;
    ptrs.m = m;
    ptrs.d = d;
    ptrs.compStride = compStride;
    ptrs.residual = residual;
    ptrs.inBias = inBias;
    ptrs.cache = cache;
    ptrs.gamma = gamma;
    ptrs.bias = bias;
    ptrs.sum = sum;
    ptrs.yTail = yTail;
    VaeWinoNormParams p{};
    p.positions = positions;
    p.channels = channels;
    p.frames = frames;
    p.order = R;
    p.halfSlabs = sizeof(TD) == 2 && !cuda::std::is_same_v<TD, __nv_bfloat16>;
    p.cacheFrames = cacheFrames;
    p.scale = scale;
    p.eps = eps;
    p.referenceRounding = REF;
    vaeWinoNormSiluBody<T, TD, R, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, IN_WINO>(ptrs, p);
}

} // namespace vaewino
