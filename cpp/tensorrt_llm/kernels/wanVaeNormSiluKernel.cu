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
#include "tensorrt_llm/common/tllmException.h"
#include "wanVaeNormSiluKernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kRowsPerWarp = 4;
constexpr int kElemsPerVec = 8; // one uint4 of bf16

struct Vec8
{
    float v[kElemsPerVec];
};

__device__ __forceinline__ Vec8 toFloat(uint4 const& raw)
{
    Vec8 out;
    __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
    for (int i = 0; i < kElemsPerVec / 2; ++i)
    {
        float2 const f = __bfloat1622float2(pairs[i]);
        out.v[2 * i] = f.x;
        out.v[2 * i + 1] = f.y;
    }
    return out;
}

__device__ __forceinline__ uint4 toBf16(Vec8 const& vals)
{
    uint4 raw;
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < kElemsPerVec / 2; ++i)
    {
        pairs[i] = __float22bfloat162_rn(make_float2(vals.v[2 * i], vals.v[2 * i + 1]));
    }
    return raw;
}

__device__ __forceinline__ uint4 loadVec(__nv_bfloat16 const* ptr)
{
    return *reinterpret_cast<uint4 const*>(ptr);
}

__device__ __forceinline__ void storeVec(__nv_bfloat16* ptr, uint4 const& raw)
{
    *reinterpret_cast<uint4*>(ptr) = raw;
}

__device__ __forceinline__ Vec8 zeros()
{
    Vec8 out;
#pragma unroll
    for (int i = 0; i < kElemsPerVec; ++i)
    {
        out.v[i] = 0.0f;
    }
    return out;
}

// One warp per pixel row; lane l owns channel vectors l, l + 32, ... (at most MAX_VEC of them).
// A warp works on ROWS rows at once so that ROWS * MAX_VEC independent 16-byte loads are in flight
// per lane before the first reduction; ROWS * MAX_VEC is kept at 4 so the register footprint is the
// same for every channel width. The per-channel vectors are loaded once per warp.
template <int MAX_VEC, bool WRITE_Y>
__global__ void __launch_bounds__(kWarpsPerBlock* kWarpSize) wanVaeNormSiluKernel(WanVaeNormSiluParams p)
{
    constexpr int ROWS = 4 / MAX_VEC;
    static_assert(ROWS * MAX_VEC == 4 && kRowsPerWarp % ROWS == 0, "unsupported tile");
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif
    int const lane = threadIdx.x % kWarpSize;
    int const warp = threadIdx.x / kWarpSize;
    int const nvec = p.channels / kElemsPerVec;
    int64_t const total_rows = p.rows_per_batch * p.batch;
    int64_t const first_row = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * kRowsPerWarp;
    if (first_row >= total_rows)
    {
        return;
    }

    Vec8 bias[MAX_VEC];
    Vec8 gamma[MAX_VEC];
#pragma unroll
    for (int i = 0; i < MAX_VEC; ++i)
    {
        int const vec = lane + i * kWarpSize;
        bias[i] = zeros();
        gamma[i] = zeros();
        if (vec < nvec)
        {
            int const c = vec * kElemsPerVec;
            if (p.bias_x != nullptr)
            {
                bias[i] = toFloat(loadVec(p.bias_x + c));
            }
            if (p.bias_res != nullptr)
            {
                Vec8 const b = toFloat(loadVec(p.bias_res + c));
#pragma unroll
                for (int k = 0; k < kElemsPerVec; ++k)
                {
                    bias[i].v[k] += b.v[k];
                }
            }
            if constexpr (WRITE_Y)
            {
                gamma[i] = toFloat(loadVec(p.gamma + c));
            }
        }
    }

    for (int r0 = 0; r0 < kRowsPerWarp; r0 += ROWS)
    {
        uint4 xr[ROWS][MAX_VEC];
        uint4 rr[ROWS][MAX_VEC];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            int64_t const row = first_row + r0 + r;
            bool const live = row < total_rows;
#pragma unroll
            for (int i = 0; i < MAX_VEC; ++i)
            {
                int const vec = lane + i * kWarpSize;
                xr[r][i] = make_uint4(0, 0, 0, 0);
                rr[r][i] = make_uint4(0, 0, 0, 0);
                if (live && vec < nvec)
                {
                    int64_t const off = row * p.channels + vec * kElemsPerVec;
                    xr[r][i] = loadVec(p.x + off);
                    if (p.residual != nullptr)
                    {
                        rr[r][i] = loadVec(p.residual + off);
                    }
                }
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            int64_t const row = first_row + r0 + r;
            if (row >= total_rows)
            {
                break;
            }
            int64_t const row_base = row * p.channels;
            Vec8 h[MAX_VEC];
            float sumsq = 0.0f;
#pragma unroll
            for (int i = 0; i < MAX_VEC; ++i)
            {
                int const vec = lane + i * kWarpSize;
                h[i] = toFloat(xr[r][i]);
                if (p.residual != nullptr)
                {
                    Vec8 const res = toFloat(rr[r][i]);
#pragma unroll
                    for (int k = 0; k < kElemsPerVec; ++k)
                    {
                        h[i].v[k] += res.v[k];
                    }
                }
#pragma unroll
                for (int k = 0; k < kElemsPerVec; ++k)
                {
                    h[i].v[k] += bias[i].v[k];
                    sumsq += h[i].v[k] * h[i].v[k];
                }
                if (p.h != nullptr && vec < nvec)
                {
                    storeVec(p.h + row_base + vec * kElemsPerVec, toBf16(h[i]));
                }
            }

            if constexpr (WRITE_Y)
            {
#pragma unroll
                for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
                {
                    sumsq += __shfl_xor_sync(0xffffffffu, sumsq, offset);
                }
                float const rcp = rsqrtf(sumsq / static_cast<float>(p.channels) + p.eps);
                int64_t const b = row / p.rows_per_batch;
                __nv_bfloat16* y_row = p.y + b * p.y_batch_stride + (row - b * p.rows_per_batch) * p.channels;
#pragma unroll
                for (int i = 0; i < MAX_VEC; ++i)
                {
                    int const vec = lane + i * kWarpSize;
                    if (vec < nvec)
                    {
                        Vec8 y;
#pragma unroll
                        for (int k = 0; k < kElemsPerVec; ++k)
                        {
                            float const n = h[i].v[k] * rcp * gamma[i].v[k];
                            y.v[k] = __fdividef(n, 1.0f + __expf(-n));
                        }
                        storeVec(y_row + vec * kElemsPerVec, toBf16(y));
                    }
                }
            }
        }
    }
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <int MAX_VEC>
void launchForWidth(WanVaeNormSiluParams const& p, cudaStream_t stream)
{
    int64_t const total_rows = p.rows_per_batch * p.batch;
    int64_t const rows_per_block = static_cast<int64_t>(kWarpsPerBlock) * kRowsPerWarp;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>((total_rows + rows_per_block - 1) / rows_per_block));
    cfg.blockDim = dim3(kWarpsPerBlock * kWarpSize);
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    if (p.y != nullptr)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, wanVaeNormSiluKernel<MAX_VEC, true>, p));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, wanVaeNormSiluKernel<MAX_VEC, false>, p));
    }
}

} // namespace

void launchWanVaeNormSilu(WanVaeNormSiluParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.channels % kElemsPerVec == 0 && params.channels >= kElemsPerVec,
        "wanVaeNormSilu: channels (%d) must be a positive multiple of %d", params.channels, kElemsPerVec);
    TLLM_CHECK_WITH_INFO(params.h != nullptr || params.y != nullptr, "wanVaeNormSilu: nothing to write");
    TLLM_CHECK_WITH_INFO(params.y == nullptr || params.gamma != nullptr, "wanVaeNormSilu: y needs gamma");
    if (params.rows_per_batch * params.batch == 0)
    {
        return;
    }
    int const nvec = params.channels / kElemsPerVec;
    if (nvec <= kWarpSize)
    {
        launchForWidth<1>(params, stream);
    }
    else if (nvec <= 2 * kWarpSize)
    {
        launchForWidth<2>(params, stream);
    }
    else if (nvec <= 4 * kWarpSize)
    {
        launchForWidth<4>(params, stream);
    }
    else
    {
        TLLM_THROW("wanVaeNormSilu: channels (%d) must be at most %d", params.channels, 4 * kWarpSize * kElemsPerVec);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
