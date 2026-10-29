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
#include "wanVaeUpsampleKernel.h"
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kBlockSize = 256;
constexpr int kVecsPerThread = 4;

// One 16-byte channel vector per thread per iteration; a warp covers 32 consecutive vectors of one
// output row, i.e. the copy of one input row's slice to its 2x2 destinations is four coalesced writes.
__global__ void __launch_bounds__(kBlockSize)
    wanVaeUpsample2xKernel(uint4 const* __restrict__ in, uint4* __restrict__ out, int64_t const height,
        int64_t const width, int64_t const vecs_per_pixel, int64_t const total_out_vecs)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif
    int64_t const out_width = 2 * width;
    int64_t const out_row_vecs = out_width * vecs_per_pixel;
    int64_t const out_image_vecs = 2 * height * out_row_vecs;
    int64_t const stride = static_cast<int64_t>(gridDim.x) * kBlockSize;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * kBlockSize + threadIdx.x;
#pragma unroll
    for (int k = 0; k < kVecsPerThread; ++k, idx += stride)
    {
        if (idx >= total_out_vecs)
        {
            return;
        }
        int64_t const n = idx / out_image_vecs;
        int64_t const rem = idx - n * out_image_vecs;
        int64_t const oh = rem / out_row_vecs;
        int64_t const rem_row = rem - oh * out_row_vecs;
        int64_t const ow = rem_row / vecs_per_pixel;
        int64_t const vec = rem_row - ow * vecs_per_pixel;
        int64_t const src = ((n * height + (oh >> 1)) * width + (ow >> 1)) * vecs_per_pixel + vec;
        out[idx] = in[src];
    }
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

} // namespace

void launchWanVaeUpsample2x(
    void const* in, void* out, int64_t batch, int64_t height, int64_t width, int64_t channels, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        channels % 8 == 0, "wanVaeUpsample2x: channels (%ld) must be a multiple of 8", static_cast<long>(channels));
    int64_t const vecs_per_pixel = channels / 8;
    int64_t const total_out_vecs = batch * 4 * height * width * vecs_per_pixel;
    if (total_out_vecs == 0)
    {
        return;
    }
    int64_t const per_block = static_cast<int64_t>(kBlockSize) * kVecsPerThread;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>((total_out_vecs + per_block - 1) / per_block));
    cfg.blockDim = dim3(kBlockSize);
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, wanVaeUpsample2xKernel, static_cast<uint4 const*>(in),
        static_cast<uint4*>(out), height, width, vecs_per_pixel, total_out_vecs));
}

} // namespace kernels

TRTLLM_NAMESPACE_END
