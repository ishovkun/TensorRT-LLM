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

#ifndef TRTLLM_WANVAENORMSILUKERNEL_H
#define TRTLLM_WANVAENORMSILUKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// The glue between two convolutions of the Wan VAE, in one pass over channels-last bf16 video.
//
// Per pixel row (C contiguous channels), in fp32:
//   h = x + bias_x + residual + bias_res        (every term after x optional)
//   y = silu(h * rsqrt(mean(h^2) + eps) * gamma)
// h is the convolution output with its bias folded in and the block's shortcut added (the residual
// stream the next block reads); y is the next convolution's input. Either output may be skipped:
// h == nullptr computes y only, y == nullptr computes h only (a vectorized add with per-channel biases).
// h may alias x.
//
// x, residual, h: [batch * rows_per_batch, channels] contiguous.
// y: row r of batch b lives at y + b * y_batch_stride + r * channels, so the output can land inside a
//    larger buffer (the causal convolution's temporally padded input) without a copy.
// channels: multiple of 8, at most 1024.
struct WanVaeNormSiluParams
{
    __nv_bfloat16 const* x;
    __nv_bfloat16 const* residual;
    __nv_bfloat16 const* bias_x;
    __nv_bfloat16 const* bias_res;
    __nv_bfloat16 const* gamma;
    __nv_bfloat16* h;
    __nv_bfloat16* y;
    int64_t rows_per_batch;
    int64_t y_batch_stride;
    int batch;
    int channels;
    float eps;
};

void launchWanVaeNormSilu(WanVaeNormSiluParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_WANVAENORMSILUKERNEL_H
