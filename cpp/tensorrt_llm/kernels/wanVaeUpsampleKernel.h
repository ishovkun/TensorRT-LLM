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

#ifndef TRTLLM_WANVAEUPSAMPLEKERNEL_H
#define TRTLLM_WANVAEUPSAMPLEKERNEL_H

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// 2x nearest upsample of channels-last images: out[n, 2h + a, 2w + b, :] = in[n, h, w, :].
// in: [batch, height, width, channels] contiguous 16-bit elements; out: [batch, 2 * height, 2 * width, channels].
// channels must be a multiple of 8.
void launchWanVaeUpsample2x(
    void const* in, void* out, int64_t batch, int64_t height, int64_t width, int64_t channels, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_WANVAEUPSAMPLEKERNEL_H
