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

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cuda_runtime.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Supported channel-vector lengths: multiples of 8 up to this bound.
constexpr int kVaeRmsNormSiluMaxChannels = 2048;

// y = SiLU(x / max(||x||_2, eps) * scale * gamma + bias), one channel vector per row.
//
// x, y:        bf16, [numRows, channels], rows contiguous (a channels-last video viewed as 2D).
// gamma, bias: bf16, [channels]; bias may be null.
// referenceRounding: reproduce the eager PyTorch sequence (round to bf16 after the
//                    normalize, after each multiply and after the bias add; SiLU on
//                    that value). Otherwise everything stays fp32 until the store.
// inBias:      optional bf16 [channels]: a convolution bias not yet applied to x. It is
//              added first as round_bf16(x + inBias), identical to the eager bias add.
// residual/sum: optional (both null or both set). With them, the normalized row is
// s = round_bf16(x + residual) and s is written to ``sum`` for the next block.
void launchVaeRmsNormSilu(void const* x, void* y, void const* gamma, void const* bias, void const* inBias,
    void const* residual, void* sum, int64_t numRows, int channels, float scale, float eps, bool referenceRounding,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
