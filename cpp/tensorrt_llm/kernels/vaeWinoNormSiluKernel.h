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

// RMSNorm + SiLU that also performs the temporal Winograd F(2,3) transforms around a
// causal 3x3x3 convolution.
//
// Frames of a batch-1 channels-last video are [T, P, C] (frame, position, channel),
// rows contiguous. Output frames come in pairs; pair p uses the normalized frames
// (f0, f1, f2, f3) = (2p-2, 2p-1, 2p, 2p+1), where frames -2 and -1 are the cache
// (zeros where the cache is shorter), and writes
//   D0 = f0 - f2, D1 = f1 + f2, D2 = f2 - f1, D3 = f1 - f3,
// each D_k as [T/2, P, C]. The four conv2d results m0..m3 of such a conv feed the
// next norm: frame 2p is m0 + m1 + m2 and frame 2p+1 is m1 - m2 - m3 (summed in that
// order), which this kernel reads directly when ``m`` is given instead of ``x``.
//
// Inputs:  x [T, P, C] or m[4] each [T/2, P, C]; residual [T, P, C] (optional);
//          inBias [C] (optional; deferred conv bias, added first); cache
//          [cacheFrames, P, C] of normalized frames; gamma [C]; bias [C] (optional).
// Outputs: d[4] each [T/2, P, C]; yTail [2, P, C] (the last two normalized frames,
//          the next chunk's cache); sum [T, P, C] when a residual is given.
struct VaeWinoNormParams
{
    int64_t positions;      // P
    int channels;           // C (multiple of 8, <= kVaeRmsNormSiluMaxChannels)
    int frames;             // T, even and >= 2
    int cacheFrames;        // 0, 1 or 2
    float scale;            // sqrt(C)
    float eps;              // 1e-12
    bool referenceRounding; // reproduce the eager rounding sequence
};

struct VaeWinoNormPointers
{
    void const* x;           // null when m is used
    void const* m[4];        // null when x is used
    void const* residual;    // optional
    void const* inBias;      // optional
    void const* cache;       // optional when cacheFrames == 0
    void const* gamma;
    void const* bias;        // optional
    void* sum;               // required iff residual
    void* yTail;
    void* d[4];
};

void launchVaeWinoNormSilu(VaeWinoNormPointers const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
