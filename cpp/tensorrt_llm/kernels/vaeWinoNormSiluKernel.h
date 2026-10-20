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
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// RMSNorm + SiLU that also performs the temporal Winograd F(R,3) transforms around a
// causal 3x3x3 convolution, R = 2 or 4 output frames per group.
//
// Frames of a batch-1 channels-last video are [T, P, C] (frame, position, channel),
// rows contiguous. Output frames come in groups of R; group g uses the R + 2 normalized
// frames f_i = frame g*R - 2 + i (i = 0..R+1), where frames -2 and -1 are the cache
// (zeros where the cache is shorter), and writes the R + 2 slabs
//   D_k = sum_i BT[k][i] f_i          (BT: the F(R,3) input transform),
// each D_k as [T/R, P, C] in the slab dtype (bf16 or fp16). The R + 2 conv2d results
// m_0..m_{R+1} of such a conv feed the next norm: frame g*R + j is sum_k AT[j][k] m_k,
// summed in index order, which this kernel reads directly when ``m`` is given instead
// of ``x``.
//   R = 2: BT = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]], AT = [[1,1,1,0],[0,1,-1,-1]].
//   R = 4: the F(4,3) transforms on the points (0, 1, -1, 2, -1/2, inf); every AT entry
//          is a power of two, every BT entry a multiple of 1/2, so all products of 16-bit
//          inputs are exact in fp32 and the sums round once.
//
// Inputs:  x [T, P, C] or m[R+2] each [T/R, P, C]; residual [T, P, C] (optional);
//          inBias [C] (optional; deferred conv bias, added first); cache
//          [cacheFrames, P, C] of normalized frames; gamma [C]; bias [C] (optional).
// Outputs: d[R+2] each [T/R, P, C]; yTail [2, P, C] (the last two normalized frames,
//          the next chunk's cache); sum [T, P, C]: the pre-norm frames (combine + bias
//          + residual), written whenever the pointer is set.
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

void launchVaeWinoNormSilu(VaeWinoNormPointers const& ptrs, VaeWinoNormParams const& p, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
