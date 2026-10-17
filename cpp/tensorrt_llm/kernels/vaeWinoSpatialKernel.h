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

// Spatial Winograd F(2x2, 3x3) transforms for channels-last bf16 frames; the transformed
// data (V, M) is fp16, which keeps three more mantissa bits than bf16 through the GEMMs.
//
// A frame [H, W, C] is covered by ceil(H/2) x ceil(W/2) output tiles of 2x2 pixels; tile
// (i, j) is computed from the 4x4 input window at rows 2i-1..2i+2, cols 2j-1..2j+2 (zero
// outside the frame, the conv's padding of 1; output pixels past the frame are dropped).
// The input transform writes, for a range of tile rows [row0, row1) of every frame,
// V[coord][frame, tile][C] = B^T d B; the GEMMs produce M in the same layout; the output
// transform writes y[frame, 2i+a, 2j+b, C] = A^T M A.
//
// The N frames form `components` groups of N / components frames (temporal Winograd
// components); V and M are [components * 16, (N / components) * ntilesStrip, C] with
// ntilesStrip = (row1 - row0) * ceil(W/2) and tile index = pair * ntilesStrip + (i - row0) * ceil(W/2) + j.
struct VaeWinoSpatialParams
{
    int frames;     // N
    int height;     // H
    int width;      // W
    int channels;
    int components; // temporal components; frame n belongs to component n / (N / components)
    int row0;       // first tile row of the strip
    int row1;       // one past the last tile row
};

void launchVaeWinoSpatialInput(void const* frames, void* v, VaeWinoSpatialParams const& p, cudaStream_t stream);
void launchVaeWinoSpatialOutput(void const* m, void* frames, VaeWinoSpatialParams const& p, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
