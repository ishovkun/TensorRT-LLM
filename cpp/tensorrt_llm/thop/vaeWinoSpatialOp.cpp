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

#include "tensorrt_llm/kernels/vaeWinoSpatialKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
tensorrt_llm::kernels::VaeWinoSpatialParams params(
    torch::Tensor const& frames, int64_t components, int64_t row0, int64_t row1)
{
    TORCH_CHECK(frames.dim() == 4, "frames must be [N, H, W, C]");
    CHECK_INPUT(frames, torch::kBFloat16);
    tensorrt_llm::kernels::VaeWinoSpatialParams p{};
    p.frames = static_cast<int>(frames.size(0));
    p.height = static_cast<int>(frames.size(1));
    p.width = static_cast<int>(frames.size(2));
    p.channels = static_cast<int>(frames.size(3));
    p.components = static_cast<int>(components);
    p.row0 = static_cast<int>(row0);
    p.row1 = static_cast<int>(row1);
    return p;
}
} // namespace

// V = B^T d B for tile rows [row0, row1): frames [N, H, W, C] bf16 -> fp16 [components*16,
// (N/components)*ntiles, C].
torch::Tensor vae_wino_spatial_in(torch::Tensor const& frames, int64_t components, int64_t row0, int64_t row1)
{
    auto const p = params(frames, components, row0, row1);
    int64_t const ntiles = static_cast<int64_t>(p.row1 - p.row0) * ((p.width + 1) / 2);
    auto v = torch::empty(
        {components * 16, (p.frames / components) * ntiles, p.channels}, frames.options().dtype(torch::kHalf));
    tensorrt_llm::kernels::launchVaeWinoSpatialInput(
        frames.data_ptr(), v.data_ptr(), p, at::cuda::getCurrentCUDAStream(frames.get_device()));
    return v;
}

// y[:, 2i+a, 2j+b, :] = A^T M A for tile rows [row0, row1); writes into ``frames`` and returns it.
torch::Tensor vae_wino_spatial_out(
    torch::Tensor const& m, torch::Tensor const& frames, int64_t components, int64_t row0, int64_t row1)
{
    auto const p = params(frames, components, row0, row1);
    CHECK_INPUT(m, torch::kHalf);
    int64_t const ntiles = static_cast<int64_t>(p.row1 - p.row0) * ((p.width + 1) / 2);
    TORCH_CHECK(m.dim() == 3 && m.size(0) == components * 16 && m.size(1) == (p.frames / components) * ntiles
            && m.size(2) == p.channels,
        "m must be [components*16, (N/components)*ntiles, C]");
    tensorrt_llm::kernels::launchVaeWinoSpatialOutput(
        m.data_ptr(), frames.data_ptr(), p, at::cuda::getCurrentCUDAStream(frames.get_device()));
    return frames;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("vae_wino_spatial_in(Tensor frames, int components, int row0, int row1) -> Tensor");
    m.def("vae_wino_spatial_out(Tensor m, Tensor(a!) frames, int components, int row0, int row1) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("vae_wino_spatial_in", &vae_wino_spatial_in);
    m.impl("vae_wino_spatial_out", &vae_wino_spatial_out);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
