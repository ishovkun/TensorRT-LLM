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

#include "tensorrt_llm/kernels/wanVaeUpsampleKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// x: [batch, channels, height, width] in channels_last memory format, 16-bit dtype.
// Returns the 2x nearest upsample in the same layout.
torch::Tensor wan_vae_upsample2x(torch::Tensor const& x)
{
    TORCH_CHECK(x.is_cuda() && x.dim() == 4, "x must be a 4-D CUDA tensor");
    TORCH_CHECK(x.element_size() == 2, "x must have a 16-bit dtype");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "x must be channels_last");
    int64_t const batch = x.size(0);
    int64_t const channels = x.size(1);
    int64_t const height = x.size(2);
    int64_t const width = x.size(3);
    TORCH_CHECK(channels % 8 == 0, "channels must be a multiple of 8, got ", channels);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");

    auto out = torch::empty({batch, channels, 2 * height, 2 * width}, x.options(), at::MemoryFormat::ChannelsLast);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    tensorrt_llm::kernels::launchWanVaeUpsample2x(x.data_ptr(), out.data_ptr(), batch, height, width, channels, stream);
    return out;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("wan_vae_upsample2x(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("wan_vae_upsample2x", &wan_vae_upsample2x);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
