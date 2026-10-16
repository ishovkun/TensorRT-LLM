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

#include "tensorrt_llm/kernels/vaeRmsNormSiluKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// SiLU(RMSNorm(x)) for the Wan VAE: x is a channels-last activation viewed as
// [rows, channels] (rows contiguous). gamma/bias are the module parameters flattened
// to [channels]. in_bias is a deferred convolution bias added to x first. See
// vaeRmsNormSiluKernel.h for the two numerical modes.
namespace
{
void const* channelVectorPtr(std::optional<torch::Tensor> const& t, int64_t channels, char const* name)
{
    if (!t.has_value())
    {
        return nullptr;
    }
    CHECK_INPUT(t.value(), torch::kBFloat16);
    TORCH_CHECK(t->numel() == channels, name, " must have one value per channel");
    return t->data_ptr();
}
} // namespace

torch::Tensor vae_rms_norm_silu(torch::Tensor const& x, torch::Tensor const& gamma,
    std::optional<torch::Tensor> const& bias, double scale, bool reference_rounding,
    std::optional<torch::Tensor> const& in_bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, channels]");
    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_INPUT(gamma, torch::kBFloat16);
    int64_t const channels = x.size(1);
    TORCH_CHECK(gamma.numel() == channels, "gamma must have one value per channel");
    void const* biasPtr = channelVectorPtr(bias, channels, "bias");
    void const* inBiasPtr = channelVectorPtr(in_bias, channels, "in_bias");
    TORCH_CHECK(channels % 8 == 0 && channels <= tensorrt_llm::kernels::kVaeRmsNormSiluMaxChannels,
        "channels must be a multiple of 8 up to ", tensorrt_llm::kernels::kVaeRmsNormSiluMaxChannels, ", got ",
        channels);

    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    tensorrt_llm::kernels::launchVaeRmsNormSilu(x.data_ptr(), y.data_ptr(), gamma.data_ptr(), biasPtr, inBiasPtr,
        nullptr, nullptr, x.size(0), static_cast<int>(channels), static_cast<float>(scale), 1e-12f, reference_rounding,
        stream);
    return y;
}

// Residual-add variant: s = x + residual (rounded like an eager bf16 add) is normalized
// and also returned, since the next block uses it as its residual.
std::tuple<torch::Tensor, torch::Tensor> vae_add_rms_norm_silu(torch::Tensor const& x, torch::Tensor const& residual,
    torch::Tensor const& gamma, std::optional<torch::Tensor> const& bias, double scale, bool reference_rounding,
    std::optional<torch::Tensor> const& in_bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, channels]");
    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_INPUT(residual, torch::kBFloat16);
    CHECK_INPUT(gamma, torch::kBFloat16);
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual must match x");
    int64_t const channels = x.size(1);
    TORCH_CHECK(gamma.numel() == channels, "gamma must have one value per channel");
    void const* biasPtr = channelVectorPtr(bias, channels, "bias");
    void const* inBiasPtr = channelVectorPtr(in_bias, channels, "in_bias");
    TORCH_CHECK(channels % 8 == 0 && channels <= tensorrt_llm::kernels::kVaeRmsNormSiluMaxChannels,
        "channels must be a multiple of 8 up to ", tensorrt_llm::kernels::kVaeRmsNormSiluMaxChannels, ", got ",
        channels);

    auto sum = torch::empty_like(x);
    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    tensorrt_llm::kernels::launchVaeRmsNormSilu(x.data_ptr(), y.data_ptr(), gamma.data_ptr(), biasPtr, inBiasPtr,
        residual.data_ptr(), sum.data_ptr(), x.size(0), static_cast<int>(channels), static_cast<float>(scale), 1e-12f,
        reference_rounding, stream);
    return {sum, y};
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "vae_rms_norm_silu(Tensor x, Tensor gamma, Tensor? bias, float scale, bool reference_rounding, "
        "Tensor? in_bias=None) -> Tensor");
    m.def(
        "vae_add_rms_norm_silu(Tensor x, Tensor residual, Tensor gamma, Tensor? bias, float scale, "
        "bool reference_rounding, Tensor? in_bias=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("vae_rms_norm_silu", &vae_rms_norm_silu);
    m.impl("vae_add_rms_norm_silu", &vae_add_rms_norm_silu);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
