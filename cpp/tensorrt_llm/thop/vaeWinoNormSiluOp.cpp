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
#include "tensorrt_llm/kernels/vaeWinoNormSiluKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
void const* optionalPtr(std::optional<torch::Tensor> const& t, c10::IntArrayRef shape, char const* name)
{
    if (!t.has_value())
    {
        return nullptr;
    }
    CHECK_INPUT(t.value(), torch::kBFloat16);
    TORCH_CHECK(t->sizes() == shape, name, " has the wrong shape");
    return t->data_ptr();
}
} // namespace

// RMSNorm + SiLU with the temporal Winograd transforms (see vaeWinoNormSiluKernel.h).
// x: [T, P, C] frames, or m: four [T/2, P, C] tensors from the previous Winograd conv.
// Returns (sum [T, P, C] (the pre-norm frames; empty when the input is a plain x with
// neither bias nor residual), y_tail [2, P, C], d: four [T/2, P, C]).
std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> vae_wino_norm_silu(
    std::optional<torch::Tensor> const& x, std::optional<std::vector<torch::Tensor>> const& m,
    std::optional<torch::Tensor> const& residual, std::optional<torch::Tensor> const& in_bias,
    std::optional<torch::Tensor> const& cache, torch::Tensor const& gamma, std::optional<torch::Tensor> const& bias,
    double scale, bool reference_rounding, int64_t frames)
{
    TORCH_CHECK(x.has_value() != m.has_value(), "exactly one of x and m");
    TORCH_CHECK(frames >= 2 && frames % 2 == 0, "frames must be even and >= 2");
    torch::Tensor const& shapeSrc = x.has_value() ? x.value() : m->at(0);
    TORCH_CHECK(shapeSrc.dim() == 3, "frame tensors must be [T, P, C]");
    CHECK_INPUT(shapeSrc, torch::kBFloat16);
    int64_t const P = shapeSrc.size(1);
    int64_t const C = shapeSrc.size(2);
    TORCH_CHECK(C % 8 == 0 && C <= tensorrt_llm::kernels::kVaeRmsNormSiluMaxChannels, "unsupported channel count ", C);
    CHECK_INPUT(gamma, torch::kBFloat16);
    TORCH_CHECK(gamma.numel() == C, "gamma must have one value per channel");

    tensorrt_llm::kernels::VaeWinoNormPointers ptrs{};
    tensorrt_llm::kernels::VaeWinoNormParams p{};
    p.positions = P;
    p.channels = static_cast<int>(C);
    p.frames = static_cast<int>(frames);
    p.scale = static_cast<float>(scale);
    p.eps = 1e-12f;
    p.referenceRounding = reference_rounding;

    std::vector<int64_t> const frameShape{frames, P, C};
    std::vector<int64_t> const pairShape{frames / 2, P, C};
    if (x.has_value())
    {
        TORCH_CHECK(x->sizes() == c10::IntArrayRef(frameShape), "x must be [T, P, C]");
        ptrs.x = x->data_ptr();
    }
    else
    {
        TORCH_CHECK(m->size() == 4, "m must hold four tensors");
        for (int k = 0; k < 4; ++k)
        {
            CHECK_INPUT(m->at(k), torch::kBFloat16);
            TORCH_CHECK(m->at(k).sizes() == c10::IntArrayRef(pairShape), "m[k] must be [T/2, P, C]");
            ptrs.m[k] = m->at(k).data_ptr();
        }
    }
    ptrs.residual = optionalPtr(residual, frameShape, "residual");
    ptrs.inBias = optionalPtr(in_bias, {C}, "in_bias");
    ptrs.bias = optionalPtr(bias, {C}, "bias");
    ptrs.gamma = gamma.data_ptr();
    p.cacheFrames = 0;
    if (cache.has_value() && cache->numel() > 0)
    {
        CHECK_INPUT(cache.value(), torch::kBFloat16);
        TORCH_CHECK(cache->dim() == 3 && cache->size(1) == P && cache->size(2) == C && cache->size(0) <= 2,
            "cache must be [0..2, P, C]");
        p.cacheFrames = static_cast<int>(cache->size(0));
        ptrs.cache = cache->data_ptr();
    }

    auto opts = shapeSrc.options();
    bool const wantSum = residual.has_value() || in_bias.has_value() || !x.has_value();
    torch::Tensor sum = wantSum ? torch::empty(frameShape, opts) : torch::empty({0}, opts);
    torch::Tensor yTail = torch::empty({2, P, C}, opts);
    std::vector<torch::Tensor> d;
    for (int k = 0; k < 4; ++k)
    {
        d.push_back(torch::empty(pairShape, opts));
        ptrs.d[k] = d.back().data_ptr();
    }
    ptrs.sum = wantSum ? sum.data_ptr() : nullptr;
    ptrs.yTail = yTail.data_ptr();

    auto stream = at::cuda::getCurrentCUDAStream(shapeSrc.get_device());
    tensorrt_llm::kernels::launchVaeWinoNormSilu(ptrs, p, stream);
    return {sum, yTail, d};
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "vae_wino_norm_silu(Tensor? x, Tensor[]? m, Tensor? residual, Tensor? in_bias, Tensor? cache, Tensor gamma, "
        "Tensor? bias, float scale, bool reference_rounding, int frames) -> (Tensor, Tensor, Tensor[])");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("vae_wino_norm_silu", &vae_wino_norm_silu);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
