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

#include "tensorrt_llm/kernels/wanVaeNormSiluKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool aligned16(void const* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

__nv_bfloat16 const* channelVector(std::optional<torch::Tensor> const& t, int64_t channels, char const* name)
{
    if (!t.has_value())
    {
        return nullptr;
    }
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous(), name,
        " must be a contiguous bf16 CUDA tensor");
    TORCH_CHECK(t->dim() == 1 && t->size(0) == channels, name, " must have shape [channels]");
    TORCH_CHECK(aligned16(t->data_ptr()), name, " must be 16-byte aligned");
    return static_cast<__nv_bfloat16 const*>(t->data_ptr());
}

void checkRows(torch::Tensor const& t, torch::Tensor const& x, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), name,
        " must be a contiguous bf16 CUDA tensor");
    TORCH_CHECK(t.sizes() == x.sizes(), name, " must have the same shape as x");
    TORCH_CHECK(aligned16(t.data_ptr()), name, " must be 16-byte aligned");
}

} // namespace

// x: [batch, rows, channels] contiguous (a channels-last video viewed as pixel rows).
// y: [batch, rows, channels] with unit channel stride and row stride == channels; its batch
//    stride is free, so it may be a slice of a larger buffer.
void wan_vae_norm_silu(torch::Tensor const& x, std::optional<torch::Tensor> const& bias_x,
    std::optional<torch::Tensor> const& residual, std::optional<torch::Tensor> const& bias_res,
    std::optional<torch::Tensor> const& gamma, double eps, std::optional<torch::Tensor> const& h,
    std::optional<torch::Tensor> const& y)
{
    TORCH_CHECK(x.dim() == 3, "x must be [batch, rows, channels]");
    checkRows(x, x, "x");
    int64_t const batch = x.size(0);
    int64_t const rows = x.size(1);
    int64_t const channels = x.size(2);
    TORCH_CHECK(channels % 8 == 0 && channels > 0 && channels <= 1024,
        "channels must be a positive multiple of 8 and at most 1024, got ", channels);
    TORCH_CHECK(h.has_value() || y.has_value(), "at least one of h and y must be given");

    tensorrt_llm::kernels::WanVaeNormSiluParams params{};
    params.x = static_cast<__nv_bfloat16 const*>(x.data_ptr());
    params.bias_x = channelVector(bias_x, channels, "bias_x");
    params.bias_res = channelVector(bias_res, channels, "bias_res");
    params.gamma = channelVector(gamma, channels, "gamma");
    params.rows_per_batch = rows;
    params.batch = static_cast<int>(batch);
    params.channels = static_cast<int>(channels);
    params.eps = static_cast<float>(eps);
    if (residual.has_value())
    {
        checkRows(*residual, x, "residual");
        params.residual = static_cast<__nv_bfloat16 const*>(residual->data_ptr());
    }
    if (h.has_value())
    {
        checkRows(*h, x, "h");
        params.h = static_cast<__nv_bfloat16*>(h->data_ptr());
    }
    if (y.has_value())
    {
        TORCH_CHECK(gamma.has_value(), "y needs gamma");
        TORCH_CHECK(y->is_cuda() && y->scalar_type() == torch::kBFloat16, "y must be a bf16 CUDA tensor");
        TORCH_CHECK(y->dim() == 3 && y->size(0) == batch && y->size(1) == rows && y->size(2) == channels,
            "y must have the same shape as x");
        TORCH_CHECK(y->stride(2) == 1 && y->stride(1) == channels, "y rows must be contiguous");
        TORCH_CHECK(aligned16(y->data_ptr()) && (y->stride(0) * 2) % 16 == 0, "y must be 16-byte aligned");
        params.y = static_cast<__nv_bfloat16*>(y->data_ptr());
        params.y_batch_stride = y->stride(0);
    }

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    tensorrt_llm::kernels::launchWanVaeNormSilu(params, stream);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "wan_vae_norm_silu(Tensor x, Tensor? bias_x, Tensor? residual, Tensor? bias_res, Tensor? gamma, float eps, "
        "Tensor(a!)? h, Tensor(b!)? y) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("wan_vae_norm_silu", &wan_vae_norm_silu);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
