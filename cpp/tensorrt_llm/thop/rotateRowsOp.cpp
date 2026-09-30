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

#include "tensorrt_llm/kernels/rotateRows.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <torch/extension.h>

namespace torch_ext
{

//! In place: every row of ``self`` becomes ``torch.roll(row, -shift)``, i.e. rotated left by
//! ``shift`` (any integer; normalized modulo the row length). ``self`` is 1-D or 2-D with a
//! unit-stride last dimension; rows may be strided; any dtype (the rotation permutes bytes).
//! No scratch memory, no allocation, safe inside CUDA graph capture.
torch::Tensor rotate_rows_left_(torch::Tensor self, int64_t shift)
{
    CHECK_TH_CUDA(self);
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "rotate_rows_left_: expected a 1-D or 2-D tensor");
    TORCH_CHECK(self.stride(-1) == 1, "rotate_rows_left_: the last dimension must be contiguous");
    int64_t const elemSize = self.element_size();
    TORCH_CHECK(elemSize == 1 || elemSize == 2 || elemSize == 4 || elemSize == 8 || elemSize == 16,
        "rotate_rows_left_: unsupported element size");

    int64_t const rows = self.dim() == 2 ? self.size(0) : 1;
    int64_t const cols = self.size(-1);
    int64_t const rowStride = self.dim() == 2 ? self.stride(0) : cols;
    if (rows == 0 || cols <= 1)
    {
        return self;
    }
    int64_t const normalized = ((shift % cols) + cols) % cols;
    if (normalized == 0)
    {
        return self;
    }
    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());
    tensorrt_llm::kernels::invokeRotateRowsLeft(
        self.data_ptr(), rows, cols, rowStride, normalized, static_cast<int>(elemSize), stream);
    return self;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("rotate_rows_left_(Tensor(a!) self, int shift) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("rotate_rows_left_", &torch_ext::rotate_rows_left_);
}
