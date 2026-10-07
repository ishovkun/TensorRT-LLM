# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Wan VAE glue kernels: bias + shortcut + channel RMSNorm + SiLU in one pass, and the
channels-last 2x nearest upsample."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (loads the trtllm torch ops)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EPS = 1e-12


def reference(x, bias_x, residual, bias_res, gamma):
    h = x.float()
    if bias_x is not None:
        h = h + bias_x.float()
    if residual is not None:
        h = h + residual.float()
    if bias_res is not None:
        h = h + bias_res.float()
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()
    return h, torch.nn.functional.silu(n)


@pytest.mark.parametrize("channels", [96, 256, 512, 1024])
@pytest.mark.parametrize("with_residual", [False, True])
def test_norm_silu_matches_the_fp32_reference(channels, with_residual):
    torch.manual_seed(channels)
    batch, rows = 2, 301
    x = torch.randn(batch, rows, channels, device="cuda").to(torch.bfloat16)
    bias_x = torch.randn(channels, device="cuda").to(torch.bfloat16)
    gamma = (torch.rand(channels, device="cuda") * 2).to(torch.bfloat16)
    residual = bias_res = None
    if with_residual:
        residual = torch.randn(batch, rows, channels, device="cuda").to(torch.bfloat16)
        bias_res = torch.randn(channels, device="cuda").to(torch.bfloat16)
    h_ref, y_ref = reference(x, bias_x, residual, bias_res, gamma)

    # y lands inside a larger buffer: two padding rows in front of every batch's block.
    pad = 2
    buf = torch.zeros(batch, pad + rows, channels, device="cuda", dtype=torch.bfloat16)
    y = buf[:, pad:, :]
    h = torch.empty_like(x)
    torch.ops.trtllm.wan_vae_norm_silu(x, bias_x, residual, bias_res, gamma, EPS, h, y)
    torch.testing.assert_close(h.float(), h_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y.float(), y_ref, rtol=2e-2, atol=2e-2)
    assert buf[:, :pad, :].abs().sum().item() == 0  # the padding rows were not touched

    # h may alias x; y alone (no residual stream) and h alone (an add, no norm) also work.
    x2 = x.clone()
    y2 = torch.empty_like(x)
    torch.ops.trtllm.wan_vae_norm_silu(x2, bias_x, residual, bias_res, gamma, EPS, x2, y2)
    torch.testing.assert_close(x2, h)
    torch.testing.assert_close(y2, y)
    y3 = torch.empty_like(x)
    torch.ops.trtllm.wan_vae_norm_silu(x, bias_x, residual, bias_res, gamma, EPS, None, y3)
    torch.testing.assert_close(y3, y)
    h4 = torch.empty_like(x)
    torch.ops.trtllm.wan_vae_norm_silu(x, bias_x, residual, bias_res, None, EPS, h4, None)
    torch.testing.assert_close(h4, h)


def test_norm_silu_without_bias_is_the_plain_norm():
    torch.manual_seed(0)
    x = torch.randn(1, 64, 256, device="cuda").to(torch.bfloat16)
    gamma = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    torch.ops.trtllm.wan_vae_norm_silu(x, None, None, None, gamma, EPS, None, y)
    _, y_ref = reference(x, None, None, None, gamma)
    torch.testing.assert_close(y.float(), y_ref, rtol=2e-2, atol=2e-2)


def test_norm_silu_rejects_bad_inputs():
    x = torch.randn(1, 4, 256, device="cuda").to(torch.bfloat16)
    gamma = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="at least one"):
        torch.ops.trtllm.wan_vae_norm_silu(x, None, None, None, gamma, EPS, None, None)
    with pytest.raises(RuntimeError, match="needs gamma"):
        torch.ops.trtllm.wan_vae_norm_silu(
            x, None, None, None, None, EPS, None, torch.empty_like(x)
        )
    with pytest.raises(RuntimeError, match="multiple of 8"):
        bad = torch.randn(1, 4, 100, device="cuda").to(torch.bfloat16)
        torch.ops.trtllm.wan_vae_norm_silu(
            bad, None, None, None, None, EPS, torch.empty_like(bad), None
        )


@pytest.mark.parametrize("shape", [(2, 16, 5, 7), (1, 512, 30, 52), (3, 8, 1, 1)])
def test_upsample2x_matches_nearest_interpolation(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda").to(torch.bfloat16).to(memory_format=torch.channels_last)
    out = torch.ops.trtllm.wan_vae_upsample2x(x)
    assert out.is_contiguous(memory_format=torch.channels_last)
    ref = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest-exact")
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_upsample2x_rejects_other_layouts():
    x = torch.randn(2, 16, 5, 7, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="channels_last"):
        torch.ops.trtllm.wan_vae_upsample2x(x)
