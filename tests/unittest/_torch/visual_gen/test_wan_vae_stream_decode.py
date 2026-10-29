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
"""The Wan VAE decoded a few latent frames at a time equals one decode of the clip."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan import wan_vae
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanRMSNorm, WanVAE, WanVAEConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("chunks", [[1, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1], [1, 3, 3]])
def test_stream_decode_matches_whole_clip_decode(chunks):
    torch.manual_seed(0)
    vae = (
        WanVAE(WanVAEConfig(base_dim=16, z_dim=4, num_res_blocks=1))
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    frames = sum(chunks)
    z = torch.randn(1, 4, frames, 8, 8, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        whole = vae.decode(z, return_dict=False, temporal_chunk_size=2)[0]
        vae.decode_stream_start()
        parts, start = [], 0
        for n in chunks:
            parts.append(vae.decode_stream_step(z[:, :, start : start + n]))
            start += n
        vae.decode_stream_end()
        streamed = torch.cat(parts, dim=2)
    assert streamed.shape == whole.shape
    torch.testing.assert_close(streamed, whole, rtol=2e-2, atol=2e-2)
    with pytest.raises(ValueError, match="one latent frame"):
        vae.decode_stream_start()
        vae.decode_stream_step(z[:, :, :2])
    vae.decode_stream_end()


def test_fused_channel_norm_matches_the_eager_norm():
    """Channels-last bf16 video goes through the fused channel RMSNorm; any other
    layout or dtype through the eager chain. The two agree to bf16 rounding."""
    torch.manual_seed(1)
    norm = WanRMSNorm(96, images=False).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.gamma.mul_(torch.rand_like(norm.gamma) * 2)
    x = (torch.randn(1, 96, 3, 20, 24, device="cuda") * 3).to(torch.bfloat16)
    eager = norm(x)  # NCTHW contiguous: eager path
    fused = norm(x.to(memory_format=torch.channels_last_3d))
    assert fused.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(fused.contiguous(), eager, rtol=2e-2, atol=2e-2)
    ref = torch.nn.functional.normalize(x.float(), dim=1) * norm.scale * norm.gamma.float()
    for name, y in (("eager", eager), ("fused", fused)):
        err = (y.float() - ref).abs().max().item()
        assert err < 0.05 * ref.abs().max().item(), f"{name}: {err}"


def _small_vae():
    return (
        WanVAE(WanVAEConfig(base_dim=16, z_dim=4, num_res_blocks=1))
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )


def _randomize(vae):
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("bias"):
                p.normal_(std=0.5)
            elif name.endswith("gamma"):
                p.mul_(torch.rand_like(p) * 2)


def test_fused_glue_matches_the_eager_blocks(monkeypatch):
    """Decoder and encoder with the fused bias+shortcut+norm+SiLU kernel agree with the
    eager chain (separate bias add, residual add, norm, SiLU, cache concat)."""
    assert wan_vae._norm_silu_op() is not None, (
        "the wan_vae_norm_silu op is missing from this build"
    )
    torch.manual_seed(2)
    vae = _small_vae()
    _randomize(vae)
    z = torch.randn(2, 4, 5, 8, 8, device="cuda", dtype=torch.bfloat16)
    video = torch.randn(2, 3, 9, 32, 32, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        fused = vae.decode(z, return_dict=False, temporal_chunk_size=2)[0]
        fused_latent = vae.encode(video, return_dict=False)[0].mode()
        monkeypatch.setattr(wan_vae, "_norm_silu_op", lambda: None)
        eager = vae.decode(z, return_dict=False, temporal_chunk_size=2)[0]
        eager_latent = vae.encode(video, return_dict=False)[0].mode()
    torch.testing.assert_close(fused, eager, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(fused_latent, eager_latent, rtol=2e-2, atol=2e-2)


def test_pending_add_resolves_in_place():
    x = (
        torch.randn(1, 16, 2, 4, 4, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    residual = torch.randn_like(x)
    bias = torch.randn(16, device="cuda").to(torch.bfloat16)
    expected = x + bias.view(1, -1, 1, 1, 1) + residual
    pending = wan_vae._PendingAdd(x, bias, residual)
    out = pending.materialize()
    assert out is x
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    assert pending.materialize() is x  # already resolved: nothing is added twice
    torch.testing.assert_close(x, expected, rtol=1e-2, atol=1e-2)
