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
"""Fused TRT-LLM attention over CausalKVCacheManager against a dense SDPA reference.

Random inputs; the reference keeps its own dense copy of every K/V ever written
and attends over exactly what the mask says is visible.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.causal_trtllm import CausalTrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def reference_attention(q, keys, values):
    """q [T, H, D]; keys/values [S, H_kv, D] already restricted to the visible set."""
    rep = NUM_HEADS // NUM_KV_HEADS
    kx = keys.repeat_interleave(rep, dim=1)
    vx = values.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).float().unsqueeze(0),
        kx.transpose(0, 1).float().unsqueeze(0),
        vx.transpose(0, 1).float().unsqueeze(0),
    )
    return out.squeeze(0).transpose(0, 1).reshape(q.shape[0], -1).to(q.dtype)


@pytest.fixture(params=[16, 64], ids=["tpb16", "tpb64"])
def cache(request):
    mgr = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_block=request.param,
        prompt_capacity=40,
        window_tokens=64,
        chunk_tokens=32,
    )
    try:
        yield mgr
    finally:
        mgr.shutdown()


@pytest.mark.parametrize(
    "prompt_len", [0, 17, 40], ids=["no-prompt", "partial-prompt", "full-prompt"]
)
def test_rollout_matches_dense_reference(cache, prompt_len):
    torch.manual_seed(0)
    cache.open(prompt_len=prompt_len)
    attn = CausalTrtllmAttention(
        cache,
        layer_idx=0,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )

    prompt_k = torch.randn(prompt_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    prompt_v = torch.randn_like(prompt_k)
    cache.write_prompt_kv(0, prompt_k, prompt_v)

    history_k, history_v = [], []  # the reference's dense copy of committed generator K/V
    chunk = cache.chunk_tokens
    for step in range(7):  # long enough to slide the window more than once
        q = torch.randn(chunk, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(chunk, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        v = torch.randn_like(k)

        out = attn.forward(q, k, v)
        torch.cuda.synchronize()

        empty = k.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        hist_k = torch.cat(history_k)[-cache.window_tokens :] if history_k else empty
        hist_v = torch.cat(history_v)[-cache.window_tokens :] if history_v else empty
        keys = torch.cat([prompt_k, hist_k, k], dim=0)
        vals = torch.cat([prompt_v, hist_v, v], dim=0)
        expected = reference_attention(q, keys, vals)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"step {step}")

        # The fused call wrote this chunk's K/V where the next forward expects them.
        positions = torch.arange(cache.past_tokens, cache.past_tokens + chunk, device=DEVICE)
        k_back, v_back = cache.read_kv(0, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)

        cache.commit_chunk()
        history_k.append(k)
        history_v.append(v)


def test_dirty_steps_overwrite_in_place(cache):
    """Several forwards at the same ``past`` leave only the last K/V in the cache."""
    torch.manual_seed(1)
    cache.open(prompt_len=9)
    attn = CausalTrtllmAttention(
        cache,
        layer_idx=0,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )
    cache.write_prompt_kv(
        0,
        torch.randn(9, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE),
        torch.randn(9, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE),
    )
    chunk = cache.chunk_tokens
    past = cache.past_tokens
    last_k = last_v = None
    for _ in range(4):
        q = torch.randn(chunk, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        last_k = torch.randn(chunk, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        last_v = torch.randn_like(last_k)
        attn.forward(q, last_k, last_v)
    torch.cuda.synchronize()
    assert cache.past_tokens == past, "dirty steps must not advance the window"
    positions = torch.arange(past, past + chunk, device=DEVICE)
    k_back, v_back = cache.read_kv(0, positions)
    torch.testing.assert_close(k_back, last_k)
    torch.testing.assert_close(v_back, last_v)
