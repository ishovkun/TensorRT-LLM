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
"""Attention backends over CausalKVCacheManager against a dense SDPA reference.

Random inputs; the reference keeps its own dense copy of every K/V ever written
and attends over exactly what is resident: prompt, history (stale tokens
included), and the new tokens.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
PROMPT_CAPACITY = 64
WINDOW = 64
CHUNK = 40  # not a page multiple: exercises partial pages and stale tokens


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
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


@pytest.fixture
def cache():
    mgr = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_block=32,
        prompt_capacity=PROMPT_CAPACITY,
        window_tokens=WINDOW,
        chunk_tokens=CHUNK,
    )
    try:
        yield mgr
    finally:
        mgr.shutdown()


def make_backend(name):
    if name == "cudnn":
        return CuDNNAttention(
            layer_idx=0,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
            dtype=DTYPE,
        )
    return TrtllmAttention(
        layer_idx=0,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        dtype=DTYPE,
        max_seq_len=PROMPT_CAPACITY + WINDOW + CHUNK + 32,
        attention_metadata_state={},
    )


def run(attn, cache, q, k, v, segment_len=None):
    """Per-token [T, H, D] in, per-token [T, H, D] out, whichever backend."""
    out = attn.forward(
        q[None],
        k[None],
        v[None],
        batch_size=1,
        seq_len=q.shape[0],
        kv_cache=cache,
        segment_len=segment_len,
    )
    return out.reshape(q.shape[0], NUM_HEADS, HEAD_DIM)


def rand_qkv(n):
    q = torch.randn(n, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, torch.randn_like(k)


BACKENDS = ["cudnn", "trtllm"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "prompt_len", [0, 17, 64], ids=["no-prompt", "unaligned-prompt", "page-exact-prompt"]
)
def test_rollout_matches_dense_reference(cache, backend, prompt_len):
    torch.manual_seed(0)
    cache.open(prompt_len=prompt_len)
    attn = make_backend(backend)

    prompt_k, prompt_v = rand_qkv(prompt_len)[1:]
    cache.write_prompt_kv(0, prompt_k, prompt_v)

    history_k, history_v = [], []  # the reference's dense copy of committed generator K/V
    empty = prompt_k.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
    saw_stale = False
    for step in range(8):  # long enough to rotate the table more than once
        q, k, v = rand_qkv(CHUNK)
        out = run(attn, cache, q, k, v)
        torch.cuda.synchronize()

        # Resident history is the tail of what was committed, stale tokens included.
        n_hist = cache.history_tokens
        saw_stale |= cache.stale_tokens > 0
        hist_k = torch.cat(history_k)[-n_hist:] if n_hist else empty
        hist_v = torch.cat(history_v)[-n_hist:] if n_hist else empty
        expected = reference_attention(
            q, torch.cat([prompt_k, hist_k, k]), torch.cat([prompt_v, hist_v, v])
        )
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"step {step}")
        if step > 0 or prompt_len > 0:
            chunk_only = reference_attention(q, k, v)
            assert (out.float() - chunk_only.float()).abs().max() > 1e-2, (
                f"step {step}: output equals chunk-only attention, the cached prefix was ignored"
            )

        # The call wrote this chunk's K/V where the next forward expects them.
        positions = torch.arange(cache.past_tokens, cache.past_tokens + CHUNK, device=DEVICE)
        k_back, v_back = cache.read_kv(0, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)

        cache.commit_chunk()
        history_k.append(k)
        history_v.append(v)
    assert saw_stale, "test geometry should attend over stale tokens at some step"


def check_segments(attn, cache, prompt_k, prompt_v, num_segments, chunk):
    """One call cut into segments: segment i sees the cache plus segments <= i."""
    n_hist = cache.history_tokens
    hist = torch.arange(cache.prompt_len, cache.prompt_len + n_hist, device=DEVICE)
    hk, hv = cache.read_kv(0, hist)
    segment_len = chunk // num_segments
    q, k, v = rand_qkv(chunk)
    out = run(attn, cache, q, k, v, segment_len=segment_len)
    torch.cuda.synchronize()
    for i in range(num_segments):
        lo, hi = i * segment_len, (i + 1) * segment_len
        expected = reference_attention(
            q[lo:hi], torch.cat([prompt_k, hk, k[:hi]]), torch.cat([prompt_v, hv, v[:hi]])
        )
        torch.testing.assert_close(out[lo:hi], expected, rtol=2e-2, atol=2e-2, msg=f"segment {i}")
        if i < num_segments - 1:  # a segment must not see the segments after it
            leaky = reference_attention(
                q[lo:hi], torch.cat([prompt_k, hk, k]), torch.cat([prompt_v, hv, v])
            )
            assert (out[lo:hi].float() - leaky.float()).abs().max() > 1e-2, f"segment {i} leaks"
    positions = torch.arange(cache.past_tokens, cache.past_tokens + chunk, device=DEVICE)
    k_back, v_back = cache.read_kv(0, positions)
    torch.testing.assert_close(k_back, k)
    torch.testing.assert_close(v_back, v)
    with pytest.raises(ValueError):
        run(attn, cache, q, k, v, segment_len=7)


def test_cudnn_segments_at_any_alignment(cache):
    """The clean pass: one launch, four segments of 10 sharing pages mid-way."""
    torch.manual_seed(3)
    cache.open(prompt_len=9)
    pk, pv = rand_qkv(9)[1:]
    cache.write_prompt_kv(0, pk, pv)
    for _ in range(3):
        _, k, v = rand_qkv(CHUNK)
        cache.write_range(0, cache.past_tokens, k, v)
        cache.commit_chunk()
    check_segments(make_backend("cudnn"), cache, pk, pv, num_segments=4, chunk=CHUNK)


def test_trtllm_segments_need_page_aligned_starts(cache):
    """The fused update kernel zeroes V after each request to its page end, so
    segments that share a page are refused on this backend rather than corrupted."""
    torch.manual_seed(3)
    cache.open(prompt_len=9)
    cache.write_prompt_kv(0, *rand_qkv(9)[1:])
    q, k, v = rand_qkv(CHUNK)
    with pytest.raises(NotImplementedError, match="page boundary"):
        run(make_backend("trtllm"), cache, q, k, v, segment_len=CHUNK // 4)


def test_trtllm_page_aligned_segments():
    """With segment starts on page boundaries the fused path handles segments too."""
    torch.manual_seed(4)
    chunk = 64
    mgr = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_block=32,
        prompt_capacity=32,
        window_tokens=128,
        chunk_tokens=chunk,
    )
    try:
        mgr.open(prompt_len=32)
        pk, pv = rand_qkv(32)[1:]
        mgr.write_prompt_kv(0, pk, pv)
        _, k, v = rand_qkv(chunk)
        mgr.write_range(0, mgr.past_tokens, k, v)
        mgr.commit_chunk()
        assert mgr.past_tokens % 32 == 0
        check_segments(make_backend("trtllm"), mgr, pk, pv, num_segments=2, chunk=chunk)
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("backend", BACKENDS)
def test_dirty_steps_overwrite_in_place(cache, backend):
    """Several forwards at the same ``past`` leave only the last K/V in the cache."""
    torch.manual_seed(1)
    cache.open(prompt_len=9)
    attn = make_backend(backend)
    cache.write_prompt_kv(0, *rand_qkv(9)[1:])
    past = cache.past_tokens
    last_k = last_v = None
    for _ in range(4):
        q, last_k, last_v = rand_qkv(CHUNK)
        run(attn, cache, q, last_k, last_v)
    torch.cuda.synchronize()
    assert cache.past_tokens == past, "dirty steps must not advance the window"
    positions = torch.arange(past, past + CHUNK, device=DEVICE)
    k_back, v_back = cache.read_kv(0, positions)
    torch.testing.assert_close(k_back, last_k)
    torch.testing.assert_close(v_back, last_v)
