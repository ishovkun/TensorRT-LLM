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
and attends over exactly what the model's window allows: the fixed region (the
prompt), the ``WINDOW`` tokens before each block, the earlier blocks of the chunk
and the block itself. Stale tokens the whole-page eviction keeps resident must
not be seen.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.modules.attention import Attention
from tensorrt_llm.visual_gen.args import AttentionConfig

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


def exact_reference(q, pk, pv, hk, hv, k, v, start, end):
    """Attention of the new tokens ``[start, end)`` of the chunk ``k``/``v`` over what the
    window allows: the prompt, the ``WINDOW`` keys before ``start`` and the chunk up
    to ``end``. ``hk``/``hv`` are every history token committed so far, oldest first."""
    keys = torch.cat([pk, hk, k[:end]])
    values = torch.cat([pv, hv, v[:end]])
    pos = torch.arange(keys.shape[0], device=keys.device)
    visible = (pos < pk.shape[0]) | (pos >= pk.shape[0] + hk.shape[0] + start - WINDOW)
    return reference_attention(q[start:end], keys[visible], values[visible])


@pytest.fixture(params=[32, 64], ids=["tpb32", "tpb64"])
def cache(request):
    mgr = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=request.param,
        fixed_capacity=PROMPT_CAPACITY,
        window_tokens=WINDOW,
        chunk_tokens=CHUNK,
        causal_block_sizes=(CHUNK, CHUNK // 4),
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


def run(attn, cache, q, k, v, causal_block_size=None):
    """Per-token [T, H, D] in, per-token [T, H, D] out, whichever backend."""
    out = attn.forward(
        q[None],
        k[None],
        v[None],
        batch_size=1,
        seq_len=q.shape[0],
        kv_cache=cache,
        causal_block_size=causal_block_size,
    )
    return out.reshape(q.shape[0], NUM_HEADS, HEAD_DIM)


def read_kv(cache, layer, positions):
    """Gather ``[T, num_kv_heads, head_dim]`` K and V at logical ``positions`` straight from the pool."""
    buf = cache.kv_buffer(layer)
    table = cache.table.long()
    page = table[positions // cache.tokens_per_page]
    slot = positions % cache.tokens_per_page
    return buf[page, 0, :, slot, :], buf[page, 1, :, slot, :]


def rand_qkv(n):
    q = torch.randn(n, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, torch.randn_like(k)


def open_with_prompt(cache, prompt_len):
    """Open pinning ``prompt_len`` tokens, write a random prompt at position 0 and
    commit it; returns its K and V."""
    cache.open(pin_tokens=prompt_len)
    pk, pv = rand_qkv(prompt_len)[1:]
    cache.write_range(0, 0, pk, pv)
    if prompt_len:
        cache.commit(prompt_len)
    return pk, pv


BACKENDS = ["cudnn", "trtllm"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "prompt_len", [0, 17, 64], ids=["no-prompt", "unaligned-prompt", "page-exact-prompt"]
)
def test_rollout_matches_dense_reference(cache, backend, prompt_len):
    torch.manual_seed(0)
    prompt_k, prompt_v = open_with_prompt(cache, prompt_len)
    attn = make_backend(backend)
    if backend == "trtllm" and cache.tokens_per_page != 32:
        # trtllm-gen has paged context kernels for 32-token pages only; other sizes
        # would silently drop the prefix, so the backend must refuse them.
        q, k, v = rand_qkv(CHUNK)
        with pytest.raises(NotImplementedError, match="32-token"):
            run(attn, cache, q, k, v)
        return

    history_k, history_v = [], []  # the reference's dense copy of committed generator K/V
    empty = prompt_k.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
    saw_stale = False
    for step in range(8):  # long enough to rotate the table more than once
        q, k, v = rand_qkv(CHUNK)
        out = run(attn, cache, q, k, v)
        torch.cuda.synchronize()

        saw_stale |= cache.history_tokens > WINDOW
        hk = torch.cat(history_k) if history_k else empty
        hv = torch.cat(history_v) if history_v else empty
        expected = exact_reference(q, prompt_k, prompt_v, hk, hv, k, v, 0, CHUNK)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"step {step}")
        if step > 0 or prompt_len > 0:
            chunk_only = reference_attention(q, k, v)
            assert (out.float() - chunk_only.float()).abs().max() > 1e-2, (
                f"step {step}: output equals chunk-only attention, the cached prefix was ignored"
            )

        # The call wrote this chunk's K/V where the next forward expects them.
        positions = torch.arange(cache.past_tokens, cache.past_tokens + CHUNK, device=DEVICE)
        k_back, v_back = read_kv(cache, 0, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)

        cache.commit()
        history_k.append(k)
        history_v.append(v)
    assert saw_stale, "test geometry should hold stale tokens at some step"


def check_causal_blocks(attn, cache, prompt_k, prompt_v, hk, hv, num_causal_blocks, chunk):
    """One call cut into causal blocks: block i sees the prompt, its own window of
    history, the blocks before it and itself. ``hk``/``hv``: all committed history."""
    causal_block_size = chunk // num_causal_blocks
    q, k, v = rand_qkv(chunk)
    out = run(attn, cache, q, k, v, causal_block_size=causal_block_size)
    torch.cuda.synchronize()
    for i in range(num_causal_blocks):
        lo, hi = i * causal_block_size, (i + 1) * causal_block_size
        expected = exact_reference(q, prompt_k, prompt_v, hk, hv, k, v, lo, hi)
        torch.testing.assert_close(
            out[lo:hi], expected, rtol=2e-2, atol=2e-2, msg=f"causal block {i}"
        )
        if i < num_causal_blocks - 1:  # a causal block must not see the causal blocks after it
            leaky = reference_attention(
                q[lo:hi], torch.cat([prompt_k, hk, k]), torch.cat([prompt_v, hv, v])
            )
            assert (out[lo:hi].float() - leaky.float()).abs().max() > 1e-2, (
                f"causal block {i} leaks"
            )
    positions = torch.arange(cache.past_tokens, cache.past_tokens + chunk, device=DEVICE)
    k_back, v_back = read_kv(cache, 0, positions)
    torch.testing.assert_close(k_back, k)
    torch.testing.assert_close(v_back, v)
    with pytest.raises(ValueError):
        run(attn, cache, q, k, v, causal_block_size=7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_causal_blocks_at_any_alignment(cache, backend):
    """The clean pass: one launch, four causal blocks of 10 sharing pages mid-way,
    each with its own window start, over a rotated table with stale tokens."""
    torch.manual_seed(3)
    if backend == "trtllm" and cache.tokens_per_page != 32:
        pytest.skip("trtllm-gen: 32-token pages only")
    pk, pv = open_with_prompt(cache, 9)
    history_k, history_v = [], []
    for _ in range(3):  # 120 tokens committed, one page dropped: 88 resident, 24 stale
        _, k, v = rand_qkv(CHUNK)
        cache.write_range(0, cache.past_tokens, k, v)
        cache.commit()
        history_k.append(k)
        history_v.append(v)
    assert cache.history_tokens > WINDOW, "test geometry should hold stale tokens here"
    check_causal_blocks(
        make_backend(backend),
        cache,
        pk,
        pv,
        torch.cat(history_k),
        torch.cat(history_v),
        num_causal_blocks=4,
        chunk=CHUNK,
    )


def indicator_values(watch, first, count):
    """V for tokens ``[first, first + count)`` of the sequence: 1 in dimension ``d`` of
    K/V head ``h`` for the token ``watch[h][d]``, 0 everywhere else."""
    ids = torch.arange(first, first + count, device=DEVICE)
    return (ids[:, None, None] == watch[None]).to(DTYPE)


@pytest.mark.parametrize("num_causal_blocks", [1, 4])
@pytest.mark.parametrize("backend", BACKENDS)
def test_each_block_sees_exactly_its_window(cache, backend, num_causal_blocks):
    """Exact visible-key sets, no tolerance games: with q = 0 every visible key gets
    the same weight 1/N, so with V one-hot per watched token the output is
    ``[token visible] / N``. An off-by-one window edge, a stale or evicted token
    read, or a key read twice shows up as a 1, a 0 or a 2 where it should not."""
    if backend == "trtllm" and cache.tokens_per_page != 32:
        pytest.skip("trtllm-gen: 32-token pages only")
    torch.manual_seed(7)
    prompt, commits = 9, 3
    hist = commits * CHUNK  # 120 committed: some evicted, some stale, the rest visible
    total = prompt + hist + CHUNK
    # Watch every token of the prompt, the last 79 history tokens (both window
    # edges of every block, stale ones included) and the whole chunk: 128 ids,
    # one per (K/V head, dimension).
    ids = torch.cat(
        [
            torch.arange(prompt),
            torch.arange(prompt + hist - 79, prompt + hist),
            torch.arange(prompt + hist, total),
        ]
    ).to(DEVICE)
    watch = ids.view(NUM_KV_HEADS, HEAD_DIM)

    def keys(n):
        return torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    cache.open(pin_tokens=prompt)
    cache.write_range(0, 0, keys(prompt), indicator_values(watch, 0, prompt))
    cache.commit(prompt)
    for c in range(commits):
        first = prompt + c * CHUNK
        cache.write_range(0, cache.past_tokens, keys(CHUNK), indicator_values(watch, first, CHUNK))
        cache.commit()
    assert cache.history_tokens > WINDOW, "test geometry should hold stale tokens here"

    size = CHUNK // num_causal_blocks
    q = torch.zeros(CHUNK, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = indicator_values(watch, prompt + hist, CHUNK)
    out = run(make_backend(backend), cache, q, keys(CHUNK), v, causal_block_size=size)
    torch.cuda.synchronize()

    token = torch.arange(CHUNK, device=DEVICE)
    start = token // size * size  # each query's block start within the chunk
    end = start + size
    # Global ids: prompt [0, prompt), history [prompt, prompt + hist), chunk after.
    g = watch.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=0)[None]  # [1, H, D]
    win_start = (prompt + hist + start - WINDOW)[:, None, None]
    visible = (g < prompt) | ((g >= win_start) & (g < (prompt + hist + end)[:, None, None]))
    num_visible = (prompt + torch.clamp(hist + start, max=WINDOW) + size)[:, None, None]
    seen = out.float() * num_visible
    assert torch.allclose(seen, visible.float(), atol=0.05), (
        f"{(seen.round() != visible.float()).sum().item()} (query, key) pairs wrong"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_padding_tokens_are_neither_written_nor_attended(cache, backend):
    """``seq_len`` counts the real tokens: a padded chunk behaves like the unpadded one."""
    if backend == "trtllm" and cache.tokens_per_page != 32:
        pytest.skip("trtllm-gen: 32-token pages only")
    torch.manual_seed(6)
    pk, pv = open_with_prompt(cache, 9)
    attn = make_backend(backend)
    pad = 8
    q, k, v = rand_qkv(CHUNK + pad)
    out = attn.forward(
        q[None],
        k[None],
        v[None],
        batch_size=1,
        seq_len=CHUNK,
        kv_cache=cache,
    ).reshape(CHUNK + pad, NUM_HEADS, HEAD_DIM)
    torch.cuda.synchronize()
    expected = reference_attention(
        q[:CHUNK], torch.cat([pk, k[:CHUNK]]), torch.cat([pv, v[:CHUNK]])
    )
    torch.testing.assert_close(out[:CHUNK], expected, rtol=2e-2, atol=2e-2)
    assert out[CHUNK:].abs().max().item() == 0.0, "padding rows must be zero"
    positions = torch.arange(cache.past_tokens, cache.past_tokens + CHUNK, device=DEVICE)
    k_back, v_back = read_kv(cache, 0, positions)
    torch.testing.assert_close(k_back, k[:CHUNK])
    torch.testing.assert_close(v_back, v[:CHUNK])
    for bad in (0, CHUNK + pad + 1):
        with pytest.raises(ValueError):
            attn.forward(q[None], k[None], v[None], batch_size=1, seq_len=bad, kv_cache=cache)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dirty_steps_overwrite_in_place(cache, backend):
    """Several forwards at the same ``past`` leave only the last K/V in the cache."""
    if backend == "trtllm" and cache.tokens_per_page != 32:
        pytest.skip("trtllm-gen: 32-token pages only")
    torch.manual_seed(1)
    open_with_prompt(cache, 9)
    attn = make_backend(backend)
    past = cache.past_tokens
    last_k = last_v = None
    for _ in range(4):
        q, last_k, last_v = rand_qkv(CHUNK)
        run(attn, cache, q, last_k, last_v)
    torch.cuda.synchronize()
    assert cache.past_tokens == past, "dirty steps must not advance the window"
    positions = torch.arange(past, past + CHUNK, device=DEVICE)
    k_back, v_back = read_kv(cache, 0, positions)
    torch.testing.assert_close(k_back, last_k)
    torch.testing.assert_close(v_back, last_v)


@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_replay_survives_commit(cache, backend):
    """A forward captured in a CUDA graph stays correct after commit() moves the cache,
    including across a page rotation: writes land on the new pages and attention
    reads the new lengths. The step loop refreshes TRTLLM metadata before replay."""
    if backend == "trtllm" and cache.tokens_per_page != 32:
        pytest.skip("trtllm-gen: 32-token pages only")
    torch.manual_seed(5)
    pk, pv = open_with_prompt(cache, 9)
    attn = make_backend(backend)
    history_k, history_v = [], []
    for _ in range(2):
        _, k, v = rand_qkv(CHUNK)
        cache.write_range(0, cache.past_tokens, k, v)
        cache.commit()
        history_k.append(k)
        history_v.append(v)

    q, k, v = rand_qkv(CHUNK)  # static buffers the graph reads
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # eager warmup creates lengths and metadata before capture
            run(attn, cache, q, k, v)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(attn, cache, q, k, v)

    for step in range(3):  # the third commit rotates the table
        history_k.append(k.clone())
        history_v.append(v.clone())
        cache.commit()
        if backend == "trtllm":
            attn.metadata.prepare_with_kv_cache(cache, 1, CHUNK)  # the step loop's job
        q2, k2, v2 = rand_qkv(CHUNK)
        q.copy_(q2), k.copy_(k2), v.copy_(v2)
        graph.replay()
        torch.cuda.synchronize()

        hk, hv = torch.cat(history_k), torch.cat(history_v)
        expected = exact_reference(q, pk, pv, hk, hv, k, v, 0, CHUNK)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"replay {step}")
        positions = torch.arange(cache.past_tokens, cache.past_tokens + CHUNK, device=DEVICE)
        k_back, v_back = read_kv(cache, 0, positions)
        torch.testing.assert_close(k_back, k, msg=f"replay {step}: K landed on stale pages")
        torch.testing.assert_close(v_back, v, msg=f"replay {step}: V landed on stale pages")


@pytest.mark.parametrize("num_causal_blocks", [1, 4])
@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_captured_while_the_window_fills(backend, num_causal_blocks):
    """A forward captured on the rollout's first chunk, with no history yet, replays
    correctly through a dozen commits: while the window fills, once it is full, and
    across many rotations of a pool of five 32-token pages. Lengths the kernels
    read must come from the cache at replay, not from the moment of capture."""
    torch.manual_seed(8)
    prompt, size = 8, CHUNK // num_causal_blocks
    cache = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=32,
        fixed_capacity=prompt,
        window_tokens=WINDOW,
        chunk_tokens=CHUNK,
        causal_block_sizes=(CHUNK, CHUNK // 4),
    )
    try:
        assert cache.num_pages == 5
        pk, pv = open_with_prompt(cache, prompt)
        attn = make_backend(backend)
        q, k, v = rand_qkv(CHUNK)  # static buffers the graph reads
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # eager warmup creates lengths and metadata before capture
                run(attn, cache, q, k, v, causal_block_size=size)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(attn, cache, q, k, v, causal_block_size=size)

        history_k, history_v = [], []
        empty = pk.new_zeros((0, NUM_KV_HEADS, HEAD_DIM))
        for step in range(12):
            q2, k2, v2 = rand_qkv(CHUNK)
            q.copy_(q2), k.copy_(k2), v.copy_(v2)
            if backend == "trtllm":  # the step loop's job, after every commit
                attn.metadata.prepare_with_kv_cache(cache, num_causal_blocks, size)
            graph.replay()
            torch.cuda.synchronize()
            hk = torch.cat(history_k) if history_k else empty
            hv = torch.cat(history_v) if history_v else empty
            expected = torch.cat(
                [
                    exact_reference(q, pk, pv, hk, hv, k, v, i * size, (i + 1) * size)
                    for i in range(num_causal_blocks)
                ]
            )
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2, msg=f"step {step}")
            history_k.append(k.clone())
            history_v.append(v.clone())
            cache.commit()
        assert len(history_k) * CHUNK >= 3 * cache.capacity, "the pool should cycle three times"
    finally:
        cache.shutdown()


def test_backend_without_cache_support_refuses_a_cache(cache):
    """A cache routed to a backend that cannot use it raises instead of being ignored."""
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=NUM_HEADS * HEAD_DIM,
            num_attention_heads=NUM_HEADS,
            attention_head_dim=HEAD_DIM,
            eps=1e-6,
        ),
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=True,
    )
    attn = Attention(
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        config=config,
    )
    assert not attn.attn.support_kv_cache()
    assert make_backend("cudnn").support_kv_cache()
    assert make_backend("trtllm").support_kv_cache()
    q = torch.zeros(1, 8, NUM_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
    with pytest.raises(NotImplementedError, match="does not support a K/V cache"):
        attn._attn_impl(q, q, q, kv_cache=cache)
