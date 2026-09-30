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
"""CausalKVCacheManager: allocation, addressing, eviction, shared-page refill -- no model, no kernel."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU for the K/V pool"
)

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 16
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def make_cache(tokens_per_block: int):
    """Geometry that scales with the page size so every test exercises partial
    pages, stale tokens and rotation: chunk is a page plus 8 tokens, window two pages."""
    tpb = tokens_per_block
    return CausalKVCacheManager(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_block=tpb,
        prompt_capacity=tpb + 8,
        window_tokens=2 * tpb,
        chunk_tokens=tpb + 8,
    )


@pytest.fixture(params=[32, 128], ids=["tpb32", "tpb128"])
def cache(request):
    mgr = make_cache(request.param)
    try:
        yield mgr
    finally:
        mgr.shutdown()


def rand_kv(n):
    k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return k, torch.randn_like(k)


def read_kv(cache, layer, positions):
    """Gather ``[T, num_kv_heads, head_dim]`` K and V at logical ``positions`` straight from the pool."""
    buf = cache.kv_buffer(layer)
    table = cache.page_table()[0].long()
    page = table[positions // cache.tokens_per_block]
    slot = positions % cache.tokens_per_block
    return buf[page, 0, :, slot, :], buf[page, 1, :, slot, :]


def write_kv_reference(cache, layer, positions, k, v):
    """Scatter one token at a time; the slow, obviously-correct write ``write_range`` must match."""
    buf = cache.kv_buffer(layer)
    table = cache.page_table()[0].long()
    page = table[positions // cache.tokens_per_block]
    slot = positions % cache.tokens_per_block
    buf[page, 0, :, slot, :] = k
    buf[page, 1, :, slot, :] = v


def test_geometry_holds_prompt_window_stale_and_chunk(cache):
    tpb = cache.tokens_per_block
    assert cache.page_view_scale == NUM_LAYERS, "layers share a slot; one layer's view is strided"
    tokens = cache.prompt_capacity + cache.window_tokens + cache.chunk_tokens
    assert cache.num_pages == -(-tokens // tpb) + 1
    assert cache.capacity == cache.num_pages * tpb
    # Worst case resident: full prompt, window plus a page of stale, a chunk.
    assert cache.capacity >= tokens + tpb - 1


def test_open_backs_every_page_once_and_publishes_the_table(cache):
    cache.open(prompt_len=13)
    assert cache.past_tokens == 13, "prompt sits at its real length, no padding"
    table = cache.block_table()
    assert len(table) == cache.num_pages
    assert len(set(table)) == cache.num_pages, "pages must be distinct"
    assert min(table) >= 0
    assert cache._fixed == []  # 13 tokens do not fill a page
    torch.testing.assert_close(
        cache.page_table()[0].cpu(),
        torch.tensor(table, dtype=torch.int32) * cache.page_view_scale,
    )

    cache.close()
    cache.open(prompt_len=cache.tokens_per_block + 5)
    assert len(cache._fixed) == 1, "the full prompt page never rotates"


def test_open_twice_is_an_error_and_close_is_idempotent(cache):
    cache.open(prompt_len=0)
    with pytest.raises(RuntimeError):
        cache.open(prompt_len=0)
    cache.close()
    cache.close()
    cache.open(prompt_len=5)
    assert cache.prompt_len == 5


def test_write_range_matches_indexed_write(cache):
    """The run-based fast write lands bytes exactly where the per-token write does."""
    cache.open(prompt_len=20)
    tpb = cache.tokens_per_block
    for start, n in ((0, 20), (20, 40), (7, 3), (tpb - 1, 2 * tpb + 5), (5, cache.capacity - 5)):
        k, v = rand_kv(n)
        positions = torch.arange(start, start + n, device=DEVICE)
        for layer in range(NUM_LAYERS):
            write_kv_reference(cache, layer, positions, -k, -v)  # poison first
            cache.write_range(layer, start, k, v)
            k_back, v_back = read_kv(cache, layer, positions)
            torch.testing.assert_close(k_back, k)
            torch.testing.assert_close(v_back, v)
            write_kv_reference(cache, layer, positions, k, v)  # and the reference agrees
            k_back, v_back = read_kv(cache, layer, positions)
            torch.testing.assert_close(k_back, k)
            torch.testing.assert_close(v_back, v)

    # Independently: the raw pool at (table[t // tpb], t % tpb) holds token t.
    layer = NUM_LAYERS - 1
    buf = cache.kv_buffer(layer)
    assert buf.shape[1:] == (2, NUM_KV_HEADS, tpb, HEAD_DIM)
    table = cache.block_table()
    positions = torch.arange(cache.capacity, device=DEVICE)
    k_back, _ = read_kv(cache, layer, positions)
    for t in (0, 19, 20, tpb - 1, tpb, cache.capacity - 1):
        page, slot = table[t // tpb] * cache.page_view_scale, t % tpb
        torch.testing.assert_close(buf[page, 0, :, slot, :], k_back[t])

    with pytest.raises(ValueError):
        cache.write_range(0, cache.capacity - 1, *rand_kv(2))
    with pytest.raises(ValueError):
        cache.write_prompt_kv(0, *rand_kv(19))


def test_eviction_keeps_the_window_and_the_prompt(cache):
    """Content check across many chunks with a prompt that shares a page with the history."""
    prompt_len = 13
    cache.open(prompt_len=prompt_len)
    tpb, chunk, window = cache.tokens_per_block, cache.chunk_tokens, cache.window_tokens
    allocated = sorted(cache.block_table())
    pk, pv = rand_kv(prompt_len)
    for layer in range(NUM_LAYERS):
        cache.write_prompt_kv(layer, pk, pv)
    written = []  # one stamp per committed generator token, oldest first
    versions = {cache.table_version}
    saw_stale = saw_rotation = False

    for c in range(1, 12):
        stamp = torch.full((chunk, NUM_KV_HEADS, HEAD_DIM), float(c), device=DEVICE, dtype=DTYPE)
        for layer in range(NUM_LAYERS):
            cache.write_range(layer, cache.past_tokens, stamp, -stamp)
        before = cache.table_version
        cache.commit()
        written.extend([c] * chunk)
        saw_rotation |= cache.table_version != before
        versions.add(cache.table_version)

        # Whole-page eviction leaves fewer than one page of stale tokens, all attended.
        assert cache.history_tokens <= window + tpb - 1
        assert 0 <= cache.stale_tokens < tpb
        saw_stale |= cache.stale_tokens > 0
        assert cache.history_tokens - cache.stale_tokens == min(len(written), window)
        assert cache.past_tokens == prompt_len + cache.history_tokens

        for layer in range(NUM_LAYERS):
            # Resident history, oldest first, is the tail of what was written.
            hist = torch.arange(prompt_len, prompt_len + cache.history_tokens, device=DEVICE)
            k_back, v_back = read_kv(cache, layer, hist)
            expect = torch.tensor(written[-cache.history_tokens :], device=DEVICE, dtype=DTYPE)
            torch.testing.assert_close(k_back[:, 0, 0], expect)
            torch.testing.assert_close(v_back[:, 0, 0], -expect)
            # The prompt survived every rotation of the page it shares with the history.
            k_p, v_p = read_kv(cache, layer, torch.arange(prompt_len, device=DEVICE))
            torch.testing.assert_close(k_p, pk)
            torch.testing.assert_close(v_p, pv)

        # Eviction recycles pages; it never allocates or frees any.
        table = cache.block_table()
        assert sorted(table) == allocated
        assert len(set(table)) == len(table)
        scaled = torch.tensor(table, dtype=torch.int32) * cache.page_view_scale
        for row in cache.page_table(cache.MAX_CAUSAL_BLOCKS).cpu():
            torch.testing.assert_close(row, scaled)

    assert saw_stale, "test geometry should produce stale tokens"
    assert saw_rotation, "test geometry should rotate the table"


def test_causal_block_lengths_are_causal_across_blocks_and_persistent(cache):
    cache.open(prompt_len=3)
    chunk = cache.chunk_tokens
    q, kv = cache.causal_block_lengths(1, chunk)
    assert q.dtype == kv.dtype == torch.int32
    assert q.tolist() == [chunk] and kv.tolist() == [3 + chunk]

    q4, kv4 = cache.causal_block_lengths(4, chunk // 4)
    assert q4.data_ptr() == q.data_ptr() and kv4.data_ptr() == kv.data_ptr(), "same buffers"
    assert q4.tolist() == [chunk // 4] * 4
    assert kv4.tolist() == [3 + (i + 1) * chunk // 4 for i in range(4)]

    cache.commit()
    _, kv_after = cache.causal_block_lengths(4, chunk // 4)
    assert kv_after.tolist() == [3 + chunk + (i + 1) * chunk // 4 for i in range(4)]

    with pytest.raises(ValueError):
        cache.causal_block_lengths(cache.MAX_CAUSAL_BLOCKS + 1, 1)
    with pytest.raises(ValueError):
        cache.causal_block_lengths(2, chunk)  # two full chunks do not fit one chunk
    with pytest.raises(ValueError):
        cache.page_table(cache.MAX_CAUSAL_BLOCKS + 1)


def test_copy_batch_block_offsets_encodes_our_table(cache):
    cache.open(prompt_len=3)
    for _ in range(5):
        cache.commit()
    dst = torch.full((1, 4, 2, cache.max_blocks_per_seq), -7, dtype=torch.int32, device="cuda")
    cache.copy_batch_block_offsets(dst, [cache.REQUEST_ID] * 3, 1, 3, 3)
    torch.cuda.synchronize()
    table = torch.tensor(cache.block_table(), dtype=torch.int32, device="cuda")
    n = table.numel()
    scale = int(cache.index_scales[0])
    kv_offset = int(cache.kv_offset[0])
    for block_size in range(3):  # every causal block row is the one sequence
        torch.testing.assert_close(dst[0, block_size, 0, :n], table * scale)
        torch.testing.assert_close(dst[0, block_size, 1, :n], table * scale + kv_offset)
        assert torch.count_nonzero(dst[0, block_size, :, n:]) == 0, (
            "unused entries are the safe page 0"
        )
    assert (dst[0, 3] == -7).all(), "rows beyond num_seqs are untouched"

    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, [1], 1, 1, 1)
    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, [cache.REQUEST_ID, 1], 1, 2, 2)
    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, [cache.REQUEST_ID] * 2, 1, 2, 1)


def test_rejects_bad_geometry():
    with pytest.raises(ValueError):
        make_cache(0)
    for bad in (12, 24, 100):
        with pytest.raises(ValueError, match="power of two"):
            make_cache(bad)
    with pytest.raises(ValueError):
        CausalKVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
            dtype=torch.float32,
            tokens_per_block=32,
            prompt_capacity=8,
            window_tokens=8,
            chunk_tokens=8,
        )
