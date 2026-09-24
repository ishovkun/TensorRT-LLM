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
"""CausalKVCacheManager: allocation, addressing, eviction, masking — no model, no kernel."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU for the K/V pool"
)

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 16


def make_cache(tokens_per_block: int, *, prompt_capacity=20, window_tokens=40, chunk_tokens=16):
    return CausalKVCacheManager(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        tokens_per_block=tokens_per_block,
        prompt_capacity=prompt_capacity,
        window_tokens=window_tokens,
        chunk_tokens=chunk_tokens,
    )


@pytest.fixture(params=[32], ids=["tpb32"])
def cache(request):
    mgr = make_cache(request.param)
    try:
        yield mgr
    finally:
        mgr.shutdown()


def test_geometry_rounds_prompt_to_whole_pages(cache):
    tpb = cache.tokens_per_block
    assert cache.prompt_capacity % tpb == 0
    assert cache.prompt_capacity >= 20
    assert cache.prompt_pages * tpb == cache.prompt_capacity
    # History region can hold the window, a chunk, and one page of eviction slack.
    assert cache.history_pages * tpb >= cache.window_tokens + cache.chunk_tokens + tpb - 1


def test_open_backs_every_page_once(cache):
    cache.open(prompt_len=13)
    table = cache.block_table()
    assert len(table) == -(-cache.seq_len // cache.tokens_per_block)
    assert table[: cache.prompt_pages] == cache._prompt_table
    all_pages = cache._prompt_table + list(cache._ring)
    assert len(all_pages) == cache.num_pages
    assert len(set(all_pages)) == cache.num_pages, "pages must be distinct"
    assert min(all_pages) >= 0


def test_open_twice_is_an_error_and_close_is_idempotent(cache):
    cache.open(prompt_len=0)
    with pytest.raises(RuntimeError):
        cache.open(prompt_len=0)
    cache.close()
    cache.close()
    cache.open(prompt_len=5)
    assert cache.prompt_len == 5


def test_roundtrip_through_pool(cache):
    """Bytes written at a logical position come back from the same logical position."""
    cache.open(prompt_len=20)
    device = torch.device("cuda")
    positions = torch.arange(cache.seq_len, device=device)
    for layer in range(NUM_LAYERS):
        k = torch.randn(cache.seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache.write_kv(layer, positions, k, v)
        k_back, v_back = cache.read_kv(layer, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)

    # Independently: the raw pool at (table[t // tpb], t % tpb) holds token t.
    layer = NUM_LAYERS - 1
    buf = cache.kv_buffer(layer)
    assert buf.shape[1:] == (2, NUM_KV_HEADS, cache.tokens_per_block, HEAD_DIM)
    assert buf.shape[0] >= cache.num_pages
    table = cache.block_table()
    k_back, _ = cache.read_kv(layer, positions)
    for t in (0, cache.prompt_capacity - 1, cache.prompt_capacity, cache.seq_len - 1):
        page, slot = table[t // cache.tokens_per_block], t % cache.tokens_per_block
        torch.testing.assert_close(buf[page, 0, :, slot, :], k_back[t])


def test_prompt_write_zeroes_padding(cache):
    cache.open(prompt_len=7)
    device = torch.device("cuda")
    k = torch.randn(7, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cache.write_prompt_kv(0, k, v)
    positions = torch.arange(cache.prompt_capacity, device=device)
    k_back, v_back = cache.read_kv(0, positions)
    torch.testing.assert_close(k_back[:7], k)
    torch.testing.assert_close(v_back[:7], v)
    assert torch.count_nonzero(k_back[7:]) == 0
    assert torch.count_nonzero(v_back[7:]) == 0


def test_eviction_is_whole_pages_and_keeps_the_window(cache):
    """Content check across many chunks: the resident history is exactly the tail of what was written."""
    cache.open(prompt_len=0)
    device = torch.device("cuda")
    tpb, chunk, window = cache.tokens_per_block, cache.chunk_tokens, cache.window_tokens
    allocated = sorted(cache._prompt_table + list(cache._ring))
    written = []  # one stamp per committed generator token, oldest first

    for c in range(1, 12):
        past = cache.past_tokens
        positions = torch.arange(past, past + chunk, device=device)
        stamp = torch.full(
            (chunk, NUM_KV_HEADS, HEAD_DIM), float(c), device=device, dtype=torch.bfloat16
        )
        cache.write_kv(0, positions, stamp, -stamp)
        cache.commit_chunk()
        written.extend([c] * chunk)

        # Whole-page eviction leaves fewer than one page of stale tokens.
        assert cache.history_tokens <= window + tpb - 1
        assert 0 <= cache.stale_tokens < tpb
        assert cache.history_tokens - cache.stale_tokens == min(len(written), window)

        # What is resident, oldest first, is the tail of what was written.
        hist = torch.arange(
            cache.prompt_capacity, cache.prompt_capacity + cache.history_tokens, device=device
        )
        k_back, v_back = cache.read_kv(0, hist)
        expect = torch.tensor(written[-cache.history_tokens :], device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(k_back[:, 0, 0], expect)
        torch.testing.assert_close(v_back[:, 0, 0], -expect)

        # Eviction recycles pages; it never allocates or frees any.
        assert sorted(cache._prompt_table + list(cache._ring)) == allocated
        assert len(set(cache.block_table())) == len(cache.block_table())


def test_attention_mask_covers_prompt_padding_and_stale_head(cache):
    cache.open(prompt_len=cache.prompt_capacity)
    device = torch.device("cuda")
    assert cache.attention_mask(device) is None, "full prompt, no history: nothing to mask"

    cache.close()
    cache.open(prompt_len=5)
    mask = cache.attention_mask(device)
    assert mask.shape == (cache.chunk_tokens, cache.seq_len)
    assert mask[:, :5].all()
    assert not mask[:, 5 : cache.prompt_capacity].any()
    assert mask[:, cache.prompt_capacity :].all()

    # Fill past the window so stale tokens appear (unless the page size divides the excess).
    for _ in range(8):
        cache.commit_chunk()
    mask = cache.attention_mask(device)
    lo = cache.prompt_capacity
    if cache.stale_tokens:
        assert not mask[:, lo : lo + cache.stale_tokens].any()
    assert mask[:, lo + cache.stale_tokens :].all()
    assert mask.shape[1] == cache.seq_len


def test_copy_batch_block_offsets_encodes_our_table(cache):
    cache.open(prompt_len=3)
    for _ in range(5):
        cache.commit_chunk()
    dst = torch.full((1, 1, 2, cache.max_blocks_per_seq), -7, dtype=torch.int32, device="cuda")
    cache.copy_batch_block_offsets(dst, [cache.REQUEST_ID], 1, 1, 1)
    torch.cuda.synchronize()
    table = torch.tensor(cache.block_table(), dtype=torch.int32, device="cuda")
    n = table.numel()
    scale = int(cache.index_scales[0])
    kv_offset = int(cache.kv_offset[0])
    torch.testing.assert_close(dst[0, 0, 0, :n], table * scale)
    torch.testing.assert_close(dst[0, 0, 1, :n], table * scale + kv_offset)
    assert torch.count_nonzero(dst[0, 0, :, n:]) == 0, "unused entries are the safe page 0"

    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, [1], 1, 1, 1)
    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, [cache.REQUEST_ID, 1], 1, 2, 2)


def test_rejects_bad_geometry():
    with pytest.raises(ValueError):
        make_cache(0)
    for bad in (8, 12, 16, 64, 128):
        with pytest.raises(ValueError, match="must be 32"):
            make_cache(bad)
    with pytest.raises(ValueError):
        CausalKVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
            dtype=torch.float32,
            tokens_per_block=8,
            prompt_capacity=8,
            window_tokens=8,
            chunk_tokens=8,
        )
