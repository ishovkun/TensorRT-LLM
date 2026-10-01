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
"""CausalKVCacheManager: pages, the fixed region, the rolling window, and the
per-block rows attention reads."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 64
DEVICE = torch.device("cuda")
DTYPE = torch.float16  # integer stamps up to 2048 stay exact; bf16 loses them above 256


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
        fixed_capacity=tpb + 8,
        window_tokens=2 * tpb,
        chunk_tokens=tpb + 8,
        causal_block_sizes=(tpb + 8, (tpb + 8) // 4),  # the whole chunk, and four blocks
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


def stamped_kv(positions):
    """K/V whose every element is the token's logical position: reading a slot back
    tells which token sits there."""
    stamp = positions.to(DTYPE)[:, None, None].expand(-1, NUM_KV_HEADS, HEAD_DIM).contiguous()
    return stamp, -stamp


def read_kv(cache, layer, positions):
    """Gather ``[T, num_kv_heads, head_dim]`` K and V at logical ``positions`` straight from the pool."""
    buf = cache.kv_buffer(layer)
    table = cache.table.long()
    page = table[positions // cache.tokens_per_block]
    slot = positions % cache.tokens_per_block
    return buf[page, 0, :, slot, :], buf[page, 1, :, slot, :]


def write_kv_reference(cache, layer, positions, k, v):
    """Scatter one token at a time; the slow, obviously-correct write ``write_range`` must match."""
    buf = cache.kv_buffer(layer)
    table = cache.table.long()
    page = table[positions // cache.tokens_per_block]
    slot = positions % cache.tokens_per_block
    buf[page, 0, :, slot, :] = k
    buf[page, 1, :, slot, :] = v


def open_with_fixed(cache, fixed_len):
    """Open, write ``fixed_len`` random tokens at position 0 in every layer, pin them."""
    cache.open()
    k, v = rand_kv(fixed_len)
    for layer in range(NUM_LAYERS):
        cache.write_range(layer, 0, k, v)
    if fixed_len:
        cache.pin(fixed_len)
    return k, v


def row_keys(cache, layer, block_size, i):
    """The K stamps block ``i``'s row presents to the kernel, in row order."""
    buf = cache.kv_buffer(layer)
    rows = cache.page_table(block_size)
    _, kv_len = cache.causal_block_lengths(block_size)
    n = int(kv_len[i])
    pos = torch.arange(n, device=DEVICE)
    page = rows[i].long()[pos // cache.tokens_per_block]
    slot = pos % cache.tokens_per_block
    return buf[page, 0, 0, slot, 0].float()


def test_geometry_holds_fixed_window_stale_and_chunk(cache):
    tpb = cache.tokens_per_block
    assert cache.page_view_scale == NUM_LAYERS, "layers share a slot; one layer's view is strided"
    tokens = cache.fixed_capacity + cache.window_tokens + cache.chunk_tokens
    assert cache.num_pages == -(-tokens // tpb) + 1
    assert cache.capacity == cache.num_pages * tpb
    # Worst case resident: full fixed region, window plus a page of stale, a chunk.
    assert cache.capacity >= tokens + tpb - 1


def test_open_backs_every_page_once_and_publishes_the_table(cache):
    cache.open()
    assert cache.fixed_tokens == cache.history_tokens == cache.past_tokens == 0
    table = cache.block_table()
    assert len(table) == cache.num_pages
    assert len(set(table)) == cache.num_pages, "pages must be distinct"
    assert min(table) >= 0
    torch.testing.assert_close(
        cache.table.cpu(), torch.tensor(table, dtype=torch.int32) * cache.page_view_scale
    )
    with pytest.raises(RuntimeError):
        cache.open()
    cache.close()
    cache.close()


def test_pin_makes_fresh_tokens_or_the_oldest_history_fixed(cache):
    tpb = cache.tokens_per_block
    cache.open()
    with pytest.raises(RuntimeError):
        make_cache(tpb).table  # not open
    k, v = rand_kv(13)
    for layer in range(NUM_LAYERS):
        cache.write_range(layer, 0, k, v)
    cache.pin(13)
    assert (cache.fixed_tokens, cache.history_tokens, cache.past_tokens) == (13, 0, 13)
    assert cache._fixed_pages == 0  # 13 tokens do not fill a page; the page is shared
    with pytest.raises(ValueError):
        cache.pin(0)
    with pytest.raises(ValueError):
        cache.pin(cache.fixed_capacity)  # over capacity

    # A chunk becomes history; its oldest 8 tokens become fixed (sink frames).
    chunk = cache.chunk_tokens
    hk, hv = rand_kv(chunk)
    for layer in range(NUM_LAYERS):
        cache.write_range(layer, cache.past_tokens, hk, hv)
    cache.commit()
    with pytest.raises(ValueError):
        cache.pin(chunk + 1)  # more than the resident history
    version = cache.table_version
    cache.pin(8)
    assert (cache.fixed_tokens, cache.history_tokens, cache.past_tokens) == (
        21,
        chunk - 8,
        13 + chunk,
    )
    assert cache.table_version > version
    for layer in range(NUM_LAYERS):
        k_back, v_back = read_kv(cache, layer, torch.arange(21, device=DEVICE))
        torch.testing.assert_close(k_back, torch.cat([k, hk[:8]]))
        torch.testing.assert_close(v_back, torch.cat([v, hv[:8]]))
    # Pinned tokens survive rotations like any fixed token.
    for _ in range(6):
        _, kk, vv = (None, *rand_kv(chunk))
        for layer in range(NUM_LAYERS):
            cache.write_range(layer, cache.past_tokens, kk, vv)
        cache.commit()
    for layer in range(NUM_LAYERS):
        k_back, _ = read_kv(cache, layer, torch.arange(21, device=DEVICE))
        torch.testing.assert_close(k_back, torch.cat([k, hk[:8]]))


def test_write_range_matches_indexed_write(cache):
    """The run-based fast write lands bytes exactly where the per-token write does."""
    open_with_fixed(cache, 20)
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


def test_eviction_keeps_the_window_and_the_fixed_region(cache):
    """Content check across many chunks with a fixed region that shares a page with the history."""
    fixed = 13
    pk, pv = open_with_fixed(cache, fixed)
    tpb, chunk, window = cache.tokens_per_block, cache.chunk_tokens, cache.window_tokens
    allocated = sorted(cache.block_table())
    written = []  # one stamp per committed generator token, oldest first
    saw_stale = saw_rotation = False

    for c in range(1, 12):
        stamp = torch.full((chunk, NUM_KV_HEADS, HEAD_DIM), float(c), device=DEVICE, dtype=DTYPE)
        for layer in range(NUM_LAYERS):
            cache.write_range(layer, cache.past_tokens, stamp, -stamp)
        before = cache.table_version
        cache.commit()
        written.extend([c] * chunk)
        saw_rotation |= cache.table_version != before

        # Whole-page eviction leaves fewer than one page of stale tokens resident.
        assert cache.history_tokens <= window + tpb - 1
        stale = max(0, cache.history_tokens - window)
        assert stale < tpb
        saw_stale |= stale > 0
        assert cache.history_tokens - stale == min(len(written), window)
        assert cache.past_tokens == fixed + cache.history_tokens

        for layer in range(NUM_LAYERS):
            hist = torch.arange(fixed, fixed + cache.history_tokens, device=DEVICE)
            k_back, v_back = read_kv(cache, layer, hist)
            expect = torch.tensor(written[-cache.history_tokens :], device=DEVICE, dtype=DTYPE)
            torch.testing.assert_close(k_back[:, 0, 0], expect)
            torch.testing.assert_close(v_back[:, 0, 0], -expect)
            # The fixed region survived every rotation of the page it shares with the history.
            k_p, v_p = read_kv(cache, layer, torch.arange(fixed, device=DEVICE))
            torch.testing.assert_close(k_p, pk)
            torch.testing.assert_close(v_p, pv)

        # Eviction recycles pages; it never allocates or frees any.
        table = cache.block_table()
        assert sorted(table) == allocated
        assert len(set(table)) == len(table)

    assert saw_stale, "test geometry should produce stale tokens"
    assert saw_rotation, "test geometry should rotate the table"


def test_block_rows_present_exactly_the_window(cache):
    """Every block's row holds the fixed region, exactly ``window_tokens`` of history
    before the block, the earlier blocks and itself; nothing stale, nothing later,
    nothing twice. Checked by stamping every token with its position."""
    chunk, window = cache.chunk_tokens, cache.window_tokens
    fixed = 13
    cache.open()
    for layer in range(NUM_LAYERS):
        cache.write_range(layer, 0, *stamped_kv(torch.arange(fixed, device=DEVICE)))
    cache.pin(fixed)
    num_blocks, size = 4, chunk // 4

    written: list = []  # stamp of every committed token, oldest first

    def expected_stamps(win_start, start, end):
        # Fixed stamps are their positions; history stamps are the positions the
        # tokens had when written (eviction shifted them since); chunk stamps are
        # current positions. Resident history is the tail of what was committed.
        past = cache.past_tokens
        resident = written[len(written) - cache.history_tokens :]
        return sorted(list(range(fixed)) + resident[win_start - fixed :] + list(range(past, end)))

    def check(step):
        past = cache.past_tokens
        positions = torch.arange(past, past + chunk, device=DEVICE)
        k, v = stamped_kv(positions)
        for layer in range(NUM_LAYERS):
            cache.write_chunk(layer, k, v, size)
        rows = cache.page_table(size)
        assert rows.shape[0] == num_blocks and rows.dtype == torch.int32
        cached = cache.cached_tokens(size)
        _, kv_len = cache.causal_block_lengths(size)
        for i in range(num_blocks):
            start, end = past + i * size, past + (i + 1) * size
            win_start = max(fixed, start - window)
            expected = expected_stamps(win_start, start, end)
            assert cached[i] == len(expected) - size
            assert int(kv_len[i]) == len(expected)
            for layer in range(NUM_LAYERS):
                got = row_keys(cache, layer, size, i).tolist()
                assert sorted(got) == expected, f"step {step} block {i} layer {layer}"
        # A one-block forward reads the whole chunk with the same window.
        for layer in range(NUM_LAYERS):
            cache.write_chunk(layer, k, v, chunk)
        win_start = max(fixed, past - window)
        expected = expected_stamps(win_start, past, past + chunk)
        assert cache.cached_tokens(chunk) == [len(expected) - chunk]
        for layer in range(NUM_LAYERS):
            assert sorted(row_keys(cache, layer, chunk, 0).tolist()) == expected
        # A shorter forward of the same block size uses the leading blocks only.
        half = 2 * size
        for layer in range(NUM_LAYERS):
            cache.write_chunk(layer, k[:half], v[:half], size)
        for i in range(2):
            start, end = past + i * size, past + (i + 1) * size
            expected = expected_stamps(max(fixed, start - window), start, end)
            assert sorted(row_keys(cache, layer, size, i).tolist()) == expected
        # The shared pages hold the chunk too, for later blocks and chunks.
        for layer in range(NUM_LAYERS):
            k_back, _ = read_kv(cache, layer, positions)
            torch.testing.assert_close(k_back, k)

    saw_stale = False
    for step in range(8):  # from empty history through saturation and several rotations
        check(step)
        saw_stale |= cache.history_tokens > window
        written.extend(range(cache.past_tokens, cache.past_tokens + chunk))
        cache.commit()
    assert saw_stale, "test geometry should hold stale tokens at some step"

    with pytest.raises(ValueError, match="not declared"):
        cache.causal_block_lengths(chunk // 2)
    with pytest.raises(ValueError, match="not declared"):
        cache.write_chunk(0, *rand_kv(chunk // 2))


def test_causal_block_lengths_follow_the_exact_window(cache):
    open_with_fixed(cache, 3)
    chunk, window = cache.chunk_tokens, cache.window_tokens
    q, kv = cache.causal_block_lengths(chunk)
    assert q.dtype == kv.dtype == torch.int32
    assert q.tolist() == [chunk] and kv.tolist() == [3 + chunk]
    assert cache.max_causal_blocks == 4

    size = chunk // 4

    def exact(history):  # block i sees fixed + min(window, history + i*size) + itself
        return [3 + min(window, history + i * size) + size for i in range(4)]

    q4, kv4 = cache.causal_block_lengths(size)
    assert q4.tolist() == [size] * 4
    assert kv4.tolist() == exact(0)
    assert cache.causal_block_lengths(size)[1] is kv4, "one persistent pair per block size"

    cache.commit()
    # Refreshed in place by commit(), without anyone asking for them again.
    assert kv.tolist() == [3 + min(window, chunk) + chunk]
    assert kv4.tolist() == exact(chunk)

    for _ in range(4):
        cache.commit()
    assert cache.history_tokens > window
    # Saturated: exactly the window before each block, however many stale tokens are resident.
    assert kv.tolist() == [3 + window + chunk]
    assert kv4.tolist() == [3 + window + size] * 4

    with pytest.raises(ValueError, match="not declared"):
        cache.causal_block_lengths(7)


def test_copy_batch_block_offsets_encodes_the_block_rows(cache):
    open_with_fixed(cache, 3)
    for _ in range(5):
        cache.commit()
    chunk = cache.chunk_tokens
    num_blocks, size = 3, chunk // 4  # three of the four blocks: a shorter forward
    dst = torch.full((1, 4, 2, cache.max_blocks_per_seq), -7, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):  # the block size must be declared first
        cache.copy_batch_block_offsets(
            dst, cache.request_ids(num_blocks), 1, num_blocks, num_blocks
        )
    cache.set_block_offsets_block_size(size)
    cache.copy_batch_block_offsets(dst, cache.request_ids(num_blocks), 1, num_blocks, num_blocks)
    torch.cuda.synchronize()
    rows = cache.page_table(size)
    n = rows.shape[1]
    kv_offset = int(cache.kv_offset[0])
    for i in range(num_blocks):
        torch.testing.assert_close(dst[0, i, 0, :n], rows[i] * cache.kv_factor)
        torch.testing.assert_close(dst[0, i, 1, :n], rows[i] * cache.kv_factor + kv_offset)
        assert torch.count_nonzero(dst[0, i, :, n:]) == 0, "unused entries are the safe page 0"
    assert (dst[0, 3] == -7).all(), "rows beyond num_seqs are untouched"

    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, [1], 1, 1, 1)
    with pytest.raises(ValueError):
        cache.copy_batch_block_offsets(dst, cache.request_ids(1) + [1], 1, 2, 2)
    with pytest.raises(ValueError):  # more blocks than the declared size cuts the chunk into
        cache.copy_batch_block_offsets(dst, cache.request_ids(5), 1, 5, 5)


def test_write_chunk_matches_write_range(cache):
    """The device-indexed chunk write lands exactly where the host-sliced write does."""
    open_with_fixed(cache, 13)
    for _ in range(3):
        cache.commit()  # move past off a page boundary and rotate once
    chunk = cache.chunk_tokens
    positions = torch.arange(cache.past_tokens, cache.past_tokens + chunk, device=DEVICE)
    for layer in range(NUM_LAYERS):
        k, v = rand_kv(chunk)
        cache.write_range(layer, cache.past_tokens, -k, -v)  # poison first
        cache.write_chunk(layer, k, v, own_tokens=False)
        k_back, v_back = read_kv(cache, layer, positions)
        torch.testing.assert_close(k_back, k)
        torch.testing.assert_close(v_back, v)
    with pytest.raises(ValueError):
        cache.write_chunk(0, *rand_kv(chunk + 1))


def test_inherited_table_accessors_report_the_rotated_table(cache):
    """V2's own accessors must agree with the logical table, or raise."""
    open_with_fixed(cache, 13)
    for _ in range(6):  # rotate at least once
        cache.commit()
    expected = cache.table.tolist()
    assert cache.get_batch_cache_indices(cache.request_ids(1)) == [expected]
    two = cache.get_batch_cache_indices(cache.request_ids(2), num_blocks_per_seq=[3, 2])
    assert two == [expected[:3], expected[:2]]
    flat = cache.get_batch_cache_indices_flat(cache.request_ids(2), [3, 2])
    assert flat.dtype == torch.int32 and flat.tolist() == expected[:3] + expected[:2]
    with pytest.raises(ValueError):
        cache.get_batch_cache_indices([7])


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
            fixed_capacity=8,
            window_tokens=8,
            chunk_tokens=8,
            causal_block_sizes=(8,),
        )
    with pytest.raises(ValueError, match="tile"):
        CausalKVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
            dtype=torch.float16,
            tokens_per_block=32,
            fixed_capacity=8,
            window_tokens=64,
            chunk_tokens=40,
            causal_block_sizes=(40, 7),
        )


def test_commit_takes_the_tokens_actually_written(cache):
    """A rollout's first chunk is one frame: committing it must not promote the rest
    of the chunk's slots to history."""
    tpb = cache.tokens_per_block
    open_with_fixed(cache, 9)
    first = tpb + 3  # shorter than a chunk, not a page multiple
    k, v = rand_kv(first)
    cache.write_range(0, cache.past_tokens, k, v)
    cache.commit(first)
    assert cache.history_tokens == first
    assert cache.past_tokens == 9 + first
    k_back, v_back = read_kv(cache, 0, torch.arange(9, 9 + first, device=DEVICE))
    torch.testing.assert_close(k_back, k)
    torch.testing.assert_close(v_back, v)
    with pytest.raises(ValueError):
        cache.commit(0)
    with pytest.raises(ValueError):
        cache.commit(cache.chunk_tokens + 1)
