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
"""Paged K/V cache for a causal video rollout.

One rollout is one logical sequence laid out as

    [ fixed region | rolling history | in-flight chunk ]

held in a single pool of ``KVCacheManagerV2``. The manager provides the pool
and the pages; this class owns the page table and the sliding window. Nothing
here uses the manager's own sliding-window eviction, block reuse, or request
scheduling, and no ``LlmRequest`` is ever created.

The fixed region is whatever the model wants every frame to see for the whole
rollout: it is written with ``write_range`` and then ``pin``-ned. The cache does
not know whether it holds a text prompt, sink frames, or both. The rolling
history slides: ``commit`` turns the in-flight chunk into history and drops
whole pages from the front once the history exceeds the window, by rotating
the table; no history moves. When the fixed region ends mid-page the first
history tokens share its last page, and after a rotation the fixed region's
tail is copied into the page that now starts the history.

Attention reads exactly what the model's window says. A paged kernel reads
whole pages in table order and cuts only at the end, so every causal block of a
forward gets its own table row: all pages that are legitimate in full, then a
private region holding the few legitimate slots of partially-legitimate pages
followed by the block's own tokens. The kernel is told the cached length up to
the block's own tokens and reads nothing older than the window and nothing
of later blocks. See ``_CausalBlockLayout``.

Rotary positions are the caller's business and are absolute over the rollout;
storage positions here never grow past the resident capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

_DTYPES = {
    torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
    torch.float16: tensorrt_llm.bindings.DataType.HALF,
}

# The V2 sequence id of the one rollout. V2 registers every sequence under an
# integer id that normally comes from an LlmRequest; a rollout has none, so it
# gets a fixed one, as V2's own guard-page and capture-dummy sequences do.
_ROLLOUT_REQUEST_ID = 0


@dataclass
class _CausalBlockLayout:
    """Everything a forward that cuts the chunk into causal blocks of one size reads.

    Block ``i`` holds chunk tokens ``[i*block_size, (i+1)*block_size)`` and may see the
    fixed region, the ``window_tokens`` before it, the earlier blocks and itself. Its
    table row lists every page that is legitimate in full, then ``region_pages``
    private pages holding, in order, the legitimate slots of the partially
    legitimate pages (at most three: where the fixed region ends, where the window
    starts, where the block starts) and the block's own tokens. ``cached[i]`` counts
    the keys before the block's own tokens. Slots holding fixed or history tokens
    are copied at ``commit``/``pin``; slots holding this forward's chunk tokens
    (earlier blocks' tokens in the block's start page, and the block's own) are
    written by every ``write_chunk``. All tensors are persistent and rewritten in
    place by ``commit``/``pin``, never on the forward path.
    """

    num_blocks: int
    block_size: int
    region_pages: int
    regions: torch.Tensor  # [num_blocks, region_pages] int64 layer-0 view indices
    rows: torch.Tensor  # [num_blocks, row_len] int32 layer-0 view indices, 0-padded
    seq_len_q: torch.Tensor  # [num_blocks] int32, all block_size
    seq_len_kv: torch.Tensor  # [num_blocks] int32, cached + block_size
    cached: List[int]  # host copy
    k_rows: torch.Tensor  # [num_blocks*block_size*H] int64: the block tokens' private rows
    v_rows: torch.Tensor
    # Earlier blocks' tokens that sit in a block's start page: chunk rows to gather
    # and the private rows to put them at. A fixed (tpb-1)*H entries per block, padded
    # with a harmless repeat of the block's own first token, so the first ``n``
    # blocks' entries are a prefix and a captured forward replays them.
    extra_src: torch.Tensor  # [num_blocks*(tpb-1)*H] int64 rows into the chunk's [T*H, D]
    extra_k_dst: torch.Tensor
    extra_v_dst: torch.Tensor


class CausalKVCacheManager(KVCacheManagerV2):
    """``KVCacheManagerV2`` driven as one long-lived sequence with a caller-owned table.

    Args:
        num_layers: attention layers that persist K/V (the generator tower).
        num_kv_heads: K/V heads this rank holds, i.e. the head count the attention
            backend it serves was built with. Head sharding (tensor parallel,
            Ulysses) is settled by the model before K/V reach the cache; the
            cache is a per-rank pool and knows nothing about the topology.
        head_dim, dtype: K/V geometry.
        tokens_per_block: page size, a power of two (both paged kernels require
            it). Each backend may add its own constraint and raises if unmet.
        fixed_capacity: most tokens the fixed region may hold (prompt, sink frames).
        window_tokens: history each block may see before itself, in tokens
            (``(window_frames - 1) * tokens_per_frame``).
        chunk_tokens: tokens written per forward (``chunk_frames * tokens_per_frame``).
        causal_block_sizes: every causal block size a forward may use; each must tile
            the chunk. The whole chunk as one block is a denoising step; one frame per
            block is the clean pass. Each size gets its own private pages.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_block: int,
        fixed_capacity: int,
        window_tokens: int,
        chunk_tokens: int,
        causal_block_sizes: Sequence[int],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"CausalKVCacheManager supports bf16/fp16 K/V, got {dtype}")
        if min(tokens_per_block, window_tokens, chunk_tokens) <= 0 or fixed_capacity < 0:
            raise ValueError(
                "tokens_per_block, window_tokens, chunk_tokens must be positive "
                "and fixed_capacity non-negative"
            )
        if tokens_per_block & (tokens_per_block - 1):
            # cuDNN's paged SDPA and trtllm-gen's KV block array both require it.
            raise ValueError(f"tokens_per_block must be a power of two, got {tokens_per_block}")
        sizes = tuple(dict.fromkeys(int(b) for b in causal_block_sizes))
        if not sizes or any(b <= 0 or chunk_tokens % b for b in sizes):
            raise ValueError(
                f"causal_block_sizes {tuple(causal_block_sizes)} must be positive and tile the "
                f"{chunk_tokens}-token chunk"
            )
        self.causal_block_sizes = sizes

        tpb = tokens_per_block
        self.tokens_per_block = tpb
        self.fixed_capacity = fixed_capacity
        self.window_tokens = window_tokens
        self.chunk_tokens = chunk_tokens
        self._num_layers = num_layers
        # Resident tokens peak at fixed + window + (tpb - 1) stale + chunk; the
        # extra page covers the stale tokens and an unaligned fixed-region end.
        self.num_pages = ceil_div(fixed_capacity + window_tokens + chunk_tokens, tpb) + 1
        self.capacity = self.num_pages * tpb
        # Private pages: for every block size, each block needs room for up to three
        # partial pages' worth of slots plus its own tokens.
        self._region_pages_for = lambda block_size: ceil_div(3 * (tpb - 1) + block_size, tpb)
        self._num_private_pages = sum(
            (chunk_tokens // b) * self._region_pages_for(b) for b in sizes
        )
        pool_tokens = (self.num_pages + self._num_private_pages) * tpb

        kv_cache_config = KvCacheConfig(
            max_tokens=pool_tokens,
            max_attention_window=[pool_tokens],
            enable_block_reuse=False,
            host_cache_size=0,  # nothing is ever suspended, so no host tier
        )
        super().__init__(
            kv_cache_config,
            CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tpb,
            max_seq_len=pool_tokens,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=_DTYPES[dtype],
            max_num_tokens=max(chunk_tokens, fixed_capacity, 1),
            execution_stream=stream,
        )
        if self.max_seq_len < pool_tokens:
            raise RuntimeError(
                f"KVCacheManagerV2 clamped max_seq_len to {self.max_seq_len} < {pool_tokens}; "
                "the pool could not back the resident window"
            )
        self._pool_tokens = pool_tokens
        if self.num_pools != 1:
            raise RuntimeError(f"expected one K/V pool, got {self.num_pools}")
        # A physical slot holds every layer's page; ``get_buffers(layer)`` is a view
        # whose index unit is one K/V page pair, so base page ``p`` of a layer sits
        # at view index ``p * page_view_scale`` (1 for a single layer).
        scales = {self.get_layer_page_index_scale(i) for i in range(num_layers)}
        if len(scales) != 1:
            raise RuntimeError(f"layers disagree on page index scale: {sorted(scales)}")
        (scale,) = scales
        if scale % self.kv_factor:
            raise RuntimeError(
                f"page index scale {scale} not a multiple of kv_factor {self.kv_factor}"
            )
        self.page_view_scale = scale // self.kv_factor

        self._kv_cache = None
        self._fixed_tokens = 0
        self._history_tokens = 0
        # Device-side state read by the kernels. Every tensor here lives for the
        # life of an open cache and is rewritten in place by open(), pin() and
        # commit(), never on the forward path, so a CUDA graph captured around a
        # forward keeps reading correct values after the cache moves.
        #
        # The table: layer-0 view indices in logical order, the fixed region's full
        # pages first (never rotated), then the ring: the page the fixed region's
        # tail shares with the first history tokens (if any), history, free pages.
        self._table: Optional[torch.Tensor] = None  # [num_pages] int32
        self._fixed_pages = 0
        self._private: Optional[torch.Tensor] = None  # [num_private_pages] int64 view indices
        self._k_rows: Optional[torch.Tensor] = None  # [chunk_tokens * H] int64 pool rows (shared)
        self._v_rows: Optional[torch.Tensor] = None
        # Scratch for recomputing the rows without allocating: [chunk_tokens] each,
        # int64 except _view_page, which gathers from the int32 table.
        self._logical: Optional[torch.Tensor] = None
        self._logical_page: Optional[torch.Tensor] = None
        self._view_page: Optional[torch.Tensor] = None
        self._slot: Optional[torch.Tensor] = None
        self._head: Optional[torch.Tensor] = None  # [H] int64, constant
        self._layouts: Dict[int, _CausalBlockLayout] = {}  # by causal block size
        self._block_offsets_size: Optional[int] = None
        self._kv_heads_local = 0
        self._rows_per_page = 0  # pool rows of head_dim per view page: 2 * H * tpb
        self._table_version = 0

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> None:
        """Allocate every page the rollout will ever use. The sequence starts empty."""
        if self._kv_cache is not None:
            raise RuntimeError("cache already open; call close() first")

        kv_cache = self._create_kv_cache(_ROLLOUT_REQUEST_ID, None, None, is_dummy=True)
        if kv_cache is None:
            raise RuntimeError("KVCacheManagerV2 has no free sequence slot")
        try:
            if not kv_cache.resume(self._stream.cuda_stream):
                raise RuntimeError("KVCacheManagerV2 could not resume the rollout sequence")
            # We never commit tokens; the pages are ours to address directly.
            kv_cache.stop_committing()
            if not kv_cache.resize(self._pool_tokens):
                raise RuntimeError(
                    f"KVCacheManagerV2 could not back {self._pool_tokens // self.tokens_per_block} "
                    "pages for the rollout"
                )
        except Exception:
            self._release(kv_cache)
            raise

        # V2 owns and may rewrite that buffer: copy it.
        total_pages = self.num_pages + self._num_private_pages
        pages = np.array(kv_cache.get_base_page_indices(0), dtype=np.int64)[:total_pages]
        if pages.size != total_pages or pages.min() < 0:
            self._release(kv_cache)
            raise RuntimeError(f"expected {total_pages} backed pages, got {pages}")
        # Logical order is ours to choose. Ascending keeps consecutive page ids
        # consecutive after rotation whenever V2 handed out a contiguous range, which
        # lets write_range copy a prompt in few pieces; nothing depends on it.
        pages.sort()

        self._kv_cache = kv_cache
        self._fixed_tokens = 0
        self._history_tokens = 0
        self._fixed_pages = 0
        buf = self.kv_buffer(0)
        device = buf.device
        self._kv_heads_local = buf.shape[2]
        self._rows_per_page = 2 * self._kv_heads_local * self.tokens_per_block
        # Layer ``l``'s page view is layer 0's shifted by ``l`` pages; commit-time
        # copies address every layer through layer 0's view with that offset.
        page_bytes = buf[0].numel() * buf.element_size()
        for layer in range(1, self._num_layers):
            if self.kv_buffer(layer).data_ptr() != buf.data_ptr() + layer * page_bytes:
                self._release(kv_cache)
                raise RuntimeError("K/V pool layout changed: layers are not page-interleaved")
        scaled = torch.from_numpy(pages * self.page_view_scale).to(device=device, dtype=torch.int32)
        self._table = scaled[: self.num_pages].clone()
        self._private = scaled[self.num_pages :].to(torch.int64)
        rows = self.chunk_tokens * self._kv_heads_local
        self._k_rows = torch.empty(rows, dtype=torch.int64, device=device)
        self._v_rows = torch.empty(rows, dtype=torch.int64, device=device)
        self._logical, self._logical_page, self._slot = (
            torch.empty(self.chunk_tokens, dtype=torch.int64, device=device) for _ in range(3)
        )
        self._view_page = torch.empty(self.chunk_tokens, dtype=torch.int32, device=device)
        self._head = torch.arange(self._kv_heads_local, dtype=torch.int64, device=device)
        self._layouts = {}
        self._block_offsets_size = None
        first = 0
        for size in self.causal_block_sizes:
            n, region_pages = self.chunk_tokens // size, self._region_pages_for(size)
            self._layouts[size] = self._new_layout(
                size, self._private[first : first + n * region_pages]
            )
            first += n * region_pages
        self._table_version += 1
        self._refresh_device_state()

    def close(self) -> None:
        if self._kv_cache is None:
            return
        kv_cache, self._kv_cache = self._kv_cache, None
        self._table = self._k_rows = self._v_rows = self._private = None
        self._logical = self._logical_page = self._view_page = self._slot = self._head = None
        self._layouts = {}
        self._block_offsets_size = None
        self._release(kv_cache)

    def _release(self, kv_cache) -> None:
        self.kv_cache_map.pop(_ROLLOUT_REQUEST_ID, None)
        kv_cache.discard_pending_stats()
        kv_cache.close()
        self.impl.clear_stats_excluded(_ROLLOUT_REQUEST_ID)
        self.index_mapper.remove_sequence(_ROLLOUT_REQUEST_ID)

    def shutdown(self) -> None:
        self.close()
        super().shutdown()

    # ------------------------------------------------------------------ geometry

    @property
    def fixed_tokens(self) -> int:
        """Tokens every block sees for the whole rollout; logical ``[0, fixed_tokens)``."""
        return self._fixed_tokens

    @property
    def history_tokens(self) -> int:
        """Rolling tokens resident before the in-flight chunk, stale ones included."""
        return self._history_tokens

    @property
    def past_tokens(self) -> int:
        """Logical position where the in-flight chunk's K/V are written."""
        return self._fixed_tokens + self._history_tokens

    @property
    def table_version(self) -> int:
        """Increments whenever the block table changes; lets callers cache derived metadata."""
        return self._table_version

    @property
    def table(self) -> torch.Tensor:
        """``[num_pages]`` int32 device table of layer-0 view indices in logical order."""
        self._require_open()
        return self._table

    def request_ids(self, num_causal_blocks: int) -> List[int]:
        """The request ids attention metadata must carry: one per causal block, all the
        one rollout sequence."""
        return [_ROLLOUT_REQUEST_ID] * num_causal_blocks

    def block_table(self) -> List[int]:
        """Every base page in logical order, read back from the device table. For
        ``write_range`` and for tests; kernels read ``page_table`` directly."""
        self._require_open()
        return (self._table // self.page_view_scale).tolist()

    # ------------------------------------------------------------------ the window

    def pin(self, num_tokens: int) -> None:
        """Make the ``num_tokens`` tokens right after the fixed region fixed too.

        They are either the oldest resident history (sink frames, pinned after their
        clean pass) or, when the history is empty, tokens just written with
        ``write_range`` at ``past_tokens`` (a prompt). No data moves.
        """
        self._require_open()
        if num_tokens <= 0:
            raise ValueError(f"pin of {num_tokens} tokens")
        if self._fixed_tokens + num_tokens > self.fixed_capacity:
            raise ValueError(
                f"pinning {num_tokens} tokens exceeds fixed_capacity {self.fixed_capacity} "
                f"(fixed so far: {self._fixed_tokens})"
            )
        if 0 < self._history_tokens < num_tokens:
            raise ValueError(
                f"pin of {num_tokens} tokens but only {self._history_tokens} history tokens are "
                "resident; pin the oldest history or fresh tokens, not a mix"
            )
        if self._history_tokens:
            self._history_tokens -= num_tokens
        self._fixed_tokens += num_tokens
        self._fixed_pages = self._fixed_tokens // self.tokens_per_block
        self._table_version += 1
        self._refresh_device_state()

    def commit(self, num_tokens: Optional[int] = None) -> None:
        """The in-flight chunk's K/V are final; advance the window by ``num_tokens``
        (default: a full chunk). A rollout's first chunk is a single frame."""
        self._require_open()
        if num_tokens is None:
            num_tokens = self.chunk_tokens
        if not 0 < num_tokens <= self.chunk_tokens:
            raise ValueError(
                f"commit of {num_tokens} tokens; a chunk holds at most {self.chunk_tokens}"
            )
        self._history_tokens += num_tokens
        excess = self._history_tokens - self.window_tokens
        if excess > 0:
            drop_pages = excess // self.tokens_per_block
            if drop_pages:
                # Recycle: the oldest pages become the newest free pages. Rotated in
                # place, so the table's address, which captured graphs hold, never changes.
                ring = self._table[self._fixed_pages :]
                old_head = ring[0].clone()
                torch.ops.trtllm.rotate_rows_(ring, -drop_pages)  # dropped pages go to the tail
                self._refill_shared_page(old_head, ring[0])
                self._history_tokens -= drop_pages * self.tokens_per_block
                self._table_version += 1
        self._refresh_device_state()

    def _refill_shared_page(self, old_page: torch.Tensor, new_page: torch.Tensor) -> None:
        """Copy the fixed region's tail into the page that now starts the history.

        ``old_page``/``new_page`` are 0-d int32 view indices on the device; indexing
        with them keeps the commit free of host syncs.
        """
        tail = self._fixed_tokens % self.tokens_per_block
        if tail == 0:
            return
        old_page, new_page = old_page.long().view(1), new_page.long().view(1)
        for layer in range(self._num_layers):
            buf = self.kv_buffer(layer)
            buf[new_page, :, :, :tail] = buf[old_page, :, :, :tail]

    # ------------------------------------------------------------------ causal blocks

    def _new_layout(self, size: int, private: torch.Tensor) -> _CausalBlockLayout:
        n = self.chunk_tokens // size
        region_pages = self._region_pages_for(size)
        device = self._table.device
        heads = self._kv_heads_local
        extra = n * (self.tokens_per_block - 1) * heads
        return _CausalBlockLayout(
            num_blocks=n,
            block_size=size,
            region_pages=region_pages,
            regions=private.view(n, region_pages),
            rows=torch.zeros((n, self.num_pages + region_pages), dtype=torch.int32, device=device),
            seq_len_q=torch.full((n,), size, dtype=torch.int32, device=device),
            seq_len_kv=torch.empty(n, dtype=torch.int32, device=device),
            cached=[0] * n,
            k_rows=torch.empty(n * size * heads, dtype=torch.int64, device=device),
            v_rows=torch.empty(n * size * heads, dtype=torch.int64, device=device),
            extra_src=torch.empty(extra, dtype=torch.int64, device=device),
            extra_k_dst=torch.empty(extra, dtype=torch.int64, device=device),
            extra_v_dst=torch.empty(extra, dtype=torch.int64, device=device),
        )

    def _layout(self, causal_block_size: int) -> _CausalBlockLayout:
        self._require_open()
        layout = self._layouts.get(causal_block_size)
        if layout is None:
            raise ValueError(
                f"causal block size {causal_block_size} was not declared; this cache serves "
                f"{self.causal_block_sizes}"
            )
        return layout

    @property
    def max_causal_blocks(self) -> int:
        """Most causal blocks any declared size cuts the chunk into."""
        return max(self.chunk_tokens // b for b in self.causal_block_sizes)

    def causal_block_lengths(self, causal_block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-block int32 device tensors ``(seq_len_q, seq_len_kv)`` for every block of
        that size in a full chunk; a forward of fewer tokens uses the leading ones.

        ``seq_len_kv[i]`` is the number of keys in block ``i``'s table row: the fixed
        region, the window before the block, the earlier blocks and the block itself.
        Persistent, refreshed by ``commit``/``pin``, so a captured forward keeps
        reading the right lengths.
        """
        layout = self._layout(causal_block_size)
        return layout.seq_len_q, layout.seq_len_kv

    def cached_tokens(self, causal_block_size: int) -> List[int]:
        """Keys in each block's row before the block's own tokens (host ints)."""
        return list(self._layout(causal_block_size).cached)

    def page_table(self, causal_block_size: int) -> torch.Tensor:
        """``[num_blocks, row_len]`` int32 device table of layer-0 view indices.

        Row ``i`` is block ``i``'s key sequence; entries past its length are 0 and
        never read. Persistent, refreshed in place by ``commit``/``pin``.
        """
        return self._layout(causal_block_size).rows

    def set_block_offsets_block_size(self, causal_block_size: int) -> None:
        """Declare the block size the next ``copy_batch_block_offsets`` describes."""
        self._layout(causal_block_size)
        self._block_offsets_size = causal_block_size

    # ------------------------------------------------------------------ device state

    def _refresh_device_state(self) -> None:
        """Rewrite every kernel-facing device tensor for the current table, fixed region
        and ``past``. Runs on the host in ``open``, ``pin`` and ``commit``, never on the
        forward path."""
        # Pool viewed as rows of head_dim: row of (page, kv, head, slot) is
        # ((page*2 + kv)*H + head)*tpb + slot, with page already a view index.
        tpb, heads = self.tokens_per_block, self._kv_heads_local
        torch.arange(self.past_tokens, self.past_tokens + self.chunk_tokens, out=self._logical)
        torch.floor_divide(self._logical, tpb, out=self._logical_page)
        torch.index_select(self._table, 0, self._logical_page, out=self._view_page)
        torch.remainder(self._logical, tpb, out=self._slot)
        k_rows = self._k_rows.view(self.chunk_tokens, heads)
        k_rows.copy_(self._view_page[:, None]).mul_(2 * heads).add_(self._head).mul_(tpb)
        k_rows.add_(self._slot[:, None])
        self._v_rows.copy_(self._k_rows).add_(heads * tpb)
        for layout in self._layouts.values():
            self._refresh_layout(layout)

    def _refresh_layout(self, blk: _CausalBlockLayout) -> None:
        """Rebuild one block size's rows, lengths and private regions for the current state."""
        tpb, heads, rpp = self.tokens_per_block, self._kv_heads_local, self._rows_per_page
        table = self._table.cpu().numpy().astype(np.int64)
        regions = blk.regions.cpu().numpy()
        fixed, past, window = self._fixed_tokens, self.past_tokens, self.window_tokens
        n, size = blk.num_blocks, blk.block_size
        rows = np.zeros((n, blk.rows.shape[1]), dtype=np.int32)
        cached: List[int] = []
        k_rows = np.empty((n, size, heads), dtype=np.int64)
        head_rows = np.arange(heads, dtype=np.int64)[None, :] * tpb  # (1, H)
        piece_src: List[np.ndarray] = []
        piece_dst: List[np.ndarray] = []
        per_block = (tpb - 1) * heads
        extra_src = np.zeros((n, per_block), dtype=np.int64)
        extra_dst = np.zeros((n, per_block), dtype=np.int64)
        for i in range(n):
            # Dynamic pieces for this block, padded to a fixed count with a repeat of
            # the block's own first token (same value written twice).
            dyn_src: List[np.ndarray] = []
            dyn_dst: List[np.ndarray] = []
            start = past + i * size
            win_start = max(fixed, start - window)
            # Legitimate keys before the block: [0, fixed) and [win_start, start).
            # Per logical page, how many of its slots are legitimate.
            num_pages = ceil_div(start, tpb) if start else 0
            page = np.arange(num_pages, dtype=np.int64)
            lo, hi = page * tpb, (page + 1) * tpb
            legit = np.clip(np.minimum(hi, fixed) - lo, 0, tpb) + np.clip(
                np.minimum(hi, start) - np.maximum(lo, win_start), 0, tpb
            )
            whole = page[legit == tpb]
            partial = page[(legit > 0) & (legit < tpb)]
            # Slots of the partial pages, in logical order. Fixed and history tokens
            # are copied now; tokens of this forward's chunk (earlier blocks in this
            # block's start page) are written by write_chunk.
            region = regions[i]
            num_pieces = 0
            for p in partial.tolist():
                for slot in range(tpb):
                    pos = p * tpb + slot
                    if not (pos < fixed or win_start <= pos < start):
                        continue
                    dst = region[num_pieces // tpb] * rpp + (num_pieces % tpb)
                    if pos < past:
                        piece_src.append(np.array([table[p] * rpp + slot]))
                        piece_dst.append(np.array([dst]))
                    else:
                        dyn_src.append((pos - past) * heads + np.arange(heads))
                        dyn_dst.append(dst + head_rows[0])
                    num_pieces += 1
            # The row: whole pages, then the region.
            row = np.concatenate([table[whole], region])
            rows[i, : row.size] = row
            cached.append(int(whole.size * tpb + num_pieces))
            # The block's own tokens follow the pieces inside the region.
            j = num_pieces + np.arange(size)
            k_rows[i] = (region[j // tpb] * rpp + (j % tpb))[:, None] + head_rows
            src = np.concatenate(dyn_src) if dyn_src else np.zeros(0, dtype=np.int64)
            dst = np.concatenate(dyn_dst) if dyn_dst else np.zeros(0, dtype=np.int64)
            if src.size > per_block:
                raise RuntimeError("more chunk tokens in a start page than a page holds")
            extra_src[i, : src.size] = src
            extra_src[i, src.size :] = i * size * heads  # the block's first token, head 0
            extra_dst[i, : dst.size] = dst
            extra_dst[i, dst.size :] = k_rows[i, 0, 0]
        blk.rows.copy_(torch.from_numpy(rows))
        blk.cached = cached
        blk.seq_len_kv.copy_(torch.tensor([c + size for c in cached], dtype=torch.int32))
        blk.k_rows.copy_(torch.from_numpy(k_rows.reshape(-1)))
        torch.add(blk.k_rows, heads * tpb, out=blk.v_rows)
        blk.extra_src.copy_(torch.from_numpy(extra_src.reshape(-1)))
        blk.extra_k_dst.copy_(torch.from_numpy(extra_dst.reshape(-1)))
        torch.add(blk.extra_k_dst, heads * tpb, out=blk.extra_v_dst)
        if piece_src:
            self._copy_pieces(np.concatenate(piece_src), np.concatenate(piece_dst))

    def _copy_pieces(self, src: np.ndarray, dst: np.ndarray) -> None:
        """Copy pool slots ``src`` to ``dst`` (layer-0 rows of one slot, all heads implied)
        for every layer, K and V: one gather and one scatter over the whole pool."""
        tpb, heads, rpp = self.tokens_per_block, self._kv_heads_local, self._rows_per_page
        device = self._table.device
        # Expand slot rows to (layer, kv, head) rows: + layer*rpp + (kv*H + head)*tpb.
        layer = np.arange(self._num_layers, dtype=np.int64)[None, :, None] * rpp
        kv_head = (np.arange(2 * heads, dtype=np.int64) * tpb)[None, None, :]
        src = (src[:, None, None] + layer + kv_head).reshape(-1)
        dst = (dst[:, None, None] + layer + kv_head).reshape(-1)
        src_t = torch.from_numpy(src).to(device, non_blocking=True)
        dst_t = torch.from_numpy(dst).to(device, non_blocking=True)
        pool = self.kv_buffer(0)
        pool_rows = pool.view(-1, pool.shape[-1])
        pool_rows.index_copy_(0, dst_t, pool_rows.index_select(0, src_t))

    # ------------------------------------------------------------------ direct pool access

    def kv_buffer(self, layer_idx: int) -> torch.Tensor:
        """The layer's pool view ``[view_pages, 2, num_kv_heads, tokens_per_block, head_dim]``.

        Base page ``p`` of this layer is ``buf[p * page_view_scale]``.
        """
        buf = self.get_buffers(layer_idx, kv_layout="HND")
        if buf is None:
            raise RuntimeError(f"layer {layer_idx} has no K/V buffer")
        return buf

    def write_range(self, layer_idx: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write ``k``/``v`` ``[T, num_kv_heads, head_dim]`` at logical ``[start, start + T)``.

        Eager, host-addressed: a partial page at either end is one copy each, whole
        pages go one group of physically consecutive pages at a time. For the fixed
        region and for tests; the per-forward write is ``write_chunk``.
        """
        self._require_open()
        n = k.shape[0]
        if v.shape != k.shape:
            raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if n == 0:
            return
        if not 0 <= start <= start + n <= self.capacity:
            raise ValueError(f"[{start}, {start + n}) outside the cache's {self.capacity} tokens")
        buf = self.kv_buffer(layer_idx)
        tpb = self.tokens_per_block
        table = self.block_table()
        if k.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")

        end = start + n
        first_whole, end_whole = ceil_div(start, tpb), end // tpb  # whole pages [first, end)
        if first_whole > end_whole:  # the whole range lies inside one page
            self._write_page_slots(buf, table[start // tpb], start % tpb, k, v)
            return
        if start % tpb:
            head = first_whole * tpb - start
            self._write_page_slots(buf, table[start // tpb], start % tpb, k[:head], v[:head])
        page_idx = first_whole
        while page_idx < end_whole:
            run_end = page_idx + 1
            while run_end < end_whole and table[run_end] == table[run_end - 1] + 1:
                run_end += 1
            t0 = page_idx * tpb - start
            t1 = run_end * tpb - start
            self._write_whole_pages(buf, table[page_idx], run_end - page_idx, k[t0:t1], v[t0:t1])
            page_idx = run_end
        if end % tpb:
            tail = end_whole * tpb - start
            self._write_page_slots(buf, table[end_whole], 0, k[tail:], v[tail:])

    def _write_page_slots(
        self, buf: torch.Tensor, page: int, slot: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """A run of tokens inside one page, at ``slot`` onward; one copy per tensor."""
        page *= self.page_view_scale
        buf[page, 0, :, slot : slot + k.shape[0]].copy_(k.transpose(0, 1))
        buf[page, 1, :, slot : slot + v.shape[0]].copy_(v.transpose(0, 1))

    def _write_whole_pages(
        self, buf: torch.Tensor, first_page: int, count: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """``count`` whole, physically consecutive pages starting at ``first_page``; one copy per tensor."""
        tpb, vs = self.tokens_per_block, self.page_view_scale
        dst = slice(first_page * vs, (first_page + count) * vs, vs)
        buf[dst, 0].copy_(k.view(count, tpb, -1, k.shape[-1]).transpose(1, 2))
        buf[dst, 1].copy_(v.view(count, tpb, -1, v.shape[-1]).transpose(1, 2))

    def write_chunk(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_block_size: Optional[int] = None,
        *,
        own_tokens: bool = True,
    ) -> None:
        """Write the in-flight chunk's first ``T`` tokens of K/V, cut into causal blocks
        of ``causal_block_size`` (default: one block of ``T``).

        ``k``/``v`` are ``[T, num_kv_heads, head_dim]``. They go to the logical
        positions from ``past_tokens`` (where later blocks and, after commit, later
        chunks read them) and to the blocks' private regions: the earlier blocks'
        tokens each block's start page holds, and, with ``own_tokens``, each block's
        own tokens. ``index_copy_`` kernels over the pool viewed as rows of
        ``head_dim``, driven by row indices rebuilt on ``commit``/``pin``, so the
        write replays correctly inside a CUDA graph. A kernel that writes the new
        tokens into the region itself (trtllm-gen) passes ``own_tokens=False``.
        """
        self._require_open()
        num_tokens, heads, head_dim = k.shape
        if v.shape != k.shape:
            raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if num_tokens > self.chunk_tokens or heads != self._kv_heads_local:
            raise ValueError(
                f"chunk write of [{num_tokens}, {heads}] does not fit "
                f"[{self.chunk_tokens}, {self._kv_heads_local}]"
            )
        buf = self.kv_buffer(layer_idx)
        if k.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")
        rows = num_tokens * heads
        pool_rows = buf.view(-1, head_dim)
        k2, v2 = k.reshape(rows, head_dim), v.reshape(rows, head_dim)
        pool_rows.index_copy_(0, self._k_rows[:rows], k2)
        pool_rows.index_copy_(0, self._v_rows[:rows], v2)
        size = causal_block_size or num_tokens
        layout = self._layout(size)
        if num_tokens % size:
            raise ValueError(f"{num_tokens} tokens do not split into causal blocks of {size}")
        num_blocks = num_tokens // size
        if own_tokens:
            pool_rows.index_copy_(0, layout.k_rows[:rows], k2)
            pool_rows.index_copy_(0, layout.v_rows[:rows], v2)
        if num_blocks > 1:
            extra = num_blocks * (self.tokens_per_block - 1) * heads
            src = k2.index_select(0, layout.extra_src[:extra])
            pool_rows.index_copy_(0, layout.extra_k_dst[:extra], src)
            torch.index_select(v2, 0, layout.extra_src[:extra], out=src)
            pool_rows.index_copy_(0, layout.extra_v_dst[:extra], src)

    # ------------------------------------------------------------------ manager hook

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        """Hand the attention metadata *our* rows instead of the manager's.

        ``dst_tensor`` is ``[num_pools, max_num_sequences, 2, max_blocks_per_seq]``
        of int32: K offsets then V offsets per sequence, each ``page * index_scale``
        (+ ``kv_offset`` for V) -- the same encoding the manager's device copy uses.
        Every request is a causal block of the size declared with
        ``set_block_offsets_block_size``; row ``i`` is that block's key sequence.
        """
        ids = list(request_ids)
        if (
            not 0 < len(ids) <= self.max_causal_blocks
            or any(r != _ROLLOUT_REQUEST_ID for r in ids)
            or num_seqs != len(ids)
            or beam_width != 1
        ):
            raise ValueError(
                f"CausalKVCacheManager serves one sequence (id {_ROLLOUT_REQUEST_ID}) in up to "
                f"{self.max_causal_blocks} causal blocks; got request_ids={ids}, "
                f"num_seqs={num_seqs}, beam_width={beam_width}"
            )
        size = self._block_offsets_size
        if size is None or num_seqs > self._layouts[size].num_blocks:
            raise ValueError(
                f"block offsets for {num_seqs} sequences need set_block_offsets_block_size "
                f"first with a size that cuts the chunk into at least that many blocks; "
                f"declared: {size}"
            )
        rows = self._layouts[size].rows[:num_seqs]
        n = rows.shape[1]
        # The op wants base_page * index_scale; the rows hold view indices, which
        # are base_page * index_scale / kv_factor.
        k_offsets = rows * self.kv_factor
        dst_tensor[0, :num_seqs, 0, :n] = k_offsets
        dst_tensor[0, :num_seqs, 1, :n] = k_offsets + int(self.kv_offset[0])
        if n < dst_tensor.shape[-1]:
            dst_tensor[0, :num_seqs, :, n:].zero_()

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        """The rotated table in V2's per-layer view units, one row per request.

        Overrides the inherited accessor, which would report the pages in
        allocation order and disagree with what the attention backends read.
        Every request is the one sequence.
        """
        ids = list(request_ids)
        if any(r != _ROLLOUT_REQUEST_ID for r in ids):
            raise ValueError(
                f"CausalKVCacheManager serves request {_ROLLOUT_REQUEST_ID} only; got {ids}"
            )
        row = self._table.tolist()
        widths = [len(row)] * len(ids) if num_blocks_per_seq is None else list(num_blocks_per_seq)
        return [row[:n] for n in widths]

    def get_batch_cache_indices_flat(
        self, request_ids: List[int], num_blocks: List[int], layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        rows = self.get_batch_cache_indices(request_ids, layer_idx, num_blocks)
        return torch.tensor([p for row in rows for p in row], dtype=torch.int32)

    def _require_open(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("cache is not open; call open() first")
