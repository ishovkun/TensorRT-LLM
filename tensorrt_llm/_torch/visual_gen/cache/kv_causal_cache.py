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

    [ prompt | generator history | in-flight chunk ]

held in a single pool of ``KVCacheManagerV2``. The manager provides the pool
and the pages; this class owns the block table and the sliding window. Nothing
here uses the manager's own sliding-window eviction, block reuse, or request
scheduling, and no ``LlmRequest`` is ever created.

Paged attention kernels index logical token ``t`` at page ``table[t // tpb]``,
slot ``t % tpb``, and read ``[0, seq_len)`` of that sequence. Two consequences
shape everything below:

* The prompt lives in the same pool as the history, at logical position 0 and
  at its real length, so one attention call reads both. When its length is not
  a page multiple, the first history tokens share its last page.
* Eviction moves the logical stream by whole pages: the table rotates and no
  history moves. The shared page is the one exception -- after a rotation the
  prompt's tail is copied into the new first history page, at most
  ``tpb - 1`` tokens per layer. Up to ``tpb - 1`` history tokens older than the
  window stay resident at the front of the history; they are attended.

Rotary positions are the caller's business and are absolute over the rollout;
storage positions here never grow past the resident capacity.
"""

from __future__ import annotations

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


class CausalKVCacheManager(KVCacheManagerV2):
    """``KVCacheManagerV2`` driven as one long-lived sequence with a caller-owned table.

    Args:
        num_layers: attention layers that persist K/V (the generator tower).
        num_kv_heads, head_dim, dtype: K/V geometry, per layer, per rank.
        tokens_per_block: page size, a power of two (both paged kernels require
            it). Each backend may add its own constraint and raises if unmet.
        prompt_capacity: largest prompt this cache accepts, in tokens
            (``text_cache_max_len``).
        window_tokens: generator history kept attendable, in tokens
            (``window_frames * tokens_per_frame``).
        chunk_tokens: tokens written per forward (``chunk_frames * tokens_per_frame``).
        mapping: parallel topology; ``tp_size`` here must already fold in any
            head sharding done by Ulysses, since the manager divides heads by it.
    """

    # Most causal blocks one forward may cut the in-flight chunk into. Sizes the
    # persistent per-block tensors so their pointers are stable across forwards
    # of different causal block counts.
    MAX_CAUSAL_BLOCKS = 8

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_block: int,
        prompt_capacity: int,
        window_tokens: int,
        chunk_tokens: int,
        mapping: Optional[Mapping] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"CausalKVCacheManager supports bf16/fp16 K/V, got {dtype}")
        if min(tokens_per_block, window_tokens, chunk_tokens) <= 0 or prompt_capacity < 0:
            raise ValueError(
                "tokens_per_block, window_tokens, chunk_tokens must be positive "
                "and prompt_capacity non-negative"
            )
        if tokens_per_block & (tokens_per_block - 1):
            # cuDNN's paged SDPA and trtllm-gen's KV block array both require it.
            raise ValueError(f"tokens_per_block must be a power of two, got {tokens_per_block}")

        self.tokens_per_block = tokens_per_block
        self.prompt_capacity = prompt_capacity
        self.window_tokens = window_tokens
        self.chunk_tokens = chunk_tokens
        self._num_layers = num_layers
        # Resident tokens peak at prompt + window + (tpb - 1) stale + chunk; the
        # extra page covers the stale tokens and an unaligned prompt start.
        self.num_pages = (
            ceil_div(prompt_capacity + window_tokens + chunk_tokens, tokens_per_block) + 1
        )
        self.capacity = self.num_pages * tokens_per_block

        kv_cache_config = KvCacheConfig(
            max_tokens=self.capacity,
            max_attention_window=[self.capacity],
            enable_block_reuse=False,
        )
        super().__init__(
            kv_cache_config,
            CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=self.capacity,
            max_batch_size=1,
            mapping=mapping if mapping is not None else Mapping(world_size=1, tp_size=1, rank=0),
            dtype=_DTYPES[dtype],
            max_num_tokens=max(chunk_tokens, prompt_capacity, 1),
            execution_stream=stream,
        )
        if self.max_seq_len < self.capacity:
            raise RuntimeError(
                f"KVCacheManagerV2 clamped max_seq_len to {self.max_seq_len} < {self.capacity}; "
                "the pool could not back the resident window"
            )
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
        self._prompt_len = 0
        self._history_tokens = 0
        # Device-side state read by the kernels. Every tensor here lives for the
        # life of an open cache and is rewritten in place by open() and commit(),
        # never on the forward path, so a CUDA graph captured around a forward
        # keeps reading correct values after the cache moves.
        #
        # The table: pool-view page indices in logical order, full prompt pages
        # first (never rotated), then the ring: the page shared with the prompt's
        # tail (if any), history, free pages. One identical row per causal block,
        # because cuDNN reads a [blocks, pages] table with a real row stride.
        self._table: Optional[torch.Tensor] = None  # [MAX_CAUSAL_BLOCKS, num_pages] int32
        self._fixed_pages = 0
        self._k_rows: Optional[torch.Tensor] = None  # [chunk_tokens * num_kv_heads] int64 pool rows
        self._v_rows: Optional[torch.Tensor] = None  # same shape; the V rows
        # Scratch for recomputing the rows without allocating: [chunk_tokens] each,
        # int64 except _view_page, which gathers from the int32 table.
        self._logical: Optional[torch.Tensor] = None
        self._logical_page: Optional[torch.Tensor] = None
        self._view_page: Optional[torch.Tensor] = None
        self._slot: Optional[torch.Tensor] = None
        self._head: Optional[torch.Tensor] = None  # [num_kv_heads] int64, constant
        # (num_blocks, block_size) -> (seq_len_q [num_blocks], seq_len_kv [num_blocks]) int32
        self._block_lengths: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._kv_heads_local = 0
        self._table_version = 0

    # ------------------------------------------------------------------ lifecycle

    def open(self, prompt_len: int) -> None:
        """Allocate every page the rollout will ever use and declare the prompt length."""
        if self._kv_cache is not None:
            raise RuntimeError("cache already open; call close() first")
        if not 0 <= prompt_len <= self.prompt_capacity:
            raise ValueError(
                f"prompt_len {prompt_len} exceeds prompt_capacity {self.prompt_capacity}"
            )

        kv_cache = self._create_kv_cache(_ROLLOUT_REQUEST_ID, None, None, is_dummy=True)
        if kv_cache is None:
            raise RuntimeError("KVCacheManagerV2 has no free sequence slot")
        try:
            if not kv_cache.resume(self._stream.cuda_stream):
                raise RuntimeError("KVCacheManagerV2 could not resume the rollout sequence")
            # We never commit tokens; the pages are ours to address directly.
            kv_cache.stop_committing()
            if not kv_cache.resize(self.capacity):
                raise RuntimeError(
                    f"KVCacheManagerV2 could not back {self.num_pages} pages for the rollout"
                )
        except Exception:
            self._release(kv_cache)
            raise

        # V2 owns and may rewrite that buffer: copy it.
        pages = np.array(kv_cache.get_base_page_indices(0), dtype=np.int64)[: self.num_pages]
        if pages.size != self.num_pages or pages.min() < 0:
            self._release(kv_cache)
            raise RuntimeError(f"expected {self.num_pages} backed pages, got {pages}")
        # Logical order is ours to choose. Ascending keeps consecutive page ids
        # consecutive after rotation whenever V2 handed out a contiguous range, which
        # lets write_range copy the prompt in few pieces; nothing depends on it.
        pages.sort()

        self._kv_cache = kv_cache
        self._prompt_len = prompt_len
        self._history_tokens = 0
        self._fixed_pages = prompt_len // self.tokens_per_block
        buf = self.kv_buffer(0)
        device = buf.device
        self._kv_heads_local = buf.shape[2]
        scaled = torch.from_numpy(pages * self.page_view_scale).to(device=device, dtype=torch.int32)
        self._table = scaled.repeat(self.MAX_CAUSAL_BLOCKS, 1)
        rows = self.chunk_tokens * self._kv_heads_local
        self._k_rows = torch.empty(rows, dtype=torch.int64, device=device)
        self._v_rows = torch.empty(rows, dtype=torch.int64, device=device)
        self._logical, self._logical_page, self._slot = (
            torch.empty(self.chunk_tokens, dtype=torch.int64, device=device) for _ in range(3)
        )
        self._view_page = torch.empty(self.chunk_tokens, dtype=torch.int32, device=device)
        self._head = torch.arange(self._kv_heads_local, dtype=torch.int64, device=device)
        self._block_lengths = {}
        self._table_version += 1
        self._refresh_device_state()

    def close(self) -> None:
        if self._kv_cache is None:
            return
        kv_cache, self._kv_cache = self._kv_cache, None
        self._table = self._k_rows = self._v_rows = None
        self._logical = self._logical_page = self._view_page = self._slot = self._head = None
        self._block_lengths = {}
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
    def prompt_len(self) -> int:
        return self._prompt_len

    @property
    def history_tokens(self) -> int:
        """Generator tokens resident before the in-flight chunk, stale ones included."""
        return self._history_tokens

    @property
    def past_tokens(self) -> int:
        """Logical position where the in-flight chunk's K/V are written."""
        return self._prompt_len + self._history_tokens

    @property
    def table_version(self) -> int:
        """Increments whenever the block table changes; lets callers cache derived metadata."""
        return self._table_version

    def request_ids(self, num_causal_blocks: int) -> List[int]:
        """The request ids attention metadata must carry: one per causal block, all the
        one rollout sequence."""
        return [_ROLLOUT_REQUEST_ID] * num_causal_blocks

    def block_table(self) -> List[int]:
        """Every base page in logical order, read back from the device table. For the
        prompt write and for tests; kernels read ``page_table`` directly."""
        self._require_open()
        return (self._table[0] // self.page_view_scale).tolist()

    def page_table(self, num_causal_blocks: int = 1) -> torch.Tensor:
        """``[num_causal_blocks, num_pages]`` int32 device table in ``kv_buffer`` view indices.

        Every row is the one sequence. Persistent, refreshed in place on commit; the
        page table for kernels that read the ``kv_buffer`` view directly (cuDNN).
        """
        self._require_open()
        if not 0 < num_causal_blocks <= self.MAX_CAUSAL_BLOCKS:
            raise ValueError(
                f"num_causal_blocks {num_causal_blocks} outside (0, {self.MAX_CAUSAL_BLOCKS}]"
            )
        return self._table[:num_causal_blocks]

    def causal_block_lengths(
        self, num_causal_blocks: int, causal_block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-block int32 device tensors ``(seq_len_q, seq_len_kv)``, each ``[num_causal_blocks]``.

        A forward cuts the tokens it adds into consecutive causal blocks. Block ``i``
        holds tokens ``[past + i*causal_block_size, past + (i+1)*causal_block_size)`` and attends
        over ``[0, past + (i+1)*causal_block_size)``: full within a block, causal across
        blocks. One persistent pair per blocking, created on first use (never during
        graph capture) and refreshed by ``commit()``, so a captured forward keeps
        reading the right lengths.
        """
        self._require_open()
        if not 0 < num_causal_blocks <= self.MAX_CAUSAL_BLOCKS:
            raise ValueError(
                f"num_causal_blocks {num_causal_blocks} outside (0, {self.MAX_CAUSAL_BLOCKS}]"
            )
        if not 0 < num_causal_blocks * causal_block_size <= self.chunk_tokens:
            raise ValueError(
                f"{num_causal_blocks} causal blocks of {causal_block_size} tokens do not fit a "
                f"{self.chunk_tokens}-token chunk"
            )
        key = (num_causal_blocks, causal_block_size)
        pair = self._block_lengths.get(key)
        if pair is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"causal blocking {key} first seen during CUDA graph capture; run the "
                    "forward eagerly once before capturing"
                )
            device = self._table.device
            pair = (
                torch.full(
                    (num_causal_blocks,), causal_block_size, dtype=torch.int32, device=device
                ),
                torch.empty(num_causal_blocks, dtype=torch.int32, device=device),
            )
            self._block_lengths[key] = pair
            self._fill_block_kv_lengths(key, pair[1])
        return pair

    def commit(self) -> None:
        """The in-flight chunk's K/V are final; advance the window."""
        self._require_open()
        self._history_tokens += self.chunk_tokens
        excess = self._history_tokens - self.window_tokens
        if excess > 0:
            drop_pages = excess // self.tokens_per_block
            if drop_pages:
                # Recycle: the oldest pages become the newest free pages. Rotated in
                # place, so the table's address, which captured graphs hold, never changes.
                ring = self._table[:, self._fixed_pages :]  # [MAX_CAUSAL_BLOCKS, num_pages - fixed]
                old_head = ring[0, 0].clone()
                torch.ops.trtllm.rotate_rows_(ring, -drop_pages)  # dropped pages go to the tail
                self._refill_shared_page(old_head, ring[0, 0])
                self._history_tokens -= drop_pages * self.tokens_per_block
                self._table_version += 1
        self._refresh_device_state()

    def _refill_shared_page(self, old_page: torch.Tensor, new_page: torch.Tensor) -> None:
        """Copy the prompt's tail into the page that now starts the history.

        ``old_page``/``new_page`` are 0-d int32 view indices on the device; indexing
        with them keeps the commit free of host syncs.
        """
        tail = self._prompt_len % self.tokens_per_block
        if tail == 0:
            return
        old_page, new_page = old_page.long().view(1), new_page.long().view(1)
        for layer in range(self._num_layers):
            buf = self.kv_buffer(layer)
            buf[new_page, :, :, :tail] = buf[old_page, :, :, :tail]

    def _refresh_device_state(self) -> None:
        """Rewrite every kernel-facing device tensor for the current table and ``past``.

        Runs on the host in ``open()`` and ``commit()``, never on the forward path:
        the page table, the pool row of every (token, head) of the in-flight chunk,
        and the K/V lengths of every blocking seen so far.
        """
        # Pool viewed as rows of head_dim: row of (page, kv, head, slot) is
        # ((page*2 + kv)*H + head)*tpb + slot, with page already a view index.
        tpb, heads = self.tokens_per_block, self._kv_heads_local
        torch.arange(self.past_tokens, self.past_tokens + self.chunk_tokens, out=self._logical)
        torch.floor_divide(self._logical, tpb, out=self._logical_page)
        torch.index_select(self._table[0], 0, self._logical_page, out=self._view_page)
        torch.remainder(self._logical, tpb, out=self._slot)
        k_rows = self._k_rows.view(self.chunk_tokens, heads)
        k_rows.copy_(self._view_page[:, None]).mul_(2 * heads).add_(self._head).mul_(tpb)
        k_rows.add_(self._slot[:, None])
        self._v_rows.copy_(self._k_rows).add_(heads * tpb)

        for key, (_, kv_lengths) in self._block_lengths.items():
            self._fill_block_kv_lengths(key, kv_lengths)

    def _fill_block_kv_lengths(self, key: Tuple[int, int], kv_lengths: torch.Tensor) -> None:
        num_blocks, block_size = key
        ends = torch.arange(1, num_blocks + 1, dtype=torch.int32, device=kv_lengths.device)
        kv_lengths.copy_(ends * block_size + self.past_tokens)

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
        pages go one group of physically consecutive pages at a time. For the prompt
        (``start=0``) and for tests; the per-forward write is ``write_chunk``.
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

    def write_chunk(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write the in-flight chunk's first ``T`` tokens of K/V at ``past_tokens``.

        ``k``/``v`` are ``[T, num_kv_heads, head_dim]``. Two ``index_copy_`` kernels over
        the pool viewed as rows of ``head_dim``, driven by row indices the cache
        rebuilds on ``commit()``, so the write replays correctly inside a CUDA graph.
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
        pool_rows.index_copy_(0, self._k_rows[:rows], k.reshape(rows, head_dim))
        pool_rows.index_copy_(0, self._v_rows[:rows], v.reshape(rows, head_dim))

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
        """Hand the attention metadata *our* table instead of the manager's.

        ``dst_tensor`` is ``[num_pools, max_num_sequences, 2, max_blocks_per_seq]``
        of int32: K offsets then V offsets per sequence, each ``page * index_scale``
        (+ ``kv_offset`` for V) -- the same encoding the manager's device copy uses.
        Every request is a causal block of the one sequence, so every row gets the
        same table.
        """
        ids = list(request_ids)
        if (
            not 0 < len(ids) <= self.MAX_CAUSAL_BLOCKS
            or any(r != _ROLLOUT_REQUEST_ID for r in ids)
            or num_seqs != len(ids)
            or beam_width != 1
        ):
            raise ValueError(
                f"CausalKVCacheManager serves one sequence (id {_ROLLOUT_REQUEST_ID}) in up to "
                f"{self.MAX_CAUSAL_BLOCKS} causal blocks; got request_ids={ids}, num_seqs={num_seqs}, "
                f"beam_width={beam_width}"
            )
        # The op wants base_page * index_scale; the table holds view indices, which
        # are base_page * index_scale / kv_factor.
        n = self.num_pages
        k_offsets = self._table[0] * self.kv_factor
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
        row = self._table[0].tolist()
        widths = [len(row)] * len(ids) if num_blocks_per_seq is None else list(num_blocks_per_seq)
        return [row[:n] for n in widths]

    def get_batch_cache_indices_flat(
        self, request_ids: List[int], num_blocks: List[int], layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        rows = self.get_batch_cache_indices(request_ids, layer_idx, num_blocks)
        return torch.tensor([p for row in rows for p in row], dtype=torch.int32)

    def _require_open(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("cache is not open; call open(prompt_len) first")
