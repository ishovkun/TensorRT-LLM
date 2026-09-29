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

from collections import deque
from typing import Deque, Iterator, List, Optional, Tuple

import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

_DTYPES = {
    torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
    torch.float16: tensorrt_llm.bindings.DataType.HALF,
}


# The only page size with shipped trtllm-gen paged context kernels.
_TOKENS_PER_PAGE = 32


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _contiguous_runs(pages: List[int]) -> Iterator[Tuple[int, int, int]]:
    """Split a page list into ``(first_index, first_page, count)`` runs of consecutive pages."""
    i = 0
    while i < len(pages):
        j = i + 1
        while j < len(pages) and pages[j] == pages[j - 1] + 1:
            j += 1
        yield i, pages[i], j - i
        i = j


class CausalKVCacheManager(KVCacheManagerV2):
    """``KVCacheManagerV2`` driven as one long-lived sequence with a caller-owned table.

    Args:
        num_layers: attention layers that persist K/V (the generator tower).
        num_kv_heads, head_dim, dtype: K/V geometry, per layer, per rank.
        tokens_per_block: page size; must be 32, the only value with shipped
            trtllm-gen paged context kernels.
        prompt_capacity: largest prompt this cache accepts, in tokens
            (``text_cache_max_len``).
        window_tokens: generator history kept attendable, in tokens
            (``window_frames * tokens_per_frame``).
        chunk_tokens: tokens written per forward (``chunk_frames * tokens_per_frame``).
        mapping: parallel topology; ``tp_size`` here must already fold in any
            head sharding done by Ulysses, since the manager divides heads by it.
    """

    REQUEST_ID = 0
    # Most segments one forward may cut the in-flight chunk into (the clean pass uses
    # one per frame). Sizes the persistent per-segment tensors so their pointers are
    # stable across forwards of different segment counts.
    MAX_SEGMENTS = 8

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
        if tokens_per_block != _TOKENS_PER_PAGE:
            # The page size is part of the trtllm-gen kernel hash and the only paged
            # context cubins shipped are for 32 tokens per page. Any other value misses
            # the lookup, and the attention op then falls back to an unfused path that
            # silently ignores the cached prefix instead of raising.
            raise ValueError(f"tokens_per_block must be {_TOKENS_PER_PAGE}, got {tokens_per_block}")

        self.tokens_per_block = tokens_per_block
        self.prompt_capacity = prompt_capacity
        self.window_tokens = window_tokens
        self.chunk_tokens = chunk_tokens
        self._num_layers = num_layers
        # Resident tokens peak at prompt + window + (tpb - 1) stale + chunk; the
        # extra page covers the stale tokens and an unaligned prompt start.
        self.num_pages = (
            _ceil_div(prompt_capacity + window_tokens + chunk_tokens, tokens_per_block) + 1
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
        self._stale_tokens = 0
        self._fixed: List[int] = []  # prompt-only pages, never rotated
        self._ring: Deque[int] = deque()  # shared page (if any), history, free pages
        self._page_table: Optional[torch.Tensor] = None
        self._seq_len_q: Optional[torch.Tensor] = None
        self._seq_len_kv: Optional[torch.Tensor] = None
        self._segment_key: Optional[tuple] = None
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

        kv_cache = self._create_kv_cache(self.REQUEST_ID, None, None, is_dummy=True)
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

        pages = [int(p) for p in kv_cache.get_base_page_indices(0)][: self.num_pages]
        if len(pages) != self.num_pages or min(pages) < 0:
            self._release(kv_cache)
            raise RuntimeError(f"expected {self.num_pages} backed pages, got {pages}")
        # Logical order is ours to choose; ascending keeps the ring a cyclic shift of
        # consecutive pages, so a chunk's pages form at most two contiguous runs.
        pages.sort()

        self._kv_cache = kv_cache
        self._prompt_len = prompt_len
        self._history_tokens = 0
        self._stale_tokens = 0
        full_prompt_pages = prompt_len // self.tokens_per_block
        self._fixed = pages[:full_prompt_pages]
        self._ring = deque(pages[full_prompt_pages:])
        device = self.kv_buffer(0).device
        self._page_table = torch.empty(
            self.MAX_SEGMENTS, self.num_pages, dtype=torch.int32, device=device
        )
        self._seq_len_q = torch.zeros(self.MAX_SEGMENTS, dtype=torch.int32, device=device)
        self._seq_len_kv = torch.zeros(self.MAX_SEGMENTS, dtype=torch.int32, device=device)
        self._segment_key = None
        self._publish_table()

    def close(self) -> None:
        if self._kv_cache is None:
            return
        kv_cache, self._kv_cache = self._kv_cache, None
        self._page_table = self._seq_len_q = self._seq_len_kv = None
        self._release(kv_cache)

    def _release(self, kv_cache) -> None:
        self.kv_cache_map.pop(self.REQUEST_ID, None)
        kv_cache.discard_pending_stats()
        kv_cache.close()
        self.impl.clear_stats_excluded(self.REQUEST_ID)
        self.index_mapper.remove_sequence(self.REQUEST_ID)

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
    def stale_tokens(self) -> int:
        """Leading history tokens older than the window that whole-page eviction could not drop."""
        return self._stale_tokens

    @property
    def past_tokens(self) -> int:
        """Logical position where the in-flight chunk's K/V are written."""
        return self._prompt_len + self._history_tokens

    @property
    def seq_len(self) -> int:
        """Logical length attended by a full-chunk forward: everything resident plus the chunk."""
        return self.past_tokens + self.chunk_tokens

    @property
    def table_version(self) -> int:
        """Increments whenever the block table changes; lets callers cache derived metadata."""
        return self._table_version

    def block_table(self) -> List[int]:
        """Every base page in logical order. Kernels read only the first ``ceil(seq_len / tpb)``."""
        self._require_open()
        return self._fixed + list(self._ring)

    def page_table(self, num_segments: int = 1) -> torch.Tensor:
        """``[num_segments, num_pages]`` int32 device table in ``kv_buffer`` view indices.

        Every row is the one sequence. Persistent, refreshed in place on commit; the
        page table for kernels that read the ``kv_buffer`` view directly (cuDNN).
        """
        self._require_open()
        if not 0 < num_segments <= self.MAX_SEGMENTS:
            raise ValueError(f"num_segments {num_segments} outside (0, {self.MAX_SEGMENTS}]")
        return self._page_table[:num_segments]

    def segment_lengths(
        self, num_segments: int, segment_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-segment int32 device tensors ``(seq_len_q, seq_len_kv)``, each ``[num_segments]``.

        A forward cuts the tokens it adds into consecutive segments. Segment ``i``
        holds tokens ``[past + i*segment_len, past + (i+1)*segment_len)`` and attends
        over ``[0, past + (i+1)*segment_len)``: full within a segment, causal across
        segments. Paged kernels read these lengths from device memory; they are
        rewritten in place only when the values change, so every layer of a forward
        and every denoising step over one chunk share a single write.
        """
        self._require_open()
        if not 0 < num_segments <= self.MAX_SEGMENTS:
            raise ValueError(f"num_segments {num_segments} outside (0, {self.MAX_SEGMENTS}]")
        if not 0 < num_segments * segment_len <= self.chunk_tokens:
            raise ValueError(
                f"{num_segments} segments of {segment_len} tokens do not fit a "
                f"{self.chunk_tokens}-token chunk"
            )
        key = (self.past_tokens, num_segments, segment_len)
        if key != self._segment_key:
            ends = torch.arange(1, num_segments + 1, dtype=torch.int32) * segment_len
            self._seq_len_q[:num_segments].fill_(segment_len)
            self._seq_len_kv[:num_segments].copy_(ends + self.past_tokens)
            self._segment_key = key
        return self._seq_len_q[:num_segments], self._seq_len_kv[:num_segments]

    def commit(self) -> None:
        """The in-flight chunk's K/V are final; advance the window."""
        self._require_open()
        self._history_tokens += self.chunk_tokens
        excess = self._history_tokens - self.window_tokens
        if excess > 0:
            drop_pages = excess // self.tokens_per_block
            if drop_pages:
                old_head = self._ring[0]
                for _ in range(drop_pages):
                    # Recycle: the oldest page becomes the newest free page.
                    self._ring.append(self._ring.popleft())
                self._refill_shared_page(old_head, self._ring[0])
                self._history_tokens -= drop_pages * self.tokens_per_block
                self._publish_table()
            self._stale_tokens = self._history_tokens - self.window_tokens
        assert 0 <= self._stale_tokens < self.tokens_per_block, self._stale_tokens
        assert self.seq_len <= self.capacity

    def _refill_shared_page(self, old_page: int, new_page: int) -> None:
        """Copy the prompt's tail into the page that now starts the history."""
        tail = self._prompt_len % self.tokens_per_block
        if tail == 0:
            return
        old_page, new_page = old_page * self.page_view_scale, new_page * self.page_view_scale
        for layer in range(self._num_layers):
            buf = self.kv_buffer(layer)
            buf[new_page, :, :, :tail].copy_(buf[old_page, :, :, :tail])

    def _publish_table(self) -> None:
        self._table_version += 1
        table = torch.tensor(self.block_table(), dtype=torch.int32) * self.page_view_scale
        self._page_table.copy_(table.expand(self.MAX_SEGMENTS, -1))

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

        One copy per run of physically consecutive pages, plus one for each
        partial page at either end. In steady state the ring is a cyclic shift of
        consecutive pages, so a chunk costs two or three copies per tensor.
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
        tpb, vs = self.tokens_per_block, self.page_view_scale
        table = self.block_table()
        if k.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")

        t = 0
        head = (-start) % tpb  # tokens that complete the page `start` falls in
        if head:
            m = min(head, n)
            page, slot = table[start // tpb] * vs, start % tpb
            buf[page, 0, :, slot : slot + m].copy_(k[:m].transpose(0, 1))
            buf[page, 1, :, slot : slot + m].copy_(v[:m].transpose(0, 1))
            t = m
        full = (n - t) // tpb
        if full:
            first = (start + t) // tpb
            src_k = k[t : t + full * tpb].view(full, tpb, -1, k.shape[-1]).transpose(1, 2)
            src_v = v[t : t + full * tpb].view(full, tpb, -1, v.shape[-1]).transpose(1, 2)
            for i, page, count in _contiguous_runs(table[first : first + full]):
                dst = slice(page * vs, (page + count) * vs, vs)
                buf[dst, 0].copy_(src_k[i : i + count])
                buf[dst, 1].copy_(src_v[i : i + count])
            t += full * tpb
        if t < n:
            page = table[(start + t) // tpb] * vs
            buf[page, 0, :, : n - t].copy_(k[t:].transpose(0, 1))
            buf[page, 1, :, : n - t].copy_(v[t:].transpose(0, 1))

    def write_prompt_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write the prompt's K/V (``[prompt_len, num_kv_heads, head_dim]``) at logical 0."""
        self._require_open()
        if k.shape[0] != self._prompt_len:
            raise ValueError(
                f"prompt K/V has {k.shape[0]} tokens, cache was opened with {self._prompt_len}"
            )
        self.write_range(layer_idx, 0, k, v)

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
        Every request is a segment of the one sequence, so every row gets the
        same table.
        """
        ids = list(request_ids)
        if (
            not 0 < len(ids) <= self.MAX_SEGMENTS
            or any(r != self.REQUEST_ID for r in ids)
            or num_seqs != len(ids)
            or beam_width != 1
        ):
            raise ValueError(
                f"CausalKVCacheManager serves one sequence (id {self.REQUEST_ID}) in up to "
                f"{self.MAX_SEGMENTS} segments; got request_ids={ids}, num_seqs={num_seqs}, "
                f"beam_width={beam_width}"
            )
        table = torch.tensor(self.block_table(), dtype=torch.int32)
        scale = int(self.index_scales[0])
        kv_offset = int(self.kv_offset[0])
        n = table.numel()
        offsets = torch.stack((table * scale, table * scale + kv_offset))
        dst_tensor[0, :num_seqs, :, :n].copy_(offsets.expand(num_seqs, -1, -1), non_blocking=True)
        if n < dst_tensor.shape[-1]:
            dst_tensor[0, :num_seqs, :, n:].zero_()

    def _require_open(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("cache is not open; call open(prompt_len) first")
