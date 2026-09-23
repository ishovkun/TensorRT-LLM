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

    [ prompt (padded to whole pages) | generator history | in-flight chunk ]

held in a single pool of ``KVCacheManagerV2``. The manager provides the pool
and the pages; this class owns the block table and the sliding window. Nothing
here uses the manager's own sliding-window eviction, block reuse, or request
scheduling, and no ``LlmRequest`` is ever created.

The attention kernel resolves every K/V address as ``pool_base + page * stride``
against one base pointer per layer, and indexes logical token ``t`` at
``table[t // tokens_per_block]``, slot ``t % tokens_per_block``. Two
consequences shape everything below:

* The prompt has to live in the same pool as the history, or one attention call
  could not read both. It is therefore the leading pages of this sequence.
* Eviction must move the logical stream by whole pages, or slot offsets inside
  pages would no longer match. Up to ``tokens_per_block - 1`` tokens older than
  the window therefore stay resident at the front of the history; they, and the
  unused tail of the prompt region, are excluded with an attention mask.

Rotary positions are the caller's business and are absolute over the rollout;
storage positions here restart at zero after each eviction and never grow past
the resident capacity.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

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


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class CausalKVCacheManager(KVCacheManagerV2):
    """``KVCacheManagerV2`` driven as one long-lived sequence with a caller-owned table.

    Args:
        num_layers: attention layers that persist K/V (the generator tower).
        num_kv_heads, head_dim, dtype: K/V geometry, per layer, per rank.
        tokens_per_block: page size, a power of two (the TRT-LLM K/V kernels
            require it). Otherwise a free performance knob; the mask makes
            correctness independent of it.
        prompt_capacity: largest prompt this cache accepts, in tokens
            (``text_cache_max_len``). Rounded up to whole pages.
        window_tokens: generator history kept attendable, in tokens
            (``window_frames * tokens_per_frame``).
        chunk_tokens: tokens written per forward (``chunk_frames * tokens_per_frame``).
        mapping: parallel topology; ``tp_size`` here must already fold in any
            head sharding done by Ulysses, since the manager divides heads by it.
    """

    REQUEST_ID = 0

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
        if min(tokens_per_block, prompt_capacity, window_tokens, chunk_tokens) <= 0:
            raise ValueError(
                "tokens_per_block, prompt_capacity, window_tokens, chunk_tokens must be positive"
            )
        if tokens_per_block & (tokens_per_block - 1) or tokens_per_block < 16:
            # Power of two: the paged K/V kernels assert it. At least 16: the page size is
            # part of the trtllm-gen kernel hash and no kernel exists below 16, in which
            # case the op falls back to an unfused path that ignores the cached prefix
            # without raising.
            raise ValueError(
                f"tokens_per_block must be a power of two >= 16, got {tokens_per_block}"
            )

        self.tokens_per_block = tokens_per_block
        self.prompt_pages = _ceil_div(prompt_capacity, tokens_per_block)
        self.prompt_capacity = self.prompt_pages * tokens_per_block
        self.window_tokens = window_tokens
        self.chunk_tokens = chunk_tokens
        # History may exceed the window by up to a page after a whole-page
        # eviction, and the in-flight chunk sits after it.
        self.history_pages = _ceil_div(window_tokens + chunk_tokens, tokens_per_block) + 1
        self.num_pages = self.prompt_pages + self.history_pages
        capacity = self.num_pages * tokens_per_block

        kv_cache_config = KvCacheConfig(
            max_tokens=capacity,
            max_attention_window=[capacity],
            enable_block_reuse=False,
        )
        super().__init__(
            kv_cache_config,
            CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=capacity,
            max_batch_size=1,
            mapping=mapping if mapping is not None else Mapping(world_size=1, tp_size=1, rank=0),
            dtype=_DTYPES[dtype],
            max_num_tokens=max(chunk_tokens, self.prompt_capacity),
            execution_stream=stream,
        )
        if self.max_seq_len < capacity:
            raise RuntimeError(
                f"KVCacheManagerV2 clamped max_seq_len to {self.max_seq_len} < {capacity}; "
                "the pool could not back the resident window"
            )
        if self.num_pools != 1:
            raise RuntimeError(f"expected one K/V pool, got {self.num_pools}")

        self._kv_cache = None
        self._prompt_len = 0
        self._history_tokens = 0
        self._stale_tokens = 0
        self._prompt_table: List[int] = []
        self._ring: Deque[int] = deque()

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
            if not kv_cache.resize(self.num_pages * self.tokens_per_block):
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

        self._kv_cache = kv_cache
        self._prompt_len = prompt_len
        self._history_tokens = 0
        self._stale_tokens = 0
        self._prompt_table = pages[: self.prompt_pages]
        self._ring = deque(pages[self.prompt_pages :])

    def close(self) -> None:
        if self._kv_cache is None:
            return
        kv_cache, self._kv_cache = self._kv_cache, None
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
    def is_open(self) -> bool:
        return self._kv_cache is not None

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
        return self.prompt_capacity + self._history_tokens

    @property
    def seq_len(self) -> int:
        """Logical length attended by a forward: everything resident plus the chunk."""
        return self.past_tokens + self.chunk_tokens

    def block_table(self) -> List[int]:
        """Base page indices, in logical order, covering ``seq_len`` tokens."""
        self._require_open()
        needed = _ceil_div(self.seq_len, self.tokens_per_block)
        assert needed <= self.num_pages, (needed, self.num_pages)
        history_needed = needed - self.prompt_pages
        return self._prompt_table + [self._ring[i] for i in range(history_needed)]

    def attention_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        """Dense ``[chunk_tokens, seq_len]`` bool mask, or ``None`` when nothing needs masking.

        Excludes the unused tail of the prompt region and the stale head of the
        history. Full attention otherwise; the batched clean pass adds its own
        frame-causal structure on top of this.
        """
        self._require_open()
        prompt_pad = self.prompt_capacity - self._prompt_len
        if prompt_pad == 0 and self._stale_tokens == 0:
            return None
        allowed = torch.ones(self.seq_len, dtype=torch.bool, device=device)
        allowed[self._prompt_len : self.prompt_capacity] = False
        allowed[self.prompt_capacity : self.prompt_capacity + self._stale_tokens] = False
        return allowed.unsqueeze(0).expand(self.chunk_tokens, -1).contiguous()

    def commit_chunk(self) -> None:
        """The in-flight chunk's K/V are final; advance the window."""
        self._require_open()
        self._history_tokens += self.chunk_tokens
        excess = self._history_tokens - self.window_tokens
        if excess > 0:
            drop_pages = excess // self.tokens_per_block
            for _ in range(drop_pages):
                # Recycle: the oldest page becomes the newest free page.
                self._ring.append(self._ring.popleft())
            self._history_tokens -= drop_pages * self.tokens_per_block
            self._stale_tokens = self._history_tokens - self.window_tokens
        assert 0 <= self._stale_tokens < self.tokens_per_block, self._stale_tokens
        assert self.seq_len <= self.num_pages * self.tokens_per_block

    # ------------------------------------------------------------------ direct pool access

    def kv_buffer(self, layer_idx: int) -> torch.Tensor:
        """The layer's pool as ``[num_pages, 2, num_kv_heads, tokens_per_block, head_dim]``."""
        buf = self.get_buffers(layer_idx, kv_layout="HND")
        if buf is None:
            raise RuntimeError(f"layer {layer_idx} has no K/V buffer")
        return buf

    def _slots(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        table = torch.tensor(self.block_table(), dtype=torch.long, device=positions.device)
        return table[positions // self.tokens_per_block], positions % self.tokens_per_block

    def write_kv(
        self, layer_idx: int, positions: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write ``k``/``v`` of shape ``[T, num_kv_heads, head_dim]`` at logical ``positions``.

        Used for the prompt, which the reasoner produces outside any attention
        call. The in-flight chunk is written by the attention kernel itself.
        """
        buf = self.kv_buffer(layer_idx)
        page, slot = self._slots(positions)
        # Advanced indices on dims 0 and 3 broadcast to [T] and move to the front,
        # so the target is [T, num_kv_heads, head_dim] — the same layout as k/v.
        buf[page, 0, :, slot, :] = k.to(buf.dtype)
        buf[page, 1, :, slot, :] = v.to(buf.dtype)

    def read_kv(self, layer_idx: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read back ``[T, num_kv_heads, head_dim]`` K and V at logical ``positions``."""
        buf = self.kv_buffer(layer_idx)
        page, slot = self._slots(positions)
        return buf[page, 0, :, slot, :], buf[page, 1, :, slot, :]

    def write_prompt_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write the prompt's K/V (``[prompt_len, num_kv_heads, head_dim]``) and zero its padding."""
        self._require_open()
        if k.shape[0] != self._prompt_len:
            raise ValueError(
                f"prompt K/V has {k.shape[0]} tokens, cache was opened with {self._prompt_len}"
            )
        buf = self.kv_buffer(layer_idx)
        positions = torch.arange(self.prompt_capacity, device=buf.device)
        page, slot = self._slots(positions)
        buf[page, :, :, slot, :] = 0
        if self._prompt_len:
            self.write_kv(layer_idx, positions[: self._prompt_len], k, v)

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
        (+ ``kv_offset`` for V) — the same encoding the manager's device copy uses.
        """
        if list(request_ids) != [self.REQUEST_ID] or num_seqs != 1 or beam_width != 1:
            raise ValueError(
                f"CausalKVCacheManager serves exactly one sequence (id {self.REQUEST_ID}); "
                f"got request_ids={list(request_ids)}, num_seqs={num_seqs}, beam_width={beam_width}"
            )
        table = torch.tensor(self.block_table(), dtype=torch.int32)
        scale = int(self.index_scales[0])
        kv_offset = int(self.kv_offset[0])
        n = table.numel()
        offsets = torch.stack((table * scale, table * scale + kv_offset))
        dst_tensor[0, 0, :, :n].copy_(offsets, non_blocking=True)
        if n < dst_tensor.shape[-1]:
            dst_tensor[0, 0, :, n:].zero_()

    def _require_open(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("cache is not open; call open(prompt_len) first")
