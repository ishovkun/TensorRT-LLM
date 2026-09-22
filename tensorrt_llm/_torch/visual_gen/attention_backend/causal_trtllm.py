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
"""TRT-LLM attention over a :class:`CausalKVCacheManager`.

The diffusion TRTLLM backend in this package is the LLM-side ``TrtllmAttention``
constructed with ``kv_cache_manager=None``. This module is the same backend with
the manager plugged in: one fused call per layer that writes the in-flight
chunk's K/V into the cache and attends over ``[prompt | history | chunk]``.

Two facts about the kernel path decide how this is wired:

* The write is fused. With a fused QKV input and ``update_kv_cache=True`` the
  op writes K/V at ``[past, past + chunk)`` and reads ``[0, past + chunk)`` in the
  same call. Denoising steps overwrite the chunk's slots in place; the clean
  pass writes last. ``past`` moves only on ``commit_chunk``.
* The mask is a dense ``[chunk, seq_len]`` bool tensor. On Blackwell the fused
  trtllm-gen FMHA refuses custom masks, and the backend's FMHA registry then
  selects its Triton custom-mask context kernel, which reads the paged cache
  through the same block table. Correctness does not depend on which kernel
  runs; only the benchmark does.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    CustomAttentionMask,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.visual_gen.kv_cache import CausalKVCacheManager


class CausalTrtllmAttention:
    """One attention layer reading and writing a :class:`CausalKVCacheManager`.

    Inputs are per-token, already normalised and rotated by the model:
    ``q`` is ``[chunk, num_heads, head_dim]``, ``k``/``v`` are
    ``[chunk, num_kv_heads, head_dim]``. Output is ``[chunk, num_heads * head_dim]``.
    """

    def __init__(
        self,
        cache: CausalKVCacheManager,
        *,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.cache = cache
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # ``pos_embd_params=None``: rotary embedding is applied by the model,
        # the backend caches K as given.
        self._attn = TrtllmAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
        )
        self._metadata: Optional[TrtllmAttentionMetadata] = None

    def _prepared_metadata(self, chunk: int, device: torch.device) -> TrtllmAttentionMetadata:
        md = self._metadata
        if md is None or md.max_num_tokens < chunk:
            md = TrtllmAttentionMetadata(
                max_num_requests=1,
                max_num_tokens=chunk,
                max_num_sequences=1,
                kv_cache_manager=self.cache,
                mapping=self.cache.mapping,
                # A context request with cached tokens is chunked prefill to
                # this backend; that is what routes it to the paged context FMHA.
                runtime_features=AttentionRuntimeFeatures(chunked_prefill=True),
            )
            self._metadata = md
        md.seq_lens = torch.tensor([chunk], dtype=torch.int32)
        md.num_contexts = 1
        md.request_ids = [self.cache.REQUEST_ID]
        md.prompt_lens = [chunk]
        md.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[self.cache.past_tokens],
        )
        md.prepare()
        return md

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        chunk = q.shape[0]
        if chunk != self.cache.chunk_tokens:
            raise ValueError(f"expected {self.cache.chunk_tokens} query tokens, got {chunk}")
        if k.shape[0] != chunk or v.shape[0] != chunk:
            raise ValueError("q, k, v must carry the same number of tokens")

        md = self._prepared_metadata(chunk, q.device)
        # The fused op wants one [chunk, (H + 2 H_kv) * D] tensor. This concat is a
        # copy of the chunk's Q/K/V; the existing diffusion TRTLLM backend pays the
        # same one in its ``_concat_qkv``.
        qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)

        mask = self.cache.attention_mask(q.device)
        if mask is None:
            return self._attn.forward(
                qkv, None, None, md, attention_mask=PredefinedAttentionMask.FULL
            )
        return self._attn.forward(
            qkv,
            None,
            None,
            md,
            attention_mask=CustomAttentionMask.CUSTOM,
            attention_mask_data=mask,
        )
