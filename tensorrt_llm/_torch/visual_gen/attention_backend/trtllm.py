# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Diffusion TRTLLM Attention Backend

Wraps TrtllmAttention with simplified metadata for visual generation (diffusion) models.
Handles the specifics of no-KV-cache operation and fused QKV requirements.
"""

from typing import Optional, Union

import torch

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention.backends.interface import AttentionRuntimeFeatures, PredefinedAttentionMask
from ...attention.backends.sparse.skip_softmax import SkipSoftmaxParams
from ...attention.backends.trtllm import TrtllmAttention as BaseTrtllmAttention
from ...attention.backends.trtllm import TrtllmAttentionMetadata as BaseTrtllmAttentionMetadata
from ...metadata import KVCacheParams
from ..cache import CausalKVCacheManager
from .interface import AttentionBackend, AttentionTensorLayout

# The only page size with shipped trtllm-gen paged context kernels: every paged
# context cubin under kernels/trtllmGenKernels/fmha/cubin is a ``P32`` variant, and
# log2(tokens per page) is part of the kernel hash, so any other value misses the
# lookup and the attention op falls back to an unfused path that silently ignores
# the cached prefix. Nothing in the tree exposes this number; it lives in the cubin
# inventory only.
TRTLLM_GEN_TOKENS_PER_PAGE = 32


class TrtllmAttentionMetadata:
    """
    Simplified metadata adapter for diffusion models using TRTLLM backend.

    Lazy initialization with auto-growing capacity:
    - Metadata created only when capacity needs increase
    - prepare() called only when seq_lens actually change
    - Automatically reallocates when batch_size or seq_len exceeds current capacity

    Args:
        device: Target device for tensors.
        attention_metadata_state: Mutable model-scoped state shared by all
            attention layers in one model instance.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        attention_metadata_state: Optional[dict] = None,
    ):
        self.device = device or torch.device("cuda")
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM attention requires `attention_metadata_state` to be provided "
                "by visual-gen config for model-scoped metadata sharing."
            )
        self._metadata_state = attention_metadata_state

        # Lazily created BaseTrtllmAttentionMetadata objects. Diffusion blocks
        # can launch video and audio attention back-to-back with different
        # sequence lengths, so keep separate metadata buffers per shape instead
        # of mutating one shared object while kernels may still be in flight.
        self._metadata_cache = self._metadata_state.setdefault("metadata_cache", {})
        self._metadata: Optional[BaseTrtllmAttentionMetadata] = None

        # Track prepared state
        self._cached_seq_lens: Optional[torch.Tensor] = None
        self._prepared = False

    def _needs_prepare(self, batch_size: int, seq_lens: torch.Tensor) -> bool:
        """Check if we need to call prepare() (current request seq_lens or shared metadata object seq_lens changed).

        Assumes uniform sequence length per batch; if per-sample lengths vary,
        we may need to check seq_lens tensor instead.

        In addition, multiple visual gen attention modules share one metadata object.  A
        different module may have prepared it for another sequence length even
        when this wrapper's local cached seq_lens are unchanged.
        """
        if not self._prepared:
            return True
        if self._cached_seq_lens is None:
            return True
        if self._cached_seq_lens.shape[0] != batch_size:
            return True
        if not torch.equal(self._cached_seq_lens[:batch_size], seq_lens):
            return True

        metadata = self._metadata
        if metadata is None:
            return True
        if getattr(metadata, "num_contexts", None) != batch_size:
            return True

        max_seq_len = seq_lens.max().item()
        if getattr(metadata, "max_seq_len", None) != max_seq_len:
            return True

        metadata_seq_lens = getattr(metadata, "seq_lens", None)
        if metadata_seq_lens is None or metadata_seq_lens.shape[0] < batch_size:
            return True
        if not torch.equal(metadata_seq_lens[:batch_size].to(seq_lens.device), seq_lens):
            return True

        return False

    def _create_metadata(self, batch_size: int, max_seq_len: int) -> None:
        """Create new metadata with given capacity."""
        self._metadata = BaseTrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=batch_size * max_seq_len,
            max_num_sequences=batch_size,
            kv_cache_manager=None,  # No KV cache for diffusion
            mapping=Mapping(),
            runtime_features=AttentionRuntimeFeatures(),
        )
        self._prepared = False  # Reset prepare state on new metadata

    def _select_cached_metadata(self, cached) -> None:
        self._metadata = cached["metadata"]
        self._prepared = cached["prepared"]
        self._cached_seq_lens = cached["seq_lens"]

    def prepare(
        self,
        batch_size: int,
        seq_lens: Union[int, torch.Tensor],
    ) -> BaseTrtllmAttentionMetadata:
        """
        Prepare metadata for a forward pass.

        Lazy behavior:
        - Creates metadata only when capacity needs increase
        - Calls prepare() only when (batch_size, max_seq_len) actually change
        """
        if isinstance(seq_lens, int):
            seq_lens_tensor = torch.full((batch_size,), seq_lens, dtype=torch.int32)
        else:
            seq_lens_tensor = seq_lens.to(dtype=torch.int32)
        max_seq_len = seq_lens_tensor.max().item()
        # Keep CUDA graph-captured metadata buffers stable per batch/seq-lens shape.
        cache_key = (batch_size, tuple(int(x) for x in seq_lens_tensor.tolist()))

        cached = self._metadata_cache.get(cache_key)
        if cached is None:
            self._create_metadata(batch_size, max_seq_len)
            cached = {
                "metadata": self._metadata,
                "prepared": False,
                "seq_lens": None,
            }
            self._metadata_cache[cache_key] = cached

        self._select_cached_metadata(cached)

        if self._needs_prepare(batch_size, seq_lens_tensor):
            cached_seq_lens = seq_lens_tensor.clone()
            self._metadata.seq_lens = cached_seq_lens
            self._metadata.num_contexts = batch_size
            self._metadata.max_seq_len = max_seq_len
            self._metadata.request_ids = list(range(batch_size))
            self._metadata.prepare()

            # Cache per-shape state without sharing the tensor across entries.
            cached["prepared"] = True
            cached["seq_lens"] = cached_seq_lens

            self._select_cached_metadata(cached)

        return self._metadata

    def prepare_with_kv_cache(
        self, kv_cache: CausalKVCacheManager, num_causal_blocks: int, causal_block_size: int
    ) -> BaseTrtllmAttentionMetadata:
        """Metadata over ``kv_cache``: ``num_causal_blocks`` context requests of
        ``causal_block_size`` tokens, request ``i`` with ``past + i*causal_block_size``
        tokens already cached. All requests are the one sequence and share its table.

        One object per (cache, blocking) for the life of the cache, shared by every
        layer through the model-scoped state. Its device buffers are allocated once
        and ``prepare()`` re-fills them in place whenever the cache's table or
        ``past`` moved, which is what keeps a CUDA graph captured around the
        forward valid after ``commit()``. Neither creation nor re-preparation may
        happen during capture: warm up eagerly first, and call this before replay.
        """
        cache_key = ("kv_cache", id(kv_cache), num_causal_blocks, causal_block_size)
        state = (kv_cache.table_version, kv_cache.past_tokens)
        cached = self._metadata_cache.get(cache_key)
        if cached is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "K/V cache attention metadata first needed during CUDA graph capture; "
                    "run the forward eagerly once before capturing"
                )
            self._drop_metadata_of_shut_down_caches()
            metadata = BaseTrtllmAttentionMetadata(
                max_num_requests=kv_cache.max_causal_blocks,
                max_num_tokens=kv_cache.chunk_tokens,
                max_num_sequences=kv_cache.max_causal_blocks,
                kv_cache_manager=kv_cache,
                mapping=kv_cache.mapping,
                runtime_features=AttentionRuntimeFeatures(chunked_prefill=True),
            )
            metadata.seq_lens = torch.full(
                (num_causal_blocks,), causal_block_size, dtype=torch.int32
            )
            metadata.num_contexts = num_causal_blocks
            metadata.request_ids = kv_cache.request_ids(num_causal_blocks)
            metadata.prompt_lens = [causal_block_size] * num_causal_blocks
            cached = {
                "metadata": metadata,
                "prepared": False,
                "seq_lens": metadata.seq_lens,
                "kv_state": None,
            }
            self._metadata_cache[cache_key] = cached
        metadata = cached["metadata"]
        if cached["kv_state"] != state:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "K/V cache moved since the metadata was prepared; prepare before capture "
                    "or replay, not inside it"
                )
            # Each block's row holds the fixed region, its window, the earlier blocks
            # and then its own tokens; the kernel writes the block right after the
            # cached count, inside the block's private pages.
            metadata.kv_cache_params = KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=kv_cache.cached_tokens(causal_block_size)[
                    :num_causal_blocks
                ],
            )
            kv_cache.set_causal_block_size(causal_block_size)
            metadata.prepare()
            cached["prepared"] = True
            cached["kv_state"] = state
        return metadata

    def _drop_metadata_of_shut_down_caches(self) -> None:
        """Forget metadata built over caches that were shut down. Each entry holds its
        cache, so without this a model that builds a new cache per rollout would keep
        every old one alive. A merely closed cache keeps its metadata: it may be
        reopened, and graphs captured over it still read those buffers."""
        stale = [
            key
            for key, entry in self._metadata_cache.items()
            if key[0] == "kv_cache" and entry["metadata"].kv_cache_manager.is_shut_down
        ]
        for key in stale:
            del self._metadata_cache[key]


class TrtllmAttention(BaseTrtllmAttention, AttentionBackend):
    """
    TRTLLM Attention wrapper for diffusion models.

    Handles:
    - Fused QKV requirement for TRTLLM kernel (used when no quant_attention_config is provided)
    - Metadata creation and preparation
    - No KV cache operation
    - SageAttention per-block QKV quantization (when a quant_attention_config is provided. requires unfused QKV)
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        dtype: Optional[torch.dtype] = None,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        attention_metadata_state: Optional[dict] = None,
        sparse_params: Optional[SkipSoftmaxParams] = None,
    ):
        num_kv_heads = num_kv_heads or num_heads

        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
            sparse_params=sparse_params,
            dtype=dtype,
        )

        # TRTLLM expects flat [B*S, H*D] format
        self._preferred_layout = AttentionTensorLayout.NHD

        self.metadata = TrtllmAttentionMetadata(
            attention_metadata_state=attention_metadata_state,
        )

        self.quant_attention_config = quant_attention_config

    # Needed to work with torch compile cause of attention metadata
    # make attn metadata as input for it to work
    @torch.compiler.disable
    def _prepare_metadata(self, batch_size: int, seq_len: int):
        return self.metadata.prepare(batch_size, seq_len)

    @torch.compiler.disable
    def _prepare_kv_cache_metadata(
        self, kv_cache: CausalKVCacheManager, num_causal_blocks: int, causal_block_size: int
    ):
        return self.metadata.prepare_with_kv_cache(kv_cache, num_causal_blocks, causal_block_size)

    @torch.compile
    def _concat_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int,
    ):
        # Separate Q, K, V provided - fuse them
        q = q.view(batch_size * seq_len, -1)
        k = k.view(batch_size * kv_seq_len, -1)
        v = v.view(batch_size * kv_seq_len, -1)
        qkv = torch.cat([q, k, v], dim=-1)
        return qkv

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        kv_cache: Optional[CausalKVCacheManager] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with automatic metadata handling.

        Dimensions are derived from tensor shapes (NHD layout: ``[B, S, H, D]``).

        For diffusion models, expects:
        - Fused QKV: q contains [Q, K, V] concatenated, k and v are None
            - does not support SageAttention
        - OR separate Q, K, V which:
            - for regular TRTLLM attention, will be fused internally
            - for SageAttention, will be used directly

        Args:
            q: Query tensor [B, S, H, D] or fused QKV [B, S, H_qkv, D]
            k: Key tensor [B, S_kv, H_kv, D] or None if fused
            v: Value tensor [B, S_kv, H_kv, D] or None if fused
            batch_size: Batch size
            seq_len: Number of real query tokens. Without ``kv_cache`` it equals
                ``S``. With ``kv_cache`` it may be smaller: rows of ``q``/``k``/``v``
                past ``seq_len`` are padding added so the sequence splits evenly
                across ranks; they are neither written to the cache nor attended,
                and their output rows are zero.
            attention_mask: Attention mask type
            seq_len_kv: Sequence length for K/V (for cross-attention, defaults to seq_len)
            kv_cache: A ``CausalKVCacheManager``. When given, ``k``/``v`` are the new
                tokens only: the fused kernel writes them at ``past_tokens`` and
                attends over everything cached before them plus themselves.
                ``batch_size`` must be 1.
            timestep: Keyword; normalized diffusion timestep forwarded to
                timestep-varying sparse attention (no effect otherwise).
            causal_block_size: Keyword understood by the ``kv_cache`` path only. Cuts
                the new tokens into consecutive causal blocks: full attention within a
                block, causal across blocks. Use it when the cache must hold each
                block's K/V as if the blocks had been generated one at a time, so later
                blocks never leak into earlier ones. Absent: one causal block.

        Returns:
            Output tensor [B, S, H*D]
        """
        if kv_cache is not None:
            output = self._forward_with_kv_cache(
                q,
                k,
                v,
                batch_size,
                seq_len,
                kv_cache,
                kwargs.pop("causal_block_size", None),
                attention_mask,
            )
            return output.view(1, q.shape[1], -1)
        output = self._forward_without_kv_cache(
            q, k, v, batch_size, seq_len, attention_mask, seq_len_kv, **kwargs
        )
        return output.view(batch_size, seq_len, -1)

    def _forward_without_kv_cache(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask,
        seq_len_kv: Optional[int],
        **kwargs,
    ) -> torch.Tensor:
        """Plain diffusion attention; returns ``[B*S, H*D]``."""
        if kwargs.pop("causal_block_size", None) is not None:
            raise NotImplementedError(
                "causal_block_size is only implemented over a K/V cache; pass kv_cache."
            )
        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len
        prepared_metadata = self._prepare_metadata(batch_size, seq_len)
        timestep = kwargs.pop("timestep", None)

        if self.quant_attention_config is not None:
            assert k is not None and v is not None, (
                "SageAttention requires separate Q, K, V tensors"
            )
            quant_cfg = self.quant_attention_config
            q = q.reshape(batch_size * seq_len, -1).contiguous()
            k = k.reshape(batch_size * kv_seq_len, -1).contiguous()
            v = v.reshape(batch_size * kv_seq_len, -1).contiguous()
            output = super().forward(
                q=q,
                k=k,
                v=v,
                metadata=prepared_metadata,
                attention_mask=attention_mask,
                timestep=timestep,
                sage_attn_num_elts_per_blk_q=quant_cfg.q_block_size,
                sage_attn_num_elts_per_blk_k=quant_cfg.k_block_size,
                sage_attn_num_elts_per_blk_v=quant_cfg.v_block_size,
                sage_attn_qk_int8=(quant_cfg.qk_dtype == "int8"),
            )
        else:
            if k is None and v is None:
                qkv = q.reshape(batch_size * seq_len, -1)
            else:
                qkv = self._concat_qkv(q, k, v, batch_size, seq_len, kv_seq_len)
            output = super().forward(
                q=qkv,
                k=None,
                v=None,
                metadata=prepared_metadata,
                attention_mask=attention_mask,
                timestep=timestep,
            )
        return output

    def _forward_with_kv_cache(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        kv_cache: CausalKVCacheManager,
        causal_block_size: Optional[int],
        attention_mask: PredefinedAttentionMask,
    ) -> torch.Tensor:
        """Attention over ``kv_cache`` with the ``seq_len`` real new tokens; returns
        ``[S, H*D]`` with zero rows past ``seq_len``."""
        if attention_mask != PredefinedAttentionMask.FULL:
            raise NotImplementedError("K/V cache attention is full attention over the cache.")
        if self.quant_attention_config is not None:
            raise NotImplementedError("K/V cache attention does not combine with SageAttention.")
        if self.sparse_params is not None:
            raise NotImplementedError("K/V cache attention does not combine with sparse attention.")
        if k is None or v is None:
            raise ValueError("K/V cache attention needs separate q, k, v.")
        if kv_cache.tokens_per_page != TRTLLM_GEN_TOKENS_PER_PAGE:
            raise NotImplementedError(
                f"trtllm-gen ships paged context kernels for {TRTLLM_GEN_TOKENS_PER_PAGE}-token "
                f"pages only; the cache uses {kv_cache.tokens_per_page}. Build the cache with "
                f"tokens_per_page={TRTLLM_GEN_TOKENS_PER_PAGE} or use the CUDNN backend."
            )
        batch, num_rows, _, _ = q.shape
        if batch != 1 or batch_size != 1 or k.shape[1] != num_rows:
            raise ValueError(
                "K/V cache attention takes one video: q, k, v of [1, S, heads, head_dim]."
            )
        if k.shape[2] != kv_cache.num_kv_heads:
            raise ValueError(
                f"k has {k.shape[2]} heads but the cache holds {kv_cache.num_kv_heads} per rank."
            )
        if q.shape[2:] != (self.num_heads, self.head_dim) or k.shape[3] != self.head_dim:
            raise ValueError(
                f"q is {tuple(q.shape)}, k is {tuple(k.shape)}; this backend was built for "
                f"{self.num_heads} heads of {self.head_dim}"
            )
        if not 0 < seq_len <= num_rows:
            raise ValueError(
                f"seq_len {seq_len} outside (0, {num_rows}]: it counts the real tokens; "
                "rows past it are padding."
            )
        num_tokens = seq_len
        if num_tokens < num_rows:
            q, k, v = q[:, :num_tokens], k[:, :num_tokens], v[:, :num_tokens]
        causal_block_size = causal_block_size or num_tokens
        if num_tokens % causal_block_size:
            raise ValueError(
                f"{num_tokens} tokens do not split into causal blocks of {causal_block_size}."
            )
        num_causal_blocks = num_tokens // causal_block_size
        # The fused kernel writes each block's own tokens into the block's private
        # pages; the shared pages, which later blocks and later chunks read, get
        # the chunk here.
        kv_cache.write_chunk(self.layer_idx, k[0], v[0], causal_block_size, own_tokens=False)
        metadata = self._prepare_kv_cache_metadata(kv_cache, num_causal_blocks, causal_block_size)
        qkv = self._concat_qkv(q, k, v, 1, num_tokens, num_tokens)
        output = super().forward(
            q=qkv, k=None, v=None, metadata=metadata, attention_mask=PredefinedAttentionMask.FULL
        )
        if num_tokens == num_rows:
            return output
        full = output.new_empty(num_rows, output.shape[-1])
        full[:num_tokens].copy_(output.view(num_tokens, -1))
        full[num_tokens:].zero_()
        return full

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    def support_fused_qkv(self) -> bool:
        """Standard path fuses QKV; SageAttention path does not."""
        return self.quant_attention_config is None

    def support_kv_cache(self) -> bool:
        return True
