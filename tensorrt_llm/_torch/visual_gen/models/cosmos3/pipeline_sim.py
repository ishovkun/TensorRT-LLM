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
"""Cosmos3 Nano Sim: the chunked, causal rollout pipeline.

The checkpoint's ``_class_name`` is ``Cosmos3NanoSimBimanualPipeline``. Registering
it under that exact name is what keeps ``AutoPipeline`` from routing the checkpoint
to the bidirectional ``Cosmos3OmniMoTPipeline`` through its ``"Cosmos3" in
class_name`` fallback, where it would load and produce wrong video.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div

from ...cache import BlockCausalKVCacheManager, RolloutSpec
from ...cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from ...pipeline_registry import register_pipeline
from ...utils import postprocess_video_tensor
from ..wan.vae_loader import load_wan_vae
from .action import normalize_action_mode
from .config_resolver import Cosmos3Config
from .defaults import VIDEO_RES_SIZE_INFO
from .pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from .sim_packing import SIM_CONFIG_KEY

# Tokens per K/V cache page. The cache takes any power of two; the attention kernels
# decide: trtllm-gen ships its paged context kernels for 32-token pages, cuDNN's paged
# graph takes any size. The pool is built only after the backend confirms the size.
KV_TOKENS_PER_PAGE = 32


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def split_row_domain_ids(domain_id: Any) -> tuple[Any, Optional[tuple[int, ...]]]:
    """A request's ``domain_id`` is one id or one id per action row. Returns the id
    the generic resolution should see (the first row's) and the per-row tuple, or
    ``(domain_id, None)`` for a single id."""
    if not isinstance(domain_id, (list, tuple)):
        return domain_id, None
    if not domain_id:
        raise ValueError("Cosmos3 Sim domain_id list must not be empty.")
    row_ids = tuple(int(d) for d in domain_id)
    return row_ids[0], row_ids


@dataclass(frozen=True)
class SimEmbodiment:
    """One embodiment the checkpoint was trained on: its domain id and the width
    of the action rows it expects (``None`` where the block does not say)."""

    domain_id: Optional[int]
    action_dim: Optional[int]


@dataclass(frozen=True)
class Cosmos3SimSettings:
    """The rollout geometry of a Sim checkpoint, taken from the resolved
    :class:`Cosmos3Config`. Frames are latent frames throughout."""

    sigmas: tuple[float, ...]
    """The fixed denoising schedule, ``fixed_step_sampler_config.t_list``."""
    chunk_frames: int
    """Latent frames generated together after the first chunk."""
    chunk_partition: tuple[int, ...]
    """Leading chunk sizes from the start; the last repeats."""
    history_frames: Optional[int]
    """Recent frames before a chunk that it attends to, sinks excluded: the window.
    ``None``: full history; the pipeline sizes the cache from the memory left after
    warmup."""
    sink_frames: int
    """Leading frames pinned for the whole rollout."""
    text_cache_max_len: int
    """Most prompt tokens the fixed region may hold; longer prompts are rejected."""
    action_tokens_per_frame: int
    """Action rows packed before each frame's vision tokens."""
    video_temporal_causal: bool
    embodiments: dict[str, SimEmbodiment] = field(default_factory=dict)
    """Embodiments declared under ``conditioning.embodiments``, by lower-case name."""
    default_embodiment: Optional[str] = None
    """``conditioning.default_embodiment``: used when a request names no domain."""

    @staticmethod
    def largest_frame() -> tuple[int, int]:
        """``(height, width)`` of the biggest 480p bucket shape: the default warmup
        shape when the user config lists no warmup resolutions."""
        shapes = [(h, w) for (w, h) in VIDEO_RES_SIZE_INFO["480"].values()]
        return max(shapes, key=lambda hw: hw[0] * hw[1])

    @classmethod
    def from_config(cls, config: Cosmos3Config) -> "Cosmos3SimSettings":
        if not config.is_autoregressive:
            raise ValueError(
                f"Not a Cosmos3 Sim checkpoint: transformer/config.json has no "
                f"'{SIM_CONFIG_KEY}' block."
            )
        action, chunking, kv = config.action, config.chunking, config.kv_cache
        # Schema 3 names the width per embodiment (`raw_action_dim`); schema 5 has
        # `input_action_dim` per embodiment and one `input_contract.action_dim`.
        embodiments = {}
        for name, spec in action.embodiments.items():
            width = _get(spec, "input_action_dim")
            if width is None:
                width = _get(spec, "raw_action_dim", action.input_action_dim)
            domain_id = _get(spec, "domain_id")
            embodiments[str(name).lower()] = SimEmbodiment(
                domain_id=None if domain_id is None else int(domain_id),
                action_dim=None if width is None else int(width),
            )
        return cls(
            sigmas=config.sampling.fixed_sigmas,
            chunk_frames=chunking.chunk_frames,
            chunk_partition=chunking.partition,
            history_frames=kv.history_frames,
            sink_frames=kv.sink_frames,
            text_cache_max_len=config.prompt.text_cache_max_len,
            action_tokens_per_frame=action.tokens_per_frame,
            video_temporal_causal=chunking.video_temporal_causal,
            embodiments=embodiments,
            default_embodiment=action.default_embodiment,
        )

    def resolve_embodiment(
        self,
        domain_name: Optional[str],
        domain_id: Optional[int],
        raw_action_dim: Optional[int],
    ) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """Complete a request's ``(domain_name, domain_id, raw_action_dim)`` from the
        checkpoint's embodiment table: a missing domain falls back to the declared
        default, a missing id or width is filled in, and a given one that disagrees
        with the checkpoint is an error. Embodiments the checkpoint does not list
        pass through untouched."""
        if (domain_name is None or not str(domain_name).strip()) and domain_id is None:
            domain_name = self.default_embodiment
        if domain_name is not None and str(domain_name).strip():
            name = str(domain_name).strip().lower()
        else:
            name = next(
                (n for n, e in self.embodiments.items() if e.domain_id == int(domain_id)), None
            )
        spec = self.embodiments.get(name) if name is not None else None
        if spec is None:
            return domain_name, domain_id, raw_action_dim
        if spec.domain_id is not None:
            if domain_id is not None and int(domain_id) != spec.domain_id:
                raise ValueError(
                    f"Cosmos3 Sim domain_id={domain_id} contradicts the checkpoint, which "
                    f"maps {name!r} to domain_id={spec.domain_id}."
                )
            domain_id = spec.domain_id
        if spec.action_dim is not None:
            if raw_action_dim is not None and int(raw_action_dim) != spec.action_dim:
                raise ValueError(
                    f"Cosmos3 Sim raw_action_dim={raw_action_dim} contradicts the checkpoint, "
                    f"which expects {spec.action_dim} values per action row for {name!r}."
                )
            raw_action_dim = spec.action_dim
        return name, domain_id, raw_action_dim

    def chunk_ranges(self, num_latent_frames: int) -> list[tuple[int, int]]:
        """Latent frame ranges ``[f0, f1)`` of the rollout's chunks, in order: the
        partition's chunks first, then ``chunk_frames`` at a time, the last possibly
        shorter."""
        ranges: list[tuple[int, int]] = []
        f0 = 0
        sizes: Sequence[int] = self.chunk_partition
        i = 0
        while f0 < num_latent_frames:
            size = sizes[i] if i < len(sizes) else self.chunk_frames
            f1 = min(f0 + size, num_latent_frames)
            ranges.append((f0, f1))
            f0, i = f1, i + 1
        return ranges


class _OverlappedDecode:
    """Pixel frames of a rollout, decoded chunk by chunk on a side stream as the
    chunks finish. Each submitted latent slice is read on the side stream after the
    main stream wrote it. Its uint8 frames are copied straight into one pinned host
    buffer, so GPU memory does not grow with the video's length; ``finish`` returns
    that buffer, ``[1, T, H, W, 3]``, the video the pipeline would otherwise decode
    after the rollout.

    The side stream may live on another GPU holding its own copy of the VAE: each
    chunk's latents (a few hundred KB) hop over as they finish, so the decode leaves
    the generator's GPU alone."""

    def __init__(self, pipe, vae, stream: torch.cuda.Stream, num_pixel_frames: int) -> None:
        self.pipe = pipe
        self.vae = vae
        self.stream = stream
        self.device = stream.device
        self.num_pixel_frames = num_pixel_frames
        self.host: Optional[torch.Tensor] = None
        self.written = 0
        self.held: list[torch.Tensor] = []
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            vae.decode_stream_start()

    def submit(self, chunk: torch.Tensor) -> None:
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            ready.wait()
            self.held.append(chunk)  # the main stream never writes these frames again
            if chunk.device != self.device:
                chunk = chunk.to(self.device, non_blocking=True)
            z = self.pipe._scale_latents_for_decode(chunk)
            frames = postprocess_video_tensor(self.vae.decode_stream_step(z))
            if self.host is None:
                _, _, height, width, channels = frames.shape
                self.host = torch.empty(
                    (1, self.num_pixel_frames, height, width, channels),
                    dtype=frames.dtype,
                    pin_memory=prefer_pinned(),
                )
            t = frames.shape[1]
            # The copy is queued on the side stream; the frames' memory goes back to
            # that stream's pool and is reused only by work ordered after the copy.
            self.host[:, self.written : self.written + t].copy_(frames, non_blocking=True)
            self.written += t

    def finish(self) -> torch.Tensor:
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.vae.decode_stream_end()
        self.stream.synchronize()  # the host buffer is read on the CPU from here on
        self.held.clear()
        if self.written != self.num_pixel_frames:
            raise RuntimeError(
                f"Cosmos3 Sim streamed decode wrote {self.written} of "
                f"{self.num_pixel_frames} pixel frames."
            )
        video, self.host = self.host, None
        return video


@register_pipeline(
    "Cosmos3NanoSimBimanualPipeline",
    hf_ids=["nvidia/Cosmos3-Nano-Sim-Bimanual"],
    doc="Cosmos3 Nano Sim: chunked causal world model with a rolling K/V cache.",
)
class Cosmos3NanoSimBimanualPipeline(Cosmos3OmniMoTPipeline):
    """``Cosmos3OmniMoTPipeline`` with the chunked causal rollout in place of the
    whole-clip denoising loop. Loading, tokenization, action preparation, VAE and
    guardrails are inherited unchanged."""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.sim: Optional[Cosmos3SimSettings] = None  # set by load_standard_components
        # The page pool's size, settled once by warmup from the memory left after the
        # peak; what it holds in frames depends on each request's frame geometry.
        self._pool_tokens: Optional[int] = None
        self._kv_cache: Optional[BlockCausalKVCacheManager] = None
        # Finished chunks are decoded on this stream while later chunks denoise; on
        # another GPU when one is named, with its own VAE instance.
        self._decode_stream: Optional[torch.cuda.Stream] = None
        self._streamed_video: Optional[torch.Tensor] = None
        decode_device = os.environ.get("TRTLLM_COSMOS3_SIM_DECODE_DEVICE")
        self._decode_device = torch.device(decode_device) if decode_device else None
        self._decode_vae = None

    def load_standard_components(self, checkpoint_dir, device, skip_components=None):
        super().load_standard_components(checkpoint_dir, device, skip_components)
        # The full checkpoint (every file, not only the transformer config) decides.
        self.sim = Cosmos3SimSettings.from_config(self.cosmos3_config)
        if self.sim.history_frames is None:
            logger.info("Cosmos3 Sim: full history, the K/V cache is sized from free memory.")
        else:
            logger.info(
                f"Cosmos3 Sim: each chunk attends to {self.sim.sink_frames} sink frames and "
                f"the {self.sim.history_frames} latent frames before it."
            )
        if self._decode_device is not None and getattr(self, "vae", None) is not None:
            logger.info(f"Cosmos3 Sim: decoding on {self._decode_device}")
            self._decode_vae = load_wan_vae(
                checkpoint_dir, self._decode_device, dtype=torch.bfloat16
            )

    def _resolve_request(
        self,
        *,
        action_mode: Optional[str],
        domain_name: Optional[str],
        domain_id: Optional[int],
        raw_action_dim: Optional[int],
        **kwargs: Any,
    ):
        """The checkpoint, not the generic embodiment table, says which domain id and
        action width an embodiment has on this model."""
        domain_id, row_ids = split_row_domain_ids(domain_id)
        if normalize_action_mode(action_mode) is not None:
            domain_name, domain_id, raw_action_dim = self.sim.resolve_embodiment(
                domain_name, domain_id, raw_action_dim
            )
            if row_ids is not None:
                for d in set(row_ids):
                    self.sim.resolve_embodiment(None, d, raw_action_dim)
        resolved = super()._resolve_request(
            action_mode=action_mode,
            domain_name=domain_name,
            domain_id=domain_id,
            raw_action_dim=raw_action_dim,
            **kwargs,
        )
        if row_ids is not None:
            if (
                resolved.action_chunk_size is not None
                and len(row_ids) != resolved.action_chunk_size
            ):
                raise ValueError(
                    f"Cosmos3 Sim got {len(row_ids)} domain ids for {resolved.action_chunk_size} action rows."
                )
            resolved = replace(resolved, action_row_domain_ids=row_ids)
        self.check_envelope(resolved.height, resolved.width, resolved.num_frames)
        return resolved

    def _prepare_request_latents(self, request, **kwargs):
        prepared = super()._prepare_request_latents(request, **kwargs)
        prepared.action_row_domain_ids = request.action_row_domain_ids
        return prepared

    def check_envelope(self, height: int, width: int, num_frames: int) -> None:
        """Refuse, before any compute, a rollout the K/V cache cannot hold at this frame
        geometry: a configured window that does not fit the pool, or a full-history
        rollout longer than the pool affords. Nothing to check before the pool is sized."""
        if self._pool_tokens is None:
            return
        history = self._history_frames_for(self._tokens_per_frame(height, width))
        self._check_history_fits(self._latent_frames(num_frames), history)

    def _latent_frames(self, num_pixel_frames: int) -> int:
        return (num_pixel_frames - 1) // self.vae_scale_factor_temporal + 1

    def _check_history_fits(self, num_latent_frames: int, history_frames: int) -> None:
        """Full history (no window declared): the last frame reads every frame before
        it, so the pool must hold ``num_latent_frames - 1`` frames. Longer rollouts are
        refused rather than run over a window nobody asked for."""
        if self.sim.history_frames is not None or num_latent_frames - 1 <= history_frames:
            return
        longest = history_frames * self.vae_scale_factor_temporal + 1
        requested = (num_latent_frames - 1) * self.vae_scale_factor_temporal + 1
        raise ValueError(
            f"Cosmos3 Sim runs full history and its K/V cache holds {history_frames} latent "
            f"frames of history at this frame size: at most {longest} pixel frames, got "
            f"{requested}. Raise kv_cache_config.free_gpu_memory_fraction or shorten the request."
        )

    # ------------------------------------------------------------------ warmup and sizing

    @property
    def default_warmup_resolutions(self):
        return [self.sim.largest_frame()]

    @property
    def default_warmup_num_frames(self):
        """The shortest rollout that runs every chunk shape: the partition's chunks,
        then a one-frame tail. Kernels and graphs do not depend on the rollout's
        length, and the decoded video is kept in host memory, so a longer warmup
        would measure nothing more."""
        latent = sum(self.sim.chunk_partition) + 1
        return [(latent - 1) * self.vae_scale_factor_temporal + 1]

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        """The first warmup shape runs twice: once over a pool just large enough for it,
        to measure the peak memory everything but the cache needs, and once over the
        pool sized from what is left, so the graphs captured are the ones served. Later
        shapes run once over the settled pool."""
        if self._pool_tokens is not None:
            super()._run_warmup(height, width, num_frames, steps)
            return
        fraction = self.pipeline_config.kv_cache.free_gpu_memory_fraction
        tokens_per_frame = self._tokens_per_frame(height, width)
        self._pool_tokens = self.pool_tokens_for_history(
            self._measuring_history(num_frames), tokens_per_frame
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            super()._run_warmup(height, width, num_frames, steps)
            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            free, total = torch.cuda.mem_get_info()
            outside_torch = max(0, (total - free) - stats["allocated_bytes.all.current"])
            peak = stats["allocated_bytes.all.peak"] + outside_torch
            measuring_pool = self._kv_cache.pool_bytes if self._kv_cache is not None else 0
        finally:
            self._shutdown_kv_cache()
        available = int((total - peak + measuring_pool) * fraction)
        self._pool_tokens = self._pool_tokens_for_bytes(available)
        gib = 1 << 30
        history = self._history_frames_for(tokens_per_frame)
        if history < self.sim.chunk_frames:
            raise RuntimeError(
                f"Cosmos3 Sim cannot hold one chunk of history: warmup peak {peak / gib:.2f} GiB "
                f"of {total / gib:.2f} GiB leaves {available / gib:.2f} GiB for the K/V cache at "
                f"free_gpu_memory_fraction={fraction}; {history} frames of {tokens_per_frame} "
                f"tokens fit, {self.sim.chunk_frames} are needed."
            )
        logger.info(
            f"Cosmos3 Sim K/V cache: {self._pool_tokens // KV_TOKENS_PER_PAGE} pages of "
            f"{KV_TOKENS_PER_PAGE} tokens ({self._pool_bytes_for_tokens(self._pool_tokens) / gib:.2f} GiB): "
            f"{history} latent frames of history at {width}x{height} ({tokens_per_frame} tokens "
            f"per frame). Warmup peak {peak / gib:.2f} GiB of {total / gib:.2f} GiB, fraction {fraction}."
        )
        super()._run_warmup(height, width, num_frames, steps)

    def _measuring_history(self, num_pixel_frames: int) -> int:
        """History the measuring pool holds: enough for the warmup rollout itself, for
        the configured window (the request check asks for it), and never below two
        chunks. The pool's own bytes are added back to the budget, so its size does
        not bias the measurement."""
        return max(
            2 * self.sim.chunk_frames,
            self._latent_frames(num_pixel_frames) - 1,
            self.sim.history_frames or 0,
        )

    def _kv_geometry(self) -> dict:
        tf = self.transformer
        return dict(
            num_layers=len(tf.gen_layers),
            num_kv_heads=tf.cache_kv_heads,
            head_dim=tf.gen_layers[0].cross_attention.head_dim,
            dtype=self.dtype,
        )

    def _pool_bytes_for_tokens(self, pool_tokens: int) -> int:
        return BlockCausalKVCacheManager.bytes_for_pool_tokens(pool_tokens, **self._kv_geometry())

    def _pool_tokens_for_bytes(self, budget_bytes: int) -> int:
        """Pool tokens (whole pages) whose K/V fit ``budget_bytes``. The pool allocator
        maps whole physical chunks of at most 32 MiB, so the mapped size may exceed the
        budget by less than one chunk; the budget is a fraction of free memory and
        leaves far more than that."""
        g = self._kv_geometry()
        bytes_per_token = (
            g["num_layers"]
            * 2
            * g["num_kv_heads"]
            * g["head_dim"]
            * torch.tensor([], dtype=g["dtype"]).element_size()
        )
        return budget_bytes // (KV_TOKENS_PER_PAGE * bytes_per_token) * KV_TOKENS_PER_PAGE

    def pool_tokens_for_history(self, history_frames: int, tokens_per_frame: int) -> int:
        """Pool tokens a rollout with this much history needs at this frame geometry."""
        return BlockCausalKVCacheManager.pool_tokens_for(
            self._rollout_spec(history_frames, tokens_per_frame),
            tokens_per_page=KV_TOKENS_PER_PAGE,
        )

    def _history_frames_for(self, tokens_per_frame: int) -> int:
        """Frames of history a rollout at this frame geometry gets from the pool: the
        configured window, which must fit, or for full history the most that fit."""
        window = self.sim.history_frames
        if window is not None:
            needed = self.pool_tokens_for_history(window, tokens_per_frame)
            if needed > self._pool_tokens:
                raise ValueError(
                    f"Cosmos3 Sim's window of {window} latent frames needs {needed} pool tokens "
                    f"at {tokens_per_frame} tokens per frame; the K/V cache holds "
                    f"{self._pool_tokens}. Raise kv_cache_config.free_gpu_memory_fraction or "
                    "use a smaller frame."
                )
            return window
        # The pool holds (ceil((pins + window + staged) / page) + 1 + private) pages, where
        # the private pages of the causal blocks do not depend on the window: every page
        # left after them, the spare and the fixed tokens is window.
        tpp = KV_TOKENS_PER_PAGE
        spec = self._rollout_spec(1, tokens_per_frame)
        fixed = spec.max_pin_tokens + spec.max_staged_tokens
        private_tokens = (
            self.pool_tokens_for_history(1, tokens_per_frame)
            - (ceil_div(fixed + tokens_per_frame, tpp) + 1) * tpp
        )
        window_pages = (self._pool_tokens - private_tokens) // tpp - 1
        return max(0, (window_pages * tpp - fixed) // tokens_per_frame)

    def _rollout_spec(self, history_frames: int, tokens_per_frame: int) -> RolloutSpec:
        """The rollout's cache geometry: the window, one chunk staged per forward, and the
        causal blocks a forward cuts it into. A denoising step is one causal block of its
        whole chunk, which is ``k`` frames for a chunk of ``k <= chunk_frames`` frames
        (the partition's first chunks and the last one may be shorter); the clean pass
        is one causal block per frame."""
        sim = self.sim
        frames = sim.chunk_frames
        return RolloutSpec(
            window_tokens=max(history_frames, 1) * tokens_per_frame,
            max_staged_tokens=frames * tokens_per_frame,
            causal_blocks={k * tokens_per_frame: frames // k for k in range(1, frames + 1)},
            max_pin_tokens=sim.text_cache_max_len + sim.sink_frames * tokens_per_frame,
        )

    def _tokens_per_frame(self, height: int, width: int) -> int:
        s = self.vae_scale_factor_spatial
        Hp, Wp, _, _ = self.transformer._pad_to_patch_size(height // s, width // s)
        return self.sim.action_tokens_per_frame + Hp * Wp

    def _shutdown_kv_cache(self) -> None:
        if self._kv_cache is not None:
            # Graphs captured over this cache read its pages and tables by address.
            for runner in self._cuda_graph_runners.values():
                runner.clear()
            self._kv_cache.shutdown()
            self._kv_cache = None

    # ------------------------------------------------------------------ CUDA graphs

    def _setup_cuda_graphs(self) -> None:
        """Run the chunk forward under CUDA graphs. The bidirectional forward is not
        wrapped: the rollout never calls it. Keys: the tensor shapes, the packing,
        the pass and the forward plan's ``graph_key`` (cache, rollout spec, token
        count, causal block size). Commits rewrite the cache's device tensors in
        place, so one graph replays for every chunk of the same shape."""
        if not self.pipeline_config.cuda_graph.enable:
            return
        if self.offloader.stages():
            raise NotImplementedError(
                "CUDA graphs are not supported with visual generation offloading yet. "
                "Disable either cuda_graph_config.enable or cpu_offload_config.enable."
            )
        runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
        runner.register_extra_key_fn("clean_pass", lambda *a, **k: bool(k.get("clean_pass")))
        runner.register_extra_key_fn("first_chunk", lambda *a, **k: k.get("first_frame") == 0)
        runner.register_extra_key_fn("packing", lambda *a, **k: k.get("packing"))
        runner.register_extra_key_fn("kv", lambda *a, kv=None, **k: kv.graph_key if kv else None)
        tf = self.transformer
        tf._forward_causal_core = runner.wrap(tf._forward_causal_core)
        self._cuda_graph_runners["transformer.forward_causal"] = runner
        logger.info("CUDA graph runner: wrapping transformer.forward_causal")

    # ------------------------------------------------------------------ the rollout

    def _kv_cache_manager(self) -> BlockCausalKVCacheManager:
        """The one resident page pool, built once and kept across requests of every
        frame geometry: ``close()``/``open()`` keep its device tensors, so graphs
        captured over it stay valid, and a request's spec takes the pages it needs."""
        if self._kv_cache is not None:
            return self._kv_cache
        if self._pool_tokens is None:
            # Warmup was skipped, so nothing measured the peak: size the pool from what
            # is free right now, as the LLM executor does without estimation.
            free, _ = torch.cuda.mem_get_info()
            fraction = self.pipeline_config.kv_cache.free_gpu_memory_fraction
            self._pool_tokens = self._pool_tokens_for_bytes(int(free * fraction))
            logger.warning(
                f"Cosmos3 Sim K/V cache sized without warmup: {self._pool_tokens} pool tokens "
                f"from {free / (1 << 30):.2f} GiB free at fraction {fraction}."
            )
        tf = self.transformer
        attention = tf.gen_layers[0].cross_attention.attn
        if not attention.support_kv_cache_page_size(KV_TOKENS_PER_PAGE, self.dtype):
            raise RuntimeError(
                f"the {tf.gen_layers[0].cross_attention.attn_backend} attention backend has no "
                f"kernel for {KV_TOKENS_PER_PAGE}-token K/V pages with head_dim "
                f"{attention.head_dim} in {self.dtype} on this GPU; use the CUDNN backend."
            )
        self._kv_cache = BlockCausalKVCacheManager(
            tokens_per_page=KV_TOKENS_PER_PAGE,
            pool_tokens=self._pool_tokens,
            **self._kv_geometry(),
        )
        return self._kv_cache

    def _denoise_request(
        self,
        request,
        prepared,
        *,
        cond_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        uncond_ids: torch.Tensor,
        uncond_mask: torch.Tensor,
        generator: torch.Generator,
        timer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """The chunked causal rollout: the prompt's K/V once, then per chunk the fixed
        denoising schedule with the cache read-only, the clean pass that writes the
        finished frames' K/V, and a commit. The last latent frame gets no clean pass:
        nothing reads it."""
        del uncond_ids, uncond_mask  # guidance is 1 on a distilled checkpoint
        if prepared.video_latents is None:
            raise RuntimeError("Cosmos3 Sim request reached the rollout without latents.")
        if request.enable_audio:
            raise ValueError("Cosmos3 Sim does not generate audio.")
        sim, tf = self.sim, self.transformer
        latents = prepared.video_latents
        _, _, num_latent_frames, H, W = latents.shape
        Hp, Wp, _, _ = tf._pad_to_patch_size(H, W)
        A = sim.action_tokens_per_frame
        tokens_per_frame = A + Hp * Wp
        actions, domains = self._sim_action_rows(request, prepared, num_latent_frames, A, latents)
        velocity_mask = prepared.velocity_mask
        offload = self.offloader.context_if_requested

        kv_cache = self._kv_cache_manager()
        history_frames = self._history_frames_for(tokens_per_frame)
        self._check_history_fits(num_latent_frames, history_frames)
        tf.reset_cache()
        timer.mark_denoise_start()
        text_len = int(cond_mask.sum().item())
        if text_len > sim.text_cache_max_len:
            raise ValueError(
                f"prompt of {text_len} tokens exceeds the checkpoint's text cache of "
                f"{sim.text_cache_max_len} tokens"
            )
        decoder = self._start_overlapped_decode(latents)
        kv_cache.open(
            self._rollout_spec(history_frames, tokens_per_frame),
            pin_tokens=text_len + sim.sink_frames * tokens_per_frame,
        )
        try:
            tf.write_prompt_kv(kv_cache, cond_ids, cond_mask, offload_context=offload)
            kv_cache.commit(text_len)
            steps = len(sim.sigmas)
            step_kwargs = self.sampling.scheduler_step_kwargs(generator)
            zero_t = torch.zeros(1, device=self.device)
            for f0, f1 in sim.chunk_ranges(num_latent_frames):
                x = latents[:, :, f0:f1]
                chunk_actions = actions[:, f0 * A : f1 * A]
                domain = domains[f0 * A : f1 * A]
                mask = velocity_mask[:, :, f0:f1] if velocity_mask is not None else None
                if mask is None or bool((mask > 0).any()):
                    # The scheduler restarts per chunk: same four noise levels,
                    # fresh SDE noise from the request's generator.
                    self.sampling.set_timesteps(self.scheduler, steps, device=self.device)
                    for t in self.scheduler.timesteps:
                        t_vec = t.reshape(1).to(self.device)
                        out = tf.forward_causal(
                            x,
                            t_vec,
                            kv_cache=kv_cache,
                            first_frame=f0,
                            text_len=text_len,
                            action_latents=chunk_actions,
                            action_domain_ids=domain,
                            fps=request.frame_rate,
                            action_fps=request.action_fps,
                            timestep=t_vec / self.scheduler.config.num_train_timesteps,
                            offload_context=offload,
                        )
                        velocity = out.video if mask is None else out.video * mask
                        x = self.scheduler.step(velocity, t, x, return_dict=False, **step_kwargs)[0]
                        if mask is not None and prepared.condition_latents is not None:
                            x = mask * x + (1.0 - mask) * prepared.condition_latents[:, :, f0:f1]
                latents[:, :, f0:f1] = x
                if decoder is not None:
                    decoder.submit(latents[:, :, f0:f1])
                if f1 < num_latent_frames:
                    tf.forward_causal(
                        x,
                        zero_t,
                        kv_cache=kv_cache,
                        first_frame=f0,
                        text_len=text_len,
                        action_latents=chunk_actions,
                        action_domain_ids=domain,
                        clean_pass=True,
                        fps=request.frame_rate,
                        action_fps=request.action_fps,
                        timestep=zero_t,
                        offload_context=offload,
                    )
                    kv_cache.commit()
        finally:
            kv_cache.close()
        if decoder is not None:
            self._streamed_video = decoder.finish()
        self._release_scheduler_solver_state()
        return latents, None, prepared.action_latents, False

    # ------------------------------------------------------------------ overlapped VAE decode

    def _start_overlapped_decode(self, latents: torch.Tensor) -> Optional["_OverlappedDecode"]:
        """Decode finished chunks on a second stream while the rollout continues,
        when this rank decodes alone and the VAE is the native streaming decoder;
        otherwise the usual decode after the rollout applies."""
        self._streamed_video = None
        vae = self.vae
        if (
            self._parallel_vae_enabled
            or self.rank != 0
            or not hasattr(vae, "decode_stream_step")
            or self.offloader.stages()
        ):
            return None
        if self._decode_stream is None:
            self._decode_stream = torch.cuda.Stream(device=self._decode_device)
        num_pixel_frames = (latents.shape[2] - 1) * self.vae_scale_factor_temporal + 1
        return _OverlappedDecode(
            self, self._decode_vae or vae, self._decode_stream, num_pixel_frames
        )

    def _decode_latents(self, latents):
        video = self._streamed_video
        if video is None:
            return super()._decode_latents(latents)
        self._streamed_video = None
        return video

    def _sim_action_rows(
        self, request, prepared, num_latent_frames: int, A: int, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[1, T * A, action_dim]`` action rows, frame-major: frame ``f`` gets the ``A``
        rows that lead into it, and ``[T * A]`` embodiment ids, one per row. Frame 0
        has nothing before it and gets null (zero) rows in the first real row's
        embodiment; without an action stream every frame does."""
        tf = self.transformer
        rows = torch.zeros(
            1, num_latent_frames * A, tf.action_dim, device=self.device, dtype=self.dtype
        )
        first_domain = prepared.action_domain_id or 0
        domains = torch.full((num_latent_frames * A,), first_domain, dtype=torch.long)
        given = prepared.action_latents
        if given is not None:
            if prepared.action_state_rows:
                raise ValueError("Cosmos3 Sim does not take a state row; pass the trajectory only.")
            needed = (num_latent_frames - 1) * A
            if given.shape[1] < needed:
                raise ValueError(
                    f"{given.shape[1]} action rows for {num_latent_frames} latent frames; "
                    f"frames 1.. need {needed} ({A} per latent frame)"
                )
            rows[:, A : A + needed] = given[:, :needed]
            if prepared.action_row_domain_ids is not None:
                ids = torch.tensor(prepared.action_row_domain_ids[:needed], dtype=torch.long)
                domains[A : A + needed] = ids
                domains[:A] = ids[0]
        domains = domains.to(self.device)
        tf.action_proj_in.validate_domain_ids(domains)
        tf.domain_ids_validated = True
        return rows, domains


__all__ = [
    "Cosmos3NanoSimBimanualPipeline",
    "Cosmos3SimSettings",
    "SIM_CONFIG_KEY",
    "SimEmbodiment",
]
