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

import contextlib
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import torch
from diffusers.utils.torch_utils import randn_tensor

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div

from ...cache import BlockCausalKVCacheManager, RolloutSpec
from ...cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from ...pipeline import AdvanceOutcome, InputSpec, SessionCapabilities, StreamSpec
from ...pipeline_registry import register_pipeline
from ...session import input_nbytes
from ...utils import make_noise_generator, postprocess_video_tensor
from ..wan.vae_loader import load_wan_vae
from .action import load_action_tensor, normalize_action_mode, pad_action_to_dim
from .config_resolver import Cosmos3Config
from .defaults import VIDEO_RES_SIZE_INFO
from .guardrails import check_video_safety
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


class _ChunkDecoder:
    """Pixel frames of a rollout, decoded chunk by chunk as the chunks finish. Each
    chunk's uint8 frames are copied straight into one pinned host buffer, so GPU
    memory does not grow with the video's length; ``finish`` returns that buffer,
    ``[1, T, H, W, 3]``, the video the pipeline would otherwise decode after the
    rollout.

    On the generator's GPU the decode runs on the main stream, between one chunk's
    last denoising step and the next chunk's first: a GPU at its power cap gains
    nothing from overlapping the two. With ``stream`` on another GPU holding its own
    copy of the VAE, each chunk's latents (a few hundred KB) hop over as they finish
    and the decode leaves the generator's GPU alone."""

    def __init__(
        self, pipe, vae, stream: Optional[torch.cuda.Stream], num_pixel_frames: int
    ) -> None:
        self.pipe = pipe
        self.vae = vae
        self.stream = stream
        self.num_pixel_frames = num_pixel_frames
        self.host: Optional[torch.Tensor] = None
        self.written = 0
        self.held: list[torch.Tensor] = []
        with self._on_decode_stream():
            vae.decode_stream_start()

    @contextlib.contextmanager
    def _on_decode_stream(self):
        if self.stream is None:
            yield
            return
        with torch.cuda.device(self.stream.device), torch.cuda.stream(self.stream):
            yield

    def submit(self, chunk: torch.Tensor) -> None:
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
        with self._on_decode_stream():
            if self.stream is not None:
                ready.wait()
                self.held.append(chunk)  # the main stream never writes these frames again
                if chunk.device != self.stream.device:
                    chunk = chunk.to(self.stream.device, non_blocking=True)
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
            # The copy is queued on the decoding stream; the frames' memory goes back
            # to that stream's pool and is reused only by work ordered after the copy.
            self.host[:, self.written : self.written + t].copy_(frames, non_blocking=True)
            self.written += t

    def finish(self) -> torch.Tensor:
        with self._on_decode_stream():
            self.vae.decode_stream_end()
            torch.cuda.current_stream().synchronize()  # the CPU reads the host buffer next
        self.held.clear()
        if self.written != self.num_pixel_frames:
            raise RuntimeError(
                f"Cosmos3 Sim chunk decode wrote {self.written} of "
                f"{self.num_pixel_frames} pixel frames."
            )
        video, self.host = self.host, None
        return video


@dataclass
class _SimRollout:
    """One open rollout: the request, the opened K/V cache and what every chunk needs."""

    request: Any
    prepared: Any
    seed: int
    kv_cache: BlockCausalKVCacheManager
    text_len: int
    tokens_per_frame: int
    history_frames: int
    latent_shape: tuple[int, int, int]
    offload: Any


@dataclass
class _SimSessionSetup:
    request: Any
    prompt: str
    negative_prompt: Optional[str]
    seed: int
    use_guardrails: bool
    use_duration_template: bool
    use_resolution_template: bool


class _SimSession:
    """Pipeline-side state of one session: the rollout, the chunk ranges still to
    run, the embodiment and the streaming decoder's lifetime."""

    def __init__(self, *, rollout, ranges, domain_id, use_guardrails, remaining, device):
        self.rollout = rollout
        self.device = device
        self._ranges = ranges
        self._pending: Optional[tuple[int, int]] = None
        self.domain_id = domain_id
        self.use_guardrails = use_guardrails
        self.remaining = remaining
        self.decode_started = False
        self.closed = False

    def peek_range(self) -> tuple[int, int]:
        if self._pending is None:
            self._pending = next(self._ranges)
        return self._pending

    def take_range(self) -> tuple[int, int]:
        r = self.peek_range()
        self._pending = None
        return r

    def domain_tensor(self, count: int) -> torch.Tensor:
        return torch.full((count,), self.domain_id, dtype=torch.long, device=self.device)


def _rows_tensor(value) -> torch.Tensor:
    """Action rows as a float ``[rows, width]`` tensor, from a tensor, a list of rows
    or a NumPy array."""
    rows = load_action_tensor(value)
    if rows.ndim != 2:
        raise ValueError(f"actions must be a [rows, width] array, got shape {tuple(rows.shape)}")
    return rows


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
        # Finished chunks are decoded as they complete; on a second GPU, when the
        # serving config names one, with its own VAE instance and stream.
        self._decode_stream: Optional[torch.cuda.Stream] = None
        self._decoded_video: Optional[torch.Tensor] = None
        decode_device = pipeline_config.vae_decode_device
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

    def _chunk_generator(self, seed: int, first_frame: int) -> torch.Generator:
        """The random stream of one chunk, seeded by the request seed and the chunk's
        first latent frame: a chunk draws the same noise whether the rollout runs as
        one request or as a session advanced one chunk at a time."""
        return make_noise_generator(int(seed) + first_frame, self.device)

    def _rollout_begin(self, request, prepared, *, cond_ids, cond_mask, seed) -> "_SimRollout":
        """Open the rollout: the prompt's K/V once, then the cache is ready for chunks."""
        if prepared.video_latents is None:
            raise RuntimeError("Cosmos3 Sim request reached the rollout without latents.")
        if request.enable_audio:
            raise ValueError("Cosmos3 Sim does not generate audio.")
        sim, tf = self.sim, self.transformer
        _, C, _, H, W = prepared.video_latents.shape
        Hp, Wp, _, _ = tf._pad_to_patch_size(H, W)
        A = sim.action_tokens_per_frame
        tokens_per_frame = A + Hp * Wp
        kv_cache = self._kv_cache_manager()
        history_frames = self._history_frames_for(tokens_per_frame)
        tf.reset_cache()
        text_len = int(cond_mask.sum().item())
        if text_len > sim.text_cache_max_len:
            raise ValueError(
                f"prompt of {text_len} tokens exceeds the checkpoint's text cache of "
                f"{sim.text_cache_max_len} tokens"
            )
        offload = self.offloader.context_if_requested
        kv_cache.open(
            self._rollout_spec(history_frames, tokens_per_frame),
            pin_tokens=text_len + sim.sink_frames * tokens_per_frame,
        )
        try:
            tf.write_prompt_kv(kv_cache, cond_ids, cond_mask, offload_context=offload)
            kv_cache.commit(text_len)
        except Exception:
            kv_cache.close()
            raise
        return _SimRollout(
            request=request,
            prepared=prepared,
            seed=int(seed),
            kv_cache=kv_cache,
            text_len=text_len,
            tokens_per_frame=tokens_per_frame,
            history_frames=history_frames,
            latent_shape=(C, H, W),
            offload=offload,
        )

    def _rollout_chunk(
        self,
        ro: "_SimRollout",
        f0: int,
        f1: int,
        chunk_actions: torch.Tensor,
        domain: torch.Tensor,
        *,
        commit: bool,
    ) -> torch.Tensor:
        """Denoise latent frames ``[f0, f1)`` against the committed history and, with
        ``commit``, run the clean pass that writes their K/V. Frames the request
        conditions (the first image) are pinned through the velocity mask. Returns the
        chunk's clean latents ``[1, C, f1 - f0, H, W]``."""
        sim, tf, kv_cache = self.sim, self.transformer, ro.kv_cache
        request, prepared = ro.request, ro.prepared
        k = f1 - f0
        C, H, W = ro.latent_shape
        generator = self._chunk_generator(ro.seed, f0)
        x = randn_tensor((1, C, k, H, W), generator=generator, device=self.device, dtype=self.dtype)
        mask = None
        cond = None
        if (
            prepared.velocity_mask is not None
            and prepared.condition_latents is not None
            and f0 < prepared.velocity_mask.shape[2]
        ):
            n = min(f1, prepared.velocity_mask.shape[2]) - f0
            mask = torch.ones((1, 1, k, 1, 1), device=self.device, dtype=self.dtype)
            mask[:, :, :n] = prepared.velocity_mask[:, :, f0 : f0 + n]
            cond = torch.zeros_like(x)
            cond[:, :, :n] = prepared.condition_latents[:, :, f0 : f0 + n]
            x = mask * x + (1.0 - mask) * cond
        if mask is None or bool((mask > 0).any()):
            # The scheduler restarts per chunk: same noise levels, this chunk's SDE noise.
            steps = len(sim.sigmas)
            self.sampling.set_timesteps(self.scheduler, steps, device=self.device)
            step_kwargs = self.sampling.scheduler_step_kwargs(generator)
            for t in self.scheduler.timesteps:
                t_vec = t.reshape(1).to(self.device)
                out = tf.forward_causal(
                    x,
                    t_vec,
                    kv_cache=kv_cache,
                    first_frame=f0,
                    text_len=ro.text_len,
                    action_latents=chunk_actions,
                    action_domain_ids=domain,
                    fps=request.frame_rate,
                    action_fps=request.action_fps,
                    timestep=t_vec / self.scheduler.config.num_train_timesteps,
                    offload_context=ro.offload,
                )
                velocity = out.video if mask is None else out.video * mask
                x = self.scheduler.step(velocity, t, x, return_dict=False, **step_kwargs)[0]
                if mask is not None:
                    x = mask * x + (1.0 - mask) * cond
        if commit:
            zero_t = torch.zeros(1, device=self.device)
            tf.forward_causal(
                x,
                zero_t,
                kv_cache=kv_cache,
                first_frame=f0,
                text_len=ro.text_len,
                action_latents=chunk_actions,
                action_domain_ids=domain,
                clean_pass=True,
                fps=request.frame_rate,
                action_fps=request.action_fps,
                timestep=zero_t,
                offload_context=ro.offload,
            )
            kv_cache.commit()
        return x

    def _rollout_end(self, ro: "_SimRollout") -> None:
        ro.kv_cache.close()
        self._release_scheduler_solver_state()

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
        """The chunked causal rollout of one request: the prompt's K/V once, then per
        chunk the fixed denoising schedule with the cache read-only, the clean pass
        that writes the finished frames' K/V, and a commit. The last latent frame gets
        no clean pass: nothing reads it."""
        del uncond_ids, uncond_mask  # guidance is 1 on a distilled checkpoint
        sim = self.sim
        latents = prepared.video_latents
        if latents is None:
            raise RuntimeError("Cosmos3 Sim request reached the rollout without latents.")
        num_latent_frames = latents.shape[2]
        A = sim.action_tokens_per_frame
        actions, domains = self._sim_action_rows(request, prepared, num_latent_frames, A, latents)
        timer.mark_denoise_start()
        ro = self._rollout_begin(
            request, prepared, cond_ids=cond_ids, cond_mask=cond_mask, seed=generator.initial_seed()
        )
        try:
            self._check_history_fits(num_latent_frames, ro.history_frames)
            decoder = self._start_chunk_decode(latents)
            for f0, f1 in sim.chunk_ranges(num_latent_frames):
                x = self._rollout_chunk(
                    ro,
                    f0,
                    f1,
                    actions[:, f0 * A : f1 * A],
                    domains[f0 * A : f1 * A],
                    commit=f1 < num_latent_frames,
                )
                latents[:, :, f0:f1] = x
                if decoder is not None:
                    decoder.submit(latents[:, :, f0:f1])
        finally:
            self._rollout_end(ro)
        if decoder is not None:
            self._decoded_video = decoder.finish()
        return latents, None, prepared.action_latents, False

    # ------------------------------------------------------------------ chunk VAE decode

    def _start_chunk_decode(self, latents: torch.Tensor) -> Optional["_ChunkDecoder"]:
        """Decode finished chunks during the rollout when this rank decodes alone and
        the VAE is the native streaming decoder; otherwise the usual decode after the
        rollout applies."""
        self._decoded_video = None
        if not self._streams_decode():
            return None
        if self._decode_device is not None and self._decode_stream is None:
            self._decode_stream = torch.cuda.Stream(device=self._decode_device)
        num_pixel_frames = (latents.shape[2] - 1) * self.vae_scale_factor_temporal + 1
        return _ChunkDecoder(
            self, self._decode_vae or self.vae, self._decode_stream, num_pixel_frames
        )

    def _streams_decode(self) -> bool:
        """Whether this rank decodes finished chunks itself with the streaming VAE."""
        return (
            not self._parallel_vae_enabled
            and self.rank == 0
            and hasattr(self.vae, "decode_stream_step")
            and not self.offloader.stages()
        )

    def _decode_latents(self, latents):
        video = self._decoded_video
        if video is None:
            return super()._decode_latents(latents)
        self._decoded_video = None
        return video

    # ------------------------------------------------------------------ sessions

    @property
    def supports_sessions(self) -> bool:
        return True

    def _session_setup(self, req) -> "_SimSessionSetup":
        """Resolve a session request the way ``infer`` resolves a generation request:
        action rows condition the rollout (forward dynamics), the first image is the
        first frame, and the frame geometry comes from the request or the mode table.
        Nothing here touches the GPU."""
        params = req.params
        extra = params.extra_params or {}
        specified = params.model_fields_set

        def as_given(name):
            value = getattr(params, name)
            return value if name in specified else None

        refs_i = params.image_reference
        image = refs_i[0].content if refs_i else None
        latent = sum(self.sim.chunk_partition) + 1
        num_frames = (latent - 1) * self.vae_scale_factor_temporal + 1
        request = self._resolve_request(
            image=image,
            height=as_given("height"),
            width=as_given("width"),
            num_frames=num_frames,
            num_inference_steps=as_given("num_inference_steps"),
            guidance_scale=as_given("guidance_scale"),
            max_sequence_length=params.max_sequence_length,
            frame_rate=as_given("frame_rate"),
            use_system_prompt=extra.get("use_system_prompt"),
            enable_audio=False,
            output_type="video",
            video=None,
            flow_shift=None,
            action_mode="forward_dynamics",
            domain_name=extra.get("domain_name"),
            domain_id=extra.get("domain_id"),
            raw_action_dim=extra.get("raw_action_dim"),
            action_chunk_size=num_frames - 1,
            use_state=None,
            action_resolution=extra.get("action_resolution"),
            action_fps=extra.get("action_fps"),
            transfer_config=None,
        )
        if request.raw_action_dim is None:
            raise ValueError("Cosmos3 Sim session needs the action width: name the embodiment")
        if not self._streams_decode() and self.rank == 0:
            raise ValueError(
                "Cosmos3 Sim sessions decode each chunk with the streaming Wan VAE on rank 0; "
                "parallel VAE and CPU offload are not supported with sessions"
            )
        return _SimSessionSetup(
            request=request,
            prompt=req.prompt[0],
            negative_prompt=params.negative_prompt,
            seed=int(params.seed),
            use_guardrails=bool(extra.get("use_guardrails", True)),
            use_duration_template=bool(extra.get("use_duration_template", False)),
            use_resolution_template=bool(extra.get("use_resolution_template", False)),
        )

    def _session_ranges(self):
        """Chunk ranges of an unbounded rollout, in order."""
        sim = self.sim
        f0 = 0
        i = 0
        while True:
            size = sim.chunk_partition[i] if i < len(sim.chunk_partition) else sim.chunk_frames
            yield f0, f0 + size
            f0, i = f0 + size, i + 1

    def _session_length(self, request) -> Optional[int]:
        """Advances a session may run: open-ended with a configured window; with full
        history, as many chunks as the cache holds frames for."""
        if self.sim.history_frames is not None:
            return None
        if self._pool_tokens is None:
            self._kv_cache_manager()  # sizes the pool when warmup did not
        tokens_per_frame = self._tokens_per_frame(request.height, request.width)
        history = self._history_frames_for(tokens_per_frame)
        count = 0
        for f0, f1 in self._session_ranges():
            if f1 - 1 > history:
                break
            if f1 > 1:
                count += 1
        return count

    def _action_rows_per_advance(self) -> tuple[int, ...]:
        """Row counts one advance may take: generated frames of a chunk times the rows
        per frame. The first chunk generates one frame fewer when it holds the image."""
        sim = self.sim
        A = sim.action_tokens_per_frame
        counts = set()
        for i, (f0, f1) in zip(range(len(sim.chunk_partition) + 1), self._session_ranges()):
            generated = f1 - max(f0, 1)
            if generated > 0:
                counts.add(generated * A)
        return tuple(sorted(counts))

    def session_capabilities(self, req) -> SessionCapabilities:
        setup = self._session_setup(req)
        request = setup.request
        sim = self.sim
        width = int(request.raw_action_dim)
        allowed = self._action_rows_per_advance()

        def validate_rows(value):
            rows = _rows_tensor(value)
            if rows.shape[1] != width:
                raise ValueError(f"actions: {rows.shape[1]} values per row, {width} expected")
            if rows.shape[0] not in allowed:
                raise ValueError(
                    f"actions: {rows.shape[0]} rows; an advance takes one of {allowed}"
                )

        max_generated = max(allowed) // sim.action_tokens_per_frame
        frames_per_advance = max_generated * self.vae_scale_factor_temporal
        # The most the largest accepted value can cost in its largest supported
        # form, a list of rows of Python numbers however it was built (a tensor
        # or array of the same shape is smaller).
        action_bytes = input_nbytes([[0.0] * width for _ in range(max(allowed))], worst_case=True)
        return SessionCapabilities(
            inputs=(InputSpec(name="actions", max_bytes=action_bytes, validator=validate_rows),),
            streams=(
                StreamSpec(
                    name="video",
                    kind="video",
                    dtype=torch.uint8,
                    max_shape=(1, frames_per_advance, request.height, request.width, 3),
                    rate=float(request.frame_rate),
                ),
            ),
            initial_result=True,
            length=self._session_length(request),
            lockstep=True,
        )

    def open_session(self, req, initial_slot):
        setup = self._session_setup(req)
        request = setup.request
        if self._is_text_blocked([setup.prompt], setup.negative_prompt, setup.use_guardrails):
            raise ValueError("Cosmos3 Sim session: the prompt was blocked by the guardrails")
        generator = make_noise_generator(setup.seed, self.device)
        cond_ids, cond_mask, _, _ = self._tokenize_request_prompts(
            request,
            [setup.prompt],
            setup.negative_prompt,
            use_duration_template=setup.use_duration_template,
            use_resolution_template=setup.use_resolution_template,
        )
        # The prepared latents supply the image's latent and the conditioning mask of
        # the first chunk; the rollout draws its own noise per chunk, and the rows here
        # are placeholders the session replaces at every advance.
        placeholder_rows = torch.zeros(request.action_chunk_size, int(request.raw_action_dim))
        prepared = self._prepare_request_latents(
            request,
            video=None,
            condition_video_latent_indexes=None,
            condition_video_keep=None,
            action=placeholder_rows,
            generator=generator,
        )
        ro = self._rollout_begin(
            request, prepared, cond_ids=cond_ids, cond_mask=cond_mask, seed=setup.seed
        )
        return self._session_start(
            ro,
            prepared,
            initial_slot,
            domain_id=int(prepared.action_domain_id or 0),
            use_guardrails=setup.use_guardrails,
            remaining=self._session_length(request),
        )

    def _session_start(
        self,
        ro: "_SimRollout",
        prepared,
        initial_slot,
        *,
        domain_id: int,
        use_guardrails: bool,
        remaining: Optional[int],
    ) -> tuple["_SimSession", AdvanceOutcome]:
        """From an open rollout to a session ready for its first advance: commit the
        leading chunks that generate nothing (the image alone) and decode the first
        frame into ``initial_slot``."""
        session = _SimSession(
            rollout=ro,
            ranges=self._session_ranges(),
            domain_id=domain_id,
            use_guardrails=use_guardrails,
            remaining=remaining,
            device=self.device,
        )
        try:
            A = self.sim.action_tokens_per_frame
            while True:
                f0, f1 = session.peek_range()
                if f1 > 1:
                    break
                session.take_range()
                self._rollout_chunk(
                    ro,
                    f0,
                    f1,
                    self._null_rows((f1 - f0) * A),
                    session.domain_tensor((f1 - f0) * A),
                    commit=True,
                )
            if initial_slot is not None:
                self.vae.decode_stream_start()
                session.decode_started = True
                image_latent = prepared.condition_latents[:, :, 0:1]
                outcome = self._session_write_frames(session, image_latent, initial_slot)
            else:
                outcome = AdvanceOutcome(event=self._completion_event())
        except Exception:
            self.close_session(session)
            raise
        return session, outcome

    def validate_input(self, state, name, value) -> None:
        del state, name  # the capability's validator already checked shape and width
        rows = _rows_tensor(value)
        if not torch.isfinite(rows).all():
            raise ValueError("actions must be finite")

    def advance(self, state: "_SimSession", inputs, slot) -> AdvanceOutcome:
        sim, tf = self.sim, self.transformer
        A = sim.action_tokens_per_frame
        f0, f1 = state.take_range()
        generated = f1 - max(f0, 1)
        rows = _rows_tensor(inputs["actions"])
        if rows.shape[0] != generated * A:
            raise ValueError(
                f"advance over latent frames [{f0}, {f1}) takes {generated * A} action rows "
                f"({A} per generated frame), got {rows.shape[0]}"
            )
        padded = pad_action_to_dim(rows, int(tf.action_dim)).to(self.device, self.dtype)
        if f0 == 0:
            padded = torch.cat([self._null_rows(A)[0], padded], dim=0)
        chunk_actions = padded.unsqueeze(0)
        x = self._rollout_chunk(
            state.rollout, f0, f1, chunk_actions, state.domain_tensor((f1 - f0) * A), commit=True
        )
        done = False
        if state.remaining is not None:
            state.remaining -= 1
            done = state.remaining <= 0
        if slot is None:
            return AdvanceOutcome(event=self._completion_event(), done=done)
        new_frames = x[:, :, (1 if f0 == 0 else 0) :]
        outcome = self._session_write_frames(state, new_frames, slot)
        outcome.done = done
        return outcome

    def _session_write_frames(self, state: "_SimSession", latents, slot) -> AdvanceOutcome:
        """Decode latent frames with the streaming VAE straight into the slot."""
        z = self._scale_latents_for_decode(latents)
        frames = postprocess_video_tensor(self.vae.decode_stream_step(z))
        if state.use_guardrails and getattr(self, "safety_checker", None) is not None:
            frames = check_video_safety(frames, self.safety_checker)
            if frames is None:
                raise ValueError("Cosmos3 Sim session: the video was blocked by the guardrails")
        view = slot.buffer("video", tuple(frames.shape), torch.uint8)
        view.copy_(frames, non_blocking=True)
        return AdvanceOutcome(event=self._completion_event())

    def _completion_event(self) -> Optional[torch.cuda.Event]:
        """An event recorded behind everything this rank queued on the current
        stream: the rollout's cache writes and any slot copy. ``None`` off CUDA,
        where that work ran synchronously."""
        if self.device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        return event

    def close_session(self, state: "_SimSession") -> None:
        if state.decode_started:
            self.vae.decode_stream_end()
            state.decode_started = False
        if not state.closed:
            state.closed = True
            self._rollout_end(state.rollout)

    def _null_rows(self, count: int) -> torch.Tensor:
        return torch.zeros(
            1, count, self.transformer.action_dim, device=self.device, dtype=self.dtype
        )

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
