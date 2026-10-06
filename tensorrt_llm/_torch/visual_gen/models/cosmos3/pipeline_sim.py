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

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from tensorrt_llm.logger import logger

from ...cache import CausalKVCacheManager
from ...pipeline_registry import register_pipeline
from .action import normalize_action_mode
from .defaults import VIDEO_RES_SIZE_INFO
from .pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from .sim_packing import SIM_CONFIG_KEY


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


@dataclass(frozen=True)
class SimEmbodiment:
    """One embodiment the checkpoint was trained on: its domain id and the width
    of the action rows it expects (``None`` where the block does not say)."""

    domain_id: Optional[int]
    action_dim: Optional[int]


@dataclass(frozen=True)
class Cosmos3SimSettings:
    """The rollout geometry a Sim checkpoint declares in ``transformer/config.json``
    under ``cosmos3_nano_sim_bimanual``. Every field but the noise schedule has a
    declared default, so older exports that omit a key still resolve.

    Frames are latent frames throughout.
    """

    sigmas: tuple[float, ...]
    """The fixed denoising schedule, ``fixed_step_sampler_config.t_list``."""
    chunk_frames: int = 4
    """Latent frames generated together after the first chunk."""
    chunk_partition: tuple[int, ...] = (1, 4)
    """Frames per chunk from the start: the first chunk is one frame, then fours."""
    window_frames: Optional[int] = 96
    """Frames the history holds, counting the chunk's own first frame. ``None``:
    the checkpoint declares unbounded history and the pipeline sizes the window
    from the memory left after warmup."""
    sink_frames: int = 0
    """Leading frames pinned for the whole rollout; 0 on this checkpoint."""
    text_cache_max_len: int = 512
    """Most prompt tokens the fixed region may hold; longer prompts are rejected."""
    action_tokens_per_frame: int = 4
    """Action rows packed before each frame's vision tokens."""
    video_temporal_causal: bool = True
    max_pixels: int = 640 * 640
    """Largest frame the checkpoint serves, in pixels: the 480p bucket."""
    max_pixel_frames: int = 901
    """Longest clip the checkpoint serves, in pixel frames."""
    embodiments: dict[str, SimEmbodiment] = field(default_factory=dict)
    """Embodiments declared under ``conditioning.embodiments``, by lower-case name."""
    default_embodiment: Optional[str] = None
    """``conditioning.default_embodiment``: used when a request names no domain."""

    @property
    def history_frames(self) -> Optional[int]:
        """Finished frames a chunk may read: the window less its own first frame;
        ``None`` when the checkpoint leaves the window to the pipeline."""
        return None if self.window_frames is None else self.window_frames - 1

    def max_latent_frames(self, temporal_compression_factor: int) -> int:
        return (self.max_pixel_frames - 1) // temporal_compression_factor + 1

    def largest_frame(self) -> tuple[int, int]:
        """``(height, width)`` of the biggest 480p bucket shape within ``max_pixels``:
        the shape warmup runs, so the measured peak covers every served frame."""
        shapes = [
            (h, w) for (w, h) in VIDEO_RES_SIZE_INFO["480"].values() if h * w <= self.max_pixels
        ]
        if not shapes:
            raise ValueError(f"no 480p bucket shape fits max_pixels={self.max_pixels}")
        return max(shapes, key=lambda hw: hw[0] * hw[1])

    @classmethod
    def from_pretrained_config(cls, pretrained_config: Any) -> "Cosmos3SimSettings":
        block = _get(pretrained_config, SIM_CONFIG_KEY)
        if block is None:
            raise ValueError(
                f"Not a Cosmos3 Sim checkpoint: transformer/config.json has no "
                f"'{SIM_CONFIG_KEY}' block."
            )
        sampler = _get(block, "fixed_step_sampler_config") or {}
        t_list = _get(sampler, "t_list")
        if not t_list:
            raise ValueError(
                f"'{SIM_CONFIG_KEY}.fixed_step_sampler_config.t_list' is missing or empty; "
                "the Sim rollout needs its fixed denoising schedule."
            )
        chunk_frames = int(_get(block, "chunk_size", cls.chunk_frames))
        partition = _get(block, "chunk_partition") or (1, chunk_frames)
        conditioning = _get(block, "conditioning") or {}
        # Schema 3 names the width per embodiment (`raw_action_dim`); schema 5 has
        # `input_action_dim` per embodiment and one `input_contract.action_dim`.
        contract_dim = _get(_get(conditioning, "input_contract") or {}, "action_dim")
        embodiments = {}
        for name, spec in (_get(conditioning, "embodiments") or {}).items():
            width = _get(spec, "input_action_dim")
            if width is None:
                width = _get(spec, "raw_action_dim", contract_dim)
            domain_id = _get(spec, "domain_id")
            embodiments[str(name).lower()] = SimEmbodiment(
                domain_id=None if domain_id is None else int(domain_id),
                action_dim=None if width is None else int(width),
            )
        default_embodiment = _get(conditioning, "default_embodiment")
        settings = cls(
            sigmas=tuple(float(s) for s in t_list),
            chunk_frames=chunk_frames,
            chunk_partition=tuple(int(n) for n in partition),
            window_frames=_optional_int(_get(block, "window_frames", cls.window_frames)),
            max_pixels=int(_get(block, "max_pixels", cls.max_pixels)),
            max_pixel_frames=int(_get(block, "max_num_frames", cls.max_pixel_frames)),
            sink_frames=int(_get(block, "sink_frames", cls.sink_frames)),
            text_cache_max_len=int(_get(block, "text_cache_max_len", cls.text_cache_max_len)),
            action_tokens_per_frame=int(
                _get(conditioning, "action_tokens_per_frame", cls.action_tokens_per_frame)
            ),
            video_temporal_causal=bool(
                _get(block, "video_temporal_causal", cls.video_temporal_causal)
            ),
            embodiments=embodiments,
            default_embodiment=None
            if default_embodiment is None
            else str(default_embodiment).lower(),
        )
        settings.validate()
        return settings

    def action_contract(
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

    def validate(self) -> None:
        window = self.window_frames if self.window_frames is not None else 1
        if min(self.chunk_frames, window, self.text_cache_max_len, self.max_pixels) <= 0:
            raise ValueError(f"Cosmos3 Sim settings must be positive: {self}")
        if self.max_pixel_frames <= 0:
            raise ValueError(f"max_num_frames must be positive: {self.max_pixel_frames}")
        if not self.chunk_partition or any(n <= 0 for n in self.chunk_partition):
            raise ValueError(
                f"chunk_partition must be positive frame counts: {self.chunk_partition}"
            )
        if self.chunk_partition[-1] != self.chunk_frames:
            raise ValueError(
                f"chunk_partition {self.chunk_partition} must end with chunk_size {self.chunk_frames}"
            )
        if self.sink_frames < 0 or (
            self.window_frames is not None and self.sink_frames >= self.window_frames
        ):
            raise ValueError(f"sink_frames {self.sink_frames} must be below window_frames")
        if not self.video_temporal_causal:
            raise ValueError(
                "Cosmos3 Sim requires video_temporal_causal; this checkpoint declares it false"
            )
        if any(not 0.0 < s <= 1.0 for s in self.sigmas) or list(self.sigmas) != sorted(
            self.sigmas, reverse=True
        ):
            raise ValueError(f"t_list must decrease within (0, 1]: {self.sigmas}")

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
        self.sim = Cosmos3SimSettings.from_pretrained_config(
            pipeline_config.primary_pretrained_config
        )
        # Finished frames a chunk may read. Declared by the checkpoint, or settled
        # once by warmup from the memory left at the largest served shape.
        self._history_frames: Optional[int] = self.sim.history_frames
        # One cache is resident at a time, keyed by its frame geometry.
        self._sim_cache_obj: Optional[CausalKVCacheManager] = None
        self._sim_cache_key: Optional[int] = None

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
        if normalize_action_mode(action_mode) is not None:
            domain_name, domain_id, raw_action_dim = self.sim.action_contract(
                domain_name, domain_id, raw_action_dim
            )
        resolved = super()._resolve_request(
            action_mode=action_mode,
            domain_name=domain_name,
            domain_id=domain_id,
            raw_action_dim=raw_action_dim,
            **kwargs,
        )
        self.check_envelope(resolved.height, resolved.width, resolved.num_frames)
        return resolved

    def check_envelope(self, height: int, width: int, num_frames: int) -> None:
        """Refuse what the checkpoint does not serve. The cache was sized at the
        largest served shape, so a bigger request would not have been measured."""
        sim = self.sim
        if height * width > sim.max_pixels or num_frames > sim.max_pixel_frames:
            raise ValueError(
                f"Cosmos3 Sim serves at most {sim.max_pixels} pixels per frame "
                f"({'x'.join(map(str, reversed(sim.largest_frame())))}) and "
                f"{sim.max_pixel_frames} frames; got {width}x{height}, {num_frames} frames."
            )

    # ------------------------------------------------------------------ warmup and sizing

    @property
    def default_warmup_resolutions(self):
        return [self.sim.largest_frame()]

    @property
    def default_warmup_num_frames(self):
        return [self.sim.max_pixel_frames]

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        """Warm up at the given shape. With no declared window, the run uses a
        two-chunk cache, its peak is measured, and the window is sized from what
        is left, the way the LLM executor sizes its K/V pool."""
        if self._history_frames is not None:
            super()._run_warmup(height, width, num_frames, steps)
            return
        fraction = self.pipeline_config.kv_cache.free_gpu_memory_fraction
        self._history_frames = 2 * self.sim.chunk_frames
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
            temp_cache = self._sim_cache_obj.pool_bytes if self._sim_cache_obj is not None else 0
        finally:
            self._shutdown_sim_cache()
        available = int((total - peak + temp_cache) * fraction)
        tokens_per_frame = self._tokens_per_frame(height, width)
        cap = self.sim.max_latent_frames(self.vae_scale_factor_temporal) - 1
        frames = self.history_frames_for_budget(available, tokens_per_frame, cap)
        gib = 1 << 30
        if frames < self.sim.chunk_frames:
            raise RuntimeError(
                f"Cosmos3 Sim cannot hold one chunk of history: warmup peak {peak / gib:.2f} GiB "
                f"of {total / gib:.2f} GiB leaves {available / gib:.2f} GiB for the K/V cache at "
                f"free_gpu_memory_fraction={fraction}; {frames} frames of {tokens_per_frame} tokens fit, "
                f"{self.sim.chunk_frames} are needed."
            )
        self._history_frames = frames
        logger.info(
            f"Cosmos3 Sim K/V window: {frames} latent frames (cap {cap}). Warmup peak "
            f"{peak / gib:.2f} GiB of {total / gib:.2f} GiB, {available / gib:.2f} GiB for the cache "
            f"at fraction {fraction}, {tokens_per_frame} tokens per frame at {width}x{height}."
        )

    def history_frames_for_budget(self, budget_bytes: int, tokens_per_frame: int, cap: int) -> int:
        """Most frames of history whose cache fits ``budget_bytes``, at most ``cap``."""
        lo, hi = 0, max(cap, 0)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cache_bytes(mid, tokens_per_frame) <= budget_bytes:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _cache_bytes(self, history_frames: int, tokens_per_frame: int) -> int:
        tf, sim = self.transformer, self.sim
        return CausalKVCacheManager.pool_bytes_for(
            num_layers=len(tf.gen_layers),
            num_kv_heads=tf.cache_kv_heads,
            head_dim=tf.gen_layers[0].cross_attention.head_dim,
            dtype=self.dtype,
            tokens_per_page=32,
            fixed_capacity=sim.text_cache_max_len + sim.sink_frames * tokens_per_frame,
            window_tokens=max(history_frames, 1) * tokens_per_frame,
            chunk_tokens=sim.chunk_frames * tokens_per_frame,
            causal_block_sizes=tuple(k * tokens_per_frame for k in range(sim.chunk_frames, 0, -1)),
        )

    def _tokens_per_frame(self, height: int, width: int) -> int:
        s = self.vae_scale_factor_spatial
        Hp, Wp, _, _ = self.transformer._pad_to_patch_size(height // s, width // s)
        return self.sim.action_tokens_per_frame + Hp * Wp

    def _shutdown_sim_cache(self) -> None:
        if self._sim_cache_obj is not None:
            self._sim_cache_obj.shutdown()
            self._sim_cache_obj = None
            self._sim_cache_key = None

    # ------------------------------------------------------------------ the rollout

    def _sim_cache(self, tokens_per_frame: int) -> CausalKVCacheManager:
        """The resident cache for this frame geometry, kept across requests:
        ``close()``/``open()`` keep its device tensors, so graphs captured over it
        stay valid. A request with another geometry replaces it; two pools would
        split the memory the window was sized for."""
        if self._sim_cache_obj is not None and self._sim_cache_key == tokens_per_frame:
            return self._sim_cache_obj
        self._shutdown_sim_cache()
        if self._history_frames is None:
            # Warmup was skipped, so nothing measured the peak: size the window from
            # what is free right now, as the LLM executor does without estimation.
            free, _ = torch.cuda.mem_get_info()
            fraction = self.pipeline_config.kv_cache.free_gpu_memory_fraction
            cap = self.sim.max_latent_frames(self.vae_scale_factor_temporal) - 1
            self._history_frames = self.history_frames_for_budget(
                int(free * fraction), tokens_per_frame, cap
            )
            logger.warning(
                f"Cosmos3 Sim K/V window sized without warmup: {self._history_frames} latent "
                f"frames from {free / (1 << 30):.2f} GiB free at fraction {fraction}."
            )
        if self._history_frames < self.sim.chunk_frames:
            raise RuntimeError(
                f"Cosmos3 Sim K/V window of {self._history_frames} frames is below one chunk "
                f"({self.sim.chunk_frames} frames)."
            )
        sim, tf = self.sim, self.transformer
        self._sim_cache_obj = CausalKVCacheManager(
            num_layers=len(tf.gen_layers),
            num_kv_heads=tf.cache_kv_heads,
            head_dim=tf.gen_layers[0].cross_attention.head_dim,
            dtype=self.dtype,
            tokens_per_page=32,
            fixed_capacity=sim.text_cache_max_len + sim.sink_frames * tokens_per_frame,
            window_tokens=self._history_frames * tokens_per_frame,
            chunk_tokens=sim.chunk_frames * tokens_per_frame,
            causal_block_sizes=tuple(k * tokens_per_frame for k in range(sim.chunk_frames, 0, -1)),
        )
        self._sim_cache_key = tokens_per_frame
        return self._sim_cache_obj

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
        actions = self._sim_action_rows(request, prepared, num_latent_frames, A, latents)
        domain = (
            torch.tensor([prepared.action_domain_id], dtype=torch.long, device=self.device)
            if prepared.action_domain_id is not None
            else None
        )
        velocity_mask = prepared.velocity_mask
        offload = self.offloader.context_if_requested

        cache = self._sim_cache(tokens_per_frame)
        tf.reset_cache()
        timer.mark_denoise_start()
        text_len = int(cond_mask.sum().item())
        cache.open(pin_tokens=text_len + sim.sink_frames * tokens_per_frame)
        try:
            tf.write_prompt_kv(cache, cond_ids, cond_mask, offload_context=offload)
            cache.commit(text_len)
            steps = len(sim.sigmas)
            step_kwargs = self.sampling.scheduler_step_kwargs(generator)
            zero_t = torch.zeros(1, device=self.device)
            for f0, f1 in sim.chunk_ranges(num_latent_frames):
                x = latents[:, :, f0:f1]
                chunk_actions = actions[:, f0 * A : f1 * A]
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
                            kv_cache=cache,
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
                if f1 < num_latent_frames:
                    tf.forward_causal(
                        x,
                        zero_t,
                        kv_cache=cache,
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
                    cache.commit((f1 - f0) * tokens_per_frame)
        finally:
            cache.close()
        self._release_scheduler_solver_state()
        return latents, None, prepared.action_latents, False

    def _sim_action_rows(
        self, request, prepared, num_latent_frames: int, A: int, latents: torch.Tensor
    ) -> torch.Tensor:
        """``[1, T * A, action_dim]`` action rows, frame-major: frame ``f`` gets the ``A``
        rows that lead into it. Frame 0 has nothing before it and gets null (zero)
        rows; without an action stream every frame does."""
        rows = torch.zeros(
            1,
            num_latent_frames * A,
            self.transformer.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        given = prepared.action_latents
        if given is None:
            return rows
        if prepared.action_state_rows:
            raise ValueError("Cosmos3 Sim does not take a state row; pass the trajectory only.")
        needed = (num_latent_frames - 1) * A
        if given.shape[1] < needed:
            raise ValueError(
                f"{given.shape[1]} action rows for {num_latent_frames} latent frames; "
                f"frames 1.. need {needed} ({A} per latent frame)"
            )
        rows[:, A : A + needed] = given[:, :needed]
        return rows


__all__ = [
    "Cosmos3NanoSimBimanualPipeline",
    "Cosmos3SimSettings",
    "SIM_CONFIG_KEY",
    "SimEmbodiment",
]
