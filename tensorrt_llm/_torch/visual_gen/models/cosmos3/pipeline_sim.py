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

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from ...cache import CausalKVCacheManager
from ...pipeline_registry import register_pipeline
from .pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from .sim_packing import SIM_CONFIG_KEY


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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
    window_frames: int = 96
    """Frames the history holds, counting the chunk's own first frame."""
    sink_frames: int = 0
    """Leading frames pinned for the whole rollout; 0 on this checkpoint."""
    text_cache_max_len: int = 512
    """Most prompt tokens the fixed region may hold; longer prompts are rejected."""
    action_tokens_per_frame: int = 4
    """Action rows packed before each frame's vision tokens."""
    video_temporal_causal: bool = True

    @property
    def history_frames(self) -> int:
        """Finished frames a chunk may read: the window less its own first frame."""
        return self.window_frames - 1

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
        settings = cls(
            sigmas=tuple(float(s) for s in t_list),
            chunk_frames=chunk_frames,
            chunk_partition=tuple(int(n) for n in partition),
            window_frames=int(_get(block, "window_frames", cls.window_frames)),
            sink_frames=int(_get(block, "sink_frames", cls.sink_frames)),
            text_cache_max_len=int(_get(block, "text_cache_max_len", cls.text_cache_max_len)),
            action_tokens_per_frame=int(
                _get(conditioning, "action_tokens_per_frame", cls.action_tokens_per_frame)
            ),
            video_temporal_causal=bool(
                _get(block, "video_temporal_causal", cls.video_temporal_causal)
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if min(self.chunk_frames, self.window_frames, self.text_cache_max_len) <= 0:
            raise ValueError(f"Cosmos3 Sim settings must be positive: {self}")
        if not self.chunk_partition or any(n <= 0 for n in self.chunk_partition):
            raise ValueError(
                f"chunk_partition must be positive frame counts: {self.chunk_partition}"
            )
        if self.chunk_partition[-1] != self.chunk_frames:
            raise ValueError(
                f"chunk_partition {self.chunk_partition} must end with chunk_size {self.chunk_frames}"
            )
        if not 0 <= self.sink_frames < self.window_frames:
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
        self._sim_caches: dict[int, CausalKVCacheManager] = {}

    # ------------------------------------------------------------------ the rollout

    def _sim_cache(self, tokens_per_frame: int) -> CausalKVCacheManager:
        """One cache per frame geometry, kept across requests: ``close()``/``open()``
        keep its device tensors, so graphs captured over it stay valid."""
        cache = self._sim_caches.get(tokens_per_frame)
        if cache is None:
            sim, tf = self.sim, self.transformer
            cache = CausalKVCacheManager(
                num_layers=len(tf.gen_layers),
                num_kv_heads=tf.cache_kv_heads,
                head_dim=tf.gen_layers[0].cross_attention.head_dim,
                dtype=self.dtype,
                tokens_per_page=32,
                fixed_capacity=sim.text_cache_max_len + sim.sink_frames * tokens_per_frame,
                window_tokens=sim.history_frames * tokens_per_frame,
                chunk_tokens=sim.chunk_frames * tokens_per_frame,
                causal_block_sizes=tuple(
                    k * tokens_per_frame for k in range(sim.chunk_frames, 0, -1)
                ),
            )
            self._sim_caches[tokens_per_frame] = cache
        return cache

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


__all__ = ["Cosmos3NanoSimBimanualPipeline", "Cosmos3SimSettings", "SIM_CONFIG_KEY"]
