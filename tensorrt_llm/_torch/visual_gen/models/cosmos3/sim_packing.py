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
"""Cosmos3 Nano Sim: frame-major token packing and absolute rotary positions.

One latent frame is one block of ``A`` action rows followed by ``P`` vision tokens
(``A = 4``, ``P = ceil(H/2) * ceil(W/2)``; 394 at 480p). A chunk of ``n`` frames is
``n`` such blocks in time order, so every frame's tokens are contiguous and the
chunk can be cut into per-frame causal blocks and cached frame by frame. The
bidirectional model packs ``[all vision | all action]`` instead.

Rotary time positions count from the start of the rollout: a chunk whose first
latent frame is ``f0`` takes the positions frames ``f0 ..`` would have in one long
clip, so eviction never renumbers anything. The prompt's position space is kept
even though its tokens are no longer in the sequence after frame 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SIM_CONFIG_KEY = "cosmos3_nano_sim_bimanual"
"""The block of ``transformer/config.json`` that marks a Sim checkpoint."""


@dataclass(frozen=True)
class FramePacking:
    """Token layout of a chunk of ``num_frames`` latent frames."""

    num_frames: int
    action_tokens: int
    vision_tokens: int

    @property
    def tokens_per_frame(self) -> int:
        return self.action_tokens + self.vision_tokens

    @property
    def num_tokens(self) -> int:
        return self.num_frames * self.tokens_per_frame

    def interleave(self, vision: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``vision [B, n*P, D]`` and ``action [B, n*A, D]`` (both frame-major) to the
        packed ``[B, n*(A+P), D]``. One concatenation; the bidirectional model pays
        the same to append its action block."""
        n, a, p = self.num_frames, self.action_tokens, self.vision_tokens
        if vision.shape[1] != n * p or action.shape[1] != n * a:
            raise ValueError(
                f"packing {n} frames of {a} action + {p} vision tokens, got vision "
                f"{tuple(vision.shape)} and action {tuple(action.shape)}"
            )
        batch, _, dim = vision.shape
        packed = torch.cat([action.view(batch, n, a, dim), vision.view(batch, n, p, dim)], dim=2)
        return packed.view(batch, n * (a + p), dim)

    def split(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The inverse: ``(vision [B, n*P, D], action [B, n*A, D])``. Views of
        ``packed`` where the strides allow, which the row-major pack does not, so
        each half is one copy on the way out."""
        n, a, p = self.num_frames, self.action_tokens, self.vision_tokens
        batch, _, dim = packed.shape
        blocks = packed.view(batch, n, a + p, dim)
        return blocks[:, :, a:].reshape(batch, n * p, dim), blocks[:, :, :a].reshape(
            batch, n * a, dim
        )

    def frame_slices(self, frame: int) -> tuple[slice, slice]:
        """Token ranges of ``frame``'s action rows and vision tokens within the chunk."""
        start = frame * self.tokens_per_frame
        return (
            slice(start, start + self.action_tokens),
            slice(start + self.action_tokens, start + self.tokens_per_frame),
        )


def sim_position_ids(
    packing: FramePacking,
    *,
    first_frame: int,
    grid_h: int,
    grid_w: int,
    text_len: int,
    temporal_modality_margin: int,
    fps: float,
    action_fps: float,
    base_fps: float,
    temporal_compression_factor: int,
    enable_fps_modulation: bool,
) -> torch.Tensor:
    """3-D rotary position ids ``[3, packing.num_tokens]`` of a chunk starting at
    latent frame ``first_frame``, in the packed token order.

    Vision tokens of absolute frame ``f`` sit at time ``text_len + margin +
    f * base_fps / fps``; frame ``f``'s four action rows sit a quarter frame apart
    just before it, ending at its time, like the bidirectional model's actions
    for frames ``1 ..`` with ``action_start_frame_offset = 1``. Frame 0's rows are
    the null action and all sit at frame 0's time.
    """
    from .transformer_cosmos3 import (  # the transformer imports this module
        compute_mrope_position_ids_action,
        compute_mrope_position_ids_text,
        compute_mrope_position_ids_vision,
    )

    if grid_h * grid_w != packing.vision_tokens:
        raise ValueError(
            f"grid {grid_h}x{grid_w} has {grid_h * grid_w} tokens, packing expects "
            f"{packing.vision_tokens}"
        )
    _, text_end = compute_mrope_position_ids_text(text_len, temporal_offset=0)
    media_offset = text_end + temporal_modality_margin
    n, a = packing.num_frames, packing.action_tokens
    vision, _ = compute_mrope_position_ids_vision(
        n,
        grid_h,
        grid_w,
        temporal_offset=media_offset,
        fps=fps,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=enable_fps_modulation,
        start_frame_offset=first_frame,
    )
    # Action rows run at frame rate: row j of frame f is source frame
    # f*tcf - (tcf - 1 - j), i.e. the tcf rows that lead into f.
    action, _ = compute_mrope_position_ids_action(
        n * a,
        temporal_offset=media_offset,
        action_fps=action_fps,
        base_fps=base_fps,
        base_temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=enable_fps_modulation,
        start_frame_offset=first_frame * temporal_compression_factor - (a - 1),
    )
    dtype = torch.promote_types(vision.dtype, action.dtype)
    action = action.to(dtype).view(3, n, a)
    vision = vision.to(dtype).view(3, n, packing.vision_tokens)
    if first_frame == 0:
        # Frame 0 has no action before it: its rows are the null action, which the
        # reference places at frame 0's own time rather than leading into it.
        action[0, 0] = vision[0, 0, 0]
    per_frame = torch.cat([action, vision], dim=2)
    return per_frame.reshape(3, packing.num_tokens)
