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
"""Cosmos3 Nano Sim: checkpoint dispatch and the declared rollout settings."""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_sim import (
    SIM_CONFIG_KEY,
    Cosmos3NanoSimBimanualPipeline,
    Cosmos3SimSettings,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline

CHECKPOINT_BLOCK = {
    "chunk_size": 4,
    "window_frames": 96,
    "sink_frames": 0,
    "text_cache_max_len": 512,
    "video_temporal_causal": True,
    "fixed_step_sampler_config": {
        "sample_type": "sde",
        "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
    },
    "conditioning": {"mode": "action", "action_tokens_per_frame": 4},
}


def test_checkpoint_resolves_to_the_sim_pipeline(tmp_path):
    """The exact class name wins over the ``"Cosmos3" in class_name`` fallback that
    would otherwise load the causal checkpoint into the bidirectional pipeline."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Cosmos3NanoSimBimanualPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "Cosmos3NanoSimBimanualPipeline"
    entry = PIPELINE_REGISTRY["Cosmos3NanoSimBimanualPipeline"]
    assert entry.pipeline_cls is Cosmos3NanoSimBimanualPipeline
    assert "nvidia/Cosmos3-Nano-Sim-Bimanual" in entry.hf_ids
    # Sibling Sim checkpoints are not claimed: they still fall through.
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Cosmos3NanoSimTransferPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "Cosmos3OmniMoTPipeline"


def test_settings_read_the_checkpoint_block():
    cfg = SimpleNamespace(**{SIM_CONFIG_KEY: CHECKPOINT_BLOCK})
    s = Cosmos3SimSettings.from_pretrained_config(cfg)
    assert s.sigmas == (1.0, 0.9375, 0.8333333333333334, 0.625)
    assert (s.chunk_frames, s.window_frames, s.sink_frames) == (4, 96, 0)
    assert s.chunk_partition == (1, 4)
    assert s.history_frames == 95
    assert s.text_cache_max_len == 512 and s.action_tokens_per_frame == 4
    # 961 pixel frames: 241 latent frames, one then sixty fours.
    ranges = s.chunk_ranges(241)
    assert ranges[:3] == [(0, 1), (1, 5), (5, 9)] and ranges[-1] == (237, 241)
    assert len(ranges) == 61
    # A shorter rollout ends with a partial chunk.
    assert s.chunk_ranges(7) == [(0, 1), (1, 5), (5, 7)]


def test_settings_defaults_and_rejections():
    minimal = {"fixed_step_sampler_config": {"t_list": [1.0, 0.5]}}
    s = Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: minimal})
    assert (s.chunk_frames, s.chunk_partition, s.window_frames) == (4, (1, 4), 96)
    with pytest.raises(ValueError, match="Not a Cosmos3 Sim checkpoint"):
        Cosmos3SimSettings.from_pretrained_config(SimpleNamespace())
    with pytest.raises(ValueError, match="t_list"):
        Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: {}})
    bad = dict(CHECKPOINT_BLOCK, video_temporal_causal=False)
    with pytest.raises(ValueError, match="video_temporal_causal"):
        Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: bad})
    bad = dict(CHECKPOINT_BLOCK, chunk_partition=[1, 2])
    with pytest.raises(ValueError, match="chunk_partition"):
        Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: bad})
    bad = dict(CHECKPOINT_BLOCK, fixed_step_sampler_config={"t_list": [0.5, 1.0]})
    with pytest.raises(ValueError, match="decrease"):
        Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: bad})
