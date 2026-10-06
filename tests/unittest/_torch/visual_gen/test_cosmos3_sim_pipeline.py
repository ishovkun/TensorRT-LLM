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
    SimEmbodiment,
    split_row_domain_ids,
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

# conditioning blocks as the two export schemas write them
CONDITIONING_SCHEMA_3 = {
    "mode": "action",
    "action_tokens_per_frame": 4,
    "embodiments": {
        "agibotworld": {"domain_id": 15, "raw_action_dim": 29},
        "camera_pose": {"domain_id": 2, "raw_action_dim": 9},
    },
}
CONDITIONING_SCHEMA_5 = {
    "mode": "action",
    "action_tokens_per_frame": 4,
    "default_embodiment": "agibotworld",
    "input_contract": {"action_dim": 59},
    "embodiments": {
        "agibotworld": {"domain_id": 15, "input_action_dim": 59},
        "abc_yam": {"domain_id": 16, "input_action_dim": 59},
        "camera_pose": {"domain_id": 2},
    },
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


def test_settings_read_the_embodiment_table():
    s3 = Cosmos3SimSettings.from_pretrained_config(
        {SIM_CONFIG_KEY: dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_3)}
    )
    assert s3.embodiments["agibotworld"] == SimEmbodiment(domain_id=15, action_dim=29)
    assert s3.default_embodiment is None
    s5 = Cosmos3SimSettings.from_pretrained_config(
        {SIM_CONFIG_KEY: dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_5)}
    )
    assert s5.embodiments["agibotworld"] == SimEmbodiment(domain_id=15, action_dim=59)
    assert s5.embodiments["abc_yam"] == SimEmbodiment(domain_id=16, action_dim=59)
    # width missing per embodiment: the contract's width applies
    assert s5.embodiments["camera_pose"] == SimEmbodiment(domain_id=2, action_dim=59)
    assert s5.default_embodiment == "agibotworld"
    # a block without the table leaves requests to the generic resolution
    s = Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: CHECKPOINT_BLOCK})
    assert s.embodiments == {} and s.action_contract("agibotworld", None, None) == (
        "agibotworld",
        None,
        None,
    )


def test_action_contract_completes_and_guards_a_request():
    s = Cosmos3SimSettings.from_pretrained_config(
        {SIM_CONFIG_KEY: dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_5)}
    )
    # by name, by id, by default
    assert s.action_contract("AgiBotWorld", None, None) == ("agibotworld", 15, 59)
    assert s.action_contract(None, 16, None) == ("abc_yam", 16, 59)
    assert s.action_contract(None, None, None) == ("agibotworld", 15, 59)
    # consistent explicit values pass, contradicting ones do not
    assert s.action_contract("agibotworld", 15, 59) == ("agibotworld", 15, 59)
    with pytest.raises(ValueError, match="raw_action_dim=29 contradicts"):
        s.action_contract("agibotworld", None, 29)
    with pytest.raises(ValueError, match="domain_id=3 contradicts"):
        s.action_contract("agibotworld", 3, None)
    # an embodiment the checkpoint does not list is left alone
    assert s.action_contract("droid_lerobot", None, 8) == ("droid_lerobot", None, 8)
    assert s.action_contract(None, 7, None) == (None, 7, None)


def test_settings_window_and_envelope():
    """A checkpoint may leave the window to the pipeline (``null``); the serving
    envelope defaults to the model card (480p bucket, 901 frames) unless declared."""
    s = Cosmos3SimSettings.from_pretrained_config(
        {SIM_CONFIG_KEY: dict(CHECKPOINT_BLOCK, window_frames=None, sink_frames=0)}
    )
    assert s.window_frames is None and s.history_frames is None
    assert (s.max_pixels, s.max_pixel_frames) == (640 * 640, 901)
    assert s.max_latent_frames(4) == 226
    assert s.largest_frame() == (640, 640)
    declared = Cosmos3SimSettings.from_pretrained_config(
        {SIM_CONFIG_KEY: dict(CHECKPOINT_BLOCK, max_pixels=832 * 480, max_num_frames=121)}
    )
    assert declared.largest_frame() == (480, 832) and declared.max_latent_frames(4) == 31
    with pytest.raises(ValueError, match="no 480p bucket"):
        Cosmos3SimSettings.from_pretrained_config(
            {SIM_CONFIG_KEY: dict(CHECKPOINT_BLOCK, max_pixels=100)}
        ).largest_frame()


def test_envelope_check_refuses_larger_requests():
    pipe = Cosmos3NanoSimBimanualPipeline.__new__(Cosmos3NanoSimBimanualPipeline)
    pipe.sim = Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: CHECKPOINT_BLOCK})
    pipe.check_envelope(480, 832, 901)
    pipe.check_envelope(640, 640, 33)
    with pytest.raises(ValueError, match="at most 409600 pixels"):
        pipe.check_envelope(720, 1280, 33)
    with pytest.raises(ValueError, match="901 frames"):
        pipe.check_envelope(480, 832, 905)


def test_domain_id_may_be_one_per_action_row():
    assert split_row_domain_ids(None) == (None, None)
    assert split_row_domain_ids(15) == (15, None)
    assert split_row_domain_ids([15, 15, 2, 2]) == (15, (15, 15, 2, 2))
    with pytest.raises(ValueError, match="must not be empty"):
        split_row_domain_ids([])
    # every row's id is checked against the checkpoint's embodiment table
    s = Cosmos3SimSettings.from_pretrained_config(
        {SIM_CONFIG_KEY: dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_5)}
    )
    assert s.action_contract(None, 2, 59) == ("camera_pose", 2, 59)
    with pytest.raises(ValueError, match="contradicts the checkpoint"):
        s.action_contract(None, 2, 29)
