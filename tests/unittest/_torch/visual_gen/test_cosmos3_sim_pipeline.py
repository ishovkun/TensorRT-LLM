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
import torch
from cosmos3_sim_testing import sim_settings

from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_sim import (
    Cosmos3NanoSimBimanualPipeline,
    SimEmbodiment,
    _OverlappedDecode,
    split_row_domain_ids,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm._utils import prefer_pinned

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
    s = sim_settings(CHECKPOINT_BLOCK)
    assert s.sigmas == (1.0, 0.9375, 0.8333333333333334, 0.625)
    assert (s.chunk_frames, s.history_frames, s.sink_frames) == (4, 95, 0)
    assert s.chunk_partition == (1, 4)
    assert s.text_cache_max_len == 512 and s.action_tokens_per_frame == 4
    # 961 pixel frames: 241 latent frames, one then sixty fours.
    ranges = s.chunk_ranges(241)
    assert ranges[:3] == [(0, 1), (1, 5), (5, 9)] and ranges[-1] == (237, 241)
    assert len(ranges) == 61
    # A shorter rollout ends with a partial chunk.
    assert s.chunk_ranges(7) == [(0, 1), (1, 5), (5, 7)]


def test_settings_require_the_checkpoint_keys():
    with pytest.raises(ValueError, match="Not a Cosmos3 Sim checkpoint"):
        sim_settings(None)
    minimal = {"fixed_step_sampler_config": {"t_list": [1.0, 0.5]}}
    with pytest.raises(ValueError, match="cannot resolve"):
        sim_settings(minimal)
    with pytest.raises(ValueError, match="t_list"):
        no_schedule = {
            k: v for k, v in CHECKPOINT_BLOCK.items() if k != "fixed_step_sampler_config"
        }
        sim_settings(no_schedule)
    bad = dict(CHECKPOINT_BLOCK, video_temporal_causal=False)
    with pytest.raises(ValueError, match="video_temporal_causal"):
        sim_settings(bad)
    bad = dict(CHECKPOINT_BLOCK, chunk_partition=[1, 2])
    with pytest.raises(ValueError, match="partition"):
        sim_settings(bad)
    bad = dict(CHECKPOINT_BLOCK, fixed_step_sampler_config={"t_list": [0.5, 1.0]})
    with pytest.raises(ValueError, match="decrease"):
        sim_settings(bad)


def test_settings_read_the_embodiment_table():
    s3 = sim_settings(dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_3))
    assert s3.embodiments["agibotworld"] == SimEmbodiment(domain_id=15, action_dim=29)
    assert s3.default_embodiment is None
    s5 = sim_settings(dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_5))
    assert s5.embodiments["agibotworld"] == SimEmbodiment(domain_id=15, action_dim=59)
    assert s5.embodiments["abc_yam"] == SimEmbodiment(domain_id=16, action_dim=59)
    # width missing per embodiment: the contract's width applies
    assert s5.embodiments["camera_pose"] == SimEmbodiment(domain_id=2, action_dim=59)
    assert s5.default_embodiment == "agibotworld"
    # a block without the table leaves requests to the generic resolution
    s = sim_settings(CHECKPOINT_BLOCK)
    assert s.embodiments == {} and s.resolve_embodiment("agibotworld", None, None) == (
        "agibotworld",
        None,
        None,
    )


def test_resolve_embodiment_completes_and_guards_a_request():
    s = sim_settings(dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_5))
    # by name, by id, by default
    assert s.resolve_embodiment("AgiBotWorld", None, None) == ("agibotworld", 15, 59)
    assert s.resolve_embodiment(None, 16, None) == ("abc_yam", 16, 59)
    assert s.resolve_embodiment(None, None, None) == ("agibotworld", 15, 59)
    # consistent explicit values pass, contradicting ones do not
    assert s.resolve_embodiment("agibotworld", 15, 59) == ("agibotworld", 15, 59)
    with pytest.raises(ValueError, match="raw_action_dim=29 contradicts"):
        s.resolve_embodiment("agibotworld", None, 29)
    with pytest.raises(ValueError, match="domain_id=3 contradicts"):
        s.resolve_embodiment("agibotworld", 3, None)
    # an embodiment the checkpoint does not list is left alone
    assert s.resolve_embodiment("droid_lerobot", None, 8) == ("droid_lerobot", None, 8)
    assert s.resolve_embodiment(None, 7, None) == (None, 7, None)


def test_settings_window_and_envelope():
    """A checkpoint may leave the window to the pipeline (``null``); the default
    warmup shape is the largest 480p bucket."""
    s = sim_settings(dict(CHECKPOINT_BLOCK, window_frames=None, sink_frames=0))
    assert s.history_frames is None
    assert s.largest_frame() == (640, 640)


def _envelope_pipe(window_frames=None, pool_history=None, tokens_per_frame=394):
    """A pipeline whose pool holds ``pool_history`` frames at ``tokens_per_frame``
    (394: 832x480), or no pool yet."""
    pipe = Cosmos3NanoSimBimanualPipeline.__new__(Cosmos3NanoSimBimanualPipeline)
    pipe.sim = sim_settings(dict(CHECKPOINT_BLOCK, window_frames=window_frames))
    pipe.vae_scale_factor_temporal = 4
    pipe.vae_scale_factor_spatial = 16
    pipe.transformer = SimpleNamespace(
        _pad_to_patch_size=lambda h, w: ((h + 1) // 2, (w + 1) // 2, 0, 0)
    )
    pipe._pool_tokens = (
        None
        if pool_history is None
        else pipe.pool_tokens_for_history(pool_history, tokens_per_frame)
    )
    return pipe


def test_full_history_that_does_not_fit_is_refused():
    _envelope_pipe().check_envelope(480, 832, 1801)  # pool not sized yet: nothing to refuse
    pipe = _envelope_pipe(pool_history=225)
    assert pipe._history_frames_for(394) == 225
    pipe.check_envelope(480, 832, 901)  # 226 latent frames: 225 of history
    with pytest.raises(ValueError, match="at most 901 pixel frames, got 905"):
        pipe.check_envelope(480, 832, 905)
    # A larger frame has more tokens, so the same pool holds fewer frames of it.
    assert pipe._history_frames_for(404) < 225
    with pytest.raises(ValueError, match="full history"):
        pipe.check_envelope(640, 640, 901)


def test_history_for_pool_is_exact():
    """The closed form agrees with the page formula: the most frames that fit, and one
    more does not, for pools of every size and two frame geometries."""
    pipe = _envelope_pipe()
    for tokens_per_frame in (394, 404):
        for pool_pages in range(1, 400, 7):
            pipe._pool_tokens = pool_pages * 32

            def fits(frames):
                return pipe.pool_tokens_for_history(frames, tokens_per_frame) <= pipe._pool_tokens

            h = pipe._history_frames_for(tokens_per_frame)
            if h == 0:
                assert not fits(1)
            else:
                assert fits(h) and not fits(h + 1)


def test_window_must_fit_the_pool():
    pipe = _envelope_pipe(window_frames=8, pool_history=7)
    pipe.check_envelope(480, 832, 1801)  # a window slides; the length is free
    with pytest.raises(ValueError, match="window of 7 latent frames"):
        pipe.check_envelope(640, 640, 33)


def test_measuring_pool_holds_the_warmup_and_the_window():
    pipe = _envelope_pipe()  # full history, 4-frame chunks
    assert pipe._measuring_history(13) == 8  # 4 latent frames need 3; the floor is two chunks
    assert pipe._measuring_history(901) == 225
    windowed = _envelope_pipe(window_frames=64)
    assert windowed._measuring_history(13) == 63


def test_warmup_runs_every_chunk_shape_once():
    pipe = _envelope_pipe()
    pipe.sim = sim_settings(dict(CHECKPOINT_BLOCK, chunk_size=2))
    # chunks of 1, 2 and a 1-frame tail: 4 latent frames, 13 pixel frames
    assert pipe.default_warmup_num_frames == [13]


def test_domain_id_may_be_one_per_action_row():
    assert split_row_domain_ids(None) == (None, None)
    assert split_row_domain_ids(15) == (15, None)
    assert split_row_domain_ids([15, 15, 2, 2]) == (15, (15, 15, 2, 2))
    with pytest.raises(ValueError, match="must not be empty"):
        split_row_domain_ids([])
    # every row's id is checked against the checkpoint's embodiment table
    s = sim_settings(dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_5))
    assert s.resolve_embodiment(None, 2, 59) == ("camera_pose", 2, 59)
    with pytest.raises(ValueError, match="contradicts the checkpoint"):
        s.resolve_embodiment(None, 2, 29)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="streams and pinned memory need CUDA")
def test_streamed_decode_lands_in_one_host_buffer():
    """Each chunk's frames go to pinned host memory as they are decoded; the result
    is the frames in order, on the CPU."""

    class _FrameVAE:
        """Decodes latent frame ``i`` into pixel frames filled with ``i``: one for the
        first latent frame, four for every later one."""

        def __init__(self):
            self.seen = 0

        def decode_stream_start(self):
            self.seen = 0

        def decode_stream_step(self, z):
            frames = []
            for _ in range(z.shape[2]):
                n = 1 if self.seen == 0 else 4
                frames.append(torch.full((1, 3, n, 8, 8), self.seen / 100.0, device=z.device))
                self.seen += 1
            return torch.cat(frames, dim=2)

        def decode_stream_end(self):
            pass

    pipe = SimpleNamespace(_scale_latents_for_decode=lambda z: z)
    stream = torch.cuda.Stream()
    latents = torch.zeros(1, 4, 5, 2, 2, device="cuda")
    decoder = _OverlappedDecode(pipe, _FrameVAE(), stream, num_pixel_frames=17)
    for f0, f1 in [(0, 1), (1, 3), (3, 5)]:
        decoder.submit(latents[:, :, f0:f1])
    video = decoder.finish()
    assert video.device.type == "cpu" and video.is_pinned() == prefer_pinned()
    assert video.shape == (1, 17, 8, 8, 3) and video.dtype == torch.uint8
    expected = postprocess_video_tensor(
        torch.cat(
            [torch.full((1, 3, 1 if i == 0 else 4, 8, 8), i / 100.0) for i in range(5)], dim=2
        )
    )
    torch.testing.assert_close(video, expected)
