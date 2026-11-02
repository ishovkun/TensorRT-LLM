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
"""The Cosmos3 Sim session adapter against the Sim rollout it is built from: the
same two-layer random-weight transformer as ``test_cosmos3_sim_transformer``, a
stub streaming decoder, and the executor's delivery slots. A session advanced one
chunk at a time must produce the frames the one-request rollout produces."""

from types import SimpleNamespace

import pytest
import torch
from test_cosmos3_sim_transformer import (  # noqa: F401  (the ``model`` fixture)
    ACTION_DIM,
    DEVICE,
    DTYPE,
    A,
    H,
    W,
    make_sim_pipeline,
    model,
    prompt,
)

from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import _PreparedLatents
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer
from tensorrt_llm._torch.visual_gen.session import SessionCapabilities, StreamSpec
from tensorrt_llm._torch.visual_gen.slot_pool import SlotPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rollout needs CUDA")

UP = 16  # the stub decoder's spatial upsampling; the VAE scale factor the pipeline assumes


class StubStreamingVAE:
    """A deterministic stand-in for the causal Wan VAE's streaming decode: latent
    frame ``i`` becomes one pixel frame for the first latent frame and four after,
    each an upsampled, squashed copy of the latent's first three channels. The
    frames depend only on the latents, so two paths decoding the same latents
    produce the same video."""

    dtype = DTYPE
    config = SimpleNamespace(scaling_factor=1.0)

    def __init__(self):
        self.first = True

    def decode_stream_start(self):
        self.first = True

    def decode_stream_step(self, z):
        frames = []
        for i in range(z.shape[2]):
            n = 1 if self.first else 4
            self.first = False
            frame = torch.tanh(z[:, :3, i].float())  # [1, 3, h, w] in [-1, 1]
            frame = frame.repeat_interleave(UP, dim=-2).repeat_interleave(UP, dim=-1)
            frames.append(frame.unsqueeze(2).expand(-1, -1, n, -1, -1))
        return torch.cat(frames, dim=2)

    def decode_stream_end(self):
        pass


def _request():
    return SimpleNamespace(frame_rate=24.0, action_fps=24.0, enable_audio=False)


def _prepared(model, num_latent_frames, seed):  # noqa: F811
    """Frame 0 given (a conditioned first frame), the rest generated; one action row
    block for frames 1.. in the model's action width."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latents = torch.randn(
        1,
        model.latent_channel_size,
        num_latent_frames,
        H,
        W,
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    rows = torch.randn(
        1, (num_latent_frames - 1) * A, ACTION_DIM, device=DEVICE, dtype=DTYPE, generator=generator
    )
    mask = torch.ones(1, 1, num_latent_frames, 1, 1, device=DEVICE, dtype=DTYPE)
    mask[:, :, 0] = 0
    return _PreparedLatents(
        video_latents=latents.clone(),
        action_latents=rows,
        action_domain_id=2,
        velocity_mask=mask,
        condition_latents=latents.clone(),
    )


def _offline_video(model, prepared, seed):  # noqa: F811
    pipe = make_sim_pipeline(model)
    pipe.vae = StubStreamingVAE()
    pipe._setup_cuda_graphs()
    text_ids, text_mask = prompt()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.inference_mode():
        latents, _, _, _ = pipe._denoise_request(
            _request(),
            prepared,
            cond_ids=text_ids,
            cond_mask=text_mask,
            uncond_ids=text_ids,
            uncond_mask=text_mask,
            generator=generator,
            timer=CudaPhaseTimer(),
        )
        video = pipe._decode_latents(latents)
    torch.cuda.synchronize()
    return video


def _session_video(model, prepared, seed, num_latent_frames):  # noqa: F811
    pipe = make_sim_pipeline(model)
    pipe.vae = StubStreamingVAE()
    pipe._setup_cuda_graphs()
    text_ids, text_mask = prompt()
    caps = SessionCapabilities(
        inputs=(),
        streams=(
            StreamSpec(
                "video", "video", torch.uint8, (1, 4 * pipe.sim.chunk_frames, H * UP, W * UP, 3)
            ),
        ),
        initial_result=True,
    )
    pool = SlotPool(caps, num_slots=2, device=torch.device(DEVICE), local=True)
    frames = []
    with torch.inference_mode():
        ro = pipe._rollout_begin(
            _request(), prepared, cond_ids=text_ids, cond_mask=text_mask, seed=seed
        )
        slot = pool.reserve()
        session, outcome = pipe._session_start(
            ro, prepared, slot, domain_id=2, use_guardrails=False, remaining=None
        )
        if outcome.event is not None:
            outcome.event.synchronize()
        shape, dtype = slot.written()["video"]
        frames.append(slot.buffer("video", shape, dtype).clone())
        pool.release(slot.slot_id)
        rows = prepared.action_latents
        f = 1
        while f < num_latent_frames:
            f0, f1 = session.peek_range()
            assert f0 == f
            chunk_rows = rows[0, (f0 - 1) * A : (f1 - 1) * A]
            slot = pool.reserve()
            outcome = pipe.advance(session, {"actions": chunk_rows}, slot)
            if outcome.event is not None:
                outcome.event.synchronize()
            shape, dtype = slot.written()["video"]
            assert shape[1] == 4 * (f1 - f0)
            frames.append(slot.buffer("video", shape, dtype).clone())
            pool.release(slot.slot_id)
            f = f1
        pipe.close_session(session)
    pool.close()
    assert not pipe._kv_cache.is_open
    return torch.cat(frames, dim=1)


def test_session_advances_produce_the_rollouts_frames(model):  # noqa: F811
    """Nine latent frames: the one-request rollout's video, decoded chunk by chunk,
    equals the frames a session delivers over its initial result and two advances."""
    prepared = _prepared(model, 9, seed=31)
    offline = _offline_video(model, _prepared(model, 9, seed=31), seed=31)
    streamed = _session_video(model, prepared, seed=31, num_latent_frames=9)
    assert offline.shape == streamed.shape == (1, 33, H * UP, W * UP, 3)
    torch.testing.assert_close(streamed.cpu(), offline.cpu(), rtol=0, atol=0)


def test_advance_rejects_the_wrong_row_count(model):  # noqa: F811
    prepared = _prepared(model, 5, seed=5)
    pipe = make_sim_pipeline(model)
    pipe.vae = StubStreamingVAE()
    text_ids, text_mask = prompt()
    with torch.inference_mode():
        ro = pipe._rollout_begin(
            _request(), prepared, cond_ids=text_ids, cond_mask=text_mask, seed=5
        )
        session, _ = pipe._session_start(
            ro, prepared, None, domain_id=2, use_guardrails=False, remaining=None
        )
        with pytest.raises(ValueError, match="action rows"):
            pipe.advance(session, {"actions": prepared.action_latents[0, :A]}, None)
        pipe.close_session(session)


def test_session_rows_per_advance_follow_the_partition(model):  # noqa: F811
    pipe = make_sim_pipeline(model)
    sizes = pipe._action_rows_per_advance()
    assert sizes == (pipe.sim.chunk_frames * A,)
    ranges = pipe._session_ranges()
    assert [next(ranges) for _ in range(3)] == [(0, 1), (1, 5), (5, 9)]
