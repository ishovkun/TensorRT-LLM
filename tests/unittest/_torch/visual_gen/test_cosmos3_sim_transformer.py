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
"""Cosmos3 Nano Sim: the causal chunk forward against the bidirectional transformer.

A two-layer random-weight Cosmos3-Nano transformer. The references are the model's
own bidirectional forward and the causal forward run with a different chunking,
so no attention arithmetic is re-implemented here.
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionPipelineConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import _PreparedLatents
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_sim import (
    Cosmos3NanoSimBimanualPipeline,
    Cosmos3SimSettings,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import Cosmos3SamplingPolicy
from tensorrt_llm._torch.visual_gen.models.cosmos3.sim_packing import SIM_CONFIG_KEY
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = [
    pytest.mark.cosmos3,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
ACTION_DIM, A = 64, 4
H = W = 8  # latent cells; patch 2 -> 4 x 4 = 16 vision tokens per frame
TEXT_LEN, MAX_TEXT = 29, 48
CHUNK_FRAMES, WINDOW_FRAMES = 4, 8
NUM_TRAIN_TIMESTEPS = 1000.0
SIM_BLOCK = {
    "chunk_size": CHUNK_FRAMES,
    "window_frames": WINDOW_FRAMES,
    "sink_frames": 0,
    "text_cache_max_len": MAX_TEXT,
    "fixed_step_sampler_config": {"t_list": [1.0, 0.9375, 0.8333, 0.625]},
    "conditioning": {"action_tokens_per_frame": A},
}


def _model_config(backend: str):
    checkpoint_dir = get_checkpoint("Cosmos3-Nano")
    args = VisualGenArgs(
        model=checkpoint_dir, torch_compile_config=TorchCompileConfig(enable=False)
    )
    model_config = DiffusionPipelineConfig.from_pretrained(
        checkpoint_dir, args=args
    ).primary_model_config
    cfg = model_config.pretrained_config
    cfg.num_hidden_layers = 2
    cfg.action_gen = True
    cfg.action_dim = ACTION_DIM
    cfg.num_embodiment_domains = 4
    cfg.sound_gen = False
    setattr(cfg, SIM_CONFIG_KEY, SIM_BLOCK)  # keeps TRTLLM for the generator attention
    model_config.attention.backend = backend
    model_config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "TRTLLM" else None
    )
    return model_config


def build_model(backend: str) -> Cosmos3VFMTransformer:
    """Same random weights for every backend: the seed covers the initialisation."""
    torch.manual_seed(0)
    m = Cosmos3VFMTransformer(model_config=_model_config(backend)).to(DEVICE).eval()
    with torch.no_grad():
        for name, param in m.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                param.fill_(1.0)
            elif param.numel() > 0:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    m.post_load_weights()
    return m


@pytest.fixture(params=["CUDNN", "TRTLLM"])
def model(request):
    m = build_model(request.param)
    yield m
    del m
    torch.cuda.empty_cache()


def make_cache(model):
    tokens_per_frame = A + (H // 2) * (W // 2)
    chunk = CHUNK_FRAMES * tokens_per_frame
    return CausalKVCacheManager(
        num_layers=len(model.gen_layers),
        num_kv_heads=model.cache_kv_heads,
        head_dim=model.gen_layers[0].cross_attention.head_dim,
        dtype=DTYPE,
        tokens_per_page=32,
        fixed_capacity=MAX_TEXT,
        window_tokens=(WINDOW_FRAMES - 1) * tokens_per_frame,
        chunk_tokens=chunk,
        causal_block_sizes=tuple(k * tokens_per_frame for k in range(CHUNK_FRAMES, 0, -1)),
    )


def prompt():
    torch.manual_seed(1)
    text_ids = torch.randint(1, 1000, (1, MAX_TEXT), device=DEVICE)
    text_mask = torch.zeros(1, MAX_TEXT, device=DEVICE, dtype=torch.long)
    text_mask[:, :TEXT_LEN] = 1
    return text_ids, text_mask


def frames(num_frames, seed, channels=48):
    torch.manual_seed(seed)
    latents = torch.randn(1, channels, num_frames, H, W, device=DEVICE, dtype=DTYPE)
    actions = torch.randn(1, num_frames * A, ACTION_DIM, device=DEVICE, dtype=DTYPE)
    return latents, actions


def open_with_prompt(model, cache):
    text_ids, text_mask = prompt()
    cache.open(pin_tokens=TEXT_LEN)
    assert model.write_prompt_kv(cache, text_ids, text_mask) == TEXT_LEN
    cache.commit(TEXT_LEN)
    return text_ids, text_mask


def fraction_off(out, ref):
    """Share of elements that differ by more than 1% of the reference's scale: a
    few bf16 ulps at the largest values, so rounding differences between kernels
    or token orders stay below it."""
    scale = ref.float().abs().max().item()
    return ((out.float() - ref.float()).abs() > 0.01 * scale).float().mean().item()


def assert_same_model_output(out, ref, *, label, off_floor=0.0):
    """Two bf16 forwards of the same math in a different token order or kernel
    path agree to rounding: ``off_floor`` is the share of elements off by more than
    1% of scale measured between two kernel backends, when the caller has it. A
    wrong position, a missing key or a misplaced token moves a large share of the
    elements by the output's own scale; the negative control in the first-chunk
    test trips every one of these bounds."""
    diff = (out.float() - ref.float()).abs()
    scale = ref.float().abs().max().item()
    off = fraction_off(out, ref)
    assert diff.max().item() < 0.05 * scale, f"{label}: max error {diff.max().item()} of {scale}"
    assert off < max(0.02, 2 * off_floor), (
        f"{label}: {100 * off:.1f}% of elements differ by more than 1% of scale "
        f"(floor between backends {100 * off_floor:.1f}%)"
    )
    # bf16 keeps about 3 significant digits: the mean error is a few ulps of the
    # typical magnitude, far below the scale of the values themselves.
    rel_mean = diff.mean().item() / ref.float().abs().mean().item()
    assert rel_mean < 0.02, f"{label}: mean error {100 * rel_mean:.2f}% of the mean magnitude"


def assert_different_model_output(out, ref, *, label):
    assert fraction_off(out, ref) > 0.2, f"{label}: outputs should differ broadly"


def first_chunk(model, cache, first_frame=0):
    latents, actions = frames(CHUNK_FRAMES, seed=2, channels=model.latent_channel_size)
    t = torch.tensor([625.0], device=DEVICE)
    domain = torch.tensor([3], device=DEVICE)
    with torch.inference_mode():
        out = model.forward_causal(
            latents,
            t,
            kv_cache=cache,
            first_frame=first_frame,
            text_len=TEXT_LEN,
            action_latents=actions,
            action_domain_ids=domain,
            fps=24.0,
        )
    torch.cuda.synchronize()
    return out, (latents, actions, t, domain)


def test_first_chunk_matches_the_bidirectional_forward():
    """A first chunk sees the prompt and itself, which is what the bidirectional
    forward sees over the same frames when its action rows carry the same
    positions. One thing the bidirectional model cannot express: frame 0's rows
    are the null action, placed at frame 0's own time with their values dropped
    from the cache, so the action output is not compared; the video output is.
    cuDNN serves both paths (the bidirectional one cannot run on trtllm-gen); a
    chunk placed one frame later is the negative control. A cached frame would
    not be a valid comparison at all: bidirectional keys of a conditioned frame
    depend on the frames after it, causal ones do not."""
    model = build_model("CUDNN")
    cache = make_cache(model)
    try:
        text_ids, text_mask = open_with_prompt(model, cache)
        out, (latents, actions, t, domain) = first_chunk(model, cache)
        shifted, _ = first_chunk(model, cache, first_frame=1)
        with torch.inference_mode():
            model.reset_cache()
            ref = model(
                hidden_states=latents,
                timestep=t / NUM_TRAIN_TIMESTEPS,
                raw_timestep=t,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=(CHUNK_FRAMES, H, W),
                fps=24.0,
                action_latents=actions,
                action_domain_ids=domain,
                action_noisy_mask=torch.zeros(1, actions.shape[1], 1, device=DEVICE),
                action_start_frame_offset=-(A - 1),  # rows of frames 0.., as the Sim packing
            )
            model.reset_cache()
        torch.cuda.synchronize()
        assert_same_model_output(out.video, ref.video, label="video")
        assert_different_model_output(shifted.video, ref.video, label="chunk one frame later")
        assert cache.past_tokens == TEXT_LEN, "a denoising forward commits nothing"
    finally:
        cache.shutdown()
        del model
        torch.cuda.empty_cache()


def test_trtllm_gen_matches_cudnn_on_the_causal_path():
    """Same weights, same chunk, both cache backends."""
    outputs = []
    for backend in ("CUDNN", "TRTLLM"):
        model = build_model(backend)
        cache = make_cache(model)
        try:
            open_with_prompt(model, cache)
            out, _ = first_chunk(model, cache)
            outputs.append((out.video.clone(), out.action.clone()))
        finally:
            cache.shutdown()
            del model
            torch.cuda.empty_cache()
    assert_same_model_output(outputs[1][0], outputs[0][0], label="video")
    assert_same_model_output(outputs[1][1], outputs[0][1], label="action")


def two_pass_rollout(model, chunking, first, second):
    """Clean passes over ``first`` (5 frames) cut as ``chunking``, then one denoising
    forward of ``second``; returns its velocities."""
    domain = torch.tensor([1], device=DEVICE)
    tokens_per_frame = A + (H // 2) * (W // 2)
    cache = make_cache(model)
    try:
        open_with_prompt(model, cache)
        with torch.inference_mode():
            for f0, f1 in chunking:
                model.forward_causal(
                    first[0][:, :, f0:f1],
                    torch.zeros(1, device=DEVICE),
                    kv_cache=cache,
                    first_frame=f0,
                    text_len=TEXT_LEN,
                    action_latents=first[1][:, f0 * A : f1 * A],
                    action_domain_ids=domain,
                    clean_pass=True,
                    fps=24.0,
                )
                cache.commit((f1 - f0) * tokens_per_frame)
            out = model.forward_causal(
                second[0],
                torch.tensor([937.5], device=DEVICE),
                kv_cache=cache,
                first_frame=5,
                text_len=TEXT_LEN,
                action_latents=second[1],
                action_domain_ids=domain,
                fps=24.0,
            )
        torch.cuda.synchronize()
        assert cache.history_tokens == 5 * tokens_per_frame
        return out.video.clone(), out.action.clone()
    finally:
        cache.shutdown()


def test_rollout_is_independent_of_the_chunking(model):
    """Frames 1..4 as one chunk or as [1,2] then [3,4]: either way every frame's
    clean K/V saw the prompt, the frames before it and itself, so the next chunk
    sees the same history and predicts the same velocity. Covers the clean pass's
    per-frame causal blocks, commit, history reads and a chunk shorter than four
    frames. Frame 0 is alone in both, as the checkpoint's partition has it: its
    null-action values are zeroed once it is history, so a chunk holding it
    together with later frames would not be the same rollout.

    Two passes compound bf16 rounding, so the tolerance's noise floor is measured:
    the same rollout on the other kernel backend."""
    channels = model.latent_channel_size
    first, second = frames(5, 3, channels), frames(CHUNK_FRAMES, 4, channels)
    one = two_pass_rollout(model, [(0, 1), (1, 5)], first, second)
    split = two_pass_rollout(model, [(0, 1), (1, 3), (3, 5)], first, second)
    other_backend = (
        "TRTLLM"
        if model.gen_layers[0].cross_attention.attn.__class__.__name__.startswith("CuDNN")
        else "CUDNN"
    )
    other = build_model(other_backend)
    try:
        floor = two_pass_rollout(other, [(0, 1), (1, 5)], first, second)
    finally:
        del other
        torch.cuda.empty_cache()
    for i, label in enumerate(("video", "action")):
        off_floor = fraction_off(floor[i], one[i])
        assert_same_model_output(split[i], one[i], label=label, off_floor=off_floor)


def test_frame_zero_action_values_are_zeroed_after_its_clean_pass(model):
    """The null-action rows of frame 0 keep their keys and lose their values in the
    cache, as the reference stores them."""
    cache = make_cache(model)
    try:
        open_with_prompt(model, cache)
        latents, actions = frames(1, seed=5, channels=model.latent_channel_size)
        with torch.inference_mode():
            model.forward_causal(
                latents,
                torch.zeros(1, device=DEVICE),
                kv_cache=cache,
                first_frame=0,
                text_len=TEXT_LEN,
                action_latents=actions,
                clean_pass=True,
                fps=24.0,
            )
        torch.cuda.synchronize()
        for layer in range(len(model.gen_layers)):
            buf = cache.kv_buffer(layer)
            table = cache.table.long()
            pos = torch.arange(TEXT_LEN, TEXT_LEN + A, device=DEVICE)
            page, slot = table[pos // cache.tokens_per_page], pos % cache.tokens_per_page
            assert buf[page, 1, :, slot].abs().max().item() == 0.0, f"layer {layer}: V not zeroed"
            assert buf[page, 0, :, slot].abs().max().item() > 0.0, f"layer {layer}: K lost"
    finally:
        cache.shutdown()


def test_rollout_past_the_window_stays_bounded(model):
    cache = make_cache(model)
    tokens_per_frame = A + (H // 2) * (W // 2)
    try:
        open_with_prompt(model, cache)
        with torch.inference_mode():
            f0 = 0
            for chunk_frames in (1, 4, 4, 4, 4):  # 17 frames through an 8-frame window
                latents, actions = frames(chunk_frames, 10 + f0, model.latent_channel_size)
                for step, sigma in enumerate((1.0, 0.9375, 0.8333, 0.625)):
                    out = model.forward_causal(
                        latents,
                        torch.tensor([sigma * NUM_TRAIN_TIMESTEPS], device=DEVICE),
                        kv_cache=cache,
                        first_frame=f0,
                        text_len=TEXT_LEN,
                        action_latents=actions,
                        fps=24.0,
                    )
                    assert torch.isfinite(out.video).all(), f"frame {f0} step {step}"
                model.forward_causal(
                    latents,
                    torch.zeros(1, device=DEVICE),
                    kv_cache=cache,
                    first_frame=f0,
                    text_len=TEXT_LEN,
                    action_latents=actions,
                    clean_pass=True,
                    fps=24.0,
                )
                cache.commit(chunk_frames * tokens_per_frame)
                f0 += chunk_frames
                assert cache.history_tokens <= (WINDOW_FRAMES - 1) * tokens_per_frame + 31
        assert cache.fixed_tokens == TEXT_LEN
    finally:
        cache.shutdown()


# ----------------------------------------------------------------------- the chunk loop


def make_sim_pipeline(model, cuda_graphs: bool = False):
    """The Sim pipeline's rollout over the tiny transformer, without loading a
    checkpoint: only the attributes the rollout touches are set."""
    pipe = Cosmos3NanoSimBimanualPipeline.__new__(Cosmos3NanoSimBimanualPipeline)
    torch.nn.Module.__init__(pipe)  # BasePipeline is a Module; skip its loading constructor
    pipe.transformer = model
    pipe._cuda_graph_runners = {}
    pipe._decode_stream = None
    pipe._streamed_video = None
    pipe.vae = None  # no decoder in these tests: the rollout decodes nothing
    pipe._parallel_vae_enabled = False
    pipe._rank = 0
    pipe.sim = Cosmos3SimSettings.from_pretrained_config({SIM_CONFIG_KEY: SIM_BLOCK})
    pipe._history_frames = pipe.sim.window_frames - 1  # the tests pin a small window
    pipe._sim_cache_obj = None
    pipe._sim_cache_key = None
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "stochastic_sampling": True,
            "fixed_step_sampler_config": {
                "sample_type": "sde",
                "t_list": list(SIM_BLOCK["fixed_step_sampler_config"]["t_list"]),
            },
        }
    )
    pipe.sampling = Cosmos3SamplingPolicy.from_scheduler(pipe.scheduler)
    pipe.offloader = SimpleNamespace(
        context_if_requested=lambda name: nullcontext(), stages=lambda: []
    )
    pipe._device = DEVICE
    pipe.pipeline_config = SimpleNamespace(
        torch_dtype=DTYPE,
        kv_cache=SimpleNamespace(free_gpu_memory_fraction=0.9),
        cuda_graph=SimpleNamespace(enable=cuda_graphs),
    )
    pipe.vae_scale_factor_temporal = 4
    pipe.vae_scale_factor_spatial = 16
    pipe._scheduler_cache = {}
    pipe._release_scheduler_solver_state = lambda: None
    return pipe


def rollout(
    model, num_latent_frames, seed, conditioned_first_frame=False, cuda_graphs: bool = False
):
    pipe = make_sim_pipeline(model, cuda_graphs=cuda_graphs)
    pipe._setup_cuda_graphs()
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
    prepared = _PreparedLatents(
        video_latents=latents.clone(),
        action_latents=torch.randn(
            1,
            (num_latent_frames - 1) * A,
            ACTION_DIM,
            device=DEVICE,
            dtype=DTYPE,
            generator=generator,
        ),
        action_domain_id=2,
    )
    if conditioned_first_frame:
        mask = torch.ones(1, 1, num_latent_frames, 1, 1, device=DEVICE, dtype=DTYPE)
        mask[:, :, 0] = 0
        prepared.velocity_mask = mask
        prepared.condition_latents = latents.clone()
    request = SimpleNamespace(frame_rate=24.0, action_fps=24.0, enable_audio=False)
    text_ids, text_mask = prompt()
    with torch.inference_mode():
        out, audio, actions, do_audio = pipe._denoise_request(
            request,
            prepared,
            cond_ids=text_ids,
            cond_mask=text_mask,
            uncond_ids=text_ids,
            uncond_mask=text_mask,
            generator=generator,
            timer=CudaPhaseTimer(),
        )
    torch.cuda.synchronize()
    assert audio is None and do_audio is False and actions is prepared.action_latents
    cache = pipe._sim_cache_obj
    assert pipe._sim_cache_key == A + (H // 2) * (W // 2)
    return out, latents, cache


def test_chunk_loop_runs_the_rollout(model):
    """Nine latent frames: chunks [0], [1..4], [5..8]. Every frame changes from its
    noise, the run is seeded, and the cache ends closed with the clean frames
    0..4 as history: the last chunk gets no clean pass."""
    out, noise, cache = rollout(model, 9, seed=11)
    assert out.shape == noise.shape and torch.isfinite(out).all()
    assert not torch.equal(out, noise)
    assert not cache.is_open
    tokens_per_frame = A + (H // 2) * (W // 2)
    assert (cache.fixed_tokens, cache.history_tokens) == (TEXT_LEN, 5 * tokens_per_frame)
    again, _, _ = rollout(model, 9, seed=11)
    torch.testing.assert_close(again, out, rtol=0, atol=0)
    other, _, _ = rollout(model, 9, seed=12)
    assert not torch.equal(other, out)


def test_conditioned_first_frame_is_kept_and_cached(model):
    """Forward dynamics: frame 0 is given. It is not denoised, its clean pass still
    feeds the cache, and the generated frames differ from the unconditioned run."""
    out, source, cache = rollout(model, 5, seed=13, conditioned_first_frame=True)
    torch.testing.assert_close(out[:, :, 0], source[:, :, 0], rtol=0, atol=0)
    assert torch.isfinite(out).all()
    tokens_per_frame = A + (H // 2) * (W // 2)
    assert cache.history_tokens == 1 * tokens_per_frame


def test_chunk_loop_under_cuda_graphs_matches_eager(model):
    """The same seeded rollout with the chunk forward captured and replayed: the
    graphs see the cache's lengths through the refreshed metadata, so every chunk
    reads the history it should. Thirteen latent frames: four chunk shapes in
    play, commits and page rotations between replays."""
    eager, _, _ = rollout(model, 13, seed=21)
    graphed, _, cache = rollout(model, 13, seed=21, cuda_graphs=True)
    torch.cuda.synchronize()
    assert_same_model_output(graphed, eager, label="rollout under CUDA graphs")
    assert cache.is_open is False


def test_per_row_domain_ids_match_one_id_when_they_agree(model):
    """A chunk with one id per action row, all the same, equals the one-id path."""
    cache = make_cache(model)
    try:
        open_with_prompt(model, cache)
        latents, actions = frames(CHUNK_FRAMES, seed=17, channels=model.latent_channel_size)
        one = torch.tensor([3], device=DEVICE)
        rows = torch.full((actions.shape[1],), 3, device=DEVICE)
        outs = []
        with torch.inference_mode():
            for ids in (one, rows):
                model.reset_cache()
                out = model.forward_causal(
                    latents,
                    torch.full((1,), 500.0, device=DEVICE),
                    kv_cache=cache,
                    first_frame=0,
                    text_len=TEXT_LEN,
                    action_latents=actions,
                    action_domain_ids=ids,
                    fps=24.0,
                )
                outs.append((out.video.clone(), out.action.clone()))
        torch.cuda.synchronize()
        # per-row ids go through a batched matmul (one small weight per row), the
        # one-id path through one matmul: same math, different summation order
        assert_same_model_output(outs[1][0], outs[0][0], label="video, per-row domain ids")
        assert_same_model_output(outs[1][1], outs[0][1], label="action, per-row domain ids")
    finally:
        cache.shutdown()


def test_sim_action_rows_carry_per_row_domains(model):
    """Frame f's rows get the action rows that lead into it and their embodiment;
    frame 0's null rows take the first real row's embodiment."""
    pipe = make_sim_pipeline(model)
    T, given_rows = 5, (5 - 1) * A
    given = torch.randn(1, given_rows, model.action_dim, device=DEVICE, dtype=DTYPE)
    ids = tuple([3] * (2 * A) + [1] * (2 * A))  # frames 1-2 embodiment 3, frames 3-4 embodiment 1
    prepared = SimpleNamespace(
        action_latents=given, action_state_rows=0, action_domain_id=3, action_row_domain_ids=ids
    )
    rows, domains = pipe._sim_action_rows(None, prepared, T, A, None)
    assert rows.shape == (1, T * A, model.action_dim) and domains.shape == (T * A,)
    assert rows[:, :A].abs().max().item() == 0.0
    torch.testing.assert_close(rows[:, A:], given)
    assert domains[:A].tolist() == [3] * A
    assert domains[A:].tolist() == list(ids)
    assert model.domain_ids_validated
    # one id for the whole request fills every row
    prepared.action_row_domain_ids = None
    _, domains = pipe._sim_action_rows(None, prepared, T, A, None)
    assert domains.tolist() == [3] * (T * A)
