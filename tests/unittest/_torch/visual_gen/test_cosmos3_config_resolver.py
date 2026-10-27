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
"""The Cosmos3 precedence table, on synthetic files and on the published checkpoints."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.config_resolver import (
    CHECKPOINT,
    CONFIG,
    MODEL_INDEX,
    SCHEDULER,
    TRANSFORMER,
    VAE,
    Conflict,
    Cosmos3ConfigResolver,
    Default,
    Key,
    Rule,
    Serving,
    architecture_defaults,
    fill_pretrained_config,
)

FIXTURE = Path(__file__).parent / "test_data" / "cosmos3_checkpoint_configs.json"
SIM = "cosmos3_nano_sim_bimanual"
T_LIST = [1.0, 0.9375, 0.8333333333333334, 0.625]


def _write(checkpoint_dir: Path, files: dict) -> Path:
    for name, doc in files.items():
        path = checkpoint_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
    return checkpoint_dir


def _resolver(tmp_path: Path, files: dict, serving=None) -> Cosmos3ConfigResolver:
    _write(tmp_path, files)
    return Cosmos3ConfigResolver(str(tmp_path), files.get(TRANSFORMER, {}), serving)


class TestRule:
    def test_first_present_source_wins(self, tmp_path):
        r = _resolver(tmp_path, {CONFIG: {"a": 1}, MODEL_INDEX: {"a": 1}})
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a"), Default(7, "x")))
        assert r.resolve_one("a", rule).value == 1
        assert r.resolve_one("a", rule).source == "config.json:a"

    def test_falls_through_missing_and_null_sources(self, tmp_path):
        r = _resolver(tmp_path, {CONFIG: {"a": None}, MODEL_INDEX: {"a": 2}})
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a"), Key(CHECKPOINT, "a")))
        assert r.resolve_one("a", rule).value == 2

    def test_nullable_keeps_an_explicit_null(self, tmp_path):
        r = _resolver(tmp_path, {CONFIG: {"a": None}, MODEL_INDEX: {"a": 2}})
        rule = Rule(
            (Key(CONFIG, "a"), Key(MODEL_INDEX, "a")), conflict=Conflict.IGNORE, nullable=True
        )
        assert r.resolve_one("a", rule).value is None

    def test_default_only_when_nothing_is_declared(self, tmp_path):
        r = _resolver(tmp_path, {})
        rule = Rule((Key(CONFIG, "a"), Default(7, "x")))
        resolved = r.resolve_one("a", rule)
        assert (resolved.value, resolved.source) == (7, "default (x)")

    def test_unresolvable_required_setting_is_an_error(self, tmp_path):
        r = _resolver(tmp_path, {})
        with pytest.raises(ValueError, match="cannot resolve a"):
            r.resolve_one("a", Rule((Key(CONFIG, "a"),)))
        assert r.resolve_one("a", Rule((Key(CONFIG, "a"),), required=False)) is None

    def test_conflict_policies(self, tmp_path):
        r = _resolver(tmp_path, {CONFIG: {"a": 1}, MODEL_INDEX: {"a": 2}})
        sources = (Key(CONFIG, "a"), Key(MODEL_INDEX, "a"))
        with pytest.raises(ValueError, match="model_index.json:a=2"):
            r.resolve_one("a", Rule(sources, conflict=Conflict.ERROR))
        warned = r.resolve_one("a", Rule(sources, conflict=Conflict.WARN))
        assert warned.value == 1 and warned.disagreeing == ("model_index.json:a=2",)

    def test_float_lists_compare_within_tolerance(self, tmp_path):
        r = _resolver(
            tmp_path, {CONFIG: {"a": [0.5, 1 / 3]}, MODEL_INDEX: {"a": [0.5, 0.3333333333]}}
        )
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a")), conflict=Conflict.ERROR)
        assert r.resolve_one("a", rule).disagreeing == ()

    def test_serving_config_is_a_source(self, tmp_path):
        serving = SimpleNamespace(kv_cache=SimpleNamespace(free_gpu_memory_fraction=0.5))
        r = _resolver(tmp_path, {}, serving)
        rule = Rule((Serving("kv_cache.free_gpu_memory_fraction"),))
        assert r.resolve_one("f", rule).value == 0.5


def _sim_files(**sim_overrides):
    block = {
        "chunk_size": 2,
        "window_frames": None,
        "sink_frames": 0,
        "text_cache_max_len": 1024,
        "video_temporal_causal": True,
        "fixed_step_sampler_config": {"t_list": T_LIST},
        "conditioning": {"action_tokens_per_frame": 4},
        **sim_overrides,
    }
    return {
        TRANSFORMER: {SIM: block},
        SCHEDULER: {"fixed_step_sampler_config": {"t_list": T_LIST}},
        CONFIG: {"model": {"config": {"vlm_config": {"use_system_prompt": False}}}},
        MODEL_INDEX: {"default_use_system_prompt": True, f"_{SIM}": {"text_cache_max_len": 512}},
        CHECKPOINT: {SIM: {"chunk_partition": [1, 2], "text_cache_max_len": 512}},
        VAE: {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
    }


class TestSim:
    def test_training_config_beats_the_diffusers_index(self, tmp_path):
        cfg = _resolver(tmp_path, _sim_files()).resolve()
        assert cfg.prompt.use_system_prompt is False
        assert cfg.source_of("prompt.use_system_prompt") == (
            "config.json:model.config.vlm_config.use_system_prompt"
        )
        assert cfg.prompt.text_cache_max_len == 1024
        assert len(cfg.resolved["prompt.text_cache_max_len"].disagreeing) == 2

    def test_rollout_sections(self, tmp_path):
        cfg = _resolver(tmp_path, _sim_files()).resolve()
        assert cfg.is_autoregressive
        assert (cfg.chunking.chunk_frames, cfg.chunking.partition) == (2, (1, 2))
        assert (cfg.kv_cache.inference_size, cfg.kv_cache.sink_frames) == (None, 0)
        assert cfg.kv_cache.history_frames is None
        assert cfg.sampling.fixed_sigmas == tuple(T_LIST)
        assert cfg.action.tokens_per_frame == 4

    def test_partition_must_match_the_reference_rollout(self, tmp_path):
        files = _sim_files()
        files[CHECKPOINT][SIM]["chunk_partition"] = [1, 4]
        with pytest.raises(ValueError, match="reference rollout"):
            _resolver(tmp_path, files).resolve()

    def test_missing_partition_follows_the_reference_rollout(self, tmp_path):
        files = _sim_files()
        del files[CHECKPOINT][SIM]["chunk_partition"]
        cfg = _resolver(tmp_path, files).resolve()
        assert cfg.chunking.partition == (1, 2)

    def test_schedule_copies_must_agree(self, tmp_path):
        files = _sim_files()
        files[SCHEDULER]["fixed_step_sampler_config"]["t_list"] = [1.0, 0.5]
        with pytest.raises(ValueError, match="sampling.fixed_sigmas"):
            _resolver(tmp_path, files).resolve()

    def test_serving_window_overrides_full_history(self, tmp_path):
        """The reference recipe: 64 frames counting 8 sinks and the frame itself."""
        files = _sim_files()
        files[CONFIG]["model"]["config"].update(kv_cache_inference_size=None, attention_sink_size=0)
        serving = SimpleNamespace(
            kv_cache=SimpleNamespace(kv_cache_inference_size=64, attention_sink_size=8)
        )
        cfg = _resolver(tmp_path, files, serving).resolve()
        assert (cfg.kv_cache.inference_size, cfg.kv_cache.sink_frames) == (64, 8)
        assert cfg.kv_cache.history_frames == 55
        assert (
            cfg.source_of("kv_cache.inference_size") == "serving:kv_cache.kv_cache_inference_size"
        )
        assert cfg.resolved["kv_cache.sink_frames"].disagreeing == ()

    def test_unset_serving_window_keeps_the_checkpoint(self, tmp_path):
        files = _sim_files()
        files[CONFIG]["model"]["config"].update(kv_cache_inference_size=None, attention_sink_size=0)
        serving = SimpleNamespace(
            kv_cache=SimpleNamespace(kv_cache_inference_size=None, attention_sink_size=None)
        )
        cfg = _resolver(tmp_path, files, serving).resolve()
        assert cfg.kv_cache.inference_size is None and cfg.kv_cache.sink_frames == 0
        assert cfg.source_of("kv_cache.inference_size") == (
            "config.json:model.config.kv_cache_inference_size"
        )

    def test_sim_block_window_counts_the_chunk_first_frame(self, tmp_path):
        cfg = _resolver(tmp_path, _sim_files(window_frames=96)).resolve()
        assert (cfg.kv_cache.inference_size, cfg.kv_cache.history_frames) == (96, 95)

    def test_sinks_need_a_window(self, tmp_path):
        files = _sim_files()
        files[CONFIG]["model"]["config"].update(kv_cache_inference_size=None)
        serving = SimpleNamespace(
            kv_cache=SimpleNamespace(kv_cache_inference_size=None, attention_sink_size=8)
        )
        with pytest.raises(ValueError, match="need a kv_cache_inference_size"):
            _resolver(tmp_path, files, serving).resolve()

    def test_whole_clip_checkpoint_has_no_rollout_sections(self, tmp_path):
        files = {k: v for k, v in _sim_files().items() if k in (CONFIG, VAE)}
        files[TRANSFORMER] = {}
        cfg = _resolver(tmp_path, files).resolve()
        assert not cfg.is_autoregressive and cfg.chunking is None and cfg.kv_cache is None

    def test_system_prompt_must_be_declared(self, tmp_path):
        files = _sim_files()
        del files[CONFIG]["model"]["config"]["vlm_config"]["use_system_prompt"]
        del files[MODEL_INDEX]["default_use_system_prompt"]
        with pytest.raises(ValueError, match="cannot resolve prompt.use_system_prompt"):
            _resolver(tmp_path, files).resolve()


def test_fill_pretrained_config_sets_only_missing_attributes():
    declared = SimpleNamespace(temporal_compression_factor=8)
    fill_pretrained_config(declared)
    assert declared.temporal_compression_factor == 8
    assert declared.position_embedding_type == architecture_defaults()["position_embedding_type"]
    assert declared.max_position_embeddings == 262144
    assert declared.temporal_compression_factor_sound == 1


# Disagreements between a checkpoint's own files that the table expects and resolves.
EXPECTED_DISAGREEMENTS = {
    "nvidia/Cosmos3-Nano-Sim-Bimanual": {"prompt.use_system_prompt", "prompt.text_cache_max_len"},
}

EXPECTED = {
    # repo: (causal, use_system_prompt, native_flow_schedule, distilled)
    "nvidia/Cosmos3-Nano": (False, False, False, False),
    "nvidia/Cosmos3-Super": (False, False, False, False),
    "nvidia/Cosmos3-Edge": (False, False, True, False),
    "nvidia/Cosmos3-Super-Image2Video": (False, False, False, False),
    "nvidia/Cosmos3-Super-Image2Video-4Step": (False, False, False, True),
    "nvidia/Cosmos3-Super-Text2Image": (False, False, False, False),
    "nvidia/Cosmos3-Super-Text2Image-4Step": (False, False, False, True),
    "nvidia/Cosmos3-Nano-Policy-DROID": (False, False, False, False),
    "nvidia/Cosmos3-Edge-Policy-DROID": (False, False, True, False),
    "nvidia/Cosmos3-Nano-Sim-Bimanual": (True, False, False, True),
}


@pytest.mark.parametrize("repo", sorted(EXPECTED))
def test_published_checkpoint_resolves(tmp_path, repo):
    files = json.loads(FIXTURE.read_text())[repo]
    cfg = _resolver(tmp_path, files).resolve()
    causal, system_prompt, native, distilled = EXPECTED[repo]
    assert cfg.is_autoregressive is causal
    assert cfg.prompt.use_system_prompt is system_prompt
    assert cfg.sampling.native_flow_schedule is native
    assert (cfg.sampling.fixed_sigmas is not None) is distilled
    assert (cfg.vae.temporal_compression, cfg.vae.spatial_compression) == (4, 16)
    assert cfg.architecture.position_embedding_type == "unified_3d_mrope"
    disagreeing = {name for name, r in cfg.resolved.items() if r.disagreeing}
    assert disagreeing == EXPECTED_DISAGREEMENTS.get(repo, set())
    if causal:
        assert (cfg.chunking.chunk_frames, cfg.chunking.partition) == (2, (1, 2))
        assert (cfg.kv_cache.inference_size, cfg.kv_cache.sink_frames) == (None, 0)
        assert cfg.prompt.text_cache_max_len == 1024


def test_fixture_covers_every_expected_checkpoint():
    assert set(json.loads(FIXTURE.read_text())) == set(EXPECTED)


def test_kv_cache_config_validates_the_window():
    from tensorrt_llm.visual_gen.args import KVCacheConfig

    config = KVCacheConfig(kv_cache_inference_size=64, attention_sink_size=8)
    assert (config.kv_cache_inference_size, config.attention_sink_size) == (64, 8)
    assert KVCacheConfig().kv_cache_inference_size is None
    with pytest.raises(ValueError, match="requires kv_cache_inference_size"):
        KVCacheConfig(attention_sink_size=8)
    with pytest.raises(ValueError, match="leaves no recent frame"):
        KVCacheConfig(kv_cache_inference_size=10, attention_sink_size=9)
    with pytest.raises(ValueError):
        KVCacheConfig(kv_cache_inference_size=1)
