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
"""Minimal on-disk Cosmos3 Sim checkpoints for tests.

The Sim settings are resolved from the whole checkpoint directory, never from the
transformer config alone, so tests that want settings for a given Sim block write
the smallest directory the resolver accepts.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from tensorrt_llm._torch.visual_gen.models.cosmos3.config_resolver import Cosmos3ConfigResolver
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_sim import Cosmos3SimSettings
from tensorrt_llm._torch.visual_gen.models.cosmos3.sim_packing import SIM_CONFIG_KEY


def write_sim_checkpoint(
    checkpoint_dir: Path,
    sim_block: Optional[dict[str, Any]],
    *,
    use_system_prompt: bool = False,
) -> Path:
    """Write the files the resolver reads: the transformer config carrying
    ``sim_block`` (none when ``sim_block`` is ``None``), a scheduler config with the
    same schedule, the training config's system-prompt flag and the VAE config."""
    checkpoint_dir = Path(checkpoint_dir)
    transformer = {} if sim_block is None else {SIM_CONFIG_KEY: sim_block}
    sampler = (sim_block or {}).get("fixed_step_sampler_config") or {}
    files = {
        "transformer/config.json": transformer,
        "scheduler/scheduler_config.json": {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "stochastic_sampling": True,
            "fixed_step_sampler_config": sampler,
        },
        "config.json": {
            "model": {"config": {"vlm_config": {"use_system_prompt": use_system_prompt}}}
        },
        "vae/config.json": {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
    }
    for name, doc in files.items():
        path = checkpoint_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
    return checkpoint_dir


def sim_settings(sim_block: Optional[dict[str, Any]], checkpoint_dir=None) -> Cosmos3SimSettings:
    """The Sim settings a checkpoint with this Sim block resolves to."""
    checkpoint_dir = Path(checkpoint_dir or tempfile.mkdtemp(prefix="cosmos3-sim-"))
    write_sim_checkpoint(checkpoint_dir, sim_block)
    config = Cosmos3ConfigResolver(str(checkpoint_dir), None).resolve()
    return Cosmos3SimSettings.from_config(config)
