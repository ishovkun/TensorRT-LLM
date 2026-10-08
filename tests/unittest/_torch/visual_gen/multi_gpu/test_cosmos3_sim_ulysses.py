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
"""The Cosmos3 Sim rollout under Ulysses-2 equals the single-rank rollout.

Two ranks build the same model (same seed), run the same clean passes and one
denoising forward through the K/V cache with the sequence sharded between them,
and every rank's gathered output must match what one rank computes alone.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping

WORLD = 2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} GPUs",
)


def _sim_helpers():
    """The single-GPU Sim transformer test module: model builder, cache, rollout."""
    path = Path(__file__).resolve().parent.parent / "test_cosmos3_sim_transformer.py"
    spec = importlib.util.spec_from_file_location("sim_transformer_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rollouts(helpers, model):
    first = helpers.frames(5, seed=3)
    second = helpers.frames(2, seed=4)
    chunking = [(0, 1), (1, 3), (3, 5)]
    video, action = helpers.two_pass_rollout(model, chunking, first, second)
    return video, action


def _worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        helpers = _sim_helpers()
        reference = _rollouts(helpers, helpers.build_model("TRTLLM"))
        mapping = VisualGenMapping(world_size=world_size, rank=rank, ulysses_size=world_size)
        sharded_model = helpers.build_model("TRTLLM", mapping=mapping)
        out = _rollouts(helpers, sharded_model)
        helpers.assert_same_model_output(out[0], reference[0], label=f"rank {rank} video")
        helpers.assert_same_model_output(out[1], reference[1], label=f"rank {rank} action")
    finally:
        dist.destroy_process_group()


def test_sim_rollout_under_ulysses_matches_one_rank():
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(lambda port: mp.spawn(_worker, args=(WORLD, port), nprocs=WORLD, join=True))
