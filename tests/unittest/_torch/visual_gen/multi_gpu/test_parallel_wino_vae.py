# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for the spatially sharded WinoVae (``ParallelWinoVae``).

The sharded model must reproduce the single-GPU WinoVae with the same options: the fused
norm / slab / Winograd paths exchange halo lines instead of recomputing the borders.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_parallel_wino_vae.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae import WinoVae, WinoVaeOptions
from tensorrt_llm._torch.visual_gen.modules.vae.parallel_vae_interface import ParallelVAEFactory


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# Cosmos3-shaped (residual blocks, patch 2, 12 in/out channels) but tiny; the same config
# as test_wino_vae.py.
SMALL_CONFIG = {
    "base_dim": 16,
    "decoder_base_dim": 24,
    "z_dim": 8,
    "dim_mult": [1, 2, 2, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
}

ALL_ON = {
    "fused_norm": True,
    "fused_residual": True,
    "fused_upsample": True,
    "fused_bias": True,
    "first_frame_conv2d": True,
    "winograd_temporal": True,
    "winograd_spatial": "all",
    "conv_slabs": True,
    "fused_attention": True,
    "fold_encoder_tail": True,
    "fused_output": True,
}
ALL_OFF: dict = {}
SLABS_ONLY = {"fused_norm": True, "fused_residual": True, "fused_bias": True, "conv_slabs": True}
OPTION_SETS = {"all_on": ALL_ON, "all_off": ALL_OFF, "slabs": SLABS_ONLY}


def _init_worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _distributed_worker(rank, world_size, test_fn, port, args):
    try:
        _init_worker(rank, world_size, port)
        test_fn(rank, world_size, *args)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size: int, test_fn: Callable, *args):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, test_fn, port, args),
            nprocs=world_size,
            join=True,
        )
    )


def _model(device: str, options: dict) -> tuple[WinoVae, WanVAE]:
    """The bf16 WinoVae under test and the native model in fp32 on the same weights (the
    reference both the sharded and the single-GPU run are judged against)."""
    torch.manual_seed(0)
    model = WinoVae(SMALL_CONFIG, WinoVaeOptions.from_dict(options)).to(device=device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    native = WanVAE(SMALL_CONFIG).to(device=device)
    native.load_state_dict(model.state_dict(), strict=True)
    model = model.to(dtype=torch.bfloat16).eval()
    return model, native.to(dtype=torch.float32).eval()


def _parallel(model: WinoVae, world_size: int, split_dim: str):
    ranks = list(range(world_size))
    pg = dist.new_group(ranks, use_local_synchronization=False)
    adj = [
        dist.new_group([ranks[i], ranks[i + 1]], use_local_synchronization=False)
        for i in range(world_size - 1)
    ]
    return ParallelVAEFactory.from_vae(model, split_dim, pg, adj).eval()


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)).item()


def _check(
    got: torch.Tensor, single: torch.Tensor, ref32: torch.Tensor, split_dim: str, what: str
) -> None:
    """The sharded result must sit as close to the fp32 reference as the single-GPU one
    (different conv algorithms and Winograd tile phases change the bf16 rounding, so the two
    bf16 runs are not bitwise equal) and show no error spike at the rank boundary."""
    assert got.shape == single.shape, (got.shape, single.shape)
    rel_single = _rel_l2(single, ref32)
    rel_shard = _rel_l2(got, ref32)
    assert rel_shard <= 1.25 * rel_single + 1e-3, (
        f"{what}: sharded rel_l2={rel_shard:.3e} vs single {rel_single:.3e}"
    )
    axis = 3 if split_dim == "height" else 4
    err = (got.float() - single.float()).abs()
    dims = tuple(d for d in range(err.dim()) if d != axis)
    line = err.mean(dim=dims)
    n = line.shape[0]
    world = dist.get_world_size()
    seam = max(line[b - 1 : b + 1].max().item() for b in range(n // world, n, n // world))
    assert seam <= 2.5 * line.median().item() + 1e-4, (
        f"{what}: error at the rank boundary {seam:.3e} vs median line {line.median().item():.3e}"
    )


def _logic_decode(rank: int, world_size: int, split_dim: str, option_set: str, frames: int):
    device = f"cuda:{rank}"
    options = OPTION_SETS[option_set]
    model, native = _model(device, options)
    gen = torch.Generator().manual_seed(1)
    # Latent 8 x 12: divisible by two along either axis; pixels 128 x 192.
    latent = torch.randn(1, SMALL_CONFIG["z_dim"], frames, 8, 12, generator=gen).to(
        device, torch.bfloat16
    )
    with torch.inference_mode():
        ref32 = native.decode(latent.float(), return_dict=False)[0]
        single = model.decode(latent, return_dict=False)[0].clone()
        parallel = _parallel(model, world_size, split_dim)
        got = parallel.decode(latent, return_dict=False)[0]
    _check(got, single, ref32, split_dim, f"rank {rank}: decode {split_dim} {option_set}")


def _logic_encode(rank: int, world_size: int, split_dim: str, option_set: str, frames: int):
    device = f"cuda:{rank}"
    options = OPTION_SETS[option_set]
    model, native = _model(device, options)
    gen = torch.Generator().manual_seed(2)
    pixels = (torch.rand(1, 3, frames, 128, 192, generator=gen) * 2 - 1).to(device, torch.bfloat16)
    with torch.inference_mode():
        ref32 = native.encode(pixels.float(), return_dict=False)[0].mode()
        single = model.encode(pixels, return_dict=False)[0].mode().clone()
        parallel = _parallel(model, world_size, split_dim)
        got = parallel.encode(pixels, return_dict=False)[0].mode()
    _check(got, single, ref32, split_dim, f"rank {rank}: encode {split_dim} {option_set}")


class TestParallelWinoVaeDecode:
    @pytest.mark.parametrize("split_dim", ["height", "width"])
    @pytest.mark.parametrize("option_set", list(OPTION_SETS))
    @pytest.mark.parametrize("frames", [1, 5])
    def test_decode_2gpu(self, split_dim: str, option_set: str, frames: int) -> None:
        _run(2, _logic_decode, split_dim, option_set, frames)


class TestParallelWinoVaeEncode:
    @pytest.mark.parametrize("split_dim", ["height", "width"])
    @pytest.mark.parametrize("option_set", ["all_on", "all_off"])
    def test_encode_2gpu(self, split_dim: str, option_set: str) -> None:
        _run(2, _logic_encode, split_dim, option_set, 9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
