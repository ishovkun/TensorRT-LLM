# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VaeBlaze component tests: baseline equals the native WanVAE; each option stays within tolerance."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.vae_blaze import VaeBlazeOptions, VaeBlaze
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE

# Cosmos3-shaped (residual blocks, patch 2, 12 in/out channels) but tiny.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _pair(options: VaeBlazeOptions | None = None) -> tuple[WanVAE, VaeBlaze]:
    torch.manual_seed(0)
    native = WanVAE(SMALL_CONFIG)
    fast = VaeBlaze(SMALL_CONFIG, options)
    fast.load_state_dict(native.state_dict(), strict=True)
    return (
        native.to(device=DEVICE, dtype=DTYPE).eval(),
        fast.to(device=DEVICE, dtype=DTYPE).eval(),
    )


def _inputs(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    pixels = (torch.rand(1, 3, frames, 64, 96, generator=gen) * 2 - 1).to(DEVICE, DTYPE)
    latents = torch.randn(1, SMALL_CONFIG["z_dim"], (frames - 1) // 4 + 1, 4, 6, generator=gen).to(DEVICE, DTYPE)
    return pixels, latents


@pytest.mark.parametrize("frames", [1, 5, 9])
def test_vae_blaze_baseline_matches_native_exactly(frames: int) -> None:
    native, fast = _pair()
    pixels, latents = _inputs(frames)
    with torch.inference_mode():
        assert torch.equal(
            fast.encode(pixels, return_dict=False)[0].mode(),
            native.encode(pixels, return_dict=False)[0].mode(),
        )
        assert torch.equal(
            fast.decode(latents, return_dict=False)[0],
            native.decode(latents, return_dict=False)[0],
        )


def test_vae_blaze_state_dict_keys_match_native() -> None:
    native, fast = _pair()
    assert native.state_dict().keys() == fast.state_dict().keys()


def test_vae_blaze_options_reject_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown VaeBlazeOptions"):
        VaeBlazeOptions.from_dict({"nope": True})


OPTION_SETS = [
    {"fused_norm": True, "norm_mode": "fp32"},
    {"fused_norm": True, "norm_mode": "reference"},
    {"fused_norm": True, "fused_residual": True},
    {"fused_norm": True, "fused_residual": True, "norm_mode": "reference"},
    {"fused_upsample": True},
    {"fused_norm": True, "fused_residual": True, "fused_upsample": True},
    {"fused_norm": True, "fused_residual": True, "fused_bias": True},
    {"fused_norm": True, "fused_residual": True, "fused_bias": True, "norm_mode": "reference"},
    {"fused_norm": True, "fused_residual": True, "fused_upsample": True, "fused_bias": True},
    {"conv_out_frame_split": 3},
    {"first_frame_conv2d": True},
    {"fused_norm": True, "fused_residual": True, "fused_upsample": True, "fused_bias": True, "first_frame_conv2d": True},
    {"cudnn_benchmark": "auto"},
    {"fused_norm": True, "fused_residual": True, "fused_upsample": True, "fused_bias": True, "conv_out_frame_split": 3, "cudnn_benchmark": "auto"},
]


@pytest.mark.parametrize("option", OPTION_SETS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
@pytest.mark.parametrize("frames", [1, 9])
def test_each_option_stays_close_to_native(option: dict, frames: int) -> None:
    native, fast = _pair(VaeBlazeOptions(**option))
    pixels, latents = _inputs(frames)
    with torch.inference_mode():
        for actual, expected in (
            (fast.encode(pixels, return_dict=False)[0].mode(), native.encode(pixels, return_dict=False)[0].mode()),
            (fast.decode(latents, return_dict=False)[0], native.decode(latents, return_dict=False)[0]),
        ):
            diff = (actual.float() - expected.float())
            rel_l2 = diff.norm() / expected.float().norm().clamp_min(1e-12)
            assert torch.isfinite(actual.float()).all()
            # Both modes drift from native only through last-bit effects (fp32 mode: fewer
            # roundings; reference mode: fp32 reduction order), amplified by the random-weight
            # network. Bitwise equality of the kernel itself is checked in test_blaze_kernels.
            assert rel_l2.item() < 2e-2, f"{option}: rel_l2={rel_l2.item():.3e}"
