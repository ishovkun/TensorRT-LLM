# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm+SiLU kernel vs the eager WanRMSNorm + SiLU chain."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (
    can_fuse_rms_norm_silu,
    rms_norm_silu,
    rms_norm_silu_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")

# Decoder/encoder channel counts of the Cosmos3 VAE plus a few odd ones.
CHANNELS = [48, 96, 160, 256, 320, 512, 640, 1024, 12, 7, 16, 24, 32]


def _inputs(
    C: int, frames: int = 3, hw: tuple[int, int] = (6, 10), dtype=torch.bfloat16, bias: bool = False
):
    gen = torch.Generator(device="cuda").manual_seed(C)
    x = torch.randn(2, C, frames, *hw, generator=gen, device="cuda").to(dtype)
    x = x.to(memory_format=torch.channels_last_3d)
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(dtype)
    b = (0.05 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(dtype) if bias else None
    return x, gamma, b


@pytest.mark.parametrize("C", CHANNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_reference_rounding_matches_eager(C: int, bias: bool) -> None:
    x, gamma, b = _inputs(C, bias=bias)
    assert can_fuse_rms_norm_silu(x)
    scale = C**0.5
    expected = rms_norm_silu_reference(x, gamma, b if b is not None else 0.0, scale)
    actual = rms_norm_silu(x, gamma, b, scale, reference_rounding=True)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    # Only the fp32 reduction order can differ: allow one bf16 ulp on a handful of elements.
    diff = (actual.float() - expected.float()).abs()
    ulp = expected.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = (diff > ulp).float().mean().item()
    assert mismatched < 1e-3, f"C={C}: {mismatched:.2e} of elements differ by more than 1 bf16 ulp"
    assert torch.equal(actual, expected) or diff.max().item() <= 2 * ulp.max().item()


@pytest.mark.parametrize("C", CHANNELS)
def test_fp32_mode_is_closer_to_fp32_than_eager(C: int) -> None:
    x, gamma, b = _inputs(C, bias=True)
    scale = C**0.5
    truth = rms_norm_silu_reference(x.float(), gamma.float(), b.float(), scale)
    eager = rms_norm_silu_reference(x, gamma, b, scale).float()
    fused = rms_norm_silu(x, gamma, b, scale, reference_rounding=False).float()
    err_eager = (eager - truth).norm() / truth.norm()
    err_fused = (fused - truth).norm() / truth.norm()
    assert torch.isfinite(fused).all()
    assert err_fused <= err_eager * 1.05, f"C={C}: fused {err_fused:.3e} vs eager {err_eager:.3e}"
    assert err_fused < 1e-2


def test_zero_rows_and_small_norms() -> None:
    C = 64
    x, gamma, _ = _inputs(C)
    x[0, :, 0, 0, 0] = 0  # all-zero channel vector -> F.normalize divides by eps -> 0
    x[1, :, 1, 2, 3] = 1e-20
    for mode in (True, False):
        out = rms_norm_silu(x, gamma, None, C**0.5, reference_rounding=mode)
        assert torch.isfinite(out.float()).all()
        assert out[0, :, 0, 0, 0].abs().max().item() == 0.0


def test_fallback_conditions() -> None:
    x = torch.randn(1, 32, 2, 4, 4, device="cuda", dtype=torch.bfloat16)  # NCTHW, not channels-last
    assert not can_fuse_rms_norm_silu(x)
    assert can_fuse_rms_norm_silu(x.to(memory_format=torch.channels_last_3d))
    assert not can_fuse_rms_norm_silu(x.float().to(memory_format=torch.channels_last_3d))
    assert not can_fuse_rms_norm_silu(x.cpu())


def test_4d_images_layout() -> None:
    C = 96
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = (
        torch.randn(3, C, 8, 8, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last)
    )
    gamma = torch.ones(C, 1, 1, device="cuda", dtype=torch.bfloat16)
    expected = rms_norm_silu_reference(x, gamma, 0.0, C**0.5)
    actual = rms_norm_silu(x, gamma, None, C**0.5, reference_rounding=True)
    assert (actual.float() - expected.float()).abs().max().item() < 2e-2


# ---------------------------------------------------------------------------
# CUDA op (trtllm::vae_rms_norm_silu) against the Triton prototype and eager
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (  # noqa: E402
    add_rms_norm_silu_cuda,
    add_rms_norm_silu_reference,
    can_use_cuda_rms_norm_silu,
    cuda_op_available,
    rms_norm_silu_cuda,
)

cuda_op = pytest.mark.skipif(not cuda_op_available(), reason="trtllm::vae_rms_norm_silu not built")
CUDA_CHANNELS = [c for c in CHANNELS if c % 8 == 0]  # 7 and 12 take the Triton/eager path


@cuda_op
@pytest.mark.parametrize("C", CUDA_CHANNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_reference_mode_matches_eager(C: int, bias: bool) -> None:
    x, gamma, b = _inputs(C, frames=5, hw=(9, 13), bias=bias)
    assert can_use_cuda_rms_norm_silu(x)
    scale = C**0.5
    expected = rms_norm_silu_reference(x, gamma, b if b is not None else 0.0, scale)
    actual = rms_norm_silu_cuda(x, gamma, b, scale, reference_rounding=True)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    diff = (actual.float() - expected.float()).abs()
    ulp = expected.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = (diff > ulp).float().mean().item()
    assert mismatched < 1e-3, f"C={C}: {mismatched:.2e} of elements differ by more than 1 bf16 ulp"


@cuda_op
@pytest.mark.parametrize("C", CUDA_CHANNELS)
@pytest.mark.parametrize("reference", [False, True])
def test_cuda_matches_triton(C: int, reference: bool) -> None:
    x, gamma, b = _inputs(C, frames=5, hw=(9, 13), bias=True)
    scale = C**0.5
    cuda = rms_norm_silu_cuda(x, gamma, b, scale, reference_rounding=reference)
    tri = rms_norm_silu(x, gamma, b, scale, reference_rounding=reference)
    diff = (cuda.float() - tri.float()).abs()
    ulp = tri.float().abs().clamp_min(1e-3) * 2**-7
    # Only the fp32 reduction order can differ between the two implementations.
    assert (diff > ulp).float().mean().item() < 1e-3


@cuda_op
def test_cuda_tail_rows_and_zero_rows() -> None:
    C = 256
    gen = torch.Generator(device="cuda").manual_seed(7)
    # 1 batch x 3 frames x 7 x 11 = 231 rows: not a multiple of any block size.
    x = (
        torch.randn(1, C, 3, 7, 11, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    x[0, :, 0, 0, 0] = 0
    gamma = torch.ones(C, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    for reference in (False, True):
        out = rms_norm_silu_cuda(x, gamma, None, C**0.5, reference_rounding=reference)
        assert torch.isfinite(out.float()).all()
        assert out[0, :, 0, 0, 0].abs().max().item() == 0.0
        expected = rms_norm_silu_reference(x, gamma, 0.0, C**0.5)
        assert (out.float() - expected.float()).abs().max().item() < 2e-2


@cuda_op
@pytest.mark.parametrize("C", [256, 512, 160, 48])
def test_cuda_large_tensor_matches_triton(C: int) -> None:
    # ~300k rows: every persistent block streams many shared-memory stages.
    gen = torch.Generator(device="cuda").manual_seed(C + 1)
    x = (
        torch.randn(1, C, 3, 320, 320, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(torch.bfloat16)
    scale = C**0.5
    cuda = rms_norm_silu_cuda(x, gamma, None, scale)
    tri = rms_norm_silu(x, gamma, None, scale)
    ulp = tri.float().abs().clamp_min(1e-3) * 2**-7
    assert ((cuda.float() - tri.float()).abs() > ulp).float().mean().item() < 1e-3
    ref = rms_norm_silu_cuda(x, gamma, None, scale, reference_rounding=True)
    expected = rms_norm_silu_reference(x, gamma, 0.0, scale)
    assert ((ref.float() - expected.float()).abs() > ulp).float().mean().item() < 1e-3


@cuda_op
@pytest.mark.parametrize("C", [256, 512, 160, 48])
@pytest.mark.parametrize("reference", [True, False])
def test_cuda_residual_variant(C: int, reference: bool) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C + 7)
    x = (
        torch.randn(1, C, 3, 160, 160, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    r = (
        torch.randn(1, C, 3, 160, 160, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(torch.bfloat16)
    scale = C**0.5
    s_ref, a_ref = add_rms_norm_silu_reference(x, r, gamma, 0.0, scale)
    s, a = add_rms_norm_silu_cuda(x, r, gamma, None, scale, reference_rounding=reference)
    assert torch.equal(s, s_ref), "the returned sum must be bit-identical to the eager bf16 add"
    assert a.is_contiguous(memory_format=torch.channels_last_3d)
    ulp = a_ref.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = ((a.float() - a_ref.float()).abs() > ulp).float().mean().item()
    assert mismatched < (1e-3 if reference else 5e-2)
    # fp32 mode must also agree with the standalone norm applied to the sum
    if not reference:
        a2 = rms_norm_silu_cuda(s, gamma, None, scale)
        assert torch.equal(a, a2)


@cuda_op
@pytest.mark.parametrize("C", [256, 512, 160, 48, 16])
@pytest.mark.parametrize("reference", [True, False])
@pytest.mark.parametrize("with_residual", [False, True])
def test_cuda_in_bias(C: int, reference: bool, with_residual: bool) -> None:
    """A deferred conv bias is added first as a bf16 add; the sum stays bit-identical to eager."""
    gen = torch.Generator(device="cuda").manual_seed(C + 11)
    shape = (1, C, 2, 40, 40)
    x = (
        torch.randn(*shape, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    r = (
        torch.randn(*shape, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(torch.bfloat16)
    in_bias = (0.3 * torch.randn(C, generator=gen, device="cuda")).to(torch.bfloat16)
    scale = C**0.5
    if with_residual:
        s_ref, a_ref = add_rms_norm_silu_reference(x, r, gamma, 0.0, scale, in_bias=in_bias)
        s, a = add_rms_norm_silu_cuda(
            x, r, gamma, None, scale, reference_rounding=reference, in_bias=in_bias
        )
        assert torch.equal(s, s_ref)
    else:
        a_ref = rms_norm_silu_reference(x, gamma, 0.0, scale, in_bias=in_bias)
        a = rms_norm_silu_cuda(x, gamma, None, scale, reference_rounding=reference, in_bias=in_bias)
    ulp = a_ref.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = ((a.float() - a_ref.float()).abs() > ulp).float().mean().item()
    assert mismatched < (1e-3 if reference else 5e-2)
    # The in-bias path must equal the standalone kernel applied to the eagerly biased input.
    xb = (x + in_bias.view(1, -1, 1, 1, 1)).contiguous(memory_format=torch.channels_last_3d)
    if with_residual:
        _, a2 = add_rms_norm_silu_cuda(xb, r, gamma, None, scale, reference_rounding=reference)
    else:
        a2 = rms_norm_silu_cuda(xb, gamma, None, scale, reference_rounding=reference)
    assert torch.equal(a, a2)


# ---------------------------------------------------------------------------
# Upsample: packed 2x2 conv + phase gather vs nearest-2x + 3x3 conv (+ DupUp3D shortcut)
# ---------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402

from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (  # noqa: E402
    cuda_upsample_available,
    pack_upsample_weights,
    upsample_gather_cuda,
    upsample_gather_reference,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import DupUp3D  # noqa: E402

upsample_op = pytest.mark.skipif(
    not cuda_upsample_available(), reason="trtllm::vae_upsample_gather not built"
)


def _eager_upsample(x4: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    up = F.interpolate(x4.float(), scale_factor=2.0, mode="nearest-exact").to(x4.dtype)
    return F.conv2d(up, weight, bias, padding=1)


@pytest.mark.parametrize("shape", [(3, 16, 5, 7), (2, 8, 1, 1), (1, 24, 4, 6)])
def test_packed_upsample_algebra_fp32(shape: tuple[int, int, int, int]) -> None:
    bt, cin, h, w = shape
    cout = 16
    gen = torch.Generator(device="cuda").manual_seed(h * 31 + w)
    x = torch.randn(bt, cin, h, w, generator=gen, device="cuda")
    weight = torch.randn(cout, cin, 3, 3, generator=gen, device="cuda")
    bias = torch.randn(cout, generator=gen, device="cuda")
    # fp32 convolutions default to TF32 on Ampere+, which would hide the algebra behind
    # ~1e-2 of conv rounding; this test is about the exact identity.
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        expected = _eager_upsample(x, weight, bias)
        packed, packed_bias = pack_upsample_weights(weight, bias)
        temp = F.conv2d(x, packed, packed_bias, padding=1)
    finally:
        torch.backends.cudnn.allow_tf32 = prev
    got = upsample_gather_reference(temp.contiguous(memory_format=torch.channels_last), bt, 1)
    got = got[:, :, 0]  # frames=1 -> [bt, cout, 2h, 2w]
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


@upsample_op
@pytest.mark.parametrize("factor_t,first_chunk", [(1, False), (2, False), (2, True)])
# repeats = cout*factor/cin: equal to the factor (64->64), below it (64->32, strided
# span), above it (16->64 and 32->64, several output channels per source channel).
@pytest.mark.parametrize("cin_out", [(64, 64), (64, 32), (16, 64), (32, 64)])
def test_cuda_upsample_gather_with_shortcut(
    factor_t: int, first_chunk: bool, cin_out: tuple[int, int]
) -> None:
    cin, cout = cin_out
    b, tr, h, w = 1, 3, 9, 13
    frames = tr * factor_t - (factor_t - 1 if first_chunk else 0)
    gen = torch.Generator(device="cuda").manual_seed(cout + factor_t)
    # upsampler input after the temporal part: [b, cout_in=cin, frames, h, w]
    x5 = torch.randn(b, cin, frames, h, w, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (0.1 * torch.randn(cout, cin, 3, 3, generator=gen, device="cuda")).to(torch.bfloat16)
    bias = (0.1 * torch.randn(cout, generator=gen, device="cuda")).to(torch.bfloat16)
    residual = torch.randn(b, cin, tr, h, w, generator=gen, device="cuda").to(torch.bfloat16)
    residual = residual.to(memory_format=torch.channels_last_3d)
    dup = DupUp3D(cin, cout, factor_t=factor_t, factor_s=2)

    x4 = (
        x5.permute(0, 2, 1, 3, 4)
        .reshape(b * frames, cin, h, w)
        .contiguous(memory_format=torch.channels_last)
    )
    eager = _eager_upsample(x4, weight, bias)
    eager = eager.view(b, frames, cout, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
    eager = (eager + dup(residual, first_chunk=first_chunk)).contiguous(
        memory_format=torch.channels_last_3d
    )

    packed, packed_bias = pack_upsample_weights(weight, bias)
    temp = F.conv2d(x4, packed, packed_bias, padding=1).contiguous(
        memory_format=torch.channels_last
    )
    crop = factor_t - 1 if first_chunk else 0
    got = upsample_gather_cuda(temp, residual, b, frames, factor_t, dup.repeats, crop)
    assert got.shape == eager.shape and got.is_contiguous(memory_format=torch.channels_last_3d)

    # The kernel's gather + shortcut add must be bit-identical to the reference gather
    # followed by the eager DupUp3D add (same bf16 inputs, one bf16 rounding).
    via_reference = upsample_gather_reference(
        temp, b, frames, dup(residual, first_chunk=first_chunk)
    )
    assert torch.equal(got, via_reference)
    plain = upsample_gather_cuda(temp, None, b, frames)
    assert torch.equal(plain, upsample_gather_reference(temp, b, frames))

    # Deferred conv bias: the gather adds it before the shortcut, bit-identical to the
    # reference gather of the unbiased conv output followed by the eager bias add.
    temp_nb = F.conv2d(x4, packed, None, padding=1).contiguous(memory_format=torch.channels_last)
    got_nb = upsample_gather_cuda(
        temp_nb, residual, b, frames, factor_t, dup.repeats, crop, bias=bias
    )
    ref_nb = upsample_gather_reference(
        temp_nb, b, frames, dup(residual, first_chunk=first_chunk), bias=bias
    )
    assert torch.equal(got_nb, ref_nb)
    assert torch.equal(
        upsample_gather_cuda(temp_nb, None, b, frames, bias=bias),
        upsample_gather_reference(temp_nb, b, frames, bias=bias),
    )

    # Against the original upsample + 3x3 conv the only difference is the packed weights
    # rounding the summed taps once more: last-bit noise, never a mapping-sized error.
    diff = (got.float() - eager.float()).abs()
    assert (
        diff.norm() / eager.float().norm()
    ).item() < 5e-3  # one extra bf16 rounding of the summed taps
    assert diff.max().item() <= 2**-6 * eager.float().abs().max().item()


# ---------------------------------------------------------------------------
# Temporal Winograd norm kernel vs its eager reference
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (  # noqa: E402
    cuda_wino_available,
    wino_norm_silu_cuda,
    wino_norm_silu_reference,
)

wino_op = pytest.mark.skipif(
    not cuda_wino_available(), reason="trtllm::vae_wino_norm_silu not built"
)


@wino_op
@pytest.mark.parametrize("C", [256, 512, 1024, 160, 48])
@pytest.mark.parametrize("cache_frames", [0, 1, 2])
@pytest.mark.parametrize("source", ["x", "m"])
@pytest.mark.parametrize("extras", ["plain", "residual_bias"])
def test_wino_norm_silu_reference_mode_is_bitwise(
    C: int, cache_frames: int, source: str, extras: str
) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C + cache_frames)
    T, P = 4, 9 * 13
    bf = torch.bfloat16
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
    x = rnd(T, P, C) if source == "x" else None
    m = [rnd(T // 2, P, C) for _ in range(4)] if source == "m" else None
    residual = rnd(T, P, C) if extras == "residual_bias" else None
    in_bias = (
        (0.3 * torch.randn(C, generator=gen, device="cuda")).to(bf)
        if extras == "residual_bias"
        else None
    )
    cache = rnd(cache_frames, P, C) if cache_frames else None
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    scale = C**0.5
    s_ref, tail_ref, d_ref = wino_norm_silu_reference(
        x, m, residual, in_bias, cache, gamma, 0.0, scale
    )
    s, tail, d = wino_norm_silu_cuda(
        x, m, residual, in_bias, cache, gamma, None, scale, T, reference_rounding=True
    )
    if residual is not None:
        assert torch.equal(s, s_ref), "sum must be bit-identical to the eager adds"
    # The normalized frames (cache) and the slabs match eager up to the fp32 reduction
    # order of the sum of squares: at most 1 bf16 ulp on a tiny fraction of elements.
    for name, a, b in [("cache", tail, tail_ref), *((f"D{k}", d[k], d_ref[k]) for k in range(4))]:
        diff = (a.float() - b.float()).abs()
        ulp = b.float().abs().clamp_min(1e-3) * 2**-7
        assert (diff > ulp).float().mean().item() < 1e-3, f"{name} differs from the eager form"


@wino_op
@pytest.mark.parametrize("C", [256, 160])
def test_wino_norm_silu_fp32_mode_close(C: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C)
    T, P = 6, 20 * 20
    bf = torch.bfloat16
    x = torch.randn(T, P, C, generator=gen, device="cuda").to(bf)
    cache = torch.randn(2, P, C, generator=gen, device="cuda").to(bf)
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    _, tail_ref, d_ref = wino_norm_silu_reference(x, None, None, None, cache, gamma, 0.0, C**0.5)
    _, tail, d = wino_norm_silu_cuda(x, None, None, None, cache, gamma, None, C**0.5, T)
    for a, b in [(tail, tail_ref), *zip(d, d_ref)]:
        rel = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()
        assert rel < 1e-2, rel


# ---------------------------------------------------------------------------
# Frame ops: Winograd combine (+ bias + residual) and the time interleave (+ bias)
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (  # noqa: E402
    cuda_frame_ops_available,
    interleave_bias_cuda,
    interleave_bias_reference,
    wino_combine_cuda,
    wino_combine_reference,
)

frame_ops = pytest.mark.skipif(
    not cuda_frame_ops_available(), reason="trtllm::vae_wino_combine not built"
)


@frame_ops
@pytest.mark.parametrize("C", [256, 160, 48])
@pytest.mark.parametrize("source", ["x", "m"])
@pytest.mark.parametrize("extras", ["none", "bias", "residual", "bias_residual"])
def test_wino_combine_is_bitwise(C: int, source: str, extras: str) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C)
    T, P = 4, 7 * 11
    bf = torch.bfloat16
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
    x = rnd(T, P, C) if source == "x" else None
    m = [rnd(T // 2, P, C) for _ in range(4)] if source == "m" else None
    bias = (0.3 * torch.randn(C, generator=gen, device="cuda")).to(bf) if "bias" in extras else None
    residual = rnd(T, P, C) if "residual" in extras else None
    ref = x if x is not None else wino_combine_reference(m)
    if bias is not None:
        ref = ref + bias.reshape(1, 1, -1)
    if residual is not None:
        ref = ref + residual
    got = wino_combine_cuda(x, m, bias, residual, T, 7, 11)
    assert torch.equal(got, ref)
    # zero-padded output (the downsampler's ZeroPad2d((0, 1, 0, 1))) in the same pass
    padded = wino_combine_cuda(x, m, bias, residual, T, 7, 11, 1, 1).view(T, 8, 12, C)
    ref_img = F.pad(ref.view(T, 7, 11, C).permute(0, 3, 1, 2), (0, 1, 0, 1)).permute(0, 2, 3, 1)
    assert torch.equal(padded, ref_img)


@frame_ops
@pytest.mark.parametrize("C", [1024, 640, 48])
@pytest.mark.parametrize("with_bias", [False, True])
def test_interleave_bias_is_bitwise(C: int, with_bias: bool) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C)
    T, P = 3, 9 * 13
    u = torch.randn(T, P, 2 * C, generator=gen, device="cuda").to(torch.bfloat16)
    bias = (
        (0.3 * torch.randn(2 * C, generator=gen, device="cuda")).to(torch.bfloat16)
        if with_bias
        else None
    )
    assert torch.equal(interleave_bias_cuda(u, bias), interleave_bias_reference(u, bias))


from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import avg_down_add_cuda  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import AvgDown3D  # noqa: E402


@frame_ops
@pytest.mark.parametrize("cin_cout_ft", [(160, 160, 1), (160, 320, 2), (320, 640, 2)])
@pytest.mark.parametrize("frames", [4, 1])
def test_avg_down_add_matches_eager(cin_cout_ft: tuple[int, int, int], frames: int) -> None:
    cin, cout, ft = cin_cout_ft
    gen = torch.Generator(device="cuda").manual_seed(cin + frames)
    h, w = 12, 20
    bf = torch.bfloat16
    r = (
        torch.randn(1, cin, frames, h, w, generator=gen, device="cuda")
        .to(bf)
        .to(memory_format=torch.channels_last_3d)
    )
    dup = AvgDown3D(cin, cout, factor_t=ft, factor_s=2)
    ref_avg = dup(r)
    x = (
        torch.randn(ref_avg.shape, generator=gen, device="cuda")
        .to(bf)
        .to(memory_format=torch.channels_last_3d)
    )
    ref = x + ref_avg
    pad_t = (ft - frames % ft) % ft
    got = avg_down_add_cuda(
        r.new_empty(0)
        if False
        else x.permute(0, 2, 3, 4, 1).reshape(x.shape[2], h // 2 * (w // 2), cout),
        r.permute(0, 2, 3, 4, 1).reshape(frames, h * w, cin),
        h // 2,
        w // 2,
        w,
        ft,
        2,
        dup.group_size,
        pad_t,
    )
    got = got.view(1, x.shape[2], h // 2, w // 2, cout).permute(0, 4, 1, 2, 3)
    assert torch.equal(got, ref), "group size 4 means the mean is exact: bitwise with eager"


from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import rms_norm_reference  # noqa: E402


@cuda_op
@pytest.mark.parametrize("C", [1024, 640, 256])
@pytest.mark.parametrize("reference", [True, False])
def test_cuda_norm_without_activation(C: int, reference: bool) -> None:
    x, gamma, _ = _inputs(C, frames=2, hw=(9, 13))
    expected = rms_norm_reference(x, gamma, C**0.5)
    actual = rms_norm_silu_cuda(x, gamma, None, C**0.5, reference_rounding=reference, silu=False)
    ulp = expected.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = ((actual.float() - expected.float()).abs() > ulp).float().mean().item()
    assert mismatched < (1e-3 if reference else 5e-2)


# --- spatial Winograd F(2x2, 3x3) transforms -------------------------------------------

spatial_op = pytest.mark.skipif(
    not (hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_wino_spatial_in")),
    reason="trtllm::vae_wino_spatial_in not built",
)
_BT = torch.tensor([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=torch.float32)
_AT = torch.tensor([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=torch.float32)


def _ref_spatial_in(frames: torch.Tensor, components: int, row0: int, row1: int) -> torch.Tensor:
    """frames [N, H, W, C] -> V [components*16, (N/components) * strip tiles, C] in fp32."""
    n, h, w, c = frames.shape
    x = frames.permute(0, 3, 1, 2).float()
    tiles = F.unfold(
        F.pad(x, (1, 1 + w % 2, 1, 1 + h % 2)), kernel_size=4, stride=2
    )  # [n, c*16, L]
    th, tw = (h + 1) // 2, (w + 1) // 2
    tiles = tiles.view(n, c, 4, 4, th, tw)[..., row0:row1, :].reshape(n, c, 4, 4, -1)
    bt = _BT.to(frames.device)
    v = torch.einsum("xa,ncabl,yb->nxylc", bt, tiles, bt)
    pairs = n // components
    return (
        v.reshape(components, pairs, 16, -1, c)
        .permute(0, 2, 1, 3, 4)
        .reshape(components * 16, -1, c)
    )


def _ref_spatial_out(
    m: torch.Tensor, n: int, h: int, w: int, components: int, row0: int, row1: int
) -> torch.Tensor:
    """M [components*16, (N/components) * strip tiles, C] -> y [N, 2*(row1-row0), W, C] in fp32."""
    comp16, _, c = m.shape
    pairs = n // components
    rows, tw = row1 - row0, (w + 1) // 2
    mm = m.float().view(components, 4, 4, pairs, rows, tw, c)
    at = _AT.to(m.device)
    y = torch.einsum("rx,nxyphwc,sy->nphrwsc", at, mm, at)
    return y.reshape(n, rows * 2, tw * 2, c)[:, : min(rows * 2, h - 2 * row0), :w]


@spatial_op
@pytest.mark.parametrize("C", [8, 160, 256])
@pytest.mark.parametrize("hw", [(6, 10), (5, 9)])
@pytest.mark.parametrize("strip", [(0, 3), (1, 2), (2, 3)])
def test_spatial_winograd_transforms(C: int, strip: tuple[int, int], hw: tuple[int, int]) -> None:
    torch.manual_seed(0)
    n, components = 8, 4
    (h, w), (row0, row1) = hw, strip
    frames = torch.randn(n, h, w, C, device="cuda").to(torch.bfloat16)
    v = torch.ops.trtllm.vae_wino_spatial_in(frames, components, row0, row1)
    v_ref = _ref_spatial_in(frames, components, row0, row1)
    assert v.shape == v_ref.shape
    assert torch.equal(v.float(), v_ref.to(torch.bfloat16).float())

    m = torch.randn(components * 16, v.shape[1], C, device="cuda").to(torch.bfloat16)
    out = torch.full((n, h, w, C), float("nan"), device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.vae_wino_spatial_out(m, out, components, row0, row1)
    got = out[:, 2 * row0 : min(2 * row1, h)]
    assert torch.equal(
        got.float(), _ref_spatial_out(m, n, h, w, components, row0, row1).to(torch.bfloat16).float()
    )
    untouched = torch.cat([out[:, : 2 * row0], out[:, 2 * row1 :]], dim=1)
    assert torch.isnan(untouched.float()).all()


@spatial_op
@pytest.mark.parametrize("cin_cout", [(16, 24), (64, 32)])
@pytest.mark.parametrize("hw", [(6, 10), (5, 9)])
def test_spatial_winograd_conv_matches_conv2d(
    cin_cout: tuple[int, int], hw: tuple[int, int]
) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (
        wino_spatial_conv_cuda,
        winograd_spatial_weights,
        winograd_weights,
    )

    torch.manual_seed(1)
    cin, cout = cin_cout
    h, w = hw
    pairs = 3
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    d = torch.randn(4, pairs, h * w, cin, device="cuda").to(torch.bfloat16)
    got = wino_spatial_conv_cuda(d, winograd_spatial_weights(weight), h, w)
    g = winograd_weights(weight)
    for k in range(4):
        ref = F.conv2d(
            d[k].view(pairs, h, w, cin).permute(0, 3, 1, 2).float(), g[k].float(), None, 1, 1
        )
        ref = ref.permute(0, 2, 3, 1).reshape(pairs, h * w, cout)
        rel = ((got[k].float() - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, f"component {k}: rel_l2={rel:.2e}"


@spatial_op
@pytest.mark.parametrize("cin_cout_hw", [(64, 128, 6, 10), (128, 256, 9, 13)])
def test_spatial_winograd_cute_matches_cublas_path(cin_cout_hw: tuple[int, int, int, int]) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.blaze_kernels import (
        cute_wino_available,
        wino_spatial_conv_cuda,
        wino_spatial_conv_cute,
        winograd_spatial_weights,
    )

    if not cute_wino_available():
        pytest.skip("CuTe DSL spatial Winograd needs Blackwell and nvidia-cutlass-dsl")
    torch.manual_seed(2)
    cin, cout, h, w = cin_cout_hw
    pairs = 3
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    d = torch.randn(4, pairs, h * w, cin, device="cuda").to(torch.bfloat16)
    u = winograd_spatial_weights(weight)
    ref = wino_spatial_conv_cuda(d, u, h, w).float()
    got = wino_spatial_conv_cute(d, u.transpose(1, 2).contiguous(), h, w).float()
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"rel_l2={rel:.2e}"
