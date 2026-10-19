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

"""VaeBlaze: optimized Wan VAE.

``VaeBlaze`` mirrors ``WanVAE`` module for module and loads the same state
dict. Every optimization is a flag in ``VaeBlazeOptions``; with all flags off
the computation is the unmodified native path, which is the baseline the
benchmark compares against.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

from .blaze_kernels import (
    add_rms_norm_silu_cuda,
    apply_in_bias,
    avg_down_add_cuda,
    can_fuse_rms_norm_silu,
    can_use_cuda_rms_norm_silu,
    can_use_cuda_upsample_gather,
    cuda_frame_ops_available,
    cuda_op_available,
    cuda_wino_available,
    cuda_wino_spatial_available,
    cute_spatial_filters,
    cute_wino_available,
    frames_to_video,
    frames_view,
    images_to_pairs,
    interleave_bias_cuda,
    kf_spatial_eligible,
    kf_spatial_filters,
    kf_wino_available,
    pack_upsample_weights,
    pairs_to_images,
    rms_norm_silu,
    rms_norm_silu_cuda,
    unpatchify_clamp_available,
    unpatchify_clamp_into,
    upsample_gather_cuda,
    upsample_gather_reference,
    wino_combine_cuda,
    wino_combine_reference,
    wino_norm_silu_cuda,
    wino_spatial_conv_cuda,
    wino_spatial_conv_cute,
    wino_spatial_conv_frames,
    wino_spatial_conv_kf,
    winograd_spatial_taps_weights,
    winograd_spatial_weights,
    winograd_weights,
)
from .wan_vae import (
    CACHE_T,
    AvgDown3D,
    DupUp3D,
    WanAttentionBlock,
    WanConv2d,
    WanUpsample,
    WanVAEConfig,
    _activation,
    _causal_conv_with_cache,
    _channels_last_2d_if_needed,
    _channels_last_3d_if_needed,
    _decode_chunk_slices,
    _to_channels_last,
    _to_device_if_needed,
    patchify,
    unpatchify,
)


@dataclass
class VaeBlazeOptions:
    """Optimization switches. All off reproduces the native ``WanVAE`` exactly."""

    # RMSNorm + SiLU in one kernel (checklist item 1). ``norm_mode`` picks the
    # numerics: "fp32" keeps everything in fp32 until the store; "reference"
    # reproduces the eager path's intermediate roundings for a bitwise-style gate.
    fused_norm: bool = False
    norm_mode: str = "fp32"
    # "cuda": the trtllm::vae_rms_norm_silu op (falls back to Triton when the op is not
    # built or the shape is unsupported); "triton": the prototype kernel.
    norm_kernel: str = "cuda"
    # Checklist item 2: a residual block hands its ``conv2 + residual`` add to the next
    # block's norm kernel instead of running it as a separate pass. Needs fused_norm and
    # the CUDA op; consumers that are not a norm materialize the add themselves.
    fused_residual: bool = False
    # Checklist item 3: nearest-2x upsample + 3x3 conv as one 2x2 conv on the low-res
    # grid followed by a phase gather that also adds the DupUp3D shortcut.
    fused_upsample: bool = False
    # Checklist item 4: a convolution runs without its bias and the consumer kernel adds
    # it (the fused norm, or the upsample gather). Needs fused_norm + fused_residual (the
    # bias rides on the same PendingAdd) and the CUDA op. In "fp32" norm mode a
    # channel-changing block merges its shortcut conv's bias into the same vector; in
    # "reference" mode that shortcut bias stays eager so the result is still bitwise.
    fused_bias: bool = False
    # Checklist item 5 follow-ups. ``cudnn_benchmark``: "off" uses cuDNN's heuristics
    # everywhere, "on" runs cudnn.benchmark for every conv, "auto" benchmarks the
    # encoder and single-frame (image) decodes, where it measured faster, and keeps the
    # heuristics for video decodes, where it measured slower.
    cudnn_benchmark: str = "off"
    # ``conv_out_frame_split``: run the decoder's narrow conv_out (256 -> 12 channels) on
    # at most this many input frames per call (0 = off); cuDNN picks a kernel ~10x slower
    # for that shape once the input has more than three frames.
    conv_out_frame_split: int = 0
    # Checklist item 7: on a chunk with no causal history (the first chunk of a video,
    # every image) a 3x3x3 causal conv on one frame equals a 3x3 conv2d with the
    # current-frame weight slice; the two zero-history taps are skipped (cuDNN conv2d, or
    # spatial Winograd with ``winograd_spatial_first_frame``).
    first_frame_conv2d: bool = False
    # Checklist item 8: temporal Winograd F(2,3) on the residual-block convs of chunks
    # with an even number (>= 2) of frames. The norm kernel emits the four transformed
    # frame slabs, the conv runs as four 3x3 conv2d calls, and the next norm kernel
    # combines the four results while reading. Needs fused_norm with the CUDA op.
    winograd_temporal: bool = False
    # Checklist item 22: spatial Winograd F(2x2,3x3) on top of the temporal one. The four
    # 3x3 conv2d calls become an input transform, 64 cuBLAS batched GEMMs and an output
    # transform. Single-frame calls (the first frame of a clip, every image, and the
    # one-frame chunks of a streaming decode at the low-resolution levels) run spatial
    # Winograd alone: the three temporal taps are summed inside the GEMM (K = 3 * Cin), the
    # zero-history first frame keeps one tap. It pays where cuDNN's implicit GEMM is weak
    # (few output pixels per channel): "auto" uses it on the measured winners (see
    # ``spatial_eligible``), "all" on every eligible conv (tests), "off" never.
    winograd_spatial: str = "off"
    # Spatial-Winograd kernel for the frame-pair path: "kf" = the Kernel Factory F(4x4, 3x3)
    # pipeline (CuTe DSL on Blackwell; Cin % 64 == 0, Cout % 256 == 0, width % 4 == 0; the
    # portable path elsewhere), the fastest on every eligible layer measured on GB200;
    # "cublas" = the portable path (transform kernels + cuBLAS batched GEMMs); "cute" = the
    # CuTe DSL two-pass F(2x2, 3x3) kernel on Blackwell for Cout % 256 == 0 and frames of at
    # least 45x80 pixels (cuBLAS elsewhere). Single-frame calls always use the portable path.
    winograd_spatial_kernel: str = "kf"
    # Spatial tile of the portable path: 2 = F(2x2, 3x3), 4 = F(4x4, 3x3) (1.8x fewer GEMM FLOPs
    # and transformed bytes; fp16 storage keeps its error below the bf16 F2 path).
    winograd_spatial_tile: int = 4
    # Tile for the single-frame calls with a causal history (spatial Winograd alone, the three
    # taps summed in the GEMM). They replace a direct cuDNN conv3d, so F(2x2, 3x3), whose
    # per-conv error matches direct bf16, keeps the model's accuracy; F(4x4, 3x3) is faster
    # and ~2x noisier per conv.
    winograd_spatial_single_tile: int = 2
    # Run the zero-history first frame (and every image) as spatial Winograd on the
    # current-frame tap instead of cuDNN conv2d. Faster at the 1024-channel levels with
    # ``winograd_spatial_tile = 4``, at ~2x the per-conv error of the conv2d it replaces.
    winograd_spatial_first_frame: bool = False
    # Checklist item 9: the convs that still run as 3D with a causal cache (one frame per
    # chunk at the low-res levels, conv_out) read a per-conv frame buffer the norm kernel
    # writes into, so the cache is two frames already in place: no cat, no F.pad, no memset.
    conv_slabs: bool = False
    # Attention block: fused RMSNorm (no activation), the two 1x1 convs as GEMMs with the
    # bias in the epilogue on the row view (no permute/contiguous copy), and the residual
    # add handed to the next block's norm as a PendingAdd.
    fused_attention: bool = False
    # Checklist item 15: the encoder's conv_out (3x3x3) and quant_conv (1x1x1) have no
    # activation between them, so quant_conv folds into conv_out's weights (composed in
    # fp32, rounded once); ``encode_mode`` then runs only the mean half of the folded conv.
    # Off in "reference" mode, where every rounding of the eager path is reproduced.
    fold_encoder_tail: bool = False
    # Checklist item 20: pixel frames per encoder pass after the first one-frame pass. Must
    # be a multiple of 4 so every pass ends at the same temporal-downsampler parity. Larger
    # passes give the convs more rows and fewer launches: 720p encode 726 ms at 4 frames,
    # 628 at 24 (flat beyond), 480p 224 -> 184; peak memory 9 -> 29 GB at 720p. From 8 frames
    # up the deepest levels see frame pairs and take the Winograd pair path, which moves the
    # encode error from 1.27e-2 to 1.35e-2 against the fp32 reference (decode's level).
    encoder_chunk_frames: int = 24
    # Checklist item 14: the decoder writes each chunk's unpatchified, clamped frames
    # straight into the preallocated output (one kernel per chunk) instead of
    # cat + unpatchify + clamp over the whole video. Needs the CUDA op; bf16, batch 1.
    fused_output: bool = False

    def __post_init__(self) -> None:
        if self.norm_mode not in ("fp32", "reference"):
            raise ValueError(f"norm_mode must be 'fp32' or 'reference', got {self.norm_mode!r}")
        if self.norm_kernel not in ("cuda", "triton"):
            raise ValueError(f"norm_kernel must be 'cuda' or 'triton', got {self.norm_kernel!r}")
        if self.cudnn_benchmark not in ("off", "on", "auto"):
            raise ValueError(
                f"cudnn_benchmark must be 'off', 'on' or 'auto', got {self.cudnn_benchmark!r}"
            )
        if self.winograd_spatial not in ("off", "auto", "all"):
            raise ValueError(
                f"winograd_spatial must be 'off', 'auto' or 'all', got {self.winograd_spatial!r}"
            )
        if self.winograd_spatial_single_tile not in (2, 4):
            raise ValueError(
                f"winograd_spatial_single_tile must be 2 or 4, got {self.winograd_spatial_single_tile!r}"
            )
        if self.winograd_spatial_tile not in (2, 4):
            raise ValueError(
                f"winograd_spatial_tile must be 2 or 4, got {self.winograd_spatial_tile!r}"
            )
        if self.winograd_spatial_kernel not in ("kf", "cublas", "cute"):
            raise ValueError(
                "winograd_spatial_kernel must be 'kf', 'cublas' or 'cute', "
                f"got {self.winograd_spatial_kernel!r}"
            )
        if self.conv_out_frame_split < 0 or self.conv_out_frame_split in (1, 2):
            raise ValueError("conv_out_frame_split must be 0 (off) or at least 3 input frames")
        if self.encoder_chunk_frames < 4 or self.encoder_chunk_frames % 4:
            raise ValueError(
                f"encoder_chunk_frames must be a positive multiple of 4, got {self.encoder_chunk_frames!r}"
            )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VaeBlazeOptions":
        valid = {field.name for field in fields(cls)}
        unknown = set(values) - valid
        if unknown:
            raise ValueError(f"Unknown VaeBlazeOptions: {sorted(unknown)}; valid: {sorted(valid)}")
        return cls(**values)

    @property
    def use_attention(self) -> bool:
        return (
            self.fused_attention
            and self.fused_norm
            and self.norm_kernel == "cuda"
            and cuda_op_available()
        )

    @property
    def use_slabs(self) -> bool:
        return (
            self.conv_slabs
            and self.fused_norm
            and self.norm_kernel == "cuda"
            and cuda_op_available()
        )

    @property
    def use_winograd(self) -> bool:
        return (
            self.winograd_temporal
            and self.fused_norm
            and self.norm_kernel == "cuda"
            and cuda_wino_available()
        )

    @property
    def use_winograd_spatial(self) -> bool:
        return (
            self.winograd_spatial != "off" and self.use_winograd and cuda_wino_spatial_available()
        )

    @property
    def defer_conv_bias(self) -> bool:
        return (
            self.fused_bias
            and self.fused_norm
            and self.fused_residual
            and self.norm_kernel == "cuda"
        )

    def enabled(self) -> list[str]:
        return [field.name for field in fields(self) if getattr(self, field.name) is True]

    def benchmark_convs(self, *, encoder: bool, frames: int) -> bool:
        if self.cudnn_benchmark == "on":
            return True
        if self.cudnn_benchmark == "auto":
            return encoder or frames == 1
        return False


@dataclass
class PendingAdd:
    """``(a + bias) + b`` not yet computed: the next fused norm adds it while reading.

    ``bias`` is a per-channel convolution bias the conv did not apply; ``b`` is a
    residual. Either may be absent. When ``wino`` is set, ``a`` is None and the frames
    are the Winograd combine of the four conv2d results ``wino`` (each
    ``[pairs, C, H, W]`` channels-last): frame 2p = m0 + m1 + m2, frame 2p+1 = m1 - m2 - m3.
    """

    a: torch.Tensor | None
    b: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    wino: list[torch.Tensor] | None = None

    @property
    def frames(self) -> int:
        return self.a.shape[2] if self.a is not None else 2 * self.wino[0].shape[0]

    def combined(self) -> torch.Tensor:
        """``a`` as a tensor (eager Winograd combine when needed)."""
        if self.a is not None:
            return self.a
        assert self.wino is not None
        h, w = self.wino[0].shape[-2:]
        return frames_to_video(
            wino_combine_reference([images_to_pairs(mk) for mk in self.wino]), h, w
        )

    def materialize(self) -> torch.Tensor:
        """The frames as a tensor: one fused pass when the op is built, else the eager chain."""
        src = self.a if self.a is not None else self.wino[0]
        batch_one = (self.a is None or self.a.shape[0] == 1) and (
            self.b is None or self.b.shape[0] == 1
        )
        if cuda_frame_ops_available() and src.dtype == torch.bfloat16 and src.is_cuda and batch_one:
            h, w = src.shape[-2:]
            return frames_to_video(self._combine_cuda(0, 0), h, w)
        s = apply_in_bias(self.combined(), self.bias)
        if self.b is not None:
            s = s + self.b
        return _channels_last_3d_if_needed(s)

    def _combine_cuda(self, pad_h: int, pad_w: int) -> torch.Tensor:
        src = self.a if self.a is not None else self.wino[0]
        h, w = src.shape[-2:]
        x_f = frames_view(_channels_last_3d_if_needed(self.a)) if self.a is not None else None
        m_f = [images_to_pairs(mk) for mk in self.wino] if self.wino is not None else None
        r_f = frames_view(_channels_last_3d_if_needed(self.b)) if self.b is not None else None
        return wino_combine_cuda(x_f, m_f, self.bias, r_f, self.frames, h, w, pad_h, pad_w)

    def can_fuse(self) -> bool:
        src = self.a if self.a is not None else self.wino[0]
        batch_one = (self.a is None or self.a.shape[0] == 1) and (
            self.b is None or self.b.shape[0] == 1
        )
        return (
            cuda_frame_ops_available() and src.dtype == torch.bfloat16 and src.is_cuda and batch_one
        )

    def materialize_padded_2d(self) -> torch.Tensor:
        """The frames with one zero row and column appended, as ``[T, C, H+1, W+1]`` channels-last
        (the downsampler's input), in one pass."""
        src = self.a if self.a is not None else self.wino[0]
        h, w = src.shape[-2:]
        c = src.shape[1]
        y = self._combine_cuda(1, 1)  # [T, (H+1)*(W+1), C]
        return y.view(self.frames, h + 1, w + 1, c).permute(0, 3, 1, 2)


@dataclass
class WinoSlabs:
    """The four transformed input slabs ``D0..D3`` of a Winograd conv, each ``[pairs, C, H, W]``
    channels-last, plus the buffer they are views of as ``[4, pairs, H*W, C]``."""

    d: list[torch.Tensor]
    stacked: torch.Tensor


def _pending_like(x: "torch.Tensor | PendingAdd") -> torch.Tensor:
    """A tensor with the frames' batch/channel/spatial shape, without materializing."""
    if not isinstance(x, PendingAdd):
        return x
    if x.a is not None:
        return x.a
    return x.b if x.b is not None else x.wino[0]


def _into(out: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Copy ``value`` into ``out`` when a destination was requested."""
    if out is None:
        return value
    out.copy_(value)
    return out


def _tensor(x: "torch.Tensor | PendingAdd") -> torch.Tensor:
    return x.materialize() if isinstance(x, PendingAdd) else x


class BlazeRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        options: VaeBlazeOptions | None = None,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0
        self.options = options or VaeBlazeOptions()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dim = 1 if self.channel_first else -1
        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
            t in str(x.dtype) for t in ("float4_", "float8_")
        )
        normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=norm_dim).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias

    def forward_silu(
        self, x: "torch.Tensor | PendingAdd", out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """SiLU(norm(x)); one fused pass when enabled and supported, else the eager chain.
        ``out``: optional channels-last destination (a conv frame-buffer window)."""
        if isinstance(x, PendingAdd):
            if x.wino is not None:
                return self._forward_silu(x.materialize(), None, out)
            if x.b is not None:
                return self.forward_silu_pending(x, out)[1]
            return self._forward_silu(x.a, x.bias, out)
        return self._forward_silu(x, None, out)

    def forward_silu_wino(
        self, x: "torch.Tensor | PendingAdd", cache: torch.Tensor | None, frames: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, WinoSlabs]:
        """Norm + SiLU that writes the Winograd slabs for the conv that follows.

        ``cache`` holds the last 0..2 normalized frames of the previous chunk as
        ``[1, C, n, H, W]``. Returns ``(sum or None, new cache [1, C, 2, H, W], slabs)``."""
        pending = x if isinstance(x, PendingAdd) else PendingAdd(x)
        src = pending.a if pending.a is not None else pending.wino[0]
        h, w = src.shape[-2:]
        bias = self.bias if isinstance(self.bias, torch.Tensor) else None
        s, y_tail, d = wino_norm_silu_cuda(
            frames_view(pending.a) if pending.a is not None else None,
            [images_to_pairs(mk) for mk in pending.wino] if pending.wino is not None else None,
            frames_view(pending.b) if pending.b is not None else None,
            pending.bias,
            frames_view(cache) if cache is not None and cache.shape[2] > 0 else None,
            self.gamma,
            bias,
            self.scale,
            frames,
            reference_rounding=self.options.norm_mode == "reference",
        )
        slabs = WinoSlabs([pairs_to_images(dk, h, w) for dk in d], d)
        return (
            (frames_to_video(s, h, w) if s is not None else None),
            frames_to_video(y_tail, h, w),
            slabs,
        )

    def _forward_silu(
        self, x: torch.Tensor, in_bias: torch.Tensor | None, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not (self.options.fused_norm and self.channel_first):
            return _into(out, F.silu(self.forward(apply_in_bias(x, in_bias))))
        bias = self.bias if isinstance(self.bias, torch.Tensor) else None
        reference = self.options.norm_mode == "reference"
        if self.options.norm_kernel == "cuda" and can_use_cuda_rms_norm_silu(x):
            return rms_norm_silu_cuda(
                x,
                self.gamma,
                bias,
                self.scale,
                reference_rounding=reference,
                in_bias=in_bias,
                out=out,
            )
        x = apply_in_bias(x, in_bias)
        if can_fuse_rms_norm_silu(x):
            return _into(
                out, rms_norm_silu(x, self.gamma, bias, self.scale, reference_rounding=reference)
            )
        return _into(out, F.silu(self.forward(x)))

    def forward_silu_pending(
        self, pending: PendingAdd, out: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(s, SiLU(norm(s)))`` with ``s = (a + bias) + b``, added inside the kernel when possible."""
        bias = self.bias if isinstance(self.bias, torch.Tensor) else None
        reference = self.options.norm_mode == "reference"
        if pending.wino is not None:
            s = pending.materialize()
            return s, self._forward_silu(s, None, out)
        if (
            pending.b is not None
            and self.options.fused_norm
            and self.channel_first
            and self.options.norm_kernel == "cuda"
            and can_use_cuda_rms_norm_silu(pending.a)
            and can_use_cuda_rms_norm_silu(pending.b)
        ):
            return add_rms_norm_silu_cuda(
                pending.a,
                pending.b,
                self.gamma,
                bias,
                self.scale,
                reference_rounding=reference,
                in_bias=pending.bias,
                out=out,
            )
        s = pending.materialize()
        return s, self.forward_silu(s, out)


class BlazeCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self._padding = (0, 0, 0, 0, 2 * self.padding[0], 0)
        self.padding = (0, self.padding[1], self.padding[2])
        # When set, forward() leaves the bias out; the caller hands it to the consumer kernel.
        self.defer_bias = False
        # When set, a stride-1 temporal conv runs on at most this many input frames per
        # cuDNN call and concatenates the output frames (see VaeBlazeOptions.conv_out_frame_split).
        self.max_in_frames: int | None = None
        # When set, a single-frame call with no cache runs as conv2d on the last temporal
        # weight slice (see VaeBlazeOptions.first_frame_conv2d).
        self.first_frame_conv2d = False
        self._weight_2d: tuple[int, int, torch.Tensor] | None = None
        self._wino_w: tuple[int, int, list[torch.Tensor]] | None = None
        # Spatial Winograd mode for forward_winograd ("off", "auto", "all"), set by the block.
        self.winograd_spatial = "off"
        self.winograd_spatial_kernel = "cublas"
        self.winograd_spatial_tile = 2
        self.winograd_spatial_single_tile = 2
        self.winograd_spatial_first_frame = False
        self._wino_u_nk: tuple[int, int, torch.Tensor] | None = None
        self._wino_u_kf: tuple[int, int, torch.Tensor] | None = None
        self._wino_u: tuple[int, int, torch.Tensor] | None = None
        self._wino_u_taps: dict[int, tuple[int, int, torch.Tensor]] = {}

    # ---- frame buffer ("slab") for the causal history -------------------------------
    # The buffer holds [1, C, S, H, W] channels-last frames; frames [pos-2, pos) are the
    # causal history (zeros before the first chunk), the producer writes the chunk's T
    # frames at [pos, pos+T), the conv reads the window [pos-2, pos+T) with no copy. When
    # the window would run off the end, the two history frames move to the front.
    kSlabChunks = 4

    def slab_reset(self) -> None:
        self._slab = None
        self._slab_pos = 0

    def slab_begin(self, cache_entry, frames: int, like: torch.Tensor) -> torch.Tensor:
        """Start a chunk: reset on the first chunk, adopt a cache tensor left by the eager
        first-chunk path (the frame history), and return the writable window."""
        adopt = cache_entry if isinstance(cache_entry, torch.Tensor) else None
        if cache_entry is None or adopt is not None:
            self.slab_reset()
        window = self.slab_window(frames, like)
        if adopt is not None:
            n = adopt.shape[2]
            self._slab[:, :, 2 - n : 2].copy_(adopt)
        return window

    def slab_window(self, frames: int, like: torch.Tensor) -> torch.Tensor:
        """A writable channels-last ``[1, C, frames, H, W]`` view for the chunk's frames."""
        c = self.in_channels
        h, w = like.shape[-2:]
        slab = getattr(self, "_slab", None)

        def allocate() -> torch.Tensor:
            return torch.empty(
                1,
                c,
                2 + self.kSlabChunks * frames,
                h,
                w,
                dtype=like.dtype,
                device=like.device,
                memory_format=torch.channels_last_3d,
            )

        if (
            slab is None
            or slab.shape[1] != c
            or tuple(slab.shape[3:]) != (h, w)
            or slab.dtype != like.dtype
        ):
            slab = allocate()
            slab[:, :, :2].zero_()
            self._slab, self._slab_pos = slab, 2
        elif self._slab_pos + frames > slab.shape[2]:
            # Wrap: the two history frames move to the front (through a copy, the ranges
            # may overlap); grow the buffer if this chunk is longer than the ones before.
            hist = slab[:, :, self._slab_pos - 2 : self._slab_pos].clone()
            if slab.shape[2] < 2 + frames:
                slab = allocate()
                self._slab = slab
            slab[:, :, :2].copy_(hist)
            self._slab_pos = 2
        return slab[:, :, self._slab_pos : self._slab_pos + frames]

    def slab_conv(self, frames: int) -> torch.Tensor:
        """Run the conv on ``[pos-2, pos+frames)`` and advance; the next two history frames stay in place."""
        slab = self._slab
        pos = self._slab_pos
        bias = None if self.defer_bias else self.bias
        if frames == 1 and self.spatial_eligible(
            slab.shape[3], slab.shape[4], self.winograd_spatial_single_tile
        ):
            y = self._forward_spatial_frames(slab[:, :, pos - 2 : pos + 1], bias)
            self._slab_pos = pos + 1
            return y
        if self.max_in_frames is not None and frames + 2 > self.max_in_frames:
            per_call = self.max_in_frames - 2
            parts = [
                F.conv3d(
                    slab[:, :, pos - 2 + t : pos + min(t + per_call, frames)],
                    self.weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                for t in range(0, frames, per_call)
            ]
            y = torch.cat(parts, dim=2)
        else:
            window = slab[:, :, pos - 2 : pos + frames]
            y = F.conv3d(
                window, self.weight, bias, self.stride, self.padding, self.dilation, self.groups
            )
        self._slab_pos = pos + frames
        return _channels_last_3d_if_needed(y)

    def slab_eligible(self, x_like: torch.Tensor) -> bool:
        return (
            x_like.shape[0] == 1
            and self.kernel_size[0] == 3
            and self.stride == (1, 1, 1)
            and self.dilation == (1, 1, 1)
            and self.groups == 1
        )

    def _winograd_weights(self) -> list[torch.Tensor]:
        key = (self.weight.data_ptr(), self.weight._version)
        if self._wino_w is None or self._wino_w[:2] != key:
            self._wino_w = (*key, winograd_weights(self.weight))
        return self._wino_w[2]

    def winograd_eligible(self, frames: int, batch: int) -> bool:
        return (
            batch == 1
            and frames >= 2
            and frames % 2 == 0
            and self.kernel_size == (3, 3, 3)
            and self.stride == (1, 1, 1)
            and self.dilation == (1, 1, 1)
            and self.groups == 1
        )

    def _winograd_spatial_weights(self) -> torch.Tensor:
        key = (self.weight.data_ptr(), self.weight._version, self.winograd_spatial_tile)
        if self._wino_u is None or self._wino_u[:3] != key:
            self._wino_u = (*key, winograd_spatial_weights(self.weight, self.winograd_spatial_tile))
        return self._wino_u[3]

    def _winograd_spatial_weights_cute(self) -> torch.Tensor:
        key = (self.weight.data_ptr(), self.weight._version)
        if self._wino_u_nk is None or self._wino_u_nk[:2] != key:
            u_nk = winograd_spatial_weights(self.weight, 2).transpose(1, 2).contiguous()
            self._wino_u_nk = (*key, cute_spatial_filters(u_nk))
        return self._wino_u_nk[2]

    def _winograd_spatial_weights_kf(self) -> torch.Tensor:
        key = (self.weight.data_ptr(), self.weight._version)
        if self._wino_u_kf is None or self._wino_u_kf[:2] != key:
            self._wino_u_kf = (*key, kf_spatial_filters(self._winograd_weights()))
        return self._wino_u_kf[2]

    def _use_kf_spatial(self, height: int, width: int) -> bool:
        return (
            self.winograd_spatial_kernel == "kf"
            and kf_wino_available()
            and kf_spatial_eligible(self.in_channels, self.out_channels, width)
        )

    def _use_cute_spatial(self, height: int, width: int) -> bool:
        # The 256-wide two-pass kernel beats the portable path from 45x80 pixels up (its
        # 256-row tiles waste too many rows below that); smaller frames stay on cuBLAS.
        return (
            self.winograd_spatial_kernel == "cute"
            and cute_wino_available()
            and self.in_channels % 64 == 0
            and self.out_channels % 256 == 0
            and height * width >= 45 * 80
        )

    def spatial_eligible(
        self, height: int, width: int, tile: int | None = None, kernel: str = "cublas"
    ) -> bool:
        tile = self.winograd_spatial_tile if tile is None else tile
        if (
            self.winograd_spatial == "off"
            or self.padding[1:] != (1, 1)
            or self.in_channels % 8
            or self.out_channels % 8
        ):
            return False
        if self.winograd_spatial == "all":
            return True
        # Measured winners on GB200: the Kernel Factory F(4x4, 3x3) pipeline on every conv
        # with at least 512 input channels (incl. 512 -> 256 at 360x640); the portable path
        # with F(4x4, 3x3) on every 1024-channel layer and the 512-channel level up to
        # 180x320, with F(2x2, 3x3) on the 1024-channel levels only.
        if kernel == "kf":
            return self.in_channels >= 512 and height * width <= 360 * 640
        if tile == 4:
            return self.in_channels >= 512 and height * width <= 180 * 320
        return self.in_channels >= 1024 and height * width <= 90 * 160

    def _winograd_taps_weights(self, taps: int, tile: int) -> torch.Tensor:
        key = (self.weight.data_ptr(), self.weight._version)
        cached = self._wino_u_taps.get((taps, tile))
        if cached is None or cached[:2] != key:
            cached = (*key, winograd_spatial_taps_weights(self.weight, tile, taps))
            self._wino_u_taps[(taps, tile)] = cached
        return cached[2]

    def _forward_spatial_frames(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """One output frame by spatial Winograd alone: ``x`` holds the taps' input frames
        (``[1, C, 1, H, W]`` with zero history, or ``[1, C, 3, H, W]``: t-2, t-1, t)."""
        _, _, taps, h, w = x.shape
        tile = self.winograd_spatial_tile if taps == 1 else self.winograd_spatial_single_tile
        u = self._winograd_taps_weights(taps, tile)
        y = wino_spatial_conv_frames(frames_view(x), u, h, w, bias)
        return frames_to_video(y.unsqueeze(0), h, w)

    def forward_winograd(self, slabs: WinoSlabs) -> list[torch.Tensor]:
        """The four 3x3 convs of the transformed slabs (bias left to the consumer): cuDNN
        conv2d per slab, or spatial Winograd on all four at once where that is faster."""
        _, _, h, w = slabs.d[0].shape
        if self._use_kf_spatial(h, w) and self.spatial_eligible(h, w, kernel="kf"):
            m = wino_spatial_conv_kf(slabs.stacked, self._winograd_spatial_weights_kf(), h, w)
            return [pairs_to_images(m[k], h, w) for k in range(4)]
        if self.spatial_eligible(h, w):
            if self._use_cute_spatial(h, w):
                m = wino_spatial_conv_cute(
                    slabs.stacked, self._winograd_spatial_weights_cute(), h, w
                )
            else:
                m = wino_spatial_conv_cuda(slabs.stacked, self._winograd_spatial_weights(), h, w)
            return [pairs_to_images(m[k], h, w) for k in range(4)]
        g = self._winograd_weights()
        pad = (self.padding[1], self.padding[2])
        return [F.conv2d(slabs.d[k], g[k], None, 1, pad) for k in range(4)]

    def _current_frame_weight(self) -> torch.Tensor:
        key = (self.weight.data_ptr(), self.weight._version)
        if self._weight_2d is None or self._weight_2d[:2] != key:
            w2 = self.weight[:, :, -1].contiguous(memory_format=torch.channels_last)
            self._weight_2d = (*key, w2)
        return self._weight_2d[2]

    def forward(
        self,
        x: torch.Tensor,
        cache_x: torch.Tensor | None = None,
        *,
        spatial_padding: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        x = _channels_last_3d_if_needed(x)
        bias = None if self.defer_bias else self.bias
        if (
            self.first_frame_conv2d
            and cache_x is None
            and self._padding[4] > 0
            and x.shape[2] == 1
            and self.stride[0] == 1
            and spatial_padding is None
        ):
            # Zero history: only the current-frame taps contribute.
            if self.winograd_spatial_first_frame and self.spatial_eligible(x.shape[3], x.shape[4]):
                return self._forward_spatial_frames(x, bias)
            y = F.conv2d(
                x[:, :, 0],
                self._current_frame_weight(),
                bias,
                self.stride[1:],
                self.padding[1:],
                self.dilation[1:],
                self.groups,
            )
            return _channels_last_3d_if_needed(y.unsqueeze(2))
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = _to_device_if_needed(cache_x, x.device)
            cache_x = _channels_last_3d_if_needed(cache_x)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        if any(padding):
            x = F.pad(x, padding)
        x = _channels_last_3d_if_needed(x)
        padding = self.padding if spatial_padding is None else (0, *spatial_padding)
        kt = self.kernel_size[0]
        if (
            kt == 3
            and x.shape[2] == 3
            and x.shape[0] == 1
            and self.stride == (1, 1, 1)
            and self.dilation == (1, 1, 1)
            and self.groups == 1
            and spatial_padding is None
            and self.spatial_eligible(x.shape[3], x.shape[4], self.winograd_spatial_single_tile)
        ):
            return self._forward_spatial_frames(x, bias)
        if (
            self.max_in_frames is not None
            and self.stride[0] == 1
            and x.shape[2] > self.max_in_frames >= kt
        ):
            per_call = self.max_in_frames - kt + 1
            total = x.shape[2] - kt + 1
            parts = [
                F.conv3d(
                    x[:, :, t : t + per_call + kt - 1],
                    self.weight,
                    bias,
                    self.stride,
                    padding,
                    self.dilation,
                    self.groups,
                )
                for t in range(0, total, per_call)
            ]
            return _channels_last_3d_if_needed(torch.cat(parts, dim=2))
        x = F.conv3d(x, self.weight, bias, self.stride, padding, self.dilation, self.groups)
        return _channels_last_3d_if_needed(x)


class BlazeResample(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        options: VaeBlazeOptions | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.options = options or VaeBlazeOptions()
        self._packed: tuple[int, int, torch.Tensor, torch.Tensor | None] | None = None
        # Downsample conv bias left out of the conv (fused_bias): the time conv absorbs it
        # (downsample3d) or the block's shortcut-add kernel adds it (downsample2d).
        self.pending_bias: torch.Tensor | None = None
        self._time_bias: tuple[int, int, int, int, torch.Tensor] | None = None
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanConv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanConv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = BlazeCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), WanConv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), WanConv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = BlazeCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))
        else:
            self.resample = nn.Identity()

    def _packed_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv = self.resample[1]
        key = (conv.weight.data_ptr(), conv.weight._version)
        if self._packed is None or self._packed[:2] != key:
            packed, packed_bias = pack_upsample_weights(conv.weight, conv.bias)
            self._packed = (key[0], key[1], packed, packed_bias)
        return self._packed[2], self._packed[3]

    def _fused_upsample(
        self,
        x_4d: torch.Tensor,
        batch_size: int,
        frames: int,
        shortcut: tuple[torch.Tensor, "DupUp3D", bool] | None,
    ) -> torch.Tensor:
        """2x2 conv on the low-res grid, then the phase gather (+ DupUp3D shortcut)."""
        packed, packed_bias = self._packed_weights()
        cout = packed.shape[0] // 4
        use_cuda = x_4d.dtype in (torch.bfloat16, torch.float16) and (
            shortcut is None
            and can_use_cuda_upsample_gather(cout, 1, 1)
            or shortcut is not None
            and can_use_cuda_upsample_gather(cout, shortcut[1].factor_t, shortcut[1].repeats)
        )
        defer_bias = use_cuda and self.options.fused_bias and packed_bias is not None
        temp = F.conv2d(x_4d, packed, None if defer_bias else packed_bias, padding=1)
        temp = _channels_last_2d_if_needed(temp)
        if use_cuda:
            gather_bias = self.resample[1].bias if defer_bias else None
            if shortcut is None:
                return upsample_gather_cuda(temp, None, batch_size, frames, bias=gather_bias)
            residual, dup, first_chunk = shortcut
            residual = _channels_last_3d_if_needed(residual)
            crop = dup.factor_t - 1 if first_chunk else 0
            return upsample_gather_cuda(
                temp,
                residual,
                batch_size,
                frames,
                dup.factor_t,
                dup.repeats,
                crop,
                bias=gather_bias,
            )
        expanded = None
        if shortcut is not None:
            residual, dup, first_chunk = shortcut
            expanded = dup(residual, first_chunk=first_chunk)
        return upsample_gather_reference(temp, batch_size, frames, expanded)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        shortcut: tuple[torch.Tensor, "DupUp3D", bool] | None = None,
    ) -> torch.Tensor:
        """``shortcut`` = (residual, DupUp3D module, first_chunk): when given, the fused
        upsample adds the expanded shortcut while writing and the caller skips its add."""
        batch_size, channels, frames, height, width = x.size()
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")
        if shortcut is not None and not (
            self.options.fused_upsample and self.mode.startswith("upsample")
        ):
            raise ValueError("shortcut is only consumed by the fused upsample path")

        if self.mode == "upsample3d" and feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :]
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] != "Rep"
                ):
                    cache_x = torch.cat(
                        [
                            _to_device_if_needed(
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                            ),
                            cache_x,
                        ],
                        dim=2,
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                fuse_interleave = (
                    cuda_frame_ops_available()
                    and batch_size == 1
                    and x.dtype == torch.bfloat16
                    and x.is_cuda
                )
                self.time_conv.defer_bias = fuse_interleave
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                if fuse_interleave:
                    # Interleave the two channel halves along time and add the time conv's bias.
                    u = frames_view(_channels_last_3d_if_needed(x))
                    x = frames_to_video(interleave_bias_cuda(u, self.time_conv.bias), height, width)
                else:
                    x = x.reshape(batch_size, 2, channels, frames, height, width)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(batch_size, channels, frames * 2, height, width)

        frames = x.shape[2]
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = _channels_last_2d_if_needed(x_4d)
        if self.options.fused_upsample and self.mode.startswith("upsample"):
            return self._fused_upsample(x, batch_size, frames, shortcut)
        if self.mode.startswith("downsample"):
            x = self._downsample_conv(self.resample[0](x), batch_size)
        else:
            x = self.resample(x)
        return self._after_resample(x, batch_size, frames, feat_cache, feat_idx)

    def _defer_downsample_bias(self, x: torch.Tensor, batch_size: int) -> bool:
        conv = self.resample[1]
        return (
            self.options.fused_bias
            and self.options.norm_mode != "reference"
            and conv.bias is not None
            and cuda_frame_ops_available()
            and batch_size == 1
            and x.is_cuda
            and x.dtype == torch.bfloat16
        )

    def _downsample_conv(self, x_4d: torch.Tensor, batch_size: int) -> torch.Tensor:
        """The stride-2 3x3 conv of the downsample modes; with a deferred bias the conv runs
        bias-free and the bias is left in ``pending_bias`` for the consumer."""
        conv = self.resample[1]
        if self._defer_downsample_bias(x_4d, batch_size):
            self.pending_bias = conv.bias
            return F.conv2d(x_4d, conv.weight, None, conv.stride, conv.padding)
        self.pending_bias = None
        return conv(x_4d)

    def take_pending_bias(self) -> torch.Tensor | None:
        bias, self.pending_bias = self.pending_bias, None
        return bias

    def _time_conv_bias(self, bias: torch.Tensor) -> torch.Tensor:
        """``time_conv``'s bias with the downsample conv's bias folded through its taps:
        every input frame of the stride-2 time conv carries ``bias``, so
        ``b' = b_t + sum_k W_t[:, :, k] @ bias`` (composed in fp32, rounded once)."""
        tc = self.time_conv
        key = (tc.weight.data_ptr(), tc.weight._version, bias.data_ptr(), bias._version)
        if self._time_bias is None or self._time_bias[:4] != key:
            with torch.inference_mode(False), torch.no_grad():
                w = tc.weight.detach().float()[:, :, :, 0, 0]  # [Cout, Cin, taps]
                b = bias.detach().float()
                bt = tc.bias.detach().float() if tc.bias is not None else torch.zeros_like(b)
                folded = (bt + torch.einsum("oik,i->o", w, b)).to(tc.weight.dtype)
            self._time_bias = (*key, folded)
        return self._time_bias[4]

    def forward_prepadded(
        self,
        x_4d: torch.Tensor,
        frames: int,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Downsample path on an input that already carries the zero row/column the
        ``ZeroPad2d`` would add; ``x_4d`` is ``[T, C, H+1, W+1]`` channels-last, batch 1."""
        if not self.mode.startswith("downsample"):
            raise ValueError("forward_prepadded is for the downsample modes")
        x = self._downsample_conv(_channels_last_2d_if_needed(x_4d), 1)
        return self._after_resample(x, 1, frames, feat_cache, feat_idx)

    def _after_resample(
        self,
        x: torch.Tensor,
        batch_size: int,
        frames: int,
        feat_cache: list[torch.Tensor | str | None] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        x_5d = x.reshape(batch_size, frames, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        x = _channels_last_3d_if_needed(x_5d)

        if self.mode == "downsample3d" and feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :]
                x_in = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2)
                bias = self.take_pending_bias()
                if bias is not None:
                    tc = self.time_conv
                    x = F.conv3d(x_in, tc.weight, self._time_conv_bias(bias), tc.stride)
                else:
                    x = self.time_conv(x_in)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class BlazeResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        options: VaeBlazeOptions | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.options = options or VaeBlazeOptions()
        self.nonlinearity = _activation(non_linearity)
        self.norm1 = BlazeRMSNorm(in_dim, images=False, options=self.options)
        self.conv1 = BlazeCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = BlazeRMSNorm(out_dim, images=False, options=self.options)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = BlazeCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            BlazeCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )
        self._merged_bias: tuple[int, int, torch.Tensor] | None = None
        # Spatial Winograd applies to both convs on every path (frame pairs, cached single
        # frames, the first frame), so the mode is fixed here, not when a path first runs.
        spatial = self.options.winograd_spatial if self.options.use_winograd_spatial else "off"
        for conv in (self.conv1, self.conv2):
            conv.winograd_spatial = spatial
            conv.winograd_spatial_kernel = self.options.winograd_spatial_kernel
            conv.winograd_spatial_tile = self.options.winograd_spatial_tile
            conv.winograd_spatial_single_tile = self.options.winograd_spatial_single_tile
            conv.winograd_spatial_first_frame = self.options.winograd_spatial_first_frame

    def _output_bias(self, with_shortcut: bool) -> torch.Tensor:
        """conv2's bias, plus the shortcut conv's when that one was deferred too."""
        if not with_shortcut:
            return self.conv2.bias
        b2, bs = self.conv2.bias, self.conv_shortcut.bias
        key = (b2._version, bs._version)
        if self._merged_bias is None or self._merged_bias[:2] != key:
            self._merged_bias = (*key, (b2.detach().float() + bs.detach().float()).to(b2.dtype))
        return self._merged_bias[2]

    def forward(
        self,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        *,
        defer_output_bias: bool = False,
        return_input: bool = False,
    ) -> "torch.Tensor | PendingAdd | tuple[torch.Tensor | PendingAdd, torch.Tensor]":
        """``defer_output_bias``: the consumer is a fused norm, so conv2's bias (and in fp32
        mode the shortcut conv's) travel on the returned PendingAdd instead of being
        added here. ``return_input``: also return the materialized block input."""
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")
        defer = self.options.defer_conv_bias
        defer_out = defer and defer_output_bias and self.options.fused_residual

        if isinstance(x, PendingAdd):
            frames = x.frames
            batch = 1 if x.wino is not None else x.a.shape[0]
        else:
            frames, batch = x.shape[2], x.shape[0]
        if (
            self.options.use_winograd
            and feat_cache is not None
            and feat_idx is not None
            and self.conv1.winograd_eligible(frames, batch)
            and self.conv2.winograd_eligible(frames, batch)
        ):
            return self._forward_winograd(x, feat_cache, feat_idx, frames, return_input)

        if (
            self.options.use_slabs
            and feat_cache is not None
            and feat_idx is not None
            and self.conv1.slab_eligible(_pending_like(x))
            and self.conv2.slab_eligible(_pending_like(x))
            and not (
                self.conv1.first_frame_conv2d and frames == 1 and feat_cache[feat_idx[0]] is None
            )
        ):
            return self._forward_slabs(
                x, feat_cache, feat_idx, frames, defer, defer_out, return_input
            )

        if isinstance(x, PendingAdd):
            x, h = self.norm1.forward_silu_pending(x)
        else:
            h = self.norm1.forward_silu(x)
        has_shortcut_conv = isinstance(self.conv_shortcut, BlazeCausalConv3d)
        defer_shortcut = defer_out and has_shortcut_conv and self.options.norm_mode != "reference"
        if has_shortcut_conv:
            self.conv_shortcut.defer_bias = defer_shortcut
        residual = self.conv_shortcut(x)

        self.conv1.defer_bias = defer
        if feat_cache is not None and feat_idx is not None:
            h = _causal_conv_with_cache(self.conv1, h, feat_cache, feat_idx)
        else:
            h = self.conv1(h)
        h = self.norm2.forward_silu(PendingAdd(h, None, self.conv1.bias) if defer else h)
        h = self.dropout(h)

        self.conv2.defer_bias = defer_out
        if feat_cache is not None and feat_idx is not None:
            h = _causal_conv_with_cache(self.conv2, h, feat_cache, feat_idx)
        else:
            h = self.conv2(h)
        if self.options.fused_residual:
            out = PendingAdd(h, residual, self._output_bias(defer_shortcut) if defer_out else None)
        else:
            out = _channels_last_3d_if_needed(h + residual)
        return (out, x) if return_input else out

    def _forward_slabs(
        self,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[torch.Tensor | str | None],
        feat_idx: list[int],
        frames: int,
        defer: bool,
        defer_out: bool,
        return_input: bool,
    ) -> "torch.Tensor | PendingAdd | tuple[torch.Tensor | PendingAdd, torch.Tensor]":
        """Plain 3D convs reading their frame buffers; the norm kernels write into them."""
        like = _pending_like(x)
        idx1 = feat_idx[0]
        win1 = self.conv1.slab_begin(feat_cache[idx1], frames, like)
        feat_cache[idx1] = self.conv1
        feat_idx[0] += 1
        if isinstance(x, PendingAdd):
            x, _ = self.norm1.forward_silu_pending(x, out=win1)
        else:
            self.norm1.forward_silu(x, out=win1)
        has_shortcut_conv = isinstance(self.conv_shortcut, BlazeCausalConv3d)
        defer_shortcut = defer_out and has_shortcut_conv and self.options.norm_mode != "reference"
        if has_shortcut_conv:
            self.conv_shortcut.defer_bias = defer_shortcut
        residual = self.conv_shortcut(x)
        self.conv1.defer_bias = defer
        h = self.conv1.slab_conv(frames)

        idx2 = feat_idx[0]
        win2 = self.conv2.slab_begin(feat_cache[idx2], frames, h)
        feat_cache[idx2] = self.conv2
        feat_idx[0] += 1
        self.norm2.forward_silu(PendingAdd(h, None, self.conv1.bias) if defer else h, out=win2)
        self.conv2.defer_bias = defer_out
        h = self.conv2.slab_conv(frames)
        if self.options.fused_residual:
            out = PendingAdd(h, residual, self._output_bias(defer_shortcut) if defer_out else None)
        else:
            out = _channels_last_3d_if_needed(h + residual)
        return (out, x) if return_input else out

    def _forward_winograd(
        self,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[torch.Tensor | str | None],
        feat_idx: list[int],
        frames: int,
        return_input: bool,
    ) -> "PendingAdd | tuple[PendingAdd, torch.Tensor]":
        """Both convs as temporal Winograd. The caches hold the last two normalized
        frames exactly as the eager path stores them; conv biases ride on the
        PendingAdd and are added by the consumer kernel (free there)."""
        # norm1 -> slabs for conv1; the sum it writes is the block input (residual source).
        idx = feat_idx[0]
        s, cache1, slabs = self.norm1.forward_silu_wino(x, feat_cache[idx], frames)
        feat_cache[idx] = cache1
        feat_idx[0] += 1
        if isinstance(x, PendingAdd):
            if s is None:  # plain input without bias or residual
                s = x.materialize()
            x_in = s
        else:
            x_in = x
        has_shortcut_conv = isinstance(self.conv_shortcut, BlazeCausalConv3d)
        defer_shortcut = has_shortcut_conv and self.options.norm_mode != "reference"
        if has_shortcut_conv:
            self.conv_shortcut.defer_bias = defer_shortcut
        residual = self.conv_shortcut(x_in)
        m1 = self.conv1.forward_winograd(slabs)

        idx = feat_idx[0]
        _, cache2, slabs2 = self.norm2.forward_silu_wino(
            PendingAdd(None, None, self.conv1.bias, wino=m1), feat_cache[idx], frames
        )
        feat_cache[idx] = cache2
        feat_idx[0] += 1
        m2 = self.conv2.forward_winograd(slabs2)
        out = PendingAdd(None, residual, self._output_bias(defer_shortcut), wino=m2)
        return (out, x_in) if return_input else out


class BlazeAttentionBlock(WanAttentionBlock):
    """The mid-block attention with the norm, projections and residual add fused."""

    def __init__(self, dim: int, options: VaeBlazeOptions | None = None) -> None:
        super().__init__(dim)
        self.options = options or VaeBlazeOptions()

    def forward(self, x: "torch.Tensor | PendingAdd") -> "torch.Tensor | PendingAdd":  # type: ignore[override]
        x = _tensor(x)
        if not (
            self.options.use_attention
            and x.shape[0] == 1
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and can_use_cuda_rms_norm_silu(x)
        ):
            return super().forward(x)
        residual = _channels_last_3d_if_needed(x)
        _, channels, frames, height, width = residual.shape
        positions = height * width
        # The GEMMs run on the channels-last row view [T*H*W, C], which is also SDPA's layout.
        normed = rms_norm_silu_cuda(
            residual,
            self.norm.gamma,
            None,
            self.norm.scale,
            reference_rounding=self.options.norm_mode == "reference",
            silu=False,
        )
        normed_rows = frames_view(normed).reshape(frames * positions, channels)
        w_qkv = self.to_qkv.weight.reshape(3 * channels, channels)
        qkv = torch.addmm(self.to_qkv.bias, normed_rows, w_qkv.t())  # bias in the GEMM epilogue
        qkv = qkv.view(frames, 1, positions, 3 * channels)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(q, k, v)  # [T, 1, HW, C]
        attn_rows = attn.reshape(frames * positions, channels)
        w_proj = self.proj.weight.reshape(channels, channels)
        proj_rows = torch.mm(attn_rows, w_proj.t())
        proj = frames_to_video(proj_rows.view(frames, positions, channels), height, width)
        # x + residual (+ the proj bias) is added by the consumer's norm kernel.
        return PendingAdd(proj, residual, self.proj.bias)


def _run_resnets(
    resnets: nn.ModuleList,
    x: "torch.Tensor | PendingAdd",
    feat_cache: list[torch.Tensor | str | None] | None,
    feat_idx: list[int] | None,
    *,
    defer_last_bias: bool,
    defer_first_bias: bool = True,
) -> tuple["torch.Tensor | PendingAdd", torch.Tensor]:
    """Run a chain of residual blocks; returns ``(output, materialized input)``.

    The first block consumes a PendingAdd input inside its norm kernel and hands back
    the sum, so the caller gets the block input without a separate add. Every block
    but the last defers its conv2 bias to the next block's norm; the last one does so
    only when ``defer_last_bias`` (its consumer is a fused norm)."""
    last = len(resnets) - 1
    entry: torch.Tensor | None = None
    for i, resnet in enumerate(resnets):
        defer = (defer_last_bias if i == last else True) and (defer_first_bias or i > 0)
        if i == 0:
            x, entry = resnet(
                x,
                feat_cache=feat_cache,
                feat_idx=feat_idx,
                defer_output_bias=defer,
                return_input=True,
            )
        else:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx, defer_output_bias=defer)
    assert entry is not None
    return x, entry


class BlazeMidBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
        options: VaeBlazeOptions | None = None,
    ):
        super().__init__()
        resnets = [BlazeResidualBlock(dim, dim, dropout, non_linearity, options)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(BlazeAttentionBlock(dim, options))
            resnets.append(BlazeResidualBlock(dim, dim, dropout, non_linearity, options))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        # The fused attention block returns its residual add as a PendingAdd for the next
        # norm; the plain one needs a materialized tensor.
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx, defer_output_bias=False)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx, defer_output_bias=True)
        return x


class BlazeResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
        options: VaeBlazeOptions | None = None,
    ):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(BlazeResidualBlock(in_dim, out_dim, dropout, options=options))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = None
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = BlazeResample(out_dim, mode=mode, options=options)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        # The last block feeds the downsampler or the shortcut add, never a norm.
        x, residual = _run_resnets(self.resnets, x, feat_cache, feat_idx, defer_last_bias=False)
        if self.downsampler is not None:
            if isinstance(x, PendingAdd) and x.can_fuse():
                # One pass writes the downsampler's zero-padded input directly.
                x = self.downsampler.forward_prepadded(
                    x.materialize_padded_2d(), x.frames, feat_cache=feat_cache, feat_idx=feat_idx
                )
            else:
                x = self.downsampler(_tensor(x), feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            x = _tensor(x)
        x = _channels_last_3d_if_needed(x)
        # The downsample conv's bias, when it ran without one (downsample2d; the time conv
        # absorbs it in downsample3d).
        bias = self.downsampler.take_pending_bias() if self.downsampler is not None else None
        dup = self.avg_shortcut
        if (
            cuda_frame_ops_available()
            and x.shape[0] == 1
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and residual.dtype == torch.bfloat16
        ):
            residual = _channels_last_3d_if_needed(residual)
            t_in = residual.shape[2]
            pad_t = (dup.factor_t - t_in % dup.factor_t) % dup.factor_t
            h_out, w_out = x.shape[-2:]
            y = avg_down_add_cuda(
                frames_view(x),
                frames_view(residual),
                h_out,
                w_out,
                residual.shape[-1],
                dup.factor_t,
                dup.factor_s,
                dup.group_size,
                pad_t,
                bias,
            )
            return frames_to_video(y, h_out, w_out)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1, 1)
        return _channels_last_3d_if_needed(x + dup(residual))


class BlazeEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
        options: VaeBlazeOptions | None = None,
    ):
        super().__init__()
        self.options = options or VaeBlazeOptions()
        dim_mult = [1, 2, 4, 4] if dim_mult is None else dim_mult
        attn_scales = [] if attn_scales is None else attn_scales
        temperal_downsample = (
            [False, True, True] if temperal_downsample is None else temperal_downsample
        )
        self.nonlinearity = _activation(non_linearity)

        dims = [dim * value for value in [1] + dim_mult]
        scale = 1.0
        self.conv_in = BlazeCausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                self.down_blocks.append(
                    BlazeResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i]
                        if i != len(dim_mult) - 1
                        else False,
                        down_flag=i != len(dim_mult) - 1,
                        options=options,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(
                        BlazeResidualBlock(in_dim, out_dim, dropout, options=options)
                    )
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(BlazeResample(out_dim, mode=mode, options=options))
                    scale /= 2.0

        self.mid_block = BlazeMidBlock(
            out_dim, dropout, non_linearity, num_layers=1, options=options
        )
        self.norm_out = BlazeRMSNorm(out_dim, images=False, options=options)
        self.conv_out = BlazeCausalConv3d(out_dim, z_dim, 3, padding=1)
        # A replacement for conv_out for the duration of one encode (the folded encoder
        # tail); not a submodule, so the state dict is untouched.
        self.conv_out_override: BlazeCausalConv3d | None = None
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        self.conv_in.defer_bias = self.options.defer_conv_bias
        if feat_cache is not None and feat_idx is not None:
            x = _causal_conv_with_cache(self.conv_in, x, feat_cache, feat_idx)
        else:
            x = self.conv_in(x)
        if self.conv_in.defer_bias:
            x = PendingAdd(x, None, self.conv_in.bias)

        for layer in self.down_blocks:
            if not isinstance(layer, (BlazeResidualBlock, BlazeResidualDownBlock)):
                x = _tensor(x)
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        conv_out = self.conv_out_override or self.conv_out
        frames = x.frames if isinstance(x, PendingAdd) else x.shape[2]
        like = _pending_like(x)
        if (
            self.options.use_slabs
            and feat_cache is not None
            and feat_idx is not None
            and like.shape[0] == 1
            and conv_out.slab_eligible(like)
            and not (
                conv_out.first_frame_conv2d and frames == 1 and feat_cache[feat_idx[0]] is None
            )
        ):
            idx = feat_idx[0]
            window = conv_out.slab_begin(feat_cache[idx], frames, like)
            feat_cache[idx] = conv_out
            feat_idx[0] += 1
            self.norm_out.forward_silu(x, out=window)
            return conv_out.slab_conv(frames)
        x = self.norm_out.forward_silu(x)
        if feat_cache is not None and feat_idx is not None:
            x = _causal_conv_with_cache(conv_out, x, feat_cache, feat_idx)
        else:
            x = conv_out(x)
        return _channels_last_3d_if_needed(x)


class BlazeResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
        options: VaeBlazeOptions | None = None,
    ):
        super().__init__()
        self.avg_shortcut = None
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2
            )

        current_dim = in_dim
        resnets = []
        for _ in range(num_res_blocks + 1):
            resnets.append(
                BlazeResidualBlock(current_dim, out_dim, dropout, non_linearity, options)
            )
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        self.upsampler = None
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = BlazeResample(
                out_dim, mode=mode, upsample_out_dim=out_dim, options=options
            )
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        # With an upsampler the last block feeds a conv, so its bias stays in eager; the
        # final up block (no upsampler) hands its output to norm_out.
        x, residual = _run_resnets(
            self.resnets, x, feat_cache, feat_idx, defer_last_bias=self.upsampler is None
        )
        if self.upsampler is not None:
            fuse_shortcut = self.avg_shortcut is not None and self.upsampler.options.fused_upsample
            x = self.upsampler(
                _tensor(x),
                feat_cache=feat_cache,
                feat_idx=feat_idx,
                shortcut=(residual, self.avg_shortcut, first_chunk) if fuse_shortcut else None,
            )
            if fuse_shortcut:
                return x
        if self.avg_shortcut is not None:
            x = _tensor(x) + self.avg_shortcut(residual, first_chunk=first_chunk)
        return x if isinstance(x, PendingAdd) else _channels_last_3d_if_needed(x)


class BlazeUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
        options: VaeBlazeOptions | None = None,
    ):
        super().__init__()
        current_dim = in_dim
        resnets = []
        for _ in range(num_res_blocks + 1):
            resnets.append(
                BlazeResidualBlock(current_dim, out_dim, dropout, non_linearity, options)
            )
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList(
                [BlazeResample(out_dim, mode=upsample_mode, options=options)]
            )
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool | None = None,
    ) -> torch.Tensor:
        del first_chunk
        x, _ = _run_resnets(
            self.resnets, x, feat_cache, feat_idx, defer_last_bias=self.upsamplers is None
        )
        if self.upsamplers is not None:
            x = self.upsamplers[0](_tensor(x), feat_cache=feat_cache, feat_idx=feat_idx)
        return x if isinstance(x, PendingAdd) else _channels_last_3d_if_needed(x)


class BlazeDecoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
        options: VaeBlazeOptions | None = None,
    ):
        super().__init__()
        self.options = options or VaeBlazeOptions()
        dim_mult = [1, 2, 4, 4] if dim_mult is None else dim_mult
        del attn_scales
        temperal_upsample = [False, True, True] if temperal_upsample is None else temperal_upsample
        self.nonlinearity = _activation(non_linearity)

        dims = [dim * value for value in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = BlazeCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = BlazeMidBlock(
            dims[0], dropout, non_linearity, num_layers=1, options=options
        )
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and not is_residual:
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            if is_residual:
                up_block = BlazeResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                    options=options,
                )
            else:
                up_block = BlazeUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                    options=options,
                )
            self.up_blocks.append(up_block)

        self.norm_out = BlazeRMSNorm(out_dim, images=False, options=options)
        self.conv_out = BlazeCausalConv3d(out_dim, out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        self.conv_in.defer_bias = self.options.defer_conv_bias
        if feat_cache is not None and feat_idx is not None:
            x = _causal_conv_with_cache(self.conv_in, x, feat_cache, feat_idx)
        else:
            x = self.conv_in(x)
        if self.conv_in.defer_bias:
            x = PendingAdd(x, None, self.conv_in.bias)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        frames = x.frames if isinstance(x, PendingAdd) else x.shape[2]
        like = _pending_like(x)
        if (
            self.options.use_slabs
            and feat_cache is not None
            and feat_idx is not None
            and like.shape[0] == 1
            and self.conv_out.slab_eligible(like)
            and not (
                self.conv_out.first_frame_conv2d and frames == 1 and feat_cache[feat_idx[0]] is None
            )
        ):
            idx = feat_idx[0]
            window = self.conv_out.slab_begin(feat_cache[idx], frames, like)
            feat_cache[idx] = self.conv_out
            feat_idx[0] += 1
            self.norm_out.forward_silu(x, out=window)
            return self.conv_out.slab_conv(frames)
        x = self.norm_out.forward_silu(x)
        if feat_cache is not None and feat_idx is not None:
            x = _causal_conv_with_cache(self.conv_out, x, feat_cache, feat_idx)
        else:
            x = self.conv_out(x)
        return _channels_last_3d_if_needed(x)


class LazyDiagonalGaussianDistribution(DiagonalGaussianDistribution):
    """``DiagonalGaussianDistribution`` whose ``std``/``var``/clamped ``logvar`` are computed
    on first use: ``mode()`` launches nothing beyond the view split."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self._logvar_raw = torch.chunk(parameters, 2, dim=1)
        self.deterministic = deterministic
        self._logvar: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._var: torch.Tensor | None = None

    @property
    def logvar(self) -> torch.Tensor:
        if self._logvar is None:
            self._logvar = torch.clamp(self._logvar_raw, -30.0, 20.0)
        return self._logvar

    @property
    def std(self) -> torch.Tensor:
        if self._std is None:
            self._std = (
                torch.zeros_like(self.mean) if self.deterministic else torch.exp(0.5 * self.logvar)
            )
        return self._std

    @property
    def var(self) -> torch.Tensor:
        if self._var is None:
            self._var = (
                torch.zeros_like(self.mean) if self.deterministic else torch.exp(self.logvar)
            )
        return self._var


class VaeBlaze(nn.Module):
    """Optimized Wan VAE; state-dict compatible with ``WanVAE`` and ``AutoencoderKLWan``.

    Beyond the ``AutoencoderKLWan`` interface: ``encode_mode(x)`` returns the posterior mean
    directly, and ``fold_latent_affine(mean, std)`` folds the latent de-normalization into
    ``post_quant_conv`` so ``decode`` takes the model's raw latents.
    """

    def __init__(
        self,
        config: WanVAEConfig | dict[str, Any] | None = None,
        options: VaeBlazeOptions | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = WanVAEConfig()
        if isinstance(config, dict):
            config = WanVAEConfig.from_dict(config)
        if options is None:
            options = VaeBlazeOptions()
        if isinstance(options, dict):
            options = VaeBlazeOptions.from_dict(options)
        self.config = config
        self.options = options
        self.z_dim = config.z_dim
        self.temperal_downsample = config.temperal_downsample
        self.temperal_upsample = config.temperal_downsample[::-1]

        self.encoder = BlazeEncoder3d(
            in_channels=config.in_channels,
            dim=config.base_dim,
            z_dim=config.z_dim * 2,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_downsample=config.temperal_downsample,
            dropout=config.dropout,
            is_residual=config.is_residual,
            options=options,
        )
        self.quant_conv = BlazeCausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = BlazeCausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = BlazeDecoder3d(
            dim=config.decoder_base_dim,
            z_dim=config.z_dim,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=config.dropout,
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            options=options,
        )

        for module in self.modules():
            if module is not self:
                _to_channels_last(module)
        for module in self.modules():
            if isinstance(module, BlazeCausalConv3d):
                module.first_frame_conv2d = options.first_frame_conv2d
        self._cached_conv_counts = {
            "decoder": sum(
                isinstance(module, BlazeCausalConv3d) for module in self.decoder.modules()
            ),
            "encoder": sum(
                isinstance(module, BlazeCausalConv3d) for module in self.encoder.modules()
            ),
        }
        self._folded_tail: dict[int | None, tuple[tuple, BlazeCausalConv3d]] = {}
        self._post_quant_folded: tuple[torch.Tensor, torch.Tensor] | None = None
        self.clear_cache()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def clear_cache(self) -> None:
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map: list[torch.Tensor | str | None] = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[torch.Tensor | str | None] = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor, *, mode_only: bool = False) -> torch.Tensor:
        prev_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self.options.benchmark_convs(
            encoder=True, frames=x.shape[2]
        )
        try:
            return self._encode_impl(x, mode_only=mode_only)
        finally:
            torch.backends.cudnn.benchmark = prev_benchmark

    def _use_folded_tail(self) -> bool:
        return self.options.fold_encoder_tail and self.options.norm_mode != "reference"

    def _folded_tail_conv(self, rows: int | None) -> BlazeCausalConv3d:
        """conv_out with quant_conv folded in: ``W' = Q A``, ``b' = Q a + q`` (composed in
        fp32, rounded once to the model dtype); ``rows`` keeps only the first output
        channels (the posterior mean) or all of them."""
        conv_out, quant = self.encoder.conv_out, self.quant_conv
        key = (
            conv_out.weight.data_ptr(),
            conv_out.weight._version,
            quant.weight.data_ptr(),
            quant.weight._version,
            rows,
        )
        cached = self._folded_tail.get(rows)
        if cached is not None and cached[0] == key:
            return cached[1]
        # Built outside inference mode even when called from it: the folded conv's weight
        # caches key on the tensor version counter, which inference tensors do not carry.
        with torch.inference_mode(False), torch.no_grad():
            folded = self._build_folded_tail(conv_out, quant, rows)
        self._folded_tail[rows] = (key, folded)
        return folded

    @staticmethod
    def _build_folded_tail(
        conv_out: BlazeCausalConv3d, quant: BlazeCausalConv3d, rows: int | None
    ) -> BlazeCausalConv3d:
        q = quant.weight.detach().float().flatten(1)  # [2z, 2z]
        qb = quant.bias.detach().float() if quant.bias is not None else torch.zeros_like(q[:, 0])
        a = conv_out.weight.detach().float()  # [2z, Cin, 3, 3, 3]
        ab = (
            conv_out.bias.detach().float()
            if conv_out.bias is not None
            else torch.zeros_like(q[:, 0])
        )
        if rows is not None:
            q, qb = q[:rows], qb[:rows]
        w = torch.einsum("om,mikhw->oikhw", q, a)
        b = q @ ab + qb
        folded = BlazeCausalConv3d(
            conv_out.in_channels, w.shape[0], conv_out.kernel_size, padding=conv_out.padding[1]
        ).to(device=a.device, dtype=conv_out.weight.dtype)
        folded.weight.copy_(w)
        folded.bias.copy_(b)
        folded.weight.requires_grad_(False)
        folded.bias.requires_grad_(False)
        folded.first_frame_conv2d = conv_out.first_frame_conv2d
        folded.max_in_frames = conv_out.max_in_frames
        return folded

    def _encode_impl(self, x: torch.Tensor, *, mode_only: bool = False) -> torch.Tensor:
        _, _, num_frame, _, _ = x.shape

        self.clear_cache()
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
        x = _channels_last_3d_if_needed(x)

        folded = self._use_folded_tail()
        if mode_only and not folded:
            raise ValueError("encode_mode needs fold_encoder_tail (and norm_mode != 'reference')")
        self.encoder.conv_out_override = (
            self._folded_tail_conv(self.z_dim if mode_only else None) if folded else None
        )
        # The first pass is one frame; every later pass is a multiple of four frames so the
        # stride-2 temporal downsamplers see chunk boundaries at the same parity each time.
        step = self.options.encoder_chunk_frames
        bounds = [(0, 1)] + [
            (start, min(start + step, num_frame)) for start in range(1, num_frame, step)
        ]
        out_chunks: list[torch.Tensor] = []
        try:
            for start, end in bounds:
                self._enc_conv_idx = [0]
                out_chunks.append(
                    self.encoder(
                        x[:, :, start:end, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                )
        finally:
            self.encoder.conv_out_override = None

        if len(out_chunks) == 1:
            out = out_chunks[0]
        else:
            out = torch.cat(out_chunks, dim=2)

        enc = out if folded else self.quant_conv(out)
        enc = _channels_last_3d_if_needed(enc)
        self.clear_cache()
        return enc

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        h = self._encode(x)
        posterior = LazyDiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def encode_mode(self, x: torch.Tensor) -> torch.Tensor:
        """The posterior mean ``[B, z, T', H', W']`` without computing the log-variance half:
        the folded encoder tail runs only its first ``z`` output channels."""
        return self._encode(x, mode_only=True)

    def _decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        *,
        temporal_chunk_size: int = 1,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        _, _, num_frame, _, _ = z.shape

        self.clear_cache()
        self.decoder.conv_out.max_in_frames = self.options.conv_out_frame_split or None
        prev_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self.options.benchmark_convs(
            encoder=False, frames=num_frame
        )
        try:
            return self._decode_impl(z, return_dict, temporal_chunk_size)
        finally:
            torch.backends.cudnn.benchmark = prev_benchmark

    def fold_latent_affine(self, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> None:
        """Fold the latent de-normalization ``u = std * z + mean`` into ``post_quant_conv``
        (``P' = P diag(std)``, ``b' = P mean + b``, composed in fp32, rounded once). After
        this call ``decode`` takes the model's raw latents; ``clear_latent_affine`` undoes it.
        The module's own weights are untouched."""
        conv = self.post_quant_conv
        with torch.inference_mode(False), torch.no_grad():
            w = conv.weight.detach().float()  # [z, z, 1, 1, 1]
            z = w.shape[0]
            mean = latents_mean.detach().to(w.device, torch.float32).reshape(z)
            std = latents_std.detach().to(w.device, torch.float32).reshape(z)
            b = (
                conv.bias.detach().float()
                if conv.bias is not None
                else torch.zeros(z, device=w.device)
            )
            folded_w = (w * std.view(1, z, 1, 1, 1)).to(conv.weight.dtype)
            folded_b = (w.view(z, z) @ mean + b).to(conv.weight.dtype)
        self._post_quant_folded = (folded_w, folded_b)

    def clear_latent_affine(self) -> None:
        self._post_quant_folded = None

    def _post_quant(self, z: torch.Tensor) -> torch.Tensor:
        if self._post_quant_folded is None:
            return self.post_quant_conv(z)
        w, b = self._post_quant_folded
        return _channels_last_3d_if_needed(F.conv3d(z, w, b))

    def _use_fused_output(self, x: torch.Tensor) -> bool:
        return (
            self.options.fused_output
            and self.config.patch_size is not None
            and x.shape[0] == 1
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and unpatchify_clamp_available()
        )

    def _decode_impl(
        self, z: torch.Tensor, return_dict: bool, temporal_chunk_size: int
    ) -> DecoderOutput | tuple[torch.Tensor]:
        _, _, num_frame, _, _ = z.shape
        z = _channels_last_3d_if_needed(z)
        x = self._post_quant(z)
        fused = self._use_fused_output(x)
        out: torch.Tensor | None = None
        frames_done = 0
        out_chunks: list[torch.Tensor] = []
        for index, frame_slice in enumerate(_decode_chunk_slices(num_frame, temporal_chunk_size)):
            self._conv_idx = [0]
            out_chunk = self.decoder(
                x[:, :, frame_slice, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=index == 0,
            )
            if fused:
                # Unpatchify + clamp of this chunk straight into the preallocated video.
                out_chunk = _channels_last_3d_if_needed(out_chunk)
                if out is None:
                    patch = self.config.patch_size
                    total = (num_frame - 1) * self.config.scale_factor_temporal + 1
                    out = torch.empty(
                        (
                            1,
                            out_chunk.shape[1] // (patch * patch),
                            total,
                            out_chunk.shape[3] * patch,
                            out_chunk.shape[4] * patch,
                        ),
                        dtype=out_chunk.dtype,
                        device=out_chunk.device,
                        memory_format=torch.channels_last_3d,
                    )
                unpatchify_clamp_into(out_chunk, out, frames_done, self.config.patch_size)
                frames_done += out_chunk.shape[2]
            else:
                out_chunks.append(out_chunk)

        if fused:
            assert out is not None
            if frames_done != out.shape[2]:
                raise RuntimeError(
                    f"decoder produced {frames_done} frames for a {out.shape[2]}-frame output"
                )
        else:
            out = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=2)
            if self.config.patch_size is not None:
                out = unpatchify(out, patch_size=self.config.patch_size)
            out = _channels_last_3d_if_needed(out)
            out = torch.clamp(out, min=-1.0, max=1.0)

        self.clear_cache()
        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)

    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        *,
        temporal_chunk_size: int = 1,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        decoded = self._decode(z, temporal_chunk_size=temporal_chunk_size).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
