# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime compilation of the Wan VAE position-walk kernel (``cuda/vae_norm_kernel.cuh``).

One kernel template covers the fused RMSNorm + SiLU, the temporal Winograd norm (slab
producer and product consumer, order 2 or 4) and the Winograd combine; its parameters (dtypes,
order, lane layout, mode flags, stages) are all known at the call site. Instead of
instantiating every combination ahead of time, this module compiles the one a call needs on
first use (nvcc from the toolkit, a few seconds), caches the cubin in memory and on disk, and
launches it through the driver API on the current torch stream (CUDA-graph safe).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass

import torch

_SRC_DIR = pathlib.Path(__file__).parent / "cuda"
_HEADERS = (
    "packed_vector.cuh",
    "vae_norm_common.cuh",
    "vae_wino_tables.cuh",
    "vae_norm_kernel.cuh",
)
_THREADS = 256
_WARPS = _THREADS // 32
_VEC = 8  # 16-byte vectors of 16-bit elements

MID_NONE, MID_NORM, MID_NORM_SILU = 0, 1, 2
OUT_SLABS, OUT_ROWS = 0, 1


def _check(result, what: str):
    err, *values = result
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return values[0] if len(values) == 1 else values


@functools.lru_cache(maxsize=None)
def _cuda_home() -> str:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is needed to compile the VAE norm kernel at run time")
    return CUDA_HOME


@functools.lru_cache(maxsize=None)
def _arch(device: int) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


@functools.lru_cache(maxsize=None)
def _sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=None)
def _sources() -> dict[str, str]:
    return {name: (_SRC_DIR / name).read_text() for name in _HEADERS}


def _extra_defines() -> list[str]:
    """Experiment knobs for the kernel source (``-D`` flags), from TRTLLM_VAE_JIT_DEFINES."""
    raw = os.environ.get("TRTLLM_VAE_JIT_DEFINES", "")
    return [f"-D{d}" for d in raw.split(",") if d]


def _cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(base) / "tensorrt_llm" / "vae_jit"


def row_layout(channels: int, fold_single: bool = True) -> tuple[int, int]:
    """(lanes per row, 16-byte vectors per lane) for a channel row. ``fold_single`` turns a
    one-vector-per-lane layout into two vectors on half the lanes: the per-row overhead
    amortizes over too few elements at one vector, and the two-slice pipeline with the
    cross-frame prefetch keeps four rows per warp in flight (measured on the image shapes:
    one vector per lane with whole-row prefetch ran 1.7x slower than the standalone norm)."""
    nvec = channels // _VEC
    lanes = 32
    while nvec % lanes:
        lanes //= 2
    k = nvec // lanes
    if fold_single and k == 1 and lanes > 1:
        lanes //= 2
        k = 2
    return lanes, k


def blocks_per_sm(k: int, in_wino: bool, order: int, out: int) -> int:
    if out == OUT_ROWS:
        return 4 if k <= 2 else 3
    if k >= 4 or order == 4:
        return 2
    return 4 if (not in_wino and k <= 2) else 3


@dataclass(frozen=True)
class Variant:
    slab_half: bool
    order: int
    lanes: int
    k: int
    reference: bool
    has_bias: bool
    has_in_bias: bool
    has_residual: bool
    in_wino: bool
    mid: int
    out: int
    padded: bool

    def name(self) -> str:
        def b(v: bool) -> str:
            return "true" if v else "false"

        td = "__half" if self.slab_half else "__nv_bfloat16"
        return (
            f"vaewino::vaeNormKernel<__nv_bfloat16, {td}, {self.order}, {_VEC}, {self.lanes}, {self.k}, "
            f"{b(self.reference)}, {b(self.has_bias)}, {b(self.has_in_bias)}, {b(self.has_residual)}, "
            f"{b(self.in_wino)}, {self.mid}, {self.out}, {b(self.padded)}>"
        )

    def signature(self) -> str:
        t, td = "__nv_bfloat16", "__half" if self.slab_half else "__nv_bfloat16"
        return (
            f"({t} const*, {td} const*, {td}*, long long, {t} const*, {t} const*, {t} const*, {t} const*, "
            f"{t} const*, {t}*, {t}*, {t}*, long long, int, int, int, int, int, int, int, float, float)"
        )

    @property
    def positions_per_block(self) -> int:
        return (32 // self.lanes) * _WARPS

    def shared_bytes(self, channels: int) -> int:
        ring = (
            (self.order + 1) * self.positions_per_block * channels * 2
            if self.out == OUT_SLABS
            else 0
        )
        return 3 * channels * 4 + ring


class _Kernel:
    def __init__(self, function, module, variant: Variant) -> None:
        self.function = function
        self.module = module
        self.variant = variant
        self.smem_configured = 0


def _compile_cubin(variant: Variant, arch: str) -> tuple[bytes, str]:
    """nvcc compiles one explicit instantiation; the mangled name comes from the cubin."""
    cuda_home = _cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise RuntimeError(
            f"nvcc not found at {nvcc}; the runtime-compiled VAE norm kernel needs the CUDA toolkit"
        )
    name = variant.name()
    src = (
        '#include "vae_norm_kernel.cuh"'
        + "\n"
        + f"template __global__ void {name}{variant.signature()};"
        + "\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        cu = os.path.join(tmp, "vae_norm_kernel_inst.cu")
        cubin = os.path.join(tmp, "k.cubin")
        pathlib.Path(cu).write_text(src)
        cmd = [
            nvcc,
            "-cubin",
            f"-arch={arch}",
            "--std=c++17",
            "-O3",
            *_extra_defines(),
            f"-I{_SRC_DIR}",
            f"-I{os.path.join(cuda_home, 'include', 'cccl')}",
            "-o",
            cubin,
            cu,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"nvcc failed for {name}: {res.stderr[-4000:]}")
        data = pathlib.Path(cubin).read_bytes()
        dump = subprocess.run(
            [os.path.join(cuda_home, "bin", "cuobjdump"), "--dump-resource-usage", cubin],
            capture_output=True,
            text=True,
        ).stdout
    lowered = None
    for line in dump.splitlines():
        line = line.strip()
        if line.startswith("Function ") and "vaeNormKernel" in line:
            lowered = line[len("Function ") :].rstrip(":")
            break
    if lowered is None:
        raise RuntimeError(f"could not find the kernel symbol in the cubin for {name}")
    return data, lowered


@functools.lru_cache(maxsize=None)
def _kernel(device: int, variant: Variant) -> _Kernel:
    from cuda.bindings import driver as cuda

    arch = _arch(device)
    name = variant.name()
    digest = hashlib.sha256()
    for h in _HEADERS:
        digest.update(_sources()[h].encode())
    digest.update(f"{name}|{arch}|{'|'.join(_extra_defines())}|v2".encode())
    cache = _cache_dir() / f"{digest.hexdigest()[:32]}.cubin"
    lowered_path = cache.with_suffix(".name")
    if cache.exists() and lowered_path.exists():
        cubin = cache.read_bytes()
        lowered = lowered_path.read_text()
    else:
        cubin, lowered = _compile_cubin(variant, arch)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(cubin)
        os.replace(tmp, cache)
        lowered_path.write_text(lowered)
    torch.cuda.set_device(device)
    torch.cuda.current_stream(device)  # make sure the primary context exists
    module = _check(cuda.cuModuleLoadData(cubin), "cuModuleLoadData")
    function = _check(cuda.cuModuleGetFunction(module, lowered.encode()), "cuModuleGetFunction")
    return _Kernel(function, module, variant)


_ARG_TYPES = (
    *(ctypes.c_void_p,) * 3,
    ctypes.c_longlong,
    *(ctypes.c_void_p,) * 8,
    ctypes.c_longlong,
    *(ctypes.c_int,) * 7,
    ctypes.c_float,
    ctypes.c_float,
)


def _addr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def _launch(
    variant: Variant,
    device: int,
    *,
    x,
    m_ptr: int,
    d,
    comp_stride: int,
    residual,
    in_bias,
    cache,
    cache_frames: int,
    gamma,
    bias,
    out_sum,
    y_tail,
    y,
    positions: int,
    channels: int,
    frames: int,
    height: int,
    width: int,
    pad_h: int,
    pad_w: int,
    scale: float,
) -> None:
    from cuda.bindings import driver as cuda

    kernel = _kernel(device, variant)
    smem = variant.shared_bytes(channels)
    if smem > kernel.smem_configured:
        _check(
            cuda.cuFuncSetAttribute(
                kernel.function,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem,
            ),
            "cuFuncSetAttribute",
        )
        kernel.smem_configured = smem
    out_positions = (height + pad_h) * (width + pad_w) if variant.padded else positions
    if out_positions <= 0:
        return
    steps = -(-out_positions // variant.positions_per_block)
    persistent = _sm_count(device) * blocks_per_sm(
        variant.k, variant.in_wino, variant.order, variant.out
    )
    blocks = int(min(steps, persistent))
    args = (
        _addr(x),
        m_ptr,
        _addr(d),
        comp_stride,
        _addr(residual),
        _addr(in_bias),
        _addr(cache) if cache_frames else 0,
        _addr(gamma),
        _addr(bias),
        _addr(out_sum),
        _addr(y_tail),
        _addr(y),
        positions,
        channels,
        frames,
        cache_frames,
        height,
        width,
        pad_h,
        pad_w,
        float(scale),
        1e-12,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    _check(
        cuda.cuLaunchKernel(
            kernel.function, blocks, 1, 1, _THREADS, 1, 1, smem, stream, (args, _ARG_TYPES), 0
        ),
        "cuLaunchKernel",
    )


def _stacked_products(m: list[torch.Tensor], slab_dtype: torch.dtype) -> tuple[int, int]:
    """(base pointer, component stride in elements) of the products, which the kernel reads as
    one stacked buffer [comps, groups, P, C]: consecutive slices of the previous norm's slab
    buffer in the model; anything else is stacked here (a copy)."""
    comps = len(m)
    numel = m[0].numel()
    base = m[0].data_ptr()
    if any(mk.dtype != slab_dtype or not mk.is_contiguous() for mk in m):
        raise ValueError("m must be contiguous tensors in the slab dtype")
    if any(m[k].data_ptr() != base + k * numel * 2 for k in range(comps)):
        stacked = torch.stack(m)
        m[:] = list(stacked.unbind(0))
        base = m[0].data_ptr()
    return base, numel


def wino_norm_silu(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    residual: torch.Tensor | None,
    in_bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    frames: int,
    *,
    order: int = 2,
    half_slabs: bool = False,
    reference_rounding: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Same contract as ``trtllm::vae_wino_norm_silu``: frame layout ``[T, P, C]`` (``m``/``D`` as
    ``[T/order, P, C]``, cache ``[0..2, P, C]``). Returns ``(sum or None, y_tail, D)``."""
    if (x is None) == (m is None):
        raise ValueError("exactly one of x and m")
    if order not in (2, 4) or (order == 4 and not half_slabs):
        raise ValueError("order must be 2 or 4; order 4 needs fp16 slabs")
    if frames < order or frames % order:
        raise ValueError("frames must be a positive multiple of order")
    comps = order + 2
    slab_dtype = torch.float16 if half_slabs else torch.bfloat16
    src = x if x is not None else m[0]
    _, positions, channels = src.shape
    if channels % _VEC:
        raise ValueError("channels must be a multiple of 8")
    device = src.device.index
    opts = dict(device=src.device, dtype=torch.bfloat16)
    want_sum = residual is not None or in_bias is not None or x is None
    out_sum = torch.empty((frames, positions, channels), **opts) if want_sum else None
    y_tail = torch.empty((2, positions, channels), **opts)
    d = torch.empty(
        (comps, frames // order, positions, channels), device=src.device, dtype=slab_dtype
    )
    cache_frames = cache.shape[0] if cache is not None and cache.numel() else 0
    m_ptr = 0
    if m is not None:
        if len(m) != comps:
            raise ValueError("m must hold order + 2 tensors")
        m_ptr, stride = _stacked_products(m, slab_dtype)
        if stride != d[0].numel():
            raise ValueError("products and slabs must have the same component size")
    lanes, k = row_layout(channels, fold_single=True)
    variant = Variant(
        half_slabs,
        order,
        lanes,
        k,
        reference_rounding,
        bias is not None,
        in_bias is not None,
        residual is not None,
        x is None,
        MID_NORM_SILU,
        OUT_SLABS,
        False,
    )
    _launch(
        variant,
        device,
        x=x,
        m_ptr=m_ptr,
        d=d,
        comp_stride=d[0].numel(),
        residual=residual,
        in_bias=in_bias,
        cache=cache,
        cache_frames=cache_frames,
        gamma=gamma,
        bias=bias,
        out_sum=out_sum,
        y_tail=y_tail,
        y=None,
        positions=positions,
        channels=channels,
        frames=frames,
        height=0,
        width=0,
        pad_h=0,
        pad_w=0,
        scale=scale,
    )
    return out_sum, y_tail, d


def rms_norm_silu_rows(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    *,
    reference_rounding: bool = False,
    in_bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    silu: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Same contract as ``trtllm::vae_rms_norm_silu`` / ``vae_add_rms_norm_silu`` on rows
    ``[N, C]`` bf16: returns ``(sum or None, y)``; ``sum = round(x (+ in_bias) + residual)`` is
    written when a residual is given; ``out`` is an optional destination for ``y``."""
    rows, channels = x.shape
    if channels % _VEC:
        raise ValueError("channels must be a multiple of 8")
    device = x.device.index
    y = out if out is not None else torch.empty_like(x)
    out_sum = torch.empty_like(x) if residual is not None else None
    lanes, k = row_layout(channels, fold_single=True)
    variant = Variant(
        False,
        1,
        lanes,
        k,
        reference_rounding,
        bias is not None,
        in_bias is not None,
        residual is not None,
        False,
        MID_NORM_SILU if silu else MID_NORM,
        OUT_ROWS,
        False,
    )
    _launch(
        variant,
        device,
        x=x,
        m_ptr=0,
        d=None,
        comp_stride=0,
        residual=residual,
        in_bias=in_bias,
        cache=None,
        cache_frames=0,
        gamma=gamma,
        bias=bias,
        out_sum=out_sum,
        y_tail=None,
        y=y,
        positions=rows,
        channels=channels,
        frames=1,
        height=0,
        width=0,
        pad_h=0,
        pad_w=0,
        scale=scale,
    )
    return out_sum, y


def combine(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    frames: int,
    height: int,
    width: int,
    pad_h: int = 0,
    pad_w: int = 0,
) -> torch.Tensor:
    """Same contract as ``trtllm::vae_wino_combine``: ``(combine(m) | x) + bias + residual`` as
    bf16 frames ``[T, (H+pad_h)*(W+pad_w), C]`` with zero rows/columns appended, every add
    rounded like the eager chain."""
    if (x is None) == (m is None):
        raise ValueError("exactly one of x and m")
    src = x if x is not None else m[0]
    positions, channels = src.shape[1], src.shape[2]
    if height * width != positions:
        raise ValueError("height * width must equal the position count")
    device = src.device.index
    order, m_ptr, stride = 1, 0, 0
    slab_half = False
    if m is not None:
        order = len(m) - 2
        if order not in (2, 4) or frames % order:
            raise ValueError(
                "m must hold order + 2 tensors (order 2 or 4) and T must be divisible by the order"
            )
        slab_half = m[0].dtype == torch.float16
        m_ptr, stride = _stacked_products(m, m[0].dtype)
    padded = pad_h > 0 or pad_w > 0
    y = torch.empty(
        (frames, (height + pad_h) * (width + pad_w), channels),
        device=src.device,
        dtype=torch.bfloat16,
    )
    lanes, k = row_layout(channels, fold_single=True)
    variant = Variant(
        slab_half,
        order,
        lanes,
        k,
        True,
        False,
        bias is not None,
        residual is not None,
        m is not None,
        MID_NONE,
        OUT_ROWS,
        padded,
    )
    _launch(
        variant,
        device,
        x=x,
        m_ptr=m_ptr,
        d=None,
        comp_stride=stride,
        residual=residual,
        in_bias=bias,
        cache=None,
        cache_frames=0,
        gamma=None,
        bias=None,
        out_sum=None,
        y_tail=None,
        y=y,
        positions=positions,
        channels=channels,
        frames=frames,
        height=height,
        width=width,
        pad_h=pad_h,
        pad_w=pad_w,
        scale=1.0,
    )
    return y
