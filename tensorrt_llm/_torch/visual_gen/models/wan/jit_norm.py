# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime compilation of the fused Winograd norm kernel.

The kernel in ``cuda/vae_wino_norm_silu.cuh`` is one template whose parameters (activation and
slab dtypes, temporal order, lane layout, five mode flags) are all known at the call site. The
ahead-of-time build would instantiate every combination; this module compiles the one a call
needs on first use (nvcc from the toolkit, a few seconds), caches the cubin in memory and on
disk, and launches it through the driver API on the current torch stream (CUDA-graph safe,
like the CuTe DSL kernels).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import pathlib

import torch

_SRC_DIR = pathlib.Path(__file__).parent / "cuda"
_HEADERS = (
    "packed_vector.cuh",
    "vae_norm_common.cuh",
    "vae_wino_tables.cuh",
    "vae_wino_norm_silu.cuh",
)
_THREADS = 256
_WARPS = _THREADS // 32
_VEC = 8  # 16-byte vectors of 16-bit elements
_MAX_COMPONENTS = 6


def _check(result, what: str):
    err, *values = result
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return values[0] if len(values) == 1 else values


@functools.lru_cache(maxsize=None)
def _cuda_include() -> str:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is needed to compile the VAE norm kernel at run time")
    return os.path.join(CUDA_HOME, "include")


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


def _cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(base) / "tensorrt_llm" / "vae_jit"


def row_layout(channels: int) -> tuple[int, int]:
    """(lanes per row, 16-byte vectors per lane) for a channel row; mirrors ``winoRowLayout``."""
    nvec = channels // _VEC
    lanes = 32
    while nvec % lanes:
        lanes //= 2
    k = nvec // lanes
    if k == 1 and lanes > 1:
        lanes //= 2
        k = 2
    return lanes, k


def blocks_per_sm(k: int, in_wino: bool, order: int) -> int:
    if k >= 4 or order == 4:
        return 2
    return 4 if (not in_wino and k <= 2) else 3


class _Kernel:
    def __init__(self, function, module, lanes: int, k: int, in_wino: bool, order: int) -> None:
        self.function = function
        self.module = module
        self.lanes = lanes
        self.k = k
        self.in_wino = in_wino
        self.order = order
        self.smem_configured = 0

    @property
    def positions_per_block(self) -> int:
        return (32 // self.lanes) * _WARPS

    def shared_bytes(self, channels: int) -> int:
        return 3 * channels * 4 + (self.order + 1) * self.positions_per_block * channels * 2


def _compile_cubin(name: str, arch: str) -> tuple[bytes, str]:
    """Compile the instantiation ``name`` for ``arch`` and return (cubin, lowered name).

    nvcc from the toolkit does the work (a few seconds per instantiation; NVRTC 13.4 crashes
    on this source). The explicit instantiation is written into a temporary .cu next to the
    headers; the mangled name is read back from the cubin's resource table.
    """
    import subprocess
    import tempfile

    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise RuntimeError(
            f"nvcc not found at {nvcc}; the runtime-compiled VAE norm kernel needs the CUDA toolkit"
        )
    inner = name[name.index("<") + 1 : name.rindex(">")]
    args = inner.split(", ")
    t, td = args[0], args[1]
    sig = (
        f"({t} const*, {td} const*, {td}*, long long, {t} const*, {t} const*, {t} const*, {t} const*, "
        f"{t} const*, {t}*, {t}*, long long, int, int, int, float, float)"
    )
    src = (
        '#include "vae_wino_norm_silu.cuh"'
        + chr(10)
        + f"template __global__ void {name}{sig};"
        + chr(10)
    )
    with tempfile.TemporaryDirectory() as tmp:
        cu = os.path.join(tmp, "vae_wino_norm_silu_inst.cu")
        cubin = os.path.join(tmp, "k.cubin")
        pathlib.Path(cu).write_text(src)
        cmd = [
            nvcc,
            "-cubin",
            f"-arch={arch}",
            "--std=c++17",
            "-O3",
            f"-I{_SRC_DIR}",
            f"-I{os.path.join(_cuda_include(), 'cccl')}",
            "-o",
            cubin,
            cu,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"nvcc failed for {name}: {res.stderr[-4000:]}")
        data = pathlib.Path(cubin).read_bytes()
        dump = subprocess.run(
            [os.path.join(CUDA_HOME, "bin", "cuobjdump"), "--dump-resource-usage", cubin],
            capture_output=True,
            text=True,
        ).stdout
    lowered = None
    for line in dump.splitlines():
        line = line.strip()
        if line.startswith("Function ") and "vaeWinoNormSiluKernel" in line:
            lowered = line[len("Function ") :].rstrip(":")
            break
    if lowered is None:
        raise RuntimeError(f"could not find the kernel symbol in the cubin for {name}")
    return data, lowered


@functools.lru_cache(maxsize=None)
def _kernel(
    device: int,
    slab_half: bool,
    order: int,
    lanes: int,
    k: int,
    reference: bool,
    has_bias: bool,
    has_in_bias: bool,
    has_residual: bool,
    in_wino: bool,
) -> _Kernel:
    from cuda.bindings import driver as cuda

    def b(v: bool) -> str:
        return "true" if v else "false"

    slab_type = "__half" if slab_half else "__nv_bfloat16"
    name = (
        f"vaewino::vaeWinoNormSiluKernel<__nv_bfloat16, {slab_type}, {order}, {_VEC}, {lanes}, {k}, "
        f"{b(reference)}, {b(has_bias)}, {b(has_in_bias)}, {b(has_residual)}, {b(in_wino)}>"
    )
    arch = _arch(device)
    digest = hashlib.sha256()
    for h in _HEADERS:
        digest.update(_sources()[h].encode())
    digest.update(f"{name}|{arch}|v1".encode())
    cache = _cache_dir() / f"{digest.hexdigest()[:32]}.cubin"
    lowered_path = cache.with_suffix(".name")
    if cache.exists() and lowered_path.exists():
        cubin = cache.read_bytes()
        lowered = lowered_path.read_text()
    else:
        cubin, lowered = _compile_cubin(name, arch)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(cubin)
        os.replace(tmp, cache)
        lowered_path.write_text(lowered)
    torch.cuda.set_device(device)
    torch.cuda.current_stream(device)  # make sure the primary context exists
    module = _check(cuda.cuModuleLoadData(cubin), "cuModuleLoadData")
    function = _check(cuda.cuModuleGetFunction(module, lowered.encode()), "cuModuleGetFunction")
    return _Kernel(function, module, lanes, k, in_wino, order)


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
    from cuda.bindings import driver as cuda

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
    if m is not None:
        if len(m) != comps:
            raise ValueError("m must hold order + 2 tensors")
        for mk in m:
            if mk.dtype != slab_dtype or not mk.is_contiguous():
                raise ValueError("m must be contiguous tensors in the slab dtype")
    device = src.device.index
    opts = dict(device=src.device, dtype=torch.bfloat16)
    group = (frames // order, positions, channels)
    want_sum = residual is not None or in_bias is not None or x is None
    out_sum = torch.empty((frames, positions, channels), **opts) if want_sum else None
    y_tail = torch.empty((2, positions, channels), **opts)
    d = torch.empty((comps, *group), device=src.device, dtype=slab_dtype)
    cache_frames = cache.shape[0] if cache is not None and cache.numel() else 0

    lanes, k = row_layout(channels)
    kernel = _kernel(
        device,
        half_slabs,
        order,
        lanes,
        k,
        reference_rounding,
        bias is not None,
        in_bias is not None,
        residual is not None,
        x is None,
    )

    def addr(t: torch.Tensor | None) -> int:
        return 0 if t is None else t.data_ptr()

    elem = 2  # 16-bit slabs and products
    comp_stride = d[0].numel()
    m_ptr = 0
    if m is not None:
        # The products are read as one stacked buffer, [comps, groups, P, C]: consecutive
        # slices of the previous norm's slab buffer in the model; anything else is stacked here.
        base = m[0].data_ptr()
        if any(m[k].data_ptr() != base + k * comp_stride * elem for k in range(comps)):
            m = list(torch.stack(m).unbind(0))
            base = m[0].data_ptr()
        m_ptr = base
    args = (
        addr(x),
        m_ptr,
        d.data_ptr(),
        comp_stride,
        addr(residual),
        addr(in_bias),
        addr(cache) if cache_frames else 0,
        gamma.data_ptr(),
        addr(bias),
        addr(out_sum),
        y_tail.data_ptr(),
        positions,
        channels,
        frames,
        cache_frames,
        float(scale),
        1e-12,
    )
    types = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    )

    smem = kernel.shared_bytes(channels)
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
    steps = -(-positions // kernel.positions_per_block)
    persistent = _sm_count(device) * blocks_per_sm(k, x is None, order)
    blocks = int(min(steps, persistent))
    if positions > 0:
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        _check(
            cuda.cuLaunchKernel(
                kernel.function, blocks, 1, 1, _THREADS, 1, 1, smem, stream, (args, types), 0
            ),
            "cuLaunchKernel",
        )
    return out_sum, y_tail, d
