# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan VAE component benchmark: Diffusers vs vLLM-Omni fast path vs native WanVAE vs VaeBlaze.

Random weights (seeded, shared by all backends), random inputs at Cosmos3
default shapes, CUDA-graph replay timed with CUPTI kernel timestamps, L2
flushed between replays. Writes one JSON per run; ``plot_vae_blaze_ladder.py``
merges runs into the optimization ladder.
"""

from __future__ import annotations

import argparse
import atexit
import bisect
import datetime as dt
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

_THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

VLLM_BACKENDS = {"vllm_lossless": "lossless", "vllm_channels_last": "channels_last"}

# Wan2.2-TI2V-5B VAE, as shipped in Cosmos3 checkpoints' vae/config.json.
COSMOS_VAE_CONFIG: dict[str, Any] = {
    "base_dim": 160,
    "decoder_base_dim": 256,
    "z_dim": 48,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
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

# Cosmos3 defaults (tensorrt_llm/_torch/visual_gen/models/cosmos3/defaults.py):
# T2I 1024x1024, 720p video 189 frames, Edge 480p video 121 frames.
SHAPES: dict[str, tuple[int, int, int]] = {
    "t2i_1024": (1, 1024, 1024),
    "t2v_720p_189f": (189, 720, 1280),
    "edge_480p_121f": (121, 480, 832),
    # Cosmos3-Nano-Sim-Bimanual streaming decode: 901 pixel frames = 226 latent frames,
    # decoded 2 latent frames (8 pixel frames) per step after a 1-frame cold start.
    "bimanual_480x832_901f": (901, 480, 832),
}
# Decoder temporal chunk size (latent frames per call) a shape is decoded with when the
# command line does not override it.
SHAPE_DECODE_CHUNK: dict[str, int] = {"bimanual_480x832_901f": 2}

_L2_FLUSH_BYTES = 128 * 1024 * 1024
_l2_flush: torch.Tensor | None = None

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("attention", re.compile(r"fmha|flash|sdpa|attention", re.I)),
    (
        "conv",
        re.compile(r"cudnn|conv|gemm|cutlass|implicit|xmma|nhwc|nchw|wgrad|dgrad|sm\d+_", re.I),
    ),
    (
        "copy",
        re.compile(r"CatArray|copy|pad|index|gather|scatter|transpose|permute|contiguous", re.I),
    ),
    (
        "norm_act",
        re.compile(
            r"reduce|norm|silu|sigmoid|Mul|Add|Div|Sub|elementwise|unrolled|vectorized|clamp|exp|upsample|nearest",
            re.I,
        ),
    ),
]


def _classify(name: str | None) -> str:
    if name is None:
        return "memcpy_memset"
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(name):
            return category
    return "other"


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _gpu_signature() -> tuple[str, str]:
    full = torch.cuda.get_device_name(0)
    short = full.replace("NVIDIA", "").strip().replace(" ", "_")
    return full, short


def _flush_l2() -> None:
    global _l2_flush
    if _l2_flush is None:
        _l2_flush = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    _l2_flush.fill_(0.0)
    torch.cuda.synchronize()


def _latent_shape(T: int, H: int, W: int) -> tuple[int, int, int, int, int]:
    return (1, COSMOS_VAE_CONFIG["z_dim"], (T - 1) // 4 + 1, H // 16, W // 16)


def _pixel_shape(T: int, H: int, W: int) -> tuple[int, int, int, int, int]:
    return (1, 3, T, H, W)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _reference_state_dict(seed: int) -> dict[str, torch.Tensor]:
    from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE

    torch.manual_seed(seed)
    model = WanVAE(COSMOS_VAE_CONFIG)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _build(
    backend: str,
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype,
    blaze_opts: dict[str, Any],
) -> torch.nn.Module:
    if backend == "diffusers" or backend in VLLM_BACKENDS:
        from diffusers import AutoencoderKLWan

        model = AutoencoderKLWan(**COSMOS_VAE_CONFIG)
    elif backend == "native":
        from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE

        model = WanVAE(COSMOS_VAE_CONFIG)
    elif backend == "blaze":
        from tensorrt_llm._torch.visual_gen.models.wan.vae_blaze import VaeBlaze

        model = VaeBlaze(COSMOS_VAE_CONFIG, blaze_opts)
    else:
        raise ValueError(backend)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device="cuda", dtype=dtype).eval().requires_grad_(False)
    if backend in VLLM_BACKENDS:
        # Vendored vLLM-Omni fast path (benchmarks/visual_gen/third_party); rebinds
        # the Diffusers module forwards in place, so decode()/encode() stay the same.
        from vllm_omni_wan_fastpath import (
            install_wan_vae_encoder_fastpath,
            install_wan_vae_fastpath,
        )

        level = VLLM_BACKENDS[backend]
        dec = install_wan_vae_fastpath(model, level=level)
        enc = install_wan_vae_encoder_fastpath(model, level=level)
        model._bench_fastpath = {
            "decoder": {"installed": dec.installed, "reason": getattr(dec, "reason", None)},
            "encoder": {"installed": enc.installed, "reason": getattr(enc, "reason", None)},
        }
        for part, rep_ in model._bench_fastpath.items():
            if not rep_["installed"]:
                print(
                    f"    [vllm-omni] {part} fast path NOT installed: {rep_['reason']}",
                    file=sys.stderr,
                )
    return model


def _make_op(
    model: torch.nn.Module, backend: str, op: str, inp: torch.Tensor, decode_chunk: int
) -> Callable[[], torch.Tensor]:
    if op == "decode":
        if backend == "diffusers" or backend in VLLM_BACKENDS:
            return lambda: model.decode(inp, return_dict=False)[0]
        return lambda: model.decode(inp, return_dict=False, temporal_chunk_size=decode_chunk)[0]
    if op == "encode":
        return lambda: model.encode(inp, return_dict=False)[0].mode()
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _try_capture(run_fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph | None:
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.inference_mode(), torch.cuda.graph(graph):
            run_fn()
        torch.cuda.synchronize()
        return graph
    except Exception as exc:  # capture-incompatible backend: fall back to eager replay
        torch.cuda.synchronize()
        print(
            f"    [graph] capture failed, timing eager: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _time_cupti(
    run_fn: Callable[[], Any], warmup: int, iters: int, use_graph: bool
) -> dict[str, Any]:
    from cupti import cupti

    with torch.inference_mode():
        run_fn()
    torch.cuda.synchronize()

    graph = _try_capture(run_fn) if use_graph else None
    replay = graph.replay if graph is not None else run_fn

    with torch.inference_mode():
        for _ in range(warmup):
            _flush_l2()
            replay()
        torch.cuda.synchronize()

    launches: list[tuple[int, int, int]] = []
    kernels: list[tuple[int, int, int, str | None]] = []

    def _buf_requested():
        return 8 * 1024 * 1024, 0

    def _buf_completed(activities):
        for a in activities:
            if a.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                name = a.name if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL else None
                kernels.append((a.start, a.end, a.correlation_id, name))
            elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append((a.start, a.end, a.correlation_id))

    kinds = [
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ]
    for kind in kinds:
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(_buf_requested, _buf_completed)

    windows: list[tuple[int, int]] = []
    with torch.inference_mode():
        for _ in range(iters):
            _flush_l2()
            t0 = cupti.get_timestamp()
            replay()
            torch.cuda.synchronize()
            t1 = cupti.get_timestamp()
            windows.append((t0, t1))

    cupti.activity_flush_all(0)
    for kind in kinds:
        cupti.activity_disable(kind)
    # No cupti.finalize() here: once per process, at exit.

    corr_to_kernels: dict[int, list[tuple[int, int, int, str | None]]] = {}
    for k in kernels:
        corr_to_kernels.setdefault(k[2], []).append(k)
    sorted_launches = sorted(launches, key=lambda e: e[0])
    launch_starts = [e[0] for e in sorted_launches]
    kernels_by_start = sorted(kernels, key=lambda k: k[0])
    kernel_starts = [k[0] for k in kernels_by_start]

    span_us: list[float] = []
    busy_us: list[float] = []
    wall_us: list[float] = []
    n_kernels: list[int] = []
    category_ns: dict[str, int] = {}
    kernel_ns: dict[str, int] = {}
    for idx, (t0, t1) in enumerate(windows):
        li = bisect.bisect_left(launch_starts, t0)
        ri = bisect.bisect_right(launch_starts, t1)
        corr_ids = {sorted_launches[i][2] for i in range(li, ri)}
        iter_kernels: list[tuple[int, int, int, str | None]] = []
        for cid in corr_ids:
            iter_kernels.extend(corr_to_kernels.get(cid, []))
        if not iter_kernels:
            # Fall back to the timestamp window (same clock as the CUPTI records).
            ki = bisect.bisect_left(kernel_starts, t0)
            kj = bisect.bisect_right(kernel_starts, t1)
            iter_kernels = [k for k in kernels_by_start[ki:kj] if k[1] <= t1]
        if not iter_kernels:
            raise RuntimeError(f"CUPTI recorded no kernels for iteration {idx}")
        min_start = min(k[0] for k in iter_kernels)
        max_end = max(k[1] for k in iter_kernels)
        span_us.append((max_end - min_start) / 1e3)
        busy_us.append(sum(k[1] - k[0] for k in iter_kernels) / 1e3)
        wall_us.append((t1 - t0) / 1e3)
        n_kernels.append(len(iter_kernels))
        for k in iter_kernels:
            d = k[1] - k[0]
            category_ns[_classify(k[3])] = category_ns.get(_classify(k[3]), 0) + d
            short = (k[3] or "<memcpy/memset>")[:160]
            kernel_ns[short] = kernel_ns.get(short, 0) + d

    def _stats(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
        return {
            "median_us": statistics.median(values),
            "p90_us": p90,
            "min_us": ordered[0],
            "max_us": ordered[-1],
        }

    top = sorted(kernel_ns.items(), key=lambda kv: -kv[1])[:15]
    return {
        "graph": graph is not None,
        "iters": iters,
        "span": _stats(span_us),
        "busy": _stats(busy_us),
        "wall": _stats(wall_us),
        "kernels_per_iter": statistics.median(n_kernels),
        "category_us_per_iter": {k: v / 1e3 / iters for k, v in sorted(category_ns.items())},
        "top_kernels_us_per_iter": [{"name": n, "us": v / 1e3 / iters} for n, v in top],
    }


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.float()
    e = expected.float()
    diff = a - e
    return {
        "rel_l2": (diff.norm() / e.norm().clamp_min(1e-12)).item(),
        "max_abs": diff.abs().max().item(),
        "rms": diff.pow(2).mean().sqrt().item(),
        "nan_count": int(torch.isnan(a).sum().item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_blaze_opts(text: str) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        key, _, value = item.partition("=")
        if not value:
            opts[key] = True
        elif value.lower() in ("1", "true", "yes", "0", "false", "no"):
            opts[key] = value.lower() in ("1", "true", "yes")
        elif value.isdigit():
            opts[key] = int(value)
        else:
            opts[key] = value
    return opts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--label", required=True, help="ladder step name, e.g. baseline, fused_norm"
    )
    parser.add_argument(
        "--backends",
        default="native,blaze,diffusers,vllm_lossless,vllm_channels_last",
        help="native runs first so every other backend gets a vs_native accuracy row",
    )
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--ops", default="decode,encode")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--ref-dtype",
        default="float32",
        choices=["float32", "none"],
        help="Diffusers accuracy reference dtype; none skips",
    )
    parser.add_argument("--blaze-opts", default="", help="comma list, e.g. fused_norm=1")
    parser.add_argument(
        "--decode-chunk",
        type=int,
        default=0,
        help="temporal_chunk_size for native/blaze decode (0: per-shape default, 1 unless the shape says otherwise)",
    )
    parser.add_argument(
        "--step-latency",
        action="store_true",
        help="also time one steady-state streaming decode step: decode(1 + chunk latent frames) minus "
        "decode(1), and the cold-start decode(1)",
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--world-size", type=int, default=1, help="recorded in the signature only")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    dtype = getattr(torch, args.dtype)
    backends = args.backends.split(",")
    shapes = args.shapes.split(",")
    ops = args.ops.split(",")
    blaze_opts = _parse_blaze_opts(args.blaze_opts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _finalize():
        try:
            from cupti import cupti

            cupti.finalize()
        except Exception:
            pass

    atexit.register(_finalize)

    repo = Path(__file__).resolve().parents[2]
    gpu_full, gpu_short = _gpu_signature()
    import diffusers

    meta = {
        "label": args.label,
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo),
        "gpu": gpu_full,
        "gpu_short": gpu_short,
        "world_size": args.world_size,
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "diffusers": diffusers.__version__,
        "dtype": args.dtype,
        "blaze_opts": blaze_opts,
        "decode_chunk": args.decode_chunk or "per-shape",
        "iters": args.iters,
        "warmup": args.warmup,
        "cudnn_benchmark": args.cudnn_benchmark,
        "graph_requested": not args.no_graph,
        "argv": sys.argv[1:],
    }
    print(json.dumps(meta, indent=1))

    state_dict = _reference_state_dict(args.seed)
    results: list[dict[str, Any]] = []

    for shape_name in shapes:
        T, H, W = SHAPES[shape_name]
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        inputs = {
            "decode": torch.randn(_latent_shape(T, H, W), generator=gen, device="cuda").to(dtype),
            "encode": (torch.rand(_pixel_shape(T, H, W), generator=gen, device="cuda") * 2 - 1).to(
                dtype
            ),
        }
        decode_chunk = args.decode_chunk or SHAPE_DECODE_CHUNK.get(shape_name, 1)
        for op in ops:
            if op not in inputs:
                raise ValueError(op)
            inp = inputs[op]
            print(
                f"\n=== {shape_name} {op} input={tuple(inp.shape)} chunk={decode_chunk} ===",
                flush=True,
            )

            reference: torch.Tensor | None = None
            native_out: torch.Tensor | None = None
            if not args.skip_accuracy and args.ref_dtype != "none":
                model = _build("diffusers", state_dict, torch.float32, {})
                with torch.inference_mode():
                    reference = (
                        _make_op(model, "diffusers", op, inp.float(), decode_chunk)().float().cpu()
                    )
                del model
                torch.cuda.empty_cache()

            for backend in backends:
                t_start = time.time()
                model = _build(backend, state_dict, dtype, blaze_opts if backend == "blaze" else {})
                run_fn = _make_op(model, backend, op, inp, decode_chunk)

                row: dict[str, Any] = {
                    "backend": backend,
                    "shape": shape_name,
                    "op": op,
                    "input_shape": list(inp.shape),
                    "decode_chunk": decode_chunk,
                }
                if hasattr(model, "_bench_fastpath"):
                    row["fastpath"] = model._bench_fastpath[
                        {"decode": "decoder", "encode": "encoder"}[op]
                    ]
                if not args.skip_accuracy:
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():
                        out = run_fn()
                    torch.cuda.synchronize()
                    row["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
                    row["output_shape"] = list(out.shape)
                    out_cpu = out.float().cpu()
                    if reference is not None:
                        row["vs_ref_fp32"] = _metrics(out_cpu, reference)
                    if backend == "native":
                        native_out = out_cpu
                    elif native_out is not None:
                        row["vs_native"] = _metrics(out_cpu, native_out)
                    del out

                row["timing"] = _time_cupti(
                    run_fn, args.warmup, args.iters, use_graph=not args.no_graph
                )
                if args.step_latency and op == "decode" and inp.shape[2] >= 1 + 2 * decode_chunk:
                    # Streaming step latency: the difference between decoding 1 + k*chunk and
                    # 1 + (k-1)*chunk latent frames is one steady-state step (the clip-level
                    # graph hides per-call costs, so this is the number a streaming consumer sees).
                    def med(n: int, m: torch.nn.Module = model) -> float:
                        sub = inp[:, :, :n].contiguous(memory_format=torch.channels_last_3d)
                        fn = _make_op(m, backend, op, sub, decode_chunk)
                        return (
                            _time_cupti(fn, args.warmup, args.iters, use_graph=not args.no_graph)[
                                "span"
                            ]["median_us"]
                            / 1e3
                        )

                    cold = med(1)
                    one = med(1 + decode_chunk)
                    two = med(1 + 2 * decode_chunk)
                    row["step_latency"] = {
                        "chunk_latent_frames": decode_chunk,
                        "cold_start_ms": cold,
                        "step_ms": one - cold,
                        "step_ms_check": two - one,
                    }
                    print(
                        f"    step latency: cold start {cold:.2f} ms, steady step ({decode_chunk} latent frames) "
                        f"{one - cold:.2f} ms (check {two - one:.2f} ms)",
                        flush=True,
                    )
                row["setup_s"] = time.time() - t_start
                results.append(row)

                t = row["timing"]
                acc = row.get("vs_ref_fp32", {})
                print(
                    f"  {backend:9s} graph={str(t['graph']):5s} span med {t['span']['median_us'] / 1e3:9.2f} ms "
                    f"p90 {t['span']['p90_us'] / 1e3:9.2f} ms  wall med {t['wall']['median_us'] / 1e3:9.2f} ms  "
                    f"kernels/iter {t['kernels_per_iter']:.0f}  "
                    + (f"rel_l2 {acc['rel_l2']:.3e} max_abs {acc['max_abs']:.3e}" if acc else ""),
                    flush=True,
                )
                del model, run_fn
                torch.cuda.empty_cache()
            del reference, native_out
        del inputs
        torch.cuda.empty_cache()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"run_{stamp}_{args.label}_{gpu_short}_ws{args.world_size}.json"
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
