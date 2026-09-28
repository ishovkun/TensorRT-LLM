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
"""One layer of causal-rollout attention at Cosmos3-Nano geometry, four ways.

    trtllm       TRTLLM backend over CausalKVCacheManager: one fused paged trtllm-gen call
    cudnn_paged  cuDNN backend over the same cache: run-copy of the new K/V + paged SDPA
    slab         one contiguous K/V buffer per layer: in-place write + cuDNN dense SDPA
    naive        materialised torch.cat of all K/V + SDPA (the VANILLA path shipping today)

Timing is CUPTI kernel timestamps on CUDA-graph replays with an L2 flush before
each iteration; ``cupti.finalize`` runs exactly once per process. Every variant
is cross-checked against an fp32 dense reference before anything is timed, so a
silent kernel fallback (one that drops the cached prefix) fails here rather than
being measured.

    python bench_kv_causal_attention.py --prompt-len 512
    python bench_kv_causal_attention.py --q-tokens 394 --kv-offset 1182   # last clean-pass frame
"""

from __future__ import annotations

import argparse
import bisect
import statistics
import sys
from functools import partial

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 32, 8, 128
TOKENS_PER_FRAME, FRAMES_PER_CHUNK, TOKENS_PER_BLOCK = 394, 4, 32
CHUNK = TOKENS_PER_FRAME * FRAMES_PER_CHUNK
DTYPE = torch.bfloat16
DEV = torch.device("cuda")


# ------------------------------------------------------------------ CUPTI timing


class CuptiTimer:
    def __init__(
        self, iters: int, warmup: int, l2_flush: bool, use_graph: bool, dump: bool = False
    ) -> None:
        from cupti import cupti

        self.cupti, self.iters, self.warmup = cupti, iters, warmup
        self.use_graph = use_graph
        self.dump = dump
        self._l2 = (
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV) if l2_flush else None
        )

    def _flush(self) -> None:
        if self._l2 is not None:
            self._l2.fill_(0)

    def time(self, run_fn, tag: str) -> dict:
        cupti = self.cupti
        run_fn()
        torch.cuda.synchronize()

        if self.use_graph:
            g_reset = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_reset):
                self._flush()
            g_run = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_run):
                run_fn()
            torch.cuda.synchronize()
            reset, run = g_reset.replay, g_run.replay
        else:
            reset, run = self._flush, run_fn

        for _ in range(self.warmup):
            reset()
            run()
        torch.cuda.synchronize()

        launches, kernels = [], []

        def on_buffer_requested():
            return 8 * 1024 * 1024, 0

        def on_buffer_completed(_launches, _kernels, activities):
            for a in activities:
                if a.kind in (
                    cupti.ActivityKind.CONCURRENT_KERNEL,
                    cupti.ActivityKind.MEMCPY,
                    cupti.ActivityKind.MEMSET,
                ):
                    name = a.name if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL else None
                    _kernels.append((a.start, a.end, a.correlation_id, name))
                elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                    _launches.append((a.start, a.end, a.correlation_id))

        kinds = [
            cupti.ActivityKind.RUNTIME,
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.DRIVER,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
        ]
        for kind in kinds:
            cupti.activity_enable(kind)
        cupti.activity_register_callbacks(
            on_buffer_requested, partial(on_buffer_completed, launches, kernels)
        )

        stamps = []
        for _ in range(self.iters):
            reset()
            torch.cuda.synchronize()
            t0 = cupti.get_timestamp()
            run()
            torch.cuda.synchronize()
            stamps.append((t0, cupti.get_timestamp()))

        cupti.activity_flush_all(0)
        for kind in kinds:
            cupti.activity_disable(kind)

        by_corr: dict[int, list] = {}
        for k in kernels:
            by_corr.setdefault(k[2], []).append(k)
        launches.sort(key=lambda x: x[0])
        starts = [x[0] for x in launches]

        us, counts = [], []
        for idx, (t0, t1) in enumerate(stamps):
            lo, hi = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1)
            ks = [k for i in range(lo, hi) for k in by_corr.get(launches[i][2], [])]
            if not ks:
                raise RuntimeError(f"{tag}: no kernel activity recorded for iteration {idx}")
            t_start = min(k[0] for k in ks)
            us.append((max(k[1] for k in ks) - t_start) / 1e3)
            counts.append(sum(1 for k in ks if k[3] is not None))
            if self.dump and idx == 0:
                for k in sorted(ks, key=lambda r: r[0]):
                    print(
                        f"  k: {tag:6s} +{(k[0] - t_start) / 1e3:7.1f}us  {(k[1] - k[0]) / 1e3:7.1f}us  "
                        f"{k[3] or '<memcpy/memset>'}"[:150]
                    )
        us.sort()
        return {
            "median_us": statistics.median(us),
            "min_us": us[0],
            "p90_us": us[int(0.9 * (len(us) - 1))],
            "kernels": statistics.median(counts),
        }


# ------------------------------------------------------------------ variants


def build_cache(prompt_len: int, window_tokens: int, history_chunks: int, gen):
    """Open a cache, write the prompt, commit ``history_chunks`` chunks.

    Returns the manager plus dense copies of the prompt and of the *resident*
    history (the tail of everything committed, stale tokens included).
    """
    mgr = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_block=TOKENS_PER_BLOCK,
        prompt_capacity=max(prompt_len, 1),
        window_tokens=window_tokens,
        chunk_tokens=CHUNK,
    )
    mgr.open(prompt_len=prompt_len)
    kp = torch.randn(prompt_len, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    vp = torch.randn_like(kp)
    mgr.write_prompt_kv(0, kp, vp)

    hist_k, hist_v = [], []
    for _ in range(history_chunks):
        k = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
        v = torch.randn_like(k)
        mgr.write_range(0, mgr.past_tokens, k, v)
        mgr.commit_chunk()
        hist_k.append(k)
        hist_v.append(v)
    n = mgr.history_tokens
    return mgr, kp, vp, torch.cat(hist_k)[-n:], torch.cat(hist_v)[-n:]


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], enable_gqa=True
    )[0].transpose(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["trtllm", "cudnn_paged", "slab", "naive", "all"],
        default="all",
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--window-frames", type=int, default=96)
    ap.add_argument(
        "--history-chunks",
        type=int,
        default=28,
        help="chunks committed before timing; > window/chunk so the table has rotated",
    )
    ap.add_argument(
        "--q-tokens", type=int, default=CHUNK, help="new tokens per call (394 = one frame)"
    )
    ap.add_argument(
        "--kv-offset",
        type=int,
        default=0,
        help="chunk tokens already written before this call (per-frame clean pass)",
    )
    ap.add_argument("--no-l2-flush", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dump-kernels", action="store_true", help="print kernel names for iteration 0"
    )
    args = ap.parse_args()
    if args.kv_offset + args.q_tokens > CHUNK:
        raise SystemExit("--kv-offset + --q-tokens must fit in one chunk")

    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    window = args.window_frames * TOKENS_PER_FRAME
    mgr, kp, vp, k_hist, v_hist = build_cache(args.prompt_len, window, args.history_chunks, gen)
    n_new, off = args.q_tokens, args.kv_offset
    start = mgr.past_tokens + off
    seq_len = start + n_new

    # Chunk tokens before the offset were written by earlier per-frame calls.
    k_pre = torch.randn(off, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    v_pre = torch.randn_like(k_pre)
    mgr.write_range(0, mgr.past_tokens, k_pre, v_pre)

    q = torch.randn(n_new, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    k = torch.randn(n_new, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    v = torch.randn_like(k)
    print(
        f"geometry: q={n_new} keys={seq_len} (prompt {args.prompt_len} + history "
        f"{mgr.history_tokens} [{mgr.stale_tokens} stale] + chunk offset {off} + new {n_new}) "
        f"heads={NUM_HEADS}/{NUM_KV_HEADS} d={HEAD_DIM} page={TOKENS_PER_BLOCK} bf16"
    )

    trtllm_attn = TrtllmAttention(
        layer_idx=0,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        dtype=DTYPE,
        max_seq_len=mgr.capacity,
        attention_metadata_state={},
    )
    cudnn_attn = CuDNNAttention(
        layer_idx=0, num_heads=NUM_HEADS, head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS, dtype=DTYPE
    )
    q4, k4, v4 = q[None], k[None], v[None]

    def run_trtllm():
        return trtllm_attn.forward(
            q4, k4, v4, batch_size=1, seq_len=n_new, kv_cache=mgr, kv_cache_offset=off
        )

    def run_cudnn_paged():
        return cudnn_attn.forward(q4, k4, v4, kv_cache=mgr, kv_cache_offset=off)

    # Slab: what a contiguous per-layer buffer would cost. Prompt and history are
    # already in place; a call writes the new tokens and attends over a slice.
    slab_k = torch.empty(mgr.capacity, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE)
    slab_v = torch.empty_like(slab_k)
    resident_k, resident_v = torch.cat((kp, k_hist, k_pre)), torch.cat((vp, v_hist, v_pre))
    slab_k[:start].copy_(resident_k)
    slab_v[:start].copy_(resident_v)

    def run_slab():
        slab_k[start:seq_len].copy_(k)
        slab_v[start:seq_len].copy_(v)
        return sdpa(q, slab_k[:seq_len], slab_v[:seq_len])

    def run_naive():
        return sdpa(q, torch.cat((resident_k, k)), torch.cat((resident_v, v)))

    variants = {
        "trtllm": run_trtllm,
        "cudnn_paged": run_cudnn_paged,
        "slab": run_slab,
        "naive": run_naive,
    }
    chosen = list(variants) if args.variant == "all" else [args.variant]

    # Cross-check before timing: a silent fallback that ignores the prefix shows up here.
    ref = sdpa(q.float(), torch.cat((resident_k, k)).float(), torch.cat((resident_v, v)).float())
    chunk_only = sdpa(q.float(), k.float(), v.float())
    for name in chosen:
        out = variants[name]().reshape(n_new, NUM_HEADS, HEAD_DIM).float()
        torch.cuda.synchronize()
        err = (out - ref).abs().max().item()
        err_chunk = (out - chunk_only).abs().max().item()
        print(
            f"check {name:12s} max|err| vs dense fp32 = {err:.4f}   vs new-only = {err_chunk:.4f}"
        )
        if err > 2e-2 or err_chunk < 1e-2:
            raise SystemExit(f"{name}: wrong result -- not benchmarking a broken path")

    timer = CuptiTimer(
        args.iters, args.warmup, not args.no_l2_flush, not args.no_graph, args.dump_kernels
    )
    rows = []
    for name in chosen:
        try:
            r = timer.time(variants[name], name)
        except RuntimeError as e:
            if args.no_graph or "capture" not in str(e).lower():
                raise
            print(f"{name}: graph capture failed ({str(e)[:60]}); timing eager", file=sys.stderr)
            r = CuptiTimer(
                args.iters, args.warmup, not args.no_l2_flush, False, args.dump_kernels
            ).time(variants[name], name)
        rows.append((name, r))

    print(f"\n{'variant':12s} {'median us':>10s} {'min us':>9s} {'p90 us':>9s} {'kernels':>8s}")
    for name, r in rows:
        print(
            f"{name:12s} {r['median_us']:10.1f} {r['min_us']:9.1f} {r['p90_us']:9.1f} {r['kernels']:8.0f}"
        )
    mgr.shutdown()
    try:
        timer.cupti.finalize()
    except Exception:  # once-per-process teardown; nothing depends on it succeeding
        pass


if __name__ == "__main__":
    main()
