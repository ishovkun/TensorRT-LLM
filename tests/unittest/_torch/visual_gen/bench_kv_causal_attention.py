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
"""One layer of causal-rollout attention at Cosmos3-Nano geometry, three ways.

    fused   one paged trtllm-gen Dense call over [prompt | history | chunk]
    split   paged gen->gen call + dense gen->prompt call + log-sum-exp merge
    naive   materialised torch.cat of all K/V + SDPA (the VANILLA path shipping today)

Timing is CUPTI kernel timestamps on CUDA-graph replays with an L2 flush before
each iteration; ``cupti.finalize`` runs exactly once per process. The three
variants are cross-checked numerically before anything is timed, so a silent
kernel fallback (one that drops the cached prefix) fails here rather than being
measured.

    python bench_kv_causal_attention.py --variant all
"""

from __future__ import annotations

import argparse
import bisect
import statistics
import sys
from functools import partial

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention_backend.causal_kv import CausalKVAttention
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


def build_cache(prompt_len: int, window_tokens: int, gen):
    mgr = CausalKVCacheManager(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_block=TOKENS_PER_BLOCK,
        prompt_capacity=prompt_len,
        window_tokens=window_tokens,
        chunk_tokens=CHUNK,
    )
    mgr.open(prompt_len=prompt_len)
    if mgr.attention_mask(DEV) is not None:
        raise SystemExit(
            f"prompt_len={prompt_len} is not page-exact; the fused variant would take the masked path"
        )
    kp = torch.randn(prompt_len, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    vp = torch.randn_like(kp)
    mgr.write_prompt_kv(0, kp, vp)

    hist_k, hist_v = [], []
    for _ in range(window_tokens // CHUNK):
        pos = torch.arange(mgr.past_tokens, mgr.past_tokens + CHUNK, device=DEV)
        k = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
        v = torch.randn_like(k)
        mgr.write_kv(0, pos, k, v)
        mgr.commit_chunk()
        hist_k.append(k)
        hist_v.append(v)
    assert mgr.stale_tokens == 0, mgr.stale_tokens
    return mgr, kp, vp, torch.cat(hist_k), torch.cat(hist_v)


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], enable_gqa=True
    )[0].transpose(0, 1)


def small_attention_with_lse(q, k, v):
    """Dense gen->prompt attention returning (out [S,H,D], lse [S,H]).

    FA4 (one fused kernel) when its CuTe DSL build is usable; otherwise plain torch,
    which spends several launches on the same 1576 x |prompt| problem.
    """
    try:
        from tensorrt_llm._torch.visual_gen.attention_backend import flash_attn4

        cls = next(
            c
            for n, c in vars(flash_attn4).items()
            if isinstance(c, type)
            and c.__module__ == flash_attn4.__name__  # not the imported abstract base
            and hasattr(c, "forward_with_lse")
        )
        fa4 = cls(
            layer_idx=0,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
            dtype=DTYPE,
        )

        def run_fa4():
            o, lse = fa4.forward_with_lse(q[None], k[None], v[None])
            return o[0], lse[0].transpose(0, 1)

        run_fa4()
        torch.cuda.synchronize()
        return run_fa4, "FA4 (fused)"
    except (ImportError, AttributeError, RuntimeError, TypeError, StopIteration) as e:
        print(
            f"FA4 unavailable for the split's small call ({type(e).__name__}); using torch",
            file=sys.stderr,
        )

    rep = NUM_HEADS // NUM_KV_HEADS
    kx = k.repeat_interleave(rep, dim=1).float()
    vx = v.repeat_interleave(rep, dim=1).float()
    scale = HEAD_DIM**-0.5

    def run_torch():
        s = torch.einsum("shd,thd->hst", q.float(), kx) * scale
        m = s.amax(-1)
        p = torch.exp(s - m[..., None])
        denom = p.sum(-1)
        o = torch.einsum("hst,thd->shd", p / denom[..., None], vx)
        return o.to(DTYPE), (m + torch.log(denom)).transpose(0, 1)

    return run_torch, "torch (unfused, several launches)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fused", "split", "naive", "all"], default="all")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--prompt-len", type=int, default=64, help="must be a multiple of 32 for the fused path"
    )
    ap.add_argument("--window-frames", type=int, default=96)
    ap.add_argument("--no-l2-flush", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dump-kernels", action="store_true", help="print kernel names for iteration 0"
    )
    args = ap.parse_args()

    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    window = args.window_frames * TOKENS_PER_FRAME
    mgr, kp, vp, k_hist, v_hist = build_cache(args.prompt_len, window, gen)
    attn = CausalKVAttention(
        mgr,
        layer_idx=0,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )
    q = torch.randn(CHUNK, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    k = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    v = torch.randn_like(k)
    qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
    # Metadata is prepared once: between denoising steps the table and `past` do not change.
    md = attn._prepared_metadata(CHUNK, DEV)
    seq_len = mgr.seq_len
    print(
        f"geometry: q={CHUNK} keys={seq_len} (prompt {args.prompt_len} + history {mgr.history_tokens} "
        f"+ chunk {CHUNK}) heads={NUM_HEADS}/{NUM_KV_HEADS} d={HEAD_DIM} page={TOKENS_PER_BLOCK} bf16"
    )

    def run_fused():
        return attn._attn.forward(qkv, None, None, md, attention_mask=PredefinedAttentionMask.FULL)

    stats = torch.empty(CHUNK, NUM_HEADS, 2, dtype=torch.float32, device=DEV)
    small, small_name = small_attention_with_lse(q, kp, vp)

    def run_split():
        # gen->gen half. The paged call still holds the prompt's pages: 0.16% more keys
        # than a prompt-free cache, stated rather than hidden.
        o1 = attn._attn.forward(
            qkv,
            None,
            None,
            md,
            attention_mask=PredefinedAttentionMask.FULL,
            softmax_stats_tensor=stats,
        ).view(CHUNK, NUM_HEADS, HEAD_DIM)
        o2, lse2 = small()
        lse1 = stats[..., 0] + torch.log(stats[..., 1])  # trtllm-gen returns (row max, row sum)
        m = torch.maximum(lse1, lse2)
        w1, w2 = torch.exp(lse1 - m), torch.exp(lse2 - m)
        return (o1 * w1[..., None] + o2 * w2[..., None]) / (w1 + w2)[..., None]

    def run_naive():
        return sdpa(q, torch.cat((kp, k_hist, k)), torch.cat((vp, v_hist, v)))

    variants = {"fused": run_fused, "split": run_split, "naive": run_naive}
    chosen = list(variants) if args.variant == "all" else [args.variant]

    # Cross-check before timing: a silent fallback that ignores the prefix shows up here.
    ref = sdpa(q.float(), torch.cat((kp, k_hist, k)).float(), torch.cat((vp, v_hist, v)).float())
    for name in chosen:
        out = variants[name]().reshape(CHUNK, NUM_HEADS, HEAD_DIM).float()
        torch.cuda.synchronize()
        err = (out - ref).abs().max().item()
        chunk_only = (out - sdpa(q.float(), k.float(), v.float())).abs().max().item()
        print(
            f"check {name:6s} max|err| vs dense fp32 = {err:.4f}   vs chunk-only = {chunk_only:.4f}"
        )
        if err > 2e-2 or chunk_only < 1e-2:
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

    print(f"\n{'variant':8s} {'median us':>10s} {'min us':>9s} {'p90 us':>9s} {'kernels':>8s}")
    for name, r in rows:
        print(
            f"{name:8s} {r['median_us']:10.1f} {r['min_us']:9.1f} {r['p90_us']:9.1f} {r['kernels']:8.0f}"
        )
    print(f"\nsplit's small call: {small_name}")
    mgr.shutdown()
    try:
        timer.cupti.finalize()
    except Exception:  # once-per-process teardown; nothing depends on it succeeding
        pass


if __name__ == "__main__":
    main()
