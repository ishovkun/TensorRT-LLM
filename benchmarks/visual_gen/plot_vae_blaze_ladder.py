# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge bench_wan_vae.py runs into ladder.json and render the latest ladder PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_runs(run_dir: Path) -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("run_*.json"))]
    runs.sort(key=lambda r: r["meta"]["timestamp"])
    return runs


def _median_ms(row: dict) -> float:
    return row["timing"]["span"]["median_us"] / 1e3


def _ladder_rows(runs: list[dict]) -> list[dict]:
    # Reference timings (diffusers/native) come from the most recent run that measured them,
    # so a blaze-only ladder step is still expressed relative to the baseline.
    latest_ref: dict[tuple[str, str, str], dict] = {}
    for run in runs:
        for r in run["results"]:
            if r["backend"] in ("diffusers", "native"):
                latest_ref[(r["backend"], r["shape"], r["op"])] = r
    rows = []
    for run in runs:
        meta = run["meta"]
        by_key = {(r["backend"], r["shape"], r["op"]): r for r in run["results"]}
        for (backend, shape, op), r in by_key.items():
            native = by_key.get(("native", shape, op)) or latest_ref.get(("native", shape, op))
            diffusers = by_key.get(("diffusers", shape, op)) or latest_ref.get(
                ("diffusers", shape, op)
            )
            rows.append(
                {
                    "label": meta["label"],
                    "timestamp": meta["timestamp"],
                    "git_sha": meta["git_sha"],
                    "gpu": meta["gpu_short"],
                    "world_size": meta["world_size"],
                    "backend": backend,
                    "blaze_opts": meta["blaze_opts"] if backend == "blaze" else {},
                    "shape": shape,
                    "op": op,
                    "graph": r["timing"]["graph"],
                    "median_ms": _median_ms(r),
                    "p90_ms": r["timing"]["span"]["p90_us"] / 1e3,
                    "wall_median_ms": r["timing"]["wall"]["median_us"] / 1e3,
                    "kernels_per_iter": r["timing"]["kernels_per_iter"],
                    "speedup_vs_diffusers": (_median_ms(diffusers) / _median_ms(r))
                    if diffusers
                    else None,
                    "speedup_vs_native": (_median_ms(native) / _median_ms(r)) if native else None,
                    "rel_l2_vs_ref": r.get("vs_ref_fp32", {}).get("rel_l2"),
                    "rel_l2_vs_native": r.get("vs_native", {}).get("rel_l2"),
                    "peak_alloc_gb": r.get("peak_alloc_gb"),
                    "category_us": r["timing"]["category_us_per_iter"],
                }
            )
    return rows


def _plot(
    rows: list[dict], out_png: Path, gpu: str, world_size: int, only_shapes: set[str] | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = [op for op in ("decode", "encode") if any(r["op"] == op for r in rows)]
    shapes = (
        sorted({r["shape"] for r in rows} & only_shapes)
        if only_shapes
        else sorted({r["shape"] for r in rows})
    )
    # One bar per (backend/label) in chronological order; reference backends from their latest run.
    series: list[tuple[str, dict[tuple[str, str], dict]]] = []
    for backend in ("diffusers", "vllm_lossless", "vllm_channels_last", "native"):
        pts: dict[tuple[str, str], dict] = {}
        for r in rows:  # rows are chronological; later runs overwrite
            if r["backend"] == backend:
                pts[(r["shape"], r["op"])] = r
        if pts:
            series.append((backend, pts))
    # Per panel the most recent blaze measurement is drawn (a run may cover only some
    # shapes); the series is named after the newest of them. The history stays in ladder.json.
    blaze_rows = sorted((r for r in rows if r["backend"] == "blaze"), key=lambda r: r["timestamp"])
    if blaze_rows:
        pts = {(r["shape"], r["op"]): r for r in blaze_rows}  # later rows overwrite
        drawn = [r for r in pts.values() if only_shapes is None or r["shape"] in only_shapes]
        latest_label = max(drawn or blaze_rows, key=lambda r: r["timestamp"])["label"]
        series.append((f"blaze:{latest_label}", pts))

    # One panel per (op, shape): rows are ops, columns are shapes, linear y per panel.
    fig, axes = plt.subplots(
        len(ops), len(shapes), figsize=(5.2 * len(shapes), 4.0 * len(ops)), squeeze=False
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for oi, op in enumerate(ops):
        for si, shape in enumerate(shapes):
            ax = axes[oi, si]
            names, values, speedups = [], [], []
            for name, pts in series:
                r = pts.get((shape, op))
                if r is None:
                    continue
                names.append(name)
                values.append(r["median_ms"])
                speedups.append(r["speedup_vs_diffusers"])
            xs = range(len(names))
            bars = ax.bar(xs, values, color=[colors[i % len(colors)] for i in range(len(names))])
            for bar, sp in zip(bars, speedups):
                if sp is not None:
                    ax.annotate(
                        f"{sp:.2f}x",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            ax.set_xticks(list(xs))
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
            ax.set_title(f"{op} — {shape}")
            if si == 0:
                ax.set_ylabel("Median GPU time [ms]")
            ax.set_ylim(0, max(values) * 1.15 if values else 1)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Wan VAE optimization ladder — {gpu}, world_size={world_size}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="directory holding run_*.json")
    parser.add_argument(
        "--out-dir", default=None, help="where ladder.json and the PNG go (default: run-dir)"
    )
    parser.add_argument(
        "--shapes",
        default="t2i_1024,t2v_720p_189f,edge_480p_121f",
        help="comma list of shapes to draw (ladder.json keeps every shape); empty draws all",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = _load_runs(run_dir)
    if not all_runs:
        raise SystemExit(f"no run_*.json under {run_dir}")
    rows = _ladder_rows(all_runs)
    (out_dir / "ladder.json").write_text(json.dumps(rows, indent=1))

    # One plot per (GPU, world size): ladders on different hardware never share a panel.
    for stale in out_dir.glob("ladder_*.png"):
        stale.unlink()
    signatures = sorted({(r["meta"]["gpu_short"], r["meta"]["world_size"]) for r in all_runs})
    for gpu, world_size in signatures:
        sig_rows = [r for r in rows if r["gpu"] == gpu and r["world_size"] == world_size]
        out_png = out_dir / f"ladder_{gpu}_ws{world_size}.png"
        _plot(
            sig_rows, out_png, gpu, world_size, set(args.shapes.split(",")) if args.shapes else None
        )

    print(
        f"{'label':22s} {'backend':19s} {'shape':16s} {'op':6s} {'median_ms':>10s} {'p90_ms':>10s} {'x_diffusers':>11s} {'x_native':>8s} {'rel_l2_ref':>11s}"
    )
    for r in rows:
        sd = f"{r['speedup_vs_diffusers']:.3f}" if r["speedup_vs_diffusers"] else "-"
        sn = f"{r['speedup_vs_native']:.3f}" if r["speedup_vs_native"] else "-"
        rl = f"{r['rel_l2_vs_ref']:.2e}" if r["rel_l2_vs_ref"] is not None else "-"
        print(
            f"{r['label']:22s} {r['backend']:19s} {r['shape']:16s} {r['op']:6s} {r['median_ms']:10.2f} {r['p90_ms']:10.2f} {sd:>11s} {sn:>8s} {rl:>11s}"
        )
    print(f"wrote {out_dir / 'ladder.json'} and {len(signatures)} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
