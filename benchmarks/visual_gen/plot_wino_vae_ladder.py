# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge bench_wan_vae.py runs into ladder.json and render the latest ladder PNG."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

# Bars drawn per panel, in this order, and their display names. The vLLM-Omni bit-exact
# level is kept in ladder.json but not drawn: it times the same as Diffusers by construction.
SERIES = (
    ("diffusers", "diffusers"),
    ("native", "trt-llm native"),
    ("vllm_channels_last", "vllm"),
    ("winovae", "WinoVae"),
)
# Plot columns: per op the bench shape, the row field drawn, and the panel title (frames and
# resolution as in SHAPES of bench_wan_vae.py). The bimanual-sim column shows what the sim
# pays per episode: the steady-state decode step and the one-time encode of its input image.
COLUMNS: dict[str, dict[str, tuple[str, str, str]]] = {
    "t2i": {
        "decode": ("t2i_1024", "median_ms", "decode — t2i, 1 frame @ 1024x1024"),
        "encode": ("t2i_1024", "median_ms", "encode — t2i, 1 frame @ 1024x1024"),
    },
    "t2v_720p": {
        "decode": ("t2v_720p_189f", "median_ms", "decode — t2v, 189 frames @ 720x1280"),
        "encode": ("t2v_720p_189f", "median_ms", "encode — t2v, 189 frames @ 720x1280"),
    },
    "edge_480p": {
        "decode": ("edge_480p_121f", "median_ms", "decode — edge, 121 frames @ 480x832"),
        "encode": ("edge_480p_121f", "median_ms", "encode — edge, 121 frames @ 480x832"),
    },
    "bimanual_sim": {
        "decode": (
            "bimanual_480x832_901f",
            "step_ms",
            "bimanual-sim decode step: 2 latents -> 8 frames @ 480x832",
        ),
        "encode": (
            "bimanual_cond_480x832",
            "median_ms",
            "bimanual-sim encode: 1 image @ 480x832, once per episode",
        ),
    },
}


def _load_runs(run_dir: Path) -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("run_*.json"))]
    # Runs written before the rename call the backend "blaze".
    for run in runs:
        meta = run["meta"]
        if "blaze_opts" in meta:
            meta["winovae_opts"] = meta.pop("blaze_opts")
        for r in run["results"]:
            if r["backend"] == "blaze":
                r["backend"] = "winovae"
    runs.sort(key=lambda r: r["meta"]["timestamp"])
    return runs


def _median_ms(row: dict) -> float:
    return row["timing"]["span"]["median_us"] / 1e3


def _ladder_rows(runs: list[dict]) -> list[dict]:
    # Reference timings (diffusers/native) come from the most recent run that measured them,
    # so a winovae-only ladder step is still expressed relative to the baseline.
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
                    "winovae_opts": meta["winovae_opts"] if backend == "winovae" else {},
                    "shape": shape,
                    "op": op,
                    "graph": r["timing"]["graph"],
                    "median_ms": _median_ms(r),
                    "p90_ms": r["timing"]["span"]["p90_us"] / 1e3,
                    "wall_median_ms": r["timing"]["wall"]["median_us"] / 1e3,
                    "kernels_per_iter": r["timing"]["kernels_per_iter"],
                    "step_ms": r.get("step_latency", {}).get("step_ms"),
                    "cold_start_ms": r.get("step_latency", {}).get("cold_start_ms"),
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
    rows: list[dict], out_png: Path, gpu: str, world_size: int, columns: list[str], newest: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = ("encode", "decode")  # encode on top, decode below
    present = {r["shape"] for r in rows}
    columns = [c for c in columns if any(COLUMNS[c][op][0] in present for op in ops)]
    # Per panel the most recent measurement of each backend is drawn (a run may cover only
    # some shapes). The history stays in ladder.json.
    series: list[tuple[str, dict[tuple[str, str], dict]]] = []
    for backend, name in SERIES:
        pts: dict[tuple[str, str], dict] = {}
        for r in sorted((r for r in rows if r["backend"] == backend), key=lambda r: r["timestamp"]):
            pts[(r["shape"], r["op"])] = r
        if pts:
            series.append((name, pts))

    # One panel per (op, column), linear y per panel; speedups are against the panel's
    # Diffusers bar.
    fig, axes = plt.subplots(
        len(ops), len(columns), figsize=(5.2 * len(columns), 4.0 * len(ops)), squeeze=False
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_index = {name: i for i, (_, name) in enumerate(SERIES)}  # one color per backend
    for oi, op in enumerate(ops):
        for si, column in enumerate(columns):
            ax = axes[oi, si]
            shape, field, title = COLUMNS[column][op]
            names, values = [], []
            for name, pts in series:
                r = pts.get((shape, op))
                if r is None or r.get(field) is None:
                    continue
                names.append(name)
                values.append(r[field])
            base = dict(zip(names, values)).get(SERIES[0][1])
            speedups = [base / v if base else None for v in values]
            xs = range(len(names))
            bars = ax.bar(xs, values, color=[colors[series_index[n] % len(colors)] for n in names])
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
            ax.set_title(title, fontsize=10)
            if si == 0:
                ax.set_ylabel("Median GPU time [ms]")
            ax.set_ylim(0, max(values) * 1.15 if values else 1)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Wan VAE optimization ladder — {gpu}, world_size={world_size} (newest run {newest})"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="directory holding run_*.json")
    parser.add_argument(
        "--out-dir", default=None, help="where ladder.json and the PNG go (default: run-dir)"
    )
    parser.add_argument(
        "--columns",
        default=",".join(COLUMNS),
        help="comma list of plot columns to draw, in order (ladder.json keeps every shape)",
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

    # One plot per (GPU, world size): ladders on different hardware never share a panel. A
    # plot is redrawn only when a run newer than the file exists, so an untouched ladder keeps
    # its date instead of passing as fresh.
    signatures = sorted({(r["meta"]["gpu_short"], r["meta"]["world_size"]) for r in all_runs})
    for gpu, world_size in signatures:
        sig_rows = [r for r in rows if r["gpu"] == gpu and r["world_size"] == world_size]
        newest = max(r["timestamp"] for r in sig_rows)
        out_png = out_dir / f"ladder_{gpu}_ws{world_size}.png"
        if out_png.exists():
            newest_epoch = datetime.fromisoformat(newest).timestamp()
            if out_png.stat().st_mtime >= newest_epoch:
                print(f"kept {out_png.name}: no run newer than the file (newest run {newest[:16]})")
                continue
        _plot(sig_rows, out_png, gpu, world_size, args.columns.split(","), newest[:16])

    print(
        f"{'label':22s} {'backend':19s} {'shape':16s} {'op':6s} {'median_ms':>10s} {'p90_ms':>10s}"
        f" {'x_diffusers':>11s} {'x_native':>8s} {'rel_l2_ref':>11s}"
    )
    for r in rows:
        sd = f"{r['speedup_vs_diffusers']:.3f}" if r["speedup_vs_diffusers"] else "-"
        sn = f"{r['speedup_vs_native']:.3f}" if r["speedup_vs_native"] else "-"
        rl = f"{r['rel_l2_vs_ref']:.2e}" if r["rel_l2_vs_ref"] is not None else "-"
        print(
            f"{r['label']:22s} {r['backend']:19s} {r['shape']:16s} {r['op']:6s} {r['median_ms']:10.2f}"
            f" {r['p90_ms']:10.2f} {sd:>11s} {sn:>8s} {rl:>11s}"
        )
    print(f"wrote {out_dir / 'ladder.json'} and {len(signatures)} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
