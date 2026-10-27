# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plots of ``bench_wino_methods.py`` results.

Per-conv time (CUPTI, all kernels of the method) of each conv method, grouped by decoder layer
shape with the shipped method outlined; or, with ``--sweeps``, one panel per channel count over
frame sizes (``--speedup``: relative to direct cuDNN). Slide-sized.

    python benchmarks/visual_gen/plot_wino_methods.py wino_methods.json --out wino_methods.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DARK, MUTED = "#1F2937", "#6B7280"
COLORS = {
    "direct cuDNN": "#9CA3AF",
    "F(2×2) cuBLAS": "#3B6EA8",
    "F(4×4) cuBLAS": "#7FA7D6",
    "F(4×4) KF": "#C9711C",
    "fused 1D F(2,3)": "#76B900",
}
LABELS = {"fused 1D F(2,3)": "fused: width F(2,3), direct height"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json", nargs="+", help="JSON file(s); several resolution sweeps overlay by channel count"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--sweeps",
        action="store_true",
        help="positional JSONs are resolution sweeps, one panel each",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="with --sweeps: plot t_cuDNN / t_method instead of the time",
    )
    args = parser.parse_args()
    if args.sweeps:
        gpu = json.loads(open(args.json[0]).read())["meta"]["gpu"]
        sweeps(args.json, args.out, gpu, speedup=args.speedup)
        return
    data = json.loads(open(args.json[0]).read())
    layers = data["layers"]
    methods = [m for m in COLORS if any(m in r["methods"] for r in layers)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(12, 5.0))
    x = np.arange(len(layers))
    width = 0.8 / len(methods)
    for k, m in enumerate(methods):
        xs = x - 0.4 + width * (k + 0.5)
        vals = [r["methods"][m]["span_us"] if m in r["methods"] else np.nan for r in layers]
        edge = [DARK if r["shipped"] == m else "none" for r in layers]
        ax.bar(
            xs,
            vals,
            width * 0.92,
            color=COLORS[m],
            edgecolor=edge,
            linewidth=2.0,
            label=LABELS.get(m, m),
        )
        for xi, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(
                    xi,
                    v * 1.04,
                    f"{v / 1e3:.2f} ms" if v >= 1e3 else f"{v:.0f} µs",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=DARK,
                    rotation=90,
                )
    ax.set_yscale("log")
    ax.set_ylabel("time per conv call, µs (CUPTI; transforms + GEMMs)", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels([r["layer"] for r in layers], fontsize=11.5, color=DARK)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(top=ax.get_ylim()[1] * 3.0)
    ax.legend(
        fontsize=10.5,
        frameon=False,
        loc="upper left",
        ncol=3,
        title="outlined = what the decoder ships",
        title_fontsize=10,
    )
    title = args.title or f"3×3 conv of one chunk's slabs, per decoder layer: {data['meta']['gpu']}"
    fig.suptitle(title, fontsize=13, color=DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)


def sweeps(res_jsons: list[str], out: str, gpu: str, speedup: bool = False) -> None:
    """One panel per channel count, frame size on a log x axis, one line per method.

    ``speedup`` plots t_cuDNN / t_method on a linear axis (direct cuDNN is the flat line at 1);
    otherwise the time per conv on a log axis, panels sharing y.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    runs = sorted(
        (json.loads(open(p).read())["layers"] for p in res_jsons), key=lambda rows: rows[0]["cin"]
    )
    fig, axs = plt.subplots(1, len(runs), figsize=(5.0 * len(runs), 5.2), sharey=True)
    top = 1.0
    for ax, rows in zip(axs, runs):
        xs = [r["h"] * r["w"] for r in rows]
        for m in COLORS:
            pts = []
            for x, r in zip(xs, rows):
                mm = r["methods"]
                if m not in mm or "span_us" not in mm[m]:
                    continue
                if speedup:
                    if "span_us" not in mm.get("direct cuDNN", {}):
                        continue
                    pts.append((x, mm["direct cuDNN"]["span_us"] / mm[m]["span_us"]))
                else:
                    pts.append((x, mm[m]["span_us"]))
            if not pts:
                continue
            top = max(top, max(p[1] for p in pts))
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker="o",
                ms=6,
                lw=2.4,
                color=COLORS[m],
                label=LABELS.get(m, m),
            )
        ax.set_xscale("log")
        if speedup:
            ax.axhline(1.0, color=COLORS["direct cuDNN"], lw=1.2, ls="--")
        else:
            ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [r["layer"] for r in rows], fontsize=10.5, color=DARK, rotation=35, ha="right"
        )
        ax.set_title(f"C_in = C_out = {rows[0]['cin']}", fontsize=13, color=DARK)
        ax.set_xlabel("frame size (pixels)", color=DARK)
        ax.tick_params(axis="y", labelsize=10.5, colors=MUTED, labelleft=True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", which="major", color="#D1D5DB", lw=0.9)
        ax.set_axisbelow(True)
    if speedup:
        axs[0].set_ylim(0, top * 1.15)
        axs[0].set_ylabel("speedup over direct cuDNN  (t_cuDNN / t_method)", color=DARK)
    else:
        axs[0].set_ylabel("time per conv call, µs (CUPTI; transforms + GEMMs)", color=DARK)
    handles, labels = {}, []
    for ax in axs:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    fig.legend(
        [handles[lab] for lab in labels],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(labels),
        fontsize=11,
        frameon=False,
    )
    fig.suptitle(
        f"3×3 conv of one chunk's slabs (4 component frames): {gpu}, CUDA-graph replay, CUPTI",
        fontsize=13.5,
        color=DARK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
