# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error of every backend against the fp32 Diffusers reference, all workloads in one bar plot.

Reads bench_wan_vae.py runs (``run_*.json``): for each (shape, op) group the bars are the
backends' relative L2 error against the fp32 reference, the most recent measurement per
backend at the chosen world size. The y axis is linear from zero in units of 1e-2 (``--logy``:
log axis with power-of-ten labels); fonts are sized for a slide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_wino_vae_ladder import COLUMNS, SERIES  # noqa: E402

# (column, op) in the ladder's order, with the short group label drawn under the bars.
GROUPS = tuple(
    (column, op, f"{name}\n{op}")
    for column, name in (
        ("t2i", "image 1024"),
        ("t2v_720p", "720p 189f"),
        ("edge_480p", "480p 121f"),
        ("bimanual_sim", "bimanual image"),
    )
    for op in ("encode", "decode")
)


def _load(run_dir: Path, world_size: int, gpu: str | None) -> dict[tuple[str, str, str], float]:
    """(backend, shape, op) -> rel_l2 against fp32 of the most recent run that measured it."""
    latest: dict[tuple[str, str, str], tuple[str, float]] = {}
    for path in sorted(run_dir.glob("run_*.json")):
        run = json.loads(path.read_text())
        meta = run["meta"]
        if int(meta["world_size"]) != world_size or (gpu and meta["gpu_short"] != gpu):
            continue
        for r in run["results"]:
            err = (r.get("vs_ref_fp32") or {}).get("rel_l2")
            if err is None:
                continue
            backend = "winovae" if r["backend"] == "blaze" else r["backend"]
            key = (backend, r["shape"], r["op"])
            if key not in latest or latest[key][0] < meta["timestamp"]:
                latest[key] = (meta["timestamp"], float(err))
    return {k: v for k, (_, v) in latest.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True, help="PNG path")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gpu", default=None, help="gpu_short filter, e.g. GB200")
    parser.add_argument("--title", default="Wan VAE: relative L2 error against the fp32 reference")
    parser.add_argument("--logy", action="store_true", help="log y axis")
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="COLUMN=SHAPE",
        help="draw COLUMN from bench shape SHAPE instead of the ladder's (e.g. "
        "t2v_720p=t2v_720p_121f); the label's frame count follows",
    )
    parser.add_argument(
        "--op",
        choices=("encode", "decode"),
        default=None,
        help="plot only this op (half-slide size)",
    )
    args = parser.parse_args()
    errors = _load(Path(args.run_dir), args.world_size, args.gpu)
    if not errors:
        raise SystemExit("no accuracy rows match")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    # The bimanual step run measures no accuracy (reference skipped); its conditioning
    # image stands for that column.
    groups = []
    for column, op, label in GROUPS:
        if args.op and op != args.op:
            continue
        if args.op:
            label = label.split("\n")[0].replace(" image", "\nimage")
        shape = COLUMNS[column][op][0]
        if shape == "bimanual_480x832_901f":
            shape = "bimanual_cond_480x832"
        for item in args.shape:
            col, _, override = item.partition("=")
            if col == column:
                shape = override
                frames = re.search(r"_(\d+)f$", override)
                if frames:
                    label = re.sub(r"\d+f", frames.group(1) + "f", label)
        if any(k[1] == shape and k[2] == op for k in errors):
            groups.append((shape, op, label))
    names = list(dict(SERIES).items())
    width = 0.8 / len(names)
    scale = 1.0 if args.logy else 1e2
    half = args.op is not None
    fs = (
        {"tick": 16, "label": 16, "legend": 15, "value": 12, "title": 19}
        if half
        else {"tick": 13, "label": 14, "legend": 14, "value": 10, "title": 16}
    )
    fig, ax = plt.subplots(figsize=(8.0, 6.0) if half else (2.0 * len(groups) + 1, 5.6))
    for i, (backend, name) in enumerate(names):
        xs, ys = [], []
        for g, (shape, op, _) in enumerate(groups):
            if (backend, shape, op) in errors:
                xs.append(g + (i - (len(names) - 1) / 2) * width)
                ys.append(errors[(backend, shape, op)] * scale)
        if not xs:
            continue
        bars = ax.bar(xs, ys, width=width * 0.95, label=name, color=f"C{i}")
        for bar, y in zip(bars, ys):
            ax.annotate(
                f"{y:.2e}".replace("e-0", "e-") if args.logy else f"{y:.2f}",
                (bar.get_x() + bar.get_width() / 2, y),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=6.5 if args.logy else fs["value"],
                rotation=90 if args.logy or half else 0,
            )
    if args.logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.set_ylabel(r"relative $L_2$ error vs fp32", fontsize=fs["label"])
        ax.set_ylim(min(errors.values()) / 2, max(errors.values()) * 4)
    else:
        ax.set_ylabel(r"relative $L_2$ error vs fp32 [$10^{-2}$]", fontsize=fs["label"])
        ax.set_ylim(0, max(errors.values()) * scale * (1.6 if half else 1.3))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label for _, _, label in groups], fontsize=fs["tick"])
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(
        loc="upper center",
        ncol=2 if half else len(names),
        bbox_to_anchor=(0.5, 1.0),
        fontsize=fs["legend"],
    )
    ax.set_title(args.title, pad=12, fontsize=fs["title"])
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
