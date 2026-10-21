# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-count scaling of the Wan VAE backends from bench_wan_vae.py runs.

One line per backend, x = GPUs (world size), y = median GPU time, one panel per (op, shape).

The runs are bench_wan_vae.py outputs (``run_*.json``) taken on one node at world sizes 1, 2,
4, 8 with the same ``--label`` prefix; a backend's point at a world size is its most recent
measurement there. Annotations (``--note backend=text``) are printed under the legend, e.g. the
single-GPU vLLM-Omni point being its fast path while the multi-GPU points are its spatial-shard
decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SERIES = (
    ("diffusers", "diffusers"),
    ("native", "trt-llm native"),
    ("vllm_channels_last", "vllm-omni"),
    ("vllm_spatial_shard", "vllm-omni"),
    ("winovae", "WinoVae"),
)
PANELS = (
    ("decode", "t2v_720p_189f", "decode — 189 frames @ 720x1280"),
    ("decode", "edge_480p_121f", "decode — 121 frames @ 480x832"),
    ("decode", "bimanual_480x832_901f", "bimanual-sim decode step (2 latents -> 8 frames)"),
    ("encode", "t2v_720p_189f", "encode — 189 frames @ 720x1280"),
    ("encode", "edge_480p_121f", "encode — 121 frames @ 480x832"),
    ("decode", "t2i_1024", "decode — 1 image @ 1024x1024"),
)


def _median_ms(row: dict) -> float:
    if row.get("step_latency") and row["step_latency"].get("step_ms") is not None:
        return float(row["step_latency"]["step_ms"])
    return row["timing"]["span"]["median_us"] / 1e3


def _load(run_dir: Path, prefix: str) -> dict[tuple[str, str, str, int], float]:
    """(backend, shape, op, world_size) -> median ms of the most recent matching run."""
    points: dict[tuple[str, str, str, int], tuple[str, float]] = {}
    for path in sorted(run_dir.glob("run_*.json")):
        run = json.loads(path.read_text())
        meta = run["meta"]
        if not meta["label"].startswith(prefix):
            continue
        for r in run["results"]:
            key = (r["backend"], r["shape"], r["op"], int(meta["world_size"]))
            stamp = meta["timestamp"]
            if key not in points or points[key][0] < stamp:
                points[key] = (stamp, _median_ms(r))
    return {k: v for k, (_, v) in points.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--label-prefix", default="scaling")
    parser.add_argument("--out", required=True, help="PNG path")
    parser.add_argument("--title", default="Wan VAE scaling over GPUs")
    parser.add_argument("--note", action="append", default=[], help="backend=text footnote")
    args = parser.parse_args()
    points = _load(Path(args.run_dir), args.label_prefix)
    if not points:
        raise SystemExit("no matching runs")
    sizes = sorted({k[3] for k in points})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p for p in PANELS if any(k[1] == p[1] and k[2] == p[0] for k in points)]
    cols = 3
    rows_n = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(6.2 * cols, 4.6 * rows_n), squeeze=False)
    colors = {name: f"C{i}" for i, (_, name) in enumerate(dict(SERIES).items())}
    # Backends sharing a display name form one line (vLLM-Omni: the fast path at one GPU, the
    # sharded decode beyond), the most recent point winning where both measured a size.
    lines: dict[str, list[str]] = {}
    for backend, name in SERIES:
        lines.setdefault(name, []).append(backend)
    for ax, (op, shape, title) in zip(axes.flat, panels):
        for name, backends in lines.items():
            pts = {
                n: points[(b, shape, op, n)]
                for b in backends
                for n in sizes
                if (b, shape, op, n) in points
            }
            if not pts:
                continue
            xs = sorted(pts)
            ys = [pts[n] for n in xs]
            ax.plot(xs, ys, marker="o", color=colors[name], label=name)
            for x, y in zip(xs, ys):
                ax.annotate(
                    f"{y:.0f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=7,
                )
        ax.set_title(title, fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlabel("GPUs")
        ax.set_ylabel("median GPU time [ms]")
        ax.grid(True, which="both", alpha=0.3)
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    # Legend right under the title, where the eye lands first.
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="upper center",
        ncol=len(seen),
        bbox_to_anchor=(0.5, 0.925),
        fontsize=11,
    )
    note = "; ".join(args.note)
    fig.suptitle(args.title + (f"\n{note}" if note else ""), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
