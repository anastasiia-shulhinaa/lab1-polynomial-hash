"""Plots for the Bloom filter lab, built from the CSV files in Lab2/results.

    python -m Lab2.tools.plot                 # every chart whose CSV exists
    python -m Lab2.tools.plot --out-dir Lab2/results

    time_per_op.png   E1  time of one operation against N (bench_summary.csv)
    compare.png       E5  filters against the exact sets: time per operation
    memory.png        E5  memory of the structures
    fp_fill.png       E2  false positive rate against the load (fp_fill.csv)
    fp_hashes.png     E3  false positive rate against l (fp_hashes.csv)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Lab2.bloom.params import fp_theory  # noqa: E402
from Lab2.tools.fp_experiment import wilson  # noqa: E402

_RESULTS = Path(__file__).resolve().parent.parent / "results"

# One colour per operation, the same in every chart (colour follows the entity).
# The four pass the colour-blind separation check on a white surface in this order;
# green and yellow are below 3:1 contrast, so every series is also labelled directly.
OP_COLOUR = {"add": "#2a78d6", "pos": "#eb6834", "neg": "#1baf7a", "del": "#eda100"}
OP_LABEL = {"add": "+", "pos": "? (present)", "neg": "? (absent)", "del": "-"}
VARIANT_LABEL = {"bloom": "Bloom filter", "counting": "Counting filter",
                 "polyset": "PolyHashSet (lab 1)", "set": "built-in set"}
VARIANT_ORDER = ("bloom", "counting", "polyset", "set")

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
REFERENCE = "#898781"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _size_label(n: int) -> str:
    exponent = len(str(n)) - 1
    return f"$10^{exponent}$" if n == 10 ** exponent else f"{n:,}"


# ---- E1: time of one operation against N ----

def plot_time_per_op(summary: list[dict], out_path: Path) -> None:
    variants = [v for v in ("bloom", "counting") if any(r["variant"] == v for r in summary)]
    if not variants:
        return
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 3.8),
                             sharey=True, squeeze=False)
    top = max(float(r["us_per_op"]) for r in summary if r["variant"] in variants)
    for ax, variant in zip(axes[0], variants):
        for phase in OP_COLOUR:
            rows = sorted((r for r in summary if r["variant"] == variant and r["phase"] == phase),
                          key=lambda r: int(r["n"]))
            if not rows:
                continue
            xs = [int(r["n"]) for r in rows]
            ys = [float(r["us_per_op"]) for r in rows]
            ax.plot(xs, ys, color=OP_COLOUR[phase], linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=OP_LABEL[phase],
                    solid_capstyle="round")
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(xs, [_size_label(n) for n in xs])
        ax.minorticks_off()
        ax.set_xlim(xs[0] / 1.6, xs[-1] * 2.6)
        ax.set_ylim(0, top * 1.08)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("N, operations in the phase")
        ax.set_title(VARIANT_LABEL[variant])
    axes[0][0].set_ylabel("time per operation, µs")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.06))
    _save(fig, out_path)


# ---- E5: filters against the exact sets ----

def plot_compare(summary: list[dict], out_path: Path) -> None:
    n = max(int(r["n"]) for r in summary)
    present = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in summary)]
    if len(present) < 2:
        return
    value = {(r["variant"], r["phase"]): float(r["us_per_op"]) for r in summary if int(r["n"]) == n}
    top = max(value.values())
    fig, grid = plt.subplots(2, 2, figsize=(8.4, 0.9 * len(present) + 1.9), sharey=True)
    axes = list(grid.flat)
    ys = range(len(present))
    for ax, phase in zip(axes, OP_COLOUR):
        for y, variant in zip(ys, present):
            v = value.get((variant, phase))
            if v is None:
                ax.text(top * 0.02, y, "not supported", va="center", color=INK_SECONDARY, fontsize=9)
                continue
            ax.barh(y, v, height=0.5, color=OP_COLOUR[phase])
            ax.text(v + top * 0.03, y, f"{v:.2f}", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, top * 1.28)
        ax.set_yticks(list(ys), [VARIANT_LABEL[v] for v in present])
        ax.grid(axis="y", visible=False)
        ax.set_title(OP_LABEL[phase])
        ax.set_xlabel("µs per operation")
    axes[0].invert_yaxis()      # shared axis: first variant on top
    fig.suptitle(f"Time of one operation, N = {n:,}", x=0.01, ha="left", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_memory(summary: list[dict], out_path: Path) -> None:
    n = max(int(r["n"]) for r in summary)
    size = {r["variant"]: float(r["bytes"]) / 2**20 for r in summary if int(r["n"]) == n}
    present = [v for v in VARIANT_ORDER if v in size]
    if len(present) < 2:
        return
    top = max(size.values())
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(present) + 1.2))
    for y, variant in enumerate(present):
        ax.barh(y, size[variant], height=0.5, color=SERIES)
        ax.text(size[variant] + top * 0.015, y, f"{size[variant]:,.2f} MiB", va="center",
                color=INK, fontsize=9)
    ax.set_xlim(0, top * 1.2)
    ax.set_yticks(range(len(present)), [VARIANT_LABEL[v] for v in present])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("memory, MiB")
    ax.set_title(f"Memory of the structure holding {n:,} strings")
    _save(fig, out_path)


# ---- E2, E3: false positive rate ----

def _pooled(rows: list[dict], key: str) -> list[tuple[float, float, float, float]]:
    """(x, rate, ci_low, ci_high) with the counts of all seeds pooled, in percent."""
    hits: dict[float, int] = defaultdict(int)
    total: dict[float, int] = defaultdict(int)
    for r in rows:
        hits[float(r[key])] += int(r["false_yes"])
        total[float(r[key])] += int(r["queries"])
    points = []
    for x in sorted(hits):
        low, high = wilson(hits[x], total[x])
        points.append((x, 100 * hits[x] / total[x], 100 * low, 100 * high))
    return points


def _measured(ax, points) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    err = [[p[1] - p[2] for p in points], [p[3] - p[1] for p in points]]
    ax.errorbar(xs, ys, yerr=err, fmt="o", color=SERIES, markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, ecolor=SERIES, elinewidth=1.2, capsize=3,
                label="measured (95% interval)", zorder=3)


def plot_fp_fill(rows: list[dict], out_path: Path) -> None:
    n, m, l = int(rows[0]["n"]), int(rows[0]["m"]), int(rows[0]["l"])
    points = _pooled(rows, "load")
    top = max(p[0] for p in points)
    grid = [top * i / 200 for i in range(201)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid, [100 * fp_theory(m, l, x * n) for x in grid], color=INK_SECONDARY, linewidth=2,
            label="theory: $(1 - e^{-ln/m})^l$", solid_capstyle="round")
    _measured(ax, points)
    ax.axhline(1.0, color=REFERENCE, linewidth=0.8)
    ax.axvline(1.0, color=REFERENCE, linewidth=0.8)
    ax.text(0.01, 1.0, " target p = 1%", va="bottom", color=INK_SECONDARY, fontsize=9)
    ax.text(1.0, ax.get_ylim()[1] * 0.97, f" design load n = {n:,} ", va="top", ha="right",
            color=INK_SECONDARY, fontsize=9)
    for x, y, _, _ in points:
        if x >= 0.75:
            ax.annotate(f"{y:.2f}%", (x, y), xytext=(-8, 6), textcoords="offset points",
                        ha="right", color=INK, fontsize=9)
    ax.set_xlim(0, top * 1.03)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("stored strings / n")
    ax.set_ylabel("false positive rate, %")
    ax.set_title(f"False positive rate against the load (m = {m:,} bits, l = {l})")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    _save(fig, out_path)


def plot_fp_hashes(rows: list[dict], out_path: Path) -> None:
    n, m = int(rows[0]["n"]), int(rows[0]["m"])
    points = _pooled(rows, "l")
    ls = [int(p[0]) for p in points]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ls, [100 * fp_theory(m, l, n) for l in ls], color=INK_SECONDARY, linewidth=2,
            label="theory: $(1 - e^{-ln/m})^l$", solid_capstyle="round")
    _measured(ax, points)
    ax.axhline(1.0, color=REFERENCE, linewidth=0.8)
    ax.text(ls[-1], 1.0, "target p = 1% ", va="bottom", ha="right", color=INK_SECONDARY, fontsize=9)
    for x, y, _, _ in points:
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                    color=INK, fontsize=9)
    ax.set_xticks(ls)
    ax.set_ylim(0, max(p[3] for p in points) * 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("l, number of hash functions")
    ax.set_ylabel("false positive rate, %")
    ax.set_title(f"False positive rate against l (m = {m:,} bits, n = {n:,} strings)")
    ax.legend(loc="upper right")
    _save(fig, out_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate the charts of the Bloom filter lab.")
    parser.add_argument("--bench", type=str, default=str(_RESULTS / "bench_summary.csv"))
    parser.add_argument("--fill", type=str, default=str(_RESULTS / "fp_fill.csv"))
    parser.add_argument("--hashes", type=str, default=str(_RESULTS / "fp_hashes.csv"))
    parser.add_argument("--out-dir", type=str, default=str(_RESULTS))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    if Path(args.bench).exists():
        summary = load(Path(args.bench))
        plot_time_per_op(summary, out_dir / "time_per_op.png")
        plot_compare(summary, out_dir / "compare.png")
        plot_memory(summary, out_dir / "memory.png")
        made += 1
    if Path(args.fill).exists():
        plot_fp_fill(load(Path(args.fill)), out_dir / "fp_fill.png")
        made += 1
    if Path(args.hashes).exists():
        plot_fp_hashes(load(Path(args.hashes)), out_dir / "fp_hashes.png")
        made += 1
    if not made:
        print("no CSV files found; run Lab2.tools.benchmark and Lab2.tools.fp_experiment first")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
