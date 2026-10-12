"""Benchmark driver: mean time of the filter operations (experiments E1, E5).

Runs each variant as an isolated subprocess (Lab2.tools.variants) across a grid of
N x seed and writes one row per run to a CSV file. The summary follows the empirical
method of the lab manual (time estimates from the labour function):

    f(n_i)     labour of the phase in generalised basic operations (see LABOUR)
    t_e(n_i)   mean measured time over the runs of size n_i
    t_op(n_i)  = t_e(n_i) / f(n_i), mean time of one generalised basic operation
    t_op       mean of t_op(n_i) over the sizes
    forecast   T(n_i) = t_op * f(n_i), compared with the measured time

Examples:
    # quick smoke run
    python -m Lab2.tools.benchmark --sizes 1000 10000 --repeats 2 --out Lab2/results/bench_quick.csv

    # full run (up to N = 10**6)
    python -m Lab2.tools.benchmark --out Lab2/results/bench.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from Lab2.tools.variants import VARIANTS

_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _ROOT / "Lab2" / "results"

DEFAULT_SIZES = (10_000, 100_000, 1_000_000)
PHASES = ("add", "pos", "neg", "del")
PHASE_NAMES = {"add": "+", "pos": "? (present)", "neg": "? (absent)", "del": "-"}

# Labour of one operation = A + B * probes, counted line by line in run_ops; a probe
# is one hash function evaluated together with its access to T.
#
# BloomFilter.run_ops, every operation: loop step (1), k = from_bytes(data[i + 1]) (4),
# data[i] == b"?" (2), inner loop set-up (1) = 8; a query also calls emit (1) = 9.
# One probe: `for a, b` (2), h = (a*k + b) % q % m (5), and either the test
# T[h >> 3] & (1 << (h & 7)) with the branch (6) or the update T[h >> 3] |= ... (6) = 13.
#
# CountingBloomFilter.run_ops: a query costs 10 + 8 per probe (no bit arithmetic);
# an add of a new string 15 + 13 per probe (positions, one presence check, l increments);
# a remove of a stored string 13 + 17 per probe (positions, l checks, l decrements).
LABOUR = {
    "bloom": {"add": (8, 13), "pos": (9, 13), "neg": (9, 13)},
    "counting": {"add": (15, 13), "pos": (10, 8), "neg": (10, 8), "del": (13, 17)},
}

FIELDNAMES = [
    "n", "seed", "variant", "capacity", "p", "l", "m", "bytes", "fill", "init_time",
    "add_time", "pos_time", "neg_time", "del_time",
    "add_probes", "pos_probes", "neg_probes", "del_probes",
    "pos_yes", "neg_yes", "left_after_del",
]
SUMMARY_FIELDS = [
    "variant", "phase", "n", "runs", "mean_s", "median_s", "stdev_s", "us_per_op",
    "probes_per_op", "labour_per_op", "ns_per_basic_op", "forecast_s", "forecast_err_pct",
    "bytes",
]


def run_one(variant: str, n: int, seed: int, capacity: int, p: float) -> dict:
    cmd = [sys.executable, "-m", "Lab2.tools.variants", "--variant", variant, "--n", str(n),
           "--seed", str(seed), "--capacity", str(capacity), "--p", str(p)]
    proc = subprocess.run(cmd, capture_output=True, cwd=str(_ROOT), check=True)
    return json.loads(proc.stdout.decode())


def run_grid(sizes, variants, repeats: int, base_seed: int, capacity: int, p: float,
             out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for n in sizes:
            for rep in range(repeats):
                seed = base_seed + rep
                for variant in variants:
                    result = run_one(variant, n, seed, capacity, p)
                    writer.writerow({k: result[k] for k in FIELDNAMES if k in result})
                    f.flush()
                    phases = " ".join(f"{ph}={result[ph + '_time']:.4f}s"
                                      for ph in PHASES if ph + "_time" in result)
                    print(f"N={n:>8} seed={seed} {variant:<8}: {phases}", flush=True)


def summarize(csv_path: Path) -> list[dict]:
    """Per (variant, phase, n): mean/median time, time per operation and per basic operation."""
    groups: dict[tuple, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups.setdefault((row["variant"], int(row["n"])), []).append(row)

    rows: list[dict] = []
    for (variant, n), runs in sorted(groups.items()):
        for phase in PHASES:
            times = [float(r[phase + "_time"]) for r in runs if r.get(phase + "_time")]
            if not times:
                continue
            mean = statistics.fmean(times)
            row = {"variant": variant, "phase": phase, "n": n, "runs": len(times),
                   "mean_s": mean, "median_s": statistics.median(times),
                   "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                   "us_per_op": mean / n * 1e6,
                   "bytes": statistics.fmean(float(r["bytes"]) for r in runs)}
            if phase in LABOUR.get(variant, {}):
                per_op, per_probe = LABOUR[variant][phase]
                probes = statistics.fmean(float(r[phase + "_probes"]) for r in runs) / n
                row["probes_per_op"] = probes
                row["labour_per_op"] = per_op + per_probe * probes
                row["ns_per_basic_op"] = mean / n / row["labour_per_op"] * 1e9
            rows.append(row)

    # mean time of a basic operation over all sizes, and the forecast it gives for each size
    for variant in LABOUR:
        for phase in PHASES:
            part = [r for r in rows if r["variant"] == variant and r["phase"] == phase]
            if not part:
                continue
            t_op = statistics.fmean(r["ns_per_basic_op"] for r in part)
            for r in part:
                r["forecast_s"] = t_op * r["labour_per_op"] * r["n"] / 1e9
                r["forecast_err_pct"] = (r["forecast_s"] / r["mean_s"] - 1.0) * 100.0
    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'variant':<9} {'operation':<12} {'N':>8} {'mean s':>9} {'median s':>9} {'us/op':>7} "
          f"{'probes':>7} {'f/op':>6} {'ns/basic':>8} {'forecast s':>10} {'err %':>6}")
    for r in rows:
        tail = ""
        if "ns_per_basic_op" in r:
            tail = (f" {r['probes_per_op']:7.3f} {r['labour_per_op']:6.1f} "
                    f"{r['ns_per_basic_op']:8.1f} {r['forecast_s']:10.4f} "
                    f"{r['forecast_err_pct']:6.1f}")
        print(f"{r['variant']:<9} {PHASE_NAMES[r['phase']]:<12} {r['n']:>8} {r['mean_s']:9.4f} "
              f"{r['median_s']:9.4f} {r['us_per_op']:7.3f}{tail}")

    print("\nmean time of one generalised basic operation over all sizes (t_op):")
    for variant in LABOUR:
        parts = []
        for phase in PHASES:
            values = [r["ns_per_basic_op"] for r in rows
                      if r["variant"] == variant and r["phase"] == phase]
            if values:
                parts.append(f"{PHASE_NAMES[phase]}: {statistics.fmean(values):.1f} ns")
        if parts:
            print(f"  {variant:<9} " + ",  ".join(parts))

    print("\nmemory of the structure after the '+' phase:")
    seen = set()
    for r in rows:
        if (r["variant"], r["n"]) not in seen:
            seen.add((r["variant"], r["n"]))
            print(f"  {r['variant']:<9} N={r['n']:>8}: {r['bytes'] / 2**20:9.2f} MiB")


def write_summary(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Time the filter operations (E1, E5).")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="operations per phase")
    parser.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--repeats", type=int, default=5, help="number of seeds per size")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=10**6, help="n the filters are built for")
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--out", type=str, default=str(_RESULTS_DIR / "bench.csv"))
    parser.add_argument("--summary-only", action="store_true",
                        help="do not run anything, summarize an existing --out file")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    if not args.summary_only:
        run_grid(args.sizes, args.variants, args.repeats, args.base_seed, args.capacity, args.p, out)
    rows = summarize(out)
    print_summary([r for r in rows if r["variant"] in args.variants])
    summary_path = out.with_name(out.stem + "_summary.csv")
    write_summary(rows, summary_path)
    print(f"\nruns: {out}\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
