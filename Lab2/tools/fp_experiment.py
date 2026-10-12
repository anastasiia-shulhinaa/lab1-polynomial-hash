"""False answers of the filter measured against exact data (experiments E2-E4).

    fill      E2: false positive rate at different loads of a filter built for n strings
    hashes    E3: false positive rate for l = 3..10 hash functions, array size fixed
    counting  E4: counting filter on streams with deletions against the exact set of lab 1

Examples:
    python -m Lab2.tools.fp_experiment --experiment fill --out Lab2/results/fp_fill.csv
    python -m Lab2.tools.fp_experiment --experiment hashes --out Lab2/results/fp_hashes.csv
    python -m Lab2.tools.fp_experiment --experiment counting --out Lab2/results/fp_counting.csv

    # quick smoke run on a filter for 10**4 strings
    python -m Lab2.tools.fp_experiment --experiment fill --n 10000 --queries 100000 --repeats 1 \\
        --out Lab2/results/fp_fill_quick.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from src.hashset import PolyHashSet

from Lab2.bloom.filters import BloomFilter, CountingBloomFilter
from Lab2.bloom.params import bit_count, fp_theory, hash_count
from Lab2.tools.generator import PROFILES, distinct_words, generate, tokens

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_LOADS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25)
DEFAULT_HASHES = tuple(range(3, 11))
DEFAULT_CAPACITIES = (1_000_000, 300_000, 100_000)


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval of a proportion (Wilson score)."""
    if total == 0:
        return 0.0, 0.0
    rate = hits / total
    denom = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denom
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _false_yes(flt: BloomFilter, query: list[bytes]) -> int:
    out: list[bytes] = []
    flt.run_ops(query, len(query), out.append)
    return out.count(b"Y")


def _row(flt: BloomFilter, seed: int, added: int, queries: int, false_yes: int) -> dict:
    low, high = wilson(false_yes, queries)
    stats = flt.stats()
    return {"n": flt.n, "m": flt.m, "l": flt.l, "seed": seed, "added": added,
            "load": added / flt.n, "queries": queries, "false_yes": false_yes,
            "rate": false_yes / queries, "ci_low": low, "ci_high": high,
            "theory": fp_theory(flt.m, flt.l, added),
            "fill": stats["fill"], "fill_theory": 1 - math.exp(-flt.l * added / flt.m)}


FILL_FIELDS = ["n", "m", "l", "seed", "added", "load", "queries", "false_yes", "rate",
               "ci_low", "ci_high", "theory", "fill", "fill_theory"]


def run_fill(n: int, p: float, loads, queries: int, repeats: int, base_seed: int) -> list[dict]:
    """One filter per seed; words are added up to each load in turn, then queried."""
    rows = []
    for rep in range(repeats):
        seed = base_seed + rep
        rng = random.Random(seed)
        present = distinct_words(rng, round(max(loads) * n))
        query = tokens(b"?", distinct_words(rng, queries, banned=set(present)))
        flt = BloomFilter(n=n, p=p, seed=seed)
        done = 0
        for load in sorted(loads):
            upto = round(load * n)
            flt.run_ops(tokens(b"+", present[done:upto]), 2 * (upto - done), None)
            done = upto
            row = _row(flt, seed, done, queries, _false_yes(flt, query))
            rows.append(row)
            print(f"seed={seed} added={done:>8} ({load:4.2f} n): rate={row['rate']:.5f} "
                  f"theory={row['theory']:.5f} fill={row['fill']:.4f}", flush=True)
    return rows


def run_hashes(n: int, p: float, hashes, queries: int, repeats: int, base_seed: int) -> list[dict]:
    """Array size fixed at the lab's m; n strings added; l varies."""
    m = bit_count(n, p, hash_count(p))
    rows = []
    for rep in range(repeats):
        seed = base_seed + rep
        rng = random.Random(seed)
        present = distinct_words(rng, n)
        add = tokens(b"+", present)
        query = tokens(b"?", distinct_words(rng, queries, banned=set(present)))
        for l in hashes:
            flt = BloomFilter(n=n, p=p, seed=seed, l=l, m=m)
            flt.run_ops(add, len(add), None)
            row = _row(flt, seed, n, queries, _false_yes(flt, query))
            rows.append(row)
            print(f"seed={seed} l={l:>2}: rate={row['rate']:.5f} theory={row['theory']:.5f}",
                  flush=True)
    return rows


COUNTING_FIELDS = ["n_ops", "profile", "capacity", "m", "l", "seed", "distinct_added",
                   "final_size", "queries", "false_yes", "false_no", "false_yes_rate",
                   "false_no_rate", "fill"]


def run_counting(n_ops: int, p: float, profiles, capacities, repeats: int,
                 base_seed: int) -> list[dict]:
    """The same stream through the counting filter and through the exact set of lab 1."""
    rows = []
    for profile in profiles:
        for rep in range(repeats):
            seed = base_seed + rep
            lines = generate(n=n_ops, min_len=1, max_len=15, mixed_case=False, seed=seed,
                             **PROFILES[profile])
            data = "\n".join(lines[:-1]).encode().split()
            distinct_added = len({data[i + 1] for i in range(0, len(data), 2) if data[i] == b"+"})

            exact = PolyHashSet(expected=distinct_added, seed=seed, track_palindromes=False)
            truth: list[bytes] = []
            exact.run_ops(data, len(data), truth.append)

            for capacity in capacities:
                flt = CountingBloomFilter(n=capacity, p=p, seed=seed)
                out: list[bytes] = []
                flt.run_ops(data, len(data), out.append)
                false_yes = sum(1 for a, t in zip(out, truth) if a == b"Y" and t == b"no")
                false_no = sum(1 for a, t in zip(out, truth) if a == b"N" and t == b"yes")
                absent = truth.count(b"no")
                row = {"n_ops": n_ops, "profile": profile, "capacity": capacity, "m": flt.m,
                       "l": flt.l, "seed": seed, "distinct_added": distinct_added,
                       "final_size": exact.size, "queries": len(truth), "false_yes": false_yes,
                       "false_no": false_no,
                       "false_yes_rate": false_yes / absent if absent else 0.0,
                       "false_no_rate": false_no / (len(truth) - absent) if len(truth) > absent else 0.0,
                       "fill": flt.stats()["fill"]}
                rows.append(row)
                print(f"{profile:<12} seed={seed} capacity={capacity:>8}: queries={len(truth)} "
                      f"false Y={false_yes} false N={false_no} (stored {exact.size})", flush=True)
    return rows


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="False positive / false negative experiments.")
    parser.add_argument("--experiment", choices=("fill", "hashes", "counting"), required=True)
    parser.add_argument("--n", type=int, default=10**6,
                        help="fill, hashes: n the filter is built for; counting: number of operations")
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--queries", type=int, default=10**6, help="absent words queried (fill, hashes)")
    parser.add_argument("--loads", type=float, nargs="+", default=list(DEFAULT_LOADS),
                        help="fill: shares of n to add")
    parser.add_argument("--hashes", type=int, nargs="+", default=list(DEFAULT_HASHES),
                        help="hashes: values of l")
    parser.add_argument("--profiles", type=str, nargs="+", default=["delete", "delete_heavy"],
                        choices=sorted(PROFILES), help="counting: stream profiles")
    parser.add_argument("--capacities", type=int, nargs="+", default=list(DEFAULT_CAPACITIES),
                        help="counting: n the filter is built for")
    parser.add_argument("--repeats", type=int, default=3, help="number of seeds")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out) if args.out else _RESULTS_DIR / f"fp_{args.experiment}.csv"
    if args.experiment == "fill":
        rows = run_fill(args.n, args.p, args.loads, args.queries, args.repeats, args.base_seed)
        write_csv(rows, FILL_FIELDS, out)
    elif args.experiment == "hashes":
        rows = run_hashes(args.n, args.p, args.hashes, args.queries, args.repeats, args.base_seed)
        write_csv(rows, FILL_FIELDS, out)
    else:
        rows = run_counting(args.n, args.p, args.profiles, args.capacities, args.repeats,
                            args.base_seed)
        write_csv(rows, COUNTING_FIELDS, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
