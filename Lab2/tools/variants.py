"""Timing of the separate operations. Each run is an isolated process:

    python -m Lab2.tools.variants --variant bloom --n 100000 --seed 0

and prints one JSON line. A run generates n distinct words to store and n distinct
absent words (not timed), then times four phases, n operations each:

    add   '+' of every stored word
    pos   '?' of every stored word           (all answers Y)
    neg   '?' of every absent word           (Y here is a false positive)
    del   '-' of every stored word           (not for the bit filter)

  bloom     BloomFilter, bit array                      the lab's filter
  counting  CountingBloomFilter, one byte per counter   with deletion
  polyset   PolyHashSet of lab 1 (exact set, pre-sized, no palindrome tracking)
  set       built-in set (exact, lower bound)

For the filters the labour of a phase is counted in probes: one probe is one hash
function evaluated together with its access to T.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from src.hashset import PolyHashSet

from Lab2.bloom.filters import BloomFilter, CountingBloomFilter
from Lab2.tools.generator import present_absent, tokens

VARIANTS = ("bloom", "counting", "polyset", "set")


def _timed(run, data: list[bytes], yes: bytes) -> tuple[float, int]:
    out: list[bytes] = []
    t0 = time.perf_counter()
    run(data, len(data), out.append)
    return time.perf_counter() - t0, out.count(yes)


def run_filter(cls, present: list[bytes], absent: list[bytes], capacity: int, p: float, seed) -> dict:
    n = len(present)
    t0 = time.perf_counter()
    flt = cls(n=capacity, p=p, seed=seed)
    init_time = time.perf_counter() - t0
    add_time, _ = _timed(flt.run_ops, tokens(b"+", present), b"Y")
    stats = flt.stats()
    pos_time, pos_yes = _timed(flt.run_ops, tokens(b"?", present), b"Y")
    neg_time, neg_yes = _timed(flt.run_ops, tokens(b"?", absent), b"Y")
    result = {
        "init_time": init_time, "add_time": add_time, "pos_time": pos_time, "neg_time": neg_time,
        "pos_yes": pos_yes, "neg_yes": neg_yes,
        "add_probes": n * flt.l, "pos_probes": flt.query_probes(present),
        "neg_probes": flt.query_probes(absent),
        "l": flt.l, "m": flt.m, "bytes": stats["bytes"], "fill": stats["fill"],
    }
    if cls is CountingBloomFilter:
        result["del_time"], _ = _timed(flt.run_ops, tokens(b"-", present), b"Y")
        result["del_probes"] = n * flt.l
        result["left_after_del"] = flt.set_cells()
    return result


def run_polyset(present: list[bytes], absent: list[bytes], seed) -> dict:
    t0 = time.perf_counter()
    hs = PolyHashSet(expected=len(present), seed=seed, track_palindromes=False)
    init_time = time.perf_counter() - t0
    add_time, _ = _timed(hs.run_ops, tokens(b"+", present), b"yes")
    memory = hs.approx_memory()
    pos_time, pos_yes = _timed(hs.run_ops, tokens(b"?", present), b"yes")
    neg_time, neg_yes = _timed(hs.run_ops, tokens(b"?", absent), b"yes")
    del_time, _ = _timed(hs.run_ops, tokens(b"-", present), b"yes")
    return {"init_time": init_time, "add_time": add_time, "pos_time": pos_time,
            "neg_time": neg_time, "del_time": del_time, "pos_yes": pos_yes, "neg_yes": neg_yes,
            "bytes": memory, "left_after_del": hs.size}


def _set_ops(s: set, data: list[bytes], end: int, emit) -> None:
    for i in range(0, end, 2):
        op = data[i]
        w = data[i + 1]
        if op == b"?":
            emit(b"Y" if w in s else b"N")
        elif op == b"+":
            s.add(w)
        else:
            s.discard(w)


def run_builtin(present: list[bytes], absent: list[bytes]) -> dict:
    s: set[bytes] = set()

    def run(data, end, emit):
        _set_ops(s, data, end, emit)

    add_time, _ = _timed(run, tokens(b"+", present), b"Y")
    memory = sys.getsizeof(s) + sum(map(sys.getsizeof, s))
    pos_time, pos_yes = _timed(run, tokens(b"?", present), b"Y")
    neg_time, neg_yes = _timed(run, tokens(b"?", absent), b"Y")
    del_time, _ = _timed(run, tokens(b"-", present), b"Y")
    return {"init_time": 0.0, "add_time": add_time, "pos_time": pos_time, "neg_time": neg_time,
            "del_time": del_time, "pos_yes": pos_yes, "neg_yes": neg_yes, "bytes": memory,
            "left_after_del": len(s)}


def run_variant(variant: str, n: int, seed: int | None, capacity: int, p: float) -> dict:
    present, absent = present_absent(n, n, seed)
    if variant == "bloom":
        result = run_filter(BloomFilter, present, absent, capacity, p, seed)
    elif variant == "counting":
        result = run_filter(CountingBloomFilter, present, absent, capacity, p, seed)
    elif variant == "polyset":
        result = run_polyset(present, absent, seed)
    elif variant == "set":
        result = run_builtin(present, absent)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if result["pos_yes"] != n:
        raise RuntimeError(f"{variant}: false negatives ({n - result['pos_yes']} of {n})")
    result.update(variant=variant, n=n, seed=seed, capacity=capacity, p=p)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Time the operations of one variant.")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--n", type=int, required=True, help="operations per phase")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--capacity", type=int, default=10**6, help="n the filter is built for")
    ap.add_argument("--p", type=float, default=0.01)
    args = ap.parse_args(argv)
    result = run_variant(args.variant, args.n, args.seed, args.capacity, args.p)
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
