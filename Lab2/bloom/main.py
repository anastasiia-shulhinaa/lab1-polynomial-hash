"""CLI: Bloom filter over strings with + / ? operations (and - with counters).

Usage (from the repository root):
    python -m Lab2.bloom.main [--input FILE] [--output FILE]
                              [--filter auto|bloom|counting] [--n N] [--p P] [--seed S]
                              [--trace [FILE]] [--show-bits [FILE]] [--bits-limit N]
                              [--stats] [--timing] [--memory] [--strict]

Without --input/--output it reads stdin and writes stdout. Windows PowerShell has
no `<` input redirection, so use --input/--output there.

Main output (stdout or --output): exactly one "Y"/"N" per '?' operation, in order.
"N" is exact; "Y" may be a false positive (at most --p while at most --n strings
are stored). The hash functions are random per run, so which absent words get a
false "Y" depends on --seed.
Diagnostics (--trace, --show-bits, --stats, --timing, --memory) go to stderr or a file.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import tracemalloc

from src.hashing import MAXLEN
from src.main import find_end

from Lab2.bloom.filters import BloomFilter, CountingBloomFilter

_OPS = {
    "bloom": frozenset((b"+", b"?")),
    "counting": frozenset((b"+", b"?", b"-")),
}
_HAS_MINUS = re.compile(rb"^[ \t]*-[ \t]", re.MULTILINE)


def _valid(op: bytes, w: bytes, ops: frozenset) -> bool:
    # bytes.isalpha() is ASCII-only, so this means w matches [A-Za-z]{1,15}
    return op in ops and len(w) <= MAXLEN and w.isalpha()


def parse(raw: bytes, ops: frozenset, strict: bool) -> tuple[list[bytes], int, bool] | None:
    """Tokens [op, word, ...], their count and whether '#' was seen (None in strict mode).

    Well-formed input is split in one call. Otherwise the input is re-read line by
    line, so that a malformed line is reported and dropped without shifting the rest.
    """
    data = raw.split()
    end = find_end(data)
    if end % 2 == 0 and all(_valid(data[i], data[i + 1], ops) for i in range(0, end, 2)):
        return data, end, end < len(data)
    clean: list[bytes] = []
    terminated = False
    for no, line in enumerate(raw.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if parts == [b"#"]:
            terminated = True
            break
        if len(parts) == 2 and _valid(parts[0], parts[1], ops):
            clean += parts
            continue
        sys.stderr.write(f"warning: malformed line {no}: {line!r}\n")
        if strict:
            return None
    return clean, len(clean), terminated


def trace_ops(f: BloomFilter, data: list[bytes], end: int, emit, terminated: bool) -> list[str]:
    """Run operations one by one and build a Markdown table of what each one did.

    Uses the per-operation methods (same logic as run_ops) - meant for small inputs.
    """
    counting = f.kind == "counting"
    rows = ["| № | Операція | Позиції h0..h(l-1) | Дія над T | Вивід |",
            "|---:|---|---|---|---|"]
    for k, i in enumerate(range(0, end, 2), start=1):
        op, w = data[i], data[i + 1]
        hs = f.positions(w)
        before = [f.cell(h) for h in hs]
        present = all(before)
        answer = "—"
        if op == b"?":
            answer = "Y" if f.contains(w) else "N"
            emit(answer.encode())
            action = "усі комірки ненульові" if present else f"T[{hs[before.index(0)]}] = 0"
        elif op == b"+":
            f.add(w)
            if counting:
                action = "уже є, без змін" if present else "лічильники +1"
            else:
                fresh = len({h for h, c in zip(hs, before) if not c})
                action = f"нових одиниць: {fresh}"
        else:
            f.remove(w)
            action = "лічильники -1" if present else "немає, без змін"
        cells = ", ".join(map(str, hs))
        rows.append(f"| {k} | `{op.decode()} {w.decode()}` | {cells} | {action} | {answer} |")
    if terminated:
        rows.append(f"| {end // 2 + 1} | `#` | | | завершення |")
    return rows


def _emit_text(target: str | None, text: str) -> None:
    """target None or '-' -> stderr, otherwise a file."""
    if target in (None, "-"):
        sys.stderr.write(text)
    else:
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def _probability(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1)")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bloom filter over strings: + / ? (and - with counters).")
    ap.add_argument("--input", default=None, help="read operations from this file instead of stdin")
    ap.add_argument("--output", default=None, help="write Y/N answers to this file instead of stdout")
    ap.add_argument("--filter", choices=("auto", "bloom", "counting"), default="auto",
                    help="bloom: bit array, '+' and '?' only; counting: counters, also '-'; "
                         "auto (default): counting if the input has a '-' operation")
    ap.add_argument("--n", type=_positive, default=10**6,
                    help="maximum number of stored strings (default 1000000)")
    ap.add_argument("--p", type=_probability, default=0.01,
                    help="false positive probability (default 0.01)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the hash functions (default: random per run)")
    ap.add_argument("--trace", nargs="?", const="-", default=None, metavar="FILE",
                    help="table of operations, their cells of T and answers (Markdown; small inputs)")
    ap.add_argument("--show-bits", nargs="?", const="-", default=None, metavar="FILE",
                    help="print the array T after '#'")
    ap.add_argument("--bits-limit", type=int, default=200, help="max cells shown by --show-bits")
    ap.add_argument("--stats", action="store_true", help="filter statistics to stderr")
    ap.add_argument("--timing", action="store_true", help="phase timings to stderr")
    ap.add_argument("--memory", action="store_true",
                    help="peak memory via tracemalloc (slows the run; time it separately)")
    ap.add_argument("--strict", action="store_true", help="exit with code 1 on a malformed operation")
    return ap.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.memory:
        tracemalloc.start()
    t0 = time.perf_counter()

    if args.input:
        with open(args.input, "rb") as f:
            raw = f.read()
    else:
        raw = sys.stdin.buffer.read()
    kind = args.filter
    if kind == "auto":
        kind = "counting" if _HAS_MINUS.search(raw) else "bloom"
    parsed = parse(raw, _OPS[kind], args.strict)
    if parsed is None:
        return 1
    data, end, terminated = parsed
    del raw
    t1 = time.perf_counter()

    cls = CountingBloomFilter if kind == "counting" else BloomFilter
    flt = cls(n=args.n, p=args.p, seed=args.seed)
    t2 = time.perf_counter()

    out: list[bytes] = []
    trace_rows = None
    if args.trace:
        trace_rows = trace_ops(flt, data, end, out.append, terminated)
    else:
        flt.run_ops(data, end, out.append)
    t3 = time.perf_counter()

    answers = b"\n".join(out) + (b"\n" if out else b"")
    if args.output:
        with open(args.output, "wb") as f:
            f.write(answers)
    else:
        sys.stdout.buffer.write(answers)
        sys.stdout.buffer.flush()
    t4 = time.perf_counter()

    if trace_rows is not None:
        _emit_text(args.trace, "\n".join(trace_rows) + "\n")
    if args.show_bits:
        _emit_text(args.show_bits, "\n".join(flt.dump(args.bits_limit)) + "\n")
    if args.stats:
        ops = data[:end:2]
        yes = out.count(b"Y")
        counts = {"plus": ops.count(b"+"), "minus": ops.count(b"-"), "query": len(out),
                  "Y": yes, "N": len(out) - yes}
        sys.stderr.write(f"stats: {flt.stats() | counts}\n")
    if args.timing:
        sys.stderr.write(
            f"timing (s): read+validate={t1 - t0:.4f} init={t2 - t1:.4f} ops={t3 - t2:.4f} "
            f"write={t4 - t3:.4f} total={t4 - t0:.4f}\n"
        )
    if args.memory:
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sys.stderr.write(
            f"memory: filter={len(flt.T) / 2**20:.2f} MiB, "
            f"traced current={cur / 2**20:.1f} MiB peak={peak / 2**20:.1f} MiB\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
