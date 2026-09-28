"""CLI: set of strings with + / - / ? operations; finds all palindromes of the final set.

Usage:
    python -m src.main [--input FILE] [--output FILE] [--pal-out FILE]
                        [--mode incremental|scan] [--prealloc] [--unsorted]
                        [--trace [FILE]] [--show-table [FILE]] [--table-limit N]
                        [--seed N] [--base P] [--stats] [--timing] [--memory] [--strict]

Without --input/--output it reads stdin and writes stdout. Windows PowerShell has
no `<` input redirection, so use --input/--output there.

Main output (stdout or --output): exactly one "yes"/"no" per '?' operation, in order.
Palindromes of the final set: --pal-out FILE (one word per line), otherwise stderr.
Diagnostics (--trace, --show-table, --stats, --timing, --memory) go to stderr or a file.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc

from src.hashing import MAXLEN
from src.hashset import PolyHashSet

_OPS = frozenset((b"+", b"-", b"?"))


def _valid(op: bytes, w: bytes) -> bool:
    # bytes.isalpha() is ASCII-only; with islower() this means w matches [a-z]+
    return op in _OPS and len(w) <= MAXLEN and w.isalpha() and w.islower()


def find_end(data: list[bytes]) -> int:
    """Index of the '#' terminator (or len(data) if it is missing)."""
    i = 0
    n = len(data)
    while i < n:
        if data[i] == b"#":
            return i
        i += 2
    return n


def validate(data: list[bytes], end: int, strict: bool) -> tuple[list[bytes], int] | None:
    """Drop malformed operations with a warning on stderr (None in strict mode)."""
    if end % 2 == 0 and all(_valid(data[i], data[i + 1]) for i in range(0, end, 2)):
        return data, end
    clean: list[bytes] = []
    for k, i in enumerate(range(0, end, 2), start=1):
        op = data[i]
        w = data[i + 1] if i + 1 < end else b""
        if w and _valid(op, w):
            clean.append(op)
            clean.append(w)
            continue
        sys.stderr.write(f"warning: malformed operation #{k}: {op!r} {w!r}\n")
        if strict:
            return None
    return clean, len(clean)


def _set_repr(hs: PolyHashSet, max_words: int = 10) -> str:
    words = sorted(w.decode() for w in hs.words())
    if len(words) > max_words:
        return "{" + ", ".join(words[:max_words]) + f", ... (+{len(words) - max_words})" + "}"
    return "{" + ", ".join(words) + "}"


def trace_ops(hs: PolyHashSet, data: list[bytes], end: int, emit, terminated: bool) -> list[str]:
    """Run operations one by one and build a Markdown table of the set state.

    Uses the per-operation methods (same logic as run_ops) - meant for small inputs.
    """
    rows = ["| № | Операція | Стан множини після операції | Вивід |",
            "|---:|---|---|---|"]
    for k, i in enumerate(range(0, end, 2), start=1):
        op, w = data[i], data[i + 1]
        answer = "—"
        if op == b"+":
            hs.add(w)
        elif op == b"-":
            hs.remove(w)
        else:
            answer = "yes" if hs.contains(w) else "no"
            emit(answer.encode())
        rows.append(f"| {k} | `{op.decode()} {w.decode()}` | `{_set_repr(hs)}` | {answer} |")
    if terminated:
        rows.append(f"| {end // 2 + 1} | `#` | `{_set_repr(hs)}` | завершення |")
    return rows


def _emit_text(target: str | None, text: str) -> None:
    """target None or '-' -> stderr, otherwise a file."""
    if target in (None, "-"):
        sys.stderr.write(text)
    else:
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="String set on a polynomial-hash table; finds palindromes.")
    ap.add_argument("--input", default=None, help="read operations from this file instead of stdin")
    ap.add_argument("--output", default=None, help="write yes/no answers to this file instead of stdout")
    ap.add_argument("--pal-out", default=None,
                    help="write palindromes of the final set to this file, one per line (default: stderr)")
    ap.add_argument("--mode", choices=("incremental", "scan"), default="incremental",
                    help="incremental (default): is_pal decided once on a successful '+'; "
                         "scan: palindromes checked after '#' for the final set")
    ap.add_argument("--prealloc", action="store_true",
                    help="pre-size the table from the number of '+' operations (no resizes); "
                         "default is dynamic resizing")
    ap.add_argument("--unsorted", action="store_true", help="print palindromes in table order (no sorting)")
    ap.add_argument("--trace", nargs="?", const="-", default=None, metavar="FILE",
                    help="table of operations and set state after each one (Markdown; small inputs)")
    ap.add_argument("--show-table", nargs="?", const="-", default=None, metavar="FILE",
                    help="print the non-empty buckets of the hash table after '#'")
    ap.add_argument("--table-limit", type=int, default=50, help="max buckets shown by --show-table")
    ap.add_argument("--seed", type=int, default=None, help="seed for base p (default: random per run)")
    ap.add_argument("--base", type=int, default=None,
                    help="force base p (demo only; --base 1 makes all anagrams collide)")
    ap.add_argument("--stats", action="store_true", help="hash-table statistics to stderr")
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
            data = f.read().split()
    else:
        data = sys.stdin.buffer.read().split()
    end = find_end(data)
    terminated = end < len(data)
    checked = validate(data, end, args.strict)
    if checked is None:
        return 1
    data, end = checked
    t1 = time.perf_counter()

    expected = data[:end:2].count(b"+") if args.prealloc else 0
    hs = PolyHashSet(expected=expected, p=args.base, seed=args.seed,
                     track_palindromes=args.mode == "incremental")
    out: list[bytes] = []
    trace_rows = None
    if args.trace:
        trace_rows = trace_ops(hs, data, end, out.append, terminated)
    else:
        hs.run_ops(data, end, out.append)
    t2 = time.perf_counter()

    answers = b"\n".join(out) + (b"\n" if out else b"")
    if args.output:
        with open(args.output, "wb") as f:
            f.write(answers)
    else:
        sys.stdout.buffer.write(answers)
        sys.stdout.buffer.flush()
    t3 = time.perf_counter()

    pals = hs.palindromes()
    if not args.unsorted:
        pals.sort()
    t4 = time.perf_counter()

    words = b"\n".join(pals) + (b"\n" if pals else b"")
    if args.pal_out:
        with open(args.pal_out, "wb") as f:
            f.write(words)
    else:
        sys.stderr.buffer.write(f"palindromes ({len(pals)}):\n".encode() + words)
        sys.stderr.buffer.flush()

    if trace_rows is not None:
        _emit_text(args.trace, "\n".join(trace_rows) + "\n")
    if args.show_table:
        _emit_text(args.show_table, "\n".join(hs.dump(args.table_limit)) + "\n")
    if args.stats:
        sys.stderr.write(f"stats: {hs.stats()}\n")
    if args.timing:
        sys.stderr.write(
            f"timing (s): read+validate={t1 - t0:.4f} ops={t2 - t1:.4f} write={t3 - t2:.4f} "
            f"palindromes={t4 - t3:.4f} total={t4 - t0:.4f}\n"
        )
    if args.memory:
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sys.stderr.write(
            f"memory: set~{hs.approx_memory() / 2**20:.1f} MiB, "
            f"traced current={cur / 2**20:.1f} MiB peak={peak / 2**20:.1f} MiB\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
