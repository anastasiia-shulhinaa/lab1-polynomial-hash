"""Run the whole demonstration for the defense, step by step.

    python -m Lab2.tools.demo                 # all steps
    python -m Lab2.tools.demo --sizes 10000   # smaller scale step
    python -m Lab2.tools.demo --no-pause      # do not wait for Enter between steps

Every step prints the exact command it runs, so each one can also be typed by hand.
Files produced in Lab2/results/: demo_answers.txt, demo_trace.md, scale_<N>_answers.txt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from Lab2.tools.generator import PROFILES, generate

_ROOT = Path(__file__).resolve().parent.parent.parent
_LAB = _ROOT / "Lab2"
_RESULTS = _LAB / "results"
_DATA = _LAB / "data"


def ensure_dataset(n: int, profile: str, seed: int) -> Path:
    path = _DATA / f"n{n}_{profile}_s{seed}.txt"
    if not path.exists():
        lines = generate(n=n, min_len=1, max_len=15, mixed_case=False, seed=seed, **PROFILES[profile])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return path


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _run(args: list[str]) -> None:
    shown = " ".join(a if " " not in a else f'"{a}"' for a in args)
    print(f"\n> python -m Lab2.bloom.main {shown}\n", flush=True)
    subprocess.run([sys.executable, "-m", "Lab2.bloom.main", *args], cwd=str(_ROOT), check=True)


def _show_file(path: Path) -> None:
    print(f"\n--- {path.relative_to(_ROOT).as_posix()} ---", flush=True)
    sys.stdout.write(path.read_text(encoding="utf-8"))
    sys.stdout.flush()


def _pause(enabled: bool) -> None:
    if enabled:
        input("\n[Enter] - next step ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step-by-step demonstration of the lab.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args(argv)
    pause = not args.no_pause
    _RESULTS.mkdir(exist_ok=True)

    _header("1. Small example: + ? # -> answers file, trace of the cells each operation touches")
    _show_file(_LAB / "demo" / "demo_small.txt")
    _run(["--input", "Lab2/demo/demo_small.txt", "--seed", "1",
          "--output", "Lab2/results/demo_answers.txt",
          "--trace", "Lab2/results/demo_trace.md", "--stats"])
    for name in ("demo_trace.md", "demo_answers.txt"):
        _show_file(_RESULTS / name)
    _pause(pause)

    _header("2. False positive: 10 strings in a filter of 49 bits (n = 10, p = 0.1)")
    print("'banana' was never added, but all its bits were set by other strings -> Y.")
    _show_file(_LAB / "demo" / "demo_bits.txt")
    _run(["--input", "Lab2/demo/demo_bits.txt", "--n", "10", "--p", "0.1", "--seed", "6",
          "--trace", "--show-bits"])
    print("\nThe same input with the lab parameters (n = 10^6, p = 0.01): the answer is exact.")
    _run(["--input", "Lab2/demo/demo_bits.txt", "--seed", "6", "--stats"])
    _pause(pause)

    _header("3. Deletion with counters: duplicate +, - of a stored and of a missing string")
    _show_file(_LAB / "demo" / "demo_delete.txt")
    _run(["--input", "Lab2/demo/demo_delete.txt", "--n", "10", "--p", "0.1", "--seed", "1",
          "--trace", "--show-bits"])
    _pause(pause)

    _header("4. Malformed lines are reported and skipped; both letter cases are accepted")
    _show_file(_LAB / "demo" / "demo_malformed.txt")
    _run(["--input", "Lab2/demo/demo_malformed.txt", "--seed", "1"])
    _pause(pause)

    _header("5. Scale: generated inputs, 10^4 .. 10^6 operations (random hash functions)")
    for n in args.sizes:
        data = ensure_dataset(n, "default", 0).relative_to(_ROOT).as_posix()
        _run(["--input", data, "--output", f"Lab2/results/scale_{n}_answers.txt",
              "--timing", "--stats"])
    print("\nCharts and tables: Lab2/results/*.png, bench_summary.csv, fp_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
