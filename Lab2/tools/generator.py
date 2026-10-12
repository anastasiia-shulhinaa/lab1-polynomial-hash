"""Generate inputs for the Bloom filter lab.

Operation files ('+', '?', '-' lines ending with '#'):
    python -m Lab2.tools.generator --n 100000 --seed 1 --out Lab2/data/n100000_s1.txt
    python -m Lab2.tools.generator --n 100000 --profile delete_heavy --seed 1 --out ...

The experiments use the word sets directly: present_absent() gives distinct
strings to add and distinct strings that are guaranteed not to be among them.
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

from tools.generator import random_word

_MIXED = string.ascii_letters

# Ready-made profiles: shares of the operations, P(query/remove hits an added word),
# P(add repeats an added word)
PROFILES = {
    "default": dict(plus=0.5, query=0.5, minus=0.0, hit=0.5, repeat=0.1),
    "query_heavy": dict(plus=0.2, query=0.8, minus=0.0, hit=0.5, repeat=0.1),
    "delete": dict(plus=0.5, query=0.3, minus=0.2, hit=0.5, repeat=0.1),
    "delete_heavy": dict(plus=0.4, query=0.2, minus=0.4, hit=0.5, repeat=0.1),
}


def _word(rng: random.Random, min_len: int, max_len: int, mixed_case: bool) -> str:
    if mixed_case:
        return "".join(rng.choices(_MIXED, k=rng.randint(min_len, max_len)))
    return random_word(rng, min_len, max_len)


def distinct_words(rng: random.Random, count: int, min_len: int = 1, max_len: int = 15,
                   mixed_case: bool = False, banned: set[bytes] | None = None) -> list[bytes]:
    """`count` different words, none of them in `banned`, in generation order."""
    seen: set[bytes] = set()
    words: list[bytes] = []
    while len(words) < count:
        w = _word(rng, min_len, max_len, mixed_case).encode()
        if w in seen or (banned is not None and w in banned):
            continue
        seen.add(w)
        words.append(w)
    return words


def present_absent(n_present: int, n_absent: int, seed: int | None, min_len: int = 1,
                   max_len: int = 15, mixed_case: bool = False) -> tuple[list[bytes], list[bytes]]:
    """Distinct words to add and distinct words that are not among them."""
    rng = random.Random(seed)
    present = distinct_words(rng, n_present, min_len, max_len, mixed_case)
    absent = distinct_words(rng, n_absent, min_len, max_len, mixed_case, banned=set(present))
    return present, absent


def tokens(op: bytes, words: list[bytes]) -> list[bytes]:
    """[op, w0, op, w1, ...] - the token list run_ops expects."""
    data = [op] * (2 * len(words))
    data[1::2] = words
    return data


def generate(n: int, plus: float, query: float, minus: float, hit: float, repeat: float,
             min_len: int, max_len: int, mixed_case: bool, seed: int | None) -> list[str]:
    rng = random.Random(seed)
    total = plus + query + minus
    weights = [plus / total, query / total, minus / total]
    added: list[str] = []
    lines = []
    for _ in range(n):
        op = rng.choices(("+", "?", "-"), weights=weights)[0]
        chance = repeat if op == "+" else hit
        if added and rng.random() < chance:
            word = rng.choice(added)
        else:
            word = _word(rng, min_len, max_len, mixed_case)
        if op == "+":
            added.append(word)
        lines.append(f"{op} {word}")
    lines.append("#")
    return lines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate an input file for the Bloom filter lab.")
    parser.add_argument("--n", type=int, required=True, help="number of operations")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None,
                        help="use a ready-made profile (overrides the individual ratios)")
    parser.add_argument("--plus", type=float, default=0.5)
    parser.add_argument("--query", type=float, default=0.5)
    parser.add_argument("--minus", type=float, default=0.0)
    parser.add_argument("--hit", type=float, default=0.5,
                        help="share of '?' and '-' that use an already added word")
    parser.add_argument("--repeat", type=float, default=0.1,
                        help="share of '+' that repeat an already added word")
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=15)
    parser.add_argument("--mixed-case", action="store_true", help="use both a-z and A-Z")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = dict(plus=args.plus, query=args.query, minus=args.minus, hit=args.hit, repeat=args.repeat)
    if args.profile:
        cfg.update(PROFILES[args.profile])
    lines = generate(n=args.n, min_len=args.min_len, max_len=args.max_len,
                     mixed_case=args.mixed_case, seed=args.seed, **cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"wrote {len(lines) - 1} operations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
