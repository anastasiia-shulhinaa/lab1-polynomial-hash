"""Polynomial hashing over the a..z alphabet.

h(S) = (a0*p^(n-1) + a1*p^(n-2) + ... + a(n-1)*p^0) mod m, a_i = code - 96
(so 'a' -> 1 ... 'z' -> 26), computed with a double modulus
m1 = 1000000123 (prime) and m2 = 2**64. The reverse string hash equals
r(S) = a0*p^0 + ... + a(n-1)*p^(n-1), so S is a palindrome candidate iff
h(S) == r(S).

Two implementations are kept:
  - reference (hash_fwd_rev, hash_fwd, hash_single): reduce modulo m on every
    step exactly as in the lecture formulas;
  - fast (poly_value / poly_fwd_rev / hash_pair): Python ints are unbounded,
    so the exact polynomial (< 2**460 for n <= 15) is built first and reduced
    once. (x mod m is a ring homomorphism, so the results are identical; the
    tests check this.) This is ~3x faster in CPython and is what the set uses.
    poly_fwd_rev gives h(w) and h(reverse(w)) in one pass with no reversed copy.
"""

from __future__ import annotations

import random

M1 = 1_000_000_123          # fixed prime modulus
MASK = (1 << 64) - 1        # x & MASK == x mod 2**64 (Python ints do not wrap)
MAXLEN = 15
ALPHA_MAX = 26              # max(a_i)
CODE_SHIFT = 96             # ord('a') - 1


def choose_base(seed: int | None = None) -> int:
    """Random odd base p with 26 < p < M1 (fresh per run unless seeded)."""
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    p = rng.randrange(ALPHA_MAX + 1, M1) | 1    # odd -> gcd(p, 2**64) == 1
    if p >= M1:
        p -= 2
    return p


def powers(p: int, maxlen: int = MAXLEN) -> tuple[list[int], list[int]]:
    """p^0..p^maxlen modulo M1 and modulo 2**64."""
    pw1 = [1] * (maxlen + 1)
    pw2 = [1] * (maxlen + 1)
    for i in range(1, maxlen + 1):
        pw1[i] = pw1[i - 1] * p % M1
        pw2[i] = pw2[i - 1] * p & MASK
    return pw1, pw2


def code_offsets(p: int, maxlen: int = MAXLEN) -> list[int]:
    """off[n] = 96 * (p^0 + ... + p^(n-1)).

    Summing raw byte codes and subtracting off[len(w)] once gives the same
    polynomial as subtracting 96 from every character.
    """
    off = [0] * (maxlen + 1)
    geo = 0
    pk = 1
    for n in range(1, maxlen + 1):
        geo += pk
        pk *= p
        off[n] = CODE_SHIFT * geo
    return off


def poly_powers(p: int, maxlen: int = MAXLEN) -> list[int]:
    """Exact powers p^0..p^maxlen (unreduced)."""
    return [p ** i for i in range(maxlen + 1)]


def poly_value(w: bytes, p: int, off: list[int]) -> int:
    """Exact (unreduced) value of a0*p^(n-1) + ... + a(n-1)."""
    h = 0
    for c in w:
        h = h * p + c
    return h - off[len(w)]


def poly_fwd_rev(w: bytes, p: int, pw: list[int], off: list[int]) -> tuple[int, int]:
    """Exact h(w) and h(reverse(w)) in ONE pass, without building reverse(w).

    h(w)          = a0*p^(n-1) + ... + a(n-1)*p^0   (Horner, left to right)
    h(reverse(w)) = a0*p^0     + ... + a(n-1)*p^(n-1)  (a_i times p^i)
    """
    h = r = 0
    for c, q in zip(w, pw):
        h = h * p + c
        r += c * q
    n = len(w)
    return h - off[n], r - off[n]


def poly_reverse(w: bytes, pw: list[int], off: list[int]) -> int:
    """Exact h(reverse(w)) without building reverse(w)."""
    r = 0
    for c, q in zip(w, pw):
        r += c * q
    return r - off[len(w)]


def hash_pair(w: bytes, p: int, off: list[int]) -> tuple[int, int]:
    """Double hash (h mod m1, h mod 2**64)."""
    h = poly_value(w, p, off)
    return h % M1, h & MASK


def pack_key(h1: int, h2: int) -> int:
    """Both hashes in one int, so a single comparison checks both."""
    return (h1 << 64) | h2


def bucket_index(h1: int, h2: int, mask_cap: int) -> int:
    """Low bits of a mod-2**64 hash are poorly mixed; fold in high bits of h2."""
    return (h1 ^ (h2 >> 32)) & mask_cap


# ---- reference implementations (lecture formulas, reduce every step) ----

def hash_fwd_rev(
    w: bytes, p: int, pw1: list[int], pw2: list[int]
) -> tuple[int, int, int, int]:
    """(h1, h2, r1, r2) in one pass, where r = h(reverse(w))."""
    h1 = h2 = r1 = r2 = 0
    i = 0
    for c in w:
        a = c - CODE_SHIFT
        h1 = (h1 * p + a) % M1
        h2 = (h2 * p + a) & MASK
        r1 = (r1 + a * pw1[i]) % M1
        r2 = (r2 + a * pw2[i]) & MASK
        i += 1
    return h1, h2, r1, r2


def hash_fwd(w: bytes, p: int) -> tuple[int, int]:
    h1 = h2 = 0
    for c in w:
        a = c - CODE_SHIFT
        h1 = (h1 * p + a) % M1
        h2 = (h2 * p + a) & MASK
    return h1, h2


def hash_single(w: bytes, p: int) -> int:
    h1 = 0
    for c in w:
        h1 = (h1 * p + c - CODE_SHIFT) % M1
    return h1


def hash_bad(w: bytes) -> int:
    """Deliberately poor hash (length only) to demonstrate collisions."""
    return len(w)
