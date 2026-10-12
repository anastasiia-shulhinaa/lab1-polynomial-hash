"""Hash functions of the filter: l independent members of the universal class H_qm.

    h_ab(k) = ((a*k + b) mod q) mod m

The class is defined for integer keys 0 <= k <= q - 1, so a string is first mapped
to an integer by the exact (unreduced) polynomial with base 256,

    k(x) = a0*256^(n-1) + a1*256^(n-2) + ... + a(n-1),   a_i = character code,

which is the polynomial of lab 1 (src.hashing.poly_value) and equals
int.from_bytes(x, "big"). Character codes are non-zero, so different strings give
different keys: there are no collisions at this step. Keys are below 256**15 = 2**120,
hence the prime q = 2**127 - 1 covers all of them. (The textbook calls this prime p;
it is q here because p is the false positive probability.)
"""

from __future__ import annotations

import random

from src.hashing import MAXLEN

Q = (1 << 127) - 1          # Mersenne prime, larger than every key
KEY_LIMIT = 1 << (8 * MAXLEN)

Family = list[tuple[int, int]]


def key_of(w: bytes) -> int:
    """Exact polynomial of the string in base 256 (injective)."""
    return int.from_bytes(w, "big")


def make_family(l: int, seed: int | None = None) -> Family:
    """l independent pairs (a, b), each uniform in 1..q-1 (fresh per run unless seeded).

    The textbook prints a in Z_q, b in Z_q*, CLRS has a in Z_q*, b in Z_q; taking both
    from Z_q* fits either and excludes the constant function a = 0.
    """
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    return [(rng.randrange(1, Q), rng.randrange(1, Q)) for _ in range(l)]


def positions(k: int, family: Family, m: int) -> list[int]:
    """h_0(k), ..., h_{l-1}(k)."""
    return [(a * k + b) % Q % m for a, b in family]
