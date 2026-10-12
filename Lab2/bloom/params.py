"""Bloom filter parameters (lecture 4, formulas (4.1) and (4.2)).

    m = -n ln p / (ln 2)^2              optimal array size in bits        (4.1)
    l = (m / n) ln 2 = -ln p / ln 2     optimal number of hash functions  (4.2)
    p = (1 - e^(-l n / m))^l            false positive probability

l has to be an integer. Rounding it and keeping m from (4.1) gives p slightly
above the target (1.004% instead of 1% for n = 10**6), so l is rounded up first
and m is then taken from the last formula: "given l, p and n, find the number of bits".
"""

from __future__ import annotations

import math


def _check(n: int, p: float) -> None:
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")


def optimal_bits(n: int, p: float) -> float:
    """Formula (4.1), not rounded."""
    _check(n, p)
    return -n * math.log(p) / math.log(2) ** 2


def optimal_hashes(p: float) -> float:
    """Formula (4.2), not rounded."""
    _check(1, p)
    return -math.log2(p)


def hash_count(p: float) -> int:
    """l: the optimal number of hash functions rounded up."""
    return max(1, math.ceil(optimal_hashes(p)))


def bit_count(n: int, p: float, l: int) -> int:
    """Smallest m for which l hash functions and n elements give at most p."""
    _check(n, p)
    if l < 1:
        raise ValueError("l must be at least 1")
    return math.ceil(-l * n / math.log1p(-p ** (1.0 / l)))


def fp_theory(m: int, l: int, n: int) -> float:
    """False positive probability after n distinct elements were added."""
    return (1.0 - math.exp(-l * n / m)) ** l


def expected_probes(fill: float, l: int) -> float:
    """Mean number of hash evaluations of a negative query: 1 + fill + ... + fill^(l-1)."""
    if fill >= 1.0:
        return float(l)
    return (1.0 - fill ** l) / (1.0 - fill)
