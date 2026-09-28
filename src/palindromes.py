"""Palindrome checks (D1-D3 of the lab plan).

Three independent implementations are kept so they can be cross-checked
against each other (tests/test_palindromes.py) and benchmarked
(tools/benchmark.py):

  - is_pal_hash: the "production" method used by PolyHashSet - a hash
    candidate check (O(1) once hashes are known) confirmed by an exact
    comparison, so the result is always correct regardless of collisions.
  - is_pal_two_pointers: classic O(L) time, O(1) extra space baseline.
  - is_pal_reverse: O(L) time and space baseline using a slice reversal.
"""

from __future__ import annotations


def is_pal_hash(h1: int, h2: int, r1: int, r2: int, w: bytes) -> bool:
    """Hash-based candidate check confirmed by an exact comparison (R6)."""
    return h1 == r1 and h2 == r2 and w == w[::-1]


def is_pal_two_pointers(w: bytes) -> bool:
    i, j = 0, len(w) - 1
    while i < j:
        if w[i] != w[j]:
            return False
        i += 1
        j -= 1
    return True


def is_pal_reverse(w: bytes) -> bool:
    return w == w[::-1]
