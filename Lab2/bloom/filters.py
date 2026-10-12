"""Bloom filter over strings (lecture 4) and its variant with counters.

    T        bit array of m bits (BloomFilter) or m one-byte counters (CountingBloomFilter)
    l        number of hash functions h_0..h_{l-1} from the universal class H_qm
    InsBlF   add():      T[h_i(x)] <- 1 for i = 0..l-1
    ScinBlF  contains(): False as soon as some T[h_i(x)] is 0, otherwise True

"N" is always exact. "Y" may be a false positive, with probability at most p while
no more than n distinct strings are stored.
"""

from __future__ import annotations

from typing import Callable

from Lab2.bloom.hashing import Q, key_of, make_family, positions
from Lab2.bloom.params import bit_count, hash_count

_COUNTER_MAX = 255


class BloomFilter:
    __slots__ = ("n", "p", "l", "m", "family", "T")

    kind = "bloom"

    def __init__(self, n: int = 10**6, p: float = 0.01, seed: int | None = None,
                 l: int | None = None, m: int | None = None):
        self.n = n
        self.p = p
        self.l = l if l is not None else hash_count(p)
        self.m = m if m is not None else bit_count(n, p, self.l)
        self.family = make_family(self.l, seed)
        self.T = self._alloc()

    def _alloc(self) -> bytearray:
        return bytearray((self.m + 7) // 8)

    def positions(self, w: bytes) -> list[int]:
        return positions(key_of(w), self.family, self.m)

    def cell(self, h: int) -> int:
        return (self.T[h >> 3] >> (h & 7)) & 1

    # ---- public API (one operation per call; readable, used by tests / trace) ----

    def add(self, w: bytes) -> None:
        """InsBlF(T, x). Adding a string twice changes nothing."""
        T = self.T
        for h in self.positions(w):
            T[h >> 3] |= 1 << (h & 7)

    def contains(self, w: bytes) -> bool:
        """ScinBlF(T, x): stops at the first zero bit."""
        T = self.T
        m = self.m
        k = key_of(w)
        for a, b in self.family:
            h = (a * k + b) % Q % m
            if not T[h >> 3] & (1 << (h & 7)):
                return False
        return True

    def query_probes(self, words: list[bytes]) -> int:
        """Hash evaluations contains() performs on these words (labour of the queries)."""
        T = self.T
        m = self.m
        family = self.family
        total = 0
        for w in words:
            k = int.from_bytes(w, "big")
            for a, b in family:
                total += 1
                h = (a * k + b) % Q % m
                if not T[h >> 3] & (1 << (h & 7)):
                    break
        return total

    # ---- fast path: the same logic inlined into one loop ----

    def run_ops(self, data: list[bytes], end: int, emit: Callable[[bytes], None]) -> None:
        """Process tokens data[0:end] = [op, word, op, word, ...].

        Same logic as add/contains, inlined into one loop with local variables
        (as PolyHashSet.run_ops in lab 1). Tokens must already be validated
        (op in +,? ; word in [A-Za-z]{1,15}).
        """
        T = self.T
        m = self.m
        q = Q
        family = self.family
        from_bytes = int.from_bytes
        yes = b"Y"
        no = b"N"

        for i in range(0, end, 2):
            k = from_bytes(data[i + 1], "big")
            if data[i] == b"?":
                for a, b in family:
                    h = (a * k + b) % q % m
                    if not T[h >> 3] & (1 << (h & 7)):
                        emit(no)
                        break
                else:
                    emit(yes)
            else:
                for a, b in family:
                    h = (a * k + b) % q % m
                    T[h >> 3] |= 1 << (h & 7)

    # ---- results / inspection ----

    def set_cells(self) -> int:
        """Number of non-zero cells of T."""
        return int.from_bytes(self.T, "big").bit_count()

    def cells(self, limit: int) -> list[int]:
        return [self.cell(h) for h in range(min(self.m, limit))]

    def stats(self) -> dict:
        used = self.set_cells()
        fill = used / self.m
        return {
            "filter": self.kind,
            "n": self.n,
            "p": self.p,
            "l": self.l,
            "m": self.m,
            "bytes": len(self.T),
            "set_cells": used,
            "fill": round(fill, 6),
            "fp_now": float(f"{fill ** self.l:.6g}"),   # chance of a false Y right now
        }

    def dump(self, limit: int = 200) -> list[str]:
        """Human-readable view of the first `limit` cells of T (rows of 50)."""
        used = self.set_cells()
        lines = [f"T ({self.kind}): m={self.m}, l={self.l}, non-zero cells={used}, "
                 f"fill={used / self.m:.4f}"]
        cells = self.cells(limit)
        sep = "" if max(cells, default=0) < 10 else " "
        for start in range(0, len(cells), 50):
            row = cells[start:start + 50]
            groups = [sep.join(map(str, row[j:j + 10])) for j in range(0, len(row), 10)]
            lines.append(f"{start:>6}: " + "  ".join(groups))
        if self.m > limit:
            lines.append(f"... (showing first {limit} of {self.m} cells)")
        return lines

    def __contains__(self, w: bytes) -> bool:
        return self.contains(w)


class CountingBloomFilter(BloomFilter):
    """Bloom filter with a counter in every cell, which makes deletion possible.

    add() of a string the filter already reports as present changes nothing, so
    "+ x, + x, - x" leaves x absent (set semantics). The filter cannot tell an added
    string from a false positive: such an add is skipped and such a remove decrements
    counters of other strings, which may later produce a false "N". The probability
    of either is bounded by the false positive probability at that moment.
    """

    __slots__ = ()

    kind = "counting"

    def _alloc(self) -> bytearray:
        return bytearray(self.m)

    def cell(self, h: int) -> int:
        return self.T[h]

    def contains(self, w: bytes) -> bool:
        T = self.T
        m = self.m
        k = key_of(w)
        for a, b in self.family:
            if not T[(a * k + b) % Q % m]:
                return False
        return True

    def add(self, w: bytes) -> bool:
        """Increment the l counters; False if the string already looks present."""
        T = self.T
        hs = self.positions(w)
        for h in hs:
            if not T[h]:
                break
        else:
            return False
        for h in hs:
            c = T[h]
            if c < _COUNTER_MAX:        # a saturated counter stays put
                T[h] = c + 1
        return True

    def remove(self, w: bytes) -> bool:
        """Decrement the l counters; False if the string is certainly absent."""
        T = self.T
        hs = self.positions(w)
        for h in hs:
            if not T[h]:
                return False
        for h in hs:
            c = T[h]
            if 0 < c < _COUNTER_MAX:    # never below zero, saturated stays put
                T[h] = c - 1
        return True

    def query_probes(self, words: list[bytes]) -> int:
        T = self.T
        m = self.m
        family = self.family
        total = 0
        for w in words:
            k = int.from_bytes(w, "big")
            for a, b in family:
                total += 1
                if not T[(a * k + b) % Q % m]:
                    break
        return total

    def run_ops(self, data: list[bytes], end: int, emit: Callable[[bytes], None]) -> None:
        """As BloomFilter.run_ops, with the '-' operation (op in +,-,?)."""
        T = self.T
        m = self.m
        q = Q
        family = self.family
        from_bytes = int.from_bytes
        top = _COUNTER_MAX
        yes = b"Y"
        no = b"N"

        for i in range(0, end, 2):
            op = data[i]
            k = from_bytes(data[i + 1], "big")
            if op == b"?":
                for a, b in family:
                    if not T[(a * k + b) % q % m]:
                        emit(no)
                        break
                else:
                    emit(yes)
                continue

            hs = [(a * k + b) % q % m for a, b in family]
            for h in hs:
                if not T[h]:
                    break
            else:
                # looks present: '+' is a no-op, '-' decrements
                if op == b"-":
                    for h in hs:
                        c = T[h]
                        if 0 < c < top:
                            T[h] = c - 1
                continue
            if op == b"+":
                for h in hs:
                    c = T[h]
                    if c < top:
                        T[h] = c + 1

    def set_cells(self) -> int:
        return self.m - self.T.count(0)
