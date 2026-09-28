"""Set of strings on a separate-chaining hash table keyed by a double
polynomial hash, with palindrome detection through the reverse-string hash.

The table stores the WHOLE current set (every word, not only palindromes).
Node layout: (key, word, is_pal), key = pack_key(h mod m1, h mod 2**64).
Lookup order in a bucket: key (both hashes, one int comparison) -> exact word.
Equal hashes are never trusted alone, so answers are deterministic.

Palindrome modes:
  track_palindromes=True ("incremental", default): on a successful '+' the
      forward and reverse hashes are computed in one pass, is_pal is decided
      once and stored in the node; '-' keeps pal_count in sync. After '#' the
      palindromes are just the nodes with is_pal=True (no re-checking).
  track_palindromes=False ("scan"): nothing is decided on insert; reverse
      hashes are computed after '#' for the words still in the set.

Capacity: dynamic by default (starts at 16, doubles at load factor 0.75).
Passing expected=K pre-sizes the table (no resizes) - used for benchmarks.
"""

from __future__ import annotations

import sys
from typing import Callable, Iterator

from src.hashing import (
    M1, MASK, bucket_index, choose_base, code_offsets, pack_key, poly_fwd_rev,
    poly_powers, poly_reverse, poly_value,
)

_MIN_CAP = 16
_LOAD_FACTOR = 0.75

Node = tuple[int, bytes, bool]


def _cap_for(expected: int) -> int:
    cap = _MIN_CAP
    while cap * _LOAD_FACTOR < expected:
        cap <<= 1
    return cap


class PolyHashSet:
    __slots__ = (
        "p", "pw", "off", "table", "cap", "mask_cap", "limit", "size", "resizes",
        "track", "pal_count", "key_collisions", "pal_false_candidates",
    )

    def __init__(self, expected: int = 0, p: int | None = None, seed: int | None = None,
                 track_palindromes: bool = True):
        self.p = p if p is not None else choose_base(seed)
        self.pw = poly_powers(self.p)
        self.off = code_offsets(self.p)
        self.cap = _cap_for(expected)
        self.mask_cap = self.cap - 1
        self.limit = int(self.cap * _LOAD_FACTOR)
        self.table: list[list[Node] | None] = [None] * self.cap
        self.size = 0
        self.resizes = 0
        self.track = track_palindromes
        self.pal_count = 0              # kept exact in incremental mode
        self.key_collisions = 0         # equal (h1, h2) but different words
        self.pal_false_candidates = 0   # h(w) == h(reverse(w)) but not a palindrome

    # ---- helpers ----

    def _key_idx(self, h: int) -> tuple[int, int]:
        h1 = h % M1
        h2 = h & MASK
        return pack_key(h1, h2), bucket_index(h1, h2, self.mask_cap)

    def _confirm_palindrome(self, w: bytes, key: int, r: int) -> bool:
        """key = hash of w, r = exact hash of reverse(w).

        Different hashes prove w is not a palindrome. Equal hashes are only a
        candidate and are confirmed by an exact comparison.
        """
        if pack_key(r % M1, r & MASK) != key:
            return False
        if w == w[::-1]:
            return True
        self.pal_false_candidates += 1
        return False

    def _find(self, bucket: list[Node], key: int, w: bytes) -> int:
        for j in range(len(bucket)):
            node = bucket[j]
            if node[0] == key:
                if node[1] == w:
                    return j
                self.key_collisions += 1
        return -1

    # ---- public API (one operation per call; readable, used by tests / trace) ----

    def contains(self, w: bytes) -> bool:
        key, idx = self._key_idx(poly_value(w, self.p, self.off))
        bucket = self.table[idx]
        return bucket is not None and self._find(bucket, key, w) >= 0

    def add(self, w: bytes) -> bool:
        if self.track:
            h, r = poly_fwd_rev(w, self.p, self.pw, self.off)
        else:
            h, r = poly_value(w, self.p, self.off), 0
        key, idx = self._key_idx(h)
        bucket = self.table[idx]
        if bucket is None:
            bucket = self.table[idx] = []
        elif self._find(bucket, key, w) >= 0:
            return False
        is_pal = self.track and self._confirm_palindrome(w, key, r)
        bucket.append((key, w, is_pal))
        self.size += 1
        self.pal_count += is_pal
        if self.size > self.limit:
            self._resize()
        return True

    def remove(self, w: bytes) -> bool:
        key, idx = self._key_idx(poly_value(w, self.p, self.off))
        bucket = self.table[idx]
        if bucket is None:
            return False
        j = self._find(bucket, key, w)
        if j < 0:
            return False
        self.pal_count -= bucket[j][2]
        bucket[j] = bucket[-1]      # O(1) delete: swap with last, pop
        bucket.pop()
        self.size -= 1
        return True

    # ---- fast path: the same logic inlined into one loop ----

    def run_ops(self, data: list[bytes], end: int, emit: Callable[[bytes], None]) -> None:
        """Process tokens data[0:end] = [op, word, op, word, ...].

        Same logic as add/remove/contains, inlined into one loop with local
        variables: no per-operation method calls, which dominate in CPython.
        Tokens must already be validated (op in +,-,? ; word in [a-z]{1,15}).
        """
        p = self.p
        pw = self.pw
        off = self.off
        m1 = M1
        mask = MASK
        track = self.track
        table = self.table
        cmask = self.mask_cap
        limit = self.limit
        size = self.size
        pal_count = self.pal_count
        coll = self.key_collisions
        false_cand = self.pal_false_candidates
        yes = b"yes"
        no = b"no"
        r = 0

        for i in range(0, end, 2):
            op = data[i]
            w = data[i + 1]
            if track and op == b"+":
                # forward and reverse hash in one pass, no reversed copy
                h = 0
                r = 0
                for c, q in zip(w, pw):
                    h = h * p + c
                    r += c * q
                n = len(w)
                h -= off[n]
                r -= off[n]
            else:
                h = 0
                for c in w:
                    h = h * p + c
                h -= off[len(w)]
            h1 = h % m1
            h2 = h & mask
            key = (h1 << 64) | h2
            idx = (h1 ^ (h2 >> 32)) & cmask
            bucket = table[idx]

            if op == b"?":
                if bucket is not None:
                    for node in bucket:
                        if node[0] == key:
                            if node[1] == w:
                                emit(yes)
                                break
                            coll += 1
                    else:
                        emit(no)
                else:
                    emit(no)

            elif op == b"+":
                if bucket is None:
                    bucket = table[idx] = []
                else:
                    dup = False
                    for node in bucket:
                        if node[0] == key:
                            if node[1] == w:
                                dup = True
                                break
                            coll += 1
                    if dup:
                        continue
                is_pal = False
                if track and r % m1 == h1 and r & mask == h2:
                    if w == w[::-1]:
                        is_pal = True
                        pal_count += 1
                    else:
                        false_cand += 1
                bucket.append((key, w, is_pal))
                size += 1
                if size > limit:
                    self.size = size
                    self._resize()
                    table = self.table
                    cmask = self.mask_cap
                    limit = self.limit

            elif op == b"-":
                if bucket is not None:
                    for j in range(len(bucket)):
                        node = bucket[j]
                        if node[0] == key:
                            if node[1] == w:
                                pal_count -= node[2]
                                bucket[j] = bucket[-1]
                                bucket.pop()
                                size -= 1
                                break
                            coll += 1

        self.size = size
        self.pal_count = pal_count
        self.key_collisions = coll
        self.pal_false_candidates = false_cand

    def _resize(self) -> None:
        """Double the capacity; nodes are moved using cached hashes (no rehashing of words)."""
        new_cap = self.cap * 2
        new_mask = new_cap - 1
        new_table: list[list[Node] | None] = [None] * new_cap
        for bucket in self.table:
            if not bucket:
                continue
            for node in bucket:
                key = node[0]
                idx = ((key >> 64) ^ ((key & MASK) >> 32)) & new_mask
                nb = new_table[idx]
                if nb is None:
                    new_table[idx] = [node]
                else:
                    nb.append(node)
        self.table = new_table
        self.cap = new_cap
        self.mask_cap = new_mask
        self.limit = int(new_cap * _LOAD_FACTOR)
        self.resizes += 1

    # ---- results / inspection ----

    def palindromes(self) -> list[bytes]:
        """All palindromes of the current set."""
        if self.track:
            return [node[1] for bucket in self.table if bucket for node in bucket if node[2]]
        pw, off = self.pw, self.off
        return [w for bucket in self.table if bucket for key, w, _ in bucket
                if self._confirm_palindrome(w, key, poly_reverse(w, pw, off))]

    def words(self) -> Iterator[bytes]:
        for bucket in self.table:
            if bucket:
                for node in bucket:
                    yield node[1]

    def dump(self, limit: int = 50) -> list[str]:
        """Human-readable view of the non-empty buckets (at most `limit`)."""
        lines = [f"hash table: p={self.p}, capacity={self.cap}, size={self.size}, "
                 f"non-empty buckets={sum(1 for b in self.table if b)}"]
        shown = 0
        for idx, bucket in enumerate(self.table):
            if not bucket:
                continue
            if shown == limit:
                lines.append(f"... (showing first {limit} non-empty buckets)")
                break
            lines.append(f"bucket {idx}:")
            for key, w, is_pal in bucket:
                pal = "  [palindrome]" if is_pal else ""
                lines.append(f"    {w.decode():<15}  h1={key >> 64:<10}  h2={key & MASK:<20}{pal}")
            shown += 1
        return lines

    def approx_memory(self) -> int:
        """Bytes held by the table, buckets, nodes, keys and words (sys.getsizeof)."""
        total = sys.getsizeof(self.table)
        for bucket in self.table:
            if bucket is None:
                continue
            total += sys.getsizeof(bucket)
            for node in bucket:
                total += sys.getsizeof(node) + sys.getsizeof(node[0]) + sys.getsizeof(node[1])
        return total

    def stats(self) -> dict:
        chain_lens = [len(b) for b in self.table if b]
        n_nonempty = len(chain_lens)
        return {
            "size": self.size,
            "cap": self.cap,
            "load_factor": round(self.size / self.cap, 4),
            "resizes": self.resizes,
            "nonempty_buckets": n_nonempty,
            "avg_chain_len": round(sum(chain_lens) / n_nonempty, 4) if n_nonempty else 0.0,
            "max_chain_len": max(chain_lens, default=0),
            "pal_count": self.pal_count if self.track else len(self.palindromes()),
            "key_collisions": self.key_collisions,
            "pal_false_candidates": self.pal_false_candidates,
        }

    def __len__(self) -> int:
        return self.size

    def __contains__(self, w: bytes) -> bool:
        return self.contains(w)
