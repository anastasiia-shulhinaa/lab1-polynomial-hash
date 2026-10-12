"""Bloom filter with counters: deletion, set semantics, counter limits."""

import random
import string

from Lab2.bloom.filters import BloomFilter, CountingBloomFilter

_LOWER = string.ascii_lowercase.encode()


def _pool(rng: random.Random, count: int) -> list[bytes]:
    seen: set[bytes] = set()
    while len(seen) < count:
        seen.add(bytes(rng.choices(_LOWER, k=rng.randint(1, 15))))
    return sorted(seen)


def test_one_byte_per_counter():
    f = CountingBloomFilter(n=1000, seed=1)
    assert len(f.T) == f.m and f.set_cells() == 0


def test_add_then_remove_restores_the_empty_filter():
    f = CountingBloomFilter(n=1000, seed=1)
    words = _pool(random.Random(1), 300)
    for w in words:
        assert f.add(w)
    assert all(f.contains(w) for w in words)
    for w in words:
        assert f.remove(w)
    assert f.set_cells() == 0
    assert not any(f.contains(w) for w in words)


def test_duplicate_add_keeps_set_semantics():
    f = CountingBloomFilter(n=1000, seed=2)
    assert f.add(b"apple")
    assert not f.add(b"apple")          # already present: counters untouched
    assert f.remove(b"apple")
    assert not f.contains(b"apple")
    assert f.set_cells() == 0


def test_remove_of_an_absent_string_changes_nothing():
    f = CountingBloomFilter(n=1000, seed=3)
    f.add(b"apple")
    snapshot = bytes(f.T)
    assert not f.remove(b"pear")
    assert bytes(f.T) == snapshot


def test_removing_one_string_keeps_the_others():
    f = CountingBloomFilter(n=200, seed=4)      # small array: some cells are shared
    words = _pool(random.Random(4), 50)
    for w in words:
        f.add(w)
    f.remove(words[0])
    assert all(f.contains(w) for w in words[1:])


def test_same_answers_as_the_bit_filter_without_deletions():
    rng = random.Random(5)
    pool = _pool(rng, 600)
    bits = BloomFilter(n=300, seed=8)
    counters = CountingBloomFilter(n=300, seed=8)
    assert (bits.m, bits.family) == (counters.m, counters.family)
    for w in pool[:300]:
        bits.add(w)
        counters.add(w)
    assert [bits.contains(w) for w in pool] == [counters.contains(w) for w in pool]
    assert [bits.cell(h) for h in range(bits.m)] == [min(c, 1) for c in counters.T]


def test_differential_against_an_exact_set():
    """Far below capacity the filter behaves exactly like a set."""
    rng = random.Random(6)
    pool = _pool(rng, 500)
    f = CountingBloomFilter(n=20000, seed=6)
    exact: set[bytes] = set()
    for _ in range(20000):
        op = rng.choice("+-?")
        w = rng.choice(pool)
        if op == "+":
            f.add(w)
            exact.add(w)
        elif op == "-":
            f.remove(w)
            exact.discard(w)
        else:
            assert f.contains(w) == (w in exact)


def test_run_ops_equals_the_per_operation_methods():
    rng = random.Random(7)
    pool = _pool(rng, 400)
    ops = []
    for _ in range(8000):
        ops += [rng.choice((b"+", b"+", b"-", b"?", b"?")), rng.choice(pool)]

    fast = CountingBloomFilter(n=100, seed=12)      # overloaded on purpose
    out: list[bytes] = []
    fast.run_ops(ops, len(ops), out.append)

    slow = CountingBloomFilter(n=100, seed=12)
    expected = []
    for i in range(0, len(ops), 2):
        op, w = ops[i], ops[i + 1]
        if op == b"+":
            slow.add(w)
        elif op == b"-":
            slow.remove(w)
        else:
            expected.append(b"Y" if slow.contains(w) else b"N")

    assert out == expected
    assert fast.T == slow.T


def test_saturated_counter_stays_put():
    f = CountingBloomFilter(n=1000, seed=9)
    hs = f.positions(b"apple")
    for h in hs:
        f.T[h] = 254
    f.T[hs[0]] = 0
    assert f.add(b"apple")
    assert all(f.T[h] == 255 for h in hs[1:])
    f.T[hs[0]] = 255
    assert f.remove(b"apple")
    assert all(f.T[h] == 255 for h in hs)       # neither incremented nor decremented


def test_counters_never_go_below_zero():
    f = CountingBloomFilter(n=1, p=0.5, seed=1, l=3, m=1)   # every position is cell 0
    assert f.add(b"x") and f.T[0] == 3
    assert f.remove(b"x") and f.T[0] == 0
    f.T[0] = 1
    assert f.remove(b"y") and f.T[0] == 0       # three decrements of a counter equal to 1


def test_query_probes_and_dump():
    f = CountingBloomFilter(n=10, p=0.1, seed=1)
    assert f.query_probes([b"a", b"b"]) == 2
    f.add(b"a")
    assert f.query_probes([b"a"]) == f.l
    lines = f.dump()
    assert lines[0].startswith("T (counting): m=49, l=4")
