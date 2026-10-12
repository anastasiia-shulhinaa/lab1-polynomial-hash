"""Classic Bloom filter: no false negatives, false positive rate, fast path."""

import math
import random
import string

import pytest

from Lab2.bloom.filters import BloomFilter
from Lab2.bloom.params import fp_theory

_LETTERS = string.ascii_letters.encode()


def _distinct(rng: random.Random, count: int, banned: set[bytes] = frozenset()) -> list[bytes]:
    seen: set[bytes] = set()
    while len(seen) < count:
        w = bytes(rng.choices(_LETTERS, k=rng.randint(1, 15)))
        if w not in banned:
            seen.add(w)
    return sorted(seen)


def test_default_parameters_are_the_lab_ones():
    f = BloomFilter()
    assert (f.n, f.p, f.l, f.m) == (10**6, 0.01, 7, 9_592_955)
    assert len(f.T) == 1_199_120
    assert f.set_cells() == 0


def test_empty_filter_answers_no():
    f = BloomFilter(n=1000, seed=1)
    for w in _distinct(random.Random(1), 500):
        assert not f.contains(w)


@pytest.mark.parametrize("seed", range(5))
def test_no_false_negatives(seed):
    words = _distinct(random.Random(seed), 5000) + [b"a", b"Z", b"a" * 15, b"Z" * 15]
    f = BloomFilter(n=5000, seed=seed)
    for w in words:
        f.add(w)
    assert all(f.contains(w) for w in words)
    assert all(w in f for w in words)


def test_adding_twice_changes_nothing():
    f = BloomFilter(n=100, seed=2)
    f.add(b"apple")
    snapshot = bytes(f.T)
    f.add(b"apple")
    assert bytes(f.T) == snapshot
    assert f.set_cells() <= f.l


def test_case_matters():
    f = BloomFilter(n=1000, seed=3)
    f.add(b"Apple")
    assert f.contains(b"Apple")
    assert not f.contains(b"apple") and not f.contains(b"APPLE")


def test_explicit_l_and_m_are_used():
    f = BloomFilter(n=100, p=0.01, seed=1, l=3, m=77)
    assert (f.l, f.m, len(f.family), len(f.T)) == (3, 77, 3, 10)
    f.add(b"x")
    assert all(0 <= h < 77 for h in f.positions(b"x")) and f.contains(b"x")


def test_false_positive_rate_at_full_load():
    """n distinct strings in a filter built for n: about 1% of absent strings answer Y."""
    n = 100_000
    rng = random.Random(1)
    present = _distinct(rng, n)
    absent = _distinct(rng, 200_000, banned=set(present))
    f = BloomFilter(n=n, p=0.01, seed=1)
    for w in present:
        f.add(w)
    rate = sum(f.contains(w) for w in absent) / len(absent)
    assert 0.008 < rate < 0.012

    stats = f.stats()
    assert stats["fill"] == pytest.approx(1 - math.exp(-f.l * n / f.m), abs=0.005)
    assert stats["fp_now"] == pytest.approx(fp_theory(f.m, f.l, n), rel=0.05)


def test_run_ops_equals_the_per_operation_methods():
    rng = random.Random(7)
    pool = _distinct(rng, 800)
    ops = []
    for _ in range(6000):
        ops += [rng.choice((b"+", b"?", b"?")), rng.choice(pool)]

    fast = BloomFilter(n=200, seed=11)          # overloaded on purpose: false Y do occur
    out: list[bytes] = []
    fast.run_ops(ops, len(ops), out.append)

    slow = BloomFilter(n=200, seed=11)
    added: set[bytes] = set()
    expected = []
    false_yes = 0
    for i in range(0, len(ops), 2):
        op, w = ops[i], ops[i + 1]
        if op == b"+":
            slow.add(w)
            added.add(w)
        else:
            answer = slow.contains(w)
            expected.append(b"Y" if answer else b"N")
            assert answer or w not in added     # N is always exact
            false_yes += answer and w not in added

    assert out == expected
    assert fast.T == slow.T
    assert false_yes > 0


def test_query_probes_counts_hash_evaluations():
    f = BloomFilter(n=1000, seed=5)
    words = _distinct(random.Random(5), 200)
    assert f.query_probes(words) == len(words)          # empty: first bit is zero
    for w in words:
        f.add(w)
    assert f.query_probes(words) == len(words) * f.l    # present: all l bits checked


def test_dump_shows_the_bits():
    f = BloomFilter(n=10, p=0.1, seed=1)
    f.add(b"apple")
    lines = f.dump()
    assert lines[0].startswith("T (bloom): m=49, l=4")
    assert "".join(lines[1].split()[1:]).count("1") == f.set_cells()
