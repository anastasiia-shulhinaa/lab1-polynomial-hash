"""String key and the universal class H_qm."""

import itertools
import random
import string

from src.hashing import poly_value

from Lab2.bloom.hashing import KEY_LIMIT, Q, key_of, make_family, positions

_LETTERS = string.ascii_letters.encode()


def _words(rng: random.Random, count: int) -> list[bytes]:
    return [bytes(rng.choices(_LETTERS, k=rng.randint(1, 15))) for _ in range(count)]


def test_key_is_the_lab1_polynomial_with_base_256():
    no_offset = [0] * 16
    for w in _words(random.Random(1), 2000):
        assert key_of(w) == poly_value(w, 256, no_offset)


def test_key_is_injective():
    short = [bytes(t) for n in (1, 2) for t in itertools.product(_LETTERS, repeat=n)]
    words = set(short + _words(random.Random(2), 20000))
    assert len({key_of(w) for w in words}) == len(words)
    assert key_of(b"a") != key_of(b"aa") != key_of(b"aaa")


def test_every_key_is_below_the_prime():
    assert key_of(b"z" * 15) < KEY_LIMIT < Q


def test_q_is_prime():
    # Lucas-Lehmer test for the Mersenne number 2**127 - 1
    s = 4
    for _ in range(127 - 2):
        s = (s * s - 2) % Q
    assert s == 0


def test_family_is_reproducible_by_seed():
    assert make_family(7, seed=5) == make_family(7, seed=5)
    assert make_family(7, seed=5) != make_family(7, seed=6)
    assert make_family(7) != make_family(7)             # fresh per run without a seed


def test_family_members_are_distinct_and_in_range():
    family = make_family(7, seed=3)
    assert len(family) == 7 and len(set(family)) == 7
    assert all(1 <= a < Q and 1 <= b < Q for a, b in family)


def test_positions_are_inside_the_array():
    family = make_family(7, seed=4)
    for m in (1, 2, 49, 9_592_955):
        for w in _words(random.Random(m), 300):
            hs = positions(key_of(w), family, m)
            assert len(hs) == 7 and all(0 <= h < m for h in hs)


def test_pair_collision_probability_is_at_most_one_over_m():
    """Universality: over a random (a, b), two fixed keys collide with probability <= 1/m."""
    m = 101
    trials = 20000
    for seed, (x, y) in enumerate([(b"apple", b"apply"), (b"a", b"b"), (b"Zzzzzzzzzzzzzzz", b"z")]):
        kx, ky = key_of(x), key_of(y)
        family = make_family(trials, seed=seed)
        hits = sum(1 for a, b in family
                   if (a * kx + b) % Q % m == (a * ky + b) % Q % m)
        assert hits / trials < 1 / m + 0.0035       # 5 standard deviations above 1/m


def test_one_function_spreads_keys_evenly():
    m = 64
    count = 64000
    ((a, b),) = make_family(1, seed=9)
    buckets = [0] * m
    for i in range(count):
        buckets[(a * key_of(b"w" + str(i).encode()) + b) % Q % m] += 1
    assert min(buckets) > 0.85 * count / m
    assert max(buckets) < 1.15 * count / m
