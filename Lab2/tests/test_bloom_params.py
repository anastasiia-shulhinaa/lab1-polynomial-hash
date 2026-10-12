"""Filter parameters: formulas (4.1), (4.2) and the numbers used in the lab."""

import math

import pytest

from Lab2.bloom.params import (
    bit_count, expected_probes, fp_theory, hash_count, optimal_bits, optimal_hashes,
)

N = 10**6
P = 0.01


def test_optimal_values_from_the_lecture_formulas():
    assert optimal_hashes(P) == pytest.approx(6.6439, abs=1e-4)
    assert optimal_bits(N, P) == pytest.approx(9_585_058.4, abs=0.1)
    # (4.2) written both ways: l = (m / n) ln 2 = -ln p / ln 2
    assert optimal_bits(N, P) / N * math.log(2) == pytest.approx(optimal_hashes(P))


def test_lab_parameters():
    l = hash_count(P)
    m = bit_count(N, P, l)
    assert l == 7
    assert m == 9_592_955
    assert (m + 7) // 8 == 1_199_120
    assert fp_theory(m, l, N) <= P
    assert fp_theory(m - 1, l, N) > P       # m is the smallest such size


def test_formula_4_1_with_rounded_l_misses_the_target():
    m = math.ceil(optimal_bits(N, P))
    assert m == 9_585_059
    assert fp_theory(m, 7, N) > P
    assert fp_theory(m, 7, N) == pytest.approx(0.010039, abs=1e-6)


def test_lecture_example():
    # m = 10^10 bits, n = 10^9 elements: l = 7 and p ~ 0.008 < 1%
    assert round(10**10 / 10**9 * math.log(2)) == 7
    assert fp_theory(10**10, 7, 10**9) == pytest.approx(0.0082, abs=1e-4)
    # "even with 5 hash functions we get below 1%"
    assert fp_theory(10**10, 5, 10**9) < 0.01


@pytest.mark.parametrize("n", [1, 10, 1000, 10**6])
@pytest.mark.parametrize("p", [0.5, 0.1, 0.01, 0.001])
def test_chosen_size_meets_the_target(n, p):
    l = hash_count(p)
    assert l >= 1
    assert fp_theory(bit_count(n, p, l), l, n) <= p


def test_size_grows_with_n_and_with_stricter_p():
    assert bit_count(2 * N, P, 7) > bit_count(N, P, 7)
    assert bit_count(N, 0.001, hash_count(0.001)) > bit_count(N, P, 7)


def test_expected_probes_of_a_negative_query():
    assert expected_probes(0.0, 7) == 1.0
    assert expected_probes(1.0, 7) == 7.0
    assert expected_probes(0.5, 3) == pytest.approx(1 + 0.5 + 0.25)
    fill = 1 - math.exp(-7 * N / 9_592_955)
    assert expected_probes(fill, 7) == pytest.approx(2.05, abs=0.01)


@pytest.mark.parametrize("n, p", [(0, 0.01), (10, 0.0), (10, 1.0), (10, -0.1)])
def test_invalid_parameters(n, p):
    with pytest.raises(ValueError):
        bit_count(n, p, 7)
