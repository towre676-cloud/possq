import random
from fractions import Fraction as F

import pytest
import sympy as sp

from possq import Basis, MomentData, TrueMeasure, noisy_data, mass_certificate
from possq.bounds import leakage_bound_square, separation_floor
from possq.exact import (cheb_T, cheb_to_mono, certify_positive, complement, count_distinct_roots,
                         from_roots, mono_to_cheb, poly, mono_coeffs, certified_lower_bound)

x = sp.symbols("x")


def test_chebyshev_roundtrip():
    for k in range(20):
        assert mono_coeffs(cheb_to_mono(mono_to_cheb(cheb_T(k)))) == mono_coeffs(cheb_T(k))
        T = sp.chebyshevt(k, x)
        assert [F(int(c.p), int(c.q)) for c in reversed(sp.Poly(T, x).all_coeffs())] == mono_coeffs(cheb_T(k))


def test_sturm_matches_sympy():
    rng = random.Random(3)
    for _ in range(40):
        roots = [F(rng.randint(-50, 50), 37) for _ in range(rng.randint(1, 6))]
        P = from_roots(roots) * poly([F(rng.randint(1, 9)), 0, 1])
        a, b = F(rng.randint(-60, 0), 37), F(rng.randint(1, 60), 37)
        expected = len({r for r in roots if a < r <= b})
        assert count_distinct_roots(P, a, b) == expected


def test_certify_positive_soundness():
    P = poly([F(1, 10 ** 9), 0, 1])            # x^2 + 1e-9 > 0
    assert certify_positive(P, -1, 1)
    Q = poly([-F(1, 10 ** 9), 0, 1])           # negative near 0
    assert not certify_positive(Q, -1, 1)
    assert certify_positive(Q, F(1, 10), 1)


def test_complement():
    I = (F(-1), F(1))
    c = complement(I, [(F(-1, 2), F(0)), (F(1, 4), F(2))])
    assert c == [(F(-1), F(-1, 2)), (F(0), F(1, 4))]


def test_exact_theorem_zero_leak():
    B = Basis("chebyshev", 10)
    atoms = [(F(-1, 3), F(1, 2)), (F(1, 2), F(1, 2))]
    mu = TrueMeasure(atoms)
    d = MomentData(B, mu.moments(B), [F(0)] * 11)
    L = leakage_bound_square(d, from_roots([a for a, _ in atoms]), [(a - F(1, 10), a + F(1, 10)) for a, _ in atoms])
    assert L.eta == 0 and L.bound == 0


def test_separation_lemma():
    nodes = [F(-3, 5), F(-1, 10), F(1, 3), F(4, 5)]
    p = from_roots(nodes)
    for r in [F(1, 50), F(1, 10)]:
        out = complement((F(-1), F(1)), [(z - r, z + r) for z in nodes])
        assert certified_lower_bound(p * p, out) >= separation_floor(nodes, r) * F(999999, 10 ** 6)


@pytest.mark.parametrize("seed", range(6))
def test_certificates_contain_truth(seed):
    rng = random.Random(seed)
    B = Basis("chebyshev", 10)
    nodes = sorted(set(F(rng.randint(-90, 90), 100) for _ in range(3)))
    mu = TrueMeasure([(z, F(rng.randint(1, 50), 100)) for z in nodes], F(rng.randint(0, 10), 100))
    d = noisy_data(mu, B, F(1, 10 ** rng.randint(3, 7)), seed=seed)
    a = F(rng.randint(-100, 50), 100)
    S = [(a, a + F(3, 10))]
    lo = mass_certificate(d, S, "lower")
    up = mass_certificate(d, S, "upper")
    assert lo.value <= mu.mass(S, True)
    assert mu.mass(S, False) <= up.value


def test_monomial_basis_certificate():
    B = Basis("monomial", 8)
    mu = TrueMeasure([(F(-1, 2), F(1, 2)), (F(1, 2), F(1, 2))])
    d = noisy_data(mu, B, F(1, 10 ** 6))
    S = [(F(-1, 5), F(1, 5))]
    up = mass_certificate(d, S, "upper")
    assert 0 <= up.value < F(1, 100)
