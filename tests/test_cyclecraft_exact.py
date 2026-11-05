from fractions import Fraction as F

from flint import arb

from possq.cyclecraft import _arb_pipeline
from possq.cyclecraft_exact import base_kernel, compose, event_bracket, law, periodic_law


def test_small_window_kernel_conserves_and_matches_pgf():
    p, delta = periodic_law("cycle")
    ker, lost = base_kernel("cycle", 0, 2, -10, 14)
    pmf, L = law(ker, lost, p, delta)
    total_lo = sum((a for a, b in pmf.values()), arb(0))
    total_hi = sum((b for a, b in pmf.values()), arb(0)) + L
    assert total_lo <= 1 <= total_hi
    mono, periodic, mgf = _arb_pipeline()
    g = mgf(mono("cycle", F(1, 2)), 2, p, delta)
    ours_lo = sum((a * arb(2) ** (-k) for k, (a, b) in pmf.items()), arb(0))
    # window covers levels >= -10 only; the excluded far-left mass (< 1e-22) carries weights 2^{11+},
    # so compare the in-window part from below and require agreement to 1e-15.
    assert ours_lo.lower() <= g.upper()
    assert abs(float((g - ours_lo).mid())) < 1e-15


def test_doubling_brackets_direct():
    p, delta = periodic_law("star")
    k1, l1 = base_kernel("star", 0, 1, -8, 10)
    k2, l2 = compose(k1, l1, k1, l1, -12, 16)
    kd, ld = base_kernel("star", 0, 2, -12, 16)
    a = event_bracket(*law(k2, l2, p, delta), lambda k: k <= 0)
    b = event_bracket(*law(kd, ld, p, delta), lambda k: k <= 0)
    assert a[0] <= b[1] and b[0] <= a[1]
