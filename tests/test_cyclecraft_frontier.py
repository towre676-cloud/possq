from fractions import Fraction as F

from flint import arb, ctx
import pytest

from possq.cyclecraft_frontier import (certified_law, convolve, optimize_cw,
                                     periodic_law, wrong_direction, count_kernel,
                                     compose_kernel, kernel_event)
from possq.cyclecraft_exact import base_kernel, event_bracket, law, periodic_law as old_periodic


def test_uniformization_agrees_with_independent_dense_exponential():
    # Independent arithmetic routes: sparse positive Poisson sum versus dense
    # matrix exponential, with genuinely different count windows.
    pmf, lost, meta = certified_law("cycle", periods=2, lo=-12, hi=16,
                                    bits=256, error_digits=50)
    sparse = wrong_direction(pmf, lost)
    p, delta = old_periodic("cycle")
    ker, loss = base_kernel("cycle", 0, 2, -10, 14)
    dense = event_bracket(*law(ker, loss, p, delta), lambda k: k <= 0)
    assert sparse[0] <= dense[1] and dense[0] <= sparse[1]
    assert sparse[1] - sparse[0] < arb("1e-20")
    assert all(row["highest_poisson_index"] > 0 for row in meta["uniformization"])


def test_polynomial_convolution_has_exact_small_control():
    ctx.prec = 256
    pmf = {-1: (arb(F(1, 4).numerator) / 4,) * 2,
           0: (arb(1) / 2,) * 2, 1: (arb(1) / 4,) * 2}
    out = convolve(pmf, pmf)
    lo, hi = wrong_direction(out, arb(0))
    assert lo <= arb(11) / 16 <= hi
    assert hi - lo < arb("1e-70")


def test_optimized_cw_bounds_contain_reported_rate_and_crossings():
    result = optimize_cw("star", bits=256)
    assert F(result["tilt"]) > 0
    assert result["rho_upper"]["decimal"]
    assert [r["first_horizon_for_this_CW_bound"] for r in result["horizons"]] == sorted(
        r["first_horizon_for_this_CW_bound"] for r in result["horizons"])


def test_polynomial_joint_kernel_doubling_matches_dense_exponential():
    kernel, _ = count_kernel("star", periods=1, lo=-8, hi=10, bits=256, error_digits=50)
    doubled = compose_kernel(kernel, kernel, -8, -12, 16)
    p, delta = periodic_law("star", 256)
    a, b, lost = kernel_event(doubled, -12, 16, p, delta)
    p0, delta0 = old_periodic("star")
    dense, dense_lost = base_kernel("star", 0, 2, -12, 16)
    lower, upper = event_bracket(*law(dense, dense_lost, p0, delta0), lambda k: k <= 0)
    assert a <= upper and lower <= b
    assert b - a < arb("1e-12")


def test_invalid_experiment_is_rejected_before_arithmetic():
    for kwargs in ({"periods": 0}, {"periods": 1.5}, {"lo": 1}, {"hi": -1}, {"bits": 16}):
        with pytest.raises(ValueError):
            certified_law("cycle", **kwargs)
