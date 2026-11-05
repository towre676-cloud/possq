import random
from fractions import Fraction as F

import pytest

from possq.integer import IntegerData, sharp_tail, chernoff_best, left_tail_ok, q_exact


def law_data(law, zs, rad=F(0)):
    m1 = sum(w * n for n, w in law)
    m2 = sum(w * n * n for n, w in law)
    pg = [sum(w * z ** n for n, w in law) for z in zs]
    I = lambda v: (v - rad, v + rad)
    return IntegerData(2, [I(m1), I(m2)], list(zs), [I(v) for v in pg])


@pytest.mark.parametrize("seed", range(8))
def test_sharp_bracket_contains_truth(seed):
    rng = random.Random(seed)
    support = sorted(rng.sample(range(-6, 25), rng.randint(3, 9)))
    ws = [F(rng.randint(1, 30)) for _ in support]
    tot = sum(ws)
    law = [(n, w / tot) for n, w in zip(support, ws)]
    zs = [F(1, 2), F(3, 4), F(7, 8)]
    d = law_data(law, zs, F(1, 10 ** 12))
    cert, wit, sol = sharp_tail(d, 0, 20, 60)
    truth = sum(w for n, w in law if n <= 0)
    assert wit.value <= cert.value
    assert truth <= cert.value
    assert cert.value <= chernoff_best(d)[0]
    # certificate dominates the indicator on a wide integer range (beyond the window, too)
    for n in range(-80, 200):
        assert q_exact(cert.coeffs, d, n) >= (1 if n <= 0 else 0)


def test_exact_data_bracket():
    law = [(-1, F(1, 20)), (2, F(3, 10)), (5, F(13, 20))]
    d = law_data(law, [F(1, 2), F(2, 3)])
    cert, wit, _ = sharp_tail(d, 0, 20, 40)
    assert wit.value <= cert.value
    if not cert.left["atoms_at_infinity"]:
        assert cert.value == wit.value


def test_cyclecraft_single_count_is_exact():
    from pathlib import Path
    from possq.cyclecraft import load, network_data
    R = load(Path(__file__).parents[1] / "results" / "cyclecraft_precision_cert_results.json")
    d, meta = network_data(R, "cycle")
    cert, wit, _ = sharp_tail(d, 0, 25, 100)
    assert cert.value - wit.value < F(1, 10 ** 80)
    assert abs(float(cert.value) - 0.027645131743682536) < 1e-15
    assert cert.value < meta["chernoff_reported"][0]


def test_left_tail_lemma_rejects_bad_dominance():
    d = IntegerData(2, [(F(1), F(1)), (F(2), F(2))], [F(1, 2), F(3, 4)], [(F(1, 2), F(1, 2)), (F(3, 4), F(3, 4))])
    c = [F(1), F(0), F(0), F(-1), F(1)]          # negative weight on the dominant rate 2
    ok, info = left_tail_ok(c, d, 30)
    assert not ok
