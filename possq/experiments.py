"""Experiment ledger E1–E8.  Run:  python -m possq.experiments OUTDIR"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from .bounds import (MomentData, leakage_bound_square, mass_certificate, primal_witness,
                     separation_floor)
from .exact import Basis, complement, from_roots, peval
from .recover import (TrueMeasure, noisy_data, presence_radius, prony, rationalize,
                      hankel_spectrum, cluster_enclosure)

ATOMS = [(F(-3, 5), F(1, 4)), (F(-1, 10), F(3, 10)), (F(1, 3), F(1, 5)), (F(4, 5), F(1, 5))]
K = 16
R = F(1, 20)


def nbhd(nodes, r):
    return [(x - r, x + r) for x in nodes]


def fl(x):
    return None if x is None else float(x)


def E1_exact():
    B = Basis("chebyshev", K)
    mu = TrueMeasure(ATOMS)
    d = MomentData(B, mu.moments(B), [F(0)] * (K + 1))
    nodes = [x for x, _ in ATOMS]
    L = leakage_bound_square(d, from_roots(nodes), nbhd(nodes, R))
    pert = [nodes[0] + F(1, 1000)] + nodes[1:]
    L2 = leakage_bound_square(d, from_roots(pert), nbhd(pert, R))
    exact_val = ATOMS[0][1] * peval(from_roots(pert), ATOMS[0][0]) ** 2
    return dict(eta_true_nodes=str(L.eta), bound_true_nodes=str(L.bound),
                eta_perturbed=str(L2.eta), eta_perturbed_expected=str(exact_val),
                eta_perturbed_matches=(L2.eta == exact_val),
                passed=(L.eta == 0 and L.bound == 0 and L2.eta == exact_val))


def E2_separation():
    nodes = [x for x, _ in ATOMS]
    p = from_roots(nodes)
    rows = []
    ok = True
    from .exact import certified_lower_bound
    for r in [F(1, 100), F(1, 50), F(1, 20), F(1, 10), F(1, 5)]:
        out = complement((F(-1), F(1)), nbhd(nodes, r))
        g = certified_lower_bound(p * p, out)
        floor = separation_floor(nodes, r)
        rows.append(dict(r=str(r), g=float(g), floor=float(floor), ratio=float(g / floor)))
        ok &= g >= floor * F(999999, 1000000)
    return dict(rows=rows, passed=bool(ok))


def E3_scaling():
    B = Basis("chebyshev", K)
    mu = TrueMeasure(ATOMS)
    rows = []
    ok = True
    for e in [3, 4, 5, 6, 7, 8]:
        eps = F(1, 10 ** e)
        d = noisy_data(mu, B, eps)
        nodes, w = prony(d, 4)
        xh = rationalize(nodes)
        U = nbhd(xh, R)
        out = complement(d.I, U)
        L = leakage_bound_square(d, from_roots(xh), U)
        up = mass_certificate(d, out, "upper")
        wit = primal_witness(d, out, "upper", extra_points=xh + [a for iv in out for a in iv])
        truth = mu.mass(out, False)
        node_err = max(abs(a - float(x)) for a, (x, _) in zip(nodes, ATOMS))
        row = dict(eps=float(eps), node_err=node_err, eta=float(L.eta), g=float(L.g),
                   square_bound=fl(L.bound), sharp_upper=float(up.value),
                   witness=fl(wit.value) if wit else None, truth=float(truth),
                   heavy_atom_threshold=fl(L.bound))
        if wit:
            row["relative_gap"] = float((up.value - wit.value) / up.value)
        ok &= (L.bound is None or truth <= L.bound) and truth <= up.value
        ok &= (wit is None) or (wit.value <= up.value)
        rows.append(row)
    def slope(key):
        xs = [math.log10(r["eps"]) for r in rows if r[key]]
        ys = [math.log10(r[key]) for r in rows if r[key]]
        return float(np.polyfit(xs, ys, 1)[0])
    return dict(rows=rows, slope_square=slope("square_bound"), slope_sharp=slope("sharp_upper"),
                slope_node_err=slope("node_err"), passed=bool(ok))


def E4_hidden_atom():
    """An unmodelled fifth atom must be caught: certified lower bound of the
    outside mass is positive although the 4-node model was fitted."""
    B = Basis("chebyshev", K)
    rows = []
    ok = True
    for wh in [F(1, 20), F(1, 100), F(1, 1000)]:
        mu = TrueMeasure(ATOMS + [(F(3, 5), wh)])
        d = noisy_data(mu, B, F(1, 10 ** 7))
        nodes, w = prony(d, 4)
        xh = rationalize(nodes)
        U = nbhd(xh, R)
        out = complement(d.I, U)
        L = leakage_bound_square(d, from_roots(xh), U)
        lo = mass_certificate(d, out, "lower")
        up = mass_certificate(d, out, "upper")
        truth = mu.mass(out, False)
        sv = hankel_spectrum(d)
        rows.append(dict(hidden_weight=float(wh), prony_nodes=nodes, outside_truth=float(truth),
                         square_bound=fl(L.bound), certified_lower=float(lo.value),
                         certified_upper=float(up.value), detected=bool(lo.value > 0),
                         hankel_sv=[float(s) for s in sv]))
        ok &= lo.value <= truth <= up.value
    return dict(rows=rows, passed=bool(ok))


def E5_clusters():
    B = Basis("chebyshev", K)
    out = {}
    ok = True
    for name, bg in [("atomic", F(0)), ("background_1/20", F(1, 20))]:
        mu = TrueMeasure(ATOMS, bg)
        d = noisy_data(mu, B, F(1, 10 ** 6))
        rows = []
        for x, w in ATOMS:
            lo, up = cluster_enclosure(d, x, R)
            t = mu.mass([(x - R, x + R)], True)
            rows.append(dict(center=float(x), lower=float(lo.value), upper=float(up.value),
                             truth=float(t), width=float(up.value - lo.value)))
            ok &= lo.value <= t <= up.value
        out[name] = rows
    out["passed"] = bool(ok)
    return out


def E6_presence():
    B = Basis("chebyshev", K)
    mu = TrueMeasure(ATOMS)
    rows = []
    ok = True
    for e in [3, 4, 5, 6, 7, 8]:
        d = noisy_data(mu, B, F(1, 10 ** e))
        nodes, _ = prony(d, 4)
        xh = rationalize(nodes)
        radii = []
        for c, (x, _) in zip(xh, ATOMS):
            res = presence_radius(d, c)
            if res is None:
                radii.append(None)
                continue
            r, cert = res
            radii.append(float(r))
        rows.append(dict(eps=float(F(1, 10 ** e)), radii=radii))
    slopes = []
    for i in range(4):
        pts = [(math.log10(r["eps"]), math.log10(r["radii"][i])) for r in rows if r["radii"][i]]
        if len(pts) >= 3:
            slopes.append(float(np.polyfit(*zip(*pts), 1)[0]))
    return dict(rows=rows, slopes=slopes, passed=bool(ok))


def E7_selection():
    B = Basis("chebyshev", K)
    mu = TrueMeasure(ATOMS)
    eps = F(1, 10 ** 6)
    rng = random.Random(7)
    m = mu.moments(B)
    u = [F(rng.randint(-10 ** 6, 10 ** 6), 10 ** 6) for _ in m]

    def make(eps_vec):
        return MomentData(B, [mk + ek * uk for mk, ek, uk in zip(m, eps_vec, u)], eps_vec)

    base = make([eps] * (K + 1))
    U = nbhd([x for x, _ in ATOMS], R)
    out = complement(base.I, U)
    c0 = mass_certificate(base, out, "upper")
    rank = sorted(range(K + 1), key=lambda k: -abs(c0.coeffs[k]))
    res = dict(base=float(c0.value), sensitivity=[float(abs(c)) for c in c0.coeffs], ranking=rank)
    for label, sel in [("top4", rank[:4]), ("bottom4", rank[-4:]),
                       ("random4", random.Random(1).sample(range(K + 1), 4))]:
        ev = [eps / 10 if k in sel else eps for k in range(K + 1)]
        c = mass_certificate(make(ev), out, "upper")
        res[label] = dict(selected=sel, bound=float(c.value),
                          reduction=float(1 - c.value / c0.value),
                          predicted_first_order=float(sum(abs(c0.coeffs[k]) * (eps - eps / 10) for k in sel) / c0.value))
    res["passed"] = res["top4"]["bound"] < res["random4"]["bound"] and res["top4"]["bound"] < res["bottom4"]["bound"]
    return res


def E8_soundness(n_inst=24, seed=2026):
    rng = random.Random(seed)
    B = Basis("chebyshev", 12)
    checks = []
    fails = 0
    for it in range(n_inst):
        n = rng.randint(1, 5)
        nodes = sorted(set(F(rng.randint(-95, 95), 100) for _ in range(n)))
        ws = [F(rng.randint(1, 100), 100) for _ in nodes]
        bg = F(rng.choice([0, 0, rng.randint(1, 20)]), 100)
        mu = TrueMeasure(list(zip(nodes, ws)), bg)
        eps = F(1, 10 ** rng.randint(3, 8))
        d = noisy_data(mu, B, eps, seed=it)
        a = F(rng.randint(-100, 60), 100)
        b = a + F(rng.randint(5, 40), 100)
        S = [(a, min(b, F(1)))]
        lo = mass_certificate(d, S, "lower")
        up = mass_certificate(d, S, "upper")
        t_closed = mu.mass(S, False)
        t_open = mu.mass(S, True)
        ok = (lo.value <= t_open) and (t_closed <= up.value)
        fails += (not ok)
        checks.append(dict(inst=it, n_atoms=len(nodes), background=float(bg), eps=float(eps),
                           S=[float(a), float(S[0][1])], lower=float(lo.value), upper=float(up.value),
                           truth_open=float(t_open), truth_closed=float(t_closed), ok=bool(ok)))
    return dict(n=n_inst, failures=fails, checks=checks, passed=fails == 0)


def E9_basis():
    """Same per-moment noise level in monomial vs Chebyshev measurement bases."""
    mu = TrueMeasure(ATOMS)
    out = {}
    U = nbhd([x for x, _ in ATOMS], R)
    for name in ("chebyshev", "monomial"):
        B = Basis(name, K)
        d = noisy_data(mu, B, F(1, 10 ** 6))
        S = complement(d.I, U)
        c = mass_certificate(d, S, "upper")
        L = leakage_bound_square(d, from_roots([x for x, _ in ATOMS]), U)
        out[name] = dict(sharp_upper=float(c.value), square_bound=fl(L.bound), eta=float(L.eta))
    out["passed"] = True
    return out


def main(outdir=".", only=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lp = outdir / "ledger.json"
    ledger = json.loads(lp.read_text()) if (only and lp.exists()) else {}
    ledger.pop("all_passed", None)
    for name, fn in [("E1_exact_theorem", E1_exact), ("E2_separation_lemma", E2_separation),
                     ("E3_noise_scaling", E3_scaling), ("E4_hidden_atom", E4_hidden_atom),
                     ("E5_cluster_enclosures", E5_clusters), ("E6_presence_radii", E6_presence),
                     ("E7_measurement_selection", E7_selection), ("E8_soundness_campaign", E8_soundness),
                     ("E9_basis_comparison", E9_basis)]:
        if only and not any(name.startswith(o) for o in only):
            continue
        t = time.time()
        ledger[name] = fn()
        ledger[name]["seconds"] = round(time.time() - t, 2)
        print(name, "passed=", ledger[name]["passed"], ledger[name]["seconds"], "s", flush=True)
        (outdir / "ledger.json").write_text(json.dumps(ledger, indent=1))
    ledger["all_passed"] = all(v["passed"] for v in ledger.values() if isinstance(v, dict))
    (outdir / "ledger.json").write_text(json.dumps(ledger, indent=1))
    print("ALL PASSED:", ledger["all_passed"])
    return ledger


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".", sys.argv[2:] or None)
