"""Adapter: sharp certified wrong-direction tail bounds for Cyclecraft II.

Reads the certified 320-bit Arb enclosures in Cyclecraft's precision_cert_results.json
(authoritative exact dyadic endpoints, never the float fields) and feeds them to
possq.integer.  N_10 is the signed count on the target edge after 10 periods from the
certified periodic law; the enclosures already include Cyclecraft's periodic-initialisation
error delta.

The oracle (truth) is an independent binary64 computation of the full law of N_10 on a
count-lifted chain built from the same protocol constants (copied from Cyclecraft's
precision_cert.py, PROTOCOL below); it is a falsification check, not part of any certificate.
"""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

from .integer import IntegerData

# --- protocol constants, verbatim from Cyclecraft II precision_cert.py --------
N = 5
CORE = [(0, 1), (1, 2), (2, 0)]
GRAPHS = {"cycle": CORE + [(0, 3), (3, 4), (4, 0)], "star": CORE + [(0, 3), (0, 4)]}
TAU = F(2)
W = [[F(1, 4), F(4), F(4), F(1, 2), F(2)],
     [F(4), F(1, 4), F(4), F(1), F(1)],
     [F(4), F(4), F(1, 4), F(2), F(1, 2)]]
QC = [[F(1, 8), F(1, 8), F(2)], [F(2), F(1, 8), F(1, 8)], [F(1, 8), F(2), F(1, 8)]]
PERIODS = 10


def dyadic(d: Dict) -> F:
    m, e = int(d["mantissa"]), int(d["exponent2"])
    return F(m) * F(2) ** e


def ball(enc: Dict) -> Tuple[F, F]:
    return dyadic(enc["lower_dyadic"]), dyadic(enc["upper_dyadic"])


def load(path) -> Dict:
    return json.loads(Path(path).read_text())


def network_data(results: Dict, graph: str, use_poly: bool = True, use_pgf=True,
                 copies: int = 1) -> Tuple[IntegerData, Dict]:
    c = results["certificates"][graph]
    row = [r for r in c["finite_horizon_moments"]["target"] if r["periods"] == PERIODS][0]
    mlo, mhi = ball(row["mean"])
    vlo, vhi = ball(row["variance"])
    assert mlo > 0
    zs, pgf = [], []
    for z, enc in c["target_tilted_mgfs"].items():
        lo, hi = ball(enc)
        zs.append(F(z))
        pgf.append((lo ** copies, hi ** copies))     # independent copies: E z^S = (E z^N)^copies
    # sum of `copies` iid copies: mean and variance scale linearly
    m_lo, m_hi = copies * mlo, copies * mhi
    v_lo, v_hi = copies * vlo, copies * vhi
    poly = [(m_lo, m_hi), (v_lo + m_lo ** 2, v_hi + m_hi ** 2)]
    deg = 2 if use_poly else 0
    if not use_poly:
        poly = []
    if not use_pgf:
        zs, pgf = [], []
    meta = dict(mean=(m_lo, m_hi), variance=(v_lo, v_hi),
                chernoff_reported=ball(c["target_lower_tail"]["certified_probability_upper"]),
                aggregate_reported=ball(c["target_lower_tail"]["independent_copies_10_aggregate_wrong_direction_upper"]))
    return IntegerData(deg, poly, zs, pgf, label="%s x%d" % (graph, copies)), meta


# --- independent oracle -------------------------------------------------------
def _generator(graph: str, stage: int, B) -> np.ndarray:
    """Count-lifted generator on levels [-B, B]; B=None gives the plain (uncounted) generator."""
    if B is None:
        edges = GRAPHS[graph]
        auxq = F(1, 2) if graph == "cycle" else F(3, 4)
        q = QC[stage] + [auxq] * (len(edges) - 3)
        K = np.zeros((N, N))
        for e, (i, j) in enumerate(edges):
            for src, tgt in [(i, j), (j, i)]:
                rate = float(q[e] * W[stage][src])
                K[src, src] -= rate
                K[tgt, src] += rate
        return K
    edges = GRAPHS[graph]
    auxq = F(1, 2) if graph == "cycle" else F(3, 4)
    q = QC[stage] + [auxq] * (len(edges) - 3)
    size = N * (2 * B + 1)
    K = np.zeros((size, size))
    for k in range(-B, B + 1):
        for e, (i, j) in enumerate(edges):
            for src, tgt, sgn in [(i, j, 1), (j, i, -1)]:
                rate = float(q[e] * W[stage][src])
                col = (k + B) * N + src
                K[col, col] -= rate
                kk = k + (sgn if e == 0 else 0)
                if -B <= kk <= B:
                    K[(kk + B) * N + tgt, col] += rate
    return K


def oracle_law(graph: str, B: int = 70) -> Tuple[np.ndarray, float]:
    """Law of N_10 on [-B, B] from the periodic initial law; returns (pmf, escaped mass)."""
    M1 = np.eye(N)
    for s in range(3):
        M1 = expm(_generator(graph, s, None) * float(TAU)) @ M1
    w, v = np.linalg.eig(M1)
    p = np.real(v[:, np.argmin(abs(w - 1))]); p = p / p.sum()
    Ms = [expm(_generator(graph, s, B) * float(TAU)) for s in range(3)]
    x = np.zeros(N * (2 * B + 1)); x[B * N:(B + 1) * N] = p
    for _ in range(PERIODS):
        for s in range(3):
            x = Ms[s] @ x
    pmf = x.reshape(2 * B + 1, N).sum(axis=1)
    return pmf, 1 - pmf.sum()


def oracle_aggregate(pmf: np.ndarray, copies: int) -> np.ndarray:
    out = pmf.copy()
    for _ in range(copies - 1):
        out = np.convolve(out, pmf)
    return out


# --- additional certified tilts: Cyclecraft's own Arb pipeline, reproduced ----------------
# periodic_certificate and mgf follow precision_cert.py line for line (Doeblin residual delta,
# exact dyadic normalised proposal, row-sum error term); only the tilt grid is extended.
def _arb_pipeline():
    from flint import arb, arb_mat, fmpq, ctx
    ctx.prec = 320

    def A(x):
        return arb(fmpq(x.numerator, x.denominator)) if isinstance(x, F) else arb(x)

    def gen(graph, stage, z=F(1), edge=0):
        edges = GRAPHS[graph]
        auxq = F(1, 2) if graph == "cycle" else F(3, 4)
        q = QC[stage] + [auxq] * (len(edges) - 3)
        K = [[F(0)] * N for _ in range(N)]
        for e, (i, j) in enumerate(edges):
            for s, tg, sg in [(i, j, 1), (j, i, -1)]:
                rate = q[e] * W[stage][s]
                K[s][s] -= rate
                K[tg][s] += rate * (z ** (sg if e == edge else 0))
        return K

    def mono(graph, z=F(1)):
        M = arb_mat([[int(i == j) for j in range(N)] for i in range(N)])
        for s in range(3):
            M = (arb_mat([[A(v) for v in r] for r in gen(graph, s, z)]) * A(TAU)).exp() * M
        return M

    def minimum(xs):
        return min(xs, key=lambda x: x.lower())

    def maximum(xs):
        return max(xs, key=lambda x: x.upper())

    def periodic(M):
        Id = arb_mat([[int(i == j) for j in range(N)] for i in range(N)])
        B = M - Id
        for j in range(N):
            B[N - 1, j] = 1
        prop = B.solve(arb_mat(N, 1, [0] * (N - 1) + [1]))
        bits = ctx.prec - 32
        pp = []
        for i in range(N - 1):
            m, e = prop[i, 0].mid().man_exp()
            f = F(int(m)) * F(2) ** int(e)
            pp.append(F(round(f * (2 ** bits)), 2 ** bits))
        pp.append(F(1) - sum(pp, F(0)))
        p = [A(v) for v in pp]
        pv = arb_mat(N, 1, p)
        res = (M - Id) * pv
        rnorm = sum((abs(res[i, 0]) for i in range(N)), arb(0)).upper()
        mm = minimum([M[i, j].lower() for i in range(N) for j in range(N)])
        assert mm > 0
        delta = (rnorm / (N * mm)).upper()
        return pv, delta

    def mgf(Mz, n, p, delta):
        Mn = Mz ** n
        row = [sum((Mn[i, j] for i in range(N)), arb(0)) for j in range(N)]
        center = sum((row[j] * p[j, 0] for j in range(N)), arb(0))
        err = (maximum([abs(v).upper() for v in row]) * delta).upper()
        return center + arb(0, err)

    return mono, periodic, mgf


def certified_pgf(graph: str, zs, periods: int = PERIODS) -> Dict[F, Tuple[F, F]]:
    """Rigorous enclosures of E[z^{N_periods}] for rational z, as exact dyadic endpoints."""
    mono, periodic, mgf = _arb_pipeline()
    p, delta = periodic(mono(graph))
    out = {}
    for z in zs:
        g = mgf(mono(graph, F(z)), periods, p, delta)
        lo, hi = g.lower(), g.upper()
        ml, el = lo.man_exp()
        mh, eh = hi.man_exp()
        out[F(z)] = (F(int(ml)) * F(2) ** int(el), F(int(mh)) * F(2) ** int(eh))
    return out


def extended_data(results: Dict, graph: str, zs, copies: int = 1) -> IntegerData:
    base, _ = network_data(results, graph, copies=copies)
    enc = certified_pgf(graph, zs)
    allz = sorted(set(base.zs) | set(F(z) for z in zs))
    pgf = []
    for z in allz:
        if z in base.zs:
            pgf.append(base.pgf[base.zs.index(z)])
        else:
            lo, hi = enc[z]
            pgf.append((lo ** copies, hi ** copies))
    return IntegerData(2, base.poly, allz, pgf, label=base.label + " +tilts")
