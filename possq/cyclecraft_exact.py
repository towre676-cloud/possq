"""Certified exact counting laws for Cyclecraft II's dyadic three-hold protocols.

Instead of summarising the signed count N_n by a few moments, compute its whole law with rigorous
error bars.

Count lift.  On levels L = {lo..hi} x {5 states}, the generator moves the state as in Cyclecraft and
moves the level by +1/-1 on a forward/backward jump through the counted edge.  Jumps that would leave
the window are KILLED (mass removed).  The killed propagator is entrywise <= the true joint kernel, and
the mass it loses is exactly 1 - (its column sums).  Hence for every event E in the window

    killed(E)  <=  P(E)  <=  killed(E) + lost.                                   (Lemma K)

Arb.  exp(K tau) for each held stage is an Arb matrix exponential of an exact rational generator, so
every probability below is a rigorous ball; the periodic initial law is Cyclecraft's own certificate
(exact dyadic proposal p with Doeblin l1 error delta), reproduced here.

Doubling.  The joint kernel A^{(n)}_k[i,j] = P(N_n = k, X_n = i | X_0 = j) composes as
A^{(2n)}_m = sum_{k} A^{(n)}_{m-k} A^{(n)}_k (Markov property at time n).  Entries are dropped outside a
growing window and their mass is added to `lost`, so Lemma K survives every step.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction as F
from typing import Dict, List, Tuple

from flint import arb, arb_mat, ctx, fmpq

from .cyclecraft import GRAPHS, N, QC, TAU, W, _arb_pipeline

PREC = 200


def A(x):
    return arb(fmpq(x.numerator, x.denominator)) if isinstance(x, F) else arb(x)


def lifted_generator(graph: str, stage: int, edge: int, lo: int, hi: int, shortcut: F = None) -> List[List[F]]:
    """edge = index of the counted edge (-1: none).  shortcut = conductance of Cyclecraft's rare extra
    3--4 channel on the star (index len(edges)), as in precision_cert.shortcut_monodromy."""
    edges = list(GRAPHS[graph])
    auxq = F(1, 2) if graph == "cycle" else F(3, 4)
    q = QC[stage] + [auxq] * (len(edges) - 3)
    if shortcut is not None:
        edges.append((3, 4))
        q = q + [shortcut]
    nlev = hi - lo + 1
    size = N * nlev
    K = [[F(0)] * size for _ in range(size)]
    for lev in range(lo, hi + 1):
        for e, (i, j) in enumerate(edges):
            for src, tgt, sgn in [(i, j, 1), (j, i, -1)]:
                rate = q[e] * W[stage][src]
                col = (lev - lo) * N + src
                K[col][col] -= rate                  # leaving rate always counted: kill = lose mass
                nl = lev + (sgn if e == edge else 0)
                if lo <= nl <= hi:
                    K[(nl - lo) * N + tgt][col] += rate
    return K


def one_period(graph: str, edge: int, lo: int, hi: int, shortcut: F = None) -> arb_mat:
    ctx.prec = PREC
    P = None
    for s in range(3):
        K = lifted_generator(graph, s, edge, lo, hi, shortcut)
        E = (arb_mat([[A(v) for v in row] for row in K]) * A(TAU)).exp()
        P = E if P is None else E * P
    return P


Kernel = Dict[int, arb_mat]          # level -> 5x5 matrix (end state i, start state j)


def base_kernel(graph: str, edge: int, periods: int, lo: int, hi: int, shortcut: F = None) -> Tuple[Kernel, List[arb]]:
    """Joint kernel of (N_periods, X_periods) from each start state at level 0; with lost mass per start."""
    ctx.prec = PREC
    P = one_period(graph, edge, lo, hi, shortcut)
    size = P.nrows()
    X = arb_mat(size, N)
    for j in range(N):
        X[(0 - lo) * N + j, j] = 1
    for _ in range(periods):
        X = P * X
    ker: Kernel = {}
    for lev in range(lo, hi + 1):
        M = arb_mat(N, N)
        for i in range(N):
            for j in range(N):
                M[i, j] = X[(lev - lo) * N + i, j]
        ker[lev] = M
    lost = [arb(1) - sum((X[r, j] for r in range(size)), arb(0)) for j in range(N)]
    return ker, lost


def _colsums(ker: Kernel) -> List[arb]:
    return [sum((ker[k][i, j] for k in ker for i in range(N)), arb(0)) for j in range(N)]


def compose(k1: Kernel, lost1: List[arb], k2: Kernel, lost2: List[arb], lo: int, hi: int) -> Tuple[Kernel, List[arb]]:
    """Kernel of (first k1 then k2), truncated to levels [lo, hi]; lost mass rigorously enclosed.

    True kernel T1 = k1 + R1 with R1 >= 0 and column sums of R1 equal lost1 (likewise T2).  The
    composed lost mass is 1 - colsum(truncated k2*k1), computed directly; it is >= the true
    probability of any path not represented, since every represented path has its exact weight."""
    ctx.prec = PREC
    out: Kernel = {}
    for a, Ma in k1.items():
        for b, Mb in k2.items():
            m = a + b
            if m < lo or m > hi:
                continue
            prod = Mb * Ma
            out[m] = prod if m not in out else out[m] + prod
    cs = _colsums(out)
    lost = [arb(1) - c for c in cs]
    return out, lost


def law(ker: Kernel, lost: List[arb], p: arb_mat, delta: arb) -> Tuple[Dict[int, Tuple[arb, arb]], arb]:
    """Rigorous pmf enclosures [lo, hi] of N under initial law p (+- delta in l1), and total lost mass."""
    ctx.prec = PREC
    pmf = {}
    for k, M in ker.items():
        v = sum((M[i, j] * p[j, 0] for i in range(N) for j in range(N)), arb(0))
        # perturbation of the initial law: |sum_ij M_ij e_j| <= max_j colsum(M)_j * delta <= delta
        lo = (v - delta).lower()
        pmf[k] = (lo if lo > 0 else arb(0), (v + delta).upper())
    tot_lost = sum((lost[j] * p[j, 0] for j in range(N)), arb(0)) + delta
    return pmf, tot_lost.upper()


def event_bracket(pmf, lost, pred) -> Tuple[arb, arb]:
    lo = sum((pmf[k][0] for k in pmf if pred(k)), arb(0))
    hi = sum((pmf[k][1] for k in pmf if pred(k)), arb(0)) + lost
    return lo.lower(), hi.upper()


def convolve_pmf(p1, l1, p2, l2):
    """pmf of the sum of independent counts, with enclosures; lost mass adds."""
    out: Dict[int, List[arb]] = {}
    for a, (la, ha) in p1.items():
        for b, (lb, hb) in p2.items():
            m = a + b
            lo, hi = la * lb, ha * hb
            if m in out:
                out[m][0] += lo
                out[m][1] += hi
            else:
                out[m] = [lo, hi]
    return {m: (v[0].lower(), v[1].upper()) for m, v in out.items()}, (l1 + l2).upper()


def periodic_law(graph: str, shortcut: F = None):
    """Cyclecraft's periodic certificate (Doeblin), applied to the plain monodromy of the protocol."""
    mono, periodic, mgf = _arb_pipeline()        # sets 320 bits; values remain valid at PREC
    if shortcut is None:
        M = mono(graph)
    else:
        ctx.prec = 320
        M = None
        for s in range(3):
            K = lifted_generator(graph, s, -1, 0, 0, shortcut)
            E = (arb_mat([[A(v) for v in row] for row in K]) * A(TAU)).exp()
            M = E if M is None else E * M
    p, delta = periodic(M)
    ctx.prec = PREC
    return p, arb(delta)


# ----------------------------------------------------------------------------
# all-horizon exponential bounds (Collatz-Wielandt)
# ----------------------------------------------------------------------------
def tilted_monodromy(graph: str, z: F, edge: int = 0) -> arb_mat:
    mono, _, _ = _arb_pipeline()
    # _arb_pipeline's mono tilts edge 0; for other edges rebuild here
    if edge == 0:
        return mono(graph, z)
    raise NotImplementedError


def cw_bound(graph: str, z: F, p: arb_mat, delta: arb) -> Dict:
    """For every n >= 0:  P(N_n <= 0) <= E z^{N_n} <= C * rho^n, with rho, C rigorous upper bounds.

    If M(z) >= 0 entrywise and M v <= rho v for v > 0, then p' <= c v with c = max (p_j + delta)/v_j
    gives M^n p' <= c rho^n v, so E z^{N_n} = 1^T M^n p' <= c rho^n sum(v)."""
    import numpy as np
    M = tilted_monodromy(graph, z)
    ctx.prec = PREC
    Mf = np.array([[float(M[i, j].mid()) for j in range(N)] for i in range(N)])
    w, V = np.linalg.eig(Mf)
    v = np.abs(np.real(V[:, np.argmax(np.real(w))]))
    vq = [F(float(x)).limit_denominator(10 ** 12) for x in v]
    va = [A(x) for x in vq]
    Mv = [sum((M[i, j] * va[j] for j in range(N)), arb(0)) for i in range(N)]
    rho = max((Mv[i] / va[i]).upper() for i in range(N))
    rho_lo = min((Mv[i] / va[i]).lower() for i in range(N))
    c = max(((p[j, 0] + delta) / va[j]).upper() for j in range(N))
    C = (c * sum(va, arb(0))).upper()
    return dict(z=z, rho=rho, rho_lo=rho_lo, C=C)
