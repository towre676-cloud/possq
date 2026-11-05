"""Sharp certified tail bounds for integer-valued random variables from certified
moment and probability-generating-function enclosures.

Data.  An integer-valued N with enclosures
    E[1] = 1,  E[N^k] in [lo_k, hi_k] (k = 1..d),  E[z_j^N] in [lo_j, hi_j],  0 < z_j < 1 rational.
Every test function  q(n) = sum_k a_k n^k + sum_j b_j z_j^n  takes RATIONAL values on the
integers, so the whole certificate is decided in exact Fraction arithmetic.

Upper certificate for P(N <= t):  q(n) >= 1 for n <= t and q(n) >= 0 for n > t, on all of Z.
Verified exactly on a window [-L, U]; the two tails are closed analytically:

  right tail (n >= U+1):  every z_j^n <= z_j^(U+1), so q >= poly(n) - C_R, and the quadratic
                          poly - C_R is checked on [U+1, oo) at its exact real minimum;
  left tail (m = -n >= L+1): with r_j = 1/z_j, let r* be the largest r_j carrying b_j > 0.
                          If every negative b_j has r_j < r*, then for m >= 2 r*/(r*-1)
                          h(m) = sum_{b_j<0} |b_j| (r_j/r*)^m + P(m) r*^(-m)
                          is nonincreasing (Lemma T), so  b* > h(L+1)  certifies q(-m) >= 1.

Lower bracket: an atomic witness on finitely many integers whose moments equal exact rational
targets inside every enclosure, solved exactly over Q and checked nonnegative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

Interval = Tuple[F, F]


@dataclass
class IntegerData:
    degree: int                      # polynomial moments E[N^k], k = 1..degree (degree <= 2 certified)
    poly: List[Interval]             # enclosures of E[N^k], k = 1..degree
    zs: List[F]                      # rational bases 0 < z < 1
    pgf: List[Interval]              # enclosures of E[z^N]
    label: str = ""

    def __post_init__(self):
        assert self.degree <= 2 and len(self.poly) == self.degree
        assert all(0 < z < 1 for z in self.zs) and len(self.zs) == len(self.pgf)
        assert all(lo <= hi for lo, hi in self.poly + self.pgf)

    @property
    def nfun(self):
        return 1 + self.degree + len(self.zs)

    def intervals(self) -> List[Interval]:
        return [(F(1), F(1))] + list(self.poly) + list(self.pgf)

    def phi_exact(self, n: int) -> List[F]:
        return [F(n) ** k for k in range(self.degree + 1)] + [z ** n for z in self.zs]

    def upper(self, c: Sequence[F]) -> F:
        return sum((max(ck * lo, ck * hi) for ck, (lo, hi) in zip(c, self.intervals())), F(0))

    def contains(self, moments: Sequence[F]) -> bool:
        return all(lo <= m <= hi for m, (lo, hi) in zip(moments, self.intervals()))


def q_exact(c: Sequence[F], data: IntegerData, n: int) -> F:
    return sum((ck * v for ck, v in zip(c, data.phi_exact(n))), F(0))


# ----------------------------------------------------------------------------
# tails
# ----------------------------------------------------------------------------
def right_tail_slack(c, data, U) -> Optional[F]:
    """Exact lower bound of q(n) on n >= U+1 (target 0), or None if not provable."""
    d = data.degree
    a = list(c[: d + 1]) + [F(0)] * (3 - (d + 1))
    a0, a1, a2 = a
    b = c[d + 1:]
    CR = sum((-bj * z ** (U + 1) for bj, z in zip(b, data.zs) if bj < 0), F(0))
    n0 = F(U + 1)
    if a2 > 0:
        v = -a1 / (2 * a2)
        x = max(n0, v)
        m = a0 + a1 * x + a2 * x * x
    elif a2 == 0 and a1 >= 0:
        m = a0 + a1 * n0
    else:
        return None
    return m - CR


def left_tail_ok(c, data, L, thresh_target=F(1)) -> Tuple[bool, Dict]:
    """Lemma T check that q(-m) >= target for all integers m >= L+1."""
    d = data.degree
    a = list(c[: d + 1]) + [F(0)] * (3 - (d + 1))
    a0, a1, a2 = a
    b = list(c[d + 1:])
    rs = [1 / z for z in data.zs]
    pos = [j for j, bj in enumerate(b) if bj > 0]
    if not pos:
        return False, {"reason": "no positive geometric coefficient"}
    js = max(pos, key=lambda j: rs[j])
    rstar, bstar = rs[js], b[js]
    for j, bj in enumerate(b):
        if bj < 0 and rs[j] >= rstar:
            return False, {"reason": "negative coefficient at dominant rate"}
    m0 = math.ceil(2 * rstar / (rstar - 1))
    m = L + 1
    if m < m0:
        return False, {"reason": "window too short", "need_L": m0 - 1}
    # polynomial in m:  q(-m) - target = (a0 - target) - a1 m + a2 m^2 + sum b_j r_j^m
    A = abs(a0 - thresh_target)
    Bc = abs(a1)
    C = -a2 if a2 < 0 else F(0)
    h = sum((-bj * (rs[j] / rstar) ** m for j, bj in enumerate(b) if bj < 0), F(0))
    h += (A + Bc * m + C * m * m) / rstar ** m
    return bstar > h, {"rstar": str(rstar), "bstar": float(bstar), "h": float(h), "m0": m0}


# ----------------------------------------------------------------------------
# dual LP and exact certification
# ----------------------------------------------------------------------------
@dataclass
class TailCertificate:
    threshold: int
    value: F
    coeffs: List[F]
    shift: F
    L: int
    U: int
    left: Dict = field(default_factory=dict)
    lp_value: float = float("nan")

    def as_dict(self):
        return dict(threshold=self.threshold, value=str(self.value), value_float=float(self.value),
                    shift=str(self.shift), L=self.L, U=self.U, lp_value=self.lp_value,
                    coeffs=[str(x) for x in self.coeffs], coeffs_float=[float(x) for x in self.coeffs],
                    left_tail=self.left)


def _dual_lp(data: IntegerData, t: int, L: int, U: int):
    ns = np.arange(-L, U + 1)
    nf = data.nfun
    d = data.degree
    # centred/scaled polynomial basis u = (n - c0)/sc for conditioning (converted back below)
    if d >= 1:
        mlo, mhi = data.poly[0]
        c0 = float((mlo + mhi) / 2)
        sc = max(1.0, (float((data.poly[1][0] + data.poly[1][1]) / 2) - c0 ** 2) ** 0.5) if d == 2 else max(1.0, abs(c0))
    else:
        c0, sc = 0.0, 1.0
    Phi = np.array([[float(v) for v in data.phi_exact(int(n))] for n in ns])
    u = (ns - c0) / sc
    for k in range(1, d + 1):
        Phi[:, k] = u ** k
    mids = [float((lo + hi) / 2) for lo, hi in data.intervals()]
    rads = [float((hi - lo) / 2) for lo, hi in data.intervals()]
    if d >= 1:
        m1 = mids[1]
        mids[1] = (m1 - c0) / sc
        rads[1] = rads[1] / sc
    if d == 2:
        mids[2] = (mids[2] - 2 * c0 * m1 + c0 ** 2) / sc ** 2
        rads[2] = (rads[2] + 2 * abs(c0) * rads[1] * sc) / sc ** 2
    tgt = (ns <= t).astype(float)
    s = np.abs(Phi).max(axis=1)
    # column scaling by the value at n = 0 scale (max over the central band), rows by their max
    cs = np.array([max(abs(v), 1e-300) for v in mids])
    cs[1:d + 1] = 1.0
    Phs = Phi / cs[None, :] / s[:, None]
    mid = np.array(mids) / cs
    rad = np.array(rads) / cs
    cost = np.concatenate([mid, rad])
    I = np.eye(nf)
    A_ub = np.vstack([np.hstack([-Phs, np.zeros((len(ns), nf))]),
                      np.hstack([I, -I]), np.hstack([-I, -I])])
    b_ub = np.concatenate([-tgt / s, np.zeros(2 * nf)])
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nf + [(0, None)] * nf,
                  method="highs", options={"primal_feasibility_tolerance": 1e-10,
                                            "dual_feasibility_tolerance": 1e-10})
    if res.status != 0:
        raise RuntimeError("tail dual LP failed: " + res.message)
    x = res.x[:nf] / cs
    # back to the raw basis 1, n, n^2: a0' + a1' u + a2' u^2
    out = x.copy()
    if d >= 1:
        a = [x[0], x[1], x[2] if d == 2 else 0.0]
        out[0] = a[0] - a[1] * c0 / sc + a[2] * c0 ** 2 / sc ** 2
        out[1] = a[1] / sc - 2 * a[2] * c0 / sc ** 2
        if d == 2:
            out[2] = a[2] / sc ** 2
    slack = res.slack[:len(ns)]
    touching = sorted(int(ns[i]) for i in np.argsort(slack)[:nf])
    return out, res.fun, touching


def tail_certificate(data: IntegerData, t: int = 0, L: int = 25, U: Optional[int] = None,
                     max_L: int = 400) -> TailCertificate:
    """Exactly verified upper bound on P(N <= t) valid for every integer law matching the data."""
    if U is None:
        U = t + 120
    cf, lpv, S = _dual_lp(data, t, L, U)
    c = [F(float(v)) for v in cf]
    # Exact polish: interpolate the target on the LP's active set, then verify exactly.
    cp = _solve_exact([data.phi_exact(n) for n in S], [F(1) if n <= t else F(0) for n in S])
    if cp is not None:
        okw = all(q_exact(cp, data, n) >= (F(1) if n <= t else F(0)) for n in range(-L, U + 1))
        rs = right_tail_slack(cp, data, U)
        okl, info = left_tail_ok(cp, data, L)
        if okw and rs is not None and rs >= 0 and okl:
            cert = TailCertificate(t, data.upper(cp), cp, F(0), L, U, info, float(lpv))
            cert.left["polished_on"] = S
            return cert
    while True:
        # window: required constant shift (exact)
        need = F(0)
        for n in range(-L, U + 1):
            gap = (F(1) if n <= t else F(0)) - q_exact(c, data, n)
            need = max(need, gap)
        rs = right_tail_slack(c, data, U)
        if rs is None:
            raise RuntimeError("right tail not provable (leading coefficient)")
        need = max(need, -rs)
        s = need + (need / 10 ** 9 if need > 0 else F(0))
        cs = list(c)
        cs[0] += s
        ok, info = left_tail_ok(cs, data, L)
        if ok:
            return TailCertificate(t, data.upper(cs), cs, s, L, U, info, float(lpv))
        if L >= max_L:
            raise RuntimeError("left tail not provable: %s" % info)
        L = max(L + 20, info.get("need_L", 0))


# ----------------------------------------------------------------------------
# exact primal witness
# ----------------------------------------------------------------------------
@dataclass
class TailWitness:
    threshold: int
    value: F
    atoms: List[Tuple[int, F]]

    def as_dict(self):
        return dict(threshold=self.threshold, value=str(self.value), value_float=float(self.value),
                    atoms=[[n, str(w), float(w)] for n, w in self.atoms])


def _solve_exact(A: List[List[F]], b: List[F]) -> Optional[List[F]]:
    n = len(A)
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def tail_witness(data: IntegerData, t: int = 0, L: int = 30, U: Optional[int] = None) -> Optional[TailWitness]:
    """Exact integer-supported law with moments at the exact enclosure midpoints."""
    if U is None:
        U = t + 80
    ns = np.arange(-L, U + 1)
    nf = data.nfun
    Phi = np.array([[float(v) for v in data.phi_exact(int(n))] for n in ns]).T   # nf x N
    target = [(lo + hi) / 2 for lo, hi in data.intervals()]
    tf = np.array([float(v) for v in target])
    rs = np.abs(Phi).max(axis=1)
    cs = np.abs(Phi).max(axis=0)
    A = Phi / rs[:, None] / cs[None, :]
    obj = -(ns <= t).astype(float) / cs
    res = linprog(obj, A_eq=A, b_eq=tf / rs, bounds=[(0, None)] * len(ns), method="highs",
                  options={"primal_feasibility_tolerance": 1e-10})
    if res.status != 0:
        return None
    w = res.x / cs
    order = np.argsort(-w)
    support = sorted(int(ns[i]) for i in order[:nf] if w[i] > 0)
    # complete to nf points by neighbours, try a few completions
    cand = [support]
    extra = [n for n in range(-L, U + 1) if n not in support]
    extra.sort(key=lambda n: min(abs(n - s) for s in support) if support else abs(n))
    if len(support) < nf:
        cand = [sorted(support + extra[i:i + nf - len(support)]) for i in range(0, 40)]
    best = None
    for S in cand:
        Amat = [[data.phi_exact(n)[k] for n in S] for k in range(nf)]
        sol = _solve_exact(Amat, target)
        if sol is None or any(x < 0 for x in sol):
            continue
        moments = [sum((wi * data.phi_exact(n)[k] for n, wi in zip(S, sol)), F(0)) for k in range(nf)]
        if not data.contains(moments):
            continue
        val = sum((wi for n, wi in zip(S, sol) if n <= t), F(0))
        if best is None or val > best.value:
            best = TailWitness(t, val, [(n, wi) for n, wi in zip(S, sol) if wi != 0])
    return best


def chernoff_best(data: IntegerData, t: int = 0) -> Tuple[F, F]:
    """Best single-tilt bound min_j hi_j * z_j^(-t)  (the baseline being improved)."""
    vals = [(hi / z ** t, z) for z, (lo, hi) in zip(data.zs, data.pgf)]
    v, z = min(vals)
    return min(v, F(1)), z


def touching_set(cert: TailCertificate, data: IntegerData, k: Optional[int] = None) -> List[int]:
    """Integers where the certificate nearly meets its target (complementary-slackness support)."""
    k = k or data.nfun
    gaps = sorted((q_exact(cert.coeffs, data, n) - (F(1) if n <= cert.threshold else F(0)), n)
                  for n in range(-cert.L, cert.U + 1))
    return sorted(n for _, n in gaps[:k])


def witness_on_support(data: IntegerData, support: Sequence[int], t: int = 0) -> Optional[TailWitness]:
    """Exact law on the given integers with moments at the exact enclosure midpoints."""
    S = sorted(support)
    nf = data.nfun
    if len(S) != nf:
        return None
    target = [(lo + hi) / 2 for lo, hi in data.intervals()]
    sol = _solve_exact([[data.phi_exact(n)[k] for n in S] for k in range(nf)], target)
    if sol is None or any(x < 0 for x in sol):
        return None
    moments = [sum((w * data.phi_exact(n)[k] for n, w in zip(S, sol)), F(0)) for k in range(nf)]
    if not data.contains(moments):
        return None
    return TailWitness(t, sum((w for n, w in zip(S, sol) if n <= t), F(0)),
                       [(n, w) for n, w in zip(S, sol) if w != 0])


# ----------------------------------------------------------------------------
# Exact revised simplex (Bland's rule) on the window LP: certificate and witness at once
# ----------------------------------------------------------------------------
def _solve_T(B: List[List[F]], rhs: List[F]) -> List[F]:
    """Solve B^T y = rhs where B is given column-wise (B[j] = column j)."""
    m = len(B)
    return _solve_exact([[B[j][i] for j in range(m)] for i in range(m)][::1] and
                        [[B[i][j] for j in range(m)] for i in range(m)], rhs)


def _solve_cols(B: List[List[F]], rhs: List[F]) -> List[F]:
    """Solve B x = rhs where B is given column-wise."""
    m = len(B)
    return _solve_exact([[B[j][i] for j in range(m)] for i in range(m)], rhs)


@dataclass
class ExactWindowSolution:
    value: F                 # exact optimum of the window LP (data at midpoints)
    y: List[F]               # optimal dual = certificate coefficients
    support: List[int]
    weights: List[F]
    pivots: int


INF_MINUS, INF_PLUS = "-inf", "+inf"


def exact_window_lp(data: IntegerData, t: int, L: int, U: int, max_pivots: int = 4000,
                    compactify: bool = True) -> ExactWindowSolution:
    """Exact LP over laws on {-L..U}; with compactify=True also over the two points at infinity
    of the compactification of Z (directions phi(n)/r*^|n| -> e_dominant as n -> -oo and
    phi(n)/n^2 -> e_2 as n -> +oo, both with cost 0).  Its dual is then forced to satisfy the
    asymptotic sign conditions, so the window optimum is the infinite-support optimum whenever
    the tail lemmas certify it."""
    m = data.nfun
    b = [(lo + hi) / 2 for lo, hi in data.intervals()]
    window = list(range(-L, U + 1))
    col = {n: data.phi_exact(n) for n in window}
    cost = {n: (F(1) if n <= t else F(0)) for n in window}
    if compactify:
        jd = 1 + data.degree + min(range(len(data.zs)), key=lambda j: data.zs[j]) if data.zs else None
        if jd is not None:
            col[INF_MINUS] = [F(1) if r == jd else F(0) for r in range(m)]
            cost[INF_MINUS] = F(0)
            window = [INF_MINUS] + window
        if data.degree == 2:
            col[INF_PLUS] = [F(1) if r == 2 else F(0) for r in range(m)]
            cost[INF_PLUS] = F(0)
            window = window + [INF_PLUS]
    # basis entries: ('a', i) artificial unit column, or ('x', n)
    basis = [("a", i) for i in range(m)]

    def column(key):
        if key[0] == "a":
            return [F(1) if r == key[1] else F(0) for r in range(m)]
        return col[key[1]]

    pivots = 0
    for phase in (1, 2):
        def c_of(key):
            if phase == 1:
                return F(-1) if key[0] == "a" else F(0)
            return F(-10 ** 30) if key[0] == "a" else cost[key[1]]
        while True:
            B = [column(k) for k in basis]
            xB = _solve_cols(B, b)
            y = _solve_exact([[B[j][i] for i in range(m)] for j in range(m)], [c_of(k) for k in basis])
            entering = None
            inb = {k for k in basis}
            for n in window:                                   # Bland: smallest index first
                if ("x", n) in inb:
                    continue
                d = c_of(("x", n)) - sum((yi * ai for yi, ai in zip(y, col[n])), F(0))
                if d > 0:
                    entering = ("x", n)
                    break
            if entering is None:
                break
            u = _solve_cols(B, column(entering))
            ratios = [(xB[i] / u[i], i) for i in range(m) if u[i] > 0]
            if not ratios:
                raise RuntimeError("window LP unbounded")
            rmin = min(r for r, _ in ratios)
            leave = min((i for r, i in ratios if r == rmin), key=lambda i: (basis[i][0] != "a", basis[i]))
            basis[leave] = entering
            pivots += 1
            if pivots > max_pivots:
                raise RuntimeError("pivot limit")
        if phase == 1:
            B = [column(k) for k in basis]
            xB = _solve_cols(B, b)
            if any(k[0] == "a" and v != 0 for k, v in zip(basis, xB)):
                raise RuntimeError("data infeasible on window")
            # drive zero-level artificials out of the basis
            for i, k in enumerate(list(basis)):
                if k[0] != "a":
                    continue
                for n in window:
                    if ("x", n) in basis:
                        continue
                    u = _solve_cols([column(kk) for kk in basis], col[n])
                    if u[i] != 0:
                        basis[i] = ("x", n)
                        break
    B = [column(k) for k in basis]
    xB = _solve_cols(B, b)
    y = _solve_exact([[B[j][i] for i in range(m)] for j in range(m)], [cost[k[1]] for k in basis])
    sup = [k[1] for k in basis]
    value = sum((cost[n] * w for n, w in zip(sup, xB)), F(0))
    key = lambda v: (-10 ** 9 if v == INF_MINUS else 10 ** 9 if v == INF_PLUS else v)
    order = sorted(range(m), key=lambda i: key(sup[i]))
    return ExactWindowSolution(value, y, [sup[i] for i in order], [xB[i] for i in order], pivots)


def sharp_tail(data: IntegerData, t: int = 0, L: int = 25, U: Optional[int] = None, max_L: int = 200):
    """Exact sharp bracket for sup P(N <= t) over all integer laws consistent with the data.

    Returns (certificate, witness, window_solution).  The certificate is valid on all of Z;
    the witness is an exact law at the enclosure midpoints.  When the window optimum's dual
    passes both tail lemmas, certificate.value - witness.value equals the radius term only.
    """
    if U is None:
        U = t + 120
    while True:
        try:
            sol = exact_window_lp(data, t, L, U, compactify=True)
        except RuntimeError as ex:
            if "infeasible" not in str(ex) or L > max_L:
                raise
            L, U = L + 25, U + 100          # the window's moment cone is too small: enlarge it
            continue
        y = sol.y
        rs = right_tail_slack(y, data, U)
        okl, info = left_tail_ok(y, data, L)
        if rs is not None and rs >= 0 and okl:
            info["window_pivots"] = sol.pivots
            info["atoms_at_infinity"] = [str(n) for n, w in zip(sol.support, sol.weights)
                                         if isinstance(n, str) and w != 0]
            cert = TailCertificate(t, data.upper(y), y, F(0), L, U, info, float(sol.value))
            if not info["atoms_at_infinity"]:
                wsol = sol
            else:                       # attained only in the limit; exact finite witness separately
                wsol = None
                for LL, UU in ((L, U), (L + 30, U + 100), (L + 60, U + 250)):
                    try:
                        wsol = exact_window_lp(data, t, LL, UU, compactify=False)
                        break
                    except RuntimeError:
                        continue
            if wsol is None:
                return cert, None, sol
            atoms = [(n, w) for n, w in zip(wsol.support, wsol.weights) if w != 0 and not isinstance(n, str)]
            wit = TailWitness(t, wsol.value, atoms)
            assert data.contains([sum((w * data.phi_exact(n)[k] for n, w in wit.atoms), F(0))
                                  for k in range(data.nfun)])
            return cert, wit, sol
        if rs is None or rs < 0:
            U += 60
        if not okl:
            L = max(L + 20, info.get("need_L", 0))
        if L > max_L:
            raise RuntimeError("tails not provable within max_L: %s" % info)
