"""Certified spectral-mass bounds from noisy moments.

Data model.  A positive Borel measure mu on I=[lo,hi] is observed through
noisy generalized moments  mu_hat_k  with  |int phi_k dmu - mu_hat_k| <= eps_k,
k = 0..K, in a basis with phi_0 = 1.

Dual functional.  For q = sum_k c_k phi_k and every compatible mu,
    int q dmu <= U(c) := sum c_k mu_hat_k + sum |c_k| eps_k,
    int q dmu >= L(c) := sum c_k mu_hat_k - sum |c_k| eps_k.

Certificates.
  * Upper: q >= 0 on I and q >= 1 on closure(S)  ==>  mu(S) <= U(c).
  * Lower: q <= 0 on closure(I \\ S) and q <= 1 on closure(S)  ==>  mu(S) >= L(c).
Both side conditions are verified exactly (Sturm) after a rational constant shift.

Primal witnesses.  A finite atomic measure nu with exactly verified moment
compatibility gives  sup mu(S) >= nu(S)  (resp. inf mu(S) <= nu(S)), so the
sharp value is bracketed between a witness and a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from .exact import (Basis, Interval, certified_lower_bound, certified_shift, clip,
                    complement, merge, peval, to_fmpq)


@dataclass
class MomentData:
    basis: Basis
    mu_hat: List[F]
    eps: List[F]
    I: Interval = (F(-1), F(1))

    def __post_init__(self):
        K = self.basis.K
        assert len(self.mu_hat) == K + 1 and len(self.eps) == K + 1
        assert all(e >= 0 for e in self.eps)
        self.I = (F(self.I[0]), F(self.I[1]))

    def upper(self, c: Sequence[F]) -> F:
        return sum((ck * m + abs(ck) * e for ck, m, e in zip(c, self.mu_hat, self.eps)), F(0))

    def lower(self, c: Sequence[F]) -> F:
        return sum((ck * m - abs(ck) * e for ck, m, e in zip(c, self.mu_hat, self.eps)), F(0))

    def compatible(self, moments: Sequence[F]) -> bool:
        return all(abs(m - h) <= e for m, h, e in zip(moments, self.mu_hat, self.eps))


# ----------------------------------------------------------------------------
# Theorem A: the positive-square leakage bound
# ----------------------------------------------------------------------------
@dataclass
class LeakageCertificate:
    eta: F
    g: F
    bound: Optional[F]
    coeffs: List[F]
    outside: List[Interval]

    def as_dict(self):
        return dict(eta=str(self.eta), g=str(self.g),
                    bound=None if self.bound is None else str(self.bound),
                    eta_float=float(self.eta), g_float=float(self.g),
                    bound_float=None if self.bound is None else float(self.bound),
                    outside=[[str(a), str(b)] for a, b in self.outside])


def leakage_bound_square(data: MomentData, p, U: Sequence[Interval]) -> LeakageCertificate:
    """mu(I \\ U) <= eta / g  with  q = p^2  (nonnegative on R by construction)."""
    q = p * p
    c = data.basis.coeffs_of(q)
    eta = data.upper(c)
    outside = complement(data.I, U)
    if not outside:
        return LeakageCertificate(eta, F(0), F(0), c, outside)
    g = certified_lower_bound(q, outside)
    bound = eta / g if g > 0 else None
    return LeakageCertificate(eta, g, bound, c, outside)


def separation_floor(nodes: Sequence[F], r: F) -> F:
    """Lemma B floor  r^2 (Delta/2)^{2(n-1)}  for p = prod (x - x_i), r <= Delta/2."""
    nodes = sorted(F(x) for x in nodes)
    n = len(nodes)
    if n == 1:
        return r * r
    Delta = min(nodes[i + 1] - nodes[i] for i in range(n - 1))
    return r * r * (Delta / 2) ** (2 * (n - 1))


# ----------------------------------------------------------------------------
# LP machinery
# ----------------------------------------------------------------------------
def _grid(intervals: Sequence[Interval], density: int = 600, minpts: int = 12) -> np.ndarray:
    pts = []
    for a, b in intervals:
        a, b = float(a), float(b)
        if b <= a:
            pts.append(a)
            continue
        m = max(minpts, int(density * (b - a)) + 8)
        j = np.arange(m + 1)
        pts.extend(a + (b - a) * (1 - np.cos(np.pi * j / m)) / 2)
    return np.array(pts)


@dataclass
class MassCertificate:
    side: str
    value: F
    coeffs: List[F]
    shift: F
    S: List[Interval]
    lp_value: float

    def as_dict(self):
        return dict(side=self.side, value=str(self.value), value_float=float(self.value),
                    shift=str(self.shift), shift_float=float(self.shift),
                    lp_value=self.lp_value, S=[[str(a), str(b)] for a, b in self.S],
                    sensitivity=[float(abs(c)) for c in self.coeffs])


def _solve_dual(data: MomentData, S: List[Interval], side: str, degree: Optional[int],
                density: int):
    basis = data.basis
    K = basis.K
    d = K if degree is None else min(degree, K)
    A = basis.cheb_to_self_matrix()[:, : d + 1]          # self coeffs = A @ chebcoeffs
    mh = np.array([float(v) for v in data.mu_hat])
    ep = np.array([float(v) for v in data.eps])
    Sc = complement(data.I, S) if side == "upper" else None
    comp = Sc
    if side == "lower":
        comp = complement(data.I, S)
    gS = _grid(S, density)
    gC = _grid(comp, density) if comp else np.zeros(0)
    nK = K + 1
    if side == "upper":
        cost = np.concatenate([A.T @ mh, ep])
        sgn, tS = -1.0, 1.0          # constraints  -q <= -1 on S,  -q <= 0 on comp
    else:
        cost = np.concatenate([-(A.T @ mh), ep])
        sgn, tS = 1.0, 1.0           # constraints   q <= 1 on S,   q <= 0 on comp
    Z = lambda n: np.zeros((n, nK))
    box = 1e7
    bounds = [(-box, box)] * (d + 1) + [(0, None)] * nK
    fineS = _grid(S, 20 * density)
    fineC = _grid(comp, 20 * density) if comp else np.zeros(0)
    # Exchange (cutting-plane) loop: semi-infinite constraints enforced on a grid that
    # is enriched with the worst violators until the continuous violation is negligible.
    for _round in range(40):
        VS = np.polynomial.chebyshev.chebvander(gS, d)
        VC = np.polynomial.chebyshev.chebvander(gC, d) if len(gC) else np.zeros((0, d + 1))
        rows = [np.hstack([sgn * VS, Z(len(gS))])]
        rhs = [sgn * tS * np.ones(len(gS))]
        if len(gC):
            rows.append(np.hstack([sgn * VC, Z(len(gC))])); rhs.append(np.zeros(len(gC)))
        rows.append(np.hstack([A, -np.eye(nK)])); rhs.append(np.zeros(nK))
        rows.append(np.hstack([-A, -np.eye(nK)])); rhs.append(np.zeros(nK))
        Aub, bub = np.vstack(rows), np.concatenate(rhs)
        for meth in ("highs", "highs-ds", "highs-ipm"):
            res = linprog(cost, A_ub=Aub, b_ub=bub, bounds=bounds, method=meth)
            if res.status == 0:
                break
        if res.status != 0:
            if _round > 0:      # keep the last solved iterate; the exact shift repairs it
                res = last
                break
            raise RuntimeError("dual LP failed: " + res.message)
        last = res
        ch = res.x[: d + 1]
        # violations on fine grids enriched with the critical points of q
        crit = np.polynomial.chebyshev.chebroots(np.polynomial.chebyshev.chebder(ch))
        crit = crit[np.abs(crit.imag) < 1e-7].real
        lo_, hi_ = float(data.I[0]), float(data.I[1])
        crit = crit[(crit >= lo_) & (crit <= hi_)]
        fineS = np.concatenate([fineS, [x for x in crit if any(float(a) <= x <= float(b) for a, b in S)]])
        if comp:
            fineC = np.concatenate([fineC, [x for x in crit if any(float(a) <= x <= float(b) for a, b in comp)]])
        vS = sgn * np.polynomial.chebyshev.chebval(fineS, ch) - sgn * tS
        add_S = fineS[vS > 1e-12]
        add_C = np.zeros(0)
        if len(fineC):
            vC = sgn * np.polynomial.chebyshev.chebval(fineC, ch)
            add_C = fineC[vC > 1e-12]
        if len(add_S) == 0 and len(add_C) == 0:
            break
        # keep the worst few violators per round
        if len(add_S):
            idx = np.argsort(-(sgn * np.polynomial.chebyshev.chebval(add_S, ch) - sgn * tS))[:60]
            gS = np.concatenate([gS, add_S[idx]])
        if len(add_C):
            idx = np.argsort(-(sgn * np.polynomial.chebyshev.chebval(add_C, ch)))[:60]
            gC = np.concatenate([gC, add_C[idx]])
    val = res.fun if side == "upper" else -res.fun
    return ch, val, comp


def mass_certificate(data: MomentData, S: Sequence[Interval], side: str = "upper",
                     degree: Optional[int] = None, density: int = 600) -> MassCertificate:
    """Exactly verified bound on mu(S); S a finite union of intervals inside I."""
    S = merge(clip(data.I, S))
    if side == "upper" and not S:
        return MassCertificate(side, F(0), [F(0)] * (data.basis.K + 1), F(0), S, 0.0)
    ch, val, comp = _solve_dual(data, S, side, degree, density)
    from .exact import cheb_to_mono
    P = cheb_to_mono([F(float(v)).limit_denominator(10 ** 15) for v in ch])
    one = to_fmpq(1)
    if side == "upper":
        s1 = certified_shift(P - one, S)
        s2 = certified_shift(P, comp) if comp else F(0)
        s = max(s1, s2)
        P = P + to_fmpq(s)
        c = data.basis.coeffs_of(P)
        value = data.upper(c)
    else:
        s1 = certified_shift(one - P, S)
        s2 = certified_shift(-P, comp) if comp else F(0)
        s = max(s1, s2)
        P = P - to_fmpq(s)
        c = data.basis.coeffs_of(P)
        value = max(F(0), data.lower(c))
    return MassCertificate(side, value, c, s, S, float(val))


def mass_lp_value(data: MomentData, S, side="upper", degree=None, density=400) -> float:
    S = merge(clip(data.I, S))
    if not S:
        return 0.0
    return _solve_dual(data, S, side, degree, density)[1]


# ----------------------------------------------------------------------------
# primal witnesses
# ----------------------------------------------------------------------------
@dataclass
class Witness:
    side: str
    value: F
    atoms: List[Tuple[F, F]]

    def as_dict(self):
        return dict(side=self.side, value=str(self.value), value_float=float(self.value),
                    n_atoms=len(self.atoms))


def _in_union(x: F, S: Sequence[Interval], openset: bool) -> bool:
    for a, b in S:
        if (a < x < b) if openset else (a <= x <= b):
            return True
    return False


def primal_witness(data: MomentData, S: Sequence[Interval], side: str = "upper",
                   S_open: bool = False, density: int = 500,
                   extra_points: Sequence[F] = (), local_halfwidth: float = 2e-3,
                   local_pts: int = 40) -> Optional[Witness]:
    """Exactly verified atomic measure compatible with the data, extremal for mu(S)."""
    basis = data.basis
    lo, hi = data.I
    m = int(density * float(hi - lo)) + 40
    xs = sorted(set(F(round((float(lo) + float(hi - lo) * (1 - np.cos(np.pi * j / m)) / 2)
                            * 2 ** 24), 2 ** 24) for j in range(m + 1)) | {lo, hi} |
                {e for iv in S for e in iv if lo <= e <= hi} |
                {F(x) for x in extra_points} |
                {F(round((float(x) + local_halfwidth * t) * 2 ** 30), 2 ** 30)
                 for x in extra_points for t in np.linspace(-1, 1, local_pts + 1)})
    xs = [x for x in xs if lo <= x <= hi]
    inS = np.array([_in_union(x, S, S_open) for x in xs], dtype=float)
    Phi = [basis.exact_moment_atom(x) for x in xs]
    PhiF = np.array([[float(v) for v in row] for row in Phi])  # N x (K+1)
    mh = np.array([float(v) for v in data.mu_hat])
    ep = np.array([float(v) for v in data.eps])
    obj = -inS if side == "upper" else inS
    opts = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
    for tau in (1e-4, 1e-2, 1e-1, 3e-1):
        e2 = ep * (1 - tau)
        res = linprog(obj, A_ub=np.vstack([PhiF.T, -PhiF.T]),
                      b_ub=np.concatenate([mh + e2, -(mh - e2)]),
                      bounds=[(0, None)] * len(xs), method="highs", options=opts)
        if res.status != 0:
            continue
        w = [F(float(v)) if v > 0 else F(0) for v in res.x]
        mom = [sum((wi * Phi[j][k] for j, wi in enumerate(w) if wi), F(0))
               for k in range(basis.K + 1)]
        if data.compatible(mom):
            val = sum((wi for wi, s in zip(w, inS) if s and wi), F(0))
            atoms = [(x, wi) for x, wi in zip(xs, w) if wi]
            return Witness(side, val, atoms)
    return None
