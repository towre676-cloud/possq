"""Exact rational polynomial core: Chebyshev/monomial bases, Sturm counting,
certified positivity on closed rational intervals.

All certificates are decided in exact rational arithmetic (python-flint fmpq_poly).
Floating point is used only to *propose* shifts and lower bounds; every proposal
is accepted only after an exact Sturm verification.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from flint import fmpq, fmpq_poly

Interval = Tuple[F, F]


# ----------------------------------------------------------------------------
# conversions
# ----------------------------------------------------------------------------
def to_fmpq(x) -> fmpq:
    x = F(x)
    return fmpq(x.numerator, x.denominator)


def to_F(a) -> F:
    a = fmpq(a)
    return F(int(a.p), int(a.q))


def poly(coeffs: Iterable) -> fmpq_poly:
    """Monomial coefficients, low degree first."""
    return fmpq_poly([to_fmpq(c) for c in coeffs])


def mono_coeffs(P: fmpq_poly) -> List[F]:
    return [to_F(c) for c in P.coeffs()]


def peval(P: fmpq_poly, x) -> F:
    return to_F(P(to_fmpq(x)))


def from_roots(roots: Sequence) -> fmpq_poly:
    P = fmpq_poly([1])
    for r in roots:
        P *= fmpq_poly([-to_fmpq(r), 1])
    return P


# ----------------------------------------------------------------------------
# Chebyshev polynomials of the first kind (exact)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def cheb_T(k: int) -> fmpq_poly:
    if k == 0:
        return fmpq_poly([1])
    if k == 1:
        return fmpq_poly([0, 1])
    return fmpq_poly([0, 2]) * cheb_T(k - 1) - cheb_T(k - 2)


def mono_to_cheb(P: fmpq_poly) -> List[F]:
    """Exact Chebyshev coefficients of P (low degree first)."""
    d = P.degree()
    if d < 0:
        return [F(0)]
    out = [F(0)] * (d + 1)
    R = fmpq_poly(P)
    for k in range(d, -1, -1):
        if R.degree() < k:
            continue
        lead = to_F(R.coeffs()[k])
        lt = to_F(cheb_T(k).coeffs()[k])
        ck = lead / lt
        out[k] = ck
        R = R - to_fmpq(ck) * cheb_T(k)
    assert R.degree() < 0 or all(c == 0 for c in R.coeffs())
    return out


def cheb_to_mono(c: Sequence) -> fmpq_poly:
    P = fmpq_poly([0])
    for k, ck in enumerate(c):
        if ck != 0:
            P += to_fmpq(ck) * cheb_T(k)
    return P


# ----------------------------------------------------------------------------
# measurement bases
# ----------------------------------------------------------------------------
class Basis:
    """A polynomial basis phi_0..phi_K in which moments are measured.

    phi_0 = 1 in both supported bases, so a constant shift of a certificate
    acts only on coefficient 0.
    """

    def __init__(self, name: str, K: int):
        if name not in ("chebyshev", "monomial"):
            raise ValueError(name)
        self.name = name
        self.K = K

    def phi(self, k: int) -> fmpq_poly:
        if self.name == "chebyshev":
            return cheb_T(k)
        return fmpq_poly([0] * k + [1])

    def coeffs_of(self, P: fmpq_poly) -> List[F]:
        if P.degree() > self.K:
            raise ValueError("degree %d exceeds measured order %d" % (P.degree(), self.K))
        if self.name == "chebyshev":
            c = mono_to_cheb(P)
        else:
            c = mono_coeffs(P)
        return c + [F(0)] * (self.K + 1 - len(c))

    def poly_of(self, c: Sequence) -> fmpq_poly:
        if self.name == "chebyshev":
            return cheb_to_mono(c)
        return poly(c)

    def vander(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.name == "chebyshev":
            return np.polynomial.chebyshev.chebvander(x, self.K)
        return np.vander(x, self.K + 1, increasing=True)

    def cheb_to_self_matrix(self) -> np.ndarray:
        """Float matrix A with  (coeffs in self) = A @ (Chebyshev coeffs)."""
        if self.name == "chebyshev":
            return np.eye(self.K + 1)
        A = np.zeros((self.K + 1, self.K + 1))
        for j in range(self.K + 1):
            m = mono_coeffs(cheb_T(j))
            for i, v in enumerate(m):
                A[i, j] = float(v)
        return A

    def exact_moment_atom(self, x) -> List[F]:
        xq = to_fmpq(x)
        return [to_F(self.phi(k)(xq)) for k in range(self.K + 1)]

    def exact_moment_uniform(self, a, b) -> List[F]:
        """Moments of Lebesgue measure on [a,b]."""
        out = []
        for k in range(self.K + 1):
            m = mono_coeffs(self.phi(k))
            s = F(0)
            for j, cj in enumerate(m):
                s += cj * (F(b) ** (j + 1) - F(a) ** (j + 1)) / (j + 1)
            out.append(s)
        return out


# ----------------------------------------------------------------------------
# Sturm sequences and certified positivity
# ----------------------------------------------------------------------------
def sturm_sequence(P: fmpq_poly) -> List[fmpq_poly]:
    seq = [P, P.derivative()]
    while seq[-1].degree() > 0:
        r = -(seq[-2] % seq[-1])
        if r.degree() < 0:
            break
        seq.append(r)
    return seq


def _variations(seq, x: fmpq) -> int:
    signs = []
    for s in seq:
        if s.degree() < 0:
            continue
        v = s(x)
        if v != 0:
            signs.append(v > 0)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def count_distinct_roots(P: fmpq_poly, a, b) -> int:
    """Number of distinct real roots of P in (a, b] (exact)."""
    if P.degree() <= 0:
        return 0
    seq = sturm_sequence(P)
    return _variations(seq, to_fmpq(a)) - _variations(seq, to_fmpq(b))


def certify_positive(P: fmpq_poly, a, b) -> bool:
    """Exact decision of  P(x) > 0 for all x in the closed interval [a, b]."""
    a, b = F(a), F(b)
    if a > b:
        return True
    pa, pb = peval(P, a), peval(P, b)
    if pa <= 0 or pb <= 0:
        return False
    if a == b or P.degree() <= 0:
        return True
    return count_distinct_roots(P, a, b) == 0


# ----------------------------------------------------------------------------
# float-guided, exactly verified shifts and lower bounds
# ----------------------------------------------------------------------------
def _float_cheb(P: fmpq_poly) -> np.ndarray:
    return np.array([float(v) for v in mono_to_cheb(P)])


def float_min(P: fmpq_poly, a, b, dense: int = 400) -> float:
    """Floating estimate of min_[a,b] P, evaluated in the Chebyshev basis (stable on [-1,1])."""
    c = _float_cheb(P)
    a, b = float(a), float(b)
    pts = [a, b]
    if len(c) > 2:
        for r in np.polynomial.chebyshev.chebroots(np.polynomial.chebyshev.chebder(c)):
            if abs(r.imag) < 1e-7 and a <= r.real <= b:
                pts.append(r.real)
    pts.extend(np.linspace(a, b, dense))
    v = np.polynomial.chebyshev.chebval(np.array(pts), c)
    return float(np.min(v))


def certified_shift(P: fmpq_poly, intervals: Sequence[Interval]) -> F:
    """Smallest-found rational s >= 0 with P + s > 0 on every closed interval (exact)."""
    if not intervals:
        return F(0)
    m = min(float_min(P, a, b) for a, b in intervals)
    scale = 1.0 + float(np.abs(_float_cheb(P)).sum())
    base = max(0.0, -m)
    tol = 1e-15 * scale
    for _ in range(80):
        s = F(base + tol).limit_denominator(10 ** 15)
        if s < 0:
            s = F(0)
        Q = P + to_fmpq(s)
        if all(certify_positive(Q, a, b) for a, b in intervals):
            return s
        tol *= 4
    raise RuntimeError("could not certify a shift")


def certified_lower_bound(P: fmpq_poly, intervals: Sequence[Interval]) -> F:
    """Rational g with P > g on every closed interval (exact); g may be <= 0."""
    m = min(float_min(P, a, b) for a, b in intervals)
    rel = 1e-9
    for _ in range(80):
        g = F(m - rel * (abs(m) + 1e-300)).limit_denominator(10 ** 18)
        if g >= F(m):
            g = F(m) - F(1, 10 ** 30)
        Q = P - to_fmpq(g)
        if all(certify_positive(Q, a, b) for a, b in intervals):
            return g
        rel *= 4
    raise RuntimeError("could not certify lower bound")


# ----------------------------------------------------------------------------
# interval set algebra on I = [lo, hi]
# ----------------------------------------------------------------------------
def merge(intervals: Sequence[Interval]) -> List[Interval]:
    iv = sorted((F(a), F(b)) for a, b in intervals if F(a) < F(b))
    out: List[Interval] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def complement(I: Interval, opens: Sequence[Interval]) -> List[Interval]:
    """Closed intervals of I not covered by the union of the open intervals."""
    lo, hi = F(I[0]), F(I[1])
    U = merge([(max(a, lo - 1), min(b, hi + 1)) for a, b in opens])
    out = []
    cur = lo
    for a, b in U:
        if b <= lo or a >= hi:
            continue
        if a > cur:
            out.append((cur, min(a, hi)))
        cur = max(cur, b)
    if cur <= hi:
        out.append((cur, hi))
    return [(a, b) for a, b in out if a <= b]


def clip(I: Interval, intervals: Sequence[Interval]) -> List[Interval]:
    lo, hi = F(I[0]), F(I[1])
    return [(max(F(a), lo), min(F(b), hi)) for a, b in intervals if F(b) > lo and F(a) < hi]


def length(intervals: Sequence[Interval]) -> F:
    return sum((b - a for a, b in merge(intervals)), F(0))
