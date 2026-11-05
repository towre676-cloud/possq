"""Certified spectral mass directly from noisy rational resolvent readings.

For a positive measure on I, observe h(s)=integral (lambda+s)^(-1) dmu.
All denominators must be positive on I. Floating-point linear programmes propose
rational sandwiches; exact Sturm tests on their common-denominator numerators
accept them. No numerical differentiation or conversion to moments is used.
Targets are finite unions of closed intervals, relative to I, including
singletons and support endpoints. Atomic witnesses use the same closed-set
convention and are accepted only after exact compatibility checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from typing import Optional, Sequence

import numpy as np
from flint import ctx, fmpq_poly
from scipy.optimize import linprog

from .exact import Interval, certify_positive, mono_to_cheb, poly, to_fmpq


@dataclass
class ResolventData:
    I: Interval
    shifts: Sequence[F]
    hhat: Sequence[F]
    eps: Sequence[F]
    mass: F = F(1)
    mass_eps: F = F(0)

    def __post_init__(self):
        self.I = tuple(F(v) for v in self.I)
        self.shifts = [F(v) for v in self.shifts]
        self.hhat = [F(v) for v in self.hhat]
        self.eps = [F(v) for v in self.eps]
        self.mass, self.mass_eps = F(self.mass), F(self.mass_eps)
        if len(self.I) != 2 or not 0 < self.I[0] < self.I[1]:
            raise ValueError("I must be a nondegenerate interval with positive support")
        if not len(self.shifts) == len(self.hhat) == len(self.eps):
            raise ValueError("shifts, hhat, and eps must have equal lengths")
        if any(v < 0 for v in self.eps) or self.mass < 0 or self.mass_eps < 0:
            raise ValueError("error budgets and reported mass must be nonnegative")
        if any(self.I[0] + s <= 0 for s in self.shifts):
            raise ValueError("every resolvent denominator must be positive on I")

    @property
    def values(self):
        return [self.mass] + list(self.hhat)

    @property
    def budgets(self):
        return [self.mass_eps] + list(self.eps)

    def phi(self, x):
        x = F(x)
        return [F(1)] + [1 / (x + s) for s in self.shifts]

    def upper(self, coeffs):
        return sum((c * y + abs(c) * e for c, y, e in
                    zip(coeffs, self.values, self.budgets)), F(0))

    def lower(self, coeffs):
        return sum((c * y - abs(c) * e for c, y, e in
                    zip(coeffs, self.values, self.budgets)), F(0))

    def as_dict(self):
        return dict(I=[str(v) for v in self.I], shifts=[str(v) for v in self.shifts],
                    hhat=[str(v) for v in self.hhat], eps=[str(v) for v in self.eps],
                    mass=str(self.mass), mass_eps=str(self.mass_eps))

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("I", "shifts", "hhat", "eps", "mass", "mass_eps")})


def _target(data, S):
    # Closed singleton intervals must be retained: dropping one could falsely
    # certify zero mass at a genuine atom.
    lo, hi = data.I
    clipped = []
    for a, b in S:
        a, b = F(a), F(b)
        if a > b:
            raise ValueError("target interval endpoints must be ordered")
        a, b = max(lo, a), min(hi, b)
        if a <= b:
            clipped.append((a, b))
    out = []
    for a, b in sorted(clipped):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _complement(data, S):
    # Closure of I minus the closed target. Shared endpoints are deliberately
    # imposed on both pieces; this is conservative and safe for either convention.
    out, cur = [], data.I[0]
    for a, b in S:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < data.I[1]:
        out.append((cur, data.I[1]))
    return out


def _scaled_polynomials(data, coeffs):
    """N,D in x in [-1,1], q(center+radius*x)=N(x)/D(x)."""
    lo, hi = data.I
    center, radius = (lo + hi) / 2, (hi - lo) / 2
    factors = [poly([center + s, radius]) for s in data.shifts]
    D = poly([1])
    for factor in factors:
        D *= factor
    N = to_fmpq(coeffs[0]) * D
    for j, c in enumerate(coeffs[1:]):
        # Exact division retains repeated shifts without any special case.
        Q, R = divmod(D, factors[j])
        if R.degree() >= 0:
            raise ArithmeticError("denominator division was not exact")
        N += to_fmpq(c) * Q
    return N, D


def _x_intervals(data, intervals):
    lo, hi = data.I
    center, radius = (lo + hi) / 2, (hi - lo) / 2
    return [((a - center) / radius, (b - center) / radius) for a, b in intervals]


def _positive_or_zero(P, intervals):
    if P.degree() < 0:
        return True
    return all(certify_positive(P, a, b) for a, b in intervals)


@dataclass
class ResolventCertificate:
    side: str
    value: F
    coeffs: list[F]           # raw basis 1, 1/(lambda+s_j)
    shift: F                 # constant repair applied to the proposed q
    S: list[Interval]
    lp_value: float = float("nan")

    def as_dict(self):
        return dict(side=self.side, value=str(self.value), value_float=float(self.value),
                    coeffs=[str(v) for v in self.coeffs], shift=str(self.shift),
                    S=[[str(a), str(b)] for a, b in self.S], lp_value=self.lp_value,
                    target_convention="closed intervals relative to I")

    @classmethod
    def from_dict(cls, d):
        return cls(d["side"], F(d["value"]), [F(v) for v in d["coeffs"]],
                   F(d["shift"]), [(F(a), F(b)) for a, b in d["S"]],
                   float(d.get("lp_value", float("nan"))))


def verify_resolvent_certificate(data, certificate) -> bool:
    """Independently verify a certificate without an LP, grid, or float tolerance.

    Exact zero numerators are accepted; otherwise the pipeline's strict Sturm
    positivity conditions are checked. Some valid tangent sandwiches may be
    rejected by this deliberately conservative verifier.
    """
    try:
        data = ResolventData.from_dict(data.as_dict())
        cert = (ResolventCertificate.from_dict(certificate)
                if isinstance(certificate, dict) else ResolventCertificate(
                    certificate.side, F(certificate.value), [F(v) for v in certificate.coeffs],
                    F(certificate.shift), [(F(a), F(b)) for a, b in certificate.S],
                    certificate.lp_value))
        if cert.side not in ("upper", "lower") or len(cert.coeffs) != len(data.values):
            return False
        if F(cert.shift) < 0 or _target(data, cert.S) != cert.S:
            return False
        if any(data.I[0] + s <= 0 for s in data.shifts):
            return False
        N, D = _scaled_polynomials(data, [F(c) for c in cert.coeffs])
        if not certify_positive(D, -1, 1):
            return False
        S = _x_intervals(data, cert.S)
        C = _x_intervals(data, _complement(data, cert.S))
        if cert.side == "upper":
            valid = _positive_or_zero(N - D, S) and _positive_or_zero(N, C)
            value = data.upper(cert.coeffs)
        else:
            valid = _positive_or_zero(D - N, S) and _positive_or_zero(-N, C)
            value = data.lower(cert.coeffs)
        return bool(valid and F(cert.value) == value)
    except (ValueError, TypeError, ArithmeticError, IndexError, KeyError, AttributeError):
        return False


def _grid(intervals, density):
    pts = []
    for a, b in intervals:
        n = max(12, int(density * float(b - a) / 2))
        pts.extend(float(a) + float(b - a) * (1 - np.cos(np.pi * np.arange(n + 1) / n)) / 2)
    return np.array(pts)


def _critical_points(N, D):
    P = N.derivative() * D - N * D.derivative()
    if P.degree() <= 0:
        return np.zeros(0)
    # Near the left support endpoint, many partial fractions almost cancel.
    # Binary64 polynomial roots can miss the resulting narrow extrema entirely.
    # Arb isolates the exact rational derivative numerator before we take float
    # midpoints as LP proposal points; soundness still comes from Sturm checks.
    try:
        with ctx.workprec(200):
            roots = P.complex_roots()
            return np.array([float(z.real.mid()) for z, _ in roots
                             if z.imag.contains(0) and -1 <= float(z.real.mid()) <= 1])
    except (ValueError, ArithmeticError, RuntimeError):
        c = np.array([float(v) for v in mono_to_cheb(P)])
        scale = np.max(np.abs(c))
        if not np.isfinite(scale) or scale == 0:
            return np.zeros(0)
        roots = np.polynomial.chebyshev.chebroots(c / scale)
        return np.array([float(r.real) for r in roots
                         if abs(r.imag) < 1e-7 and -1 <= r.real <= 1])


def _raw_candidate(data, S, side, density, exchange_rounds):
    lo, hi = data.I
    center, radius = (lo + hi) / 2, (hi - lo) / 2
    shifts = np.array([float(s + center) / float(radius) for s in data.shifts])
    # Scaled measurement columns equal one at x=0. Coefficients are converted
    # back to the raw acquisition basis before rationalization or evaluation.
    scales = np.array([1.] + [float(1 / (center + s)) for s in data.shifts])
    def phi(x):
        x = np.asarray(x)
        return np.column_stack([np.ones(len(x))] +
                               [t / (x + t) for t in shifts])
    XS, XC = _x_intervals(data, S), _x_intervals(data, _complement(data, S))
    gS, gC = _grid(XS, density), _grid(XC, density)
    y = np.array([float(v) for v in data.values]) / scales
    e = np.array([float(v) for v in data.budgets]) / scales
    nf = len(y)
    cost = np.r_[y if side == "upper" else -y, e]
    sign = -1 if side == "upper" else 1
    last = None
    for _ in range(max(1, exchange_rounds)):
        VS, VC = phi(gS), phi(gC)
        rows = [np.c_[sign * VS, np.zeros((len(gS), nf))],
                np.c_[sign * VC, np.zeros((len(gC), nf))],
                np.c_[np.eye(nf), -np.eye(nf)],
                np.c_[-np.eye(nf), -np.eye(nf)]]
        rhs = np.r_[sign * np.ones(len(gS)), np.zeros(len(gC) + 2 * nf)]
        for method, presolve in (("highs", True), ("highs-ds", False), ("highs-ipm", False)):
            res = linprog(cost, A_ub=np.vstack(rows), b_ub=rhs,
                          bounds=[(-1e7, 1e7)] * nf + [(0, None)] * nf,
                          method=method, options={"primal_feasibility_tolerance": 1e-9,
                                                  "dual_feasibility_tolerance": 1e-9,
                                                  "presolve": presolve})
            if res.success:
                break
        if not res.success:
            if last is None:
                raise RuntimeError("resolvent dual LP failed: " + res.message)
            break
        last = res
        coeffs = [F(float(v)).limit_denominator(10 ** 12) for v in res.x[:nf] / scales]
        N, D = _scaled_polynomials(data, coeffs)
        critical = _critical_points(N, D)
        added = False
        for regions, g, target in ((XS, gS, 1.), (XC, gC, 0.)):
            pts = np.array([x for x in critical if any(float(a) <= x <= float(b) for a, b in regions)])
            if not len(pts):
                continue
            v = phi(pts) @ res.x[:nf]
            bad = pts[sign * (v - target) > 1e-10]
            if len(bad):
                if target:
                    gS = np.r_[gS, bad]
                else:
                    gC = np.r_[gC, bad]
                added = True
        if not added:
            break
    coeffs = [F(float(v)).limit_denominator(10 ** 12) for v in last.x[:nf] / scales]
    return coeffs, float(last.fun if side == "upper" else -last.fun)


def certify_resolvent_candidate(data, S, coeffs, side="upper", lp_value=float("nan")):
    """Repair ANY raw rational proposal by a verified constant shift.

    This public entry point also permits testing a deliberately inaccurate LP
    proposal. Every returned object passes the standalone exact verifier.
    """
    if side not in ("upper", "lower"):
        raise ValueError("side must be upper or lower")
    coeffs = [F(c) for c in coeffs]
    if len(coeffs) != len(data.values):
        raise ValueError("wrong number of raw response coefficients")
    S = _target(data, S)
    N, D = _scaled_polynomials(data, coeffs)
    XS, XC = _x_intervals(data, S), _x_intervals(data, _complement(data, S))
    critical = _critical_points(N, D)
    # A floating minimum merely proposes a repair. The subsequent exact Sturm
    # decisions, with exponentially enlarged margin if needed, establish it.
    worst = 0.
    value_scale = 1.
    for regions, target in ((XS, 1.), (XC, 0.)):
        pts = list(_grid(regions, 50)) + [x for x in critical
                    if any(float(a) <= x <= float(b) for a, b in regions)]
        for x in pts:
            xq = to_fmpq(F(float(x)))
            v = float(N(xq) / D(xq))
            value_scale = max(value_scale, abs(v))
            worst = max(worst, target - v if side == "upper" else v - target)
    # Preserve exact already-valid constant or strictly feasible proposals.
    value = data.upper(coeffs) if side == "upper" else data.lower(coeffs)
    initial = ResolventCertificate(side, value, coeffs, F(0), S, lp_value)
    if verify_resolvent_certificate(data, initial):
        return initial
    margin = 1e-13 * value_scale
    for _ in range(100):
        shift = F(max(0., worst) + margin).limit_denominator(10 ** 15)
        if shift <= 0:
            shift = F(1, 10 ** 15)
        fixed = coeffs.copy()
        fixed[0] += shift if side == "upper" else -shift
        value = data.upper(fixed) if side == "upper" else data.lower(fixed)
        cert = ResolventCertificate(side, value, fixed, shift, S, lp_value)
        if verify_resolvent_certificate(data, cert):
            return cert
        margin *= 4
    raise RuntimeError("could not certify a resolvent constant shift")


def resolvent_mass_certificate(data, S, side="upper", density=500, exchange_rounds=25):
    """Certify target spectral mass directly from raw resolvent samples."""
    if side not in ("upper", "lower"):
        raise ValueError("side must be upper or lower")
    S = _target(data, S)
    # Empty/full sets have exact, sharp certificates with no optimization.
    if not S:
        coeffs = [F(0)] * len(data.values)
    elif S == [data.I]:
        coeffs = [F(1)] + [F(0)] * len(data.shifts)
        if side == "lower" and data.mass < data.mass_eps:
            coeffs[0] = F(0)
    else:
        coeffs, lpv = _raw_candidate(data, S, side, density, exchange_rounds)
        cert = certify_resolvent_candidate(data, S, coeffs, side, lpv)
        # Zero is always a stronger lower certificate than a negative result.
        if side == "upper" and cert.value > data.mass + data.mass_eps:
            coeffs = [F(1)] + [F(0)] * len(data.shifts)
        elif side == "lower" and cert.value < 0:
            coeffs = [F(0)] * len(data.values)
        else:
            return cert
    return certify_resolvent_candidate(data, S, coeffs, side)


@dataclass
class ResolventWitness:
    side: str
    value: F
    atoms: list[tuple[F, F]]
    S: list[Interval]

    def as_dict(self):
        return dict(side=self.side, value=str(self.value), value_float=float(self.value),
                    atoms=[[str(x), str(w)] for x, w in self.atoms],
                    S=[[str(a), str(b)] for a, b in self.S])

    @classmethod
    def from_dict(cls, d):
        return cls(d["side"], F(d["value"]), [(F(x), F(w)) for x, w in d["atoms"]],
                   [(F(a), F(b)) for a, b in d["S"]])


def _inside(x, S):
    return any(a <= x <= b for a, b in S)


def verify_resolvent_witness(data, witness):
    try:
        data = ResolventData.from_dict(data.as_dict())
        wit = (ResolventWitness.from_dict(witness) if isinstance(witness, dict)
               else ResolventWitness(witness.side, F(witness.value),
                    [(F(x), F(w)) for x, w in witness.atoms],
                    [(F(a), F(b)) for a, b in witness.S]))
        if wit.side not in ("upper", "lower") or _target(data, wit.S) != wit.S:
            return False
        if any(not data.I[0] <= x <= data.I[1] or w < 0 for x, w in wit.atoms):
            return False
        moments = [sum((w * data.phi(x)[j] for x, w in wit.atoms), F(0))
                   for j in range(len(data.values))]
        if any(abs(m - y) > e for m, y, e in zip(moments, data.values, data.budgets)):
            return False
        return F(wit.value) == sum((w for x, w in wit.atoms if _inside(x, wit.S)), F(0))
    except (ValueError, TypeError, ArithmeticError, IndexError, KeyError, AttributeError):
        return False


def _correct_equalities(weights, Phi, data):
    """Rational correction on pivot weights; free corrections are set to zero."""
    eq = [j for j, e in enumerate(data.budgets) if e == 0]
    if not eq:
        return weights
    A = [[Phi[k][j] for k in range(len(weights))] +
         [data.values[j] - sum((w * Phi[k][j] for k, w in enumerate(weights)), F(0))]
         for j in eq]
    pivots, row = [], 0
    for col in range(len(weights)):
        p = next((i for i in range(row, len(A)) if A[i][col]), None)
        if p is None:
            continue
        A[row], A[p] = A[p], A[row]
        v = A[row][col]
        A[row] = [x / v for x in A[row]]
        for i in range(len(A)):
            if i != row and A[i][col]:
                v = A[i][col]
                A[i] = [a - v * b for a, b in zip(A[i], A[row])]
        pivots.append(col)
        row += 1
        if row == len(A):
            break
    if any(all(v == 0 for v in r[:-1]) and r[-1] != 0 for r in A):
        return None
    out = weights.copy()
    for i, col in enumerate(pivots):
        out[col] += A[i][-1]
    return out


def resolvent_primal_witness(data, S, side="upper", density=350,
                            extra_points=()) -> Optional[ResolventWitness]:
    """Find a compatible rational atomic law; return None if exact repair fails.

    For upper certificates, its target mass is a lower bound on the sharp
    supremum. For lower certificates, it is an upper bound on the sharp infimum.
    No claim of universal attainment or optimality is made.
    """
    if side not in ("upper", "lower"):
        raise ValueError("side must be upper or lower")
    S = _target(data, S)
    lo, hi = data.I
    m = max(24, int(density))
    points = {lo + (hi - lo) * F(j, m) for j in range(m + 1)}
    points.update(F(x) for x in extra_points if lo <= F(x) <= hi)
    for a, b in S:
        points.update((a, b))
        for edge in (a, b):
            for direction in (-1, 1):
                x = edge + direction * (hi - lo) / (m * 100)
                if lo <= x <= hi:
                    points.add(x)
    xs = sorted(points)
    Phi = [data.phi(x) for x in xs]
    V = np.array([[float(v) for v in row] for row in Phi]).T
    scales = np.array([max(abs(float(y)), float(e), 1e-12)
                       for y, e in zip(data.values, data.budgets)])
    V = V / scales[:, None]
    y = np.array([float(v) for v in data.values]) / scales
    e = np.array([float(v) for v in data.budgets]) / scales
    objective = np.array([float(_inside(x, S)) for x in xs])
    if side == "upper":
        objective *= -1
    exact_rows = [j for j, v in enumerate(data.budgets) if v == 0]
    noisy_rows = [j for j, v in enumerate(data.budgets) if v != 0]
    for tighten in (0.001, 0.02, 0.1, 0.35):
        Aeq, beq = (V[exact_rows], y[exact_rows]) if exact_rows else (None, None)
        Aub = np.vstack([V[noisy_rows], -V[noisy_rows]]) if noisy_rows else None
        bub = (np.r_[y[noisy_rows] + (1 - tighten) * e[noisy_rows],
                       -y[noisy_rows] + (1 - tighten) * e[noisy_rows]] if noisy_rows else None)
        res = linprog(objective, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                      bounds=[(0, None)] * len(xs), method="highs",
                      options={"primal_feasibility_tolerance": 1e-9,
                               "dual_feasibility_tolerance": 1e-9})
        if not res.success:
            continue
        support = [j for j, w in enumerate(res.x) if w > 0]
        if not support:
            candidate = ResolventWitness(side, F(0), [], S)
        else:
            weights = [F(float(res.x[j])).limit_denominator(10 ** 15) for j in support]
            weights = _correct_equalities(weights, [Phi[j] for j in support], data)
            if weights is None:
                continue
            atoms = [(xs[j], w) for j, w in zip(support, weights) if w]
            value = sum((w for x, w in atoms if _inside(x, S)), F(0))
            candidate = ResolventWitness(side, value, atoms, S)
        if verify_resolvent_witness(data, candidate):
            return candidate
    return None


# Convenient module-local names, matching the older moment API.
mass_certificate = resolvent_mass_certificate
primal_witness = resolvent_primal_witness
