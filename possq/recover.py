"""Recovery layer: synthetic exact instances, Prony estimation, cluster enclosures,
node-presence radii, and measurement sensitivity."""
from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction as F
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .bounds import (MomentData, mass_certificate, mass_lp_value)
from .exact import Basis, Interval, cheb_T, from_roots, length, merge, mono_coeffs, clip


# ----------------------------------------------------------------------------
# exact synthetic instances
# ----------------------------------------------------------------------------
@dataclass
class TrueMeasure:
    atoms: List[Tuple[F, F]]          # (node, weight)
    background: F = F(0)              # mass of uniform density on I
    I: Interval = (F(-1), F(1))

    def moments(self, basis: Basis) -> List[F]:
        m = [F(0)] * (basis.K + 1)
        for x, w in self.atoms:
            for k, v in enumerate(basis.exact_moment_atom(x)):
                m[k] += w * v
        if self.background:
            dens = self.background / (self.I[1] - self.I[0])
            for k, v in enumerate(basis.exact_moment_uniform(*self.I)):
                m[k] += dens * v
        return m

    def mass(self, S: Sequence[Interval], S_open: bool) -> F:
        tot = F(0)
        for x, w in self.atoms:
            for a, b in S:
                if (a < x < b) if S_open else (a <= x <= b):
                    tot += w
                    break
        if self.background:
            dens = self.background / (self.I[1] - self.I[0])
            tot += dens * length(clip(self.I, S))
        return tot


def noisy_data(mu: TrueMeasure, basis: Basis, eps: F, seed: int = 7) -> MomentData:
    rng = random.Random(seed)
    m = mu.moments(basis)
    u = [F(rng.randint(-10 ** 6, 10 ** 6), 10 ** 6) for _ in m]
    return MomentData(basis, [mk + eps * uk for mk, uk in zip(m, u)], [eps] * len(m), mu.I)


# ----------------------------------------------------------------------------
# Prony / matrix pencil on monomial moments derived exactly from the basis
# ----------------------------------------------------------------------------
def monomial_moments_float(data: MomentData) -> np.ndarray:
    if data.basis.name == "monomial":
        return np.array([float(v) for v in data.mu_hat])
    from .exact import mono_to_cheb, fmpq_poly
    out = []
    for j in range(data.basis.K + 1):
        c = mono_to_cheb(fmpq_poly([0] * j + [1]))
        out.append(float(sum(ck * data.mu_hat[k] for k, ck in enumerate(c))))
    return np.array(out)


def hankel_spectrum(data: MomentData) -> np.ndarray:
    m = monomial_moments_float(data)
    h = data.basis.K // 2
    H = np.array([[m[i + j] for j in range(h + 1)] for i in range(h + 1)])
    return np.linalg.svd(H, compute_uv=False)


def prony(data: MomentData, n: int) -> Tuple[List[float], List[float]]:
    m = monomial_moments_float(data)
    H0 = np.array([[m[i + j] for j in range(n)] for i in range(n)])
    H1 = np.array([[m[i + j + 1] for j in range(n)] for i in range(n)])
    ev = np.linalg.eigvals(np.linalg.solve(H0, H1))
    nodes = sorted(float(z.real) for z in ev)
    V = np.vander(np.array(nodes), len(m), increasing=True).T
    w, *_ = np.linalg.lstsq(V, m, rcond=None)
    return nodes, list(map(float, w))


def rationalize(xs: Sequence[float], den: int = 10 ** 12) -> List[F]:
    return [F(x).limit_denominator(den) for x in xs]


# ----------------------------------------------------------------------------
# cluster enclosures, presence radii, sensitivity
# ----------------------------------------------------------------------------
def cluster_enclosure(data: MomentData, center: F, r: F, degree=None):
    S = [(center - r, center + r)]
    lo = mass_certificate(data, S, "lower", degree)
    up = mass_certificate(data, S, "upper", degree)
    return lo, up


def presence_radius(data: MomentData, center: F, rmax: float = 0.5, iters: int = 14,
                    degree=None, density: int = 300, rmin: float = 1e-10) -> Optional[Tuple[F, object]]:
    """Smallest r (geometric bisection on [rmin, rmax]) with a certified mu((c-r,c+r)) > 0."""
    c = float(center)
    f = lambda r: mass_lp_value(data, [(F(c - r), F(c + r))], "lower", degree, density)
    if f(rmax) <= 0:
        return None
    lo, hi = rmin, rmax
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    for fac in (1.0, 1.02, 1.1, 1.3, 2.0):
        r = F(hi * fac).limit_denominator(10 ** 12)
        cert = mass_certificate(data, [(center - r, center + r)], "lower", degree)
        if cert.value > 0:
            return r, cert
    return None


def sensitivity_ranking(cert) -> List[Tuple[int, float]]:
    return sorted(((k, float(abs(c))) for k, c in enumerate(cert.coeffs)),
                  key=lambda t: -t[1])


def with_eps(data: MomentData, new_eps: Sequence[F]) -> MomentData:
    return MomentData(data.basis, list(data.mu_hat), list(new_eps), data.I)
