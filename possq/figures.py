"""House-style SVG figures from the ledger.  python -m possq.figures RESULTS_DIR"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bounds import mass_certificate
from .exact import Basis, complement
from .recover import TrueMeasure, noisy_data

OX, INK, MUTE, GOLD, CREAM = "#6b1d1d", "#2a2320", "#8a7d72", "#a8803a", "#f6f0e4"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM})


def fig_scaling(L, out):
    rows = L["E3_noise_scaling"]["rows"]
    e = [r["eps"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.loglog(e, [r["square_bound"] for r in rows], "o-", color=GOLD, label=r"positive-square bound $\eta/g$")
    ax.loglog(e, [r["sharp_upper"] for r in rows], "s-", color=OX, label="sharp dual certificate")
    ax.loglog(e, [r["witness"] or np.nan for r in rows], "^--", color=INK, label="primal witness (attained)")
    ax.loglog(e, [r["node_err"] for r in rows], ":", color=MUTE, label="Prony node error")
    ax.set_xlabel(r"moment error $\varepsilon$")
    ax.set_ylabel(r"mass outside $U$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.25, which="both")
    fig.tight_layout(); fig.savefig(out / "fig_scaling.svg"); plt.close(fig)


def fig_presence(L, out):
    rows = L["E6_presence_radii"]["rows"]
    e = [r["eps"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cols = [OX, GOLD, INK, MUTE]
    labels = ["$-3/5$", "$-1/10$", "$1/3$", "$4/5$"]
    for i in range(4):
        ax.loglog(e, [r["radii"][i] or np.nan for r in rows], "o-", color=cols[i], label="node " + labels[i])
    ax.set_xlabel(r"moment error $\varepsilon$"); ax.set_ylabel("certified presence radius")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25, which="both")
    fig.tight_layout(); fig.savefig(out / "fig_presence.svg"); plt.close(fig)


def fig_certificates(out):
    atoms = [(F(-3, 5), F(1, 4)), (F(-1, 10), F(3, 10)), (F(1, 3), F(1, 5)), (F(4, 5), F(1, 5))]
    B = Basis("chebyshev", 16)
    mu = TrueMeasure(atoms)
    d = noisy_data(mu, B, F(1, 10 ** 6))
    r = F(1, 20)
    U = [(x - r, x + r) for x, _ in atoms]
    S = complement(d.I, U)
    up = mass_certificate(d, S, "upper")
    lo = mass_certificate(d, [U[1]], "lower")
    xs = np.linspace(-1, 1, 4001)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, cert, title in [(axs[0], up, r"upper certificate: $q\geq 1$ off $U$, $q\geq 0$ on $I$"),
                            (axs[1], lo, r"lower certificate: $q\leq 1$ on $U_2$, $q\leq 0$ off $U_2$")]:
        P = B.poly_of(cert.coeffs)
        c = [float(v) for v in reversed([float(z) for z in P.coeffs()])]
        y = np.polyval(c, xs)
        for a, b in U:
            ax.axvspan(float(a), float(b), color=GOLD, alpha=.18, lw=0)
        ax.plot(xs, y, color=OX, lw=1.3)
        ax.axhline(0, color=INK, lw=.6); ax.axhline(1, color=MUTE, lw=.6, ls=":")
        for x, w in atoms:
            ax.plot([float(x)] * 2, [0, float(w)], color=INK, lw=2)
        ax.set_ylim(-0.4, 1.6); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_certificates.svg"); plt.close(fig)


def main(d):
    d = Path(d)
    L = json.loads((d / "ledger.json").read_text())
    fig_scaling(L, d); fig_presence(L, d); fig_certificates(d)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
