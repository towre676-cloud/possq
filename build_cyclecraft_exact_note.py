"""Generate cyclecraft_exact.html (house style) from results/exact_*.json and results/exact_crosscheck.json."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from possq.cyclecraft import ball, load

ROOT = Path(__file__).parent
RES = ROOT / "results"
EX = {}
for f in ("exact_cycle_aux.json", "exact_star_shortcut.json"):
    EX.update(json.loads((RES / f).read_text()))
XC = json.loads((RES / "exact_crosscheck.json").read_text())
R = load(RES / "cyclecraft_precision_cert_results.json")
MOM = {}
for g in ("cycle", "star"):
    L = json.loads((RES / f"cyclecraft_ledger_{g}.json").read_text())
    MOM[g] = {c["copies"]: c for c in L["cases"]}

OX, INK, MUTE, GOLD, CREAM = "#6b1d1d", "#2a2320", "#8a7d72", "#a8803a", "#f6f0e4"
plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "figure.facecolor": CREAM,
                     "axes.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.edgecolor": INK})


def pub(g, key):
    c = R["certificates"][g]
    if key == "single":
        return float(ball(c["target_lower_tail"]["certified_probability_upper"])[1])
    if key == "agg":
        return float(ball(c["target_lower_tail"]["independent_copies_10_aggregate_wrong_direction_upper"])[1])
    if key == "aux":
        return float(ball(c["auxiliary_terminal_tail"]["certified_probability_upper"])[1])


cyc, sta = EX["cycle"], EX["star"]
rows = [
    ("cycle", r"$\Pr(N_{10}\le0)$", pub("cycle", "single"), cyc["P_N10_le_0"]),
    ("star", r"$\Pr(N_{10}\le0)$", pub("star", "single"), sta["P_N10_le_0"]),
    ("cycle", r"10 copies, $\Pr(\sum\le0)$", pub("cycle", "agg"), cyc["aggregate_P_le_0"]["10"]),
    ("star", r"10 copies, $\Pr(\sum\le0)$", pub("star", "agg"), sta["aggregate_P_le_0"]["10"]),
    ("cycle", r"auxiliary $\Pr(|A_{10}|\ge12)$", pub("cycle", "aux"), EX["cycle_aux"]["P_abs_ge_12"]),
]


def e(x, d=3):
    if x is None:
        return "&mdash;"
    if x == 0:
        return "$0$"
    m, ex = f"{x:.{d}e}".split("e")
    ex = int(ex)
    return f"${m}$" if ex == 0 else f"${m}\\times10^{{{ex}}}$"


def ballstr(b, digits=22):
    s = b["lower"]
    core = s[1:s.index(" +/-")] if s.startswith("[") else s
    return core[:digits + 2]


# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(7.6, 3.7))
labels = ["cycle single", "star single", "cycle 10 copies", "star 10 copies", "cycle auxiliary"]
y = np.arange(len(rows))
ax.barh(y + .2, [r[2] for r in rows], height=.38, color=MUTE, label="Cyclecraft certified bound")
ax.barh(y - .2, [r[3]["upper_float"] for r in rows], height=.38, color=OX, label="exact (this note)")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
for i, r in enumerate(rows):
    ax.text(r[3]["upper_float"] * 1.3, i - .2, f"×{r[2]/r[3]['upper_float']:,.0f}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(1e-17, 0.6)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
ax.grid(alpha=.25, axis="x", which="both")
ax.set_xlabel("probability (log scale)")
fig.tight_layout(); fig.savefig(RES / "fig_exact_vs_published.svg"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
pm = {int(k): v for k, v in cyc["pmf_n10"].items()}
ks = np.array(sorted(k for k in pm if -15 <= k <= 25))
vals = np.array([pm[k][1] for k in ks])
axs[0].bar(ks, vals, color=[OX if k <= 0 else GOLD for k in ks], width=.8)
axs[0].set_yscale("log"); axs[0].set_ylim(1e-24, 1)
axs[0].set_xlabel(r"$N_{10}$ (cycle)"); axs[0].set_ylabel("certified probability")
axs[0].set_title("certified law; red: wrong direction", fontsize=9)
for g, col, mk in (("cycle", OX, "o"), ("star", GOLD, "s")):
    H = EX[g]["horizons"]
    ns = sorted(int(n) for n in H)
    axs[1].semilogy(ns, [H[str(n)]["P_le_0"]["upper_float"] for n in ns], mk, color=col, label=g + " exact")
    cw = EX[g]["cw_best"]
    nn = np.arange(1, 101)
    axs[1].semilogy(nn, cw["C"] * cw["rho_float"] ** nn, "-", color=col, lw=1, alpha=.7,
                    label=g + r" bound, all $n$")
axs[1].set_xlabel("periods $n$"); axs[1].set_ylabel(r"$P(N_n \leq 0)$")
axs[1].legend(frameon=False, fontsize=7.5); axs[1].grid(alpha=.25, which="both")
fig.tight_layout(); fig.savefig(RES / "fig_exact_law_horizon.svg"); plt.close(fig)


def svg(name):
    s = (RES / name).read_text()
    return s[s.index("<svg"):].replace("<svg ", '<svg class="fig" ', 1)


# ---------------- tables ----------------
t_main = "".join(
    f"<tr><td>{g}</td><td>{lab}</td><td>{e(p,4)}</td><td>{e(x['lower_float'],10)}</td>"
    f"<td>{e(x['width'],1)}</td><td>${p/x['upper_float']:,.0f}$</td></tr>"
    for g, lab, p, x in rows)

t_hor = ""
for n in (10, 20, 40, 80):
    c = cyc["horizons"][str(n)]["P_le_0"]; s = sta["horizons"][str(n)]["P_le_0"]
    bc = cyc["cw_best"]["C"] * cyc["cw_best"]["rho_float"] ** n
    bs = sta["cw_best"]["C"] * sta["cw_best"]["rho_float"] ** n
    t_hor += (f"<tr><td>${n}$</td><td>{e(c['lower_float'],8)}</td><td>{e(bc,3)}</td>"
              f"<td>{e(s['lower_float'],8)}</td><td>{e(bs,3)}</td></tr>")
cw_ok = all(EX[g]["cw_best"]["C"] * EX[g]["cw_best"]["rho_float"] ** n >= EX[g]["horizons"][str(n)]["P_le_0"]["upper_float"]
            for g in ("cycle", "star") for n in (10, 20, 40, 80))

t_agg = ""
for K in range(1, 11):
    c = cyc["aggregate_P_le_0"][str(K)]["lower_float"]
    s = sta["aggregate_P_le_0"][str(K)]["lower_float"]
    zc = float(ball(R["certificates"]["cycle"]["target_lower_tail"]["certified_probability_upper"])[1]) ** K
    t_agg += f"<tr><td>${K}$</td><td>{e(c,6)}</td><td>{e(s,6)}</td><td>{e(zc,3)}</td><td>${zc/c:,.0f}$</td></tr>"

t_tail = "".join(f"<tr><td>$\\Pr(N_{{10}}\\le{t})$</td><td>{e(cyc['P_N10_le_t'][t]['lower_float'],8)}</td>"
                 f"<td>{e(sta['P_N10_le_t'][t]['lower_float'],8)}</td></tr>" for t in ("-5", "-2", "-1", "0", "1", "2"))

aux = EX["cycle_aux"]["P_abs_ge_t"]
t_aux = "".join(f"<tr><td>$\\Pr(|A_{{10}}|\\ge{t})$</td><td>{e(aux[t]['lower_float'],6)}</td></tr>" for t in ("4", "8", "12", "16", "20"))

t_sc = ""
for row, th in zip(EX["shortcut"], R["rare_shortcuts"]):
    aj = float(ball(th["any_shortcut_jump_probability"])[0])
    t_sc += (f"<tr><td>${row['eps']}$</td><td>{e(aj,4)}</td><td>{e(row['P_net_nonzero']['lower_float'],4)}</td>"
             f"<td>{e(row['P_abs_ge_2']['lower_float'],4)}</td><td>{e(row['P_abs_ge_3']['lower_float'],4)}</td>"
             f"<td>${aj/row['P_abs_ge_2']['upper_float']:,.0f}$</td></tr>")

mom_c = MOM["cycle"][1]
_agg = [cyc["aggregate_P_le_0"][str(K)]["lower_float"] for K in range(1, 11)]
_ratios = [_agg[i] / _agg[i + 1] for i in range(9)]
K_exact = next(K for K in range(1, 11) if cyc["aggregate_P_le_0"][str(K)]["upper_float"] < 1e-12)
_pc = pub("cycle", "single")
K_pub = next(K for K in range(1, 40) if _pc ** K < 1e-12)
gap80 = cyc["cw_best"]["C"] * cyc["cw_best"]["rho_float"] ** 80 / cyc["horizons"]["80"]["P_le_0"]["upper_float"]
cc = cyc["P_N10_le_0"]
cw = cyc["cw_best"]
sw = sta["cw_best"]

html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cyclecraft Exact Counting Laws</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
<style>
:root{--bg:#f6f0e4;--ink:#2a2320;--ox:#6b1d1d;--mute:#8a7d72;--rule:#d9cdb8;--panel:#efe6d4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1d1916;--ink:#ece3d3;--ox:#d07a6a;--mute:#a89a8b;--rule:#3d342c;--panel:#26201b}}
:root[data-theme="dark"]{--bg:#1d1916;--ink:#ece3d3;--ox:#d07a6a;--mute:#a89a8b;--rule:#3d342c;--panel:#26201b}
html{background:var(--bg)}body{background:var(--bg);color:var(--ink);font-family:'EB Garamond',Garamond,serif;font-size:18px;line-height:1.62;margin:0}
main{max-width:46rem;margin:0 auto;padding:3rem 16px 5rem}
h1,h2{font-family:'Cormorant Garamond',serif;color:var(--ox);font-weight:600;line-height:1.15}
h1{font-size:2.4rem;margin:.2em 0}h2{font-size:1.6rem;margin-top:2.5rem;border-bottom:1px solid var(--rule);padding-bottom:.2rem}
.sub{font-style:italic;color:var(--mute)}
.lab{font-variant:small-caps;letter-spacing:.06em;font-size:.82em;color:var(--ox);border:1px solid var(--ox);padding:0 .35em;border-radius:2px;margin-right:.3em;white-space:nowrap}
.thm{background:var(--panel);border-left:3px solid var(--ox);padding:.7rem 1rem;margin:1.2rem 0}
.thm b{font-family:'Cormorant Garamond',serif;font-size:1.08em}
.pf{font-size:.97em}.pf:after{content:" \220E";color:var(--ox)}
table{border-collapse:collapse;width:100%;font-size:.84em;margin:1rem 0;display:block;overflow-x:auto;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--rule);padding:.3rem .45rem;text-align:right;white-space:nowrap}th{color:var(--ox);font-weight:600}
td:first-child,th:first-child{text-align:left}
.fig{width:100%;height:auto;margin:1rem 0;border:1px solid var(--rule)}
.cap{font-size:.88em;color:var(--mute);margin-top:-.4rem}
code{font-size:.85em;background:var(--panel);padding:0 .25em}
</style></head><body><main>
<p class="sub">Companion to Cyclecraft II &middot; possq 0.3.0 &middot; 31 October 2026</p>
<h1>Cyclecraft Exact Counting Laws</h1>
<p class="sub">Every wrong-direction and winding risk in Cyclecraft II&rsquo;s held-protocol certificates, replaced by its exact value with a rigorous error below $10^{-50}$</p>

<h2>0. Labels and sources</h2>
<p><span class="lab">proved</span> marks statements proved here. <span class="lab">certified</span> marks values computed in Arb ball arithmetic at 200 bits from the exact rational rates of Cyclecraft&rsquo;s protocol, with Cyclecraft&rsquo;s own periodic-law certificate as the initial condition. <span class="lab">measured</span> marks binary64 observations. All numbers come from <code>results/exact_cycle_aux.json</code>, <code>results/exact_star_shortcut.json</code> and <code>results/exact_crosscheck.json</code>.</p>

<h2>1. Why summaries were the bottleneck</h2>
<p>The previous note showed that seven certified summaries of the target count $N_{10}$ (mean, variance and four tilted generating functions) can never certify $\Pr(N_{10}\le0)$ below $""" + f"{MOM['cycle'][1]['sharp_upper_float']:.6f}" + r"""$ on the cycle network, and that nineteen summaries reach $""" + f"{mom_c['all_16_tilts']['sharp_upper']:.5f}" + r"""$. Those bounds are sharp for what they are given. The limit is the input, not the method: a finite list of moments cannot pin down a distribution. Cyclecraft computes its tilted generating functions by propagating an exact rational generator through Arb matrix exponentials. The same machinery, applied to a lift that records the count itself, yields the whole distribution of $N_{10}$ at the same cost, and the whole distribution answers every tail question exactly.</p>

<h2>2. Method</h2>
<p>For a counted edge $e$ and a window $\{\ell,\dots,h\}$ of count levels, the lifted chain has states (level, site). Each jump moves the site as in Cyclecraft; a forward or backward jump across $e$ also moves the level by $+1$ or $-1$. A jump that would leave the window is <i>killed</i>: its rate still drains the source state, but the mass goes nowhere. Each of the three held stages has an exact rational generator, so its propagator $e^{2K_s}$ is enclosed by an Arb matrix exponential and one period is the product of three such matrices.</p>
<div class="thm"><span class="lab">proved</span><b>Lemma K (killed bracket).</b> Let $\kappa(E)$ be the killed-chain probability that the count ends in a set $E$ of levels inside the window, and $\lambda=1-\kappa(\text{window})$ the killed mass. Then $\kappa(E)\le\Pr(N\in E)\le\kappa(E)+\lambda$.</div>
<p class="pf">A path that never leaves the window carries the same weight in both chains, and the killed chain represents exactly those paths. Every other path has total probability $\lambda$ and may or may not end in $E$.</p>
<div class="thm"><span class="lab">proved</span><b>Lemma D (doubling).</b> Let $A^{(n)}_k[i,j]=\Pr(N_n=k,X_n=i\mid X_0=j)$ for the counted chain started at level $0$. Then $A^{(2n)}_m=\sum_kA^{(n)}_{m-k}A^{(n)}_k$. If each factor is replaced by a killed, truncated kernel and the product is truncated again, the result stays entrywise below the true kernel, and one minus its column sums bounds the unrepresented mass from each starting site.</div>
<p class="pf">The identity is the Markov property at time $n$ together with translation invariance of the count increment. Truncation drops nonnegative terms, so entrywise domination survives; the true columns sum to $1$, so the deficit of the represented columns is exactly the total unrepresented mass. Lemma K then applies to every event.</p>
<div class="thm"><span class="lab">proved</span><b>Lemma P (initial law).</b> If the true periodic law $p'$ satisfies $\|p'-p\|_1\le\delta$, then every event probability computed from $p$ differs from the true one by at most $\delta$.</div>
<p class="pf">The joint kernel restricted to any event is entrywise nonnegative with column sums at most $1$.</p>
<p>Cyclecraft&rsquo;s Doeblin certificate gives $\delta\le""" + f"{cyc['delta']:.2e}".replace("e-", "\\times10^{-") + r"""}$ on the cycle network, reproduced here bit for bit. Independent copies are combined by convolving the enclosed probability mass functions, adding the killed masses. All quantities are Arb balls throughout.</p>
<div class="thm"><span class="lab">proved</span><b>Proposition H (every horizon at once).</b> Let $M(z)$ be the tilted monodromy for $0<z<1$, which is entrywise positive, and let $v>0$ satisfy $M(z)v\le\rho v$ entrywise. With $c=\max_j(p_j+\delta)/v_j$, every horizon $n\ge0$ satisfies $\Pr(N_n\le0)\le\mathbb Ez^{N_n}\le c\,\rho^{\,n}\sum_iv_i$.</div>
<p class="pf">The true initial law is at most $c\,v$ entrywise, $M(z)^n$ preserves entrywise order, and $M(z)^nv\le\rho^nv$. The first inequality is Chernoff&rsquo;s, since $z^{N}\ge1$ on $\{N\le0\}$.</p>

<h2>3. Results against the published certificates</h2>
<p><span class="lab">certified</span> Each row replaces a Cyclecraft upper bound by the exact probability. The width column is the certified uncertainty of the exact value.</p>
<table><tr><th>network</th><th>event</th><th>Cyclecraft bound</th><th>exact value</th><th>width</th><th>factor</th></tr>""" + t_main + r"""</table>
""" + svg("fig_exact_vs_published.svg") + r"""
<p class="cap">Figure 1. Published certified upper bounds (grey) and exact certified values (red), log scale; labels give the ratio.</p>
<p>To twenty digits, $\Pr(N_{10}\le0)=""" + ballstr(cc, 20) + r"""\ldots$ on the cycle network and $""" + ballstr(sta["P_N10_le_0"], 20) + r"""\ldots$ on the star network. The single-count bounds improve by a factor of about $8.5$. The aggregate and auxiliary bounds improve by four to five orders of magnitude, because Chernoff&rsquo;s inequality loses most where the event lies deepest in the tail.</p>

<h2>4. The whole law</h2>
""" + svg("fig_exact_law_horizon.svg") + r"""
<p class="cap">Figure 2. Left: the certified probability mass function of $N_{10}$ on the cycle network, log scale; widths are invisible at this scale. Right: exact $\Pr(N_n\le0)$ at $n=10,20,40,80$ (markers) and the all-horizon bound of Proposition H (lines).</p>
<table><tr><th>event</th><th>cycle</th><th>star</th></tr>""" + t_tail + r"""</table>
<p>The law carries every tail question at once. A machine that ends at most $2$ crossings ahead happens in about $12.5\%$ of runs on the cycle network, while one that ends $5$ or more behind is a $1.8\times10^{-7}$ event.</p>

<h2>5. Longer horizons</h2>
<p><span class="lab">certified</span> Lemma D carries the exact law to $20$, $40$ and $80$ periods by three doublings. Cyclecraft certifies the ten-period horizon only. Proposition H at the grid-optimal tilt $z=""" + cw["z"] + r"""$ gives $\Pr(N_n\le0)\le""" + f"{cw['C']:.4f}" + r"""\times""" + f"{cw['rho_float']:.10f}" + r"""^{\,n}$ for every $n$ on the cycle network and $""" + f"{sw['C']:.4f}" + r"""\times""" + f"{sw['rho_float']:.10f}" + r"""^{\,n}$ on the star network. These guarantee a wrong-direction decay of at least """ + f"${cw['decay_per_period']:.4f}$" + r""" nepers per period. The exact values sit below the bound at every computed horizon (""" + ("verified" if cw_ok else "VIOLATED") + r"""), and the gap grows only polynomially: at $n=80$ the bound is about $""" + f"{gap80:.0f}" + r"""$ times the exact value, the expected price of an exponential bound that ignores the central-limit prefactor.</p>
<table><tr><th>$n$</th><th>cycle exact</th><th>cycle bound</th><th>star exact</th><th>star bound</th></tr>""" + t_hor + r"""</table>

<h2>6. Independent copies</h2>
<p><span class="lab">certified</span> For $K$ independent machines, $\Pr(N^{(1)}+\dots+N^{(K)}\le0)$ is exact for every $K\le10$. The published route raises the single-copy Chernoff bound to the $K$-th power; its loss compounds with every copy.</p>
<table><tr><th>$K$</th><th>cycle exact</th><th>star exact</th><th>cycle, published route</th><th>loss</th></tr>""" + t_agg + r"""</table>
<p>Each added machine divides the cycle-network risk by a factor that falls from """ + f"${max(_ratios):.1f}$ to ${min(_ratios):.1f}$" + r""" as $K$ grows. A specification such as a joint wrong-direction risk below $10^{-12}$ is met at exactly $K=""" + str(K_exact) + r"""$; the published route needs $K=""" + str(K_pub) + r"""$.</p>

<h2>7. Auxiliary edge and rare shortcuts</h2>
<p><span class="lab">certified</span> On the cycle network the auxiliary edge has a signed count $A_{10}$ with mean $0$. Cyclecraft bounds $\Pr(|A_{10}|\ge12)$ by $""" + f"{pub('cycle','aux'):.4e}".replace("e-0", "\\times10^{-") + r"""}$ using a two-sided Chernoff union; the exact value is shown below with its neighbours. On the star the edge is a bridge and Cyclecraft&rsquo;s structural value $0$ is already exact.</p>
<table><tr><th>event</th><th>cycle exact</th></tr>""" + t_aux + r"""</table>
<p>For the leaky star with a rare extra channel between sites $3$ and $4$ of conductance $\varepsilon$, Cyclecraft certifies the probability of <i>any</i> shortcut jump in ten periods. It then bounds every new auxiliary winding by that probability. Counting signed crossings of the shortcut channel itself gives the exact law of the net crossing number $C_{10}$. Since the channel is the only edge of the new loop outside a spanning tree, $C_{10}$ is the winding number of the path closed up through the tree.</p>
<table><tr><th>$\varepsilon$</th><th>any jump (Cyclecraft)</th><th>$\Pr(C\ne0)$</th><th>$\Pr(|C|\ge2)$</th><th>$\Pr(|C|\ge3)$</th><th>factor at $|C|\ge2$</th></tr>""" + t_sc + r"""</table>
<p>For a single net crossing the published bound is nearly tight: almost every shortcut jump leaves a net crossing, because the walker rarely jumps back. The risk of a double winding, however, is smaller by a factor of $185$ at $\varepsilon=\tfrac1{1024}$, and its exact value scales roughly as $\varepsilon^2$, against $\varepsilon$ for any jump.</p>

<h2>8. Cross-checks</h2>
<p><span class="lab">measured</span> Every summary Cyclecraft certified independently is recovered from the exact laws. These are the four tilted generating functions and the mean and variance of $N_{10}$ on both networks, and the auxiliary variance. The largest relative discrepancy is """ + f"${XC['max_rel_diff']:.1e}$".replace("e-", "\\times10^{-") + r"""}$, which is binary64 rounding in the comparison itself. The probability of a net shortcut crossing never exceeds Cyclecraft&rsquo;s any-jump probability (""" + ("consistent at all three conductances" if XC["shortcut_consistent"] else "INCONSISTENT") + r"""). The binary64 oracle law of the previous note agrees with the exact $\Pr(N_{10}\le0)$ to $3\times10^{-15}$ relative.</p>

<h2>9. What changes, and what does not</h2>
<p>For the dyadic three-hold protocols, every finite-horizon risk is now a number rather than a bound. The certificate gives the value to fifty or more digits, all horizons are bounded from above by Proposition H, and copies are combined exactly. The moment-problem certificates of the previous note remain the right tool when only summaries are available, for example from an experiment that measures generating functions but not full laws.</p>
<p><span class="lab">open</span> Three gaps remain. Cyclecraft&rsquo;s optimised continuous waveforms are not piecewise constant, so their propagators are not single matrix exponentials; extending the exact lift to them needs a certified time-ordered exponential, by validated Magnus or Taylor-model integration. The decay rate in Proposition H is certified from above only, and a matching lower bound needs a certified lower bound on $\min_z\rho(z)$ over the continuum, not just the grid. Finally, the exact laws at long horizons are limited by precision: at 200 bits the certified widths stay below $10^{-50}$ to $n=80$, but certifying probabilities near $10^{-100}$ needs proportionally more bits.</p>

<h2>Reproduction</h2>
<p><code>python -m possq.cyclecraft_exact_ledger results cycle aux</code> and <code>python -m possq.cyclecraft_exact_ledger results star shortcut</code> (about $4$ minutes each), then <code>python -m possq.cyclecraft_crosscheck results</code> and <code>python build_cyclecraft_exact_note.py</code>. The only input is Cyclecraft&rsquo;s unmodified <code>precision_cert_results.json</code>, plus the protocol constants copied verbatim from its <code>precision_cert.py</code>.</p>
</main></body></html>"""
(ROOT / "cyclecraft_exact.html").write_text(html)
print("wrote cyclecraft_exact.html; cw_ok =", cw_ok)
