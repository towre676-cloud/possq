"""Generate cyclecraft_tails.html (house style) from results/cyclecraft_ledger_{cycle,star}.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RES = ROOT / "results"
cases = []
repro = {}
for g in ("cycle", "star"):
    L = json.loads((RES / f"cyclecraft_ledger_{g}.json").read_text())
    cases += L["cases"]
    repro.update(L["reproduction"])
C = {(c["graph"], c["copies"]): c for c in cases}
sound = all(c["truth_le_sharp"] and c["all_16_tilts"]["truth_le_sharp"] and
            all(p["truth_le_sharp"] for p in c["greedy_path"]) for c in cases)

OX, INK, MUTE, GOLD, CREAM = "#6b1d1d", "#2a2320", "#8a7d72", "#a8803a", "#f6f0e4"
plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "figure.facecolor": CREAM,
                     "axes.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.edgecolor": INK})
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
for ax, cp in zip(axs, (1, 10)):
    for g, col, mk in (("cycle", OX, "o"), ("star", GOLD, "s")):
        c = C[(g, cp)]
        ys = [c["reported_chernoff"], c["sharp_upper_float"]] + [p["sharp_upper"] for p in c["greedy_path"]]
        xs = list(range(len(ys)))
        ax.semilogy(xs, ys, mk + "-", color=col, label=g + " certified")
        ax.axhline(c["truth_binary64"], color=col, ls=":", lw=1)
    ax.set_xticks(range(8))
    ax.set_xticklabels(["Chernoff", "sharp", "+1", "+2", "+3", "+4", "+5", "+6"], fontsize=8)
    ax.set_title("single count $N_{10}$" if cp == 1 else "aggregate of 10 independent copies", fontsize=9)
    ax.set_ylabel(r"bound on $\Pr(\leq 0)$")
    ax.grid(alpha=.25, which="both")
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(RES / "fig_cyclecraft_tails.svg")
plt.close(fig)
svg = (RES / "fig_cyclecraft_tails.svg").read_text()
svg = svg[svg.index("<svg"):].replace("<svg ", '<svg class="fig" ', 1)


def e(x, d=3):
    if x is None:
        return "&mdash;"
    m, ex = f"{x:.{d}e}".split("e")
    ex = int(ex)
    return f"${m}$" if ex == 0 else f"${m}\\times10^{{{ex}}}$"


main_rows = "".join(
    f"<tr><td>{g}</td><td>${cp}$</td><td>{e(C[(g,cp)]['reported_chernoff'],4)}</td>"
    f"<td>{e(C[(g,cp)]['sharp_upper_float'],6)}</td><td>{e(C[(g,cp)]['witness_float'],6)}</td>"
    f"<td>{e(C[(g,cp)]['bracket_width'],1)}</td><td>{e(C[(g,cp)]['truth_binary64'],4)}</td>"
    f"<td>${C[(g,cp)]['improvement_factor']:.2f}$</td></tr>"
    for g in ("cycle", "star") for cp in (1, 10))

greedy_rows = ""
for g in ("cycle", "star"):
    for cp in (1, 10):
        c = C[(g, cp)]
        path = ", ".join(f"${p['added']}$" for p in c["greedy_path"])
        last = c["greedy_path"][-1]
        greedy_rows += (f"<tr><td>{g}</td><td>${cp}$</td><td style='white-space:normal'>{path}</td>"
                        f"<td>{e(last['sharp_upper'],4)}</td><td>{e(c['all_16_tilts']['sharp_upper'],4)}</td>"
                        f"<td>${c['all_16_tilts']['ratio_to_truth']:.3g}$</td></tr>")

single_cols = list(C[("cycle", 1)]["single_added_tilt"].keys())
single_rows = "".join(
    "<tr><td>%s, $%d$</td>" % (g, cp) +
    "".join(f"<td>${C[(g,cp)]['single_added_tilt'][z]/C[(g,cp)]['sharp_upper_float']:.3f}$</td>" for z in single_cols) +
    "</tr>" for g in ("cycle", "star") for cp in (1, 10))
single_head = "".join(f"<th>${z}$</th>" for z in single_cols)

c1, c10 = C[("cycle", 1)], C[("cycle", 10)]
_s10 = c10["single_added_tilt"]
_bestz = min(_s10, key=_s10.get)
agg_sentence = (
    "For the ten-copy aggregate the picture inverts. Its law is concentrated near $45.8$, so the event "
    "$\\{S\\le0\\}$ is far out in the tail and the optimal single tilt is much stronger than $\\tfrac12$. "
    f"The best single addition on the cycle network is $z={_bestz}$, which alone multiplies the sharp bound by "
    f"{e(_s10[_bestz]/c10['sharp_upper_float'],2)}, taking it from {e(c10['sharp_upper_float'],3)} to {e(_s10[_bestz],3)}. With all sixteen tilts the certified aggregate bound is "
    f"{e(c10['all_16_tilts']['sharp_upper'],3)}, within a factor ${c10['all_16_tilts']['ratio_to_truth']:.1f}$ "
    f"of the oracle value {e(c10['truth_binary64'],3)}, against the published {e(c10['reported_chernoff'],3)}.")
atoms = ", ".join(f"$({n},\\,{w:.4f})$" for n, w in c1["witness_atoms"])

html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sharp Cyclecraft Tails</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
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
table{border-collapse:collapse;width:100%;font-size:.84em;margin:1rem 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--rule);padding:.3rem .45rem;text-align:right;white-space:nowrap}th{color:var(--ox);font-weight:600}
td:first-child,th:first-child{text-align:left}
.fig{width:100%;height:auto;margin:1rem 0;border:1px solid var(--rule)}
.cap{font-size:.88em;color:var(--mute);margin-top:-.4rem}
code{font-size:.85em;background:var(--panel);padding:0 .25em}
</style></head><body><main>
<p class="sub">Companion to Cyclecraft II and to <i>Positive Squares Under Noise</i> &middot; possq 0.2.0 &middot; 31 October 2026</p>
<h1>Sharp Wrong-Direction Tails for Cyclecraft</h1>
<p class="sub">Exact moment-problem certificates on the integers from Cyclecraft&rsquo;s own 320-bit enclosures</p>

<h2>0. Labels and sources</h2>
<p><span class="lab">proved</span> marks statements proved here. <span class="lab">certified</span> marks statements decided in exact rational arithmetic from the authoritative dyadic endpoints of Cyclecraft&rsquo;s <code>precision_cert_results.json</code>; the binary64 display fields are never read. <span class="lab">measured</span> marks binary64 observations, in particular the oracle law of the count, which is an independent falsification check and enters no certificate. Every number below is generated from <code>results/cyclecraft_ledger_cycle.json</code> and <code>results/cyclecraft_ledger_star.json</code>.</p>

<h2>1. What Cyclecraft certifies, and what it leaves on the table</h2>
<p>For each of the cycle and star networks, Cyclecraft encloses, for the signed target count $N=N_{10}$ after ten periods from the certified periodic law, the mean and variance and the probability generating function $G(z)=\mathbb E\,z^{N}$ at $z\in\{\tfrac12,\tfrac23,\tfrac34,\tfrac78\}$, each in a ball of radius below $10^{-77}$ that already includes the Doeblin initialisation error. The wrong-direction bound it reports is the best single Chernoff tilt, $\Pr(N\le0)\le\min_zG(z)$, attained at $z=\tfrac12$, and the ten-copy aggregate bound is its tenth power. A single tilt uses one of seven certified numbers. The question this note answers exactly is how large $\Pr(N\le0)$ can be for <i>any</i> integer-valued law consistent with all seven enclosures at once.</p>

<h2>2. Certificates on $\mathbb Z$</h2>
<p>Test functions are $q(n)=a_0+a_1n+a_2n^2+\sum_jb_jz_j^{\,n}$. Because every $z_j$ is rational, $q$ takes rational values at every integer, so the sandwich condition $q\ge\mathbf 1_{n\le0}$ can be checked exactly on any finite window without interval arithmetic. If it holds on all of $\mathbb Z$, then $\Pr(N\le0)\le\mathbb Eq\le\sum_k\max(c_k\ell_k,c_ku_k)$ for the enclosures $[\ell_k,u_k]$, exactly as in Proposition 1 of the parent monograph. Two lemmas close the infinite tails.</p>
<div class="thm"><span class="lab">proved</span><b>Lemma T (left tail).</b> Put $r_j=1/z_j$ and let $r_*$ be the largest $r_j$ with $b_j>0$, carrying coefficient $b_*$. Suppose every $b_j<0$ has $r_j<r_*$. With $P(m)=|a_0-1|+|a_1|m+\max(0,-a_2)m^2$ and $h(m)=\sum_{b_j<0}|b_j|(r_j/r_*)^m+P(m)r_*^{-m}$, the function $h$ is nonincreasing on $m\ge2r_*/(r_*-1)$, and if $b_*>h(L+1)$ for some $L+1\ge2r_*/(r_*-1)$ then $q(-m)\ge1$ for every integer $m\ge L+1$.</div>
<p class="pf">Each $(r_j/r_*)^m$ decreases. For $P(m)r_*^{-m}$ the derivative has the sign of $P'-P\ln r_*$, and $P'\le P\ln r_*$ once $m\ge2/\ln r_*$, which holds when $m\ge2r_*/(r_*-1)$ because $\ln r_*\ge(r_*-1)/r_*$. Then $q(-m)-1\ge r_*^m\,(b_*-h(m))\ge r_*^m\,(b_*-h(L+1))>0$, dropping the positive geometric terms other than $b_*$ and the term $a_2m^2$ when $a_2\ge0$.</p>
<div class="thm"><span class="lab">proved</span><b>Lemma R (right tail).</b> For $n\ge U+1$, $q(n)\ge a_0+a_1n+a_2n^2-\sum_{b_j<0}|b_j|z_j^{U+1}$, and the right side is checked at its exact real minimum over $[U+1,\infty)$.</div>
<p class="pf">Each $z_j^{\,n}\le z_j^{U+1}$ for $n\ge U+1$.</p>
<p>The optimisation is not done in floating point. An exact revised simplex with Bland&rsquo;s rule over rationals solves the moment problem on a window $\{-L,\dots,U\}$ enlarged by the two points at infinity of $\mathbb Z$, namely the limiting directions $e_{\text{dominant tilt}}$ as $n\to-\infty$ and $e_{n^2}$ as $n\to+\infty$, both carrying zero probability. Their presence forces the optimal dual coefficients onto the sign pattern that Lemmas T and R need; without them the window optimum can put negative weight on the fastest-growing function, which is valid on the window and false beyond it. This happened on the first run and is why the compactification is there.</p>
<div class="thm"><span class="lab">proved</span><b>Theorem S (exactness).</b> Let $(y,w)$ be the optimal dual and primal of the compactified window problem with data at the enclosure midpoints. If $y$ passes Lemmas T and R, then $\sup\Pr(N\le0)$ over consistent integer laws lies between the probability of the exact finite-window law and $\sum_k\max(y_k\ell_k,y_ku_k)$. When $w$ has no mass at infinity these two numbers differ only by the enclosure-radius term, so the supremum is determined to within $10^{-77}$ and is attained.</div>
<p class="pf">The upper side is Proposition 1 with the tail lemmas. The lower side is the finite-window primal solved exactly over $\mathbb Q$: an atomic law whose moments equal the exact midpoints, verified to lie in every ball. By complementary slackness on the window, the two objective values coincide at the midpoints, and the dual functional exceeds its midpoint value by at most $\sum_k|y_k|\rho_k$.</p>

<h2>3. Sharp values</h2>
<p><span class="lab">certified</span> The reproduction gate passes first: an independent run of Cyclecraft&rsquo;s Arb pipeline (periodic certificate, Doeblin residual, generating functions) produces balls that intersect the published ones for every published tilt, for """ + ("both networks" if all(repro.values()) else "NOT both networks") + r""". The sharp values are as follows, with the binary64 oracle law of $N$ in the last column but one.</p>
<table><tr><th>network</th><th>copies</th><th>Cyclecraft</th><th>sharp upper</th><th>exact witness</th><th>width</th><th>oracle truth</th><th>gain</th></tr>""" + main_rows + r"""</table>
<p>For the single count the bracket width is of order $10^{-83}$: the supremum is exactly $""" + f"{c1['sharp_upper_float']:.12f}" + r"""\ldots$ for the cycle network, attained by the law with atoms (position, mass) """ + atoms + r""". This is a theorem about the seven certified enclosures: no argument that uses only these seven numbers can certify anything below it, and Cyclecraft&rsquo;s bound is """ + f"${c1['improvement_factor']:.2f}$" + r""" times too pessimistic. For the ten-copy aggregate, the sharp bound computed from the aggregate&rsquo;s own enclosures (moments scaled by ten, generating functions raised to the tenth power) improves the reported """ + e(c10["reported_chernoff"], 3) + r""" to """ + e(c10["sharp_upper_float"], 4) + r""". <span class="lab">measured</span> The oracle values sit below every certified number, as soundness requires; the remaining distance to them is information the seven enclosures do not contain.</p>

<h2>4. Which tilt to certify next</h2>
<p>Certifying another $G(z)$ costs one more Arb matrix power. The ledger adds each of twelve candidate tilts singly and then greedily, re-solving the sharp problem exactly each time. The single-addition table reports the sharp bound after adding $z$, as a fraction of the sharp bound without it.</p>
<table><tr><th>case</th>""" + single_head + r"""</tr>""" + single_rows + r"""</table>
<table><tr><th>network</th><th>copies</th><th>greedy order</th><th>after 6</th><th>all 16 tilts</th><th>all / truth</th></tr>""" + greedy_rows + r"""</table>
""" + svg + r"""
<p class="cap">Figure 1. Certified wrong-direction bounds along the greedy tilt path. Dotted lines are the binary64 oracle values.</p>
<p>Three mechanisms are visible. <span class="lab">certified</span> For the single count, every tilt stronger than $\tfrac12$ is inert when added alone, with ratio exactly $1$: the extremal problem can satisfy the lower half of a strong-tilt enclosure by sending vanishing mass towards $-\infty$, at no cost in probability, and the extremal law already respects the upper half. The weak tilts $\tfrac35,\tfrac58,\tfrac56,\tfrac{15}{16}$ each remove a few percent on their own, because they constrain the bulk of the law near its mean. Along the greedy path, however, strong and weak tilts alternate. Once a weak tilt has reshaped the extremal law, a strong tilt that was inert becomes binding, and the second step, a strong tilt, gives the largest single gain on both networks. Tilts are therefore complements, and a ranking by single-addition value is the wrong design rule. """ + agg_sentence + r"""</p>
<div class="thm"><span class="lab">proved</span><b>Proposition C (consistency).</b> If the enclosure radii tend to $0$ and the certified tilt set contains a sequence with an accumulation point in $(0,1)$, the sharp upper bound converges to the true $\Pr(N\le0)$.</div>
<p class="pf">Any one tilt $z_0$ bounds $\Pr(N\le-k)\le G(z_0)z_0^{\,k}$ and the second moment bounds the right tail, so consistent laws form a tight family, and the event $\{n\le0\}$ is clopen in $\mathbb Z$, so its probability is continuous under weak convergence. Consistent sets decrease, and their intersection contains only laws whose generating function agrees with the true one on a set with an accumulation point inside the annulus of analyticity; by the identity theorem that is the true law. A decreasing family of nonempty compact sets has suprema converging to the supremum on the intersection.</p>
<p><span class="lab">open</span> The rate in Proposition C, and the optimal placement of $k$ tilts for a given horizon and copy count, are not settled. The greedy paths suggest the answer depends sharply on whether the target event sits in the bulk or the far tail.</p>

<h2>5. Scope</h2>
<p>All statements concern the declared enclosures for the declared protocol and initial law; nothing here concerns hardware. The aggregate bound uses only the aggregate&rsquo;s own enclosures and ignores that its law is a ten-fold convolution; a certificate that exploits the convolution would be stronger and is not attempted. The oracle is a binary64 lifted-chain computation with escaped mass below $10^{-14}$ and is reported only as a falsification check; """ + ("every certified bound in the ledger lies above it." if sound else "AT LEAST ONE CERTIFIED BOUND FELL BELOW IT.") + r"""</p>
<h2>Reproduction</h2>
<p><code>python -m possq.cyclecraft_tails results/cyclecraft_precision_cert_results.json results cycle</code>, the same with <code>star</code>, then <code>python build_cyclecraft_note.py</code>. The input file is Cyclecraft&rsquo;s unmodified certificate output, copied into <code>results/</code>.</p>
</main></body></html>"""
(ROOT / "cyclecraft_tails.html").write_text(html)
print("wrote cyclecraft_tails.html; sound =", sound, "repro =", repro)
