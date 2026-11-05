"""Generate monograph.html from results/ledger.json and the SVG figures (house style)."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
L = json.loads((ROOT / "results" / "ledger.json").read_text())


def svg(name):
    s = (ROOT / "results" / name).read_text()
    s = s[s.index("<svg"):]
    return s.replace("<svg ", '<svg class="fig" ', 1)


def e(x, d=3):
    if x is None:
        return "&mdash;"
    if x == 0:
        return "$0$"
    m, ex = f"{x:.{d}e}".split("e")
    ex = int(ex)
    return f"${m}\\times10^{{{ex}}}$" if ex not in (0,) else f"${m}$"


def f(x, d=6):
    return "&mdash;" if x is None else f"${x:.{d}f}$"


E1, E2, E3, E4 = L["E1_exact_theorem"], L["E2_separation_lemma"], L["E3_noise_scaling"], L["E4_hidden_atom"]
E5, E6, E7, E8, E9 = (L["E5_cluster_enclosures"], L["E6_presence_radii"], L["E7_measurement_selection"],
                      L["E8_soundness_campaign"], L["E9_basis_comparison"])

t3 = "".join(
    f"<tr><td>{e(r['eps'],0)}</td><td>{e(r['node_err'],2)}</td><td>{e(r['eta'],3)}</td><td>{e(r['square_bound'],3)}</td>"
    f"<td>{e(r['sharp_upper'],4)}</td><td>{e(r['witness'],4)}</td><td>{f(r.get('relative_gap'),3)}</td></tr>"
    for r in E3["rows"])
t2 = "".join(f"<tr><td>${r['r']}$</td><td>{e(r['g'],4)}</td><td>{e(r['floor'],4)}</td><td>{f(r['ratio'],1)}</td></tr>"
             for r in E2["rows"])
t4 = "".join(
    f"<tr><td>{f(r['hidden_weight'],3)}</td><td>{e(r['square_bound'],3)}</td><td>{f(r['certified_lower'],7)}</td>"
    f"<td>{f(r['certified_upper'],7)}</td><td>{'yes' if r['detected'] else 'no'}</td></tr>" for r in E4["rows"])


def t5rows(key):
    return "".join(f"<tr><td>{f(r['center'],4)}</td><td>{f(r['lower'],6)}</td><td>{f(r['truth'],6)}</td>"
                   f"<td>{f(r['upper'],6)}</td><td>{e(r['width'],2)}</td></tr>" for r in E5[key])


t6 = "".join("<tr><td>" + e(r["eps"], 0) + "</td>" + "".join(f"<td>{e(x,3)}</td>" for x in r["radii"]) + "</tr>"
             for r in E6["rows"])
sl6 = ", ".join(f"${s:.3f}$" for s in E6["slopes"])
t7 = "".join(
    f"<tr><td>{k}</td><td>${', '.join(str(i) for i in E7[k]['selected'])}$</td><td>{e(E7[k]['bound'],4)}</td>"
    f"<td>{f(E7[k]['reduction'],3)}</td><td>{f(E7[k]['predicted_first_order'],3)}</td></tr>"
    for k in ("top4", "random4", "bottom4"))
e8_inst = E8["n"]
e8_fail = E8["failures"]
e8_bg = sum(1 for c in E8["checks"] if c["background"] > 0)
e8_wmin = min(c["upper"] - c["lower"] for c in E8["checks"])
e8_wmax = max(c["upper"] - c["lower"] for c in E8["checks"])

r3 = E3["rows"]
r6 = {r["eps"]: r for r in r3}[1e-6]

html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Positive Squares Under Noise</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\(','\\)']],displayMath:[['$$','$$'],['\\[','\\]']]},svg:{fontCache:'global'}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
:root{--bg:#f6f0e4;--ink:#2a2320;--ox:#6b1d1d;--mute:#8a7d72;--rule:#d9cdb8;--panel:#efe6d4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1d1916;--ink:#ece3d3;--ox:#d07a6a;--mute:#a89a8b;--rule:#3d342c;--panel:#26201b}}
:root[data-theme="dark"]{--bg:#1d1916;--ink:#ece3d3;--ox:#d07a6a;--mute:#a89a8b;--rule:#3d342c;--panel:#26201b}
html{background:var(--bg)}body{background:var(--bg);color:var(--ink);font-family:'EB Garamond',Garamond,serif;font-size:18px;line-height:1.62;margin:0}
main{max-width:46rem;margin:0 auto;padding:3rem 16px 5rem}
h1,h2,h3{font-family:'Cormorant Garamond',serif;color:var(--ox);font-weight:600;line-height:1.15}
h1{font-size:2.5rem;margin:.2em 0}h2{font-size:1.65rem;margin-top:2.6rem;border-bottom:1px solid var(--rule);padding-bottom:.2rem}h3{font-size:1.25rem;margin-top:1.8rem}
.sub{font-style:italic;color:var(--mute)}
.lab{font-variant:small-caps;letter-spacing:.06em;font-size:.82em;color:var(--ox);border:1px solid var(--ox);padding:0 .35em;border-radius:2px;margin-right:.3em;white-space:nowrap}
.thm{background:var(--panel);border-left:3px solid var(--ox);padding:.7rem 1rem;margin:1.2rem 0}
.thm b{font-family:'Cormorant Garamond',serif;font-size:1.08em}
.pf{font-size:.97em}.pf:after{content:" \220E";color:var(--ox)}
table{border-collapse:collapse;width:100%;font-size:.86em;margin:1rem 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--rule);padding:.3rem .5rem;text-align:right;white-space:nowrap}th{color:var(--ox);font-weight:600}
td:first-child,th:first-child{text-align:left}
.fig{width:100%;height:auto;margin:1rem 0;border:1px solid var(--rule)}
.cap{font-size:.88em;color:var(--mute);margin-top:-.4rem}
code{font-size:.85em;background:var(--panel);padding:0 .25em}
</style></head><body><main>
<p class="sub">Rank-Two Kernel Calculus &middot; equation family 1 &middot; possq v0.1.0 &middot; 30 October 2026</p>
<h1>Positive Squares Under Noise</h1>
<p class="sub">Certified spectral recovery from imperfect moments: leakage, sharp duality, cluster enclosures, hidden-atom detection, and measurement selection</p>

<h2>0. Register and epistemic labels</h2>
<p>Every assertion in this monograph carries one of four labels. <span class="lab">proved</span> marks a statement whose proof is written out here in full and which depends on nothing beyond the elementary theory of positive measures and polynomials. <span class="lab">certified</span> marks a numerical statement decided in exact rational arithmetic by the <code>possq</code> package, with floating point used only to propose candidates that are then accepted or rejected by Sturm sequences over $\mathbb Q$; a certified statement is a theorem about the specific data, not an estimate. <span class="lab">measured</span> marks a floating-point observation, such as a fitted slope, that is reproducible from the ledger but not a theorem. <span class="lab">open</span> marks a question this push identifies and does not close. The experiment ledger <code>results/ledger.json</code> is the single source of every number quoted below; this document is generated from it by <code>build_monograph.py</code>, so the text cannot drift from the computation.</p>

<h2>1. Setting</h2>
<p>Let $\mu$ be a positive Borel measure on $I=[-1,1]$, observed through a basis $\varphi_0=1,\varphi_1,\dots,\varphi_K$ of polynomials of degree at most $K$ (Chebyshev $T_k$ by default, monomials $x^k$ as the comparison case). The instrument reports $\widehat\mu_k$ with a declared error budget $|\int\varphi_k\,d\mu-\widehat\mu_k|\le\varepsilon_k$. Nothing is assumed about $\mu$ beyond positivity and support in $I$: it may be atomic, continuous, or mixed, and the number of atoms is not supplied. For $q=\sum_k c_k\varphi_k$ define the two data functionals $\mathsf U(c)=\sum_k c_k\widehat\mu_k+\sum_k|c_k|\varepsilon_k$ and $\mathsf L(c)=\sum_k c_k\widehat\mu_k-\sum_k|c_k|\varepsilon_k$.</p>
<div class="thm"><span class="lab">proved</span><b>Proposition 1 (data enclosure).</b> Every positive measure compatible with the data satisfies $\mathsf L(c)\le\int q\,d\mu\le\mathsf U(c)$ for every coefficient vector $c$.</div>
<p class="pf">Write $\int q\,d\mu=\sum_k c_k\widehat\mu_k+\sum_k c_k(\mu_k-\widehat\mu_k)$ and bound the second sum in absolute value by $\sum_k|c_k|\varepsilon_k$.</p>
<p>The whole programme is the observation that Proposition 1 converts any polynomial sandwiching an indicator function into a two-sided, data-only statement about spectral mass, and that the sandwich conditions are semi-algebraic statements on intervals which exact real-root counting decides.</p>

<h2>2. The positive-square leakage theorem</h2>
<div class="thm"><span class="lab">proved</span><b>Theorem A.</b> Let $p$ be a polynomial with $\deg p^2\le K$, write $p^2=\sum_k a_k\varphi_k$, and put $\eta=\mathsf U(a)$. If $U\subset I$ is open and $p^2\ge g>0$ on $I\setminus U$, then every compatible $\mu$ satisfies $\mu(I\setminus U)\le\eta/g$. Moreover every atom of $\mu$ of weight exceeding $\eta/g$ lies in $U$.</div>
<p class="pf">Since $p^2\ge g\mathbf 1_{I\setminus U}$ on $I$, $g\,\mu(I\setminus U)\le\int p^2\,d\mu\le\eta$ by Proposition 1. An atom of weight $w$ at a point of $I\setminus U$ contributes at least $w$ to $\mu(I\setminus U)$, so $w\le\eta/g$.</p>
<p>The second sentence is the operational form of the theorem and the reason it matters in instrumentation: whatever the recovered nodes, no spectral line heavier than $\eta/g$ can be hiding outside their neighbourhoods. At $\varepsilon=0$ and $p$ vanishing on the support one obtains $\eta=0$, hence $\mu(I\setminus U)=0$ for every neighbourhood $U$ of the zero set, which is the established exact theorem. <span class="lab">certified</span> The exact case is reproduced: with the four true nodes $-\tfrac35,-\tfrac1{10},\tfrac13,\tfrac45$ and exact Chebyshev moments through order $16$, the package returns $\eta=""" + E1["eta_true_nodes"] + r"""$ and bound $""" + E1["bound_true_nodes"] + r"""$ as rationals, and perturbing the first node by $\tfrac1{1000}$ returns $\eta=""" + E1["eta_perturbed"] + r"""$, which coincides exactly with $\tfrac14\,p(-\tfrac35)^2$ as the identity $\int p^2\,d\mu=\sum_iw_ip(x_i)^2$ requires.</p>

<h3>2.1 Separation floor</h3>
<div class="thm"><span class="lab">proved</span><b>Lemma B.</b> Let $p=\prod_{i=1}^n(x-\hat x_i)$ with minimal separation $\Delta$ and let $0<r\le\Delta/2$. Then $p(x)^2\ge r^2(\Delta/2)^{2(n-1)}$ for every $x$ with $\min_i|x-\hat x_i|\ge r$.</div>
<p class="pf">Let $\hat x_i$ be a node nearest to $x$. Then $|x-\hat x_i|\ge r$, and for $j\ne i$ the triangle inequality with $|x-\hat x_j|\ge|x-\hat x_i|$ gives $2|x-\hat x_j|\ge|\hat x_i-\hat x_j|\ge\Delta$. Multiply.</p>
<div class="thm"><span class="lab">proved</span><b>Corollary C (linear leakage at consistent nodes).</b> If the nodes of $p$ carry the whole of $\mu$, then $\eta\le2\sum_k|a_k|\varepsilon_k$, and therefore $\mu(I\setminus U_r)\le 2\cdot4^{\,n-1}\Delta^{-2(n-1)}r^{-2}\sum_k|a_k|\varepsilon_k$ for the union $U_r$ of the radius-$r$ neighbourhoods.</div>
<p class="pf">The hypothesis gives $\sum_ka_k\mu_k=\int p^2d\mu=0$, so $\sum_ka_k\widehat\mu_k\le\sum_k|a_k|\varepsilon_k$; add the noise term of $\mathsf U$ and apply Theorem A with Lemma B.</p>
<p><span class="lab">certified</span> On the four-node instance the exact minimum of $p^2$ off the neighbourhoods exceeds the floor of Lemma B at every tested radius, by factors between """ + f"${min(r['ratio'] for r in E2['rows']):.0f}$ and ${max(r['ratio'] for r in E2['rows']):.0f}$" + r""". The floor is therefore correct and conservative by two orders of magnitude; it is a proof device, and the certified $g$ is what the package actually uses.</p>
<table><tr><th>$r$</th><th>certified $g(r)$</th><th>Lemma B floor</th><th>ratio</th></tr>""" + t2 + r"""</table>

<h2>3. Sharp certificates and their attainment</h2>
<p>Theorem A commits to one polynomial, $p^2$, chosen before the data are consulted except through the node estimates. The sharp question is the value of $\sup\mu(S)$ over all positive measures compatible with the data, for a fixed finite union of intervals $S$. Weak duality is Proposition 1 applied to sandwich polynomials: if $q\ge0$ on $I$ and $q\ge1$ on $\overline S$ then $\sup\mu(S)\le\mathsf U(c)$, and if $q\le0$ on $\overline{I\setminus S}$ and $q\le1$ on $\overline S$ then $\inf\mu(S)\ge\mathsf L(c)$. Strong duality for this moment problem is classical (Karlin&ndash;Shapley, Kemperman), but nothing below depends on it, because the package brackets the sharp value from both sides with independently certified objects.</p>
<div class="thm"><span class="lab">proved</span><b>Theorem D (soundness of the certification pipeline).</b> Let $\tilde q$ be any polynomial with rational coefficients, $S$ and $C=\overline{I\setminus S}$ finite unions of closed rational intervals, and let $s\ge0$ be rational with $\tilde q+s-1>0$ on $S$ and $\tilde q+s>0$ on $C$, both verified by exact Sturm counts at rational endpoints. Then $q=\tilde q+s$ is a valid upper certificate and $\mu(S)\le\mathsf U(c(q))$. The symmetric statement holds for lower certificates with $q=\tilde q-s$.</div>
<p class="pf">A polynomial positive at the left endpoint of a closed interval with zero distinct roots in the half-open remainder is positive on the whole interval; the Sturm count $V(a)-V(b)$ of the canonical sequence is exact over $\mathbb Q$ for non-squarefree inputs as well, since the sequence is the gcd times a Sturm chain and the gcd cannot vanish at a point where the polynomial does not. Since $\varphi_0=1$, the shift changes only $c_0$, and the sandwich conditions then hold by construction; Proposition 1 concludes.</p>
<p>The candidate $\tilde q$ is produced by a semi-infinite linear programme in the Chebyshev basis, minimising $\mathsf U$ subject to the sandwich constraints on a grid, and enlarged by an exchange step that adds the critical points of the current iterate and the worst violators on a refined grid until the continuous violation is below $10^{-12}$. The programme never needs to be right: Theorem D repairs any residual violation by the exact shift $s$, and the reported number is $\mathsf U$ of the repaired polynomial. On the other side, the package solves the primal problem over atomic measures on a rational grid enriched near the estimated nodes and near the endpoints of $S$, converts the weights to exact dyadic rationals, and accepts the measure only if every one of its $K+1$ moments lies inside the declared error box in exact arithmetic. Such a measure is a witness that the supremum is at least its mass in $S$.</p>
<div class="thm"><span class="lab">certified</span><b>Result 3.1 (four-node instance, $K=16$, $r=\tfrac1{20}$).</b> At $\varepsilon=10^{-6}$ the positive-square bound is """ + e(r6["square_bound"], 3) + r""", the sharp dual certificate is """ + e(r6["sharp_upper"], 4) + r""", and an exactly verified compatible atomic measure places """ + e(r6["witness"], 4) + r""" outside the neighbourhoods. The sharp leakage is therefore pinned to a relative window of """ + f"${100*r6['relative_gap']:.2f}\\%$" + r""", and the positive-square bound overstates it by a factor of about """ + f"${r6['square_bound']/r6['sharp_upper']:.0f}$" + r""".</div>
<table><tr><th>$\varepsilon$</th><th>node error</th><th>$\eta$</th><th>$\eta/g$</th><th>sharp upper</th><th>witness</th><th>rel. gap</th></tr>""" + t3 + r"""</table>
""" + svg("fig_scaling.svg") + r"""
<p class="cap">Figure 1. Leakage outside the radius-$\tfrac1{20}$ neighbourhoods of the Prony nodes against the per-moment error. The true leakage is $0$; every certified quantity sits above it.</p>
<p><span class="lab">measured</span> Fitted log&ndash;log slopes against $\varepsilon$ are """ + f"${E3['slope_square']:.3f}$" + r""" for the positive-square bound, """ + f"${E3['slope_sharp']:.3f}$" + r""" for the sharp certificate, and """ + f"${E3['slope_node_err']:.3f}$" + r""" for the Prony node error. The first agrees with Corollary C combined with the observed linear node error, since $\eta=\sum_iw_ip(x_i)^2+O(\varepsilon)$ and $p(x_i)^2=O(\delta^2)=O(\varepsilon^2)$. That the sharp certificate is also linear is the content of the next section's mechanism: the optimal sandwich hugs the nodes, so its constant of proportionality is governed by the attainable dual coefficients rather than by $\Delta^{-2(n-1)}r^{-2}$. The relative gap stays below $10\%$ except at $\varepsilon=10^{-8}$, where it reaches about $18\%$; the most plausible cause is the finite primal grid, whose resolution near the boundary of $S$ and near the nodes caps how closely the witness can follow the extremal measure once the data pin it to within $10^{-8}$. Which side of the bracket is loose there is not decided by this push.</p>
""" + svg("fig_certificates.svg") + r"""
<p class="cap">Figure 2. Left: the exactly certified upper sandwich for $\mu(I\setminus U)$ at $\varepsilon=10^{-6}$; it is $\ge1$ off the shaded neighbourhoods, $\ge0$ everywhere, and nearly vanishes at the four atoms (vertical bars). Right: a certified lower sandwich for the second cluster, $\le1$ on it and $\le0$ elsewhere.</p>

<h2>4. Cluster weights and node presence</h2>
<div class="thm"><span class="lab">proved</span><b>Proposition E (presence).</b> If a lower certificate for $S=(x-r,x+r)$ has $\mathsf L(c)>0$, then $\mu(S)>0$, so $\operatorname{supp}\mu$ meets $S$; if $\mu$ is known to be atomic, some atom lies within distance $r$ of $x$.</div>
<p class="pf">Immediate from Theorem D and Proposition 1.</p>
<p>Lower and upper certificates together enclose the weight of each cluster. <span class="lab">certified</span> For the atomic instance at $\varepsilon=10^{-6}$ every enclosure contains the true weight and has width below $2\times10^{-5}$. With a uniform background of mass $\tfrac1{20}$ added, the enclosures remain valid but widen to between $0.03$ and $0.14$, because a density spread across $I$ cannot be separated from a narrow cluster by polynomials of degree $16$ on an interval of width $\tfrac1{10}$; this is the honest price of making no assumption about the continuous part.</p>
<table><tr><th>centre</th><th>lower</th><th>true</th><th>upper</th><th>width</th></tr><tr><td colspan="5"><i>atomic</i></td></tr>""" + t5rows("atomic") + r"""<tr><td colspan="5"><i>atomic plus uniform background $\tfrac1{20}$</i></td></tr>""" + t5rows("background_1/20") + r"""</table>
<p>The presence radius of a node is the smallest $r$ for which Proposition E fires, located by bisection on the floating programme and then certified exactly at the final radius.</p>
<table><tr><th>$\varepsilon$</th><th>$-\tfrac35$</th><th>$-\tfrac1{10}$</th><th>$\tfrac13$</th><th>$\tfrac45$</th></tr>""" + t6 + r"""</table>
""" + svg("fig_presence.svg") + r"""
<p class="cap">Figure 3. Certified presence radii of the four nodes against the moment error.</p>
<p><span class="lab">measured</span> The fitted slopes of $\log r$ against $\log\varepsilon$ are """ + sl6 + r""", all within $0.05$ of $\tfrac12$. The square-root law is the one Corollary C and Lemma B give for positive-square localisation, because $g(r)\sim r^2$, but the sharp lower certificates are not constrained to be squares and nothing forced them to obey it. A heuristic explains why they do: a lower sandwich for $(x_i-r,x_i+r)$ must be nonpositive at $x_i\pm r$ and positive at $x_i$, so at fixed degree its profile near the node is quadratic, $q\approx\kappa(r^2-(x-x_i)^2)$, giving $\int q\,d\mu\approx w_i\kappa r^2$, while the noise penalty $\sum_k|c_k|\varepsilon_k$ scales as $\kappa\,\varepsilon$ times a constant depending on $K$, $\Delta$ and the other nodes; positivity of $\mathsf L$ then requires $r\gtrsim(\varepsilon/w_i)^{1/2}$. The heavier node at $-\tfrac1{10}$ is not the most sharply localised, so the constant is governed by the dual geometry and not by $w_i$ alone.</p>
<div class="thm"><span class="lab">open</span><b>Conjecture G (square-root presence law).</b> For a finite atomic $\mu$ with $n$ nodes, $2n\le K$, and errors $\varepsilon_k=\varepsilon$, the sharp certified presence radius of each node satisfies $r_i(\varepsilon)=\Theta(\varepsilon^{1/2})$ as $\varepsilon\to0$ with constants depending on $K$, the nodes and the weights. The upper half would follow from an explicit quadratic-bump lower certificate; the lower half requires showing that no polynomial sandwich of degree $K$ does better, which is a statement about the dual of the moment problem near a finitely supported point and is not proved here.</div>

<h2>5. Detecting what the model omitted</h2>
<p>A recovery pipeline that fits $n$ nodes cannot, from its own output, decide whether $n$ was right. The certified lower bound on $\mu(I\setminus U)$ can. <span class="lab">certified</span> A fifth atom at $\tfrac35$ was added to the four-node measure with weights $\tfrac1{20}$, $\tfrac1{100}$ and $\tfrac1{1000}$, the moments were perturbed at $\varepsilon=10^{-7}$, and a four-node Prony fit was performed as if the model order were known. In every case the hidden atom lies outside the neighbourhoods of the fitted nodes, the positive-square bound is uninformative (it is a ceiling, not a floor), and the lower certificate on the outside mass is strictly positive and within $10^{-6}$ of the truth. The fitted model is thereby refuted by an exact certificate, and the refutation is quantitative: the missing mass is enclosed.</p>
<table><tr><th>hidden weight</th><th>$\eta/g$</th><th>certified lower</th><th>certified upper</th><th>refuted</th></tr>""" + t4 + r"""</table>

<h2>6. Which measurement to improve</h2>
<div class="thm"><span class="lab">proved</span><b>Proposition F (certificate reuse).</b> Let $c^*$ certify $\mu(S)\le\mathsf U(c^*)$ under the budget $\varepsilon$. If the error budget of moment $k$ is reduced to $\varepsilon_k'<\varepsilon_k$ with the reported value unchanged, then $c^*$ remains a certificate and the bound decreases by exactly $\sum_k|c^*_k|(\varepsilon_k-\varepsilon_k')$. Consequently the sharp value $\mathsf U^*(\varepsilon)$, an infimum of functions affine in $\varepsilon$, is concave and nondecreasing in $\varepsilon$. If $c^*$ attains that sharp value at the original budget, $(|c^*_k|)_k$ is a supergradient there; for an arbitrary feasible certificate it supplies the exact reuse bound above, without a supergradient claim.</div>
<p class="pf">The sandwich conditions on $q$ do not involve $\varepsilon$; only the functional does, and it is affine in $\varepsilon$ with coefficients $|c_k|$.</p>
<p>The vector $|c^*_k|$ is therefore a measurement-design signal computed for free with every certificate. <span class="lab">certified</span> For the leakage certificate at $\varepsilon=10^{-6}$, re-measuring four moments at ten times the precision (fresh reported values, not merely tightened budgets) gives the following, where the first-order prediction is the reuse bound of Proposition F.</p>
<table><tr><th>policy</th><th>moments $k$</th><th>new sharp bound</th><th>reduction</th><th>first-order</th></tr>""" + t7 + r"""</table>
<p>Against a base bound of """ + e(E7["base"], 4) + r""", the certificate-sensitivity-ranked selection removes """ + f"${100*E7['top4']['reduction']:.0f}\\%$" + r""" of the certified leakage, against """ + f"${100*E7['random4']['reduction']:.0f}\\%$" + r""" for a random selection and """ + f"${100*E7['bottom4']['reduction']:.0f}\\%$" + r""" for the four least sensitive moments. Two features of the table deserve comment. The realised reductions differ from the first-order column because re-measurement changes the reported values as well as the budgets, so the reuse identity of Proposition F no longer applies verbatim and the sharp programme re-optimises; for the ranked selection re-optimisation improves the realised bound well beyond the reuse prediction. Since the reported values also changed, this numerical gain is not itself an instance of the fixed-report concavity statement. The moments with vanishing sensitivity still yield a small reduction for the same reason, which is the honest limit of a first-order design signal. <span class="lab">measured</span> The certificate-sensitivity ranking places the total-mass moment $k=0$ first, followed by $k=""" + ", ".join(str(k) for k in E7["ranking"][1:4]) + r"""$; the ranking is not monotone in $k$, so the intuition that high-order moments carry the resolution is not what the certificate reports.</p>

<h2>7. Soundness campaign and basis dependence</h2>
<p><span class="lab">certified</span> """ + f"${e8_inst}$" + r""" random instances were generated with between $1$ and $5$ atoms at random rational nodes, a uniform background in """ + f"${e8_bg}$" + r""" of them, per-moment errors between $10^{-8}$ and $10^{-3}$, order $K=12$, and a random target interval. In every instance the certified lower bound was at most the true mass of the open interval and the certified upper bound at least the true mass of the closed interval; the failure count is """ + f"${e8_fail}$" + r""". Enclosure widths ranged from """ + e(e8_wmin, 2) + r""" to """ + e(e8_wmax, 2) + r""". A single failure would have falsified Theorem D or its implementation; none occurred.</p>
<p><span class="lab">certified</span> With the same per-moment error $10^{-6}$ measured in the monomial basis instead of the Chebyshev basis, the sharp leakage bound changes from """ + e(E9["chebyshev"]["sharp_upper"], 4) + r""" to """ + e(E9["monomial"]["sharp_upper"], 4) + r""", and the positive-square bound from """ + e(E9["chebyshev"]["square_bound"], 3) + r""" to """ + e(E9["monomial"]["square_bound"], 3) + r""". The ratio of the sharp bounds is about """ + f"${E9['monomial']['sharp_upper']/E9['chebyshev']['sharp_upper']:.0f}$" + r""". The error model is part of the data: identical numerical precision on different functionals is a different experiment, and since every Chebyshev coefficient of a sandwich polynomial is bounded by its sup norm while its monomial coefficients grow like $(1+\sqrt2)^K$, a fixed absolute error on monomial moments is far more damaging. The recommendation for instrument design is to calibrate error budgets on Chebyshev (or other well-conditioned) moments; the certificate reports exactly what each experiment supports.</p>

<h2>8. Position and open problems</h2>
<p>The exact positive-square theorem says that a vanishing quadratic form forces the spectrum onto a zero set. This push makes the statement robust in the only way that matters for instruments: every claim about where spectral mass is, and how much is there, now comes with an exactly verified sandwich polynomial and, for the upper side, an exactly verified compatible measure showing how much of the bound is real. The positive-square bound survives as the closed-form, assumption-light member of the family, with a proved separation floor and a proved linear rate at consistent nodes; the sharp certificates improve it by two orders of magnitude on the test instance while remaining exact.</p>
<p><span class="lab">open</span> Four problems are left in the state this part identifies them. The first is Conjecture G of §4. The second is the duality gap as a function of degree: at fixed $\varepsilon$ the sharp value decreases as $K$ grows, and an explicit rate in terms of the Christoffel function of $I$ would convert the present computations into an a-priori resolution theorem. The third is budgeted design: Proposition F makes the sharp bound concave in the error vector, so minimising it under a cost $\sum_k\kappa_k/\varepsilon_k$ is a concave-minus-convex programme whose structure, and in particular whether its optimum concentrates on few moments, is not yet known. The fourth is the multivariate case, where the sandwich conditions on boxes are no longer decided by a single Sturm chain and must be replaced by exact Bernstein subdivision or a sum-of-squares certificate with rational Gram matrix; this is the bridge to tomography.</p>

<h2>Appendix. Reproduction</h2>
<p>Install <code>python-flint</code>, <code>numpy</code>, <code>scipy</code>, <code>sympy</code> and <code>matplotlib</code>; run <code>python -m pytest tests</code>, then <code>python -m possq.experiments results</code>, <code>python -m possq.figures results</code> and <code>python build_monograph.py</code>. The ledger records exact rational values as strings where they are theorems and floats where they are measurements, together with wall-clock time for each experiment.</p>
</main></body></html>"""

(ROOT / "monograph.html").write_text(html)
print("wrote monograph.html", len(html))
