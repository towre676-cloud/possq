# possq 0.4.0 — raw-response spectral disclosure and certified counting frontiers

Open **Certified_Disclosure.html** for the integrated mathematical report and seven new publication figures. The original moment and Cyclecraft papers remain below.

## 0.4.0 — direct measurements, exact certificates, rarer counting laws

- `possq/resolvent.py`: rational spectral-mass certificates directly from noisy collocated responses, exact Sturm verification after constant repair, and independently compatible rational atomic witnesses. No numerical differentiation.
- `possq/resolvent_design.py`: exact certificate reuse under additional readings and narrower error intervals. Numerical optimizer regressions cannot weaken the retained bounds.
- `possq/resolvent_campaign.py`: a 32-node/80-edge rational benchmark with permanently ambiguous labelled branch weights and certifiable spectral risk. Twelve readings at budget `1e-10*h(0)` give `[0.061226001012468806, 0.08216845646628852]` for true mass `0.0625`; a compatible witness reaches `0.08215034878984287`, a 0.0221% relative optimality gap on the upper target.
- `possq/cyclecraft_frontier.py`: sparse positive uniformization with Arb polynomials, rigorous omitted-mass accounting, independent-copy convolution, chronological joint-kernel composition, and optimized rational Collatz–Wielandt bounds.
- Both networks require exactly 8, 21, 42, 70 independent ten-period copies to meet risks `1e-12`, `1e-30`, `1e-60`, `1e-100`. Every smaller count is excluded by a certified lower bound.
- Cycle probability at 70 copies: `6.034587777022165e-101` (relative enclosure width below `9.474e-24`). One continuous 700-period trajectory instead gives `2.356974956752654e-102`; these are distinct experiments.

Reproduce the new results:

```bash
python -m pip install -r requirements.txt
python replay_extension.py
python -m possq.resolvent_campaign
python -m possq.cyclecraft_frontier results cycle star
python -m possq.cyclecraft_frontier results cycle star --horizons
```

The acquisition heuristic uses the nominal benchmark to score future readings. The final spectral certificates use only the selected readings, error budgets, positive measure and support assumptions. The counting laws concern the declared exact rational generators and protocols. See `RESOLVENT_BRIDGE.md` and `CYCLECRAFT_FRONTIER.md`.

## Original positive-moment programme

Rank-Two Kernel Calculus, equation family 1 (positive moments and spectral recovery).

A positive measure $\mu$ on $[-1,1]$ is observed through moments $\widehat\mu_k$ with declared error budgets $\varepsilon_k$ in the Chebyshev or monomial basis. `possq` returns exactly verified statements about where the spectral mass is and how much of it there is. Floating point proposes; exact Sturm sequences over $\mathbb Q$ (python-flint) dispose.

Contents. `possq/exact.py` is the rational core (Chebyshev conversion, Sturm counting, certified positivity, certified shifts and lower bounds, interval algebra). `possq/bounds.py` holds Theorem A (positive-square leakage bound $\mu(I\setminus U)\le\eta/g$), the Lemma B separation floor, sharp upper and lower mass certificates obtained by an exchange LP followed by an exact constant-shift repair (Theorem D), and exactly verified primal atomic witnesses. `possq/recover.py` provides exact synthetic instances, Prony estimation, cluster enclosures, certified node-presence radii and the measurement-sensitivity ranking. `possq/experiments.py` runs the ledger E1–E9, `possq/figures.py` draws the figures, `build_monograph.py` generates `monograph.html` from the ledger, and `tests/` holds the pytest suite (Sturm counts cross-checked against exact root sets, Chebyshev conversion against sympy, random soundness instances).

Reproduction: `pip install python-flint numpy scipy sympy matplotlib pytest`, then `python -m pytest tests`, `python -m possq.experiments results`, `python -m possq.figures results`, `python build_monograph.py`.

Usage sketch:

```python
from fractions import Fraction as F
from possq import Basis, MomentData, mass_certificate, primal_witness
from possq.exact import complement
data = MomentData(Basis("chebyshev", 16), mu_hat, eps)          # exact rationals
S = complement(data.I, [(F(-13,20), F(-11,20))])                 # outside one cluster
up = mass_certificate(data, S, "upper")                          # mu(S) <= up.value, exact
wit = primal_witness(data, S, "upper")                           # sup mu(S) >= wit.value, exact
```

## 0.2.0 — integer tails and Cyclecraft II

`possq/integer.py` gives sharp certified bounds on $\Pr(N\le t)$ for an integer-valued $N$ with certified enclosures of $\mathbb E N$, $\mathbb E N^2$ and $\mathbb E z^N$ ($0<z<1$ rational). Every test function is rational on $\mathbb Z$, so the window LP is solved by an exact revised simplex (Bland) over $\mathbb Q$ on a window compactified by the two points at infinity, and the infinite tails are closed by Lemma T (left) and Lemma R (right). The same solve returns an exact witness law, so the supremum is bracketed to the enclosure radius.

`possq/cyclecraft.py` reads Cyclecraft II's `precision_cert_results.json` (dyadic endpoints only), reproduces its Arb tilt pipeline to certify additional tilts, and provides an independent binary64 oracle law for falsification. `possq/cyclecraft_tails.py` runs ledger C (single count and 10-copy aggregate, both networks, single and greedy tilt additions); `build_cyclecraft_note.py` writes `cyclecraft_tails.html`.

```
python -m possq.cyclecraft_tails results/cyclecraft_precision_cert_results.json results cycle
python -m possq.cyclecraft_tails results/cyclecraft_precision_cert_results.json results star
python build_cyclecraft_note.py
```

## 0.3.0 — exact counting laws for Cyclecraft II

`possq/cyclecraft_exact.py` certifies the whole law of a signed edge count for Cyclecraft's dyadic three-hold protocols: a killed count lift (Lemma K: killed(E) <= P(E) <= killed(E) + lost) propagated by Arb matrix exponentials of the exact rational generators from Cyclecraft's own periodic-law certificate, doubled to longer horizons (Lemma D), convolved over independent copies, and bounded for every horizon by a Collatz–Wielandt tilt (Proposition H). Rare-shortcut winding is handled by counting the shortcut channel itself.

Headline values (cycle network): P(N_10 <= 0) = 0.01092271816412091956957450240..., 10-copy aggregate 7.1680951562e-16 (published bound 4.32e-11), auxiliary P(|A_10| >= 12) = 1.1299e-7 (published 1.25e-3). `possq/cyclecraft_crosscheck.py` recovers every summary Cyclecraft certified independently to binary64 precision.

```
python -m possq.cyclecraft_exact_ledger results cycle aux
python -m possq.cyclecraft_exact_ledger results star shortcut
python -m possq.cyclecraft_crosscheck results
python build_cyclecraft_exact_note.py
```
