# Direct spectral disclosure from noisy collocated responses

For a symmetric positive-definite matrix K and collocated vector b, h(s)=bᵀ(K+sI)⁻¹b is the Stieltjes transform of a positive spectral measure. A known positive storage matrix C is handled by whitening. The spectral support interval and total mass normalization are declared inputs.

The acquisition basis is `1, 1/(lambda+s1), ..., 1/(lambda+sm)`. All denominators must remain positive on the support. For raw data `hhat[j]` with deterministic rational errors `eps[j]`, a rational dual function `q` that majorizes the target indicator gives the bound `c·hhat + |c|·eps` (including the mass coordinate). Lower sandwiches give the corresponding lower bound.

The common denominator is positive. Multiplying the inequalities by that denominator reduces their verification to polynomial signs on rational intervals. The standalone verifier uses exact rational Sturm sequences. The LP and Arb derivative-root isolation only propose a candidate; an exact constant repair and independent replay establish the reported bound. Tangencies may be rejected conservatively. Rational atomic witnesses are checked for nonnegative weights, support, total mass, every response interval, and target mass.

```python
from fractions import Fraction as F
from possq.resolvent import ResolventData, resolvent_mass_certificate, verify_resolvent_certificate
from possq.resolvent_campaign import I, TARGET, response

shifts = [F(1,16), F(1,512), F(1,32), F(1), F(1,8), F(2), F(1,256), F(1,128), F(1,4), F(1,2), F(4), F(8)]
data = ResolventData(I, shifts, [response(s) for s in shifts], [F(1,10**10)*response(F(0))]*12)
upper = resolvent_mass_certificate(data, TARGET, 'upper')
assert verify_resolvent_certificate(data, upper)
print(float(upper.value))
```

`extend_certificate` embeds a retained certificate in a larger basis and proves exact interval refinement. `best_certificate` selects among verified candidates. Thus additional readings never force the implementation to publish a worse retained bound.

The benchmark's cube-axis permutation fixes the port, changes labelled conductances, and preserves its complete response function. Matrix conjugacy is checked exactly. The target is nevertheless certifiable: at twelve nominally selected readings and budget 1e-10*h(0), its upper target is bracketed between a compatible witness 0.08215034878984287 and certificate 0.08216845646628852. The actual reference participation is 0.0625. The positive measure represents port-visible participation, not an unweighted count of every system mode.

Run `python -m possq.resolvent_campaign` and `python replay_extension.py`. Exact coefficients and witnesses are retained in `results/resolvent_campaign.json`.
