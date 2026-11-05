# Certified Cyclecraft frontier

`possq/cyclecraft_frontier.py` extends the certified finite-input probability
calculations to probabilities below 10^-100. All results concern the exact
five-state dyadic Cyclecraft protocol, with its certified periodic initial law.
The duration of one protocol period is 6 model time units; a ten-period
experiment lasts 60. Calibration or physical model error is outside this model.

## Independent ten-period experiments

The cycle and star campaigns both certify the same minimum number of independent
ten-period experiments needed to make their aggregate signed count nonpositive
with probability at most the stated target:

| Target | Minimum copies, cycle | Minimum copies, star |
|---|---:|---:|
| 10^-12 | 8 | 8 |
| 10^-30 | 21 | 21 |
| 10^-60 | 42 | 42 |
| 10^-100 | 70 | 70 |

Minimality is certified by checking the lower probability bound for **every**
smaller positive copy count. No monotonicity of aggregate tails is assumed.
Copies are identically distributed and independent, including their independent
periodic initializations. Consecutive blocks of one experiment are not treated
as independent copies.

At 70 copies the cycle probability is approximately
6.034587777022165e-101, with relative enclosure width below 9.474e-24. The star
probability is approximately 6.656257983214911e-102, with relative width below
8.597e-23. The exact dyadic endpoints are in `results/frontier_cycle.json` and
`results/frontier_star.json`. Decimal values and binary64 fields are previews;
the mantissa/exponent pairs are authoritative.

## Single trajectories and all horizons

`results/frontier_horizons_cycle.json` and its star counterpart preserve the
joint distribution of the signed count and the ending state. Doubling and
binary composition therefore retain state correlations between blocks. They
certify complete event brackets at horizons through 640 periods, and at 210,
420 and 700 periods for equal-total-time comparisons with independent copies.

For the cycle, the probabilities at 160, 320 and 640 periods are approximately
9.669782632799019e-25, 8.588223742660581e-48 and
9.549881919220722e-94. At 640 periods the relative enclosure width is below
5.474e-31; the all-omissions budget is below 5.227e-124.

At equal total duration 4200, a continuous 700-period cycle trajectory has
probability approximately 2.356974956752654e-102, about 25.60 times smaller
than the aggregate from 70 independently initialized ten-period trajectories.
Its relative enclosure width is below 2.426e-22. This comparison follows from
the separately certified correlated and independent calculations, rather than
assuming that resetting a system leaves its count law unchanged.

The optimized rational-tilt Collatz–Wielandt certificate separately gives
P(N_n <= 0) <= C rho^n for **every** integer n >= 0. For the cycle,
rho <= 0.719221915279738 and C <= 1.864678891290936; the bound is below
10^-100 from n=701. The star has rho <= 0.716983484692359 and crosses at
n=694. These are the first crossings of these particular exponential bounds,
not claims of minimal physical horizons. The rational tilt is numerically
proposed then rigorously verified; global optimality of the tilt is not claimed.

## Why sparse uniformization certifies the law

For each held stage choose a rational Lambda at least as large as every exit
rate. On the finite count window let Q_W be the killed count-lift generator and
P_W=I+Q_W/Lambda. This matrix is nonnegative and substochastic. Then

    exp(t Q_W) = exp(-Lambda t) sum_{j>=0} (Lambda t)^j P_W^j / j!.

Truncating this positive Poisson series retains only genuine path weights.
Shifting beyond the count window also deletes genuine paths. Neither operation
can add paths. The omitted Poisson tail is bounded by the next term divided by
1 - Lambda*t/(J+2), once this denominator is positive. Every matrix/vector
operation and exponential uses Arb ball arithmetic at the declared precision.

Let v be the retained final subprobability vector from the exact dyadic initial
proposal, and let delta bound its l1 distance to the true periodic law. For any
event E,

    retained(E) - delta <= P(E)
    P(E) <= retained(E) + delta + [1 - total_retained + delta].

The bracket includes the arithmetic uncertainty. The final loss is measured
from retained total mass, so it covers boundary killing and all positive-series
truncations together. In the default 512-bit ten-period calculation this loss
is below 8.18e-126 for both networks. In independent-copy convolutions a union
bound adds at most K times the base loss. In joint-kernel composition the new
loss is again certified directly by one minus retained mass.

The signed-count dependence is represented by Arb polynomials; count shifts,
truncation and convolution run in FLINT rather than forming a large dense
lifted matrix. The retained 0.3.0 dense calculation reports a cycle base-kernel
time of 97 seconds at lower precision. The new 512-bit sparse base calculation
takes about 5 seconds in this run. These are different precision settings and
implementations, so they are observations rather than a controlled speedup
benchmark. The complete 80-copy campaign takes about 10 seconds per network.

## Independent controls and reproduction

`tests/test_cyclecraft_frontier.py` compares sparse uniformization and joint
polynomial-kernel doubling with the separately implemented dense Arb matrix
exponential on small windows. An exact finite trinomial law controls polynomial
convolution. The optimized CW bound has its own positive-vector inequalities.

```bash
python -m pytest tests/test_cyclecraft_frontier.py
python -m possq.cyclecraft_frontier results cycle star
python -m possq.cyclecraft_frontier results cycle star --horizons
```

The finite brackets certify model probabilities; they are not statistical
confidence intervals. Their very small widths should not be interpreted as a
claim that a physical device's transition rates are known to that accuracy.
