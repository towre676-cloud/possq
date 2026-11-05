"""High-precision Cyclecraft tails, without a dense lifted matrix exponential.

Sparse uniformization is performed on vectors of Arb polynomials, whose
coefficient k-lo records the signed edge count k.  A shift beyond the chosen
window kills that path.  Truncating the positive Poisson series also kills
mass.  Every retained coefficient is therefore a lower approximation to the
true joint law; one minus retained total mass bounds all omissions together.

The output concerns the declared five-state model and independent copies of
its ten-period experiment.  It does not estimate model or calibration error.
Run: python -m possq.cyclecraft_frontier results cycle star
"""
from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from flint import arb, arb_mat, arb_poly, ctx, fmpq
from scipy.optimize import minimize_scalar

from .cyclecraft import GRAPHS, N, QC, TAU, W, _arb_pipeline


def A(x):
    return arb(fmpq(x.numerator, x.denominator)) if isinstance(x, F) else arb(x)


def dyadic(x):
    m, e = x.man_exp()
    return F(int(m)) * F(2) ** int(e)


def endpoint(x):
    """An exact dyadic endpoint, with a decimal preview only for presentation."""
    m, e = x.man_exp()
    return {"mantissa": str(m), "exponent2": int(e), "decimal": x.str(45)}


def bracket(lo, hi):
    lo, hi = lo.lower(), hi.upper()
    return {"lower": endpoint(lo), "upper": endpoint(hi),
            "lower_float": float(lo), "upper_float": float(hi),
            "relative_width_upper": float(((hi - lo) / lo).upper()) if lo > 0 else None}


def periodic_law(graph, bits=512):
    # The inherited factory selects 320 bits on creation.  Its closures use
    # current precision, so raising it BEFORE all calculations rebuilds every
    # exponential, dyadic proposal, residual and Doeblin bound at `bits`.
    mono, periodic, _ = _arb_pipeline()
    ctx.prec = bits
    p, delta = periodic(mono(graph))
    assert all(p[i, 0] >= 0 for i in range(N))
    assert sum((p[i, 0] for i in range(N)), arb(0)) == 1
    assert delta >= 0
    return p, delta


def _validate_experiment(graph, periods, lo, hi, bits, error_digits):
    if graph not in GRAPHS:
        raise ValueError("unknown graph")
    if any(not isinstance(v, int) or isinstance(v, bool)
           for v in (periods, lo, hi, bits, error_digits)):
        raise ValueError("periods, count endpoints and precision settings must be integers")
    if periods <= 0 or bits < 64 or error_digits <= 0:
        raise ValueError("positive periods/error budget and at least 64 bits are required")
    if not lo <= 0 <= hi:
        raise ValueError("count window must contain zero")


def _transition(graph, stage):
    edges = GRAPHS[graph]
    auxq = F(1, 2) if graph == "cycle" else F(3, 4)
    conductances = QC[stage] + [auxq] * (len(edges) - 3)
    transitions = []
    exits = [F(0)] * N
    for e, (i, j) in enumerate(edges):
        for src, tgt, sign in ((i, j, 1), (j, i, -1)):
            rate = conductances[e] * W[stage][src]
            exits[src] += rate
            transitions.append((src, tgt, sign if e == 0 else 0, rate))
    rate = max(exits)
    return rate, [F(1) - v / rate for v in exits], [
        (i, j, shift, value / rate) for i, j, shift, value in transitions]


def uniformized_stage(vector, graph, stage, width, poisson_budget):
    """Apply a killed held-stage semigroup with a positive finite Poisson sum.

    Uniformization uses Lambda >= every exit rate, hence P=I+Q/Lambda is
    substochastic after window killing. The omitted scalar Poisson tail is
    bounded by its next term times a geometric series. No negative Taylor
    coefficients or cancellation are used.
    """
    rate, stays, transitions = _transition(graph, stage)
    stays = [A(v) for v in stays]
    transitions = [(i, j, shift, A(v)) for i, j, shift, v in transitions]
    a = A(rate * TAU)
    weight = (-a).exp()
    v = vector
    total = [weight * p for p in v]
    j = 0
    while True:
        nxt_weight = weight * a / (j + 1)
        if j + 2 > rate * TAU:
            tail = nxt_weight / (1 - a / (j + 2))
            if tail.upper() < poisson_budget:
                return total, j, tail.upper()
        out = [v[i] * stays[i] for i in range(N)]
        for src, tgt, shift, probability in transitions:
            p = v[src]
            if shift == 1:
                p = p.left_shift(1).truncate(width)
            elif shift == -1:
                p = p.right_shift(1)
            out[tgt] += probability * p
        v = out
        weight = nxt_weight
        total = [total[i] + weight * v[i] for i in range(N)]
        j += 1
        if j > 10000:
            raise RuntimeError("uniformization failed to reach Poisson budget")


def certified_law(graph, periods=10, lo=-75, hi=115, bits=512, error_digits=125):
    """Certified retained pmf and all-omissions bound for the periodic start."""
    _validate_experiment(graph, periods, lo, hi, bits, error_digits)
    p, delta = periodic_law(graph, bits)
    width = hi - lo + 1
    vector = [arb_poly([0] * (-lo) + [p[i, 0]]) for i in range(N)]
    rows = []
    budget = arb(10) ** (-error_digits) / (3 * periods)
    for period in range(periods):
        for stage in range(3):
            vector, terms, tail = uniformized_stage(vector, graph, stage, width, budget)
            rows.append({"period": period + 1, "stage": stage,
                         "highest_poisson_index": terms, "poisson_tail_upper": endpoint(tail)})
    values = [sum((v[k] for v in vector), arb(0)) for k in range(width)]
    retained = sum(values, arb(0))
    lost = (1 - retained + delta).upper()
    # pmf bounds refer to the retained submeasure, not the true per-bin law.
    # Adding `lost` ONCE after event summation gives the true event upper bound.
    pmf = {lo + k: (max(arb(0), (value - delta).lower()),
                    (value + delta).upper()) for k, value in enumerate(values)}
    assert lost >= 0
    assert retained.lower() <= 1
    return pmf, lost, {"bits": bits, "periods": periods, "period_duration": str(3 * TAU),
                       "experiment_duration": str(periods * 3 * TAU), "window": [lo, hi],
                       "periodic_initial_l1_error_upper": endpoint(delta.upper()),
                       "all_omissions_upper": endpoint(lost), "uniformization": rows}


def convolve(p1, p2):
    """Interval coefficients of the convolution of two retained submeasures."""
    low1, high1 = min(p1), max(p1)
    low2, high2 = min(p2), max(p2)
    lp = arb_poly([p1[k][0] for k in range(low1, high1 + 1)]) * arb_poly(
        [p2[k][0] for k in range(low2, high2 + 1)])
    hp = arb_poly([p1[k][1] for k in range(low1, high1 + 1)]) * arb_poly(
        [p2[k][1] for k in range(low2, high2 + 1)])
    return {low1 + low2 + k: (max(arb(0), lp[k].lower()), hp[k].upper())
            for k in range(high1 + high2 - low1 - low2 + 1)}


def wrong_direction(pmf, lost):
    lower = sum((lo for k, (lo, hi) in pmf.items() if k <= 0), arb(0))
    upper = sum((hi for k, (lo, hi) in pmf.items() if k <= 0), arb(0)) + lost
    return lower.lower(), upper.upper()


def count_kernel(graph, periods=10, lo=-110, hi=115, bits=512, error_digits=125):
    """Retained joint kernel from each of the five possible starting states."""
    _validate_experiment(graph, periods, lo, hi, bits, error_digits)
    ctx.prec = bits
    width = hi - lo + 1
    budget = arb(10) ** (-error_digits) / (3 * periods)
    columns = []
    highest = 0
    for start in range(N):
        vector = [arb_poly([0] * (-lo) + [int(i == start)]) for i in range(N)]
        for _ in range(periods):
            for stage in range(3):
                vector, terms, _ = uniformized_stage(vector, graph, stage, width, budget)
                highest = max(highest, terms)
        columns.append(vector)
    return [[columns[j][i] for j in range(N)] for i in range(N)], {
        "window": [lo, hi], "periods": periods, "bits": bits,
        "highest_poisson_index": highest, "poisson_budget_per_stage": endpoint(budget.upper())}


def compose_kernel(first, second, old_lo, new_lo, new_hi):
    """Compose retained joint kernels using fast C-level polynomial products.

    Counts add and states compose.  All omitted paths remain omitted, and
    discarding coefficients outside the new window only enlarges that set.
    The final loss is certified by one minus each column's retained mass.
    """
    shift = new_lo - 2 * old_lo
    if shift < 0:
        raise ValueError("new window starts below the product's support")
    width = new_hi - new_lo + 1
    return [[sum((second[i][a] * first[a][j] for a in range(N)), arb_poly()).right_shift(
        shift).truncate(width) for j in range(N)] for i in range(N)]


def kernel_event(kernel, lo, hi, p, delta):
    """Event enclosure from a substochastic joint kernel and certified start."""
    vector = [sum((kernel[i][j] * p[j, 0] for j in range(N)), arb_poly()) for i in range(N)]
    values = [sum((v[k] for v in vector), arb(0)) for k in range(hi - lo + 1)]
    retained_event = sum((values[k] for k in range(-lo + 1)), arb(0))
    total = sum(values, arb(0))
    lost = (1 - total + delta).upper()
    lower = max(arb(0), (retained_event - delta).lower())
    upper = (retained_event + delta + lost).upper()
    return lower, upper, lost


def horizon_ledger(graph, outdir, bits=512, maximum=640, error_digits=125):
    """Single correlated trajectory, certified by joint-kernel doubling.

    Unlike the independent-copy campaign this preserves the ending state of
    every block, so correlations between adjacent blocks are retained.
    """
    started = time.time()
    p, delta = periodic_law(graph, bits)
    lo, hi = -110, 115
    kernel, metadata = count_kernel(graph, 10, lo, hi, bits, error_digits)
    result = {"schema": "possq.cyclecraft_frontier_horizons.v1", "graph": graph,
              "statement": "Single continuous trajectory from the certified periodic law; no independence between consecutive periods.",
              "base_kernel": metadata, "periodic_initial_l1_error_upper": endpoint(delta.upper()),
              "horizons": {}}
    n = 10
    blocks = {}
    while n <= maximum:
        blocks[n] = kernel
        lower, upper, lost = kernel_event(kernel, lo, hi, p, delta)
        result["horizons"][str(n)] = {"probability": bracket(lower, upper),
            "all_omissions_upper": endpoint(lost), "window": [lo, hi],
            "duration": str(n * 3 * TAU)}
        print(graph, "single horizon", n, "tail", lower.str(8), "lost", lost.str(8), flush=True)
        n *= 2
        if n <= maximum:
            new_hi = n + 115
            kernel = compose_kernel(kernel, kernel, lo, lo, new_hi)
            hi = new_hi
    # Match the aggregate durations of K=21,42,70 independent experiments.
    # Binary block composition retains hidden-state correlations throughout.
    for target in (210, 420, 700):
        remaining, combined, total_periods = target, None, 0
        for size in sorted(blocks, reverse=True):
            if size > remaining:
                continue
            if combined is None:
                combined = blocks[size]
            else:
                combined = compose_kernel(combined, blocks[size], lo, lo,
                                          total_periods + size + 115)
            total_periods += size
            remaining -= size
        if remaining:
            continue
        lower, upper, lost = kernel_event(combined, lo, target + 115, p, delta)
        result["horizons"][str(target)] = {"probability": bracket(lower, upper),
            "all_omissions_upper": endpoint(lost), "window": [lo, target + 115],
            "duration": str(target * 3 * TAU), "purpose": "Equal total model time comparison with independent ten-period copies."}
        print(graph, "single horizon", target, "tail", lower.str(8), "lost", lost.str(8), flush=True)
    result["seconds"] = round(time.time() - started, 3)
    result["verdict"] = "PASS"
    (Path(outdir) / ("frontier_horizons_%s.json" % graph)).write_text(json.dumps(result, indent=2) + "\n")
    return result


def certified_cw(graph, z, p, delta, bits):
    mono, _, _ = _arb_pipeline()
    ctx.prec = bits
    M = mono(graph, z)
    proposal = np.array([[float(M[i, j].mid()) for j in range(N)] for i in range(N)])
    eig, vectors = np.linalg.eig(proposal)
    candidate = np.abs(vectors[:, np.argmax(eig.real)].real)
    v = [A(F(float(x)).limit_denominator(10**15)) for x in candidate]
    assert all(x > 0 for x in v)
    ratios = [(sum((M[i, j] * v[j] for j in range(N)), arb(0)) / v[i]) for i in range(N)]
    rho = max(x.upper() for x in ratios)
    rho_lower = min(x.lower() for x in ratios)
    c = max(((p[j, 0] + delta) / v[j]).upper() for j in range(N))
    C = (c * sum(v, arb(0))).upper()
    assert 0 < rho < 1
    return rho, C, rho_lower


def optimize_cw(graph, bits=512):
    """Numerical tilt proposal, then exact rational tilt and Arb CW verification."""
    from .cyclecraft import _generator
    from scipy.linalg import expm

    def objective(z):
        # Five-state proposal only; none of these floating values enter bounds.
        M = np.eye(N)
        for stage in range(3):
            G = _generator(graph, stage, None)
            i, j = GRAPHS[graph][0]
            G[j, i] *= z
            G[i, j] /= z
            M = expm(G * float(TAU)) @ M
        return max(np.linalg.eigvals(M).real)

    opt = minimize_scalar(objective, bounds=(0.12, 0.7), method="bounded",
                          options={"xatol": 1e-13})
    z = F(float(opt.x)).limit_denominator(10**9)
    p, delta = periodic_law(graph, bits)
    rho, C, rho_lower = certified_cw(graph, z, p, delta, bits)
    result = {"tilt": str(z), "rho_upper": endpoint(rho), "rho_lower": endpoint(rho_lower),
              "C_upper": endpoint(C), "decay_per_period_lower": endpoint((-rho.log()).lower()),
              "statement": "For every integer n >= 0, P(N_n <= 0) <= C_upper * rho_upper^n."}
    horizons = []
    for digits in (12, 30, 60, 100):
        target = arb(10) ** (-digits)
        n = int(np.ceil((float(C.log()) + digits * np.log(10)) / -float(rho.log())))
        while (C * rho**n).upper() > target.lower():
            n += 1
        while n > 0 and (C * rho**(n - 1)).upper() <= target.lower():
            n -= 1
        horizons.append({"target": "1e-%d" % digits, "first_horizon_for_this_CW_bound": n,
                         "duration": str(n * 3 * TAU), "probability_upper": endpoint((C * rho**n).upper()),
                         "preceding_bound_upper": endpoint((C * rho**(n - 1)).upper())})
    result["horizons"] = horizons
    return result


def run(graph, outdir, bits=512, max_copies=80, lo=-75, hi=115, error_digits=125):
    t0 = time.time()
    pmf, lost, metadata = certified_law(graph, lo=lo, hi=hi, bits=bits, error_digits=error_digits)
    print(graph, "base law done", round(time.time() - t0, 2), "s; lost", lost.str(8), flush=True)
    results = {"schema": "possq.cyclecraft_frontier.v1", "graph": graph,
               "model": "Cyclecraft exact dyadic five-state three-hold protocol",
               "copy_assumption": "Independent identically distributed ten-period experiments, each initialized in its own periodic law.",
               "certificate": metadata, "copies": {}, "threshold_crossings": {}}
    thresholds = {digits: arb(10) ** (-digits) for digits in (12, 30, 60, 100)}
    acc = None
    previous = None
    exact_lower_history = []
    for copies in range(1, max_copies + 1):
        acc = pmf if copies == 1 else convolve(acc, pmf)
        L = (copies * lost).upper()  # union bound on any missing base path
        bounds = wrong_direction(acc, L)
        results["copies"][str(copies)] = bracket(*bounds)
        for digits, threshold in thresholds.items():
            key = "1e-%d" % digits
            if key not in results["threshold_crossings"] and bounds[1] < threshold:
                # This is the first certified crossing among ALL counts scanned.
                # Minimality requires all preceding counts' lower bounds > target;
                # the code checks that rather than assuming tail monotonicity.
                earlier = [lower > threshold for lower in exact_lower_history]
                assert all(earlier)
                results["threshold_crossings"][key] = {"minimal_copies": copies,
                    "all_smaller_counts_excluded": True, "probability": bracket(*bounds),
                    "previous_probability": previous,
                    "total_model_time_across_copies": str(copies * 10 * 3 * TAU)}
        previous = bracket(*bounds)
        exact_lower_history.append(bounds[0])
        if copies % 10 == 0:
            print(graph, "copies", copies, "tail", bounds[0].str(8), "relative width",
                  results["copies"][str(copies)]["relative_width_upper"], flush=True)
    results["optimized_all_horizon_cw"] = optimize_cw(graph, bits)
    results["seconds"] = round(time.time() - t0, 3)
    results["verdict"] = "PASS"
    path = Path(outdir) / ("frontier_%s.json" % graph)
    path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    parser.add_argument("graphs", nargs="+", choices=("cycle", "star"))
    parser.add_argument("--bits", type=int, default=512)
    parser.add_argument("--max-copies", type=int, default=80)
    parser.add_argument("--lo", type=int, default=-75)
    parser.add_argument("--hi", type=int, default=115)
    parser.add_argument("--error-digits", type=int, default=125)
    parser.add_argument("--horizons", action="store_true")
    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for graph in args.graphs:
        if args.horizons:
            horizon_ledger(graph, args.outdir, args.bits, error_digits=args.error_digits)
        else:
            run(graph, args.outdir, args.bits, args.max_copies, args.lo, args.hi, args.error_digits)


if __name__ == "__main__":
    main()
