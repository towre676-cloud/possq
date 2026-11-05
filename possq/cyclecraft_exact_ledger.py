"""Ledger X: certified exact counting laws for Cyclecraft II.
python -m possq.cyclecraft_exact_ledger OUTDIR {cycle|star|shortcut}"""
import json
import sys
import time
from fractions import Fraction as F
from pathlib import Path

from flint import arb

from .cyclecraft_exact import (base_kernel, compose, convolve_pmf, cw_bound, event_bracket, law,
                               periodic_law)


def S(x):
    return dict(ball=x.str(40), lo=float(x.lower()), hi=float(x.upper()))


def br(lo, hi):
    return dict(lower=lo.str(35), upper=hi.str(35), lower_float=float(lo), upper_float=float(hi),
                width=float(hi - lo))


def target(graph, out):
    t0 = time.time()
    p, delta = periodic_law(graph)
    ker, lost = base_kernel(graph, 0, 10, -40, 60)
    res = {"delta": float(delta.upper()), "kernel_seconds": round(time.time() - t0, 1)}
    pmf, L = law(ker, lost, p, delta)
    res["pmf_n10"] = {k: [float(a), float(b)] for k, (a, b) in sorted(pmf.items())}
    res["lost_n10"] = float(L)
    lo, hi = event_bracket(pmf, L, lambda k: k <= 0)
    res["P_N10_le_0"] = br(lo, hi)
    # left-tail profile
    res["P_N10_le_t"] = {t: br(*event_bracket(pmf, L, lambda k, t=t: k <= t)) for t in (-5, -2, -1, 0, 1, 2)}
    # horizons by doubling
    horizons = {10: (ker, lost)}
    windows = {20: (-40, 90), 40: (-40, 130), 80: (-40, 170)}
    cur = (ker, lost)
    for n in (20, 40, 80):
        t1 = time.time()
        cur = compose(cur[0], cur[1], cur[0], cur[1], *windows[n])
        horizons[n] = cur
        pm, LL = law(cur[0], cur[1], p, delta)
        res.setdefault("horizons", {})[n] = dict(P_le_0=br(*event_bracket(pm, LL, lambda k: k <= 0)),
                                                 lost=float(LL), seconds=round(time.time() - t1, 1))
    res.setdefault("horizons", {})[10] = dict(P_le_0=res["P_N10_le_0"], lost=res["lost_n10"])
    # aggregates of K independent copies of N_10
    agg = {}
    acc, accL = pmf, L
    agg[1] = br(*event_bracket(acc, accL, lambda k: k <= 0))
    for K in range(2, 11):
        acc, accL = convolve_pmf(acc, accL, pmf, L)
        agg[K] = br(*event_bracket(acc, accL, lambda k: k <= 0))
    res["aggregate_P_le_0"] = agg
    # all-horizon exponential bound
    best = None
    grid = [F(k, 20) for k in range(4, 20)]
    cws = []
    for z in grid:
        c = cw_bound(graph, z, p, delta)
        cws.append(dict(z=str(z), rho=float(c["rho"]), rho_lo=float(c["rho_lo"]), C=float(c["C"])))
        if best is None or c["rho"] < best["rho"]:
            best = c
    res["cw_grid"] = cws
    res["cw_best"] = dict(z=str(best["z"]), rho_upper=best["rho"].str(25), rho_float=float(best["rho"]),
                          rho_lo_float=float(best["rho_lo"]), C=float(best["C"]),
                          decay_per_period=float(-(best["rho"].log())))
    res["seconds"] = round(time.time() - t0, 1)
    out[graph] = res


def auxiliary(out):
    t0 = time.time()
    p, delta = periodic_law("cycle")
    ker, lost = base_kernel("cycle", 3, 10, -45, 45)
    pmf, L = law(ker, lost, p, delta)
    out["cycle_aux"] = dict(P_abs_ge_12=br(*event_bracket(pmf, L, lambda k: abs(k) >= 12)),
                            P_abs_ge_t={t: br(*event_bracket(pmf, L, lambda k, t=t: abs(k) >= t))
                                        for t in (4, 8, 12, 16, 20)},
                            pmf={k: [float(a), float(b)] for k, (a, b) in sorted(pmf.items())},
                            lost=float(L), seconds=round(time.time() - t0, 1))


def shortcut(out):
    rows = []
    for eps in (F(1, 1024), F(1, 256), F(1, 64)):
        t0 = time.time()
        p, delta = periodic_law("star", eps)
        ker, lost = base_kernel("star", 5, 10, -20, 20, shortcut=eps)
        pmf, L = law(ker, lost, p, delta)
        rows.append(dict(eps=str(eps),
                         P_net_nonzero=br(*event_bracket(pmf, L, lambda k: k != 0)),
                         P_abs_ge_2=br(*event_bracket(pmf, L, lambda k: abs(k) >= 2)),
                         P_abs_ge_3=br(*event_bracket(pmf, L, lambda k: abs(k) >= 3)),
                         pmf={k: [float(a), float(b)] for k, (a, b) in sorted(pmf.items()) if abs(k) <= 6},
                         seconds=round(time.time() - t0, 1)))
    out["shortcut"] = rows


if __name__ == "__main__":
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for job in sys.argv[2:]:
        {"cycle": lambda: target("cycle", out), "star": lambda: target("star", out),
         "aux": lambda: auxiliary(out), "shortcut": lambda: shortcut(out)}[job]()
        (outdir / ("exact_%s.json" % "_".join(sys.argv[2:]))).write_text(json.dumps(out, indent=1, default=str))
        print(job, "done", flush=True)
