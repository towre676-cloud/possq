"""Consistency of the exact laws with Cyclecraft's independently certified balls (and the binary64 oracle)."""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from .cyclecraft import ball, load, oracle_law


def main(resdir):
    resdir = Path(resdir)
    R = load(resdir / "cyclecraft_precision_cert_results.json")
    ex = {}
    for f in ("exact_cycle_aux.json", "exact_star_shortcut.json"):
        ex.update(json.loads((resdir / f).read_text()))
    out = {}
    for g in ("cycle", "star"):
        c = R["certificates"][g]
        pm = {int(k): v for k, v in ex[g]["pmf_n10"].items()}
        lost = ex[g]["lost_n10"]
        mid = {k: (a + b) / 2 for k, (a, b) in pm.items()}
        rows = []
        for z, enc in c["target_tilted_mgfs"].items():
            zf = float(F(z))
            ours = sum(v * zf ** k for k, v in mid.items())
            lo, hi = ball(enc)
            rows.append(dict(quantity="E z^N, z=" + z, ours=ours, cyclecraft=float(lo),
                             rel_diff=abs(ours - float(lo)) / float(lo)))
        r10 = [r for r in c["finite_horizon_moments"]["target"] if r["periods"] == 10][0]
        mean = sum(k * v for k, v in mid.items())
        var = sum(k * k * v for k, v in mid.items()) - mean ** 2
        rows.append(dict(quantity="mean", ours=mean, cyclecraft=float(ball(r10["mean"])[0]),
                         rel_diff=abs(mean - float(ball(r10["mean"])[0])) / mean))
        rows.append(dict(quantity="variance", ours=var, cyclecraft=float(ball(r10["variance"])[0]),
                         rel_diff=abs(var - float(ball(r10["variance"])[0])) / var))
        pmf_o, esc = oracle_law(g, 70)
        n = np.arange(-70, 71)
        rows.append(dict(quantity="P(N<=0) vs binary64 oracle", ours=ex[g]["P_N10_le_0"]["lower_float"],
                         cyclecraft=float(pmf_o[n <= 0].sum()),
                         rel_diff=abs(ex[g]["P_N10_le_0"]["lower_float"] - float(pmf_o[n <= 0].sum())) / float(pmf_o[n <= 0].sum())))
        out[g] = rows
    a = {int(k): (x + y) / 2 for k, (x, y) in ex["cycle_aux"]["pmf"].items()}
    raux = [r for r in R["certificates"]["cycle"]["finite_horizon_moments"]["auxiliary"] if r["periods"] == 10][0]
    va = sum(k * k * v for k, v in a.items()) - sum(k * v for k, v in a.items()) ** 2
    out["cycle_aux"] = [dict(quantity="aux variance", ours=va, cyclecraft=float(ball(raux["variance"])[0]),
                             rel_diff=abs(va - float(ball(raux["variance"])[0])) / va)]
    sc = []
    for row, theirs in zip(ex["shortcut"], R["rare_shortcuts"]):
        sc.append(dict(eps=row["eps"], any_jump_cyclecraft=float(ball(theirs["any_shortcut_jump_probability"])[0]),
                       net_nonzero_ours=row["P_net_nonzero"]["lower_float"],
                       consistent=row["P_net_nonzero"]["upper_float"] <= float(ball(theirs["any_shortcut_jump_probability"])[1])))
    out["shortcut"] = sc
    out["max_rel_diff"] = max(r["rel_diff"] for k in ("cycle", "star", "cycle_aux") for r in out[k])
    out["shortcut_consistent"] = all(s["consistent"] for s in sc)
    (resdir / "exact_crosscheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
