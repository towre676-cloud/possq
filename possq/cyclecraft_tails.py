"""Ledger C1–C5: sharp certified wrong-direction tails for Cyclecraft II.

python -m possq.cyclecraft_tails PATH/TO/precision_cert_results.json OUTDIR
"""
import json
import sys
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from .cyclecraft import (certified_pgf, load, network_data, oracle_aggregate, oracle_law)
from .integer import IntegerData, chernoff_best, sharp_tail

CANDIDATES = [F(1, 16), F(1, 10), F(1, 8), F(1, 6), F(1, 5), F(1, 4), F(1, 3), F(2, 5),
              F(3, 5), F(5, 8), F(5, 6), F(15, 16)]


def truth(graph, copies):
    B = 70
    pmf, esc = oracle_law(graph, B)
    if copies == 1:
        n = np.arange(-B, B + 1)
        return float(pmf[n <= 0].sum()), float(esc), pmf
    agg = oracle_aggregate(pmf, copies)
    n = np.arange(len(agg)) - copies * B
    return float(agg[n <= 0].sum()), float(esc), pmf


def with_tilts(base: IntegerData, enc, zs, copies):
    allz = sorted(set(base.zs) | set(zs))
    pgf = []
    for z in allz:
        if z in base.zs:
            pgf.append(base.pgf[base.zs.index(z)])
        else:
            lo, hi = enc[z]
            pgf.append((lo ** copies, hi ** copies))
    return IntegerData(base.degree, list(base.poly), allz, pgf, base.label)


def run_case(R, graph, copies, enc, U):
    base, meta = network_data(R, graph, copies=copies)
    tr, esc, _ = truth(graph, copies)
    t0 = time.time()
    cert, wit, sol = sharp_tail(base, 0, 25, U)
    ch, chz = chernoff_best(base)
    row = dict(graph=graph, copies=copies, truth_binary64=tr, oracle_escaped_mass=esc,
               reported_chernoff=float(meta["chernoff_reported"][1] if copies == 1 else meta["aggregate_reported"][1]),
               chernoff_recomputed=float(ch), chernoff_z=str(chz),
               sharp_upper=str(cert.value), sharp_upper_float=float(cert.value),
               witness=str(wit.value), witness_float=float(wit.value),
               bracket_width=float(cert.value - wit.value),
               witness_atoms=[[n, float(w)] for n, w in wit.atoms],
               atoms_at_infinity=cert.left.get("atoms_at_infinity"),
               certificate_coeffs=[float(c) for c in cert.coeffs], window=[-cert.L, cert.U],
               pivots=sol.pivots, left_tail=cert.left, seconds=round(time.time() - t0, 2),
               truth_le_sharp=tr <= float(cert.value),
               improvement_factor=float(ch / cert.value))
    # one-tilt-at-a-time value of each candidate, then greedy growth
    singles = {}
    for z in CANDIDATES:
        d = with_tilts(base, enc, [z], copies)
        c, w, _ = sharp_tail(d, 0, 25, U)
        singles[str(z)] = float(c.value)
    chosen, path, cur = [], [], base
    for step in range(6 if copies == 1 else 3):
        best = None
        for z in CANDIDATES:
            if z in chosen:
                continue
            d = with_tilts(base, enc, chosen + [z], copies)
            c, w, _ = sharp_tail(d, 0, 25, U)
            if best is None or c.value < best[1]:
                best = (z, c.value, None if w is None else w.value)
        chosen.append(best[0])
        path.append(dict(added=str(best[0]), sharp_upper=float(best[1]),
                         witness=None if best[2] is None else float(best[2]),
                         ratio_to_truth=float(best[1]) / tr, truth_le_sharp=tr <= float(best[1])))
    full = with_tilts(base, enc, CANDIDATES, copies)
    cf, wf, _ = sharp_tail(full, 0, 25, U)
    row.update(single_added_tilt=singles, greedy_path=path,
               all_16_tilts=dict(sharp_upper=float(cf.value), witness=None if wf is None else float(wf.value),
                                 ratio_to_truth=float(cf.value) / tr, truth_le_sharp=tr <= float(cf.value)))
    return row


def main(src, outdir, graphs=("cycle", "star")):
    R = load(src)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / ("cyclecraft_ledger%s.json" % ("" if len(graphs) == 2 else "_" + graphs[0]))
    ledger = json.loads(fname.read_text()) if fname.exists() else {"source": str(Path(src).name), "cases": []}
    done = {(c["graph"], c["copies"]) for c in ledger["cases"]}
    for graph in graphs:
        enc = certified_pgf(graph, CANDIDATES)
        base, _ = network_data(R, graph)
        # C0: our Arb reproduction encloses Cyclecraft's own published tilt balls
        rep = certified_pgf(graph, base.zs)
        ledger.setdefault("reproduction", {})[graph] = all(
            max(rep[z][0], b[0]) <= min(rep[z][1], b[1]) for z, b in zip(base.zs, base.pgf))
        for copies, U in ((1, 100), (10, 250)):
            if (graph, copies) in done:
                continue
            row = run_case(R, graph, copies, enc, U)
            ledger["cases"].append(row)
            print(graph, copies, row["chernoff_recomputed"], row["sharp_upper_float"],
                  row["all_16_tilts"]["sharp_upper"], row["truth_binary64"], flush=True)
            (out / ("cyclecraft_ledger%s.json" % ("" if len(graphs) == 2 else "_" + graphs[0]))).write_text(json.dumps(ledger, indent=1))
    ledger["all_sound"] = all(c["truth_le_sharp"] and c["all_16_tilts"]["truth_le_sharp"] and
                              all(p["truth_le_sharp"] for p in c["greedy_path"]) for c in ledger["cases"])
    ledger["all_reproduced"] = all(ledger["reproduction"].values())
    (out / ("cyclecraft_ledger%s.json" % ("" if len(graphs) == 2 else "_" + graphs[0]))).write_text(json.dumps(ledger, indent=1))
    print("sound:", ledger["all_sound"], "reproduced:", ledger["all_reproduced"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results",
         tuple(sys.argv[3:]) or ("cycle", "star"))
