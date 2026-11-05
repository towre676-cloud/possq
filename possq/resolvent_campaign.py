"""Raw-response spectral disclosure on an exactly soluble grounded RC network.

The benchmark has 32 labelled nodes, 80 unknown edge conductances, and one
collocated port.  Permuting the five Cartesian axes changes the labelled edge
weights while preserving every port response.  Spectral risk is nevertheless
accessible.  All input observations, uncertainty budgets and reference spectral
weights are rational.  Selection uses a floating LP; final reported bounds are
independently checked exact rational certificates.
"""
from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


GAMMA = F(1, 64)
AXES = [F(1, 32), F(1, 16), F(1, 8), F(1, 4), F(1, 2)]
N = 2 ** len(AXES)
SPECTRUM = [GAMMA + 2 * sum((w for i, w in enumerate(AXES) if mask >> i & 1), F(0))
            for mask in range(N)]
I = (GAMMA, GAMMA + 2 * sum(AXES))
TARGET = [(GAMMA, F(1, 8))]
CANDIDATES = [F(1, 512), F(1, 256), F(1, 128), F(1, 64), F(1, 32),
              F(1, 16), F(1, 8), F(1, 4), F(1, 2), F(1), F(2), F(4), F(8)]


def response(shift, spectrum=SPECTRUM):
    return sum((F(1, N) / (lam + shift) for lam in spectrum), F(0))


def network(axis_weights=AXES):
    """Exact grounded Laplacian in a dyadic integer-labelled basis."""
    K = [[GAMMA if i == j else F(0) for j in range(N)] for i in range(N)]
    for i in range(N):
        for bit, w in enumerate(axis_weights):
            j = i ^ (1 << bit)
            if i < j:
                K[i][i] += w
                K[j][j] += w
                K[i][j] -= w
                K[j][i] -= w
    return K


def exact_symmetry_gate():
    from flint import fmpq, fmpq_mat
    def q(x): return fmpq(x.numerator, x.denominator)
    K = network()
    swapped = [AXES[1], AXES[0], *AXES[2:]]
    K2 = network(swapped)
    assert K != K2
    def perm(i):
        return (i & ~3) | ((i & 1) << 1) | ((i & 2) >> 1)
    assert perm(0) == 0
    assert all(K2[i][j] == K[perm(i)][perm(j)] for i in range(N) for j in range(N))
    checks = []
    for shift in [F(0), F(1, 64), F(1, 4), F(2)]:
        A = fmpq_mat([[q(K[i][j] + (shift if i == j else 0)) for j in range(N)] for i in range(N)])
        v = A.inv()[0, 0]
        assert v == q(response(shift))
        checks.append({'shift': str(shift), 'h': str(response(shift)), 'matrix_inverse_matches': True})
    return {'nodes': N, 'edges': len(AXES) * N // 2, 'different_labelled_networks': True,
            'port_fixed_permutation_identity': True, 'all_frequency_response_identity': True,
            'direct_inverse_checks': checks}


def fast_bound(shifts, noise, side='upper', density=1600):
    """Selection score only.  Every published result uses the exact verifier."""
    lo, hi = map(float, I)
    end = float(TARGET[0][1])
    # Each region gets its own endpoint at the discontinuity.  Upper and lower
    # sandwiches are conservative there, matching the exact implementation.
    grids = [np.linspace(lo, end, density // 3), np.linspace(end, hi, density)]
    pts = np.concatenate(grids)
    z = np.array([float(s) for s in shifts])
    B = np.column_stack([np.ones(len(pts)), 1 / (pts[:, None] + z)])
    y = np.array([1., *[float(response(s)) for s in shifts]])
    eps = np.array([0., *[float(noise * response(F(0)))] * len(shifts)])
    scale = np.max(np.abs(B), axis=0)
    A = B / scale
    yh, eh = y / scale, eps / scale
    m = len(y)
    inside = np.concatenate([np.ones(len(grids[0])), np.zeros(len(grids[1]))])
    if side == 'upper':
        obj = np.r_[yh, eh]
        rows = np.block([[-A, np.zeros_like(A)], [np.eye(m), -np.eye(m)], [-np.eye(m), -np.eye(m)]])
        rhs = np.r_[-inside, np.zeros(2*m)]
    else:
        obj = np.r_[-yh, eh]
        rows = np.block([[A, np.zeros_like(A)], [np.eye(m), -np.eye(m)], [-np.eye(m), -np.eye(m)]])
        rhs = np.r_[inside, np.zeros(2*m)]
    result = linprog(obj, A_ub=rows, b_ub=rhs, bounds=[(None, None)]*m+[(0,None)]*m,
                     method='highs', options={'dual_feasibility_tolerance': 1e-9,
                                             'primal_feasibility_tolerance': 1e-9})
    if not result.success:
        return float('inf') if side == 'upper' else -float('inf')
    return result.fun if side == 'upper' else -result.fun


def select_shifts(noise, count=8):
    chosen, path = [], []
    for step in range(count):
        scores = []
        for s in CANDIDATES:
            if s in chosen: continue
            upper = fast_bound(chosen + [s], noise)
            lower = fast_bound(chosen + [s], noise, 'lower')
            scores.append((upper-lower, float(s), s, upper, lower))
        score, _, s, up, low = min(scores)
        chosen.append(s)
        path.append({'readings': step+1, 'added_shift': str(s), 'proposal_upper': up,
                     'proposal_lower': low, 'proposal_width': score})
    return chosen, path


def certify(shifts, noise, previous=None):
    from .resolvent import (ResolventData, resolvent_mass_certificate,
                           resolvent_primal_witness, verify_resolvent_certificate,
                           verify_resolvent_witness)
    data = ResolventData(I, shifts, [response(s) for s in shifts],
                         [noise * response(F(0))] * len(shifts))
    up = resolvent_mass_certificate(data, TARGET, 'upper')
    low = resolvent_mass_certificate(data, TARGET, 'lower')
    raw_values = (float(low.value), float(up.value))
    if previous is not None:
        from .resolvent import ResolventCertificate
        from .resolvent_design import extend_certificate, best_certificate
        old_data=ResolventData.from_dict(previous['input'])
        carried_up=extend_certificate(old_data,ResolventCertificate.from_dict(previous['upper']),data)
        carried_low=extend_certificate(old_data,ResolventCertificate.from_dict(previous['lower']),data)
        up=best_certificate(data,[up,carried_up])
        low=best_certificate(data,[low,carried_low])
    assert verify_resolvent_certificate(data, up)
    assert verify_resolvent_certificate(data, low)
    true_mass = sum((F(1, N) for lam in SPECTRUM if any(a <= lam <= b for a,b in TARGET)), F(0))
    assert low.value <= true_mass <= up.value
    out = {'readings': len(shifts), 'shifts': list(map(str, shifts)), 'noise_relative_h0': str(noise),
           'input': data.as_dict(),
           'upper': up.as_dict(), 'lower': low.as_dict(),
           'upper_float': float(up.value), 'lower_float': float(low.value),
           'width_float': float(up.value-low.value), 'truth': str(true_mass),
           'both_exactly_verified': True, 'raw_proposal_lower':raw_values[0],
           'raw_proposal_upper':raw_values[1], 'retained_previous_certificate': previous is not None}
    # Oracle support is supplied as optional candidate points only.  Compatibility
    # and mass are checked independently, without trusting the reference weights.
    for side in ['upper', 'lower']:
        try:
            wit = resolvent_primal_witness(data, TARGET, side, extra_points=SPECTRUM, density=500)
            if verify_resolvent_witness(data, wit):
                out[side+'_witness'] = wit.as_dict()
        except (RuntimeError, ValueError) as e:
            out[side+'_witness_status'] = str(e)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', default='results/resolvent_campaign.json')
    parser.add_argument('--quick', action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    result={'schema':'possq.raw_resolvent.campaign.v1', 'network': exact_symmetry_gate(),
            'I': list(map(str,I)), 'target': [[str(a),str(b)] for a,b in TARGET],
            'spectrum': list(map(str,SPECTRUM)), 'spectral_weights':['1/32']*N,
            'truth_mass':'1/16', 'campaigns':[],
            'design_scope':'The nominal benchmark model scores candidate acquisitions. Only the selected raw readings and declared support enter each certificate.'}
    noises = [F(1,10**6)] if args.quick else [F(1,10**4), F(1,10**6), F(1,10**8), F(1,10**10)]
    for noise in noises:
        chosen, path = select_shifts(noise, count=12)
        row={'noise_relative_h0':str(noise), 'selection_path':path, 'adaptive':[], 'baseline':[]}
        for count in ([4,8] if args.quick else [2,4,6,8,10,12]):
            print('certify',noise,'adaptive',count,chosen[:count],flush=True)
            row['adaptive'].append(certify(chosen[:count],noise,row['adaptive'][-1] if row['adaptive'] else None))
            # Wide-band acquisition with the same number of scalar readings.
            baseline_indices=np.round(np.linspace(0,len(CANDIDATES)-1,count)).astype(int)
            baseline=[CANDIDATES[i] for i in baseline_indices]
            print('certify',noise,'baseline',count,baseline,flush=True)
            row['baseline'].append(certify(baseline,noise))
        result['campaigns'].append(row)
        Path(args.out).parent.mkdir(exist_ok=True,parents=True)
        Path(args.out).write_text(json.dumps(result,indent=2))
    result['seconds']=time.perf_counter()-start
    result['verdict']='PASS'
    Path(args.out).write_text(json.dumps(result,indent=2))
    print('PASS',result['seconds'],flush=True)


if __name__=='__main__': main()
