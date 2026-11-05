"""Replay all v0.4 certificates and meaningful package tests."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import platform
import subprocess
import sys
import time

if sys.flags.optimize:
    raise RuntimeError('Certificate replay requires Python without -O')

HERE=Path(__file__).resolve().parent
from possq.resolvent import (ResolventData,verify_resolvent_certificate,
                             verify_resolvent_witness)


def dyadic(d):return F(int(d['mantissa']))*F(2)**int(d['exponent2'])


def main():
    start=time.perf_counter()
    subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=HERE,check=True)
    d=json.loads((HERE/'results/resolvent_campaign.json').read_text())
    certificates=witnesses=0
    for campaign in d['campaigns']:
        for design in ['adaptive','baseline']:
            last=None
            for row in campaign[design]:
                data=ResolventData.from_dict(row['input'])
                for side in ['upper','lower']:
                    if not verify_resolvent_certificate(data,row[side]):
                        raise RuntimeError('invalid rational spectral certificate')
                    certificates+=1
                    if side+'_witness' in row:
                        if not verify_resolvent_witness(data,row[side+'_witness']):
                            raise RuntimeError('invalid rational atomic witness')
                        witnesses+=1
                lo,hi=F(row['lower']['value']),F(row['upper']['value'])
                if not lo<=F(1,16)<=hi:raise RuntimeError('reference excluded')
                if design=='adaptive' and last is not None and not(lo>=last[0] and hi<=last[1]):
                    raise RuntimeError('retained certificate refinement failed')
                last=lo,hi
    crossings={}
    for graph in ['cycle','star']:
        ledger=json.loads((HERE/f'results/frontier_{graph}.json').read_text())
        for target,c in ledger['threshold_crossings'].items():
            value=F(target);K=c['minimal_copies']
            if not dyadic(ledger['copies'][str(K)]['upper'])<=value:
                raise RuntimeError('risk threshold not certified')
            if any(dyadic(ledger['copies'][str(j)]['lower'])<=value for j in range(1,K)):
                raise RuntimeError('minimum-copy witness incomplete')
        for path in [HERE/f'results/frontier_{graph}.json',HERE/f'results/frontier_horizons_{graph}.json']:
            raw=json.loads(path.read_text())
            probs=raw.get('copies',raw.get('horizons'))
            for row in probs.values():
                row=row.get('probability',row)
                lo,hi=dyadic(row['lower']),dyadic(row['upper'])
                if not 0<=lo<=hi<=1:raise RuntimeError('invalid dyadic probability endpoints')
        crossings[graph]={t:c['minimal_copies'] for t,c in ledger['threshold_crossings'].items()}
    sources=['possq/resolvent.py','possq/resolvent_design.py','possq/resolvent_campaign.py',
             'possq/cyclecraft_frontier.py']
    receipt={'verdict':'PASS','python':platform.python_version(),'spectral_certificates':certificates,
             'rational_atomic_witnesses':witnesses,'exact_copy_thresholds':crossings,
             'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in sources},
             'seconds':time.perf_counter()-start}
    (HERE/'results/extension_replay_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
