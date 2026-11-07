"""Fit a multiplicative uncertainty interval on validation only, report on test."""
import json
import argparse
import math
from pathlib import Path
import numpy as np

ROOT=Path('runs/2026-11-05/nearshore_no_ext_v2')
BASE=Path('runs/2026-11-05/nearshore_hybrid_v1')


def read(path):
    with np.load(path) as d:
        return {k:d[k].reshape(-1).astype(float) for k in ('target','prediction','sigma')}


def score(val,test):
    # Fit using validation examples inside the declared supervised depth domain.
    vm=(val['target']>=0)&(val['target']<=20)
    tm=(test['target']>=0)&(test['target']<=20)
    residual=np.abs(val['prediction'][vm]-val['target'][vm])
    ratio=residual/np.maximum(val['sigma'][vm],1e-6)
    # Split-conformal finite-sample order statistic, conditional on exchangeability.
    n=len(ratio)
    ranks={str(level):min(n,math.ceil((n+1)*level)) for level in (.68,.90)}
    q={level:float(np.sort(ratio)[rank-1]) for level,rank in ranks.items()}
    err=np.abs(test['prediction'][tm]-test['target'][tm]);sigma=test['sigma'][tm]
    def coverage(s): return float(np.mean(err<=s)*100)
    return {'validation_n':n,'test_n':int(tm.sum()),'calibration_quantiles':q,
            'validation_nominal_68_pct':float(np.mean(ratio<=1.0)*100),
            'validation_nominal_90_pct':float(np.mean(ratio<=1.6448536269514722)*100),
            'test_raw_68_pct':coverage(sigma),
            'test_raw_90_pct':coverage(1.6448536269514722*sigma),
            'test_calibrated_68_pct':coverage(q['0.68']*sigma),
            'test_calibrated_90_pct':coverage(q['0.9']*sigma),
            'median_calibrated_90_halfwidth_m':float(np.median(q['0.9']*sigma)),
            'method':'Validation absolute-residual / predicted-sigma conformal order statistics; no test fitting. Nominal marginal coverage requires exchangeability, weakened by spatial autocorrelation and geographic shift.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--new-root',type=Path,default=ROOT)
    new_root=parser.parse_args().new_root
    result={}
    for label,root in [('baseline',BASE),('candidate',new_root)]:
        if not (root/'test_predictions.npz').exists():continue
        result[label]=score(read(root/'validation_predictions.npz'),read(root/'test_predictions.npz'))
    new_root.mkdir(parents=True,exist_ok=True)
    (new_root/'uncertainty_calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
