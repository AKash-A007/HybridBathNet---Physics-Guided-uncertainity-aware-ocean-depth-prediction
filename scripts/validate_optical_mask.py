"""Predeclared risk/coverage audit; never turns a diagnostic into a mask."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('runs/2026-11-05/optical_support_audit_v1')
DATA=Path('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz')


def assess(path,coords,tolerance):
    with np.load(path) as d:
        target,pred,limit=(d[k].reshape(-1).astype(float) for k in
                           ('target','prediction','diagnostic_optical_limit'))
    domain=(target>=0)&(target<=20)
    retained=domain&np.isfinite(limit)&(pred<=limit)
    ids=np.flatnonzero(retained)
    coverage=float(retained.sum()/domain.sum())
    if not len(ids):
        return {'n_retained':0,'coverage_pct':0.,'rmse_m':None,
                'error_exceedance_pct':None,'upper95_exceedance_pct':None,
                'retained_clusters':0,'passes_empirical_gate':False}
    errors=np.abs(pred[ids]-target[ids]);fail=(errors>tolerance).astype(float)
    _,group=np.unique((coords[ids]+32)//200,axis=0,return_inverse=True)
    unique=np.unique(group)
    rng=np.random.RandomState(42);rep=[]
    for _ in range(2000):
        take=np.concatenate([np.flatnonzero(group==u) for u in
                             rng.choice(unique,size=len(unique),replace=True)])
        rep.append(fail[take].mean())
    upper=float(np.percentile(rep,95))
    # This checks predictive-error risk only, not physical bottom visibility.
    passes=coverage>=.20 and len(unique)>=10 and upper<=.10
    return {'n_retained':len(ids),'coverage_pct':100*coverage,
            'rmse_m':float(np.sqrt(np.mean((pred[ids]-target[ids])**2))),
            'error_exceedance_pct':float(100*fail.mean()),
            'upper95_exceedance_pct':100*upper,'retained_clusters':len(unique),
            'passes_empirical_gate':bool(passes)}


def main():
    with np.load(DATA) as d:
        coords=d['coords'];indices={s:d[s] for s in ('val_idx','test_idx')}
    report={'candidate':'model-learned per-band K/A plus temporal-std proxy, SNR3',
            'predeclared_gate':{'minimum_coverage_pct':20,'minimum_spatial_clusters':10,
                                'maximum_upper95_error_exceedance_pct':10,
                                'tolerances_m':[2,3]},
            'caveat':'GEBCO residual risk is not independent seabed truth or an optical-bottom-detection label. Test set has been previously inspected; this is exploratory.',
            'splits':{}}
    for split,idx in indices.items():
        path=ROOT/f'{split}_flags.npz'
        report['splits'][split]={str(t):assess(path,coords[idx],t) for t in (2,3)}
    report['operational_mask_enabled']=False
    (ROOT/'risk_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
