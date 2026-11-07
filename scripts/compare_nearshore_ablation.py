"""Matched-coordinate change in RMSE with spatial-cluster bootstrap intervals."""
import json
import argparse
from pathlib import Path
import numpy as np

BASE=Path('runs/2026-11-05/nearshore_hybrid_v1')
DEFAULT_NEW=Path('runs/2026-11-05/nearshore_no_ext_v2')


def read(path):
    with np.load(path) as d:
        return d['target'].reshape(-1).astype(float),d['prediction'].reshape(-1).astype(float)


def compare(a,b,coords):
    t0,p0=a;t1,p1=b
    if len(t0)!=len(t1) or not np.allclose(t0,t1,atol=1e-5):
        raise ValueError('Matched coordinates/targets are required')
    _,groups=np.unique((coords+32)//200,axis=0,return_inverse=True)
    rng=np.random.RandomState(42)
    def subset(mask):
        if not mask.any():return {'n':0}
        y=t0[mask];old=p0[mask];new=p1[mask];g=groups[mask]
        e0=(old-y)**2;e1=(new-y)**2
        unique=np.unique(g);boots=[]
        for _ in range(2000):
            draw=rng.choice(unique,size=len(unique),replace=True)
            ids=np.concatenate([np.flatnonzero(g==v) for v in draw])
            boots.append(float(np.sqrt(e1[ids].mean())-np.sqrt(e0[ids].mean())))
        return {'n':len(y),'clusters':len(unique),'baseline_rmse':float(np.sqrt(e0.mean())),
                'new_rmse':float(np.sqrt(e1.mean())),
                'new_minus_baseline_rmse':float(np.sqrt(e1.mean())-np.sqrt(e0.mean())),
                'difference_ci95':np.percentile(boots,[2.5,97.5]).tolist()}
    primary=(t0>=0)&(t0<=20)
    return {'primary_0_20':subset(primary),
        'bins':{f'{lo}-{hi}':subset(primary&(t0>=lo)&(t0<hi))
                for lo,hi in [(0,5),(5,10),(10,15),(15,20)]},
        'method':'Same coordinates and corrected targets; 2000 approximately 2km spatial-cluster bootstrap resamples, seed42. Exploratory because this test set has already been inspected.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--new-root',type=Path,default=DEFAULT_NEW)
    new_root=parser.parse_args().new_root
    with np.load(BASE/'features/patches.npz') as d:
        coords=d['coords'];splits={'validation':d['val_idx'],'test':d['test_idx']}
    output={}
    for split,idx in splits.items():
        stem='validation' if split=='validation' else 'test'
        a=read(BASE/f'{stem}_predictions.npz')
        b=read(new_root/f'{stem}_predictions.npz')
        output[split]=compare(a,b,coords[idx])
    (new_root/'matched_comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
