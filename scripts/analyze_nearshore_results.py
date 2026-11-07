"""Read-only exploratory cutoff/uncertainty diagnostics; no sample or model changes."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr

OUT=Path('runs/2026-11-05/nearshore_discussion')
BASE=Path('runs/2026-11-05/improvement/expanded_smooth/results')


def metrics(t,p):
    error=p-t
    return {'n':len(t),'rmse':float(np.sqrt(np.mean(error**2))),
            'mae':float(np.abs(error).mean()),'bias':float(error.mean()),
            'mape_pct':float(100*np.mean(np.abs(error)/np.maximum(t,1e-6))),
            'delta_lt_1_25_pct':float(100*np.mean(np.maximum(t/np.maximum(p,1e-6),p/np.maximum(t,1e-6))<1.25))}


def bootstrap(t,p,groups):
    unique=np.unique(groups); rng=np.random.RandomState(42)
    values=[]
    for _ in range(2000):
        chosen=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([np.flatnonzero(groups==group) for group in chosen])
        values.append(float(np.sqrt(np.mean((p[idx]-t[idx])**2))))
    return {'rmse_cluster_bootstrap_95ci':np.percentile(values,[2.5,97.5]).tolist(),
            'clusters':len(unique),'replicates':2000,'small_sample':len(t)<10 or len(unique)<10}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows={}
    with np.load('runs/2026-11-05/improvement/features/patches.npz') as data:
        allcoords=data['coords']; oldtargets=data['Y'][:,32,32]
        indices={name:data[f'{name}_idx'] for name in ('val','test')}
    for split,label in [('val','validation'),('test','test')]:
        with np.load(BASE/f'{label}_predictions.npz') as pred:
            t,p,sigma=(pred[k].reshape(-1).astype('float64') for k in ('target','prediction','sigma'))
        np.testing.assert_allclose(t,oldtargets[indices[split]],rtol=1e-6)
        coords=allcoords[indices[split]]+32
        # Approximately 2 km groups; 1 km groups would place each held-out patch alone.
        pairs=coords//200
        _,groups=np.unique(pairs,axis=0,return_inverse=True)
        bins=[]
        for lo,hi in ((0,5),(5,10),(10,15),(15,float('inf'))):
            m=(t>=lo)&(t<hi)
            bins.append({'depth_bin':[lo,None if np.isinf(hi) else hi],
                         **metrics(t[m],p[m]),**bootstrap(t[m],p[m],groups[m])})
        cuts=[]
        for cutoff in (15,20,25):
            m=(t>=0)&(t<=cutoff)
            cuts.append({'cutoff_m':cutoff,'coverage_pct':float(100*m.mean()),
                         **metrics(t[m],p[m]),**bootstrap(t[m],p[m],groups[m])})
        ranked=np.argsort(sigma)
        risk=[]
        for fraction in (.25,.5,.75,1):
            idx=ranked[:max(1,int(np.ceil(len(t)*fraction))) ]
            risk.append({'retained_fraction':fraction,**metrics(t[idx],p[idx]),
                         'mean_sigma_m':float(sigma[idx].mean())})
        rows[label]={'overall':metrics(t,p),'bins':bins,'cutoffs':cuts,'uncertainty_risk_coverage':risk,
                     'spearman_sigma_depth':float(spearmanr(sigma,t).statistic),
                     'spearman_sigma_absolute_error':float(spearmanr(sigma,np.abs(p-t)).statistic)}
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=['# Nearshore cutoff and uncertainty discussion','',
        'Exploratory analysis of saved expanded_smooth seed-42 predictions; no training, clipping or sample deletion. '
        'Reference-depth cutoffs are not deployable optical masks. All scores use the original GEBCO targets.', '',
        '95% intervals use 2,000 approximately 2 km spatial-cluster bootstrap resamples (seed 42), '
        'not independent-pixel resampling. Intervals are conditional on this model/split, do not include reference '
        'uncertainty and are unreliable for sparse bins/few clusters. No cutoff is selected from these test scores.']
    for split,data in rows.items():
        lines += ['',f'## {split}', '', '| Bin (m) | n | RMSE | 95% cluster CI | MAPE (%) | Delta <1.25 (%) | Clusters |',
                  '|---|---:|---:|---|---:|---:|---:|']
        for b in data['bins']:
            lo,hi=b['depth_bin']; ci=b['rmse_cluster_bootstrap_95ci']
            lines.append(f"| {lo}–{hi if hi is not None else '+'} | {b['n']} | {b['rmse']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {b['mape_pct']:.1f} | {b['delta_lt_1_25_pct']:.1f} | {b['clusters']} |")
        lines+=['','| Maximum reference depth | n | Coverage (%) | RMSE | 95% cluster CI | MAPE (%) |',
                '|---:|---:|---:|---:|---|---:|']
        for b in data['cutoffs']:
            ci=b['rmse_cluster_bootstrap_95ci']
            lines.append(f"| {b['cutoff_m']} | {b['n']} | {b['coverage_pct']:.1f} | {b['rmse']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {b['mape_pct']:.1f} |")
        lines += ['',f"Spearman sigma vs reference depth: {data['spearman_sigma_depth']:.3f}; sigma vs absolute error: {data['spearman_sigma_absolute_error']:.3f}.",'',
                  '| Lowest-uncertainty fraction retained | n | RMSE | Mean sigma (m) |','|---:|---:|---:|---:|']
        for b in data['uncertainty_risk_coverage']:
            lines.append(f"| {b['retained_fraction']:.0%} | {b['n']} | {b['rmse']:.3f} | {b['mean_sigma_m']:.3f} |")
    lines+=['','## What can be claimed','',
        'Uncertainty associated with depth/error and lower error in low-uncertainty subsets would be evidence '
        'of useful ranking on this dataset, not proof that the network detects an invisible seafloor. '
        'Depth, location, bottom type and label quality are confounded. Use within-depth-bin ranking, calibrated '
        'coverage, a geographic holdout and an independent bottom-detectability reference before making that claim.', '',
        'A future four-model table requires Stumpf, RF, CNN and Hybrid trained/evaluated on identical new labels, '
        'spatial groups, optical eligibility and depth bins. Existing scalar summaries or mismatched historical splits '
        'cannot supply that table. The user now authorizes considering limited tabular comparison, but no new training '
        'has been run because training is stopped pending ICESat-2 discussion.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows['test'],indent=2))


if __name__=='__main__':main()
