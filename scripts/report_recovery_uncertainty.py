"""Evaluate full-model ensembles and calibration on independent spatial groups."""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.hybridbathnet import conformal_recalibrate
from scripts.recovery_state import update

ROOT=Path('runs/2026-11-06/recovery_v2/gebco_uq')
SEEDS=(42,1,7)


def load(stem):
    means=[];sigmas=[];target=None
    for seed in SEEDS:
        with np.load(ROOT/f'seed_{seed}'/f'{stem}_predictions.npz') as d:
            if target is None:target=d['target'].reshape(-1).astype('float64')
            np.testing.assert_allclose(target,d['target'].reshape(-1),atol=1e-5)
            means.append(d['prediction'].reshape(-1).astype('float64'))
            sigmas.append(d['sigma'].reshape(-1).astype('float64'))
    means=np.array(means);sigmas=np.array(sigmas)
    mean=means.mean(0)
    aleatoric=np.mean(sigmas**2,axis=0)
    epistemic=np.var(means,axis=0)
    return target,mean,np.sqrt(aleatoric+epistemic),means,sigmas,aleatoric,epistemic


def metrics(target,mean,sigma,groups,q90,member_means,member_sigmas):
    mask=(target>=0)&(target<=20)
    def one(selected):
        n=int(selected.sum())
        if not n:return {'n':0}
        y,p,s=target[selected],mean[selected],sigma[selected]
        error=p-y
        logs=-.5*np.log(2*np.pi*member_sigmas[:,selected]**2)-.5*(member_means[:,selected]-y)**2/member_sigmas[:,selected]**2
        return {'n':n,'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(np.abs(error))),
            'mean_sigma_m':float(s.mean()),'coverage90_raw_pct':float(100*np.mean(np.abs(error)<=1.645*s)),
            'coverage90_calibrated_pct':float(100*np.mean(np.abs(error)<=q90*s)),
            'raw_interval90_mean_width_m':float(np.mean(2*1.645*s)),
            'calibrated_interval90_mean_width_m':float(np.mean(2*q90*s)),
            'mixture_nll':float(-np.mean(logsumexp(logs,axis=0)-np.log(len(member_means)))),
            'sigma_error_spearman':float(spearmanr(s,np.abs(error)).statistic)}
    result={'primary_0_20':one(mask),
            'bins':{f'{lo}-{hi}':one(mask&(target>=lo)&(target<hi))
                    for lo,hi in ((0,5),(5,10),(10,15),(15,20.0001))},
            'cutoffs':{str(c):one((target>=0)&(target<=c)) for c in (15,20,25)}}
    covered=np.abs(mean-target)<=q90*sigma
    unique=np.unique(groups[mask])
    result['primary_group90_all_pixels_covered_pct']=float(100*np.mean([covered[mask&(groups==g)].all() for g in unique]))
    result['primary_spatial_groups']=len(unique)
    unique=np.unique(groups[mask]);rng=np.random.RandomState(42);boot=[]
    ids=np.flatnonzero(mask)
    for _ in range(2000):
        idx=np.concatenate([ids[groups[ids]==g] for g in rng.choice(unique,len(unique),replace=True)])
        boot.append(np.sqrt(np.mean((mean[idx]-target[idx])**2)))
    result['primary_0_20']['rmse_ci95_spatial']=np.percentile(boot,[2.5,97.5]).tolist()
    return result


def main():
    if not all((ROOT/f'seed_{seed}'/'complete.json').exists() for seed in SEEDS):
        raise RuntimeError('All predefined members must complete before calibration/test comparison')
    with np.load(ROOT/'features/patches.npz') as d:
        coords=d['coords']+32
        idxs={stem:d[key] for stem,key in (('calibration','calib_idx'),('validation','val_idx'),('test','test_idx'))}
    predictions={stem:load(stem) for stem in idxs}
    report={'dataset':'new447 training/123 independent calibration/208 validation/343 fixed test centres',
        'seeds':list(SEEDS),'models':{},'confidence':.90,
        'calibration':'15 approximate2km groups; maximum normalized error in each group; finite-sample group conformal rank',
        'assumptions':'Coverage guarantees require exchangeable groups; geographic shift/GEBCO reference error remain.',
        'intervals':'symmetric q*sigma; raw uses1.645; calibration changes intervals only, not depth predictions',
        'historical_v1_comparison':'Same test coordinates/targets, but v1 used637 training centres and different feature fitting; not an isolated head ablation.'}
    for label,n_members in (('single_seed42',1),('ensemble3',3)):
        tc,_,_,mc,sc,_,_=predictions['calibration']
        mc,sc=mc[:n_members],sc[:n_members]
        mean_c=mc.mean(0);sigma_c=np.sqrt(np.mean(sc**2,0)+np.var(mc,axis=0))
        _,calib_groups=np.unique(coords[idxs['calibration']]//200,axis=0,return_inverse=True)
        score=np.abs(mean_c-tc)/np.maximum(sigma_c,1e-6)
        maxima=np.array([score[calib_groups==g].max() for g in np.unique(calib_groups)])
        q90=conformal_recalibrate(np.ones_like(maxima),maxima,.90)
        row={'q90':q90,'calibration_groups':len(maxima),'calibration_group_maxima':maxima.tolist()}
        for stem in ('validation','test'):
            target,_,_,means,sigmas,_,_=predictions[stem]
            means,sigmas=means[:n_members],sigmas[:n_members]
            mean=means.mean(0);aleatoric=np.mean(sigmas**2,axis=0);epistemic=np.var(means,axis=0)
            sigma=np.sqrt(aleatoric+epistemic)
            _,groups=np.unique(coords[idxs[stem]]//200,axis=0,return_inverse=True)
            row[stem]=metrics(target,mean,sigma,groups,q90,means,sigmas)
            np.savez(ROOT/f'{label}_{stem}_predictions.npz',target=target,prediction=mean,sigma=sigma,
                aleatoric_variance=aleatoric,epistemic_variance=epistemic,
                lower90=mean-q90*sigma,upper90=mean+q90*sigma)
        report['models'][label]=row
    path=ROOT/'uncertainty_comparison.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Full-model uncertainty recovery study','',report['dataset']+'. All models retain encoder/depth/physics/uncertainty.',
        'The spatial variance head uses detached shared features/mean and separate gradient clipping. The15 calibration groups were excluded from training and all feature fitting. Validation selects EMA checkpoints; calibration is used only after every member finishes.', '',
        '| Model | Validation RMSE | Test RMSE | Test raw90% coverage | Test calibrated90% coverage | Raw/calibrated mean interval width |',
        '|---|---:|---:|---:|---:|---:|']
    for label,row in report['models'].items():
        v=row['validation']['primary_0_20'];t=row['test']['primary_0_20']
        lines.append(f"| {label} | {v['rmse']:.3f}m | {t['rmse']:.3f}m | {t['coverage90_raw_pct']:.1f}% | {t['coverage90_calibrated_pct']:.1f}% | {t['raw_interval90_mean_width_m']:.2f}/{t['calibrated_interval90_mean_width_m']:.2f}m |")
    lines+=['','| Ensemble test bin | n | RMSE | Raw90% coverage | Calibrated90% coverage | Calibrated mean width |',
            '|---|---:|---:|---:|---:|---:|']
    for label,row in report['models']['ensemble3']['test']['bins'].items():
        lines.append(f"| {label}m | {row['n']} | {row['rmse']:.3f}m | {row['coverage90_raw_pct']:.1f}% | {row['coverage90_calibrated_pct']:.1f}% | {row['calibrated_interval90_mean_width_m']:.2f}m |")
    lines+=['',report['assumptions'],report['historical_v1_comparison'],
        'Calibrated coverage and interval width must be assessed together; interval scaling cannot improve RMSE. GEBCO agreement does not establish independent seabed accuracy. Full primary test set stays included; optical masking remains disabled.',
        '', '[All metrics](uncertainty_comparison.json) | [Dataset audit](features/metadata.json)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    update('uncertainty','complete',seeds=list(SEEDS),ensemble_test=report['models']['ensemble3']['test']['primary_0_20'])
    print(json.dumps(report['models']['ensemble3']['test']['primary_0_20'],indent=2),flush=True)


if __name__=='__main__':main()
