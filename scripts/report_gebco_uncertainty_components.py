"""Report variance-head aleatoric and independent-ensemble epistemic components.

No training, fitting, recalibration, masking or model selection is performed.
Population variance (1/M) describes the equally weighted predictive mixture;
it is not the standard error of its mean and must not be divided by M again.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SOURCE=Path('runs/2026-11-06/recovery_v2/gebco_uq')
OUT=Path('runs/2026-11-06/gebco_uncertainty_decomposition_v1')
SEEDS=(42,1,7)


def summarize(y,mean,aleatoric,epistemic,mask):
    if not mask.any():return {'n':0}
    a,e=aleatoric[mask],epistemic[mask]
    total=a+e;error=mean[mask]-y[mask]
    result={'n':int(mask.sum()),'rmse_m':float(np.sqrt(np.mean(error**2))),
        'raw_gaussian90_coverage_pct':float(100*np.mean(abs(error)<=1.645*np.sqrt(total))),
        'epistemic_fraction_of_aggregate_variance_pct':float(100*e.sum()/total.sum())}
    for name,v in (('aleatoric',a),('epistemic',e),('total',total)):
        sigma=np.sqrt(v)
        result[name]={'mean_variance_m2':float(v.mean()),'mean_sigma_m':float(sigma.mean()),
            'rms_sigma_m':float(np.sqrt(v.mean())),'median_sigma_m':float(np.median(sigma)),
            'p10_p90_sigma_m':np.percentile(sigma,[10,90]).tolist()}
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'components.json'
    if path.exists():raise RuntimeError('Completed decomposition exists; preserve it instead of overwriting')
    protocols=[json.loads((SOURCE/f'seed_{s}/protocol.json').read_text()) for s in SEEDS]
    hashes={p['dataset_sha256'] for p in protocols}
    assert len(hashes)==1 and all(p['seed']==s for s,p in zip(SEEDS,protocols))
    assert hashlib.sha256((SOURCE/'features/patches.npz').read_bytes()).hexdigest() in hashes
    report={'source':str(SOURCE),'seeds':list(SEEDS),'members':len(SEEDS),'dataset_sha256':next(iter(hashes)),
        'scope':'Existing full GEBCO models; 447 training/123 calibration/208 validation/343 test; no new training or selection',
        'formula':{'mean':'mean_m(mu_m)','aleatoric_variance':'mean_m(sigma_m**2)',
            'epistemic_variance':'mean_m((mu_m-ensemble_mean)**2), ddof=0',
            'total_variance':'aleatoric_variance+epistemic_variance'},
        'interpretation':'Head variance is a model-based estimate of conditional residual noise against GEBCO, not measured pure sensor noise. Ensemble disagreement is an approximation to epistemic uncertainty; three members may miss shared bias.',
        'units':'variances in m^2; square roots in m. Reported mean sigmas are averaged after per-point decomposition; their squares do not add after averaging. RMS sigmas do combine in quadrature.',
        'calibration':'All components are raw, before conformal scaling. Interval calibration does not identify an aleatoric/epistemic attribution.',
        'splits':{}}
    with np.load(SOURCE/'features/patches.npz') as d:
        coords=d['coords']+32;indices={s:d[k] for s,k in (('validation','val_idx'),('test','test_idx'))}
    for stem in indices:
        means=[];sigmas=[];y=None
        for seed in SEEDS:
            with np.load(SOURCE/f'seed_{seed}/{stem}_predictions.npz') as d:
                target=d['target'].reshape(-1).astype('float64')
                if y is None:y=target
                np.testing.assert_array_equal(y,target)
                means.append(d['prediction'].reshape(-1).astype('float64'))
                sigmas.append(d['sigma'].reshape(-1).astype('float64'))
        means=np.array(means);sigmas=np.array(sigmas)
        assert np.isfinite(means).all() and np.isfinite(sigmas).all() and np.all(sigmas>0)
        mean=means.mean(0);a=np.mean(sigmas**2,axis=0);e=np.var(means,axis=0,ddof=0)
        total=a+e
        # Law of total variance, independently evaluated through second moments.
        np.testing.assert_allclose(total,np.mean(sigmas**2+means**2,axis=0)-mean**2,rtol=1e-10,atol=1e-10)
        with np.load(SOURCE/f'ensemble3_{stem}_predictions.npz') as saved:
            np.testing.assert_array_equal(y,saved['target'])
            for key,value in (('prediction',mean),('aleatoric_variance',a),('epistemic_variance',e),('sigma',np.sqrt(total))):
                np.testing.assert_allclose(value,saved[key],rtol=1e-12,atol=1e-12)
        primary=(y>=0)&(y<=20)
        row={'primary_0_20':summarize(y,mean,a,e,primary),'all_saved_points':summarize(y,mean,a,e,np.ones(len(y),dtype=bool)),
             'bins':{f'{lo}-{hi}':summarize(y,mean,a,e,primary&(y>=lo)&(y<hi)) for lo,hi in ((0,5),(5,10),(10,15),(15,20.0001))}}
        report['splits'][stem]=row
        rc=coords[indices[stem]]
        table=pd.DataFrame({'row':rc[:,0],'col':rc[:,1],'reference_depth_m':y,'prediction_m':mean,
            'primary_0_20':primary,'aleatoric_variance_m2':a,'epistemic_variance_m2':e,'total_variance_m2':total,
            'aleatoric_sigma_m':np.sqrt(a),'epistemic_sigma_m':np.sqrt(e),'total_sigma_m':np.sqrt(total),
            'absolute_error_m':abs(mean-y)})
        for i,seed in enumerate(SEEDS):
            table[f'seed_{seed}_depth_m']=means[i];table[f'seed_{seed}_head_sigma_m']=sigmas[i]
        table.to_csv(OUT/f'{stem}_components.csv',index=False)
    lines=['# GEBCO uncertainty-head and deep-ensemble decomposition','',
        'Three independently trained full HybridBathNet models (seeds 42, 1, 7). Physics, depth and uncertainty remain active. This is a read-only analysis of their preserved predictions; no retraining or model promotion.',
        '', 'For each location: aleatoric variance = average head variance; epistemic variance = population variance of member depth means; total variance = their sum. This is the law of total variance for the equal-weight Gaussian mixture. Epistemic disagreement is not divided by the ensemble size a second time.',
        '', '| Split, 0–20 m | n | RMSE (m) | Mean aleatoric SD (m) | Mean epistemic SD (m) | Mean total SD (m) |',
        '|---|---:|---:|---:|---:|---:|']
    for stem,row in report['splits'].items():
        r=row['primary_0_20']
        lines.append(f"| {stem} | {r['n']} | {r['rmse_m']:.3f} | {r['aleatoric']['mean_sigma_m']:.3f} | {r['epistemic']['mean_sigma_m']:.3f} | {r['total']['mean_sigma_m']:.3f} |")
    lines+=['','| Test depth bin (m) | n | RMSE (m) | Mean aleatoric SD (m) | Mean epistemic SD (m) | Mean total SD (m) |',
            '|---|---:|---:|---:|---:|---:|']
    for label,r in report['splits']['test']['bins'].items():
        lines.append(f"| {'15–20' if label.startswith('15-') else label} | {r['n']} | {r['rmse_m']:.3f} | {r['aleatoric']['mean_sigma_m']:.3f} | {r['epistemic']['mean_sigma_m']:.3f} | {r['total']['mean_sigma_m']:.3f} |")
    t=report['splits']['test']['primary_0_20']
    lines+=['',report['units'],
        f"The RMS component SDs on the test set are {t['aleatoric']['rms_sigma_m']:.3f} m aleatoric, {t['epistemic']['rms_sigma_m']:.3f} m epistemic and {t['total']['rms_sigma_m']:.3f} m total. Ensemble disagreement contributes {t['epistemic_fraction_of_aggregate_variance_pct']:.1f}% of aggregate predictive variance.",
        '',report['interpretation'],report['calibration'],
        f"Raw Gaussian moment-based nominal 90% intervals cover only {t['raw_gaussian90_coverage_pct']:.1f}% of primary test locations. Small ensemble disagreement does not establish accuracy: all members can share bias. These values therefore do not certify calibrated true-seabed uncertainty.",
        'The preserved v1 remains the validation-selected GEBCO reference. Its checkpoint is not an ensemble member: it used a different training/calibration allocation and fitted feature references. No models or normalizations were mixed.',
        '', 'Checks: same dataset hash across members and against the prepared NPZ; exact member target/order matching; finite positive sigma; independent second-moment identity; exact agreement with the existing saved ensemble components. Test labels were used only for reporting errors/bins, never fitting or masking.',
        '', '[All metrics](components.json) | [Per-location test values](test_components.csv) | [Validation values](validation_components.csv)',
        '[Original three-model training protocol](../recovery_v2/gebco_uq/seed_42/protocol.json) | [Deep ensembles reference](https://arxiv.org/abs/1612.01474)']
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({s:r['primary_0_20'] for s,r in report['splits'].items()},indent=2))


if __name__=='__main__':main()
