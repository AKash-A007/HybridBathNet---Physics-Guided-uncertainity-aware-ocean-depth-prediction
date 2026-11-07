"""Freeze calibration-only fits, then evaluate predefined GEBCO UQ methods."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.uncertainty_calibration import ResidualCalibrator,group_weights,interval_score

SOURCE=Path('runs/2026-11-06/recovery_v2/gebco_uq')
ROOT=Path('runs/2026-11-06/gebco_uncertainty_calibration_v2')
METHODS=('raw','gaussian_scale','empirical_group_crude','hierarchical_conformal','group_max_conformal')
LEVELS=(.68,.80,.90,.95)


def predictions(stem):
    means=[];sigmas=[];target=None
    for seed in (42,1,7):
        with np.load(SOURCE/f'seed_{seed}/{stem}_predictions.npz') as d:
            y=d['target'].ravel().astype('float64')
            if target is None:target=y
            np.testing.assert_array_equal(target,y)
            means.append(d['prediction'].ravel().astype('float64'));sigmas.append(d['sigma'].ravel().astype('float64'))
    means=np.array(means);sigmas=np.array(sigmas)
    return target,means.mean(0),np.sqrt(np.mean(sigmas**2,axis=0)+np.var(means,axis=0))


def safe_mean(x,weights=None):
    value=float(np.mean(x) if weights is None else np.sum(weights*x))
    return value if np.isfinite(value) else None


def metrics(y,mean,sigma,groups,bounds,crps=None):
    weights=group_weights(groups)
    result={'n':len(y),'groups':len(np.unique(groups)),
        'rmse_m':float(np.sqrt(np.mean((mean-y)**2))),
        'raw_sigma_error_spearman':float(spearmanr(sigma,abs(mean-y)).statistic),'levels':{}}
    for level,(low,high) in bounds.items():
        covered=((y>=low)&(y<=high)).astype(float);width=high-low
        score=interval_score(y,low,high,level)
        result['levels'][str(round(100*level))]={'coverage_pct':float(100*covered.mean()),
            'group_balanced_coverage_pct':float(100*np.sum(weights*covered)),
            'width_mean_m':safe_mean(width),'width_group_balanced_m':safe_mean(width,weights),
            'interval_score_mean_m':safe_mean(score),'interval_score_group_balanced_m':safe_mean(score,weights),
            'unbounded_intervals_n':int((~np.isfinite(width)).sum())}
    result['crps_mean_m']=safe_mean(crps) if crps is not None else None
    result['crps_group_balanced_m']=safe_mean(crps,weights) if crps is not None else None
    return result


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'results.json').exists():raise RuntimeError('Study complete; preserve artifacts')
    if not (ROOT/'PROTOCOL.md').exists():raise RuntimeError('Missing predefined protocol')
    with np.load(SOURCE/'features/patches.npz') as d:
        coords=d['coords']+32;indices={s:d[k] for s,k in (('calibration','calib_idx'),('validation','val_idx'),('test','test_idx'))}
    groups={s:np.unique(coords[idx]//200,axis=0,return_inverse=True)[1] for s,idx in indices.items()}
    cy,cm,cs=predictions('calibration');cg=groups['calibration']
    assert np.all((cy>=0)&(cy<=20)) and len(np.unique(cg))==15
    # Only calibration labels have been read at this point. Every fold excludes
    # a whole group, including all of that group's correlated measurements.
    cv={}
    for method in METHODS:
        bounds={level:(np.empty(len(cy)),np.empty(len(cy))) for level in LEVELS}
        crps=np.empty(len(cy));has_crps=True
        for group in np.unique(cg):
            held=cg==group;fit=~held
            c=ResidualCalibrator(method).fit(cy[fit],cm[fit],cs[fit],cg[fit])
            for level in LEVELS:
                low,high=c.interval(cm[held],cs[held],level);bounds[level][0][held]=low;bounds[level][1][held]=high
            score=c.crps(cy[held],cm[held],cs[held])
            if score is None:has_crps=False
            else:crps[held]=score
        cv[method]=metrics(cy,cm,cs,cg,bounds,crps if has_crps else None)
    fitted={m:ResidualCalibrator(m).fit(cy,cm,cs,cg) for m in METHODS}
    frozen={'source':str(SOURCE),'primary':'gaussian_scale','selection':'predefined; no test-based choice',
        'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
        'calibration_n':len(cy),'groups':15,'methods':{m:c.state() for m,c in fitted.items()},'leave_group_out':cv}
    (ROOT/'calibration_frozen.json').write_text(json.dumps(frozen,indent=2,allow_nan=False),encoding='utf-8')
    # Validation/test files are loaded only after all method states are frozen.
    result={'protocol':'PROTOCOL.md','fitted_states':'calibration_frozen.json','primary':'gaussian_scale',
        'comparison_target':'pointwise coverage; group_max control targets stronger simultaneous group coverage',
        'assumptions':'Hierarchical exchangeability is not established for these fixed geography blocks. Empirical coverage is not a guarantee.',
        'sigma_decomposition':'Raw aleatoric/epistemic values unchanged; calibration of total intervals cannot identify their separate corrections.',
        'depth_predictions':'unchanged; no interval correction is applied to the depth mean or RMSE',
        'splits':{},'checks':{'whole_group_cv':True,'fit_frozen_before_held_predictions':True}}
    for stem in ('validation','test'):
        y,mean,sigma=predictions(stem);g=groups[stem];rc=coords[indices[stem]]
        result['splits'][stem]={}
        table=pd.DataFrame({'row':rc[:,0],'col':rc[:,1],'target_m':y,'prediction_m':mean,'raw_total_sigma_m':sigma,'primary_0_20':(y>=0)&(y<=20)})
        for method,c in fitted.items():
            bounds={level:c.interval(mean,sigma,level) for level in LEVELS};crps=c.crps(y,mean,sigma)
            def one(mask):
                return metrics(y[mask],mean[mask],sigma[mask],g[mask],
                    {level:(low[mask],high[mask]) for level,(low,high) in bounds.items()},crps[mask] if crps is not None else None)
            primary=(y>=0)&(y<=20)
            row={'primary_0_20':one(primary),'all_saved':one(np.ones(len(y),dtype=bool)),
                'bins':{f'{lo}-{hi}':one(primary&(y>=lo)&(y<hi)) for lo,hi in ((0,5),(5,10),(10,15),(15,20.0001))}}
            result['splits'][stem][method]=row
            table[f'{method}_low90_m'],table[f'{method}_high90_m']=bounds[.9]
        table.to_csv(ROOT/f'{stem}_intervals.csv',index=False)
        raw=result['splits'][stem]['raw']['primary_0_20']['rmse_m']
        assert all(r['primary_0_20']['rmse_m']==raw for r in result['splits'][stem].values())
        mask=(y>=0)&(y<=20);yy=y[mask];gg=g[mask]
        unique=np.unique(gg);rng=np.random.RandomState(42)
        bootstrap={}
        for method in METHODS[1:]:
            low=table[f'{method}_low90_m'].to_numpy()[mask];high=table[f'{method}_high90_m'].to_numpy()[mask]
            rawlow=table.raw_low90_m.to_numpy()[mask];rawhigh=table.raw_high90_m.to_numpy()[mask]
            difference=interval_score(yy,low,high,.9)-interval_score(yy,rawlow,rawhigh,.9)
            boot=[]
            for _ in range(2000):
                idx=np.concatenate([np.flatnonzero(gg==a) for a in rng.choice(unique,len(unique),replace=True)])
                boot.append(float(difference[idx].mean()))
            bootstrap[method]={'interval_score90_minus_raw_mean_m':float(difference.mean()),
                'paired_spatial_ci95_m':np.percentile(boot,[2.5,97.5]).tolist()}
        result['splits'][stem]['paired_interval_score90']=bootstrap
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# GEBCO uncertainty calibration: completed comparison','',
        'The full physics/depth/uncertainty networks and three independently trained ensemble members remain frozen. Fits use only123 separate calibration points in15 geographic groups. No training data, test locations, optical masks or depth predictions changed.',
        '', 'Primary candidate: group-balanced Gaussian variance scaling. Other methods and comparison levels were predefined; no test score selects a method.',
        '', '| Method | Validation90% coverage | Validation width (m) | Test90% coverage | Test width (m) | Test interval score90 (m, lower better) |',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        v=result['splits']['validation'][method]['primary_0_20']['levels']['90'];t=result['splits']['test'][method]['primary_0_20']['levels']['90']
        lines.append(f"| {method} | {v['coverage_pct']:.1f}% | {v['width_mean_m']:.2f} | {t['coverage_pct']:.1f}% | {t['width_mean_m']:.2f} | {t['interval_score_mean_m']:.3f} |")
    lines+=['','Mean per-point coverage is shown above; equal-group metrics,68/80/95% sensitivity, Gaussian/empirical CRPS and spatial paired intervals are in[results.json](results.json). HCP95% intervals are unbounded with15 calibration groups, correctly retained rather than silently capped.',
        '', '| Primary scaling: test depth bin | n | Raw90% coverage | Scaled90% coverage | Raw/scaled width (m) |',
        '|---|---:|---:|---:|---:|']
    for label,row in result['splits']['test']['gaussian_scale']['bins'].items():
        raw=result['splits']['test']['raw']['bins'][label]['levels']['90'];t=row['levels']['90']
        lines.append(f"| {label} | {row['n']} | {raw['coverage_pct']:.1f}% | {t['coverage_pct']:.1f}% | {raw['width_mean_m']:.2f}/{t['width_mean_m']:.2f} |")
    factor=fitted['gaussian_scale'].scale
    lines+=['',f"Fitted total-sigma multiplier: {factor:.6f}. It does not separately identify corrections to aleatoric and epistemic uncertainty. Original raw components are preserved.",
        f"Validation/test RMSE stays {result['splits']['validation']['raw']['primary_0_20']['rmse_m']:.3f}/{result['splits']['test']['raw']['primary_0_20']['rmse_m']:.3f}m. Correcting prediction intervals cannot improve point-depth RMSE or the scalar uncertainty ranking.",
        '', 'The prior group-max interval targets simultaneous coverage of a whole group; the new methods target pointwise uncertainty. Narrower intervals alone therefore do not establish a stronger statistical guarantee. HCP uses the published infinity-atom correction; its hierarchical exchangeability assumptions are not verified for this AOI. Scaling and empirical calibration have no distribution-free guarantee here. Previously inspected test results are exploratory.',
        'The CRUDE adaptation uses residual=(observed−predicted)/sigma, so positive errors shift upper quantiles upward. Depth means stay unchanged even if the calibrated uncertainty distribution is asymmetric. CRPS compares raw/scaled Gaussian moment distributions and the empirical calibrated distribution; conformal sets have no assigned CRPS or Gaussian likelihood.',
        '', 'The existing model ranking remains unchanged: the ensemble is a research candidate and preserved v1 remains the validation-selected GEBCO reference. This report improves/assesses interval estimation only.',
        '', '[Predefined protocol and literature](PROTOCOL.md) | [Calibration-only fits and leave-group-out diagnostics](calibration_frozen.json) | [Per-location test intervals](test_intervals.csv)',
        'Checks: numerical tests for analytic scaling, equal-group weights, empirical sign/CRPS, interval score, HCP infinity atom and rank; identical member target order; unchanged RMSE; all fits frozen before new held-out evaluation.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Sigma scale:',factor)
    print('\n'.join(lines[6:14]))


if __name__=='__main__':main()
