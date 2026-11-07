"""Calibrate the frozen OOF risk head; report on unchanged GEBCO holdouts."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gebco_oof_common import ROOT,SOURCE,update
from scripts.train_gebco_oof_head import CHANNELS,normalize,semantics,ensemble_predictions
from scripts.run_gebco_uncertainty_calibration import metrics
from src.oof_uncertainty import SpatialVarianceCorrection
from src.uncertainty_calibration import ResidualCalibrator,interval_score
LEVELS=(.68,.8,.9,.95)


def gaussian_crps(y,mean,sigma):
    z=(y-mean)/sigma
    return sigma*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))


def main():
    if (ROOT/'results.json').exists():print('OOF study complete; preserve outputs');return
    if not (ROOT/'risk_head/complete.json').exists():raise RuntimeError('Risk head incomplete')
    if not torch.cuda.is_available():raise RuntimeError('Use original CUDA backend')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    ckpt=ROOT/'risk_head/best.pt';saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    model=SpatialVarianceCorrection().cuda().eval();model.load_state_dict(saved['model'],strict=True)
    with np.load(SOURCE/'features/patches.npz') as d:
        x=d['X'];coords=d['coords']+32
        indices={s:d[k] for s,k in (('calibration','calib_idx'),('validation','val_idx'),('test','test_idx'))}

    def source(stem):
        p=ensemble_predictions(stem);idx=indices[stem]
        scalar=semantics(p['prediction'],p['sigma'],p['kd'],p['z_ext'])
        xp,sp=normalize(x[idx][:,CHANNELS],scalar,saved['stats'])
        predicted=[]
        with torch.no_grad():
            for start in range(0,len(idx),32):
                b=np.arange(start,min(start+32,len(idx)))
                lv=model(torch.from_numpy(xp[b]).cuda(),torch.from_numpy(sp[b]).cuda(),torch.from_numpy((p['sigma'][b]**2).astype('float32')).cuda())
                predicted.extend(torch.exp(lv*.5).cpu().numpy().tolist())
        p['risk_sigma']=np.array(predicted,dtype='float64')
        assert np.isfinite(p['risk_sigma']).all() and (p['risk_sigma']>0).all()
        return p

    update('calibrating_oof_risk_head',selected_epoch=saved['epoch'])
    cal=source('calibration');_,cg=np.unique(coords[indices['calibration']]//200,axis=0,return_inverse=True)
    original=ResidualCalibrator('gaussian_scale').fit(cal['target'],cal['prediction'],cal['sigma'],cg)
    learned=ResidualCalibrator('gaussian_scale').fit(cal['target'],cal['prediction'],cal['risk_sigma'],cg)
    prior=json.loads(Path('runs/2026-11-06/gebco_uncertainty_calibration_v2/calibration_frozen.json').read_text())
    np.testing.assert_allclose(original.scale,prior['methods']['gaussian_scale']['gaussian_scale'],atol=1e-10)
    frozen={'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'selected_epoch':saved['epoch'],
        'original':original.state(),'oof_head':learned.state(),
        'scope':'calibration labels only; unchanged depth means; uncertainty correction targets total residual risk'}
    (ROOT/'calibration_frozen.json').write_text(json.dumps(frozen,indent=2,allow_nan=False),encoding='utf-8')
    result={'selected_head_epoch':saved['epoch'],'original_sigma_scale':original.scale,'oof_sigma_scale':learned.scale,
        'source_ensemble':str(SOURCE),'depth_mean_unchanged':True,'uncertainty_source':'raw components preserved; learned total-risk correction',
        'splits':{}}
    for stem in ('validation','test'):
        p=source(stem);y,mean=p['target'],p['prediction'];idx=indices[stem]
        _,g=np.unique(coords[idx]//200,axis=0,return_inverse=True)
        with np.load(SOURCE/f'ensemble3_{stem}_predictions.npz') as prior_pred:
            np.testing.assert_allclose(mean,prior_pred['prediction'],atol=1e-12)
            np.testing.assert_array_equal(y,prior_pred['target'])
        variants={'original_raw':p['sigma'],'original_scaled':p['sigma']*original.scale,
            'oof_head_raw':p['risk_sigma'],'oof_head_scaled':p['risk_sigma']*learned.scale}
        table=pd.DataFrame({'row':coords[idx,0],'col':coords[idx,1],'target_m':y,'prediction_m':mean,
            'primary_0_20':(y>=0)&(y<=20),'raw_aleatoric_variance_m2':p['aleatoric_variance'],'raw_epistemic_variance_m2':p['epistemic_variance']})
        reports={}
        for name,sigma in variants.items():
            from scipy.special import ndtri
            bounds={level:(mean-ndtri((1+level)/2)*sigma,mean+ndtri((1+level)/2)*sigma) for level in LEVELS}
            crps=gaussian_crps(y,mean,sigma)
            def one(mask):
                r=metrics(y[mask],mean[mask],sigma[mask],g[mask],
                    {level:(a[mask],b[mask]) for level,(a,b) in bounds.items()},crps[mask])
                r['sigma_error_spearman']=r.pop('raw_sigma_error_spearman')
                r['gaussian_nll']=float(np.mean(.5*np.log(2*np.pi*sigma[mask]**2)+.5*((y[mask]-mean[mask])/sigma[mask])**2))
                r['mean_sigma_m']=float(sigma[mask].mean())
                return r
            primary=(y>=0)&(y<=20)
            reports[name]={'primary_0_20':one(primary),'all_saved':one(np.ones(len(y),dtype=bool)),
                'bins':{f'{lo}-{hi}':one(primary&(y>=lo)&(y<hi)) for lo,hi in ((0,5),(5,10),(10,15),(15,20.0001))}}
            table[f'{name}_sigma_m']=sigma;table[f'{name}_low90_m'],table[f'{name}_high90_m']=bounds[.9]
        table.to_csv(ROOT/f'{stem}_predictions.csv',index=False)
        mask=(y>=0)&(y<=20);gg=g[mask];unique=np.unique(gg);rng=np.random.RandomState(42)
        old=reports['original_scaled']['primary_0_20'];new=reports['oof_head_scaled']['primary_0_20']
        assert old['rmse_m']==new['rmse_m']
        residual=(y-mean)[mask]
        old_sigma=variants['original_scaled'][mask];new_sigma=variants['oof_head_scaled'][mask]
        z=1.6448536269514722
        score_old=interval_score(y[mask],mean[mask]-z*old_sigma,mean[mask]+z*old_sigma)
        score_new=interval_score(y[mask],mean[mask]-z*new_sigma,mean[mask]+z*new_sigma)
        samples={'interval_score90_difference':[],'width90_difference':[],'new_coverage90':[],'crps_difference':[]}
        crps_diff=gaussian_crps(y[mask],mean[mask],new_sigma)-gaussian_crps(y[mask],mean[mask],old_sigma)
        for _ in range(2000):
            b=np.concatenate([np.flatnonzero(gg==a) for a in rng.choice(unique,len(unique),replace=True)])
            samples['interval_score90_difference'].append(float((score_new[b]-score_old[b]).mean()))
            samples['width90_difference'].append(float((2*z*(new_sigma[b]-old_sigma[b])).mean()))
            samples['new_coverage90'].append(float(100*np.mean(abs(residual[b])<=z*new_sigma[b])))
            samples['crps_difference'].append(float(crps_diff[b].mean()))
        reports['paired_ci95']={key:np.percentile(value,[2.5,97.5]).tolist() for key,value in samples.items()}
        result['splits'][stem]=reports
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# GEBCO spatially held-out residual uncertainty study','',
        'Three fresh full HybridBathNet fold models generate447 OOF error examples. Training/source-cell buffers and feature refitting exclude every point from its own fold fit. A small spatial correction head learns total-risk multipliers from these errors; its checkpoint is selected by external-validation NLL, including the identity epoch0 control.',
        'The final depth means and raw aleatoric/epistemic components come from the original frozen three-member GEBCO ensemble. A separate123-point calibration set fits one sigma multiplier after head selection. No test fitting, depth correction, mask or held-out removal occurs.',
        '',f"Selected risk-head epoch:{saved['epoch']}. Original/new residual sigma calibration multipliers:{original.scale:.6f}/{learned.scale:.6f}.",
        '', '| Variant | Val90% coverage | Val width(m) | Val CRPS | Test90% coverage | Test width(m) | Test CRPS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in variants:
        v=result['splits']['validation'][name]['primary_0_20'];t=result['splits']['test'][name]['primary_0_20']
        lines.append(f"| {name} | {v['levels']['90']['coverage_pct']:.1f}% | {v['levels']['90']['width_mean_m']:.2f} | {v['crps_mean_m']:.3f} | {t['levels']['90']['coverage_pct']:.1f}% | {t['levels']['90']['width_mean_m']:.2f} | {t['crps_mean_m']:.3f} |")
    lines+=['','| Test bin | n | Previous/new90% coverage | Previous/new width(m) | Previous/new interval score90 |',
        '|---|---:|---:|---:|---:|']
    for label,row in result['splits']['test']['oof_head_scaled']['bins'].items():
        a=result['splits']['test']['original_scaled']['bins'][label]['levels']['90'];b=row['levels']['90']
        lines.append(f"| {label} | {row['n']} | {a['coverage_pct']:.1f}/{b['coverage_pct']:.1f}% | {a['width_mean_m']:.2f}/{b['width_mean_m']:.2f} | {a['interval_score_mean_m']:.3f}/{b['interval_score_mean_m']:.3f} |")
    v=result['splits']['validation'];t=result['splits']['test']
    ov=v['original_scaled']['primary_0_20'];nv=v['oof_head_scaled']['primary_0_20']
    gate=(saved['epoch']>0 and nv['levels']['90']['width_mean_m']<ov['levels']['90']['width_mean_m']-1e-4 and
        nv['levels']['90']['coverage_pct']>=90 and
        nv['levels']['90']['interval_score_mean_m']<=ov['levels']['90']['interval_score_mean_m'] and
        nv['crps_mean_m']<=ov['crps_mean_m'])
    result['validation_narrowing_gate_passed']=bool(gate)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines+=['',f"Predefined validation narrowing gate passed:{gate}. It requires narrower90% intervals, at least90% validation coverage, and non-worse validation interval score/CRPS. This is an exploratory assessment, not a coverage guarantee or automatic model promotion."]
    lines+=['',f"Validation/test depth RMSE stays{v['original_scaled']['primary_0_20']['rmse_m']:.3f}/{t['original_scaled']['primary_0_20']['rmse_m']:.3f}m. Primary test has228 fixed centres. Source v1 remains the validation-selected depth reference; this experiment does not promote the research ensemble.",
        f"Test sigma/error Spearman previous/new:{t['original_scaled']['primary_0_20']['sigma_error_spearman']:.3f}/{t['oof_head_scaled']['primary_0_20']['sigma_error_spearman']:.3f}. Paired spatial95% intervals for new-minus-previous scores/width and new coverage are in[results.json](results.json).",
        '', 'Limitations: fold models train on188/236/262 points, less than the final447-point source ensemble; their single-model error distribution differs from the final ensemble. Shared outer validation selects both depth-fold and correction-head checkpoints. Calibration and test remain separate, but the historical test was already inspected, so comparisons are exploratory. GEBCO is a coarse reference, not independent seabed survey truth. Narrowing alone is not improvement if coverage or proper scores degrade. Learned total-risk correction does not identify new pure aleatoric/epistemic components.',
        '', '[Protocol](PROTOCOL.md) | [Source fold split/feature audits](folds.json) | [Head protocol](risk_head/protocol.json) | [Frozen calibration](calibration_frozen.json) | [Per-location test values](test_predictions.csv)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    update('complete',selected_head_epoch=saved['epoch'],test_old_width=t['original_scaled']['primary_0_20']['levels']['90']['width_mean_m'],
        test_new_width=t['oof_head_scaled']['primary_0_20']['levels']['90']['width_mean_m'],
        test_new_coverage=t['oof_head_scaled']['primary_0_20']['levels']['90']['coverage_pct'])
    print('\n'.join(lines[6:14]))


if __name__=='__main__':main()
