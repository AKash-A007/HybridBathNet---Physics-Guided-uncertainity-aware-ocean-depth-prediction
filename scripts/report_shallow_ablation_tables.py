"""Aggregate saved GEBCO ablations and UQ variants; no fits or new training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtri

ROOT=Path('runs/2026-11-06/gebco_shallow_ablation_report_v1')
FULL=Path('runs/2026-11-05/nearshore_hybrid_v1')
ABL=Path('runs/2026-11-06/gebco_component_ablation_v1')
UQ=Path('runs/2026-11-06/recovery_v2/gebco_uq')
OOF=Path('runs/2026-11-06/gebco_oof_uncertainty_v1')


def scores(y,p,sigma,mask,aleatoric=None,epistemic=None):
    y,p=y[mask],p[mask];e=p-y
    r={'n':len(y),'rmse_m':float(np.sqrt(np.mean(e**2))),'mae_m':float(abs(e).mean()),'bias_m':float(e.mean()),
        'mean_sigma_m':None,'coverage90_pct':None,'width90_m':None,'aleatoric_sigma_mean_m':None,'epistemic_sigma_mean_m':None}
    if sigma is not None:
        s=sigma[mask];assert np.isfinite(s).all() and (s>0).all()
        r.update(mean_sigma_m=float(s.mean()),coverage90_pct=float(100*np.mean(abs(e)<=ndtri(.95)*s)),width90_m=float(2*ndtri(.95)*s.mean()))
    for name,value in (('aleatoric',aleatoric),('epistemic',epistemic)):
        if value is not None:r[f'{name}_sigma_mean_m']=float(np.sqrt(value[mask]).mean())
    return r


def domains(y):
    return {'0-5':(y>=0)&(y<5),'5-10':(y>=5)&(y<=10),'0-10':(y>=0)&(y<=10),
        '10-15':(y>10)&(y<15),'15-20':(y>=15)&(y<=20),'0-20':(y>=0)&(y<=20)}


def fmt(v):return 'N/A' if v is None else f'{v:.3f}'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'results.json').exists():raise RuntimeError('Preserve completed report')
    result={'scope':'GEBCO only; requested reference-depth subset0<=z<=10m, no model fitting/masking/depth selection',
        'uncertainty_definition':'mean per-location predicted standard deviation in metres; nominal90% Gaussian intervals',
        'ablations':'Matched historical637-train single-seed arms; uncertainty-off placeholders are invalid, reported null',
        'research':'Separate447-train three-member ensemble and its fixed uncertainty postprocessing; not matched ablations of the latest head',
        'splits':{}}
    flat=[]
    for stem in ('validation','test'):
        rows={};reference=None
        for label,path,uncertain in (
            ('Full HybridBathNet',FULL,True),('No physics',ABL/'no_physics',True),
            ('No uncertainty',ABL/'no_uncertainty',False),('Neither physics nor uncertainty',ABL/'no_both',False)):
            with np.load(path/f'{stem}_predictions.npz') as d:
                y=d['target'].ravel().astype('float64');p=d['prediction'].ravel().astype('float64')
                s=d['sigma'].ravel().astype('float64') if uncertain else None
            if reference is None:reference=y
            np.testing.assert_array_equal(reference,y)
            rows[label]={key:scores(y,p,s,m,aleatoric=s**2 if uncertain else None) for key,m in domains(y).items()}
        result['splits'][stem]={'matched_ablations':rows,'uncertainty_research':{}}
        with np.load(UQ/f'ensemble3_{stem}_predictions.npz') as d:
            y=d['target'].ravel();p=d['prediction'].ravel();s=d['sigma'].ravel();a=d['aleatoric_variance'];e=d['epistemic_variance']
        np.testing.assert_array_equal(reference,y)
        oof=pd.read_csv(OOF/f'{stem}_predictions.csv')
        np.testing.assert_allclose(p,oof.prediction_m.to_numpy(),atol=1e-12)
        # CSV text round-tripping can differ by a few float64 ulps.
        np.testing.assert_allclose(y,oof.target_m.to_numpy(),rtol=0,atol=1e-12)
        cal=json.loads(Path('runs/2026-11-06/gebco_uncertainty_calibration_v2/calibration_frozen.json').read_text())
        factor=cal['methods']['gaussian_scale']['gaussian_scale']
        for label,sigma,raw in (
            ('Three-member ensemble, raw',s,True),('Ensemble + global calibration',s*factor,False),
            ('Ensemble + OOF risk head',oof.oof_head_raw_sigma_m.to_numpy(),False),
            ('Ensemble + OOF head + calibration',oof.oof_head_scaled_sigma_m.to_numpy(),False)):
            result['splits'][stem]['uncertainty_research'][label]={key:scores(y,p,sigma,m,aleatoric=a if raw else None,epistemic=e if raw else None) for key,m in domains(y).items()}
        for study,models in result['splits'][stem].items():
            for model,summary in models.items():
                for domain,metrics in summary.items():flat.append({'split':stem,'study':study,'model':model,'depth_domain_m':domain,**metrics})
    lines=['# GEBCO ablation and uncertainty tables — shallow0–10m','',
        'No retraining or recalibration. Scores are recomputed from saved prediction files, using reference depth0<=z<=10m. The overall0–20m benchmark remains preserved. Validation and test remain separate.',
        'Uncertainty means average predicted per-location standard deviation (SD), not RMSE or a guaranteed error bound. Interval width is the full lower-to-upper90% Gaussian range. Uncertainty-off arms have no valid sigma; N/A never means zero error.',
        '', '## Matched component ablations','',
        'All four arms use the same637 training points, fixed validation/test targets, seed42 and input tensor shape. Physics/uncertainty configurations differ as specified in the original ablation protocol. Single-model epistemic uncertainty was not estimated.', '']
    for stem in ('validation','test'):
        models=result['splits'][stem]['matched_ablations'];n=next(iter(models.values()))['0-10']['n']
        lines += [f'### {stem},0–10m (n={n})','',
            '| Model | RMSE(m) | MAE(m) | Mean head SD(m) |90% coverage |90% width(m) |','|---|---:|---:|---:|---:|---:|']
        for model,summary in models.items():
            r=summary['0-10'];lines.append(f"| {model} | {r['rmse_m']:.3f} | {r['mae_m']:.3f} | {fmt(r['mean_sigma_m'])} | {fmt(r['coverage90_pct'])} | {fmt(r['width90_m'])} |")
    lines+=['','### Test RMSE by bin and original full domain','',
        '| Model |0–5m(n17) |5–10m(n58) |0–10m(n75) |10–15m(n64) |15–20m(n89) |0–20m(n228) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model,summary in result['splits']['test']['matched_ablations'].items():
        lines.append('| '+model+' | '+' | '.join(f"{summary[key]['rmse_m']:.3f}" for key in ('0-5','5-10','0-10','10-15','15-20','0-20'))+' |')
    lines+=['','## Later ensemble and uncertainty-head experiments','',
        'These use447 training points and123 independent calibration points rather than the original637 training points. They share test labels/locations, but are not matched component ablations. Do not attribute their RMSE differences solely to the physics/uncertainty heads.',
        'Raw aleatoric/epistemic decomposition applies only to the original ensemble. Postprocessed total uncertainty does not identify separately corrected components; original raw component files remain preserved.','']
    for stem in ('validation','test'):
        lines += [f'### {stem},0–10m','',
            '| Variant | n | RMSE(m) | Mean total SD(m) |90% coverage |90% width(m) |','|---|---:|---:|---:|---:|---:|']
        for model,summary in result['splits'][stem]['uncertainty_research'].items():
            r=summary['0-10'];lines.append(f"| {model} | {r['n']} | {r['rmse_m']:.3f} | {r['mean_sigma_m']:.3f} | {r['coverage90_pct']:.2f}% | {r['width90_m']:.3f} |")
    lines+=['','### Raw ensemble decomposition,0–10m','',
        '| Split | n | Mean aleatoric SD(m) | Mean epistemic SD(m) | Mean total SD(m) |','|---|---:|---:|---:|---:|']
    for stem in ('validation','test'):
        r=result['splits'][stem]['uncertainty_research']['Three-member ensemble, raw']['0-10']
        lines.append(f"| {stem} | {r['n']} | {r['aleatoric_sigma_mean_m']:.3f} | {r['epistemic_sigma_mean_m']:.3f} | {r['mean_sigma_m']:.3f} |")
    lines+=['','The component ablations were not rerun with the OOF head. No new matched physics-off/uncertainty-off deep ensembles exist; unavailable epistemic values must not be fabricated. This is an exploratory requested subgroup, not an optical mask or a newly selected reporting cutoff.',
        '', '[All domains/metrics JSON](results.json) | [All tables CSV](all_results.csv) | [Original ablation study](../gebco_component_ablation_v1/REPORT.md) | [OOF uncertainty study](../gebco_oof_uncertainty_v1/REPORT.md)']
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(flat).to_csv(ROOT/'all_results.csv',index=False)
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
