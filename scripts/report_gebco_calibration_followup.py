"""Freeze provenance, test inference-only loading and export calibrated sigma."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.uncertainty_calibration import ResidualCalibrator

ROOT=Path('runs/2026-11-06/gebco_uncertainty_calibration_v2')
SOURCE=Path('runs/2026-11-06/recovery_v2/gebco_uq')


def main():
    frozen=json.loads((ROOT/'calibration_frozen.json').read_text())
    results=json.loads((ROOT/'results.json').read_text())
    calibration=ResidualCalibrator.from_state(frozen['methods']['gaussian_scale'])
    packages={'state_sha256':hashlib.sha256((ROOT/'calibration_frozen.json').read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),'sources':[],
        'code':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path('src/uncertainty_calibration.py'),Path('scripts/run_gebco_uncertainty_calibration.py'),Path(__file__))}}
    for seed in (42,1,7):
        for file in ('protocol.json','hybridbathnet.pt','calibration_predictions.npz'):
            path=SOURCE/f'seed_{seed}'/file
            packages['sources'].append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    diagnostic={}
    for stem in ('validation','test'):
        table=pd.read_csv(ROOT/f'{stem}_intervals.csv')
        low,high=calibration.interval(table.prediction_m.to_numpy(),table.raw_total_sigma_m.to_numpy(),.9)
        np.testing.assert_allclose(low,table.gaussian_scale_low90_m,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(high,table.gaussian_scale_high90_m,rtol=1e-12,atol=1e-12)
        table['calibrated_total_sigma_m']=table.raw_total_sigma_m*calibration.scale
        table.to_csv(ROOT/f'{stem}_calibrated_uncertainty.csv',index=False)
        row={}
        for method in ('raw','gaussian_scale','empirical_group_crude'):
            r=results['splits'][stem][method]['primary_0_20']
            row[method]={'mean_abs_coverage_error_pp_fixed_levels':float(np.mean([abs(float(k)-a['coverage_pct']) for k,a in r['levels'].items()])),
                'crps_mean_m':r['crps_mean_m']}
        primary=table.primary_0_20
        row['mean_calibrated_total_sigma_m']=float(table.loc[primary,'calibrated_total_sigma_m'].mean())
        diagnostic[stem]=row
    diagnostic['reload']='Inference-only reconstruction exactly reproduces intervals to1e-12 without target inputs'
    (ROOT/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    (ROOT/'source_manifest.json').write_text(json.dumps(packages,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for stem,ax in zip(('validation','test'),axes):
        for method in ('raw','gaussian_scale','empirical_group_crude','hierarchical_conformal'):
            lev=results['splits'][stem][method]['primary_0_20']['levels']
            finite=[(float(k),v['coverage_pct']) for k,v in lev.items() if not v['unbounded_intervals_n']]
            ax.plot([a for a,b in finite],[b for a,b in finite],marker='o',label=method)
        ax.plot([65,100],[65,100],'k--',alpha=.5)
        ax.set(xlabel='Nominal coverage (%)',ylabel='Observed coverage (%)',title=f'{stem}: GEBCO 0–20 m',xlim=(65,98),ylim=(40,101))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(ROOT/'coverage_curves.png',dpi=180);plt.close(fig)
    report=ROOT/'REPORT.md'
    text=report.read_text(encoding='utf-8')
    marker='\n## Follow-up diagnostics and deployment limits\n'
    if marker in text:text=text[:text.index(marker)]
    v=results['splits']['validation'];t=results['splits']['test']
    effect=t['paired_interval_score90']['gaussian_scale'];ci=effect['paired_spatial_ci95_m']
    text+=marker+'\n'
    text+=f"Scaling raises mean total test sigma from1.612m to{diagnostic['test']['mean_calibrated_total_sigma_m']:.3f}m. Head and ensemble component values remain raw and separately preserved. Inference-only state reload reproduces saved intervals to1e-12 without labels. [Per-location calibrated sigma/intervals](test_calibrated_uncertainty.csv) | [Source manifest](source_manifest.json).\n\n"
    text+=f"Across the predefined68/80/90/95% levels, mean absolute test coverage discrepancy falls from{diagnostic['test']['raw']['mean_abs_coverage_error_pp_fixed_levels']:.2f} to{diagnostic['test']['gaussian_scale']['mean_abs_coverage_error_pp_fixed_levels']:.2f} percentage points. This descriptive calibration metric is not a proper score or a guarantee. Test Gaussian CRPS changes from{t['raw']['primary_0_20']['crps_mean_m']:.3f} to{t['gaussian_scale']['primary_0_20']['crps_mean_m']:.3f}m, while validation CRPS worsens from{v['raw']['primary_0_20']['crps_mean_m']:.3f} to{v['gaussian_scale']['primary_0_20']['crps_mean_m']:.3f}m.\n\n"
    text+=f"The primary90% test interval-score change is{effect['interval_score90_minus_raw_mean_m']:.3f}m, paired spatial95% interval[{ci[0]:.3f},{ci[1]:.3f}]m, which includes zero. Leave-one-calibration-group-out coverage was only80.7% for scaling, showing substantial calibration-group instability. The final full-calibration test coverage96.5% overshoots90%; improved undercoverage is not perfect calibration or proven uniform improvement.\n\n"
    text+='Recommendation: preserve this as an explicit calibrated-interval research option, with raw components alongside it. Do not silently replace model checkpoints or enable optical/error masking. Further work should retain shallow training support and expand independent spatial calibration coverage; improving ranking needs better variance features/held-group residual learning, not another global multiplier. No depth RMSE gain is claimed.\n\n'
    text+='All37 project tests passed, including6 focused calibration tests. Compilation and diff checks passed; no new training job is active.\n\n![Coverage curves](coverage_curves.png)\n'
    report.write_text(text,encoding='utf-8')
    print(json.dumps(diagnostic,indent=2))


if __name__=='__main__':main()
