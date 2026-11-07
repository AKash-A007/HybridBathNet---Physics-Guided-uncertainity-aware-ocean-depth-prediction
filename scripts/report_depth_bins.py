"""Requested depth-bin diagnostics from saved predictions; never alter samples."""
from pathlib import Path
import json
import csv
import numpy as np

OUT=Path('runs/2026-11-05/depth_bin_review')
SOURCES={
    'selected_validation':Path('runs/2026-11-05/improvement/expanded_smooth/results/validation_predictions.npz'),
    'selected_test':Path('runs/2026-11-05/improvement/expanded_smooth/results/test_predictions.npz'),
    'repeat_seed1_test':Path('runs/2026-11-05/improvement/selected_seed_1/results/test_predictions.npz'),
    'grid_pilot_direct_validation':Path('runs/2026-11-05/grid_audit/pilot_direct/validation_direct.npz'),
}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]; totals={}
    for name,path in SOURCES.items():
        if not path.exists(): continue
        with np.load(path) as d:
            t=d['target'].reshape(-1).astype('float64'); p=d['prediction'].reshape(-1).astype('float64')
            assert len(t)==len(p) and np.isfinite(t).all() and np.isfinite(p).all()
            error=p-t; sq=error**2; overall=float(np.sqrt(sq.mean()))
            totals[name]={'source':str(path),'n':len(t),'overall_rmse':overall,
                'mean_optical_bound':float(d['z_ext'].mean()) if 'z_ext' in d else None}
            if 'z_ext' in d:
                bound=d['z_ext'].reshape(-1)
                totals[name]['heuristic_prediction_support_count']=int((p<=bound).sum())
                totals[name]['heuristic_prediction_support_pct']=float(100*np.mean(p<=bound))
        for lo,hi,label in [(0,5,'0–5'),(5,10,'5–10'),(10,15,'10–15'),(15,float('inf'),'15+')]:
            m=(t>=lo)&(t<hi); n=int(m.sum())
            rows.append({'run':name,'bin_m':label,'n':n,'sample_share_pct':100*n/len(t),
                'rmse_m':float(np.sqrt(sq[m].mean())) if n else None,
                'mae_m':float(np.abs(error[m]).mean()) if n else None,
                'bias_m':float(error[m].mean()) if n else None,
                'squared_error_share_pct':float(100*sq[m].sum()/sq.sum()),'sparse':n<10})
    (OUT/'metrics.json').write_text(json.dumps({'overall':totals,'bins':rows},indent=2),encoding='utf-8')
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines=['# Requested depth-bin RMSE diagnostics','',
        'Bins use reference depths: [0,5), [5,10), [10,15), and [15,infinity) metres. '
        'All saved examples are retained. Small-bin estimates are shown explicitly with counts and are exploratory.', '',
        '| Run | Depth (m) | n | Samples (%) | RMSE (m) | MAE (m) | Bias (m) | Squared error (%) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[f"{r[k]:.3f}" if r[k] is not None else 'unavailable' for k in ('rmse_m','mae_m','bias_m')]
        lines.append(f"| {r['run']} | {r['bin_m']} | {r['n']} | {r['sample_share_pct']:.1f} | {' | '.join(values)} | {r['squared_error_share_pct']:.1f} |")
    lines+=['','## Interpretation','',
        'Squared-error share must be read with sample share: an abundant deep bin can dominate total error without '
        'establishing an abrupt optical cutoff. These bins diagnose error against GEBCO references, not an independently '
        'measured physical observability limit. A one-time aggregate cannot establish that reference quality is irrelevant.', '',
        'Any shallower operating domain should be justified beforehand using physics, measured water properties, '
        'bottom detectability and/or training-region evidence. Report both full-domain and restricted-domain coverage/accuracy '
        'when such a domain is established; never tune the depth cutoff to test scores or imply excluded regions are accurate.', '',
        'The current heuristic optical bound is not a verified exclusion mask. Its average near 6 m is inconsistent with '
        'most reference depths, and the existing signal-ratio/attenuation constants are not calibrated locally. '
        'Masking by that bound now would change the task substantially and could discard usable observations.', '',
        'No data, predictions, reference labels, training indices or test indices were changed by this analysis. '
        'Historical six-channel scalar metrics alone do not permit its per-depth-bin reconstruction.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'overall':totals,'selected_test_bins':[r for r in rows if r['run']=='selected_test']},indent=2))


if __name__=='__main__':main()
