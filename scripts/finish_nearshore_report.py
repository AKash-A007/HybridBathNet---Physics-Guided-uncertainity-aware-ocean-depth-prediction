"""Add primary-domain uncertainty diagnostics and figures after training completes."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import spearmanr
from src.hybridbathnet import calibration_metrics

ROOT=Path('runs/2026-11-05/nearshore_hybrid_v1')


def main():
    state=json.loads((ROOT/'status.json').read_text())
    if state['stage']!='complete':raise RuntimeError('Wait for completed training/evaluation')
    results=json.loads((ROOT/'results.json').read_text())
    uncertainty={}
    for split in ('validation','test'):
        with np.load(ROOT/f'{split}_predictions.npz') as data:
            t,p,s=(data[k].reshape(-1) for k in ('target','prediction','sigma'))
        valid=(t>=0)&(t<=20);t,p,s=t[valid],p[valid],s[valid]
        stats=calibration_metrics(t,p,s)
        stats.update(n=len(t),mean_sigma_m=float(s.mean()),
                     spearman_sigma_depth=float(spearmanr(s,t).statistic),
                     spearman_sigma_absolute_error=float(spearmanr(s,np.abs(t-p)).statistic))
        order=np.argsort(s); risk=[]
        for fraction in (.25,.5,.75,1):
            idx=order[:max(1,int(np.ceil(len(t)*fraction)))]
            risk.append({'fraction':fraction,'n':len(idx),'rmse':float(np.sqrt(np.mean((p[idx]-t[idx])**2)))})
        stats['risk_coverage']=risk
        uncertainty[split]=stats
    (ROOT/'uncertainty_primary_0_20.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8')
    history=json.loads((ROOT/'hybridbathnet/checkpoints/training_history.json').read_text())
    best=min(history,key=lambda e:e['val_rmse'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epochs=[e['epoch'] for e in history]
    axes[0].plot(epochs,[e['val_rmse'] for e in history])
    axes[0].set(xlabel='Epoch',ylabel='Validation RMSE (m)',title='0–20 m checkpoint selection')
    axes[1].plot(epochs,[e['loss'] for e in history],label='Composite training objective')
    axes[1].set(xlabel='Epoch',ylabel='Composite loss',title='Joint optimization')
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/'training_curves.png',dpi=160);plt.close(fig)
    bins=results['test']['primary_bins']; names=[]; rmse=[]; low=[]; high=[]
    for name,values in bins.items():
        if not values['n']:continue
        names.append(f"{name} m\nn={values['n']}");rmse.append(values['rmse'])
        lo,hi=values['rmse_ci95'];low.append(max(0,values['rmse']-lo));high.append(max(0,hi-values['rmse']))
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(names,rmse,yerr=[low,high],capsize=5)
    ax.set(ylabel='Test RMSE (m)',title='Primary 0–20 m domain: 95% spatial bootstrap intervals')
    fig.tight_layout();fig.savefig(ROOT/'test_depth_bins.png',dpi=160);plt.close(fig)
    section=['','## Joint-model training and uncertainty, primary 0–20 m domain','',
             f"Completed {history[-1]['epoch']} epochs; selected EMA epoch {best['epoch']} (validation RMSE {best['val_rmse']:.3f} m).",'',
             '| Split | n | Mean sigma (m) | Gaussian NLL | 68% coverage | 90% coverage |',
             '|---|---:|---:|---:|---:|---:|']
    for split,u in uncertainty.items():
        section.append(f"| {split} | {u['n']} | {u['mean_sigma_m']:.3f} | {u['gaussian_nll']:.3f} | {u['coverage_68pct']:.1f}% | {u['coverage_90pct']:.1f}% |")
    u=uncertainty['test']
    section+=['',f"Test sigma-depth Spearman: {u['spearman_sigma_depth']:.3f}; sigma-absolute-error Spearman: {u['spearman_sigma_absolute_error']:.3f}.",'',
        '| Lowest-uncertainty test fraction retained | n | RMSE (m) |','|---:|---:|---:|']
    for r in u['risk_coverage']:
        section.append(f"| {r['fraction']:.0%} | {r['n']} | {r['rmse']:.3f} |")
    section+=['','No post-hoc sigma calibration or test-dependent clipping was applied. '
        'These diagnostics concern errors against GEBCO, not independent bottom-detectability truth. '
        'No matched model ablation was trained under the HybridBathNet-only instruction, so component-wise accuracy gains are not established.', '',
        '![Training](training_curves.png)','', '![Depth bins](test_depth_bins.png)', '',
        '[Primary uncertainty metrics](uncertainty_primary_0_20.json). Optical-water audit found positive centre NDWI '
        'for all 208 validation and 343 test locations; six training centres were nonpositive. NDWI alone is not '
        'a definitive bottom/land truth, and no post-hoc data filtering was performed.']
    path=ROOT/'REPORT.md';text=path.read_text(encoding='utf-8')
    marker='## Joint-model training and uncertainty, primary 0–20 m domain'
    if marker in text:text=text[:text.index(marker)]
    path.write_text(text+'\n'.join(section)+'\n',encoding='utf-8')
    print(json.dumps({'best_epoch':best['epoch'],'epochs_completed':history[-1]['epoch'],
                      'test':results['test']['primary_0_20'],'uncertainty':u},indent=2))


if __name__=='__main__':main()
