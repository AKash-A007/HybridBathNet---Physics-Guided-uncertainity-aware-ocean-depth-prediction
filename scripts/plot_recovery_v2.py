"""Export recovery comparisons; each panel has its own reference/test set."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('runs/2026-11-06/recovery_v2')


def main():
    gebco=json.loads((ROOT/'gebco_uq/historical_reference_comparison.json').read_text())['bins']
    atl=json.loads((ROOT/'atl24_paired/comparison.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    x=np.arange(4);labels=['0–5','5–10','10–15','15–20']
    axes[0].bar(x-.18,[a['original_rmse'] for a in gebco.values()],.36,label='Preserved full v1')
    means=np.array([a['ensemble_rmse'] for a in gebco.values()])
    ci=np.array([a['ensemble_rmse_ci95_spatial'] for a in gebco.values()])
    axes[0].bar(x+.18,means,.36,label='Three-seed ensemble',yerr=np.stack([means-ci[:,0],ci[:,1]-means]),capsize=3)
    axes[0].set_title('GEBCO: same 228 test centres')
    axes[0].set_xticks(x,[f'{a}\n(n={r["n"]})' for a,r in zip(labels,gebco.values())])
    for offset,key,label in ((-.18,'composite_control','Composite control'),(.18,'matched_tide','Matched/tide')):
        bins=atl[key]['test']['primary_bins']
        axes[1].bar(x+offset,[a['rmse'] for a in bins.values()],.36,label=label)
    axes[1].set_title('Official ATL24 V002: same 171 test centres')
    axes[1].set_xticks(x,[f'{a}\n(n={r["n"]})' for a,r in zip(labels,bins.values())])
    for ax in axes:
        ax.set_xlabel('Reference depth bin (m)');ax.set_ylabel('Test RMSE (m)');ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Full HybridBathNet recovery comparisons — exploratory held-out results',fontsize=12)
    fig.savefig(ROOT/'depth_bin_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for arm in ('composite_control','matched_tide'):
        h=json.loads((ROOT/f'atl24_paired/{arm}/checkpoints/training_history.json').read_text())
        ax.plot([a['epoch'] for a in h],[a['val_rmse'] for a in h],label=arm)
        best=min(h,key=lambda a:a['val_rmse']);ax.scatter(best['epoch'],best['val_rmse'])
    ax.set(xlabel='Epoch',ylabel='Validation RMSE (m)',title='Official ATL24: validation-only checkpoint selection')
    ax.legend();ax.grid(alpha=.2);fig.savefig(ROOT/'atl24_validation_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
