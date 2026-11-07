"""Read saved results and validation predictions; never train or change checkpoints."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.cnn_baseline import DepthStratifiedSampler

ROOT = Path('runs/2026-11-05/improvement')
OLD = Path('runs/2026-11-04/study')


def main():
    rows=[]
    for study in (OLD, ROOT):
        for folder in study.iterdir():
            path=folder/'checkpoints/training_history.json'
            if not path.exists():
                continue
            history=json.loads(path.read_text())
            best=min(history,key=lambda r:r['val_rmse'])
            rows.append({'study':str(study),'experiment':folder.name,'epochs':history[-1]['epoch'],
                'best_epoch':best['epoch'],'best_validation_rmse':best['val_rmse'],
                'best_first_90':min(r['val_rmse'] for r in history if r['epoch']<=90),
                'best_last_30':min(r['val_rmse'] for r in history[-30:]),
                'finite_history':all(np.isfinite(r[k]) for r in history for k in ('loss','val_rmse','val_mae','val_r2'))})
    with np.load(ROOT/'expanded_smooth/results/validation_predictions.npz') as a, np.load(ROOT/'selected_seed_1/results/validation_predictions.npz') as b:
        np.testing.assert_array_equal(a['target'],b['target'])
        error=a['prediction']-a['target']
        diagnostics={'validation_rmse':float(np.sqrt(np.mean(error**2))),
            'validation_bias_m':float(error.mean()),
            'top_5_share_squared_error':float(np.sort(error**2)[-5:].sum()/(error**2).sum()),
            'two_seed_equal_weight_validation_rmse':float(np.sqrt(np.mean(((a['prediction']+b['prediction'])/2-a['target'])**2))),
            'seed_residual_correlation':float(np.corrcoef(error,b['prediction']-b['target'])[0,1]),
            'depth_bins':[]}
        for lo,hi in ((0,10),(10,20),(20,30),(30,1000)):
            mask=(a['target']>=lo)&(a['target']<hi)
            diagnostics['depth_bins'].append({'lower_m':lo,'upper_m':hi,'n':int(mask.sum()),
                'rmse':float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None,
                'bias':float(error[mask].mean()) if mask.any() else None})
    with np.load(ROOT/'features/patches.npz') as data:
        depths=data['Y'][data['train_idx'],32,32]
        sampler=DepthStratifiedSampler(depths,seed=42)
        drawn=list(sampler)
        diagnostics['sampler']={'training_patches':len(depths),'epoch_draws':len(drawn),
            'unique_draws_seed_42_epoch_1':len(set(drawn)), 'stratum_sizes':[len(s) for s in sampler.strata],
            'steps_per_epoch_batch_8':int(np.ceil(len(drawn)/8))}
    output={'histories':rows,'diagnostics':diagnostics}
    (ROOT/'review_evidence.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for name in ('long_original','expanded_restart','expanded_smooth','selected_seed_1'):
        hist=json.loads((ROOT/name/'checkpoints/training_history.json').read_text())
        ax.plot([r['epoch'] for r in hist],[r['val_rmse'] for r in hist],label=name)
    ax.set(xlabel='Epoch',ylabel='Validation RMSE (m)',ylim=(0,12),title='Completed improvement study (early values above 12 m omitted)')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(ROOT/'review_validation_curves.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
