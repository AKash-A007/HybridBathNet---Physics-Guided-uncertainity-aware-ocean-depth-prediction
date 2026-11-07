"""Read-only, split-aware diagnostics for the saved nearshore experiment."""
import json
from pathlib import Path
import numpy as np

root = Path('runs/2026-11-05/nearshore_hybrid_v1')
with np.load(root/'features/patches.npz') as d:
    train_y = d['Y'][d['train_idx'], 32, 32].astype(float)
    coords = d['coords']
    val_idx = d['val_idx']
with np.load(root/'validation_predictions.npz') as d:
    target = d['target'].reshape(-1).astype(float)
    pred = d['prediction'].reshape(-1).astype(float)
    sigma = d['sigma'].reshape(-1).astype(float)
    z_ext = d['z_ext'].reshape(-1).astype(float)
with np.load(root/'test_predictions.npz') as d:
    test_target = d['target'].reshape(-1).astype(float)
    test_pred = d['prediction'].reshape(-1).astype(float)
    test_sigma = d['sigma'].reshape(-1).astype(float)

def bins(x):
    return {f'{a}-{b}': int(((x >= a) & (x < b)).sum()) for a,b in [(0,5),(5,10),(10,15),(15,20)]}

out = {'train_bin_counts': bins(train_y), 'validation_bin_counts': bins(target),
       'validation': {}, 'n_train':len(train_y), 'n_val':len(target)}
for a,b in [(0,5),(5,10),(10,15),(15,20)]:
    sel = (target >= a) & (target < b)
    e = pred[sel] - target[sel]
    out['validation'][f'{a}-{b}'] = {'n':len(e), 'rmse':float(np.sqrt(np.mean(e**2))),
        'bias':float(np.mean(e)), 'mae':float(np.mean(abs(e))),
        'mean_sigma':float(sigma[sel].mean()),
        'mean_z_ext':float(z_ext[sel].mean()),
        'fraction_target_above_zext':float(np.mean(target[sel] > z_ext[sel]))}
print(json.dumps(out, indent=2))
select=(test_target>=10)&(test_target<15)
error=test_pred[select]-test_target[select]
locations=coords[np.load(root/'features/patches.npz')['test_idx']][select]
groups=(locations+32)//200
unique,inv=np.unique(groups,axis=0,return_inverse=True)
test_bin={'n':len(error),'rmse':float(np.sqrt(np.mean(error**2))),
          'bias':float(np.mean(error)),'median_sigma':float(np.median(test_sigma[select])),
          'cluster_count':len(unique),'clusters':[]}
for j,g in enumerate(unique):
    e=error[inv==j]
    test_bin['clusters'].append({'grid_group':g.tolist(),'n':len(e),
        'rmse':float(np.sqrt(np.mean(e**2))),'bias':float(e.mean()),
        'squared_error_fraction_pct':float(100*np.sum(e**2)/np.sum(error**2))})
test_bin['clusters'].sort(key=lambda row:row['squared_error_fraction_pct'],reverse=True)
print(json.dumps({'test_10_15_descriptive_only':test_bin},indent=2))
