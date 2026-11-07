"""Compare saved old/new predictions only at identical shallow test coordinates."""
from pathlib import Path
import json
import numpy as np

ROOT=Path('runs/2026-11-05/nearshore_hybrid_v1')
with np.load('runs/2026-11-05/improvement/features/patches.npz') as d:
    oldcoords=d['coords'][d['test_idx']]
with np.load(ROOT/'features/patches.npz') as d:
    newcoords=d['coords'][d['test_idx']]
with np.load('runs/2026-11-05/improvement/expanded_smooth/results/test_predictions.npz') as d:
    oldpred=d['prediction'].reshape(-1)
with np.load(ROOT/'test_predictions.npz') as d:
    target=d['target'].reshape(-1); newpred=d['prediction'].reshape(-1)
lookup={tuple(rc):i for i,rc in enumerate(oldcoords)}
newidx=np.array([i for i,rc in enumerate(newcoords) if tuple(rc) in lookup and 0<=target[i]<=20])
oldidx=np.array([lookup[tuple(newcoords[i])] for i in newidx])
t,old,new=target[newidx],oldpred[oldidx],newpred[newidx]
_,groups=np.unique((newcoords[newidx]+32)//200,axis=0,return_inverse=True)
u=np.unique(groups); rng=np.random.RandomState(42); diffs=[]
for _ in range(2000):
    idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(u,len(u),replace=True)])
    diffs.append(np.sqrt(np.mean((new[idx]-t[idx])**2))-np.sqrt(np.mean((old[idx]-t[idx])**2)))
result={'n':len(t),'old_model_rmse_same_corrected_targets':float(np.sqrt(np.mean((old-t)**2))),
        'new_model_rmse_same_corrected_targets':float(np.sqrt(np.mean((new-t)**2))),
        'new_minus_old_rmse_ci95':np.percentile(diffs,[2.5,97.5]).tolist(),'clusters':len(u),
        'scope':'common existing test coordinates, corrected reference depth <=20m; no retraining or checkpoint selection',
        'caveat':'Small exploratory matched subset; pipelines/training domains differ. Not an architecture/component ablation or independent truth.'}
(ROOT/'common_test_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
report=ROOT/'REPORT.md'; text=report.read_text(encoding='utf-8')
marker='## Identical-coordinate comparison with the previous selected model'
if marker in text:text=text[:text.index(marker)]
ci=result['new_minus_old_rmse_ci95']
text+='\n'+marker+'\n\n'+f"At {len(t)} common test locations with the same corrected reference depths <=20m, "
text+=f"the previous expanded_smooth model has RMSE {result['old_model_rmse_same_corrected_targets']:.3f} m "
text+=f"and this nearshore model {result['new_model_rmse_same_corrected_targets']:.3f} m. "
text+=f"The 95% paired spatial-bootstrap interval for new-minus-old RMSE is {ci[0]:.3f} to {ci[1]:.3f} m. "
text+='This is a small exploratory subset with differing training pipelines/domains, not proof of architecture superiority.\n'
report.write_text(text,encoding='utf-8')
print(json.dumps(result,indent=2))
