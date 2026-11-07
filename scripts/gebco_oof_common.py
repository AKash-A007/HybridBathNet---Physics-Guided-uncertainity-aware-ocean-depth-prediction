"""State and spatial splitting for the predefined GEBCO OOF uncertainty study."""
from pathlib import Path
import datetime
import json
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path('runs/2026-11-06/gebco_oof_uncertainty_v1')
SOURCE=Path('runs/2026-11-06/recovery_v2/gebco_uq')


def update(stage,**extra):
    ROOT.mkdir(exist_ok=True,parents=True)
    state={'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');tmp.replace(ROOT/'status.json')
    start='<!-- OOF_UNCERTAINTY_STATUS -->';end='<!-- /OOF_UNCERTAINTY_STATUS -->'
    note=start+'\nGEBCO held-region uncertainty study: **'+stage+'**. [Status](runs/2026-11-06/gebco_oof_uncertainty_v1/status.json) | [Progress](runs/2026-11-06/gebco_oof_uncertainty_v1/LIVE_PROGRESS.md) | [Report](runs/2026-11-06/gebco_oof_uncertainty_v1/REPORT.md).\n'+end
    for path in (Path('README.md'),Path('PROJECT_CONTEXT.md')):
        text=path.read_text(encoding='utf-8')
        if start in text:
            a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+note+text[b:]
        else:
            a=text.index('\n')+1;text=text[:a]+'\n'+note+'\n'+text[a:]
        path.write_text(text,encoding='utf-8')


def spatial_folds(train,centres,support,n_folds=3,seed=20261107):
    group=centres[train]//200
    unique=np.unique(group,axis=0)
    # Coordinate-only seeded balanced assignment; never uses labels or errors.
    rng=np.random.RandomState(seed);order=rng.permutation(len(unique))
    assignment={tuple(unique[i]):rank%n_folds for rank,i in enumerate(order)}
    fold=np.array([assignment[tuple(g)] for g in group])
    result=[]
    for k in range(n_folds):
        held=train[fold==k];other=train[fold!=k]
        used=set(support[held].ravel())
        kept=np.array([i for i in other if not set(support[i])&used],dtype='int64')
        kept=kept[cKDTree(centres[held]).query(centres[kept],p=np.inf)[0]>=96]
        if len(kept)<80 or len(held)<50:raise ValueError(f'Insufficient fold support:{len(kept)}/{len(held)}')
        assert not set(support[kept].ravel())&used
        assert (cKDTree(centres[held]).query(centres[kept],p=np.inf)[0]>=96).all()
        result.append((kept,held))
    np.testing.assert_array_equal(np.sort(np.concatenate([a[1] for a in result])),np.sort(train))
    return result
