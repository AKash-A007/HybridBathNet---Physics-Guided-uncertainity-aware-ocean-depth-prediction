"""Diagnostic only: can current learned K/A and temporal variation support a mask?"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.cnn_baseline import PatchDataset
from src.hybridbathnet import HybridBathNetModel
from src.optical_limit import bottom_detection_limit

BASE=Path('runs/2026-11-05/nearshore_hybrid_v1')
OUT=Path('runs/2026-11-05/optical_support_audit_v1')


def summary(target,pred,limit,sigma):
    primary=(target>=0)&(target<=20)
    visible=np.isfinite(limit)&(pred<=limit)
    conservative=np.isfinite(limit)&(pred+1.645*sigma<=limit)
    def subset(mask):
        n=int(mask.sum())
        return {'n':n,'coverage_pct':float(100*n/max(1,primary.sum())),
                'rmse_m':float(np.sqrt(np.mean((pred[mask]-target[mask])**2))) if n else None}
    return {'all':subset(primary),'point_mask':subset(primary&visible),
            'upper_depth_mask':subset(primary&conservative),
            'median_limit_m':float(np.nanmedian(limit)),
            'fraction_zero_limit_pct':float(100*np.mean(limit<=0)),
            'n_invalid_limit':int((~np.isfinite(limit)).sum())}


@torch.no_grad()
def main():
    with np.load(BASE/'features/patches.npz') as f:
        data={k:f[k] for k in ('X','Y','M','train_idx','val_idx','test_idx')}
    saved=torch.load(BASE/'hybridbathnet.pt',map_location='cpu',weights_only=False)
    state=saved['model'];mean=saved['ch_mean'];std=saved['ch_std']
    model=HybridBathNetModel(in_channels=16,mean=mean.squeeze(),std=std.squeeze(),
        r_inf=state['r_inf_buf'].reshape(-1).numpy(),mask_aware_physics=True,
        kd_aggregation='center',physics_conditioning='spatial').cuda().eval()
    model.load_state_dict(state)
    result={'assumptions':{'bands':'B02 and B03; either band detects bottom',
        'k':'model learned per-band optical K treated as both downward and upward attenuation; not measured',
        'contrast':'absolute learned A minus train-fitted shallow R_inf; not measured',
        'noise':'per-pixel temporal standard deviation across Sentinel scenes; includes environmental changes, not sensor noise',
        'snr':3,'operationally_calibrated':False},'splits':{}}
    for split in ('val_idx','test_idx'):
        idx=data[split]
        ds=PatchDataset(data['X'],data['Y'],data['M'],idx,mean=mean,std=std,
                        y_mean=saved['y_mean'],y_std=saved['y_std'])
        limits=[];preds=[];sigmas=[];targets=[]
        for start,(x,y,m,raw) in enumerate(DataLoader(ds,batch_size=8)):
            out=model(x.cuda(),mask=m.cuda().unsqueeze(1),raw_reflectance=raw.cuda())
            k=out['kl_map'][:,:2,32,32].float()
            a=out['al_map'][:,:2,32,32].float()
            contrast=(a-out['r_inf'][:,:2,0,0]).abs()
            rows=idx[start*8:start*8+len(x)]
            temporal=torch.from_numpy(data['X'][rows,12:14,32,32].copy()).to(k.device).float()
            limits_bands=bottom_detection_limit(k,k,contrast,temporal,detection_snr=3.)
            z=limits_bands['z_limit_m'].nan_to_num(nan=float('-inf')).max(dim=1).values
            z=torch.where(torch.isfinite(z),z,torch.full_like(z,float('nan')))
            limits.extend(z.cpu().numpy().tolist())
            preds.extend(out['depth'][:,0,32,32].float().cpu().numpy().tolist())
            sigmas.extend(torch.exp(.5*out['log_var'][:,0,32,32].float()).cpu().numpy().tolist())
            targets.extend(y[:,32,32].numpy().tolist())
        target=np.asarray(targets);pred=np.asarray(preds);sigma=np.asarray(sigmas);limit=np.asarray(limits)
        result['splits'][split]=summary(target,pred,limit,sigma)
        np.savez(OUT/f'{split}_flags.npz',target=target,prediction=pred,sigma=sigma,
                 diagnostic_optical_limit=limit,operationally_calibrated=np.array(False))
    (OUT/'diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    main()
