"""Inspect extinction consistency and optical reconstruction on held-out locations."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.cnn_baseline import PatchDataset
from src.hybridbathnet import HybridBathNetModel

DATA=Path('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz')
BASE=Path('runs/2026-11-05/nearshore_hybrid_v1')
DEFAULT_NEW=Path('runs/2026-11-05/nearshore_no_ext_v2')


@torch.no_grad()
def audit(root, arrays):
    saved=torch.load(root/'hybridbathnet.pt',map_location='cpu',weights_only=False)
    state=saved['model'];mean=saved['ch_mean'];std=saved['ch_std']
    train=PatchDataset(arrays['X'],arrays['Y'],arrays['M'],arrays['train_idx'],
                       mean=mean,std=std,y_mean=saved['y_mean'],y_std=saved['y_std'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=HybridBathNetModel(in_channels=16,mean=mean.squeeze(),std=std.squeeze(),
        r_inf=state['r_inf_buf'].reshape(-1).numpy(),mask_aware_physics=True,
        kd_aggregation='center',physics_conditioning='spatial').to(device)
    model.load_state_dict(state);model.eval()
    result={}
    for split in ('val_idx','test_idx'):
        ds=PatchDataset(arrays['X'],arrays['Y'],arrays['M'],arrays[split],
            mean=train.mean,std=train.std,y_mean=train.y_mean,y_std=train.y_std)
        errors=[];k=[];al=[];z=[];target=[];pred=[];transmission=[];depth_slope=[]
        for x,y,m,raw in DataLoader(ds,batch_size=8):
            x=x.to(device);m=m.to(device).unsqueeze(1);raw=raw.to(device)
            out=model(x,mask=m,raw_reflectance=raw)
            valid=m.bool().expand_as(raw)
            errors.extend(((out['r_hat']-raw)[valid].float().cpu().numpy()).tolist())
            k.extend(out['kl_map'][valid].float().cpu().numpy().tolist())
            al.extend(out['al_map'][valid].float().cpu().numpy().tolist())
            kc=out['kl_map'][:,:,32,32].float()
            ac=out['al_map'][:,:,32,32].float()
            dc=out['depth'][:,:,32,32].float()
            tc=torch.exp(-2*kc*dc)
            transmission.extend(tc.cpu().numpy().reshape(-1).tolist())
            depth_slope.extend((2*kc*(ac-out['r_inf'].reshape(1,3))*tc).abs().cpu().numpy().reshape(-1).tolist())
            z.extend(out['z_ext'].reshape(-1).float().cpu().numpy().tolist())
            target.extend(y[:,32,32].numpy().tolist())
            pred.extend(out['depth'][:,0,32,32].float().cpu().numpy().tolist())
        errors=np.asarray(errors);k=np.asarray(k);al=np.asarray(al)
        transmission=np.asarray(transmission);depth_slope=np.asarray(depth_slope)
        z=np.asarray(z);target=np.asarray(target);pred=np.asarray(pred)
        primary=(target>=0)&(target<=20)
        result[split]={'reflectance_rmse':float(np.sqrt(np.mean(errors**2))),
            'reflectance_mae':float(np.mean(abs(errors))),
            'attenuation_k_quantiles_per_m':np.quantile(k,[.05,.5,.95]).tolist(),
            'bottom_albedo_quantiles':np.quantile(al,[.05,.5,.95]).tolist(),
            'centre_bottom_signal_transmission_quantiles':np.quantile(transmission,[.05,.5,.95]).tolist(),
            'centre_reconstruction_depth_slope_quantiles_reflectance_per_m':np.quantile(depth_slope,[.05,.5,.95]).tolist(),
            'primary_target_above_heuristic_zext_pct':float(100*np.mean(target[primary]>z[primary])),
            'primary_prediction_above_heuristic_zext_pct':float(100*np.mean(pred[primary]>z[primary])),
            'primary_mean_heuristic_zext_m':float(np.mean(z[primary])),
            'n_primary':int(primary.sum())}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--new-root',type=Path,default=DEFAULT_NEW)
    new_root=parser.parse_args().new_root
    with np.load(DATA) as d:
        arrays={key:d[key] for key in ('X','Y','M','train_idx','val_idx','test_idx')}
    result={}
    for name,root in [('baseline',BASE),('candidate',new_root)]:
        if (root/'hybridbathnet.pt').exists():result[name]=audit(root,arrays)
    new_root.mkdir(parents=True,exist_ok=True)
    (new_root/'physics_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
