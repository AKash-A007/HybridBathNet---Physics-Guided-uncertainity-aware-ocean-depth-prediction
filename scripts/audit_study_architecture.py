"""Inspect actual saved model tensors and runtime shapes, without training."""
from pathlib import Path
import sys
import json
import collections
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.hybridbathnet import HybridBathNetModel
from src.oof_uncertainty import SpatialVarianceCorrection
ROOT=Path('reports/2026-11-06_project_study')


def shape(value):
    if isinstance(value,torch.Tensor):return list(value.shape)
    if isinstance(value,(tuple,list)):return [shape(v) for v in value]
    if isinstance(value,dict):return {k:shape(v) for k,v in value.items()}
    return str(type(value).__name__)


def inspect(name,model,forward):
    records={};stages={};handles=[]
    stage_names={'encoder.stem1','encoder.res1','encoder.film1','encoder.pool1','encoder.stem2','encoder.pool2',
        'encoder.stem3','encoder.pool3','encoder.bottleneck','encoder.film_btn','decoder.up3','decoder.dec3',
        'decoder.up2','decoder.dec2','decoder.up1','decoder.dec1','decoder.depth_conv','decoder.logvar_conv',
        'decoder.uncertainty_head','cond_proj','recon_head','image','head'}
    for key,module in model.named_modules():
        if key in stage_names:
            handles.append(module.register_forward_hook(lambda m,i,o,k=key:stages.update({k:{'input':shape(i),'output':shape(o)}})))
        if not key or list(module.children()):continue
        record={'model':name,'module':key,'type':type(module).__name__,'parameters':sum(p.numel() for p in module.parameters(recurse=False)),
            'specification':str(module),'executed':False,'calls':0,'input':None,'output':None}
        records[key]=record
        def hook(m,i,o,k=key):
            records[k].update(executed=True,input=shape(i),output=shape(o),calls=records[k]['calls']+1)
        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():out=forward(model)
    for h in handles:h.remove()
    all_counts=collections.Counter(a['type'] for a in records.values());active=collections.Counter(a['type'] for a in records.values() if a['executed'])
    return {'parameters_registered':sum(p.numel() for p in model.parameters()),
        'parameters_in_executed_leaves':sum(a['parameters'] for a in records.values() if a['executed']),
        'registered_leaf_types':dict(all_counts),'executed_leaf_types':dict(active),'stages':stages,'output_shapes':shape(out),
        'layers':list(records.values())}


def main():
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    reports={};flat=[]
    configs=[('Full v1','runs/2026-11-05/nearshore_hybrid_v1/hybridbathnet.pt','runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz'),
        ('Spatial uncertainty member','runs/2026-11-06/recovery_v2/gebco_uq/seed_42/hybridbathnet.pt','runs/2026-11-06/recovery_v2/gebco_uq/features/patches.npz')]
    for name,checkpoint,data in configs:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);options=saved['protocol'].get('model_options',{})
        model=HybridBathNetModel(in_channels=16,mean=saved['ch_mean'].squeeze(),std=saved['ch_std'].squeeze(),
            mask_aware_physics=True,kd_aggregation='center',physics_conditioning='spatial',**options).eval()
        model.load_state_dict(saved['model'],strict=True)
        with np.load(data) as d:
            idx=d['val_idx'][:1];x=d['X'][idx];mask=d['M'][idx,None]
        normalized=(x-saved['ch_mean'])/saved['ch_std']
        reports[name]=inspect(name,model,lambda m:m(torch.from_numpy(normalized),mask=torch.from_numpy(mask),raw_reflectance=torch.from_numpy(x[:,:3].copy())))
        reports[name]['checkpoint']=checkpoint;flat.extend(reports[name]['layers'])
    saved=torch.load('runs/2026-11-06/gebco_oof_uncertainty_v1/risk_head/best.pt',map_location='cpu',weights_only=False)
    risk=SpatialVarianceCorrection().eval();risk.load_state_dict(saved['model'],strict=True)
    reports['OOF correction head']=inspect('OOF correction head',risk,lambda m:m(torch.zeros(1,9,64,64),torch.zeros(1,4),torch.ones(1)))
    flat.extend(reports['OOF correction head']['layers'])
    (ROOT/'architecture_audit.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    pd.DataFrame(flat).to_csv(ROOT/'layer_inventory.csv',index=False)
    print(json.dumps({name:{k:r[k] for k in ('parameters_registered','parameters_in_executed_leaves','executed_leaf_types')} for name,r in reports.items()},indent=2))


if __name__=='__main__':main()
