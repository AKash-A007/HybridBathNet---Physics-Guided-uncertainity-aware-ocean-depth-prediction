"""Count optically deep control pixels confined to historical training blocks."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.features import load_stack
from src.preprocess import resample_gebco

stack,transform,crs=load_stack('runs/2026-11-04/processed/stack.tif')
depth=-resample_gebco('data/raw/gebco/gebco_subset.nc',transform,crs,
                      stack['gebco_depth'].shape)
with np.load('runs/2026-11-05/improvement/features/patches.npz') as d:
    rc=d['coords'];train_blocks={tuple(x//128) for x in rc[d['train_idx']]}
valid=stack['valid_mask']>.5
out={'train_block_count':len(train_blocks),'thresholds':{}}
with np.load('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz') as patches:
    optical=patches['X']
    for split in ('train_idx','val_idx','test_idx'):
        idx=patches[split]
        centre_std=optical[idx,12:15,32,32]
        out[split+'_temporal_std_bgr_q05_q50_q95']=np.quantile(centre_std,[.05,.5,.95],axis=0).tolist()
for minimum in (20,25,30,40,50,60,80,100):
    count=0;blocks=0;values=[]
    for br,bc in train_blocks:
        ys=slice(br*128,(br+1)*128);xs=slice(bc*128,(bc+1)*128)
        good=valid[ys,xs]&np.isfinite(depth[ys,xs])&(depth[ys,xs]>=minimum)
        n=int(good.sum());count+=n;blocks+=n>0
        if n:
            values.append(np.stack([stack[b][ys,xs][good] for b in ('B02','B03','B04','B08')],axis=1))
    entry={'pixels':count,'blocks':blocks}
    if values:
        v=np.concatenate(values)
        entry['reflectance_q05_q50_q95']=np.quantile(v,[.05,.5,.95],axis=0).tolist()
    out['thresholds'][str(minimum)]=entry
output=Path('runs/2026-11-05/optical_support_audit_v1')
output.mkdir(parents=True,exist_ok=True)
(output/'deep_controls.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
