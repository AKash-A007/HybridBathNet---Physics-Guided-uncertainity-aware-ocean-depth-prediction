"""Rebuild corrected features at historical spatial coordinates; preserve holdouts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import rasterio
from src.features import load_stack, extract_patches
from src.preprocess import resample_gebco

ROOT=Path('runs/2026-11-05/joint_v1')


def main():
    out=ROOT/'features'; out.mkdir(parents=True,exist_ok=True)
    if (out/'patches.npz').exists(): raise RuntimeError('Refusing to overwrite joint dataset')
    source=Path('runs/2026-11-04/processed/stack.tif')
    arrays,t,crs=load_stack(str(source))
    arrays['gebco_depth']=resample_gebco('data/raw/gebco/gebco_subset.nc',t,crs,arrays['gebco_depth'].shape)
    valid=(arrays['valid_mask']>.5)&np.isfinite(arrays['gebco_depth'])&(arrays['gebco_depth']<0)
    arrays['valid_mask']=valid.astype('float32')
    with np.load('runs/2026-11-05/improvement/features/patches.npz') as old:
        coords=old['coords']; groups={name:old[name].copy() for name in ('train_idx','val_idx','test_idx')}
    eligible=[]
    for i,(r,c) in enumerate(coords):
        if valid[r+32,c+32] and valid[r:r+64,c:c+64].mean()>=.5: eligible.append(i)
    eligible=set(eligible)
    for name in ('val_idx','test_idx'):
        assert set(groups[name]).issubset(eligible),'Held-out eligibility changed; inspect instead of dropping'
    original_train=len(groups['train_idx'])
    groups['train_idx']=np.array([i for i in groups['train_idx'] if i in eligible])
    fit=np.zeros_like(valid)
    for i in groups['train_idx']:
        r,c=coords[i]; fit[r:r+64,c:c+64]=True
    X,Y,M,allcoords=extract_patches(arrays,['B02','B03','B04','B08'],64,32,fit_mask=fit)
    missing=~np.isfinite(Y)
    if np.any(missing & (M>.5)):
        raise ValueError('Non-finite depth on a supervised water pixel')
    # Native coverage gaps remain excluded by M; fill storage only outside M.
    Y[missing]=0
    lookup={tuple(rc):i for i,rc in enumerate(allcoords)}
    keep=sorted(set(np.concatenate(list(groups.values()))))
    indices=np.array([lookup[tuple(coords[i])] for i in keep])
    remap={old:new for new,old in enumerate(keep)}
    np.savez(out/'patches.npz',X=X[indices],Y=Y[indices],M=M[indices],coords=coords[keep],
             **{name:np.array([remap[int(i)] for i in idx]) for name,idx in groups.items()})
    meta={'split_sizes':{name:len(idx) for name,idx in groups.items()},
          'excluded_training_by_optical_water_eligibility':original_train-len(groups['train_idx']),
          'heldout_coordinates_unchanged':True,'feature_fit':'training footprints only',
          'labels':'native GEBCO direct interpolation','image_grid':'historical fixed grid; no resampling of imagery',
          'note':'This is supervised joint data. Separate Sentinel-only builder never reads GEBCO.'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(meta,flush=True)


if __name__=='__main__': main()
