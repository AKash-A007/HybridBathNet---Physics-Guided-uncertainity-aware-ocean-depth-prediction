"""Reserve spatial calibration groups and refit features on training only."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.features import (load_stack, sun_glint_correction, lyzenga_channels,
                          stumpf_ratio, beer_lambert_depth_prior)
from src.preprocess import resample_gebco

ROOT=Path('runs/2026-11-06/recovery_v2/gebco_uq/features')
SOURCE=Path('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'patches.npz').exists():
        print('Existing calibration dataset; not rebuilding'); return
    with np.load(SOURCE) as d:
        coords,y,m,support=(d[k].copy() for k in ('coords','Y','M','native_support'))
        old={k:d[k].copy() for k in ('train_idx','val_idx','test_idx')}
    centres=coords+32
    train0=old['train_idx']
    groups=centres[train0]//200
    unique=np.unique(groups,axis=0)
    selected=unique[np.random.RandomState(20261106).choice(len(unique),15,replace=False)]
    selected_set=set(map(tuple,selected))
    is_calib=np.array([tuple(group) in selected_set for group in groups])
    calib=train0[is_calib]
    calib_support=set(support[calib].ravel())
    train=np.array([i for i in train0[~is_calib] if not set(support[i])&calib_support])
    train=train[cKDTree(centres[calib]).query(centres[train],p=np.inf)[0]>=96]
    split={'train_idx':train,'calib_idx':calib,'val_idx':old['val_idx'],'test_idx':old['test_idx']}
    assert len(train)>=100 and len(calib)>=50
    for i,left in enumerate(split):
        for right in list(split)[i+1:]:
            assert not set(support[split[left]].ravel())&set(support[split[right]].ravel()),(left,right)
    arrays,t,crs=load_stack('runs/2026-11-04/processed/stack.tif')
    elevation=resample_gebco('data/raw/gebco/gebco_subset.nc',t,crs,arrays['gebco_depth'].shape)
    water=(arrays['valid_mask']>.5)&np.isfinite(elevation)&(elevation<0)&(elevation>=-25)
    for name in ('B02','B03','B04','B08'):
        water &= np.isfinite(arrays[name])
    fit=np.zeros_like(water)
    for r,c in coords[train]:
        fit[r:r+64,c:c+64]=True
    fit &= water
    for key in ('calib_idx','val_idx','test_idx'):
        for r,c in coords[split[key]]:
            assert not fit[r:r+64,c:c+64].any(),key
    blue,green,red,nir=[arrays[k] for k in ('B02','B03','B04','B08')]
    blue,green,red=sun_glint_correction(blue,green,red,nir,fit)
    rinf=[float(np.percentile(a[fit],2)) for a in (blue,green,red)]
    layers=[blue,green,red,nir,*lyzenga_channels(blue,green,red,*rinf),
            stumpf_ratio(blue,green),beer_lambert_depth_prior(blue,green,fit_mask=fit),
            water.astype('float32'),*[arrays[f'{b}_std'] for b in ('B02','B03','B04','B08')]]
    # Extract only the fixed original coordinates, retaining unused train rows
    # for provenance but never putting them into a learning/calibration loader.
    x=np.empty((len(coords),16,64,64),dtype='float32')
    for band,layer in enumerate(layers):
        for i,(r,c) in enumerate(coords):
            x[i,band]=np.nan_to_num(layer[r:r+64,c:c+64],nan=0.,posinf=0.,neginf=0.)
    assert np.isfinite(x).all()
    np.savez(ROOT/'patches.npz',X=x,Y=y,M=m,coords=coords,native_support=support,**split)
    meta={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'split_sizes':{k:len(v) for k,v in split.items()},
          'calibration_group_seed':20261106,'calibration_groups_2km':selected.tolist(),
          'calibration_group_count':15,'selection':'spatial groups selected randomly before candidate training; no labels or residuals used',
          'train_calib_gap_pixels':96,'native_support_disjoint_all_splits':True,
          'validation_test_coordinates_and_targets_identical_to_v1':True,
          'feature_fitting':'new training footprints only; no calibration/validation/test footprint used',
          'rinf':rinf,'input_channels':16,
          'uncertainty_caveat':'Calibration groups need exchangeability for coverage guarantees; fixed geographic shift is not removed.'}
    (ROOT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    main()
