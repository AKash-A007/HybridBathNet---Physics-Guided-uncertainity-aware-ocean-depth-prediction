"""Shallow shelf patches inside existing geographic split blocks, no new test region."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio.transform import xy
from src.features import load_stack,extract_patches
from src.preprocess import resample_gebco

ROOT=Path('runs/2026-11-05/nearshore_hybrid_v1')


def main():
    out=ROOT/'features'; out.mkdir(parents=True,exist_ok=True)
    if (out/'patches.npz').exists(): raise RuntimeError('Refusing to overwrite nearshore data')
    arrays,t,crs=load_stack('runs/2026-11-04/processed/stack.tif')
    elevation=resample_gebco('data/raw/gebco/gebco_subset.nc',t,crs,arrays['gebco_depth'].shape)
    arrays['gebco_depth']=elevation
    shelf=(arrays['valid_mask']>.5)&np.isfinite(elevation)&(elevation<0)&(elevation>=-25)
    for band in ('B02','B03','B04','B08'): shelf&=np.isfinite(arrays[band])
    arrays['valid_mask']=shelf.astype('float32')
    with np.load('runs/2026-11-05/improvement/features/patches.npz') as old:
        oldcoords=old['coords']
        blocks={name:{tuple(rc//128) for rc in oldcoords[old[name]]} for name in ('train_idx','val_idx','test_idx')}
    assert not blocks['train_idx']&blocks['val_idx'] and not blocks['train_idx']&blocks['test_idx']
    coords=[]; depths=[]
    for r in range(0,shelf.shape[0]-63,32):
        for c in range(0,shelf.shape[1]-63,32):
            if r%128+64>128 or c%128+64>128: continue
            if shelf[r+32,c+32] and shelf[r:r+64,c:c+64].mean()>=.5:
                coords.append((r,c)); depths.append(float(-elevation[r+32,c+32]))
    coords=np.array(coords,dtype='int64'); depths=np.array(depths)
    candidates={name:np.array([i for i,rc in enumerate(coords) if tuple(rc//128) in group
                  and (name=='test_idx' or depths[i]<=20)],dtype='int64') for name,group in blocks.items()}
    with xr.open_dataset('data/raw/gebco/gebco_subset.nc') as nc:
        lat=np.sort(nc.latitude.values); lon=np.sort(nc.longitude.values)
    px,py=xy(t,coords[:,0]+32,coords[:,1]+32)
    gx,gy=Transformer.from_crs(crs,'EPSG:4326',always_xy=True).transform(px,py)
    rr=np.clip(np.searchsorted(lat,gy)-1,0,len(lat)-2)
    cc=np.clip(np.searchsorted(lon,gx)-1,0,len(lon)-2)
    supports=[{int((r+dr)*len(lon)+c+dc) for dr in (0,1) for dc in (0,1)} for r,c in zip(rr,cc)]
    # Reference-cell exclusion only removes training/validation candidates; test footprint remains fixed.
    test_support=set().union(*(supports[i] for i in candidates['test_idx']))
    validation=np.array([i for i in candidates['val_idx'] if not supports[i]&test_support],dtype='int64')
    held=np.concatenate([validation,candidates['test_idx']])
    held_support=set().union(*(supports[i] for i in held))
    train=[]
    for i in candidates['train_idx']:
        if supports[i]&held_support: continue
        delta=np.abs(coords[held]-coords[i])
        if np.any((delta[:,0]<96)&(delta[:,1]<96)): continue
        train.append(i)
    groups={'train_idx':np.asarray(train,dtype='int64'),'val_idx':validation,'test_idx':candidates['test_idx']}
    fit=np.zeros_like(shelf)
    for i in groups['train_idx']:
        r,c=coords[i]; fit[r:r+64,c:c+64]=True
    counts={name:len(idx) for name,idx in groups.items()}
    meta={'split_sizes':counts,'candidate_sizes_before_reference_exclusion':{k:len(v) for k,v in candidates.items()},
        'test_region':'unchanged original test block IDs; more distinct coordinates within them',
        'pre_extraction_shelf_max_depth_m':25,'training_and_validation_max_depth_m':20,
        'test_sensitivity_cutoffs_m':[15,20,25],
        'test_20_to_25m':'outside supervised training-depth domain; extrapolation sensitivity only',
        'train_heldout_pixel_buffer':32,'train_heldout_shared_bilinear_source_cells':0,
        'validation_test_shared_bilinear_source_cells':0,'feature_fitting':'training footprints only',
        'patch_size':64,'stride':32,'sample_duplication':False,
        'native_support_cell_counts':{name:len(set().union(*(supports[i] for i in idx))) for name,idx in groups.items()},
        'bin_counts':{name:{f'{lo}-{hi}':int(((depths[idx]>=lo)&(depths[idx]<hi)).sum())
                    for lo,hi in ((0,5),(5,10),(10,15),(15,20),(20,25.0001))} for name,idx in groups.items()}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)
    if counts['train_idx']<100 or counts['val_idx']<20 or counts['test_idx']<20:
        raise RuntimeError('Insufficient unique shallow coordinates for the predeclared pilot gate; inspect audit')
    X,Y,M,extracted=extract_patches(arrays,['B02','B03','B04','B08'],64,32,fit_mask=fit)
    missing=~np.isfinite(Y)
    if np.any(missing&(M>.5)): raise ValueError('Nonfinite supervised target')
    Y[missing]=0
    lookup={tuple(rc):i for i,rc in enumerate(extracted)}
    keep=np.sort(np.concatenate(list(groups.values())))
    indices=np.array([lookup[tuple(coords[i])] for i in keep]); remap={int(i):j for j,i in enumerate(keep)}
    np.savez(out/'patches.npz',X=X[indices],Y=Y[indices],M=M[indices],coords=coords[keep],
             native_support=np.array([sorted(supports[i]) for i in keep]),
             **{name:np.array([remap[int(i)] for i in idx]) for name,idx in groups.items()})


if __name__=='__main__':main()
