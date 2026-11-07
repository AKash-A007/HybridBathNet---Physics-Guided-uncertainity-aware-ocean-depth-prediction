"""Rebuild each fold's features using fold-training footprints only."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.gebco_oof_common import ROOT,SOURCE,spatial_folds,update
from src.features import load_stack,sun_glint_correction,lyzenga_channels,stumpf_ratio,beer_lambert_depth_prior
from src.preprocess import resample_gebco


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'features/patches.npz') as d:
        coords,y,m,support=(d[k].copy() for k in ('coords','Y','M','native_support'))
        indices={k:d[k].copy() for k in ('train_idx','calib_idx','val_idx','test_idx')}
    folds=spatial_folds(indices['train_idx'],coords+32,support)
    arrays,t,crs=load_stack('runs/2026-11-04/processed/stack.tif')
    elevation=resample_gebco('data/raw/gebco/gebco_subset.nc',t,crs,arrays['gebco_depth'].shape)
    water=(arrays['valid_mask']>.5)&np.isfinite(elevation)&(elevation<0)&(elevation>=-25)
    for band in ('B02','B03','B04','B08'):water &= np.isfinite(arrays[band])
    summaries=[]
    for k,(train,held) in enumerate(folds):
        out=ROOT/f'fold_{k}';out.mkdir(exist_ok=True)
        if (out/'features.npz').exists():
            summaries.append(json.loads((out/'features.json').read_text()));continue
        update('building_fold_features',fold=k,train=len(train),held=len(held))
        fit=np.zeros_like(water)
        for r,c in coords[train]:fit[r:r+64,c:c+64]=True
        fit &= water
        for i in np.r_[held,indices['calib_idx'],indices['val_idx'],indices['test_idx']]:
            r,c=coords[i];assert not fit[r:r+64,c:c+64].any()
        blue,green,red,nir=[arrays[b] for b in ('B02','B03','B04','B08')]
        blue,green,red=sun_glint_correction(blue,green,red,nir,fit)
        rinf=[float(np.percentile(a[fit],2)) for a in (blue,green,red)]
        layers=[blue,green,red,nir,*lyzenga_channels(blue,green,red,*rinf),stumpf_ratio(blue,green),
            beer_lambert_depth_prior(blue,green,fit_mask=fit),water.astype('float32'),
            *[arrays[b+'_std'] for b in ('B02','B03','B04','B08')]]
        kept=np.sort(np.r_[train,held,indices['val_idx']]);lookup={v:i for i,v in enumerate(kept)}
        x=np.empty((len(kept),16,64,64),dtype='float32')
        for b,layer in enumerate(layers):
            for j,i in enumerate(kept):
                r,c=coords[i];x[j,b]=np.nan_to_num(layer[r:r+64,c:c+64],nan=0,posinf=0,neginf=0)
        splits={key:np.array([lookup[i] for i in a],dtype='int64') for key,a in (
            ('train_idx',train),('held_idx',held),('val_idx',indices['val_idx']))}
        np.savez(out/'features.npz',X=x,Y=y[kept],M=m[kept],coords=coords[kept],original_index=kept,**splits)
        meta={'fold':k,'train':len(train),'held':len(held),'validation':len(indices['val_idx']),
            'original_train_indices':train.tolist(),'original_held_indices':held.tolist(),
            'feature_fit':'fold training footprints only; no OOF/calibration/validation/test pixels',
            'native_support_overlap':0,'minimum_centre_gap_pixels':96,'rinf':rinf,
            'dataset_sha256':hashlib.sha256((out/'features.npz').read_bytes()).hexdigest()}
        (out/'features.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');summaries.append(meta)
    (ROOT/'folds.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    update('fold_data_ready',folds=[{k:a[k] for k in ('fold','train','held')} for a in summaries])
    print(json.dumps([{k:a[k] for k in ('fold','train','held')} for a in summaries],indent=2))


if __name__=='__main__':main()
