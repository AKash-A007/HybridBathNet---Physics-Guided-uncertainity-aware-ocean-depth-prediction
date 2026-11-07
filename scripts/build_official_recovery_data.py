"""Matched V002 full-model datasets with fixed geography and explicit fallbacks.

Both arms have identical official acquisition-relative targets for reporting.
The matched arm learns scene-water-level depth; predictions are transformed back
to the original ATL24 surface for common-target reporting. Every product-eligible
test point remains, including unavailable/cloudy historical imagery.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.features import load_stack,lyzenga_channels,stumpf_ratio
from scripts.validate_recovery_tides import predict
from scripts.recovery_state import update
ROOT=Path('runs/2026-11-06/recovery_v2')
OUT=ROOT/'atl24_paired'


def far(a,b):
    return cKDTree(b).query(a,p=np.inf)[0]>=96 if len(b) else np.ones(len(a),dtype=bool)


def engineer(raw,masks,centres,source_ids,train):
    # Fit each unique source-scene/grid pixel once, training footprints only.
    ids=[];values=[]
    offsets=np.indices((64,64)).reshape(2,-1).T-32
    for i in train:
        valid=masks[i].ravel()>.5
        rc=centres[i]+offsets
        keys=source_ids[i]*100_000_000+rc[:,0]*10000+rc[:,1]
        ids.append(keys[valid]);values.append(raw[i].reshape(4,-1)[:,valid])
    _,unique=np.unique(np.concatenate(ids),return_index=True)
    fit=np.concatenate(values,axis=1)[:,unique]
    if fit.shape[1]<100:raise ValueError('Too few unique training water pixels')
    nir=fit[3];nmin=float(np.percentile(nir,5));denom=np.sum((nir-nir.mean())**2)
    slopes=np.array([np.sum((v-v.mean())*(nir-nir.mean()))/max(denom,1e-12) for v in fit[:3]])
    corrected=np.maximum(raw[:,:3]-slopes[None,:,None,None]*(raw[:,3:4]-nmin),0).astype('float32')
    corrected_fit=np.maximum(fit[:3]-slopes[:,None]*(nir[None]-nmin),0)
    rinf=np.percentile(corrected_fit,2,axis=1).astype('float32')
    surface=max(float(np.percentile(corrected_fit[1],95)),.001)
    b,g,r=corrected.transpose(1,0,2,3)
    bp=np.maximum(b,1e-6);gp=np.maximum(g,1e-6)
    kd=np.clip(.0166+.156*(bp/gp)**-1.14,.01,5)
    prior=np.log(np.clip(surface/gp,1,1e4))/(2*kd)
    layers=[b,g,r,raw[:,3],*lyzenga_channels(b,g,r,*rinf),stumpf_ratio(b,g),prior,masks]
    x=np.stack(layers,axis=1)
    return x,{'fit_unique_source_pixels':len(unique),'nir_reference':nmin,'glint_slopes':slopes.tolist(),
              'rinf_train_only':rinf.tolist(),'green_surface_reference':surface}


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'dataset_complete.json').exists():raise RuntimeError('Preserve completed data; refusing overwrite')
    pairing=ROOT/'pairing_official'
    rows=pd.read_csv(pairing/'source_samples.csv')
    match=pd.read_csv(pairing/'assignments.csv')
    scenes=json.loads((pairing/'scene_manifest.json').read_text())
    missing=[name for name in scenes if not (pairing/'crops'/f'{name}.tif').exists()]
    if missing:raise RuntimeError(f'Wait for {len(missing)} historical crops')
    arrays,_,_=load_stack('runs/2026-11-04/processed/stack.tif');arrays.pop('gebco_depth',None)
    coords=rows[['row','col']].to_numpy(dtype='int64')
    base=np.stack([np.stack([arrays[b][r-32:r+32,c-32:c+32] for b in ('B02','B03','B04','B08')]) for r,c in coords])
    std=np.stack([np.stack([arrays[b+'_std'][r-32:r+32,c-32:c+32] for b in ('B02','B03','B04','B08')]) for r,c in coords])
    std=np.nan_to_num(std,nan=0,posinf=0,neginf=0)
    mask=np.stack([arrays['valid_mask'][r-32:r+32,c-32:c+32] for r,c in coords])>.5
    mask &= np.isfinite(base).all(1)&(base[:,0]>0)&(base[:,1]>0)&(base[:,1]>base[:,3])
    base=np.nan_to_num(base,nan=0,posinf=0,neginf=0)
    # Test points are never rejected by image quality. Train eligibility follows
    # the pre-existing optical mask, before testing any historical match.
    test=np.flatnonzero(rows.split.eq('test'))
    val0=np.flatnonzero(rows.split.eq('val'));val=val0[far(coords[val0],coords[test])]
    tr0=np.flatnonzero(rows.split.eq('train')&(mask.mean((1,2))>=.5)&mask[:,32,32])
    train=tr0[far(coords[tr0],coords[np.r_[test,val]])]
    if min(len(train),len(val),len(test))<20:raise ValueError('Insufficient fixed buffered splits')
    kept=np.sort(np.r_[train,val,test]);remap={old:i for i,old in enumerate(kept)}
    indices={f'{key}_idx':np.array([remap[int(i)] for i in ids],dtype='int64') for key,ids in (('train',train),('val',val),('test',test))}
    assert np.all(far(coords[train],coords[np.r_[test,val]]))
    matched=base.copy();matched_mask=mask.copy();matched_std=std.copy()
    source_ids=np.zeros(len(rows),dtype='int64');accepted=np.zeros(len(rows),dtype=bool)
    reason=np.full(len(rows),'no_scene',dtype=object)
    for sid,(name,scene) in enumerate(sorted(scenes.items()),1):
        with rasterio.open(pairing/'crops'/f'{name}.tif') as src:
            image=src.read();tags=src.tags()
        r0,c0=int(tags['original_grid_row']),int(tags['original_grid_col'])
        for i in match.loc[match.scene_id.eq(name),'sample_index']:
            r,c=coords[i]-[r0,c0];patch=image[:4,r-32:r+32,c-32:c+32];scl=image[4,r-32:r+32,c-32:c+32]
            if patch.shape!=(4,64,64):reason[i]='crop_bounds';continue
            valid=np.isfinite(patch).all(0)&(patch[0]>0)&(patch[1]>0)&(patch[1]>patch[3])&(scl==6)
            reason[i]='cloud_or_water_quality'
            if not valid[32,32] or valid.mean()<.5:continue
            matched[i]=np.nan_to_num(patch,nan=0);matched_mask[i]=valid;matched_std[i]=0
            source_ids[i]=sid;accepted[i]=True;reason[i]='matched_clear_water'
    tide_report=json.loads((ROOT/'tides/validation.json').read_text())
    if not tide_report['cases']['tide_coastal_10km']['gate_passed']:raise RuntimeError('Tide validation gate failed')
    delta=np.zeros(len(rows),dtype='float64')
    ii=np.flatnonzero(accepted)
    atl_tide=predict(rows.loc[ii,'lon'],rows.loc[ii,'lat'],rows.loc[ii,'time'],True)
    scene_tide=predict(rows.loc[ii,'lon'],rows.loc[ii,'lat'],match.loc[ii,'scene_datetime'],True)
    d=scene_tide-atl_tide
    good=np.isfinite(d)&((rows.loc[ii,'depth_m'].to_numpy()+d)>.01)
    delta[ii[good]]=d[good]
    for i in ii[~good]:
        matched[i]=base[i];matched_mask[i]=mask[i];matched_std[i]=std[i];source_ids[i]=0;accepted[i]=False;reason[i]='tide_unavailable_or_emergent'
    rows['matched_clear_used']=accepted;rows['fallback_reason']=reason;rows['tide_delta_m']=delta
    rows['kept']=np.isin(np.arange(len(rows)),kept)
    rows.to_csv(OUT/'sample_audit.csv',index=False)
    description={'reference':'official ATL24 V002; 0-20m acquisition-relative median per10m pixel/pass',
        'test_geography':'all171 product-eligible test points preserved; original geographic blocks;96px buffer',
        'training_eligibility':'2024 optical mask >=50% patch and valid water centre, before historical matching',
        'scene_quality':'SCL6 water at centre and >=50% patch, positive finite B/G, NDWI>0; otherwise explicit2024 fallback',
        'tide_formula':'scene depth = ATL24 depth + EOT20(scene)-EOT20(ATL24); revert predictions before common-target scoring',
        'tide_limit':'coastal extrapolation<=10km; held-date surface RMSE0.269m; other water-level/seabed changes unresolved',
        'horizontal_uncertainty':'median~7m; recorded per sample; no fabricated per-axis covariance or positional correction',
        'single_scene_std':'matched inputs have zero temporal std; fallback uses original composite std',
        'split_sizes':{k:len(v) for k,v in indices.items()},'arms':{}}
    for name,raw,m,s,delta_arm,sources in (
        ('composite_control',base,mask,std,np.zeros(len(rows)),np.zeros(len(rows),dtype='int64')),
        ('matched_tide',matched,matched_mask,matched_std,delta,source_ids)):
        out=OUT/name/'features';out.mkdir(parents=True,exist_ok=True)
        if (out/'patches.npz').exists():raise RuntimeError('Refusing stale features')
        x,fit=engineer(raw,m.astype('float32'),coords,sources,train)
        x=np.concatenate([x,s],axis=1).astype('float32')
        targets=rows.depth_m.to_numpy()+delta_arm
        y=np.broadcast_to(targets[kept,None,None],(len(kept),64,64)).astype('float32').copy()
        assert x.shape[1]==16 and np.isfinite(x).all() and np.isfinite(y).all()
        np.savez(out/'patches.npz',X=x[kept],Y=y,M=m[kept].astype('float32'),coords=coords[kept]-32,
            acquisition_target=rows.depth_m.to_numpy()[kept],tide_delta=delta_arm[kept],matched=accepted[kept],
            sigma_tvu=rows.sigma_tvu.to_numpy()[kept],sigma_thu=rows.sigma_thu.to_numpy()[kept],**indices)
        stats={split:{'n':len(ids),'historical_clear_used':int(accepted[ids].sum()),
                       'fallback':int((~accepted[ids]).sum())} for split,ids in (('train',train),('val',val),('test',test))}
        metadata={**description,'arm':name,'feature_fitting':fit,'matched_quality_counts':stats,
            'target_range':[float(targets[kept].min()),float(targets[kept].max())]}
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        description['arms'][name]={'feature_fit':fit,'quality':stats}
    (OUT/'dataset_complete.json').write_text(json.dumps(description,indent=2),encoding='utf-8')
    update('atl24','paired_data_ready',split_sizes=description['split_sizes'],quality=stats)
    print(json.dumps({'splits':description['split_sizes'],'quality':stats},indent=2))


if __name__=='__main__':main()
