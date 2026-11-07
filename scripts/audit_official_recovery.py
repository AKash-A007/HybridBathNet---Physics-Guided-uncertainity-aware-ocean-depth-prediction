"""Version-consistent ATL24 quality, footprint and legacy source audit."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.recovery_state import update
ROOT=Path('runs/2026-11-06/recovery_v2')


def main():
    raw=pd.read_parquet(ROOT/'official_atl24_aoi.parquet',filters=[('class_ph','==',40)])
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as grid:
        xx,yy=Transformer.from_crs(4326,grid.crs,always_xy=True).transform(raw.lon_ph.to_numpy(),raw.lat_ph.to_numpy())
        rows,cols=rasterio.transform.rowcol(grid.transform,xx,yy)
        raw['grid_row'],raw['grid_col']=rows,cols
        raw['inside_stack']=(np.asarray(rows)>=32)&(np.asarray(cols)>=32)&(np.asarray(rows)+32<=grid.height)&(np.asarray(cols)+32<=grid.width)
    raw['x_m'],raw['y_m']=xx,yy
    with np.load('runs/2026-11-05/improvement/features/patches.npz') as d:
        blocks={split:set(map(tuple,np.unique(d['coords'][d[f'{split}_idx']]//128,axis=0))) for split in ('train','val','test')}
    raw['split']=[next((name for name,b in blocks.items() if tuple(rc) in b),'other') for rc in raw[['grid_row','grid_col']].to_numpy()//128]
    raw['date']=raw.time.dt.strftime('%Y-%m-%d')
    basic=(raw.confidence.ge(.6)&raw.depth_m.between(.01,100)&raw.sigma_tvu.gt(0)&raw.sigma_tvu.le(1)&raw.sensor_depth_exceeded.eq(0)&raw.inside_stack)
    raw['product_quality_eligible']=basic
    # Do not turn missing VIIRS Kd into a fabricated product-quality exclusion.
    profiles=raw.loc[basic].groupby(['grid_row','grid_col','split','date','rgt','cycle','beam'],as_index=False).agg(
        depth_m=('depth_m','median'),photon_count=('depth_m','size'),sigma_tvu=('sigma_tvu','median'),
        sigma_thu=('sigma_thu','median'),surface_h=('surface_h','median'),depth_min=('depth_m','min'),depth_max=('depth_m','max'),
        lon=('lon_ph','mean'),lat=('lat_ph','mean'),time=('time','first'),invalid_kd=('invalid_kd','max'))
    profiles['within_pixel_depth_range_m']=profiles.depth_max-profiles.depth_min
    profiles['primary_0_20']=profiles.depth_m.between(.01,20)
    profiles['pixel_key']=profiles.grid_row.astype(str)+':'+profiles.grid_col.astype(str)
    profiles.to_csv(ROOT/'official_profiles.csv',index=False)
    raw.to_parquet(ROOT/'official_bottom_audited.parquet',index=False)
    summary={'official_version':'002','bottom_photons':len(raw),'product_eligible_photons':int(basic.sum()),
        'aggregation':'median per10m pixel/pass/date, before0-20m cutoff; no reflectance/test error exclusions',
        'quality':'class40, confidence>=.6, 0<depth<=100, 0<sigma_tvu<=1, sensor_depth_exceeded=0, patch inside grid',
        'sigma_thu_interpretation':'total horizontal1sigma, not independent x/y sigmas',
        'no_double_refraction':True,'depth_definition':'surface_h minus refraction-corrected ortho_h; relative to acquisition water surface',
        'counts':{},'caveats':['V002 TPU is documented optimistic; isolated class40 false positives remain possible.',
        'Adjacent pixels/repeated tracks are correlated; photon counts are not independent sample counts.',
        'invalid_kd denotes missing auxiliary VIIRS Kd; ATLAS flags are not Sentinel visibility truth.']}
    for split in ('train','val','test','other'):
        a=profiles.loc[profiles.primary_0_20&profiles.split.eq(split)]
        summary['counts'][split]={'pixel_passes':len(a),'pixels':a.pixel_key.nunique(),'dates':a.date.nunique(),
            'bins':{f'{lo}-{hi}':int(((a.depth_m>=lo)&(a.depth_m<hi)).sum()) for lo,hi in ((0,5),(5,10),(10,15),(15,20.0001))},
            'median_sigma_thu_m':float(a.sigma_thu.median()) if len(a) else None,
            'single_photon_pct':float(100*a.photon_count.eq(1).mean()) if len(a) else None}
    legacy=pd.read_parquet('runs/2026-11-05/nearshore_error_reference_audit_v1/atl24_archive_with_original_split.parquet')
    legacy=legacy.loc[legacy.class_ph.eq(40)&legacy.confidence.ge(.6)].copy()
    legacy['beam']=legacy['gt'].map({10:'gt1l',20:'gt1r',30:'gt2l',40:'gt2r',50:'gt3l',60:'gt3r'})
    legacy['date']=legacy.index.strftime('%Y-%m-%d')
    lx,ly=Transformer.from_crs(4326,32643,always_xy=True).transform(legacy.lon_ph.to_numpy(),legacy.lat_ph.to_numpy())
    legacy['x_m'],legacy['y_m']=lx,ly
    matched=[]
    for key,a in legacy.groupby(['date','rgt','cycle','beam']):
        b=raw.loc[raw.date.eq(key[0])&raw.rgt.eq(key[1])&raw.cycle.eq(key[2])&raw.beam.eq(key[3])]
        if not len(b):continue
        dist,idx=cKDTree(b[['x_m','y_m']]).query(a[['x_m','y_m']])
        good=dist<=2
        for old,new,distance in zip(a.loc[good].itertuples(),b.iloc[idx[good]].itertuples(),dist[good]):
            matched.append({'distance_m':float(distance),'old_depth_m':old.depth_m,'official_depth_m':new.depth_m,
                'delta_depth_m':new.depth_m-old.depth_m,'date':key[0],'rgt':int(key[1]),'cycle':int(key[2]),'beam':key[3]})
    pd.DataFrame(matched).to_csv(ROOT/'official_legacy_crossmatch.csv',index=False)
    differences=np.array([m['delta_depth_m'] for m in matched])
    finite=np.isfinite(differences)
    differences=differences[finite]
    summary['legacy_crossmatch']={'scope':'same UTCdate/rgt/cycle/beam, nearest horizontal distance<=2m; approximate not photon identity',
        'matched':len(matched),'finite_depth_pairs':int(finite.sum()),'legacy_high_conf_photons':len(legacy),
        'median_abs_depth_difference_m':float(np.median(abs(differences))) if len(differences) else None,
        'p90_abs_depth_difference_m':float(np.percentile(abs(differences),90)) if len(differences) else None,
        'legacy_exact_source_version_verified':False}
    (ROOT/'official_quality_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    update('atl24','official_quality_audited',summary=summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
