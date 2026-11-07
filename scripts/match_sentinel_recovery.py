"""Public historical L2A scene matching and AOI-only window retrieval.

Selection uses dates, tile coverage and catalogue cloud only; no depths/errors.
Frozen ATL24 centres are retained, including unmatched/poor-quality cases.
"""
from pathlib import Path
import concurrent.futures
import json
import sys
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.transform import xy
from pyproj import Transformer
from shapely.geometry import shape,Point
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.recovery_state import update

ROOT=Path('runs/2026-11-06/recovery_v2/pairing')
AUDIT=Path('runs/2026-11-06/physics_uncertainty_research_v1')
COLLECTIONS=['sentinel-2-l2a','sentinel-2-pre-c1-l2a','sentinel-2-c1-l2a']


def fetch_date(date):
    path=ROOT/'catalogue'/f'{date}.json'
    if path.exists():return date,json.loads(path.read_text())
    cached=Path('runs/2026-11-06/recovery_v2/pairing/catalogue')/f'{date}.json'
    if cached.exists():
        data=json.loads(cached.read_text());path.write_text(json.dumps(data),encoding='utf-8');return date,data
    day=pd.Timestamp(date)
    payload={'collections':COLLECTIONS,'bbox':[76.30,8.90,76.55,9.20],
        'datetime':(day-pd.Timedelta(days=10)).strftime('%Y-%m-%dT00:00:00Z')+'/'+
                   (day+pd.Timedelta(days=11)).strftime('%Y-%m-%dT00:00:00Z'),'limit':100}
    items=[]
    response=requests.post('https://earth-search.aws.element84.com/v1/search',json=payload,timeout=90)
    response.raise_for_status();result=response.json();items+=result.get('features',[])
    while any(link.get('rel')=='next' for link in result.get('links',[])):
        link=next(link for link in result['links'] if link.get('rel')=='next')
        response=requests.post(link['href'],json=link.get('body',payload),timeout=90) if link.get('method')=='POST' else requests.get(link['href'],timeout=90)
        response.raise_for_status();result=response.json();items+=result.get('features',[])
    path.write_text(json.dumps(items),encoding='utf-8')
    return date,items


def catalogue(official=False):
    global ROOT
    if official:ROOT=Path('runs/2026-11-06/recovery_v2/pairing_official')
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'catalogue').mkdir(exist_ok=True)
    if official:
        samples=pd.read_csv(Path('runs/2026-11-06/recovery_v2/official_profiles.csv'))
        samples=samples.loc[samples.primary_0_20&samples.split.isin(['train','val','test'])].reset_index(drop=True)
        samples=samples.rename(columns={'grid_row':'row','grid_col':'col'})
        samples.to_csv(ROOT/'source_samples.csv',index=False)
    else:
        samples=pd.read_csv(AUDIT/'atl24_sample_audit.csv')
        raw=pd.read_parquet('runs/2026-11-05/nearshore_error_reference_audit_v1/atl24_archive_with_original_split.parquet')
        raw['pixel_key']=raw.grid_row.astype(str)+':'+raw.grid_col.astype(str)
        good=(raw.class_ph.eq(40)&raw.confidence.ge(.6)&raw.quality_ph.eq(0)&raw.depth_m.between(.01,100)&raw.sigma_tvu.le(1))
        raw=raw.loc[good]
        raw['date']=raw.index.strftime('%Y-%m-%d')
        dates=raw.groupby('pixel_key').date.first()
        samples['date']=samples.pixel_key.map(dates)
    assert samples.date.notna().all()
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as grid:
        xx,yy=xy(grid.transform,samples.row.to_numpy(),samples.col.to_numpy())
        lon,lat=Transformer.from_crs(grid.crs,'EPSG:4326',always_xy=True).transform(xx,yy)
    samples['lon'],samples['lat']=lon,lat
    all_dates={}
    update('pairing','querying_historical_scenes',dates=int(samples.date.nunique()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for date,items in pool.map(fetch_date,sorted(samples.date.unique())):
            all_dates[date]=items
    scenes={};rows=[]
    for i,sample in samples.iterrows():
        candidates=[]
        for item in all_dates[sample.date]:
            props=item['properties'];cloud=float(props.get('eo:cloud_cover',100))
            if cloud>40 or not all(key in item['assets'] for key in ('blue','green','red','nir','scl')):
                continue
            if not shape(item['geometry']).covers(Point(sample.lon,sample.lat)):
                continue
            gap=abs((pd.Timestamp(props['datetime']).tz_localize(None)-pd.Timestamp(sample.date)).total_seconds())/86400
            if gap>10.999:continue
            # Predeclared quality/time cost; independent of target/model errors.
            candidates.append((gap+.1*cloud,gap,cloud,item))
        candidates.sort(key=lambda x:(x[0],x[3]['id']))
        row={'sample_index':int(i),'pixel_key':sample.pixel_key,'split':sample['split'],
             'photon_date':sample.date,'row':int(sample.row),'col':int(sample.col),'matched':bool(candidates)}
        if candidates:
            _,gap,cloud,item=candidates[0]
            scenes[item['id']]=item
            row.update(scene_id=item['id'],scene_datetime=item['properties']['datetime'],gap_days=gap,cloud_pct=cloud)
        rows.append(row)
    assignments=pd.DataFrame(rows)
    assignments.to_csv(ROOT/'assignments.csv',index=False)
    (ROOT/'scene_manifest.json').write_text(json.dumps(scenes),encoding='utf-8')
    update('pairing','catalogue_matched',samples=len(samples),matched=int(assignments.matched.sum()),
           scenes=len(scenes),policy='±10day dates, <=40% catalogue cloud, gap+.1*cloud cost')
    print('Matched',int(assignments.matched.sum()),'of',len(samples),'centres to',len(scenes),'scenes',flush=True)


def crop_one(task):
    scene_id,item,rows=task
    dest=ROOT/'crops'/f'{scene_id}.tif'
    if dest.exists():return {'scene_id':scene_id,'status':'existing','bytes':dest.stat().st_size}
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as grid:
        r0=max(0,int(rows.row.min())-48);c0=max(0,int(rows.col.min())-48)
        r1=min(grid.height,int(rows.row.max())+48);c1=min(grid.width,int(rows.col.max())+48)
        window=Window(c0,r0,c1-c0,r1-r0)
        transform=grid.window_transform(window);crs=grid.crs
        shape_hw=(r1-r0,c1-c0)
        output=np.empty((5,*shape_hw),dtype='float32')
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',VSI_CACHE='TRUE'):
            for band,key in enumerate(('blue','green','red','nir','scl')):
                asset=item['assets'][key]
                if not asset['href'].startswith('https://'):
                    raise ValueError('Expected public HTTPS COG asset')
                with rasterio.open(asset['href']) as source:
                    with WarpedVRT(source,crs=grid.crs,transform=grid.transform,width=grid.width,
                                   height=grid.height,resampling=Resampling.nearest,src_nodata=0,nodata=0) as vrt:
                        values=vrt.read(1,window=window).astype('float32')
                if key!='scl':
                    specification=asset.get('raster:bands',[{}])[0]
                    if 'scale' not in specification:
                        raise ValueError(f'Missing reflectance scale for {scene_id}/{key}')
                    invalid=values==0
                    values=values*float(specification['scale'])+float(specification.get('offset',0))
                    values[invalid]=np.nan
                output[band]=values
    part=dest.with_suffix('.part')
    with rasterio.open(part,'w',driver='GTiff',width=shape_hw[1],height=shape_hw[0],count=5,
                       dtype='float32',crs=crs,transform=transform,nodata=np.nan,compress='deflate',tiled=True) as target:
        target.write(output)
        for band,name in enumerate(('B02','B03','B04','B08','SCL'),1):target.set_band_description(band,name)
        target.update_tags(scene_id=scene_id,scene_datetime=item['properties']['datetime'],
                           original_grid_row=r0,original_grid_col=c0,selection_uses_depth='false')
    part.replace(dest)
    return {'scene_id':scene_id,'status':'downloaded_crop','bytes':dest.stat().st_size}


def download_crops():
    (ROOT/'crops').mkdir(exist_ok=True)
    samples=pd.read_csv(ROOT/'assignments.csv')
    scenes=json.loads((ROOT/'scene_manifest.json').read_text())
    tasks=[(name,scenes[name],rows) for name,rows in samples.loc[samples.matched].groupby('scene_id')]
    done=[]
    update('pairing','retrieving_aoi_crops',total_scenes=len(tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(crop_one,tasks):
            done.append(result)
            (ROOT/'crop_status.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
            print('AOI crop',len(done),'/',len(tasks),result['scene_id'],flush=True)
    update('pairing','crops_ready',scenes=len(done),bytes=sum(row['bytes'] for row in done))


if __name__=='__main__':
    catalogue();download_crops()
