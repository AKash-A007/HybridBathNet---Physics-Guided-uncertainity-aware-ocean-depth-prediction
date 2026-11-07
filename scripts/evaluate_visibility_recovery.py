"""Fetch two public L1C tiles and assess a published visibility model on our AOI."""
from pathlib import Path
import concurrent.futures
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window,from_bounds
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.visibility_reference import PublishedVisibility,features_at_points,BANDS
from scripts.recovery_state import update

ROOT=Path('runs/2026-11-06/recovery_v2')
OUT=ROOT/'visibility'


def download_one(task):
    name,url,path=task
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        part=path.with_suffix('.part')
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with part.open('wb') as f:
                for chunk in response.iter_content(1024*1024):f.write(chunk)
        with rasterio.open(part) as check:
            assert check.width>0
        part.replace(path)
    return name,path.stat().st_size


def fetch():
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=json.loads((ROOT/'probe_sentinel-2-l1c.json').read_text())['features']
    tasks=[];clouds={}
    for item in scenes:
        for band in BANDS:
            url=item['assets'][band]['href']
            bucket,tail=url[5:].split('/',1)
            tasks.append((item['id']+'/'+band,f'https://{bucket}.s3.amazonaws.com/{tail}',OUT/'raw'/item['id']/f'{band}.jp2'))
        props=item['properties'];tile=props['grid:code'].replace('MGRS-','')
        prefix=f"tiles/{tile[:2]}/{tile[2]}/{tile[3:]}/{props['s2:product_uri']}/GRANULE/"
        response=requests.get('https://storage.googleapis.com/gcp-public-data-sentinel-2',
            params={'prefix':prefix,'max-keys':1000},timeout=60)
        response.raise_for_status()
        keys=[node.text for node in ET.fromstring(response.content).iter() if node.tag.endswith('}Key')]
        key=next((key for key in keys if key.endswith('/MSK_CLASSI_B00.jp2')),None)
        if key is None:raise RuntimeError(f'No native cloud classification mask: {item["id"]}')
        url='https://storage.googleapis.com/gcp-public-data-sentinel-2/'+key
        tasks.append((item['id']+'/cloud',url,OUT/'raw'/item['id']/'cloud.jp2'))
    update('visibility','downloading_native_toa',date='2024-01-13',tiles=len(scenes),files=len(tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,(name,size) in enumerate(pool.map(download_one,tasks),1):
            print('Visibility input',i,'/',len(tasks),name,size,flush=True)
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as canonical:
        bounds=canonical.bounds
    for item in scenes:
        dest=OUT/f'{item["id"]}_toa_aoi.tif'
        if dest.exists():continue
        with rasterio.open(OUT/'raw'/item['id']/'blue.jp2') as grid:
            candidate=from_bounds(*bounds,transform=grid.transform)
            window=Window(int(np.floor(candidate.col_off))-20,int(np.floor(candidate.row_off))-20,
                          int(np.ceil(candidate.width))+40,int(np.ceil(candidate.height))+40)
            window=window.intersection(Window(0,0,grid.width,grid.height))
            profile=grid.profile.copy();profile.update(driver='GTiff',count=12,dtype='uint16',
                height=int(window.height),width=int(window.width),transform=grid.window_transform(window),
                compress='deflate',tiled=True,blockxsize=256,blockysize=256)
            with rasterio.open(dest,'w',**profile) as target:
                for i,key in enumerate((*BANDS,'cloud'),1):
                    with rasterio.open(OUT/'raw'/item['id']/f'{key}.jp2') as source:
                        with WarpedVRT(source,crs=grid.crs,transform=grid.transform,width=grid.width,
                                       height=grid.height,resampling=Resampling.nearest) as vrt:
                            values=vrt.read(1,window=window)
                    if key=='cloud':values=binary_dilation(values==1,iterations=8).astype('uint16')
                    target.write(values.astype('uint16'),i);target.set_band_description(i,key)
                target.update_tags(scene=item['id'],mgrs=item['properties']['grid:code'].replace('MGRS-',''),
                                   baseline=item['properties']['s2:processing_baseline'])
    update('visibility','native_toa_crops_ready',tiles=len(scenes),date='2024-01-13')


def evaluate():
    torch.set_num_threads(2)
    model_path=ROOT/'external/optically_shallow_deep/opticallyshallowdeep/models/TOA.h5'
    model=PublishedVisibility(model_path).eval()
    check=model.numerical_check()
    (OUT/'conversion_check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    with np.load('runs/2026-11-05/nearshore_hybrid_v1/features/patches.npz') as d:
        coords=d['coords']+32
        depth=d['Y'][:,32,32]
        splits={name:d[f'{name}_idx'] for name in ('train','val','test')}
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as grid:
        xx,yy=rasterio.transform.xy(grid.transform,coords[:,0],coords[:,1])
    result=pd.DataFrame({'index':np.arange(len(coords)),'row':coords[:,0],'col':coords[:,1],
                         'target_depth':depth,'p_optically_shallow':np.nan,'valid':False,'scene':''})
    for path in sorted(OUT.glob('*_toa_aoi.tif')):
        with rasterio.open(path) as source:
            image=source.read();tags=source.tags()
            rows,cols=rasterio.transform.rowcol(source.transform,xx,yy)
        features,valid=features_at_points(image[:11],rows,cols,mgrs=tags['mgrs'],baseline=float(tags['baseline']))
        rr=np.clip(rows,0,image.shape[1]-1);cc=np.clip(cols,0,image.shape[2]-1)
        valid &= image[11,rr,cc]==0
        valid &= ~result.valid.to_numpy()
        with torch.no_grad():probability=model(torch.from_numpy(features)).numpy().ravel()
        result.loc[valid,'p_optically_shallow']=probability[valid]
        result.loc[valid,'valid']=True;result.loc[valid,'scene']=tags['scene']
    summary={'source':'Richardson et al.2024 TOA pretrained model; source commit recorded separately',
        'date':'2024-01-13','scope':'native10m neighborhood features at fixed GEBCO centres; no operational mask',
        'calibrated_to_this_aoi':False,'conversion_check':check,'splits':{}}
    for split,idx in splits.items():
        result.loc[idx,'split']=split
        selected=result.iloc[idx]
        info={'n':len(idx),'valid':int(selected.valid.sum()),
              'shallow_ge_0_6':int((selected.p_optically_shallow>=.6).sum()),
              'deep_le_0_4':int((selected.p_optically_shallow<=.4).sum())}
        if split in ('val','test'):
            stem='validation' if split=='val' else 'test'
            with np.load(f'runs/2026-11-05/nearshore_hybrid_v1/{stem}_predictions.npz') as d:
                prediction=d['prediction'].reshape(-1)
            primary=(depth[idx]>=0)&(depth[idx]<=20)
            supported=(selected.p_optically_shallow.to_numpy()>=.6)&primary
            info['primary_n']=int(primary.sum());info['supported_primary_n']=int(supported.sum())
            info['full_primary_rmse']=float(np.sqrt(np.mean((prediction[primary]-depth[idx][primary])**2)))
            info['supported_primary_rmse']=float(np.sqrt(np.mean((prediction[supported]-depth[idx][supported])**2))) if supported.any() else None
            info['supported_fraction_primary']=float(supported.sum()/primary.sum())
            info['supported_groups_2km']=int(len(np.unique(coords[idx][supported]//200,axis=0)))
        summary['splits'][split]=info
    result.to_csv(OUT/'point_visibility.csv',index=False)
    (OUT/'visibility_assessment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    update('visibility','assessed_uncalibrated',summary=summary['splits'],local_visibility_truth_available=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':fetch();evaluate()
