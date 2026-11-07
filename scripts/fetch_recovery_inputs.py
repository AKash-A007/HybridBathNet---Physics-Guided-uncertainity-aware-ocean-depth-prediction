"""Fetch catalogue/model metadata and authorized official ATL24 recovery data.

Credentials are read only for explicit official ATL24 download, never printed.
Third-party source is saved for inspection, not imported/executed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import requests

ROOT = Path('runs/2026-11-06/recovery_v2')
REPO = 'yulunwu8/Optically-Shallow-Deep'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', action='store_true')
    parser.add_argument('--access-probe', action='store_true')
    parser.add_argument('--official-atl24', action='store_true')
    parser.add_argument('--match-scenes', action='store_true')
    parser.add_argument('--match-official', action='store_true')
    parser.add_argument('--visibility-scene', action='store_true')
    parser.add_argument('--tide-download', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'external' / 'optically_shallow_deep'
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(f'https://api.github.com/repos/{REPO}/git/trees/main?recursive=1', timeout=45)
    response.raise_for_status()
    tree = response.json()
    (root/'repository_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
    commit = tree['sha']
    paths = ['LICENSE','README.md','setup.py'] + [item['path'] for item in tree['tree']
        if item['path'].startswith('opticallyshallowdeep/') and item['path'].endswith('.py')]
    if args.weights or (root/'opticallyshallowdeep/models/TOA.h5').exists():
        paths += [item['path'] for item in tree['tree'] if item['path'].endswith('.h5')]
    manifest = []
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            response = session.get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{name}', timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        manifest.append({'path':name, 'bytes':path.stat().st_size,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / 'source_manifest.json').write_text(json.dumps({'repository':REPO,'commit':commit,
        'purpose':'independent optical visibility comparison; source inspected before any execution',
        'files':manifest},indent=2),encoding='utf-8')
    # Inspect one representative historical AOI scene. This is metadata only.
    for collection in ('sentinel-2-l1c','sentinel-2-l2a'):
        response = session.post('https://earth-search.aws.element84.com/v1/search',json={
            'collections':[collection], 'bbox':[76.30,8.90,76.55,9.20],
            'datetime':'2024-01-13T00:00:00Z/2024-01-14T00:00:00Z','limit':10},timeout=60)
        response.raise_for_status()
        data = response.json()
        (ROOT / f'probe_{collection}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(collection, len(data.get('features',[])), flush=True)
        if data.get('features'):
            feature = data['features'][0]
            print(feature['id'], list(feature['assets']), flush=True)
    print('Inspected source files:', len(manifest), 'commit:',commit,flush=True)
    if args.access_probe:
        import xml.etree.ElementTree as ET
        l1 = json.loads((ROOT/'probe_sentinel-2-l1c.json').read_text())['features'][0]
        l2 = json.loads((ROOT/'probe_sentinel-2-l2a.json').read_text())['features'][0]
        url = l1['assets']['blue']['href'].replace('s3://sentinel-s2-l1c/', 'https://sentinel-s2-l1c.s3.amazonaws.com/')
        checks = {}
        for label, address in (('aws_l1c',url),('public_l2a',l2['assets']['blue']['href'])):
            with session.get(address,headers={'Range':'bytes=0-15'},stream=True,timeout=30) as response:
                checks[label] = {'status':response.status_code, 'range':response.headers.get('Content-Range')}
        prefix = 'tiles/43/P/FK/' + l1['properties']['s2:product_uri'].split('_N')[0]
        response = session.get('https://storage.googleapis.com/gcp-public-data-sentinel-2',params={'prefix':prefix,'max-keys':8},timeout=45)
        checks['gcs_status'] = response.status_code
        if response.ok:
            keys = [a.text for a in ET.fromstring(response.content).iter() if a.tag.endswith('}Key')]
            checks['gcs_keys'] = keys
        response = session.get('https://cmr.earthdata.nasa.gov/search/granules.json',params={
            'short_name':'ATL24','version':'002','bounding_box':'76.30,8.90,76.55,9.20',
            'temporal':'2018-10-14T00:00:00Z,2024-11-07T23:59:59Z','page_size':2000},timeout=60)
        checks['cmr_status'] = response.status_code
        if response.ok:
            catalogue = response.json()
            (ROOT/'atl24_v002_archive_catalogue.json').write_text(json.dumps(catalogue,indent=2),encoding='utf-8')
            checks['official_granules'] = len(catalogue.get('feed',{}).get('entry',[]))
        (ROOT/'access_probe.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        print(json.dumps(checks,indent=2),flush=True)
    if args.official_atl24:
        download_official(session)
    if args.match_scenes:
        from match_sentinel_recovery import catalogue,download_crops
        catalogue();download_crops()
    if args.match_official:
        from match_sentinel_recovery import catalogue,download_crops
        catalogue(official=True);download_crops()
    if args.visibility_scene:
        from evaluate_visibility_recovery import fetch,evaluate
        fetch();evaluate()
    if args.tide_download:
        fetch_tide_model(session)


def fetch_tide_model(session):
    import zipfile
    from recovery_state import update
    root=ROOT/'tides';root.mkdir(exist_ok=True)
    url='https://www.seanoe.org/data/00683/79489/data/85762.zip'
    path=root/'85762.zip'
    update('tides','retrieving_eot20',source=url,doi='10.17882/79489')
    if not path.exists():
        part=path.with_suffix('.part')
        with session.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            chunks=response.iter_content(1024*1024);first=next(chunks,b'')
            if not first.startswith(b'PK'):raise ValueError('Tide source is not a ZIP archive')
            with part.open('wb') as handle:
                handle.write(first)
                for chunk in chunks:handle.write(chunk)
        part.replace(path)
    target=root/'EOT20';target.mkdir(exist_ok=True)
    def extract(archive,destination):
        for info in archive.infolist():
            result=(destination/info.filename).resolve()
            if not result.is_relative_to(destination.resolve()):raise ValueError('Unsafe archive path')
            if info.is_dir():result.mkdir(parents=True,exist_ok=True);continue
            if result.suffix not in ('.nc','.zip','.txt','.pdf'):continue
            if result.exists():continue
            result.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(info) as source,result.open('wb') as output:
                import shutil
                shutil.copyfileobj(source,output)
    with zipfile.ZipFile(path) as archive:extract(archive,target)
    for nested in list(target.rglob('ocean_tides.zip')):
        with zipfile.ZipFile(nested) as archive:extract(archive,nested.parent)
    files=list(target.rglob('*ocean_eot20.nc'))
    metadata={'source':url,'doi':'10.17882/79489','licence':'CC BY4.0; Hart-Davis et al.2021',
              'archive_bytes':path.stat().st_size,'ocean_constituent_files':len(files),
              'paths':[str(p) for p in files]}
    (root/'source.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    update('tides','downloaded_pending_local_validation',**metadata)
    print(json.dumps(metadata,indent=2),flush=True)


def download_official(session):
    import sys
    import h5py
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from download_atl24_official import load_token
    token = load_token()
    catalogue = json.loads((ROOT/'atl24_v002_archive_catalogue.json').read_text())['feed']['entry']
    dest = ROOT/'atl24_v002'
    dest.mkdir(exist_ok=True)
    rows, frames = [], []
    fields = ('class_ph','confidence','delta_time','ellipse_h','index_ph','index_seg',
              'invalid_kd','invalid_wind_speed','lat_ph','lon_ph','low_confidence_flag',
              'night_flag','ortho_h','sensor_depth_exceeded','sigma_thu','sigma_tvu',
              'surface_h','x_atc','y_atc')
    for i,item in enumerate(catalogue):
        link = next(link['href'] for link in item['links']
                    if link.get('type')=='application/x-hdf5' and link['href'].startswith('https://data.nsidc.'))
        name = link.rsplit('/',1)[1]
        previous = Path('runs/2026-11-05/nearshore_error_reference_audit_v1/atl24_v002_official')/name
        path = previous if previous.exists() else dest/name
        if not path.exists():
            part = path.with_suffix('.part')
            with session.get(link,headers={'Authorization':f'Bearer {token}'},stream=True,timeout=(25,180)) as response:
                if response.status_code in (401,403):
                    raise RuntimeError('Earthdata authorization rejected; refresh protected token without posting it')
                response.raise_for_status()
                chunks = response.iter_content(1024*1024)
                first = next(chunks,b'')
                if not first.startswith(b'\x89HDF\r\n\x1a\n'):
                    raise RuntimeError('Official response is not HDF5; authorization needs verification')
                with part.open('wb') as handle:
                    handle.write(first)
                    for chunk in chunks:
                        handle.write(chunk)
            part.replace(path)
        with h5py.File(path,'r') as h5:
            for beam in ('gt1l','gt1r','gt2l','gt2r','gt3l','gt3r'):
                if beam not in h5:
                    continue
                group = h5[beam]
                lon,lat = group['lon_ph'][:],group['lat_ph'][:]
                inside = (lon>=76.30)&(lon<=76.55)&(lat>=8.90)&(lat<=9.20)
                if not inside.any():
                    continue
                frame = pd.DataFrame({field:group[field][:][inside] for field in fields if field in group})
                orbit = name.split('_')[2]
                frame['rgt'],frame['cycle'] = int(orbit[:4]),int(orbit[4:6])
                frame['granule'],frame['beam'] = name,beam
                frame['depth_m'] = frame.surface_h-frame.ortho_h
                frame['time'] = pd.Timestamp('2018-01-01',tz='UTC')+pd.to_timedelta(frame.delta_time,unit='s')
                frames.append(frame)
        rows.append({'granule':name,'path':str(path),'bytes':path.stat().st_size,'status':'verified_hdf5'})
        (ROOT/'official_download_status.json').write_text(json.dumps({'completed':len(rows),'total':len(catalogue),'files':rows},indent=2),encoding='utf-8')
        print('Official ATL24',i+1,'/',len(catalogue),name,flush=True)
    points = pd.concat(frames,ignore_index=True)
    points.to_parquet(ROOT/'official_atl24_aoi.parquet',index=False)
    high = (points.class_ph.eq(40)&points.confidence.ge(.6)&points.depth_m.between(.01,20))
    summary = {'version':'ATL24.002','aoi':[76.30,8.90,76.55,9.20],
        'granules':len(rows),'aoi_photons':len(points),'class_counts':{str(k):int(v) for k,v in points.class_ph.value_counts().items()},
        'high_confidence_shallow_photons':int(high.sum()),'fields':list(points),
        'total_bytes':sum(row['bytes'] for row in rows)}
    (ROOT/'official_extraction.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
