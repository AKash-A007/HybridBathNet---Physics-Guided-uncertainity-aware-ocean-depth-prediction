"""Validate external EOT20 tide variation against held-date ATL24 sea surface."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pyTMD
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.recovery_state import update
ROOT=Path('runs/2026-11-06/recovery_v2')


def predict(lon,lat,time,extrapolate=False):
    dt=pd.to_datetime(time,utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')
    tide=pyTMD.compute.tide_elevations(np.asarray(lon),np.asarray(lat),dt,
        DIRECTORY=ROOT/'tides',MODEL='EOT20',TIME='datetime',METHOD='linear',
        CROP=True,BOUNDS=[75,78,7,11],EXTRAPOLATE=extrapolate,CUTOFF=10)
    return np.ma.filled(tide,np.nan).astype('float64')


def main():
    surface=pd.read_parquet(ROOT/'official_atl24_aoi.parquet',filters=[('class_ph','==',41)],
        columns=['granule','beam','lon_ph','lat_ph','surface_h','x_atc','time'])
    surface['along_group']=(surface.x_atc//2000).astype('int64')
    g=surface.groupby(['granule','beam','along_group'],as_index=False).agg(
        lon=('lon_ph','mean'),lat=('lat_ph','mean'),surface_h=('surface_h','median'),
        time=('time','first'),photons=('surface_h','size'))
    g['date']=g.time.dt.strftime('%Y-%m-%d')
    # Freeze chronological pass/date partition before inspecting tide errors.
    dates=sorted(g.date.unique());fit_dates=set(dates[:max(1,int(.6*len(dates)))])
    g['fit_date']=g.date.isin(fit_dates)
    update('tides','validating_held_dates',groups=len(g),dates=len(dates))
    g['tide_strict']=predict(g.lon,g.lat,g.time)
    g['tide_coastal_10km']=predict(g.lon,g.lat,g.time,True)
    report={'model':'EOT20','pyTMD':pyTMD.__version__,'source_doi':'10.17882/79489',
        'scope':'ATLAS surface_h vs modeled sea-surface variation; not independent tide gauge validation',
        'offset':'single constant fitted only on earliest60% dates; last40% dates held out',
        'predeclared_gate':'held-date RMSE <=0.35m and >=10% reduction vs fitted constant, >=20 held groups',
        'extrapolation':'linear constituent interpolation; optional nearest coastal extrapolation <=10km',
        'cases':{}}
    for key in ('tide_strict','tide_coastal_10km'):
        finite=np.isfinite(g[key])&np.isfinite(g.surface_h)
        fit=g.loc[finite&g.fit_date];held=g.loc[finite&~g.fit_date]
        if len(fit)<20 or len(held)<20:
            report['cases'][key]={'finite_groups':int(finite.sum()),'fit_groups':len(fit),'held_groups':len(held),'gate_passed':False};continue
        offset=float(np.median(fit.surface_h-fit[key]));constant=float(fit.surface_h.median())
        rmse=float(np.sqrt(np.mean((held.surface_h-held[key]-offset)**2)))
        baseline=float(np.sqrt(np.mean((held.surface_h-constant)**2)))
        report['cases'][key]={'finite_groups':int(finite.sum()),'fit_groups':len(fit),'held_groups':len(held),
            'offset_m':offset,'held_rmse_m':rmse,'held_constant_rmse_m':baseline,
            'held_tide_surface_correlation':float(np.corrcoef(held[key],held.surface_h)[0,1]),
            'gate_passed':bool(rmse<=.35 and rmse<=.9*baseline)}
    g.to_csv(ROOT/'tides/surface_validation.csv',index=False)
    (ROOT/'tides/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    update('tides','validation_complete',cases=report['cases'])
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
