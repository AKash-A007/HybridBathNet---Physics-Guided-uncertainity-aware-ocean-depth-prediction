"""Audit original GEBCO coordinates against the actual training raster."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import rasterio
from rasterio.transform import from_origin, xy
from rasterio.warp import reproject, Resampling, transform_bounds
from pyproj import Transformer
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

OUT=Path('runs/2026-11-05/grid_audit')


def stats(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return {'n':int(a.size),'min':float(a.min()),'max':float(a.max()),
            'mean':float(a.mean()),'rmse':float(np.sqrt(np.mean(a*a)))} if a.size else {'n':0}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with xr.open_dataset('data/raw/gebco/gebco_subset.nc') as ds:
        da=ds['elevation']; print(ds,flush=True)
        print(da.attrs,flush=True)
        print({k:(v.values[:3].tolist(),v.values[-3:].tolist()) for k,v in ds.coords.items()},flush=True)
        lat=ds['latitude'].values; lon=ds['longitude'].values
        values=da.transpose('latitude','longitude').values.astype('float32')
        source_attrs=dict(da.attrs)
    ilat=np.argsort(lat); ilon=np.argsort(lon)
    lat=lat[ilat]; lon=lon[ilon]; values=values[np.ix_(ilat,ilon)]
    dx=float(np.diff(lon).mean()); dy=float(np.diff(lat).mean())
    assert np.allclose(np.diff(lon),dx) and np.allclose(np.diff(lat),dy)
    source_transform=from_origin(lon[0]-dx/2,lat[-1]+dy/2,dx,dy)
    with rasterio.open('runs/2026-11-04/processed/stack.tif') as stack:
        old=stack.read(9); mask=stack.read(10)>.5
        profile=stack.profile.copy(); transform=stack.transform; crs=stack.crs
        direct=np.full(stack.shape,np.nan,dtype='float32')
        reproject(values[::-1].copy(),direct,src_transform=source_transform,src_crs='EPSG:4326',
                  src_nodata=np.nan,dst_transform=transform,dst_crs=crs,dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        grid={'shape':stack.shape,'crs':str(crs),'transform':list(transform),'bounds':list(stack.bounds)}
    with rasterio.open('data/raw/gebco/gebco_subset.tif') as src:
        intermediate={'shape':src.shape,'crs':str(src.crs),'transform':list(src.transform),'nodata':src.nodata,
                      'range':stats(src.read(1,masked=True).compressed()),'dtype':src.dtypes}
    features=np.load('runs/2026-11-05/improvement/features/patches.npz')
    coords=features['coords']; r=coords[:,0]+32; c=coords[:,1]+32
    xs,ys=xy(transform,r,c)
    lons,lats=Transformer.from_crs(crs,'EPSG:4326',always_xy=True).transform(xs,ys)
    independent=RegularGridInterpolator((lat,lon),values,bounds_error=False,fill_value=np.nan)(np.column_stack([lats,lons]))
    # Independent interpolation on geographic cell centres detects shifts/flips.
    differences={}
    for split in ('train_idx','val_idx','test_idx'):
        idx=features[split]
        differences[split]={'old_minus_direct':stats(old[r[idx],c[idx]]-direct[r[idx],c[idx]]),
            'direct_minus_independent':stats(direct[r[idx],c[idx]]-independent[idx]),
            'old_minus_independent':stats(old[r[idx],c[idx]]-independent[idx]),
            'new_nonwater_centres':int((direct[r[idx],c[idx]]>=0).sum()),
            'new_missing_centres':int((~np.isfinite(direct[r[idx],c[idx]])).sum())}
    bbox=transform_bounds('EPSG:4326',crs,76.3,8.9,76.55,9.2,densify_pts=21)
    # Native-cell bilinear support overlap: pixel non-overlap alone is insufficient.
    cellsets={}
    for split in ('train_idx','val_idx','test_idx'):
        s=set()
        for i in features[split]:
            a=int(np.searchsorted(lat,lats[i])-1); b=int(np.searchsorted(lon,lons[i])-1)
            s.update((a+dr,b+dc) for dr in (0,1) for dc in (0,1))
        cellsets[split]=s
    report={'source_attributes':source_attrs,'source_shape':values.shape,'source_resolution_degrees':[dx,dy],
            'source_transform':list(source_transform),'intermediate':intermediate,'training_grid':grid,
            'all_corner_projected_aoi_bounds':bbox,'shared_native_support_cells':{
                'train_validation':len(cellsets['train_idx']&cellsets['val_idx']),
                'train_test':len(cellsets['train_idx']&cellsets['test_idx'])},
            'water_pixel_old_minus_direct':stats((old-direct)[mask&(old<0)&np.isfinite(direct)]),
            'split_comparisons':differences}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
    profile.update(count=1)
    with rasterio.open(OUT/'gebco_direct.tif','w',**profile) as dst:
        dst.write(direct,1); dst.set_band_description(1,'gebco_elevation_direct_m')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,5))
    for ax,arr,title in zip(axes,(old,direct,old-direct),
                            ('Old elevation','Direct native interpolation','Old minus direct')):
        im=ax.imshow(arr[::8,::8],cmap='coolwarm',vmin=-2 if 'minus' in title else -50,
                     vmax=2 if 'minus' in title else 10)
        ax.set_title(title); ax.set_xlabel('Column / 8'); ax.set_ylabel('Row / 8')
        fig.colorbar(im,ax=ax,label='metres',shrink=.7)
    fig.suptitle('Same Sentinel target grid; differences are label processing, not surveyed error')
    fig.tight_layout(); fig.savefig(OUT/'grid_comparison.png',dpi=140); plt.close(fig)
    print(json.dumps(report,indent=2,default=str),flush=True)


if __name__=='__main__': main()
