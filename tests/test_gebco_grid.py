import tempfile
from pathlib import Path
import unittest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from src.preprocess import resample_gebco, get_target_grid


class GebcoGridTests(unittest.TestCase):
    def test_pixel_centres_axis_order_and_native_preference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.nc'
            # Latitude ascending, longitude descending: catch flips/transposition.
            ds=xr.Dataset({'elevation':(('latitude','longitude'), [[-12.,-11.,-10.],[-22.,-21.,-20.]])},
                          coords={'latitude':[10.,11.], 'longitude':[22.,21.,20.]})
            ds.elevation.attrs['units']='m'; ds.to_netcdf(path)
            got=resample_gebco(str(path.with_suffix('.tif')),from_origin(19.5,11.5,1,1),'EPSG:4326',(2,3))
            np.testing.assert_allclose(got,[[-20,-21,-22],[-10,-11,-12]])

    def test_missing_coverage_remains_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.nc'
            ds=xr.Dataset({'elevation':(('latitude','longitude'), np.full((2,2),-5.))},
                          coords={'latitude':[10.,11.], 'longitude':[20.,21.]})
            ds.elevation.attrs['units']='m'; ds.to_netcdf(path)
            got=resample_gebco(str(path),from_origin(0,1,1,1),'EPSG:4326',(2,2))
            self.assertTrue(np.isnan(got).all())

    def test_grid_encloses_all_corners_at_exact_resolution(self):
        bbox=[76.3,8.9,76.55,9.2]
        t,crs,shape=get_target_grid(bbox,'EPSG:32643')
        left,bottom,right,top=transform_bounds('EPSG:4326',crs,*bbox,densify_pts=21)
        self.assertEqual(t.a,10); self.assertEqual(t.e,-10)
        self.assertLessEqual(t.c,left); self.assertGreaterEqual(t.f,top)
        self.assertGreaterEqual(t.c+shape[1]*10,right)
        self.assertLessEqual(t.f-shape[0]*10,bottom)


if __name__=='__main__': unittest.main()
