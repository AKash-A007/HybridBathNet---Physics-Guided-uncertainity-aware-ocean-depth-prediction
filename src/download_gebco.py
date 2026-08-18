"""
Download GEBCO 2020 bathymetry for the AOI via the NOAA CoastWatch ERDDAP
public API, then reproject it into a GeoTIFF aligned to your project CRS.

No manual download step required — this script fully automates the GEBCO
acquisition using the ERDDAP griddap endpoint:
  https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2020

If you already have a local GEBCO NetCDF file, pass --nc-path to skip the
download and go straight to the reprojection step.

Usage:
    python -m src.download_gebco --config config.yaml
    python -m src.download_gebco --config config.yaml --nc-path /path/to/my.nc
"""
import argparse
from pathlib import Path

import requests
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)
import xarray as xr

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("download_gebco")

# NOAA CoastWatch ERDDAP — publicly accessible, no auth required.
# GEBCO_2020 grid, variable name "elevation", dims (latitude, longitude).
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2020.nc"


def download_gebco_erddap(bbox: list, out_nc: str) -> None:
    """Download a GEBCO subset for the given bbox from NOAA ERDDAP.

    Args:
        bbox: [minlon, minlat, maxlon, maxlat] in EPSG:4326.
        out_nc: Local path to write the downloaded NetCDF file.
    """
    minlon, minlat, maxlon, maxlat = bbox
    url = (
        f"{ERDDAP_BASE}"
        f"?elevation[({minlat}):1:({maxlat})][({minlon}):1:({maxlon})]"
    )
    LOG.info("Downloading GEBCO 2020 subset from NOAA ERDDAP...")
    LOG.info("  bbox: [%.4f, %.4f, %.4f, %.4f]", minlon, minlat, maxlon, maxlat)
    LOG.info("  URL : %s", url)

    ensure_dir(str(Path(out_nc).parent))
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    with open(out_nc, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    size_kb = Path(out_nc).stat().st_size / 1e3
    LOG.info("Saved %.1f KB -> %s", size_kb, out_nc)


def convert_gebco_netcdf_to_geotiff(nc_path: str, out_tif: str, dst_crs: str) -> None:
    """Reproject a GEBCO NetCDF to a GeoTIFF in the target project CRS."""
    ds = xr.open_dataset(nc_path)
    # ERDDAP GEBCO uses "elevation"; manual GEBCO downloads use "elevation" too.
    var_name = "elevation" if "elevation" in ds.data_vars else list(ds.data_vars)[0]
    LOG.info("Converting variable '%s' -> %s (CRS: %s)", var_name, out_tif, dst_crs)
    da = ds[var_name]
    da = da.rio.write_crs("EPSG:4326", inplace=True)
    da_proj = da.rio.reproject(dst_crs)
    ensure_dir(str(Path(out_tif).parent))
    da_proj.rio.to_raster(out_tif)
    LOG.info("Wrote reprojected GEBCO GeoTIFF to %s", out_tif)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--nc-path", default=None,
        help="Path to an already-downloaded GEBCO NetCDF file. "
             "If omitted, the file is downloaded automatically via ERDDAP.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    dst_crs = cfg["project"]["crs"]
    out_tif = cfg["gebco"]["raw_path"]
    nc_path = args.nc_path or cfg["gebco"]["raw_path"].replace(".tif", ".nc")

    if not Path(nc_path).exists():
        bbox = cfg["aoi"]["bbox"]
        download_gebco_erddap(bbox, nc_path)
    else:
        LOG.info("Using existing GEBCO NetCDF: %s", nc_path)

    convert_gebco_netcdf_to_geotiff(nc_path, out_tif, dst_crs)


if __name__ == "__main__":
    main()
