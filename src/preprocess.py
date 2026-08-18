"""
Preprocess raw Sentinel-2 SAFE products into a cloud-masked, co-registered,
temporally-composited multi-band raster stack aligned with the GEBCO depth
grid, ready for feature engineering.

Usage:
    python -m src.preprocess --config config.yaml
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from pyproj import Transformer

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("preprocess")

# Sentinel-2 L2A Scene Classification Layer (SCL) codes:
#   0 no data, 1 saturated/defective, 2 dark area, 3 cloud shadow,
#   4 vegetation, 5 not vegetated, 6 water, 7 unclassified,
#   8 cloud (medium prob), 9 cloud (high prob), 10 thin cirrus, 11 snow
# "Clear" = usable land/water pixels, excluding cloud/shadow/no-data/snow.
SCL_CLEAR_CLASSES = {4, 5, 6, 7}
SCL_WATER_CLASS = 6


def unzip_all(raw_dir: str) -> list:
    raw_dir = Path(raw_dir)
    safe_dirs = list(raw_dir.glob("*.SAFE"))
    for zpath in sorted(raw_dir.glob("*.zip")):
        target_name = zpath.stem if zpath.stem.endswith(".SAFE") else f"{zpath.stem}.SAFE"
        existing = list(raw_dir.glob(target_name))
        if existing:
            for d in existing:
                if d not in safe_dirs:
                    safe_dirs.append(d)
            continue
        LOG.info("Unzipping %s (%.0f MB)...", zpath.name, zpath.stat().st_size / 1e6)
        try:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(raw_dir)
        except zipfile.BadZipFile:
            LOG.warning(
                "Skipping %s — not a valid ZIP (still downloading or corrupt).",
                zpath.name,
            )
            continue
        for d in raw_dir.glob(target_name):
            if d not in safe_dirs:
                safe_dirs.append(d)
    return sorted(list(set(safe_dirs)))


def find_band_path(safe_dir: Path, band: str, resolution: str = "10m") -> Path:
    """Locate a .jp2 band file inside a .SAFE product structure."""
    candidates = list(safe_dir.rglob(f"*_{band}_{resolution}.jp2"))
    if not candidates:
        candidates = list(safe_dir.rglob(f"*_{band}.jp2"))
    if not candidates:
        raise FileNotFoundError(f"Band {band} not found in {safe_dir}")
    return candidates[0]


def find_scl_path(safe_dir: Path) -> Path:
    candidates = list(safe_dir.rglob("*_SCL_20m.jp2"))
    if not candidates:
        raise FileNotFoundError(f"SCL band not found in {safe_dir}")
    return candidates[0]


def get_target_grid(bbox_geo: list, dst_crs: str, resolution: float = 10.0):
    """Compute target transform, CRS, and shape covering the AOI bbox."""
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    minx, miny = transformer.transform(bbox_geo[0], bbox_geo[1])
    maxx, maxy = transformer.transform(bbox_geo[2], bbox_geo[3])
    dst_width = int(np.ceil((maxx - minx) / resolution))
    dst_height = int(np.ceil((maxy - miny) / resolution))
    dst_transform = from_bounds(minx, miny, maxx, maxy, dst_width, dst_height)
    dst_shape = (dst_height, dst_width)
    return dst_transform, dst_crs, dst_shape


def load_scene_onto_grid(safe_dir: Path, bands: list, dst_transform, dst_crs, dst_shape):
    """Load requested bands + SCL mask for one scene, reprojected to target AOI grid."""
    stack = []
    for b in bands:
        path = find_band_path(safe_dir, b, "10m")
        with rasterio.open(path) as src:
            dst = np.zeros(dst_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        stack.append(dst)
    stack = np.stack(stack, axis=0)  # (bands, H, W)

    scl_path = find_scl_path(safe_dir)
    with rasterio.open(scl_path) as src:
        scl_dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=scl_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    # Valid clear pixels: SCL class in clear classes and valid non-zero reflectance
    clear_mask = np.isin(np.round(scl_dst).astype(int), list(SCL_CLEAR_CLASSES)) & (stack[0] > 0)
    return stack, clear_mask


def temporal_composite(safe_dirs: list, bands: list, dst_transform, dst_crs, dst_shape):
    """Median composite across all cloud-free pixels from all scenes."""
    all_stacks, all_masks = [], []

    for safe_dir in safe_dirs:
        try:
            stack, clear_mask = load_scene_onto_grid(safe_dir, bands, dst_transform, dst_crs, dst_shape)
        except FileNotFoundError as e:
            LOG.warning("Skipping %s: %s", safe_dir.name, e)
            continue

        stack_masked = np.where(clear_mask[None, :, :], stack, np.nan)
        all_stacks.append(stack_masked)
        all_masks.append(clear_mask)

    if not all_stacks:
        raise RuntimeError("No usable scenes found — check downloads and cloud filter.")

    cube = np.stack(all_stacks, axis=0)          # (time, bands, H, W)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(cube, axis=0)    # (bands, H, W)
    valid_any = np.any(np.stack(all_masks, axis=0), axis=0)
    return composite, valid_any


def resample_gebco(gebco_path: str, ref_transform, ref_crs, ref_shape):
    with rasterio.open(gebco_path) as src:
        depth = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=depth,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
    return depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_dir = cfg["sentinel2"]["raw_dir"]
    bands = cfg["sentinel2"]["bands"]
    processed_dir = cfg["preprocessing"]["processed_dir"]
    ensure_dir(processed_dir)

    LOG.info("Searching Sentinel-2 SAFE products...")
    safe_dirs = unzip_all(raw_dir)
    LOG.info("Found %d SAFE products", len(safe_dirs))
    if not safe_dirs:
        raise SystemExit(f"No .SAFE products found under {raw_dir} — run download_sentinel2.py first.")

    bbox = cfg["aoi"]["bbox"]
    dst_crs = cfg["project"]["crs"]
    transform, crs, shape = get_target_grid(bbox, dst_crs, resolution=10.0)
    LOG.info("Target AOI grid: %s (HxW), CRS: %s", shape, crs)

    LOG.info("Building cloud-free median composite across %d scenes (bands: %s)...", len(safe_dirs), bands)
    composite, valid_mask = temporal_composite(safe_dirs, bands, transform, crs, shape)

    LOG.info("Resampling GEBCO depth grid onto the Sentinel-2 composite grid...")
    depth = resample_gebco(cfg["gebco"]["raw_path"], transform, crs, shape)

    out_profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": len(bands) + 2,  # bands + depth + valid_mask
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }
    out_path = Path(processed_dir) / "stack.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        for i, b in enumerate(bands, start=1):
            band_arr = composite[i - 1]
            dst.write(np.nan_to_num(band_arr, nan=0.0), i)
            dst.set_band_description(i, b)
        dst.write(depth, len(bands) + 1)
        dst.set_band_description(len(bands) + 1, "gebco_depth")
        dst.write(valid_mask.astype(np.float32), len(bands) + 2)
        dst.set_band_description(len(bands) + 2, "valid_mask")

    LOG.info("Wrote preprocessed stack to %s", out_path)


if __name__ == "__main__":
    main()
