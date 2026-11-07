"""Check downloaded ATL24 photons against the frozen nearshore grid and splits.

The result is an exploratory reference audit, never a relabeling/training step.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


ROOT = Path("runs/2026-11-05/nearshore_hybrid_v1")
OUT = Path("runs/2026-11-05/nearshore_error_reference_audit_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    stem = "atl24_archive" if args.archive else "atl24"
    points = pd.read_parquet(OUT / f"{stem}_bathymetry_raw.parquet").copy()
    with rasterio.open("runs/2026-11-04/processed/stack.tif") as scene:
        x, y = Transformer.from_crs("EPSG:4326", scene.crs, always_xy=True).transform(
            points.lon_ph.to_numpy(), points.lat_ph.to_numpy())
        rr, cc = rowcol(scene.transform, x, y)
        rr, cc = np.asarray(rr), np.asarray(cc)
        inside = (rr >= 0) & (cc >= 0) & (rr < scene.height) & (cc < scene.width)
        valid_composite = np.full(len(points), False)
        for i in np.flatnonzero(inside):
            valid_composite[i] = scene.read(10, window=((int(rr[i]), int(rr[i])+1),
                                                        (int(cc[i]), int(cc[i])+1)))[0, 0] > .5
    with xr.open_dataset("data/raw/gebco/gebco_subset.nc") as ds:
        lat = ds.latitude.values
        lon = ds.longitude.values
        elev = ds.elevation.transpose("latitude", "longitude").values
    if lat[0] > lat[-1]:
        lat, elev = lat[::-1], elev[::-1]
    if lon[0] > lon[-1]:
        lon, elev = lon[::-1], elev[:, ::-1]
    gebco = -RegularGridInterpolator((lat, lon), elev, bounds_error=False,
                                    fill_value=np.nan)(points[["lat_ph", "lon_ph"]].to_numpy())
    with np.load(ROOT / "features/patches.npz") as patches:
        coords = patches["coords"] + 32
        split_indices = {s: patches[f"{s}_idx"] for s in ("train", "val", "test")}
        split_depths = patches["Y"][:, 32, 32]
    spatial = np.column_stack([rr, cc])
    near = {}
    for split, ids in split_indices.items():
        dist, match = cKDTree(coords[ids]).query(spatial)
        near[split] = {"nearest_centre_m": dist * 10,
                       "nearest_centre_reference_m": split_depths[ids][match]}
    nominal = np.isfinite(points.depth_m) & (points.depth_m > 0) & (points.depth_m <= 20)
    high = nominal & inside & valid_composite & (points.confidence >= .6)
    bbox = (points.lon_ph.between(76.30, 76.55) & points.lat_ph.between(8.90, 9.20)).to_numpy()
    # Group count by 2 km spatial cells rather than calling correlated photons independent.
    groups = spatial // 200
    report = {"source": f"SlideRule atl24x response ({'2018–Nov 2024' if args.archive else '2024 Jan–Apr'}, bathymetry class, confidence>=0.6); not NSIDC V002 download",
              "requested_bbox_lonlat": [76.30, 8.90, 76.55, 9.20],
              "all_returned_inside_requested_bbox": bool(np.all(bbox)),
              "returned_lon_range": [float(points.lon_ph.min()), float(points.lon_ph.max())],
              "returned_lat_range": [float(points.lat_ph.min()), float(points.lat_ph.max())],
              "total_photons": len(points), "finite_0_20m": int(nominal.sum()),
              "returned_time_range_utc": [str(points.index.min()), str(points.index.max())],
              "finite_0_20m_inside_scene_valid_composite": int(high.sum()),
              "finite_0_20m_with_gebco_0_25m": int((high & (gebco >= 0) & (gebco <= 25)).sum()),
              "atl24_10_15m_with_gebco_0_25m": int((high & (points.depth_m >= 10) &
                                                       (points.depth_m < 15) & (gebco >= 0) &
                                                       (gebco <= 25)).sum()),
              "unique_tracks": points[["rgt", "cycle", "region"]].drop_duplicates().to_dict("records"),
              "nominal_depth_quantiles_m": np.quantile(points.depth_m[nominal], [.1, .5, .9]).tolist() if nominal.any() else [],
              "valid_quality_flag_values": points.quality_ph.value_counts(dropna=False).to_dict(),
              "median_vertical_uncertainty_m": float(np.median(points.sigma_tvu[high])) if high.any() else None,
              "independent_spatial_group_count": int(len(np.unique(groups[high], axis=0))),
              "geographic_match": {},
              "caveat": "ATL24 depths are sea-surface minus refracted-bottom orthometric heights. GEBCO may represent a different datum/epoch and coarse cells. Nearby-photon differences are not a model score. Preserve original geographic test groups and all GEBCO labels."}
    for split in ("train", "val", "test"):
        dist = near[split]["nearest_centre_m"]
        report["geographic_match"][split] = {
            "photons_0_20m_within_50m": int((high & (dist <= 50)).sum()),
            "photons_0_20m_within_100m": int((high & (dist <= 100)).sum()),
            "photons_0_20m_within_320m": int((high & (dist <= 320)).sum()),
            "median_nearest_distance_m": float(np.median(dist[high])) if high.any() else None}
    coloc = high & np.isfinite(gebco)
    dif = points.depth_m.to_numpy()[coloc] - gebco[coloc]
    report["gebco_colocation"] = {
        "n": int(coloc.sum()), "median_atl24_minus_gebco_m": float(np.median(dif)) if len(dif) else None,
        "median_absolute_difference_m": float(np.median(np.abs(dif))) if len(dif) else None,
        "rmse_difference_m": float(np.sqrt(np.mean(dif**2))) if len(dif) else None,
        "warning": "Photon counts are highly correlated and this is not independent validation of the trained network."}
    points["gebco_depth_m"] = gebco
    points["inside_stack"] = inside
    points["composite_valid_any_mask"] = valid_composite
    points["grid_row"] = rr
    points["grid_col"] = cc
    points["depth_0_20m_inside_valid_composite"] = high
    for split in near:
        points[f"distance_to_{split}_centre_m"] = near[split]["nearest_centre_m"]
    points.to_parquet(OUT / f"{stem}_bathymetry_audited.parquet", index=True)
    (OUT / f"{stem}_reference_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
