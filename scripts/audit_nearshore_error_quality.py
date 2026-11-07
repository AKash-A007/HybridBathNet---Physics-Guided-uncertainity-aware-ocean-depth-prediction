"""Exploratory, read-only error/quality audit of the frozen nearshore v1 run.

Composite proxies are descriptive. This script never selects or filters samples.
"""
from pathlib import Path
import json
import sys

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import xy as pixel_xy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import find_scl_path


ROOT = Path("runs/2026-11-05/nearshore_hybrid_v1")
OUT = Path("runs/2026-11-05/nearshore_error_reference_audit_v1")


def summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "features/patches.npz") as data:
        coords = data["coords"].copy()
        x = data["X"]
        y = data["Y"]
        mask = data["M"] > .5
        indices = {split: data[f"{split}_idx"].copy() for split in ("train", "val", "test")}
        centre = (32, 32)
        # X uses glint-corrected visible bands, raw NIR and composite std.
        blue, green, red, nir = [x[:, i, *centre].astype(float) for i in range(4)]
        std_bg = np.hypot(x[:, 12, *centre], x[:, 13, *centre]).astype(float)
        valid_fraction = mask.mean(axis=(1, 2))
        # Patch-scale interpolation variation, not an independent GEBCO quality flag.
        depth_range = np.array([np.ptp(y[i][mask[i]]) if mask[i].any() else np.nan
                                for i in range(len(x))])
        depth = y[:, *centre].astype(float)
    with rasterio.open("runs/2026-11-04/processed/stack.tif") as source:
        transform = source.transform
        crs = str(source.crs)
    xx, yy = pixel_xy(transform, (coords + 32)[:, 0], (coords + 32)[:, 1])
    # Sample native SCL at each patch centre, retaining one classification per
    # acquisition date. Multiple SAFE tiles can cover the same date/location.
    scl_by_date = {}
    for safe in sorted(Path("data/raw/sentinel2").glob("*.SAFE")):
        date = safe.name.split("_")[2][:8]
        with rasterio.open(find_scl_path(safe)) as scl:
            sx, sy = Transformer.from_crs(crs, scl.crs, always_xy=True).transform(xx, yy)
            values = np.fromiter((int(v[0]) for v in scl.sample(zip(sx, sy))),
                                 dtype=np.int16, count=len(coords))
        if date in scl_by_date:
            previous = scl_by_date[date]
            previous[previous == 0] = values[previous == 0]
        else:
            scl_by_date[date] = values
    scl = np.stack(list(scl_by_date.values()))
    scene_available = (scl != 0).sum(axis=0)
    scene_clear = np.isin(scl, [4, 5, 6, 7]).sum(axis=0)
    scene_water = (scl == 6).sum(axis=0)
    scene_cloud_shadow = np.isin(scl, [3, 8, 9, 10]).sum(axis=0)
    ndwi = (green - nir) / (green + nir + 1e-6)
    # High NIR / red may indicate glint, turbidity or bottom/shore effects.
    # They cannot uniquely label any of these mechanisms.
    proxies = {"composite_blue": blue, "composite_green": green,
               "composite_red": red, "composite_nir": nir,
               "composite_ndwi": ndwi, "composite_bg_temporal_std_l2": std_bg,
               "valid_patch_fraction": valid_fraction,
               "patch_gebco_depth_range_m": depth_range,
               "scl_available_scene_count": scene_available,
               "scl_clear_scene_count": scene_clear,
               "scl_water_scene_count": scene_water,
               "scl_cloud_shadow_scene_count": scene_cloud_shadow}
    out = {"scope": "Frozen v1. No data, model, split, or predictions modified.",
           "reference": "GEBCO interpolated from ~15-arc-second source; no native quality/type flag in saved patches.",
           "proxy_caveat": "SCL counts are centre classifications across distinct dates, not a whole-patch image-quality assessment. Temporal standard deviation is across-date variability, not sensor noise; NIR/red/NDWI do not independently distinguish glint, turbidity, bottom type or cloud.",
           "scene_dates": sorted(scl_by_date),
           "crs": crs, "grid_transform": list(transform)[:6], "splits": {}}
    for split, idx in indices.items():
        split_data = {}
        bin_mask = (depth[idx] >= 10) & (depth[idx] < 15)
        split_data["10_15_m"] = {"n": int(bin_mask.sum()),
                                "proxy_summary": {name: summary(values[idx][bin_mask])
                                                  for name, values in proxies.items()}}
        split_data["other_0_20_m"] = {
            "n": int(((depth[idx] >= 0) & (depth[idx] <= 20) & ~bin_mask).sum()),
            "proxy_summary": {name: summary(values[idx][(depth[idx] >= 0) &
                                                              (depth[idx] <= 20) & ~bin_mask])
                              for name, values in proxies.items()}}
        if split in ("val", "test"):
            stem = "validation" if split == "val" else "test"
            with np.load(ROOT / f"{stem}_predictions.npz") as pred:
                target = pred["target"].astype(float)
                prediction = pred["prediction"].astype(float)
            np.testing.assert_allclose(target, depth[idx], rtol=1e-5, atol=1e-4)
            error = prediction - target
            xy = coords[idx] + 32
            groups = xy // 200  # ~2 km grid cells, descriptive only.
            selected = np.flatnonzero(bin_mask)
            group_rows = []
            for group in np.unique(groups[selected], axis=0):
                members = selected[np.all(groups[selected] == group, axis=1)]
                group_rows.append({"grid_group": group.tolist(), "n": int(len(members)),
                                   "bias_m": float(error[members].mean()),
                                   "rmse_m": float(np.sqrt(np.mean(error[members] ** 2))),
                                   "squared_error_fraction_of_bin": float(
                                       np.sum(error[members] ** 2) / np.sum(error[selected] ** 2)),
                                   "proxy_medians": {name: float(np.median(values[idx][members]))
                                                     for name, values in proxies.items()}})
            group_rows.sort(key=lambda row: row["squared_error_fraction_of_bin"], reverse=True)
            split_data["10_15_m"].update({"rmse_m": float(np.sqrt(np.mean(error[selected] ** 2))),
                                         "bias_m": float(error[selected].mean()),
                                         "grid_group_count": len(group_rows),
                                         "groups": group_rows,
                                         "top_four_squared_error_fraction": float(sum(
                                             row["squared_error_fraction_of_bin"] for row in group_rows[:4]))})
        out["splits"][split] = split_data
    (OUT / "quality_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({split: {"n_10_15": row["10_15_m"]["n"],
                              "top_four_error_fraction": row["10_15_m"].get("top_four_squared_error_fraction"),
                              "nir_10_15": row["10_15_m"]["proxy_summary"]["composite_nir"]["median"]}
                      for split, row in out["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
