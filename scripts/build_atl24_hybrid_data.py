"""Build a distinct optical-input/ATL24-depth HybridBathNet dataset for the AOI.

No GEBCO depths or water labels enter the features or targets. Original geographic
block identities are reused; all dates at a pixel aggregate before assignment.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import (load_stack, sun_glint_correction, lyzenga_channels,
                          stumpf_ratio, beer_lambert_depth_prior)

ROOT = Path("runs/2026-11-06/atl24_hybrid_v1")
SOURCE = Path("runs/2026-11-05/nearshore_error_reference_audit_v1/atl24_archive_with_original_split.parquet")


def far_from(points, held, pixels=96):
    if not len(points) or not len(held):
        return np.ones(len(points), dtype=bool)
    distance, _ = cKDTree(held).query(points, p=np.inf)
    return distance >= pixels


def main():
    out = ROOT / "features"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "patches.npz"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite dataset {path}")
    raw = pd.read_parquet(SOURCE)
    # Quality criteria are declared before fitting a model. Median aggregation
    # precedes the supervised 0–20 m domain cutoff to avoid truncated medians.
    good = ((raw.class_ph == 40) & (raw.confidence >= .6) & (raw.quality_ph == 0) &
            np.isfinite(raw.depth_m) & (raw.depth_m > 0) & (raw.depth_m < 100) &
            np.isfinite(raw.sigma_tvu) & (raw.sigma_tvu <= 1) &
            raw.inside_stack & raw.composite_valid_any_mask &
            raw.original_geo_split.isin(["train", "val", "test"]))
    raw = raw.loc[good].copy()
    raw["acquisition_year"] = raw.index.year
    aggregated = raw.groupby(["grid_row", "grid_col", "original_geo_split"], as_index=False).agg(
        depth_m=("depth_m", "median"), photon_count=("depth_m", "size"),
        confidence=("confidence", "median"), sigma_tvu=("sigma_tvu", "median"),
        first_year=("acquisition_year", "min"), last_year=("acquisition_year", "max"))
    aggregated = aggregated.loc[aggregated.depth_m.between(.01, 20)].copy()
    arrays, _, _ = load_stack("runs/2026-11-04/processed/stack.tif")
    arrays.pop("gebco_depth", None)  # explicit: no GEBCO target or mask use
    blue, green, red, nir = [arrays[k] for k in ("B02", "B03", "B04", "B08")]
    valid = (arrays["valid_mask"] > .5) & np.isfinite(blue) & np.isfinite(green)
    valid &= np.isfinite(red) & np.isfinite(nir) & (blue > 0) & (green > 0)
    valid &= (green - nir) > 0  # simple optical-water proxy; not depth cutoff
    h, w = valid.shape
    centres = aggregated[["grid_row", "grid_col"]].to_numpy(dtype=np.int64)
    bounds = ((centres[:, 0] >= 32) & (centres[:, 1] >= 32) &
              (centres[:, 0] + 32 <= h) & (centres[:, 1] + 32 <= w))
    eligible = []
    for i in np.flatnonzero(bounds):
        r, c = centres[i]
        if valid[r, c] and valid[r-32:r+32, c-32:c+32].mean() >= .5:
            eligible.append(i)
    aggregated = aggregated.iloc[eligible].reset_index(drop=True)
    centres = centres[eligible]
    # Preserve every eligible test point. Exclude val near test and train near
    # both holdouts so 64px image footprints have a 32px gap across splits.
    test = np.flatnonzero(aggregated.original_geo_split.to_numpy() == "test")
    val0 = np.flatnonzero(aggregated.original_geo_split.to_numpy() == "val")
    val = val0[far_from(centres[val0], centres[test])]
    train0 = np.flatnonzero(aggregated.original_geo_split.to_numpy() == "train")
    train = train0[far_from(centres[train0], centres[np.r_[val, test]])]
    kept = np.sort(np.r_[train, val, test])
    if min(len(train), len(val), len(test)) < 20:
        raise RuntimeError(f"Insufficient buffered spatial split: {len(train)}/{len(val)}/{len(test)}")
    if np.any(~far_from(centres[train], centres[np.r_[val, test]])):
        raise AssertionError("Training patch buffer failed")
    if np.any(~far_from(centres[val], centres[test])):
        raise AssertionError("Validation/test patch buffer failed")
    fit = np.zeros((h, w), dtype=bool)
    for r, c in centres[train]:
        fit[r-32:r+32, c-32:c+32] = True
    fit &= valid
    blue_c, green_c, red_c = sun_glint_correction(blue, green, red, nir, fit)
    rinf = [float(np.percentile(a[fit], 2)) for a in (blue_c, green_c, red_c)]
    lyz = lyzenga_channels(blue_c, green_c, red_c, *rinf)
    prior = beer_lambert_depth_prior(blue_c, green_c, fit_mask=fit)
    layers = [blue_c, green_c, red_c, nir, *lyz, stumpf_ratio(blue_c, green_c),
              prior, valid.astype(np.float32),
              *[arrays[f"{band}_std"] for band in ("B02", "B03", "B04", "B08")]]
    feature_stack = np.stack([np.nan_to_num(layer, nan=0., posinf=0., neginf=0.)
                             for layer in layers]).astype(np.float32)
    x = np.stack([feature_stack[:, r-32:r+32, c-32:c+32] for r, c in centres[kept]])
    m = np.stack([valid[r-32:r+32, c-32:c+32] for r, c in centres[kept]]).astype(np.float32)
    # Centre-only supervision: fill storage with the same scalar to keep the
    # centre label invariant under even-sized-patch augmentation flips.
    y = np.broadcast_to(aggregated.depth_m.to_numpy(dtype=np.float32)[kept, None, None],
                        (len(kept), 64, 64)).copy()
    top_left = centres[kept] - 32
    remap = {int(old): new for new, old in enumerate(kept)}
    groups = {f"{name}_idx": np.array([remap[int(i)] for i in indices], dtype=np.int64)
              for name, indices in (("train", train), ("val", val), ("test", test))}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite engineered features or ATL24 targets")
    np.savez(path, X=x, Y=y, M=m, coords=top_left,
             photon_count=aggregated.photon_count.to_numpy()[kept],
             first_year=aggregated.first_year.to_numpy()[kept],
             last_year=aggregated.last_year.to_numpy()[kept], **groups)
    meta = {"source": str(SOURCE), "target": "median ATL24 bathymetry photon depth per 10m grid pixel",
            "target_screen": "class40, confidence>=0.6, quality_ph=0, sigma_tvu<=1m; median then 0-20m",
            "optical_mask": "composite valid-any + finite positive blue/green + green>NIR; no GEBCO",
            "feature_fitting": "glint, reference reflectance and prior fitted on training footprints only",
            "input_channels": 16, "rinf_train_only": rinf,
            "spatial_assignment": "original 128px train/val/test block IDs; test kept; 96px centre gap across splits",
            "split_sizes": {name: len(indices) for name, indices in groups.items()},
            "pre_buffer_eligible_sizes": {name: int((aggregated.original_geo_split == name).sum())
                                          for name in ("train", "val", "test")},
            "split_bins": {name: {f"{lo}-{hi}": int(((y[indices, 32, 32] >= lo) &
                                                    (y[indices, 32, 32] < hi)).sum())
                                  for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20.0001))}
                           for name, indices in groups.items()},
            "caveats": ["Public SlideRule archive subset may miss granules that emitted invalid-dataframe alerts.",
                        "2018–2024 depth points paired to Jan–Mar 2024 Sentinel composite; seabed change and vertical reference need review.",
                        "Within-split overlapping pixels/tracks are correlated; this is not a 10m independent survey grid.",
                        "Y stores a constant per-patch scalar solely for centre-supervised trainer augmentation."]}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"split_sizes": meta["split_sizes"], "split_bins": meta["split_bins"]}, indent=2))


if __name__ == "__main__":
    main()
