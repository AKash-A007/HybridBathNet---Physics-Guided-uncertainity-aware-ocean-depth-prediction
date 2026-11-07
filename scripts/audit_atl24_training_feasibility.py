"""Count archive ATL24 labels under the existing geographic block assignment.

No examples are removed from the original GEBCO benchmark. This is a feasibility
screen for a distinct ATL24 experiment, before any training/selection.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

OUT = Path("runs/2026-11-05/nearshore_error_reference_audit_v1")


def main():
    points = pd.read_parquet(OUT / "atl24_archive_bathymetry_audited.parquet")
    with np.load("runs/2026-11-05/improvement/features/patches.npz") as old:
        coords = old["coords"]
        blocks = {name: {tuple(row) for row in np.unique(coords[old[f"{name}_idx"]] // 128,
                                                      axis=0)} for name in ("train", "val", "test")}
    if blocks["train"] & blocks["val"] or blocks["train"] & blocks["test"] or blocks["val"] & blocks["test"]:
        raise AssertionError("Historical geographic blocks overlap")
    keys = [tuple(row) for row in (points[["grid_row", "grid_col"]].to_numpy() // 128)]
    assigned = np.array([next((split for split, split_blocks in blocks.items()
                              if key in split_blocks), "other") for key in keys])
    points["original_geo_split"] = assigned
    points["spatial_2km_group"] = [f"{r}:{c}" for r, c in
                                   (points[["grid_row", "grid_col"]].to_numpy() // 200)]
    # Declare before inspecting any model result; threshold is a product field,
    # not selected against validation/test residuals. quality_ph=0 is a stricter
    # sensitivity screen, pending official field interpretation.
    base = (points.depth_m.between(.01, 20) & points.inside_stack &
            points.composite_valid_any_mask & (points.confidence >= .6) &
            np.isfinite(points.sigma_tvu) & (points.sigma_tvu <= 1))
    screens = {"basic": base, "quality_ph_zero": base & (points.quality_ph == 0),
               "2022_plus_quality_zero": base & (points.quality_ph == 0) &
               (points.index >= pd.Timestamp("2022-01-01"))}
    report = {"source": "SlideRule current ATL24 archive AOI subset, 2018–Nov 2024",
              "screens": {},
              "caveats": ["Public SlideRule query emitted invalid-dataframe alerts for some granules; returned set may be incomplete.",
                          "Repeated photons/dates along the same RGT are correlated. Unique row/col cells and 2km groups are more informative than photon count.",
                          "Historical block assignment is descriptive; it does not yet construct leak-free ATL24 patches or certify ground truth."]}
    for label, screen in screens.items():
        counts = {}
        for split in ("train", "val", "test", "other"):
            subset = points.loc[screen & (assigned == split)]
            counts[split] = {"photons": len(subset),
                             "unique_10m_pixels": len(subset[["grid_row", "grid_col"]].drop_duplicates()),
                             "spatial_2km_groups": int(subset.spatial_2km_group.nunique()),
                             "unique_rgt": int(subset.rgt.nunique()),
                             "depth_0_5": int(subset.depth_m.between(0, 5, inclusive="left").sum()),
                             "depth_5_10": int(subset.depth_m.between(5, 10, inclusive="left").sum()),
                             "depth_10_15": int(subset.depth_m.between(10, 15, inclusive="left").sum()),
                             "depth_15_20": int(subset.depth_m.between(15, 20, inclusive="both").sum())}
        report["screens"][label] = counts
    points.to_parquet(OUT / "atl24_archive_with_original_split.parquet", index=True)
    (OUT / "atl24_training_feasibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["screens"], indent=2))


if __name__ == "__main__":
    main()
