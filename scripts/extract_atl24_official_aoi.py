"""Extract all ATL24 V002 photons inside the fixed project AOI from verified HDF5.

Preserves classifications and quality flags. Does not change existing training data.
"""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd

OUT = Path("runs/2026-11-05/nearshore_error_reference_audit_v1")
BBOX = (76.30, 8.90, 76.55, 9.20)
BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
FIELDS = ("class_ph", "confidence", "delta_time", "ellipse_h", "index_ph",
          "index_seg", "invalid_kd", "invalid_wind_speed", "lat_ph", "lon_ph",
          "low_confidence_flag", "night_flag", "ortho_h",
          "sensor_depth_exceeded", "sigma_thu", "sigma_tvu", "surface_h",
          "x_atc", "y_atc")


def main():
    files = sorted((OUT / "atl24_v002_official").glob("*.h5"))
    if len(files) != 4:
        raise RuntimeError(f"Expected four verified V002 files; found {len(files)}")
    frames = []
    counts = []
    west, south, east, north = BBOX
    for path in files:
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError(f"Invalid HDF5 signature: {path.name}")
        with h5py.File(path, "r") as h5:
            for beam in BEAMS:
                group = h5[beam]
                lon, lat = group["lon_ph"][:], group["lat_ph"][:]
                inside = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
                counts.append({"granule": path.name, "beam": beam,
                               "photon_rows_in_granule_beam": int(len(lat)),
                               "photon_rows_inside_aoi": int(inside.sum())})
                if not inside.any():
                    continue
                frame = pd.DataFrame({field: group[field][:][inside] for field in FIELDS})
                frame["granule"] = path.name
                frame["beam"] = beam
                frame["depth_m"] = frame.surface_h - frame.ortho_h
                frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = OUT / "atl24_v002_aoi_all_photons.parquet"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite official AOI extraction: {path}")
    raw.to_parquet(path, index=False)
    shallow = ((raw.class_ph == 40) & (raw.confidence >= .6) &
               np.isfinite(raw.depth_m) & (raw.depth_m > 0) & (raw.depth_m <= 20))
    report = {"source": "Official NSIDC ATL24 V002 HDF5 (four CMR bbox-intersecting granules)",
              "aoi_bbox_lonlat": list(BBOX), "all_output_rows_inside_aoi": bool(
                  raw.lon_ph.between(west, east).all() and raw.lat_ph.between(south, north).all()),
              "all_aoi_photons": int(len(raw)),
              "class_counts": {str(k): int(v) for k, v in raw.class_ph.value_counts().items()},
              "bathymetry_class_confidence_ge_0_6_depth_0_20": int(shallow.sum()),
              "granule_beam_counts": counts,
              "note": "No depth labels or splits modified. Photon and quality checks follow."}
    (OUT / "atl24_v002_extraction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("all_aoi_photons", "class_counts",
                                                   "bathymetry_class_confidence_ge_0_6_depth_0_20")}, indent=2))


if __name__ == "__main__":
    main()
