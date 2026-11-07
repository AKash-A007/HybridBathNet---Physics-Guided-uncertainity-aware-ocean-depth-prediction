"""Fetch an exploratory ATL24 photon subset via SlideRule; never change training data.

Run only after verifying service terms and availability. Source ATL24 v3 may differ
from NSIDC v2 CMR candidates. Keep version/source in the output metadata.
"""
from pathlib import Path
import argparse
import json
import pandas as pd
from sliderule import sliderule

OUT = Path("runs/2026-11-05/nearshore_error_reference_audit_v1")
POLY = [{"lon": 76.30, "lat": 8.90}, {"lon": 76.55, "lat": 8.90},
        {"lon": 76.55, "lat": 9.20}, {"lon": 76.30, "lat": 9.20},
        {"lon": 76.30, "lat": 8.90}]
PARAMS = {"poly": POLY, "t0": "2024-01-01T00:00:00Z",
          "t1": "2024-04-30T23:59:59Z",
          "atl24": {"compact": False, "class_ph": [40], "confidence_threshold": 0.6}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="store_true", help="2018–Nov 2024 AOI subset")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "atl24_archive" if args.archive else "atl24"
    parms = json.loads(json.dumps(PARAMS))
    if args.archive:
        parms["t0"] = "2018-10-14T00:00:00Z"
        parms["t1"] = "2024-11-07T23:59:59Z"
    path = OUT / f"{stem}_bathymetry_raw.parquet"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    sliderule.init("slideruleearth.io", verbose=False)
    frame = sliderule.run("atl24x", parms)
    print("returned rows", len(frame), "columns", list(frame.columns), flush=True)
    if not len(frame):
        (OUT / f"{stem}_fetch_status.json").write_text(json.dumps({
            "status": "service_returned_no_photons", "parameters": parms,
            "note": "A granule footprint is not evidence of usable bathymetry photons."}, indent=2), encoding="utf-8")
        return
    if "geometry" in frame:
        frame["lon_ph"] = frame.geometry.x
        frame["lat_ph"] = frame.geometry.y
        frame = pd.DataFrame(frame.drop(columns="geometry"))
    for key in ("lon_ph", "lat_ph", "ortho_h", "surface_h", "confidence"):
        if key not in frame:
            raise ValueError(f"SlideRule response missing {key}; columns: {list(frame.columns)}")
    frame["depth_m"] = frame["surface_h"] - frame["ortho_h"]
    frame.to_parquet(path, index=True)
    status = {"status": "downloaded_unvalidated", "parameters": parms,
              "source": "SlideRule atl24x; compare service granule version to NSIDC CMR V002 before publication",
              "rows": len(frame), "columns": list(frame.columns),
              "depth_0_20_count": int(((frame.depth_m > 0) & (frame.depth_m <= 20)).sum()),
              "note": "Photon points are spatially correlated; do not train or score before quality, datum, temporal and split checks."}
    (OUT / f"{stem}_fetch_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
