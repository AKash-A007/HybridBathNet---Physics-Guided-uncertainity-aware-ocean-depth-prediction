"""
Download OpenStreetMap coastline data for the AOI using the Overpass API,
and save it as a GeoJSON of coastline LineStrings (used for ROI/land masking
context, not a hard model input).

Usage:
    python -m src.download_coastline --config config.yaml
"""
import argparse
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("download_coastline")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_coastline(bbox: list) -> gpd.GeoDataFrame:
    """bbox = [minlon, minlat, maxlon, maxlat]"""
    minlon, minlat, maxlon, maxlat = bbox
    query = f"""
    [out:json][timeout:180];
    (
      way["natural"="coastline"]({minlat},{minlon},{maxlat},{maxlon});
    );
    out geom;
    """
    # Overpass requires a User-Agent header (returns 406 without one)
    headers = {"User-Agent": "HybridBathNet/1.0 (coastal bathymetry research)"}
    resp = requests.get(
        OVERPASS_URL,
        params={"data": query},
        headers=headers,
        timeout=200,
    )
    resp.raise_for_status()
    data = resp.json()

    lines = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) >= 2:
            lines.append(LineString(coords))

    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    bbox = cfg["aoi"]["bbox"]
    out_path = cfg["coastline"]["raw_path"]

    LOG.info("Querying Overpass API for coastline in bbox %s", bbox)
    gdf = fetch_coastline(bbox)
    LOG.info("Retrieved %d coastline segments", len(gdf))
    if len(gdf) == 0:
        LOG.warning("No coastline segments returned — check bbox order/values.")
        return

    ensure_dir(str(Path(out_path).parent))
    gdf.to_file(out_path, driver="GeoJSON")
    LOG.info("Saved coastline to %s", out_path)


if __name__ == "__main__":
    main()
