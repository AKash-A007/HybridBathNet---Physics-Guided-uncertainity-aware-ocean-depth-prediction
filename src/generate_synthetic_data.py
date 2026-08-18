"""
Generate realistic synthetic Sentinel-2 + GEBCO-like data for the AOI defined
in config.yaml. This bypasses both the Sentinel-2 download AND the manual GEBCO
step so the full pipeline can run end-to-end immediately.

The synthetic stack written to data/processed/stack.tif has the same band layout
as the real pipeline output from preprocess.py:
    band 1..N  : Sentinel-2 reflectance bands (B02 blue, B03 green, B04 red, B08 NIR)
    band N+1   : gebco_depth  (negative below sea level, in metres)
    band N+2   : valid_mask   (1.0 = usable cloud-free water pixel)

Spatial model used:
  - A ~500 x 500 pixel grid (10 m resolution, similar to S2 L2A)
  - Depth gradient: 0 m at a synthetic shoreline → -30 m offshore (west)
  - Sentinel-2 reflectance: spectrally-tuned shallow-to-deep water values +
    per-band Gaussian noise to mimic real coastal imagery
  - Turbidity/bottom type variation added via spatially-correlated noise fields.

Usage:
    python -m src.generate_synthetic_data --config config.yaml
    python -m src.generate_synthetic_data --config config.yaml --size 300
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("generate_synthetic_data")

# S2 L2A typical reflectance (0–1 scale after /10000 DN scaling) per band for
# coastal water at different depths — used to build a realistic gradient.
# Values are approximate means from published coastal S2 studies.
#   [blue, green, red, NIR]  at surface  (shallow <1 m)
_SURF_SHALLOW = np.array([0.070, 0.055, 0.030, 0.005])
#   at ~15 m depth
_SURF_MID     = np.array([0.045, 0.040, 0.020, 0.003])
#   at >25 m depth
_SURF_DEEP    = np.array([0.025, 0.022, 0.010, 0.002])
#   land pixels (not used for training but included for context)
_LAND         = np.array([0.120, 0.100, 0.090, 0.140])

# Noise standard deviations per band
_NOISE_STD    = np.array([0.006, 0.005, 0.004, 0.003])


def _smooth_noise(shape, scale, rng):
    """Return spatially-correlated noise via low-pass filtering with a box blur."""
    raw = rng.standard_normal(shape).astype(np.float32)
    # Simple box-blur repeated to approximate Gaussian smoothing
    from scipy.ndimage import uniform_filter
    return uniform_filter(raw, size=scale)


def generate_synthetic_stack(
    n_rows: int = 512,
    n_cols: int = 512,
    bands: list = None,
    seed: int = 42,
) -> tuple:
    """
    Returns:
        composite  : (n_bands, n_rows, n_cols) float32 reflectance array
        depth      : (n_rows, n_cols) float32 — negative below sea, metres
        valid_mask : (n_rows, n_cols) bool    — True where usable water pixel
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B08"]

    rng = np.random.default_rng(seed)

    # --- Depth model ---
    # Shoreline runs diagonally: pixels where col < row*0.9 + n_cols*0.15 are land.
    row_idx, col_idx = np.meshgrid(
        np.arange(n_rows, dtype=np.float32),
        np.arange(n_cols, dtype=np.float32),
        indexing="ij",
    )
    # Shoreline approximately at col = n_cols * 0.35
    shore_col = n_cols * 0.35

    dist_from_shore = (col_idx - shore_col) / (n_cols - shore_col)   # 0 at shore, 1 at far end
    dist_from_shore = np.clip(dist_from_shore, 0, 1)

    # Add spatially correlated noise to the shoreline shape
    from scipy.ndimage import uniform_filter
    shore_noise = uniform_filter(
        rng.standard_normal((n_rows, n_cols)).astype(np.float32), size=40
    ) * 0.08
    dist_noisy = np.clip(dist_from_shore + shore_noise, 0, 1)

    # Depth: 0 at shore → -30 m at max distance, with patch-scale variation
    max_depth = -30.0
    depth_noise = uniform_filter(
        rng.standard_normal((n_rows, n_cols)).astype(np.float32), size=60
    ) * 3.0
    depth = (max_depth * dist_noisy + depth_noise).astype(np.float32)

    # Land mask: west side of the shoreline
    is_land = col_idx < (shore_col + shore_noise * n_cols * 0.5)
    depth[is_land] = 5.0  # positive elevation on land

    # Valid mask: cloud-free water only
    # Simulate a few cloud patches using random ellipses
    valid = ~is_land  # start: all water pixels valid
    n_cloud_patches = rng.integers(2, 6)
    for _ in range(n_cloud_patches):
        cr = rng.integers(0, n_rows)
        cc = rng.integers(int(shore_col), n_cols)
        rr = rng.integers(20, 60)
        rc = rng.integers(20, 80)
        cloud_mask = (
            ((row_idx - cr) / rr) ** 2 + ((col_idx - cc) / rc) ** 2
        ) < 1.0
        valid &= ~cloud_mask

    LOG.info(
        "Depth range: %.1f m to %.1f m | Water pixels: %d | Cloudy pixels: %d",
        depth[~is_land].min(), depth[~is_land].max(),
        valid.sum(), (~is_land & ~valid).sum(),
    )

    # --- Reflectance model ---
    n_bands = len(bands)
    composite = np.zeros((n_bands, n_rows, n_cols), dtype=np.float32)

    # Normalised depth in [0, 1] (0 = shallow, 1 = deep)
    depth_norm = np.clip(-depth / abs(max_depth), 0, 1)
    depth_norm[is_land] = 0.0

    for bi in range(n_bands):
        # Trilinear interpolation through shallow / mid / deep water spectra
        shallow = _SURF_SHALLOW[bi]
        mid     = _SURF_MID[bi]
        deep    = _SURF_DEEP[bi]

        # Piecewise: 0-0.5 → shallow to mid, 0.5-1.0 → mid to deep
        refl = np.where(
            depth_norm < 0.5,
            shallow + (mid - shallow) * (depth_norm / 0.5),
            mid    + (deep - mid)    * ((depth_norm - 0.5) / 0.5),
        ).astype(np.float32)

        # Add turbidity / bottom-type spatial variation
        turbid = uniform_filter(
            rng.standard_normal((n_rows, n_cols)).astype(np.float32), size=80
        ) * 0.010
        refl += turbid

        # Per-pixel Gaussian instrument noise
        noise = rng.normal(0, _NOISE_STD[bi], size=(n_rows, n_cols)).astype(np.float32)
        refl += noise

        # Land reflectance
        refl[is_land] = _LAND[bi] + rng.normal(0, 0.015, size=is_land.sum()).astype(np.float32)

        # Scale to S2 DN range (0–10000) and clip
        composite[bi] = np.clip(refl * 10000, 0, 10000)

    return composite, depth, valid.astype(np.float32)


def make_affine_transform(bbox, n_rows, n_cols):
    """Return a rasterio Affine transform for the given bounding box."""
    return from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], n_cols, n_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--size", type=int, default=512,
                        help="Grid dimension in pixels (NxN, default 512).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bands  = cfg["sentinel2"]["bands"]
    bbox   = cfg["aoi"]["bbox"]          # [minlon, minlat, maxlon, maxlat]
    crs_str = cfg["project"]["crs"]
    processed_dir = cfg["preprocessing"]["processed_dir"]
    gebco_out_tif = cfg["gebco"]["raw_path"]

    n = args.size
    LOG.info("Generating %d×%d synthetic coastal scene (bands: %s)...", n, n, bands)

    composite, depth, valid_mask = generate_synthetic_stack(
        n_rows=n, n_cols=n, bands=bands, seed=args.seed
    )

    # Use a simple geographic transform (lat/lon CRS) since we're not doing
    # real reprojection in the synthetic path.
    transform = make_affine_transform(bbox, n, n)
    crs = CRS.from_epsg(4326)   # WGS-84 for synthetic data (pipeline reads CRS from file)

    # --- Write stack.tif (output of preprocess.py) ---
    ensure_dir(processed_dir)
    stack_path = Path(processed_dir) / "stack.tif"
    profile = {
        "driver":    "GTiff",
        "height":    n,
        "width":     n,
        "count":     len(bands) + 2,
        "dtype":     "float32",
        "crs":       crs,
        "transform": transform,
        "nodata":    float("nan"),
        "compress":  "lzw",
    }
    with rasterio.open(stack_path, "w", **profile) as dst:
        for i, b in enumerate(bands, start=1):
            dst.write(composite[i - 1], i)
            dst.set_band_description(i, b)
        dst.write(depth, len(bands) + 1)
        dst.set_band_description(len(bands) + 1, "gebco_depth")
        dst.write(valid_mask, len(bands) + 2)
        dst.set_band_description(len(bands) + 2, "valid_mask")
    LOG.info("Wrote synthetic stack → %s", stack_path)

    # --- Write gebco_subset.tif (raw GEBCO GeoTIFF expected by preprocess.py) ---
    ensure_dir(str(Path(gebco_out_tif).parent))
    gebco_profile = {
        "driver":    "GTiff",
        "height":    n,
        "width":     n,
        "count":     1,
        "dtype":     "float32",
        "crs":       crs,
        "transform": transform,
        "nodata":    float("nan"),
        "compress":  "lzw",
    }
    with rasterio.open(gebco_out_tif, "w", **gebco_profile) as dst:
        dst.write(depth, 1)
        dst.set_band_description(1, "elevation")
    LOG.info("Wrote synthetic GEBCO GeoTIFF → %s", gebco_out_tif)

    LOG.info(
        "Done. Run the rest of the pipeline with:\n"
        "  python run_baseline_pipeline.py --config config.yaml --synthetic\n"
        "  (or start from features: python -m src.features --config config.yaml)"
    )


if __name__ == "__main__":
    main()
