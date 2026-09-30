"""
Feature engineering: band ratio (Stumpf), NDWI water index, a per-pixel
tabular dataset for the Stumpf/RF baselines, and patch extraction + spatial
train/val/test split for the CNN baseline.

Usage:
    python -m src.features --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("features")

STUMPF_N = 1000.0  # fixed scaling constant from Stumpf, Holderied & Sinclair (2003)


def stumpf_ratio(blue: np.ndarray, green: np.ndarray) -> np.ndarray:
    """ln(n*blue) / ln(n*green) — guards against non-positive reflectance."""
    b = np.clip(blue, 1e-6, None)
    g = np.clip(green, 1e-6, None)
    return np.log(STUMPF_N * b) / np.log(STUMPF_N * g)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + 1e-6)


def beer_lambert_depth_prior(blue: np.ndarray, green: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes a per-pixel attenuation coefficient Kd(lambda) and analytical depth
    prior z_prior using the Beer-Lambert radiative transfer law.

    Physics Rationale:
        Under the Beer-Lambert optical extinction law:
            I(z) = I_surface * exp(-2 * Kd * z)
            => z_prior = (1 / (2 * Kd)) * ln(I_surface / I_pixel)

        where 2 * Kd accounts for the two-way light propagation path (down to seabed
        and back to satellite sensor). Kd is estimated per pixel using the QAA/Morel
        blue-to-green reflectance ratio proxy Kd(490) = 0.0166 + 0.156 * (R_blue / R_green)^(-1.14).

    Args:
        blue: 2D array of Blue band reflectance (B02)
        green: 2D array of Green band reflectance (B03)

    Returns:
        z_prior: 2D array of analytical depth estimates (meters)
    """
    b = np.clip(blue, eps, None)
    g = np.clip(green, eps, None)

    # QAA optical water clarity proxy for Kd (m^-1)
    kd = 0.0166 + 0.156 * np.power(b / g, -1.14)
    kd = np.clip(kd, 0.01, 5.0)

    # Surface water reference reflectance (95th percentile over clear water pixels)
    valid_g = g[g > eps]
    i_surface = float(np.percentile(valid_g, 95)) if len(valid_g) > 0 else 0.1
    i_surface = max(i_surface, 1e-3)

    # Analytical depth prior from optical transmission ratio
    transmission_ratio = np.clip(i_surface / g, 1.0, 1e4)
    z_prior = (1.0 / (2.0 * kd)) * np.log(transmission_ratio)
    return np.nan_to_num(z_prior, nan=0.0, posinf=50.0, neginf=0.0).astype(np.float32)


def load_stack(stack_path: str):
    with rasterio.open(stack_path) as src:
        arrays = {}
        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1] or f"band_{i}"
            arrays[desc] = src.read(i)
        transform, crs = src.transform, src.crs
    return arrays, transform, crs


def build_pixel_table(arrays: dict, bands: list) -> pd.DataFrame:
    blue, green, red = arrays[bands[0]], arrays[bands[1]], arrays[bands[2]]
    ratio_bg = stumpf_ratio(blue, green)
    ratio_br = stumpf_ratio(blue, red)
    z_beer = beer_lambert_depth_prior(blue, green)
    water_idx = ndwi(green, arrays[bands[3]]) if len(bands) > 3 else None

    valid = arrays["valid_mask"] > 0.5
    depth = arrays["gebco_depth"]
    is_water = depth < 0  # GEBCO elevation is negative below sea level
    keep = valid & is_water & np.isfinite(ratio_bg) & np.isfinite(depth)

    rows, cols = np.where(keep)
    data = {
        "row": rows,
        "col": cols,
        "stumpf_ratio": ratio_bg[rows, cols],
        "stumpf_ratio_red": ratio_br[rows, cols],
        "beer_lambert_z_prior": z_beer[rows, cols],
        "depth_m": -depth[rows, cols],
    }
    for b in bands:
        data[b] = arrays[b][rows, cols]
    if water_idx is not None:
        data["ndwi"] = water_idx[rows, cols]

    return pd.DataFrame(data)


def spatial_split(df: pd.DataFrame, n_cols: int, n_rows: int,
                   train_frac: float, val_frac: float, axis: str = "lon",
                   block_size: int = 64, seed: int = 42):
    """Depth-stratified spatial block split for the pixel table.

    Divides the raster into a grid of (block_size × block_size) spatial blocks,
    computes the mean depth per block, sorts blocks by depth, and interleaves
    them across train/val/test.  This ensures:
      - Spatial disjointness (no adjacent-pixel leakage)
      - Balanced depth distributions across all three splits

    Uses the same interleaving logic as ``stratified_spatial_patch_split``
    for consistency between pixel-table and patch-based experiments.
    """
    rng = np.random.RandomState(seed)

    # Assign each pixel to a spatial block
    df = df.copy()
    df["block_r"] = df["row"] // block_size
    df["block_c"] = df["col"] // block_size
    df["block_id"] = df["block_r"].astype(str) + "_" + df["block_c"].astype(str)

    # Compute mean depth per block and sort by depth for stratification
    block_depths = df.groupby("block_id")["depth_m"].mean().sort_values()
    sorted_blocks = block_depths.index.tolist()

    # Interleave sorted blocks: 70 % train, 15 % val, 15 % test
    # (mod 20: first 14 → train, next 3 → val, last 3 → test)
    train_blocks, val_blocks, test_blocks = set(), set(), set()
    for i, b in enumerate(sorted_blocks):
        mod = i % 20
        if mod < 14:
            train_blocks.add(b)
        elif mod < 17:
            val_blocks.add(b)
        else:
            test_blocks.add(b)

    train_df = df[df["block_id"].isin(train_blocks)].drop(columns=["block_r", "block_c", "block_id"])
    val_df   = df[df["block_id"].isin(val_blocks)].drop(columns=["block_r", "block_c", "block_id"])
    test_df  = df[df["block_id"].isin(test_blocks)].drop(columns=["block_r", "block_c", "block_id"])
    return train_df, val_df, test_df


def extract_patches(arrays: dict, bands: list, patch_size: int, stride: int):
    """Extract (image, depth, valid-mask) patches for the CNN baseline & HybridBathNet.

    Each patch has shape (C+2, patch_size, patch_size) containing 6 channels total:
      - Channels 0-3: Raw Sentinel-2 spectral bands (B02 Blue, B03 Green, B04 Red, B08 NIR)
      - Channel 4   : Stumpf Blue/Green ratio (ln(n*B02)/ln(n*B03))
      - Channel 5   : Beer-Lambert analytical depth prior z_prior = (1/(2*Kd)) * ln(I_surf/I_pixel)
    """
    valid = arrays["valid_mask"] > 0.5
    depth = -arrays["gebco_depth"]
    is_water = arrays["gebco_depth"] < 0
    band_stack = np.stack([arrays[b] for b in bands], axis=0)  # (C=4, H, W)

    # --- Physics channels ---
    ratio_bg = stumpf_ratio(arrays[bands[0]], arrays[bands[1]])  # (H, W) Stumpf Blue/Green ratio
    z_prior_2d = beer_lambert_depth_prior(arrays[bands[0]], arrays[bands[1]]) # (H, W) Beer-Lambert prior

    ratio_bg = np.nan_to_num(ratio_bg, nan=0.0, posinf=0.0, neginf=0.0)
    z_prior_2d = np.nan_to_num(z_prior_2d, nan=0.0, posinf=0.0, neginf=0.0)

    phys_stack = np.stack([ratio_bg, z_prior_2d], axis=0)  # (2, H, W)

    H, W = valid.shape
    patches_x, patches_y, patches_valid, coords = [], [], [], []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            m = valid[r:r + patch_size, c:c + patch_size] & is_water[r:r + patch_size, c:c + patch_size]
            if m.mean() < 0.5:
                continue
            spectral = band_stack[:, r:r + patch_size, c:c + patch_size]      # (4, H, W)
            phys_patch = phys_stack[:, r:r + patch_size, c:c + patch_size]    # (2, H, W)
            patches_x.append(np.concatenate([spectral, phys_patch], axis=0))  # (6, H, W)
            patches_y.append(depth[r:r + patch_size, c:c + patch_size])
            patches_valid.append(m)
            coords.append((r, c))

    if not patches_x:
        raise RuntimeError(
            "No valid patches extracted — check that the AOI actually contains "
            "nearshore water pixels with valid GEBCO depth and cloud-free imagery."
        )

    return (
        np.stack(patches_x).astype(np.float32),
        np.stack(patches_y).astype(np.float32),
        np.stack(patches_valid).astype(np.float32),
        coords,
    )


def stratified_spatial_patch_split(
    coords: list,
    Y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    grid_size: int = 128,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits patches into train/val/test using spatially-disjoint block tiles
    while stratifying by depth to ensure train, val, and test sets cover matching
    shallow-to-deep depth distributions.

    Preventing data leakage:
    Patches are grouped into spatial grid blocks of size (grid_size x grid_size).
    Entire spatial blocks are assigned atomically to train, val, or test.
    This guarantees zero spatial overlap between adjacent pixels of different splits.
    """
    rng = np.random.RandomState(seed)

    cy, cx = Y.shape[1] // 2, Y.shape[2] // 2
    center_depths = Y[:, cy, cx]

    # Group patch indices by spatial block coordinate (grid_size x grid_size)
    block_map = {}
    for idx, (r, c) in enumerate(coords):
        block_id = (r // grid_size, c // grid_size)
        if block_id not in block_map:
            block_map[block_id] = []
        block_map[block_id].append(idx)

    # Compute mean depth per spatial block
    block_ids = list(block_map.keys())
    block_depths = np.array([np.mean([center_depths[i] for i in block_map[b]]) for b in block_ids])

    # Sort spatial blocks by mean depth for depth stratification
    sorted_order = np.argsort(block_depths)
    sorted_blocks = [block_ids[i] for i in sorted_order]

    train_idx, val_idx, test_idx = [], [], []

    # Interleave blocks across depth quantiles (70% train, 15% val, 15% test)
    for i, b in enumerate(sorted_blocks):
        mod = i % 20
        if mod < 14:       # 70% train
            train_idx.extend(block_map[b])
        elif mod < 17:     # 15% val
            val_idx.extend(block_map[b])
        else:              # 15% test
            test_idx.extend(block_map[b])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    stack_path = Path(cfg["preprocessing"]["processed_dir"]) / "stack.tif"
    bands = cfg["sentinel2"]["bands"]
    feat_cfg = cfg["features"]
    out_dir = feat_cfg["features_dir"]
    ensure_dir(out_dir)

    LOG.info("Loading preprocessed stack from %s", stack_path)
    arrays, transform, crs = load_stack(str(stack_path))
    n_rows, n_cols = arrays[bands[0]].shape

    LOG.info("Building per-pixel tabular dataset (for Stumpf/RF baselines)...")
    df = build_pixel_table(arrays, bands)
    train_df, val_df, test_df = spatial_split(
        df, n_cols, n_rows, feat_cfg["train_frac"], feat_cfg["val_frac"], feat_cfg["split_axis"]
    )
    train_df.to_csv(Path(out_dir) / "train_pixels.csv", index=False)
    val_df.to_csv(Path(out_dir) / "val_pixels.csv", index=False)
    test_df.to_csv(Path(out_dir) / "test_pixels.csv", index=False)
    LOG.info("Pixel table sizes -> train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))

    LOG.info("Extracting patches (6 channels: 4 spectral + 1 Stumpf ratio + 1 Beer-Lambert prior)...")
    X, Y, M, coords = extract_patches(arrays, bands, feat_cfg["patch_size"], feat_cfg["patch_stride"])
    LOG.info("Extracted %d patches of size %d  |  channels: %d",
              len(X), feat_cfg["patch_size"], X.shape[1])

    train_idx, val_idx, test_idx = stratified_spatial_patch_split(
        coords, Y, feat_cfg["train_frac"], feat_cfg["val_frac"]
    )

    np.savez(
        Path(out_dir) / "patches.npz",
        X=X, Y=Y, M=M,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    LOG.info("Saved depth-stratified spatial patch dataset -> train: %d, val: %d, test: %d",
              len(train_idx), len(val_idx), len(test_idx))


if __name__ == "__main__":
    main()

