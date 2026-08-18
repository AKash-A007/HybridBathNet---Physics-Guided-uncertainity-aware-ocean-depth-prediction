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


def load_stack(stack_path: str):
    with rasterio.open(stack_path) as src:
        arrays = {}
        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1] or f"band_{i}"
            arrays[desc] = src.read(i)
        transform, crs = src.transform, src.crs
    return arrays, transform, crs


def build_pixel_table(arrays: dict, bands: list) -> pd.DataFrame:
    blue, green = arrays[bands[0]], arrays[bands[1]]
    ratio = stumpf_ratio(blue, green)
    water_idx = ndwi(green, arrays[bands[3]]) if len(bands) > 3 else None

    valid = arrays["valid_mask"] > 0.5
    depth = arrays["gebco_depth"]
    is_water = depth < 0  # GEBCO elevation is negative below sea level
    keep = valid & is_water & np.isfinite(ratio) & np.isfinite(depth)

    rows, cols = np.where(keep)
    data = {
        "row": rows,
        "col": cols,
        "stumpf_ratio": ratio[rows, cols],
        "depth_m": -depth[rows, cols],
    }
    for b in bands:
        data[b] = arrays[b][rows, cols]
    if water_idx is not None:
        data["ndwi"] = water_idx[rows, cols]

    return pd.DataFrame(data)


def spatial_split(df: pd.DataFrame, n_cols: int, n_rows: int,
                   train_frac: float, val_frac: float, axis: str = "lon"):
    """Split by column (proxy for longitude) or row (proxy for latitude) so
    train/val/test cover spatially disjoint regions — avoids leakage between
    adjacent pixels of the same stretch of coastline."""
    key = "col" if axis == "lon" else "row"
    n = n_cols if axis == "lon" else n_rows
    train_cut = int(n * train_frac)
    val_cut = int(n * (train_frac + val_frac))

    train_df = df[df[key] < train_cut]
    val_df = df[(df[key] >= train_cut) & (df[key] < val_cut)]
    test_df = df[df[key] >= val_cut]
    return train_df, val_df, test_df


def extract_patches(arrays: dict, bands: list, patch_size: int, stride: int):
    """Extract (image, depth, valid-mask) patches for the CNN baseline.

    Each patch has shape (C+1, patch_size, patch_size) where the first C
    channels are the raw Sentinel-2 spectral bands and the final channel is
    the Stumpf log-ratio physics prior (ln(n*blue)/ln(n*green)), computed
    by reusing the existing ``stumpf_ratio`` function in this module.
    """
    valid = arrays["valid_mask"] > 0.5
    depth = -arrays["gebco_depth"]
    is_water = arrays["gebco_depth"] < 0
    band_stack = np.stack([arrays[b] for b in bands], axis=0)  # (C, H, W)

    # --- Physics channel: Stumpf log-ratio (reusing the function above) ---
    # bands[0] = B02 (blue), bands[1] = B03 (green) per config.yaml convention.
    ratio_2d = stumpf_ratio(arrays[bands[0]], arrays[bands[1]])  # (H, W)
    # Guard against NaN/Inf that can appear at masked-out pixels.
    ratio_2d = np.nan_to_num(ratio_2d, nan=0.0, posinf=0.0, neginf=0.0)

    H, W = valid.shape
    patches_x, patches_y, patches_valid, coords = [], [], [], []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            m = valid[r:r + patch_size, c:c + patch_size] & is_water[r:r + patch_size, c:c + patch_size]
            if m.mean() < 0.5:
                continue
            spectral = band_stack[:, r:r + patch_size, c:c + patch_size]        # (C, H, W)
            ratio_patch = ratio_2d[r:r + patch_size, c:c + patch_size][None]    # (1, H, W)
            patches_x.append(np.concatenate([spectral, ratio_patch], axis=0))   # (C+1, H, W)
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

    LOG.info("Extracting patches (for CNN baseline, with Stumpf ratio channel)...")
    X, Y, M, coords = extract_patches(arrays, bands, feat_cfg["patch_size"], feat_cfg["patch_stride"])
    LOG.info("Extracted %d patches of size %d  |  channels: %d (bands=%d + stumpf_ratio=1)",
              len(X), feat_cfg["patch_size"], X.shape[1], len(bands))

    cols = np.array([c for _, c in coords])
    train_cut = int(n_cols * feat_cfg["train_frac"])
    val_cut = int(n_cols * (feat_cfg["train_frac"] + feat_cfg["val_frac"]))
    train_idx = np.where(cols < train_cut)[0]
    val_idx = np.where((cols >= train_cut) & (cols < val_cut))[0]
    test_idx = np.where(cols >= val_cut)[0]

    np.savez(
        Path(out_dir) / "patches.npz",
        X=X, Y=Y, M=M,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    LOG.info("Saved patch dataset -> train: %d, val: %d, test: %d",
              len(train_idx), len(val_idx), len(test_idx))


if __name__ == "__main__":
    main()
