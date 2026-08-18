"""
Two tabular baselines to compare against later:
  1. Stumpf (2003) linear band-ratio regression — the classic empirical SDB method
  2. Random Forest on raw bands + ratio + NDWI — a stronger empirical ML baseline

Usage:
    python -m src.baseline_models --config config.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("baseline_models")


def metrics(y_true, y_pred) -> dict:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    abs_diff = np.abs(y_p - y_t)
    rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
    mae = float(np.mean(abs_diff))
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    
    # Depth Accuracy metrics against GEBCO ground truth:
    # 1. Percentage of predictions within +/- 1 metre of GEBCO depth
    acc_1m = float(np.mean(abs_diff <= 1.0) * 100.0)
    # 2. Percentage of predictions within +/- 2 metres of GEBCO depth
    acc_2m = float(np.mean(abs_diff <= 2.0) * 100.0)
    # 3. Standard depth estimation threshold metric: max(y/y_hat, y_hat/y) < 1.25
    pos = (y_t > 0.1) & (y_p > 0.1)
    delta_1 = float(np.mean(np.maximum(y_t[pos] / y_p[pos], y_p[pos] / y_t[pos]) < 1.25) * 100.0) if np.any(pos) else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "acc_within_1m_%": acc_1m,
        "acc_within_2m_%": acc_2m,
        "delta_1.25_%": delta_1,
    }


def train_stumpf(train_df: pd.DataFrame, test_df: pd.DataFrame):
    model = LinearRegression()
    model.fit(train_df[["stumpf_ratio"]], train_df["depth_m"])
    preds = model.predict(test_df[["stumpf_ratio"]])
    m = metrics(test_df["depth_m"], preds)
    LOG.info("Stumpf baseline       -> RMSE %.3f  MAE %.3f  R2 %.3f", m["rmse"], m["mae"], m["r2"])
    return model, m


def train_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    max_samples = 100_000 if len(train_df) > 100_000 else None
    model = RandomForestRegressor(
        n_estimators=100, max_depth=None, min_samples_leaf=3,
        max_samples=max_samples,
        n_jobs=-1, random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["depth_m"])
    preds = model.predict(test_df[feature_cols])
    m = metrics(test_df["depth_m"], preds)
    LOG.info("RandomForest baseline -> RMSE %.3f  MAE %.3f  R2 %.3f", m["rmse"], m["mae"], m["r2"])
    return model, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    feat_dir = cfg["features"]["features_dir"]
    results_dir = cfg["evaluation"]["results_dir"]
    ensure_dir(results_dir)
    ensure_dir(cfg["training"]["checkpoint_dir"])

    train_df = pd.read_csv(Path(feat_dir) / "train_pixels.csv")
    test_df = pd.read_csv(Path(feat_dir) / "test_pixels.csv")

    bands = cfg["sentinel2"]["bands"]
    feature_cols = [c for c in [*bands, "stumpf_ratio", "ndwi"] if c in train_df.columns]

    stumpf_model, stumpf_metrics = train_stumpf(train_df, test_df)
    rf_model, rf_metrics = train_random_forest(train_df, test_df, feature_cols)

    ckpt_dir = cfg["training"]["checkpoint_dir"]
    joblib.dump(stumpf_model, Path(ckpt_dir) / "stumpf_model.joblib")
    joblib.dump(rf_model, Path(ckpt_dir) / "rf_model.joblib")

    results = {"stumpf": stumpf_metrics, "random_forest": rf_metrics}
    with open(Path(results_dir) / "tabular_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    LOG.info("Saved results to %s/tabular_baseline_results.json", results_dir)


if __name__ == "__main__":
    main()
