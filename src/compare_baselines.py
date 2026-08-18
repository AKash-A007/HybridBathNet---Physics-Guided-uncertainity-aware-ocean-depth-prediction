"""
Combine tabular + CNN baseline results into one comparison table and plot.

Usage:
    python -m src.compare_baselines --config config.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import load_config, get_logger

LOG = get_logger("compare_baselines")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_dir = Path(cfg["evaluation"]["results_dir"])

    with open(results_dir / "tabular_baseline_results.json") as f:
        tab = json.load(f)
    with open(results_dir / "cnn_baseline_results.json") as f:
        cnn = json.load(f)

    rows = [
        {"model": "Stumpf (2003)", **tab["stumpf"]},
        {"model": "Random Forest", **tab["random_forest"]},
        {"model": "Simple CNN", **cnn},
    ]
    df = pd.DataFrame(rows).set_index("model")
    LOG.info("\n%s", df.to_string())
    df.to_csv(results_dir / "baseline_comparison.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    df["rmse"].plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Baseline comparison — nearshore depth RMSE")
    plt.tight_layout()
    fig.savefig(results_dir / "baseline_comparison.png", dpi=150)
    LOG.info("Saved comparison table and plot to %s", results_dir)


if __name__ == "__main__":
    main()
