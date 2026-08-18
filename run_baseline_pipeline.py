"""
End-to-end baseline pipeline orchestrator.

Before running, make sure you've:
  1. Set CDSE_USERNAME / CDSE_PASSWORD environment variables (or they're set
     in download_sentinel2.py directly)
  2. Either downloaded GEBCO manually OR pass --synthetic to skip all downloads
     and use synthetic coastal data instead.

Usage:
    python run_baseline_pipeline.py --config config.yaml
    python run_baseline_pipeline.py --config config.yaml --synthetic     # no downloads needed
    python run_baseline_pipeline.py --config config.yaml --skip-download # use existing raw data
"""
import argparse
import subprocess
import sys


def run(cmd, stage_name=""):
    label = stage_name or " ".join(cmd[-2:])
    print(f"\n{'='*60}")
    print(f"  STAGE: {label}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n  ✓ {label} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ {label} FAILED with exit code {e.returncode}.")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download steps (use existing data/raw/*)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help=(
            "Generate synthetic Sentinel-2 + GEBCO data instead of downloading "
            "real data. Lets you run the full pipeline immediately without any "
            "internet access or manual GEBCO download."
        )
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip preprocess step (use existing data/processed/stack.tif)"
    )
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_download:
        run([py, "-m", "src.download_sentinel2", "--config", args.config],
            "Download Sentinel-2 (real tiles from CDSE)")
        run([py, "-m", "src.download_gebco",     "--config", args.config],
            "Download + convert GEBCO (auto via NOAA ERDDAP)")
        run([py, "-m", "src.download_coastline", "--config", args.config],
            "Download OSM coastline")

    if not args.skip_preprocess:
        run([py, "-m", "src.preprocess", "--config", args.config], "Preprocess")

    run([py, "-m", "src.features",          "--config", args.config], "Feature engineering")
    run([py, "-m", "src.baseline_models",   "--config", args.config], "Tabular baselines (Stumpf + RF)")
    run([py, "-m", "src.cnn_baseline",      "--config", args.config], "CNN baseline")
    run([py, "-m", "src.compare_baselines", "--config", args.config], "Compare baselines")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("  Results in results/baseline_comparison.csv + .png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
