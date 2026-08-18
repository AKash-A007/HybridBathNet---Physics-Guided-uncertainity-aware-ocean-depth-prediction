# HybridBathNet — Physics-Guided & Uncertainty-Aware Ocean Depth Prediction

This repository contains the baseline pipeline and experimental models for **HybridBathNet**: an end-to-end framework for ocean bathymetry (depth prediction) from Sentinel-2 satellite imagery.

It includes three baseline models to compare performance against:
1. **Stumpf (2003)** empirical log-band ratio regression.
2. **Random Forest** on spectral bands, band ratios, and NDWI.
3. **Simple CNN** (encoder-decoder deep learning model without physics constraints).

---

## ⚡ Quick Start: Commands to Run

### 1. Environment Setup
```bash
# Clone the repository (with LFS for model checkpoints)
git clone https://github.com/mujiishere/HybridBathNet---Physics-Guided-uncertainity-aware-ocean-depth-prediction.git
cd HybridBathNet---Physics-Guided-uncertainity-aware-ocean-depth-prediction
git lfs pull

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Entire Pipeline (One-Click)
```bash
# Run full pipeline (Data Download -> Preprocessing -> Feature Engineering -> Training -> Evaluation)
python run_baseline_pipeline.py --config config.yaml

# Or if datasets are already prepared, skip download:
python run_baseline_pipeline.py --config config.yaml --skip-download
```

### 3. Run Pipeline with Live Output Monitor
```bash
python monitor_and_run.py
```

---

## 🛠️ Step-by-Step Pipeline Commands

If you prefer to execute each stage individually:

### Step 1: Download Datasets
Set your Copernicus credentials (free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu)):
```bash
# Windows (PowerShell):
$env:CDSE_USERNAME="your-email@example.com"
$env:CDSE_PASSWORD="your-password"

# Linux/macOS:
export CDSE_USERNAME="your-email@example.com"
export CDSE_PASSWORD="your-password"
```

Download Sentinel-2 L2A scenes:
```bash
python -m src.download_sentinel2 --config config.yaml
```
*(Optionally download GEBCO bathymetry NetCDF to `data/raw/gebco/gebco_subset.nc` and run:)*
```bash
python -m src.download_gebco --config config.yaml
python -m src.download_coastline --config config.yaml
```

### Step 2: Preprocess & Coregister Data
```bash
python -m src.preprocess --config config.yaml
```

### Step 3: Feature Engineering & Spatial Splitting
```bash
python -m src.features --config config.yaml
```

### Step 4: Train & Evaluate Baseline Models
```bash
# Train Stumpf & Random Forest models
python -m src.baseline_models --config config.yaml

# Train Simple CNN baseline
python -m src.cnn_baseline --config config.yaml

# Compare all model results (generates comparison table & charts)
python -m src.compare_baselines --config config.yaml
```

---

## 📁 Repository Structure

```
hybridbathnet_baseline/
├── config.yaml                     # Main configuration (AOI, hyperparameters, paths)
├── requirements.txt                # Python dependencies
├── run_baseline_pipeline.py        # End-to-end execution script
├── monitor_and_run.py              # Automated runner with live monitoring
├── README.md                       # Documentation
├── src/                            # Source code modules
│   ├── utils.py                    # Geo utilities & logging
│   ├── download_sentinel2.py       # CDSE satellite imagery fetcher
│   ├── download_gebco.py           # GEBCO bathymetry processor
│   ├── download_coastline.py       # OpenStreetMap coastline downloader
│   ├── preprocess.py               # Cloud masking, compositing, resampling
│   ├── features.py                 # Band ratio, NDWI, spatial train/test splits
│   ├── baseline_models.py          # Stumpf & Random Forest models
│   ├── cnn_baseline.py             # Simple PyTorch CNN architecture & training
│   └── compare_baselines.py        # Benchmark results comparison & plots
├── checkpoints/                    # Pre-trained model weights (Git LFS)
│   ├── cnn_baseline.pt
│   ├── rf_model.joblib
│   └── stumpf_model.joblib
├── results/                        # Evaluation metrics & plots
│   ├── cnn_baseline_results.json
│   ├── tabular_baseline_results.json
│   ├── baseline_comparison.csv
│   └── baseline_comparison.png
└── outputs/                        # Output rasters & GeoTIFFs
```

---

## 📊 Pre-trained Models & Output Results

Trained checkpoint models are included in `checkpoints/` (tracked via **Git LFS**):
- **Stumpf Model**: `checkpoints/stumpf_model.joblib`
- **Random Forest Model**: `checkpoints/rf_model.joblib`
- **Simple CNN Model**: `checkpoints/cnn_baseline.pt`

Benchmark results are output to `results/` in JSON and CSV formats.

---

## ⚠️ Notes & Troubleshooting

- **Large File Storage (Git LFS)**: Pre-trained models (e.g. `rf_model.joblib`) are tracked with `git-lfs`. Ensure `git lfs pull` is run after cloning.
- **Copernicus Authentication**: Ensure `CDSE_USERNAME` and `CDSE_PASSWORD` environment variables are set before downloading Sentinel-2 data.
- **GPU Acceleration**: PyTorch will automatically use NVIDIA CUDA GPU if available, otherwise defaulting to CPU.
