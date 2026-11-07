"""Train joint HybridBathNet from scratch on the distinct ATL24 AOI benchmark."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import datetime
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cnn_baseline import PatchDataset, SpectralAugDataset, train_hybridbathnet, evaluate_hybridbathnet
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import JointCompositeLoss, CoverageSampler, initialize_optical_head
from scripts.run_nearshore_hybrid import diagnostics

ROOT = Path("runs/2026-11-06/atl24_hybrid_v1")


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def status(stage, extra=None):
    write(ROOT / "status.json", {
        "stage": stage, "experiment": "ATL24-supervised HybridBathNet only",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **(extra or {})})


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing slow CPU training")
    if (ROOT / "results.json").exists():
        print("Already complete; refusing duplicate training")
        return
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = ROOT / "features/patches.npz"
    with dataset.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    with np.load(dataset) as data:
        x, y, m, coords = (data[key] for key in ("X", "Y", "M", "coords"))
        indices = {key: data[key] for key in ("train_idx", "val_idx", "test_idx")}
    assert len(np.unique(coords, axis=0)) == len(coords)
    assert np.all(y[:, 32, 32] > 0) and np.all(y[:, 32, 32] <= 20)
    train_base = PatchDataset(x, y, m, indices["train_idx"])

    def loader(name, training=False):
        data = train_base if training else PatchDataset(
            x, y, m, indices[name], mean=train_base.mean, std=train_base.std,
            y_mean=train_base.y_mean, y_std=train_base.y_std)
        wrapped = SpectralAugDataset(data, augment=training, noise_std=.02,
                                     use_rotations=False)
        return DataLoader(wrapped, batch_size=8,
                          sampler=CoverageSampler(len(data), 42) if training else None)

    train, val, test = loader("train_idx", True), loader("val_idx"), loader("test_idx")
    rinf = np.array([np.percentile(train_base.X[:, band][train_base.M > .5], 2)
                     for band in range(3)], dtype=np.float32)
    model = HybridBathNetModel(
        in_channels=16, mean=train_base.mean.squeeze(), std=train_base.std.squeeze(),
        r_inf=rinf, mask_aware_physics=True, kd_aggregation="center",
        physics_conditioning="spatial").cuda()
    initialize_optical_head(model)
    criterion = JointCompositeLoss(w_mse=1, w_nll=1, w_ext=.03, w_phys=0,
                                   w_recon=.1, use_depth_weighting=False,
                                   beta_nll=.5, huber_delta=.5)
    protocol = {
        "source": "SlideRule ATL24 archive AOI subset, 2018–Nov 2024, current V003 catalogue",
        "reference": "ATL24 median bathymetry depth per 10m pixel; no GEBCO target",
        "imagery": "Jan–Mar 2024 Sentinel-2 L2A composite at same AOI",
        "model": "HybridBathNet 16 input channels; joint physics/depth/uncertainty/reconstruction",
        "seed": 42, "epochs_max": 180, "patience": 30, "batch": 8, "lr": .001,
        "cosine_period": 180, "ema": .999, "nll_warmup": 20,
        "weights": {"depth_huber": 1, "beta_nll": 1, "extinction": .03, "reconstruction": .1},
        "dataset_sha256": fingerprint,
        "split_sizes": {key: len(value) for key, value in indices.items()},
        "selection": "validation RMSE only; fixed historical geographic test blocks",
        "optical_limit": "uncalibrated diagnostic flag only, never used to exclude",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "limitations": ["ATL24 V003 public service returned invalid-dataframe alerts for some granules.",
                        "Repeated points along four RGT tracks and 2018–2024-to-2024 temporal mismatch.",
                        "Different reference/locations from GEBCO v1; RMSEs are not matched model gains."]}
    write(ROOT / "protocol.json", protocol)
    status("training", {"split_sizes": protocol["split_sizes"]})
    model, _ = train_hybridbathnet(
        model, criterion, train, val, epochs=180, lr=.001, device="cuda",
        y_mean=train_base.y_mean, y_std=train_base.y_std, nll_warmup_epochs=20,
        lr_restart_period=180, ema_decay=.999, early_stop_patience=30,
        checkpoint_dir=str(ROOT / "hybridbathnet/checkpoints"))
    torch.save({"model": model.state_dict(), "protocol": protocol,
                "ch_mean": train_base.mean, "ch_std": train_base.std,
                "y_mean": train_base.y_mean, "y_std": train_base.y_std}, ROOT / "hybridbathnet.pt")
    status("evaluating")
    validation = evaluate_hybridbathnet(
        model, val, "cuda", train_base.y_mean, train_base.y_std,
        compute_calibration=True, predictions_path=ROOT / "validation_predictions.npz")
    testing = evaluate_hybridbathnet(
        model, test, "cuda", train_base.y_mean, train_base.y_std,
        compute_calibration=True, predictions_path=ROOT / "test_predictions.npz")
    write(ROOT / "validation_metrics.json", validation)
    write(ROOT / "test_metrics.json", testing)
    val_d = diagnostics(ROOT / "validation_predictions.npz", coords[indices["val_idx"]])
    test_d = diagnostics(ROOT / "test_predictions.npz", coords[indices["test_idx"]])
    write(ROOT / "results.json", {"validation": val_d, "test": test_d,
                                  "validation_uncertainty": validation,
                                  "test_uncertainty": testing})
    lines = ["# ATL24-supervised HybridBathNet results", "",
             "A distinct, from-scratch full joint HybridBathNet trained on ATL24 depth labels.",
             "The Jan–Mar 2024 Sentinel-2 composite is the input; no GEBCO depths are targets or features.",
             "Original geographic test blocks are kept. Check [protocol](protocol.json) and [data audit](features/metadata.json).", "",
             f"Validation 0–20 m RMSE: **{val_d['primary_0_20']['rmse']:.3f} m** "
             f"(n={val_d['primary_0_20']['n']}).",
             f"Test 0–20 m RMSE: **{test_d['primary_0_20']['rmse']:.3f} m** "
             f"(n={test_d['primary_0_20']['n']}).", "",
             "| Test ATL24 depth bin | n | RMSE (m) | 95% approx. 2 km group bootstrap CI |",
             "|---|---:|---:|---|"]
    for label, score in test_d["primary_bins"].items():
        if score["n"]:
            ci = score["rmse_ci95"]
            display = "15–20" if label == "15-+" else label.replace("-", "–")
            lines.append(f"| {display} | {score['n']} | {score['rmse']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} |")
    lines += ["", "**Interpretation:** ATL24 points along repeat tracks are correlated and some public-service granules may be absent.",
              "Optical conditions/seabed may have changed between 2018–2024 reference dates and 2024 imagery.",
              "The existing GEBCO test RMSE 2.561 m is a different benchmark, not a matched comparator.",
              "The optical-limit heuristic is uncalibrated; no pixels were masked to improve reported RMSE.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    status("complete", {"validation_rmse_m": val_d["primary_0_20"]["rmse"],
                        "test_rmse_m": test_d["primary_0_20"]["rmse"]})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        ROOT.mkdir(parents=True, exist_ok=True)
        write(ROOT / "failure.json", {"error": repr(error)})
        status("failed")
        raise
