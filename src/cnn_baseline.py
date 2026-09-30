"""
CNN depth-regression baseline — encoder-only architecture, ready for dual-head
extension (depth regression head + aleatoric uncertainty head).

Architecture overview
---------------------
Input  : (B, C, patch_size, patch_size)   — raw spectral bands + Stumpf ratio
Encoder: three conv blocks (enc1 → enc2 → bottleneck) with MaxPool downsampling
Pool   : AdaptiveAvgPool2d(1) + flatten   → (B, FEATURE_DIM)  ← encoder output
Head*  : _TempDepthHead — placeholder Linear(FEATURE_DIM → 1) so training runs
         today.  Will be replaced by DepthHead + UncertaintyHead.

Label  : center-pixel GEBCO depth (scalar, metres, positive = below sea level).
         Both planned heads predict one scalar per patch (mean depth, variance),
         so the label is one scalar per patch too.

Usage:
    python -m src.cnn_baseline --config config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("cnn_baseline")

# Dimensionality of the flat feature vector the encoder produces.
# Must equal the output channels of the bottleneck block.
FEATURE_DIM = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Loads pre-extracted patches from patches.npz with global channel z-score standardization.

    Using dataset-wide global mean/std per channel preserves physical unit scaling
    (especially for channel 5 Beer-Lambert z_prior in meters) across all patches.

    Normalization stats (mean/std for both inputs and targets) must be computed
    from the TRAINING split only, and passed to val/test via the constructor.

    Returns three items per sample:
        x_norm   : (C, H, W) z-score normalized input channels
        y_norm   : scalar, z-score normalized center-pixel depth
        y_meters : scalar, original center-pixel depth in meters (for physics loss)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, indices: np.ndarray,
                 mean: np.ndarray = None, std: np.ndarray = None,
                 y_mean: float = None, y_std: float = None):
        self.X = X[indices]   # (N, C, H, W)
        self.Y = Y[indices]   # (N, H, W)

        # --- Input channel normalization stats ---
        if mean is None:
            # Compute channel-wise mean and std across spatial dimensions
            self.mean = self.X.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
            self.std = (self.X.std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)
        else:
            self.mean = mean
            self.std = std

        # --- Target depth normalization stats ---
        cy, cx = self.Y.shape[1] // 2, self.Y.shape[2] // 2
        center_depths = self.Y[:, cy, cx]
        if y_mean is None:
            self.y_mean = float(center_depths.mean())
            self.y_std  = float(center_depths.std()) + 1e-6
        else:
            self.y_mean = y_mean
            self.y_std  = y_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (C, H, W)

        # Global channel z-score standardization
        x = (x - self.mean.squeeze()[:, None, None]) / self.std.squeeze()[:, None, None]

        # Scalar depth label: center pixel of the depth patch
        cy, cx = x.shape[1] // 2, x.shape[2] // 2
        y_meters = float(self.Y[idx, cy, cx])

        # Normalized depth for loss computation
        y_norm = (y_meters - self.y_mean) / self.y_std

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(y_meters, dtype=torch.float32),
        )



# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Spatial feature encoder.

    Compresses an input patch (B, C, H, W) into a flat feature vector
    (B, FEATURE_DIM) using three convolutional blocks and global average
    pooling.  The decoder is intentionally removed — two prediction heads
    (DepthHead, UncertaintyHead) will attach to this output.

    Flow:
        enc1       : C  → 32    full resolution
        pool1      : ÷2 spatially
        enc2       : 32 → 64
        pool2      : ÷2 spatially
        bottleneck : 64 → FEATURE_DIM (128)
        gap        : AdaptiveAvgPool2d(1)  — works for ANY patch size
        flatten    → (B, FEATURE_DIM)

    Args:
        in_channels: number of input channels (spectral bands + physics channels)
    """

    def __init__(self, in_channels: int):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1       = block(in_channels, 32)
        self.pool1      = nn.MaxPool2d(2)
        self.enc2       = block(32, 64)
        self.pool2      = nn.MaxPool2d(2)
        self.bottleneck = block(64, FEATURE_DIM)

        # Resolution-agnostic pooling: (B, FEATURE_DIM, H', W') → (B, FEATURE_DIM)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features: (B, FEATURE_DIM)  — NOT a depth prediction; feed to heads
        """
        x = self.enc1(x)
        x = self.enc2(self.pool1(x))
        x = self.bottleneck(self.pool2(x))
        x = self.gap(x)
        return self.flatten(x)   # (B, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Placeholder head  (TEMPORARY — replace with DepthHead + UncertaintyHead)
# ---------------------------------------------------------------------------

class _TempDepthHead(nn.Module):
    """**PLACEHOLDER — delete when DepthHead + UncertaintyHead are added.**

    Single Linear(FEATURE_DIM → 1) so training is runnable immediately.

    Swap-out checklist when adding the real heads:
      1. Delete this class.
      2. Add DepthHead(FEATURE_DIM) → scalar mean depth.
      3. Add UncertaintyHead(FEATURE_DIM) → scalar log-variance (σ²).
      4. Switch loss from scalar_mse → Gaussian NLL.
      5. Update train() and evaluate() to handle two head outputs.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FEATURE_DIM, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, FEATURE_DIM)
        Returns:
            depth_pred: (B,)
        """
        return self.fc(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------

def scalar_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE on per-patch scalar predictions vs. center-pixel GEBCO depth."""
    return torch.mean((pred - target) ** 2)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression + bathymetry-specific accuracy metrics (scalar labels)."""
    p, t = y_pred.flatten(), y_true.flatten()
    if len(p) == 0:
        nan = float("nan")
        return {"rmse": nan, "mae": nan, "r2": nan,
                "acc_within_1m_%": nan, "acc_within_2m_%": nan,
                "delta_1.25_%": nan}

    abs_diff = np.abs(p - t)
    rmse     = float(np.sqrt(np.mean((p - t) ** 2)))
    mae      = float(np.mean(abs_diff))
    ss_res   = np.sum((t - p) ** 2)
    ss_tot   = np.sum((t - t.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Depth accuracy against GEBCO ground truth
    acc_1m  = float(np.mean(abs_diff <= 1.0) * 100.0)  # within ±1 m
    acc_2m  = float(np.mean(abs_diff <= 2.0) * 100.0)  # within ±2 m
    pos     = (t > 0.1) & (p > 0.1)
    delta_1 = (
        float(np.mean(np.maximum(t[pos] / p[pos], p[pos] / t[pos]) < 1.25) * 100.0)
        if np.any(pos) else float("nan")
    )

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "acc_within_1m_%": acc_1m, "acc_within_2m_%": acc_2m,
        "delta_1.25_%": delta_1,
    }


# ---------------------------------------------------------------------------
# Training & evaluation with HybridBathNet (Physics Decoder + Dual Heads + NLL Loss)
# ---------------------------------------------------------------------------

from .hybridbathnet import HybridBathNetModel, CompositePhysicsNLLLoss, predict_with_uncertainty


def train_hybridbathnet(model, criterion, train_loader, val_loader, epochs, lr, device, y_mean, y_std):
    """Train HybridBathNet model end-to-end with CompositePhysicsNLLLoss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_rmse = float("inf")
    best_state = None

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training HybridBathNet", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_nll = 0.0
        train_nonneg = 0.0
        train_ext = 0.0

        for x, y_norm, y_m in train_loader:
            x, y_norm, y_m = x.to(device), y_norm.to(device), y_m.to(device)
            optimizer.zero_grad()

            pred_norm, log_var, kd, z_ext = model(x)
            loss, loss_dict = criterion(pred_norm, log_var, y_norm, z_ext, y_mean=y_mean, y_std=y_std)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_nll += loss_dict["loss_nll"] * x.size(0)
            train_nonneg += loss_dict["loss_nonneg"] * x.size(0)
            train_ext += loss_dict["loss_ext"] * x.size(0)

        n_samples = len(train_loader.dataset)
        train_loss /= n_samples
        train_nll /= n_samples
        train_nonneg /= n_samples
        train_ext /= n_samples

        val_metrics = evaluate_hybridbathnet(model, val_loader, device, y_mean, y_std)

        epoch_bar.set_postfix(
            loss=f"{train_loss:.3f}",
            val_rmse=f"{val_metrics['rmse']:.2f}m",
            val_r2=f"{val_metrics['r2']:.3f}",
            val_acc_1m=f"{val_metrics['acc_within_1m_%']:.1f}%",
        )
        LOG.info(
            "Epoch %3d/%d | Total Loss: %.4f (NLL: %.4f, NonNeg: %.4f, Ext: %.4f) | "
            "Val → RMSE: %.3fm  MAE: %.3fm  R²: %.3f  Acc(±1m): %.1f%%  Acc(±2m): %.1f%%  δ<1.25: %.1f%%",
            epoch, epochs, train_loss, train_nll, train_nonneg, train_ext,
            val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"],
            val_metrics["acc_within_1m_%"], val_metrics["acc_within_2m_%"],
            val_metrics["delta_1.25_%"],
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def evaluate_hybridbathnet(model, loader, device, y_mean, y_std) -> dict:
    model.eval()
    all_pred, all_true, all_aleatoric = [], [], []

    for x, _, y_m in loader:
        x = x.to(device)
        pred_norm, log_var, _, _ = model(x)
        pred_m = pred_norm.cpu().numpy() * y_std + y_mean
        all_pred.append(pred_m)
        all_true.append(y_m.numpy())
        all_aleatoric.append(torch.exp(log_var).cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )
    metrics["mean_aleatoric_std"] = float(np.sqrt(np.mean(np.concatenate(all_aleatoric))))
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    feat_dir = cfg["features"]["features_dir"]
    data = np.load(Path(feat_dir) / "patches.npz")
    X, Y  = data["X"], data["Y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    LOG.info(
        "Patches — X: %s  Y: %s  |  channels: %d (spectral + physics)",
        X.shape, Y.shape, X.shape[1],
    )

    train_cfg = cfg["training"]
    hnet_cfg  = cfg.get("hybridbathnet", {})
    device    = train_cfg["device"] if torch.cuda.is_available() else "cpu"
    LOG.info("Using device: %s", device)

    train_ds = PatchDataset(X, Y, train_idx)
    val_ds   = PatchDataset(X, Y, val_idx  if len(val_idx)  else train_idx, mean=train_ds.mean, std=train_ds.std, y_mean=train_ds.y_mean, y_std=train_ds.y_std)
    test_ds  = PatchDataset(X, Y, test_idx if len(test_idx) else train_idx, mean=train_ds.mean, std=train_ds.std, y_mean=train_ds.y_mean, y_std=train_ds.y_std)

    if len(test_idx) == 0:
        LOG.warning("test_idx is empty — falling back to train set for test evaluation.")
    if len(val_idx) == 0:
        LOG.warning("val_idx is empty — falling back to train set for validation.")

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg["batch_size"])

    # Instantiate FRESH HybridBathNet (Physics Encoder + Physics Decoder + Dual Heads)
    # Re-initialized from random weights (no checkpoint loading to prevent overfitting)
    loss_weights = hnet_cfg.get("loss_weights", {"w_mse": 1.0, "w_nll": 1.0, "w_phys": 0.5, "w_ext": 0.5})
    model = HybridBathNetModel(
        in_channels=X.shape[1],
        feature_dim=hnet_cfg.get("feature_dim", FEATURE_DIM),
        dropout_rate=hnet_cfg.get("dropout_rate", 0.1),
        i0_over_epsilon=hnet_cfg.get("i0_over_epsilon", 100.0),
    ).to(device)

    criterion = CompositePhysicsNLLLoss(
        w_mse=loss_weights.get("w_mse", 1.0),
        w_nll=loss_weights.get("w_nll", 1.0),
        w_phys=loss_weights.get("w_phys", 0.5),
        w_ext=loss_weights.get("w_ext", 0.5),
    )

    LOG.info("FRESH HybridBathNetModel initialized from random weights (no checkpoint loaded).")
    LOG.info("Normalization stats (from TRAINING split only):")
    LOG.info("  Input channels  -> mean: %s  std: %s", train_ds.mean.squeeze().tolist(), train_ds.std.squeeze().tolist())
    LOG.info("  Target depth    -> y_mean: %.3f m  y_std: %.3f m", train_ds.y_mean, train_ds.y_std)
    LOG.info("Loss Weights -> Depth MSE: %.2f | Gaussian NLL: %.2f | Non-Negativity: %.2f | Extinction Bound: %.2f",
             loss_weights["w_mse"], loss_weights["w_nll"], loss_weights["w_phys"], loss_weights["w_ext"])


    model = train_hybridbathnet(
        model, criterion,
        train_loader, val_loader,
        train_cfg["epochs"], train_cfg["learning_rate"], device,
        y_mean=train_ds.y_mean, y_std=train_ds.y_std,
    )

    ensure_dir(train_cfg["checkpoint_dir"])
    torch.save(
        {"model": model.state_dict(), "y_mean": train_ds.y_mean, "y_std": train_ds.y_std},
        Path(train_cfg["checkpoint_dir"]) / "hybridbathnet.pt",
    )

    test_metrics = evaluate_hybridbathnet(model, test_loader, device, y_mean=train_ds.y_mean, y_std=train_ds.y_std)
    LOG.info(
        "HybridBathNet (test) → RMSE %.3f  MAE %.3f  R² %.3f  (Aleatoric std: %.3fm)",
        test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"], test_metrics["mean_aleatoric_std"],
    )


    ensure_dir(cfg["evaluation"]["results_dir"])
    with open(Path(cfg["evaluation"]["results_dir"]) / "hybridbathnet_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()

