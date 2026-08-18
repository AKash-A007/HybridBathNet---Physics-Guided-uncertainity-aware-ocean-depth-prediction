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
    """Loads pre-extracted patches from patches.npz.

    Each sample returns:
        x : (C, patch_size, patch_size) float32 — normalised spectral + physics
        y : scalar float32 — GEBCO depth at the CENTER pixel of the patch (m)

    Center-pixel label is used because the two planned heads (depth + uncertainty)
    each produce one scalar per patch, not a full depth map.
    The valid-mask (M) is dropped here — center pixels are guaranteed valid by
    construction (≥50% water coverage enforced during patch extraction).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, indices: np.ndarray):
        self.X = X[indices]   # (N, C, H, W)
        self.Y = Y[indices]   # (N, H, W) — full depth patch; center pixel used

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (C, H, W)

        # Per-channel min-max normalisation — robust whether reflectance is
        # already 0-1 (L2A) or raw DN (L1C scaled integers).
        for c in range(x.shape[0]):
            mx = x[c].max()
            if mx > 0:
                x[c] = x[c] / mx

        # Scalar depth label: center pixel of the depth patch.
        cy, cx = x.shape[1] // 2, x.shape[2] // 2
        y_center = float(self.Y[idx, cy, cx])

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y_center, dtype=torch.float32),
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
# Training & evaluation
# ---------------------------------------------------------------------------

def train(encoder, head, train_loader, val_loader, epochs, lr, device):
    """Train encoder + placeholder head end-to-end with scalar MSE loss.

    Encoder and head are passed explicitly so either can be swapped
    independently when the real dual-head is wired in.
    """
    params    = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    best_val_rmse = float("inf")
    best_enc_state, best_head_state = None, None

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training CNN encoder", unit="epoch")
    for epoch in epoch_bar:
        encoder.train()
        head.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            features = encoder(x)       # (B, FEATURE_DIM)
            pred     = head(features)   # (B,)
            loss     = scalar_mse(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        val_metrics = evaluate(encoder, head, val_loader, device)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.3f}",
            val_rmse=f"{val_metrics['rmse']:.2f}m",
            val_acc_1m=f"{val_metrics['acc_within_1m_%']:.1f}%",
            val_acc_2m=f"{val_metrics['acc_within_2m_%']:.1f}%",
        )
        LOG.info(
            "Epoch %3d/%d | loss: %.4f | Val → RMSE: %.3fm  MAE: %.3fm  R²: %.3f"
            "  Acc(±1m): %.1f%%  Acc(±2m): %.1f%%  δ<1.25: %.1f%%",
            epoch, epochs, train_loss,
            val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"],
            val_metrics["acc_within_1m_%"], val_metrics["acc_within_2m_%"],
            val_metrics["delta_1.25_%"],
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse   = val_metrics["rmse"]
            best_enc_state  = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_enc_state is not None:
        encoder.load_state_dict(best_enc_state)
        head.load_state_dict(best_head_state)

    return encoder, head


@torch.no_grad()
def evaluate(encoder, head, loader, device) -> dict:
    encoder.eval()
    head.eval()
    all_pred, all_true = [], []

    for x, y in loader:
        x        = x.to(device)
        features = encoder(x)
        pred     = head(features).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y.numpy())

    return compute_metrics(
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )


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
    # M (valid-mask) is no longer used — center-pixel labels are always valid.
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    LOG.info(
        "Patches — X: %s  Y: %s  |  channels: %d (spectral + stumpf_ratio)",
        X.shape, Y.shape, X.shape[1],
    )

    train_cfg = cfg["training"]
    device    = train_cfg["device"] if torch.cuda.is_available() else "cpu"
    LOG.info("Using device: %s", device)

    train_ds = PatchDataset(X, Y, train_idx)
    val_ds   = PatchDataset(X, Y, val_idx  if len(val_idx)  else train_idx)
    test_ds  = PatchDataset(X, Y, test_idx if len(test_idx) else train_idx)

    if len(test_idx) == 0:
        LOG.warning("test_idx is empty — falling back to train set for test evaluation.")
    if len(val_idx) == 0:
        LOG.warning("val_idx is empty — falling back to train set for validation.")

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg["batch_size"])

    # Encoder + placeholder head.
    # Replace _TempDepthHead with real DepthHead + UncertaintyHead when ready.
    encoder = CNNEncoder(in_channels=X.shape[1]).to(device)
    head    = _TempDepthHead().to(device)

    LOG.info(
        "CNNEncoder — in_channels: %d  →  feature_dim: %d",
        X.shape[1], FEATURE_DIM,
    )
    LOG.info("Head       — _TempDepthHead (placeholder, Linear %d→1)", FEATURE_DIM)

    encoder, head = train(
        encoder, head,
        train_loader, val_loader,
        train_cfg["epochs"], train_cfg["learning_rate"], device,
    )

    ensure_dir(train_cfg["checkpoint_dir"])
    torch.save(
        {"encoder": encoder.state_dict(), "head": head.state_dict()},
        Path(train_cfg["checkpoint_dir"]) / "cnn_baseline.pt",
    )

    test_metrics = evaluate(encoder, head, test_loader, device)
    LOG.info(
        "CNN baseline (test) → RMSE %.3f  MAE %.3f  R² %.3f",
        test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"],
    )

    ensure_dir(cfg["evaluation"]["results_dir"])
    with open(Path(cfg["evaluation"]["results_dir"]) / "cnn_baseline_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
