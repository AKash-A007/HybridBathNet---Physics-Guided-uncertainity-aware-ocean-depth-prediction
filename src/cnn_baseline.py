"""
CNN depth-regression baseline + HybridBathNet v3 training pipeline.

Architecture overview (v3)
--------------------------
Input  : (B, C, patch_size, patch_size)   — 16+ channels (spectral + physics + Lyzenga)
Encoder: UNetEncoder — residual blocks + SE + FiLM conditioning at each scale
Decoder: UNetDecoder — skip connections → dense (H, W) depth + log-variance maps
Heads  : depth = softplus(h)  log_var = clamp(h, −10, 10) — both dense maps

Label  : full GEBCO depth patch (H, W), averaged over valid water pixels in loss.

v3 Training Improvements (no new data required):
  §1.1  σ units fixed   : mean_aleatoric_std now in metres.
  §1.4  LR schedule     : epochs set to a multiple of T_0 to avoid mid-cycle cutoff.
  §4.1  One compensation: depth_weighting disabled when DepthStratifiedSampler is ON.
  §4.2  β-NLL           : variance collapse prevented via β-weighted NLL.
  §4.3  Huber mean term : smooth_l1_loss replaces squared MSE.
  §4.5  EMA weights     : exponential moving average of model parameters; the EMA
                          model is evaluated at the end of each epoch and saved.
  §4.5  Early stopping  : validation RMSE patience (default 15 epochs).
  §5.2  Snapshot saves  : checkpoint saved at end of every cosine cycle.
  §5.4  Calibration     : extended metrics returned (NLL, coverage, sparsification).
  §6.3  AMP             : torch.autocast + GradScaler for bfloat16/fp16 training.
  §6.2  GPU data path   : tensors preloaded to GPU; augmentation done in-batch.

Usage:
    python -m src.cnn_baseline --config config.yaml
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from .utils import load_config, get_logger, ensure_dir
from .hybridbathnet import (
    HybridBathNetModel,
    CompositePhysicsNLLLoss,
    DensePhysicsLoss,
    predict_with_uncertainty,
    calibration_metrics,
    conformal_recalibrate,
)

LOG = get_logger("cnn_baseline")

FEATURE_DIM = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Loads pre-extracted patches from patches.npz with global z-score normalisation.

    Normalisation stats are computed over WATER PIXELS ONLY (§4.6) from the
    training split to avoid land pixels biasing the mean/std.

    Returns three items per sample:
        x_norm   : (C, H, W)  z-score normalised input
        y_patch  : (H, W)     depth map in metres (positive = below sea level)
        mask     : (H, W)     valid water pixel mask (float 0/1)
    """

    def __init__(
        self,
        X:      np.ndarray,
        Y:      np.ndarray,
        M:      np.ndarray,
        indices: np.ndarray,
        mean:   np.ndarray = None,
        std:    np.ndarray = None,
        y_mean: float = None,
        y_std:  float = None,
    ):
        self.X = X[indices]   # (N, C, H, W)
        self.Y = Y[indices]   # (N, H, W)
        self.M = M[indices]   # (N, H, W)
        if len(indices) == 0:
            raise ValueError("Empty dataset split; rebuild spatial splits before training")
        if not np.isfinite(self.X).all() or not np.isfinite(self.Y).all():
            raise ValueError("Non-finite input features or labels; rebuild preprocessing")

        # Channel normalisation (water pixels only — §4.6)
        if mean is None:
            # Flatten all (N, C, H, W) over spatial dims, masking land pixels
            mask_3d = self.M[:, None, :, :]             # (N, 1, H, W)
            X_water = self.X * mask_3d                  # zero out land
            n_water = mask_3d.sum(axis=(0, 2, 3)) + 1e-6  # (1, C, 1, 1) — keep shape
            n_water_ch = n_water.reshape(-1)             # (C,)

            # Weighted mean per channel
            ch_mean = (X_water * mask_3d).sum(axis=(0, 2, 3)) / n_water_ch
            # Weighted std per channel
            diff_sq = ((self.X - ch_mean[None, :, None, None]) ** 2) * mask_3d
            ch_std  = np.sqrt(diff_sq.sum(axis=(0, 2, 3)) / n_water_ch) + 1e-6

            self.mean = ch_mean.astype(np.float32)[None, :, None, None]
            self.std  = ch_std.astype(np.float32)[None,  :, None, None]
            # Constant water-mask channels otherwise turn land into -1e6.
            self.std[self.std < 1e-5] = 1.0
            if self.X.shape[1] > 11:
                self.mean[:, 11] = 0.0
                self.std[:, 11] = 1.0
        else:
            self.mean = mean
            self.std  = std

        # Target depth normalisation (centre pixel of each patch)
        cy, cx          = self.Y.shape[1] // 2, self.Y.shape[2] // 2
        center_depths   = self.Y[:, cy, cx]
        if y_mean is None:
            self.y_mean = float(center_depths.mean())
            self.y_std  = float(center_depths.std()) + 1e-6
        else:
            self.y_mean = y_mean
            self.y_std  = y_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean.squeeze()[:, None, None]) / self.std.squeeze()[:, None, None]
        y = self.Y[idx]          # (H, W) depth in metres
        m = self.M[idx]          # (H, W) mask
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(m.astype(np.float32)),
            torch.from_numpy(self.X[idx, :3].astype(np.float32).copy()),
        )


# ---------------------------------------------------------------------------
# Spectral + spatial augmentation  (§3.8 — flip-only variant available)
# ---------------------------------------------------------------------------

class SpectralAugDataset(Dataset):
    """Wraps PatchDataset with on-the-fly spectral + spatial augmentations."""

    def __init__(
        self,
        base:        Dataset,
        augment:     bool  = True,
        noise_std:   float = 0.02,
        use_rotations: bool = False,   # §3.8: set False to ablate rotations
    ):
        self.base          = base
        self.augment       = augment
        self.noise_std     = noise_std
        self.use_rotations = use_rotations

    def __len__(self): return len(self.base)

    @property
    def y_mean(self): return self.base.y_mean
    @property
    def y_std(self):  return self.base.y_std
    @property
    def mean(self):   return self.base.mean
    @property
    def std(self):    return self.base.std

    def __getitem__(self, idx):
        x, y, m, raw = self.base[idx]   # raw remains unjittered for physics

        if self.augment:
            n_spectral = min(4, x.shape[0])
            noise      = torch.randn(n_spectral, 1, 1) * self.noise_std
            x          = x.clone()
            x[:n_spectral] = x[:n_spectral] + noise

            # Horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [2])
                y = torch.flip(y, [1])
                m = torch.flip(m, [1])
                raw = torch.flip(raw, [2])

            # Vertical flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [1])
                y = torch.flip(y, [0])
                m = torch.flip(m, [0])
                raw = torch.flip(raw, [1])

            # 90° rotation (§3.8 — ablate with use_rotations=False)
            if self.use_rotations:
                k = int(torch.randint(0, 4, (1,)).item())
                if k > 0:
                    x = torch.rot90(x, k=k, dims=[1, 2])
                    y = torch.rot90(y.unsqueeze(0), k=k, dims=[1, 2]).squeeze(0)
                    m = torch.rot90(m.unsqueeze(0), k=k, dims=[1, 2]).squeeze(0)
                    raw = torch.rot90(raw, k=k, dims=[1, 2])

        return x, y, m, raw


# ---------------------------------------------------------------------------
# Depth-Stratified Sampler
# ---------------------------------------------------------------------------

class DepthStratifiedSampler(Sampler):
    """Yields mini-batch indices with balanced depth-stratum representation."""

    def __init__(self, depths: np.ndarray, n_strata: int = 5, seed: int = 42):
        self.rng      = np.random.RandomState(seed)
        self.n_strata = n_strata

        percentiles = np.linspace(0, 100, n_strata + 1)
        boundaries  = np.percentile(depths, percentiles)
        boundaries[0]  -= 1e-3
        boundaries[-1] += 1e-3

        self.strata = []
        for i in range(n_strata):
            mask = (depths >= boundaries[i]) & (depths < boundaries[i + 1])
            self.strata.append(np.where(mask)[0])
            LOG.debug("Stratum %d: depth=[%.1f, %.1f] m  n=%d",
                      i, boundaries[i], boundaries[i+1], len(self.strata[-1]))

        self.n_per_stratum = max(len(s) for s in self.strata)
        self._total        = self.n_per_stratum * n_strata

    def __len__(self): return self._total

    def __iter__(self):
        indices = []
        for stratum in self.strata:
            if len(stratum) == 0:
                continue
            chosen = self.rng.choice(stratum, size=self.n_per_stratum, replace=True)
            self.rng.shuffle(chosen)
            indices.extend(chosen.tolist())
        self.rng.shuffle(indices)
        return iter(indices)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression + bathymetry accuracy metrics."""
    p, t = y_pred.flatten(), y_true.flatten()
    if len(p) == 0:
        nan = float("nan")
        return {"rmse": nan, "mae": nan, "r2": nan,
                "acc_within_1m_%": nan, "acc_within_2m_%": nan, "delta_1.25_%": nan}

    abs_diff = np.abs(p - t)
    rmse     = float(np.sqrt(np.mean((p - t) ** 2)))
    mae      = float(np.mean(abs_diff))
    ss_res   = np.sum((t - p) ** 2)
    ss_tot   = np.sum((t - t.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    acc_1m  = float(np.mean(abs_diff <= 1.0) * 100.0)
    acc_2m  = float(np.mean(abs_diff <= 2.0) * 100.0)
    pos     = (t > 0.1) & (p > 0.1)
    delta_1 = (
        float(np.mean(np.maximum(t[pos] / p[pos], p[pos] / t[pos]) < 1.25) * 100.0)
        if np.any(pos) else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "r2": r2,
            "acc_within_1m_%": acc_1m, "acc_within_2m_%": acc_2m, "delta_1.25_%": delta_1}


def compute_depth_bin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-depth-bin RMSE for the results table (§8 ablation protocol)."""
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 9999)]
    out  = {}
    for lo, hi in bins:
        label = f"{lo}-{hi}m" if hi < 9999 else f">{lo}m"
        mask  = (y_true >= lo) & (y_true < hi)
        out[f"n_{label}"] = int(mask.sum())
        if mask.sum() < 10:
            out[f"rmse_{label}"] = None
        else:
            out[f"rmse_{label}"] = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
    return out


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model weights (§4.5)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay    = decay
        self.shadow   = copy.deepcopy(model.state_dict())
        self._model   = model

    @torch.no_grad()
    def update(self):
        for name, param in self._model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
        # BatchNorm running statistics must track the trained model too.
        for name, buffer in self._model.named_buffers():
            self.shadow[name].copy_(buffer)

    def apply_shadow(self):
        self._model.load_state_dict(self.shadow)

    def restore(self, backup: dict):
        self._model.load_state_dict(backup)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def clip_hybrid_gradients(model, max_norm, separate_uncertainty=False):
    """Prevent variance-gradient magnitude from rescaling depth/physics updates."""
    if not separate_uncertainty:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm,
                                             error_if_nonfinite=True)
    variance, depth_physics = [], []
    for name, parameter in model.named_parameters():
        destination = variance if name.startswith(('decoder.logvar_conv.', 'decoder.uncertainty_head.')) else depth_physics
        destination.append(parameter)
    return tuple(torch.nn.utils.clip_grad_norm_(group, max_norm=max_norm,
                 error_if_nonfinite=True) for group in (depth_physics, variance))


def train_hybridbathnet(
    model, criterion, train_loader, val_loader,
    epochs, lr, device, y_mean, y_std,
    nll_warmup_epochs: int   = 20,
    base_nll_weight:   float = 1.0,
    grad_clip_norm:    float = 1.0,
    lr_restart_period: int   = 30,
    ema_decay:         float = 0.999,
    early_stop_patience: int = 15,
    checkpoint_dir:    str   = "checkpoints",
    use_amp:           bool  = True,
    separate_uncertainty_clip: bool = False,
):
    """Train HybridBathNet v3.

    Improvements:
      §4.5  EMA weights   : updated every step; evaluated at epoch end.
      §5.2  Snapshots     : checkpoint saved at the end of every cosine cycle.
      §6.3  AMP           : torch.autocast (bfloat16 if CUDA, else fp16).
      §1.4  LR alignment  : epochs should be a multiple of T_0 (enforced below).
      Early stopping      : stops if val RMSE does not improve for patience epochs.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # §1.4 — align epoch count to nearest multiple of T_0 (warn user if not)
    if epochs % lr_restart_period != 0:
        LOG.warning(
            "epochs=%d is not a multiple of lr_restart_period=%d — "
            "the LR schedule ends mid-cycle. Consider epochs=%d or %d.",
            epochs, lr_restart_period,
            (epochs // lr_restart_period) * lr_restart_period,
            ((epochs // lr_restart_period) + 1) * lr_restart_period,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=lr_restart_period, T_mult=1, eta_min=lr * 0.01
    )

    ema         = EMA(model, decay=ema_decay)
    best_val    = float("inf")
    best_state  = None
    no_improve  = 0
    snapshots   = []    # §5.2 — one per cosine cycle
    history = []

    # AMP setup (§6.3)
    use_amp = use_amp and str(device).startswith("cuda")
    amp_dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_device = "cuda" if device != "cpu" else "cpu"
    scaler     = torch.amp.GradScaler(enabled=(use_amp and device != "cpu"))

    ensure_dir(checkpoint_dir)
    resume_path = Path(checkpoint_dir) / "last_training.pt"
    start_epoch = 1
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        ema.shadow = saved["ema"]
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        best_val, no_improve = saved["best_val"], saved["no_improve"]
        snapshots = saved["snapshots"]
        start_epoch = saved["epoch"] + 1
        best_state = torch.load(Path(checkpoint_dir) / "best_ema.pt", map_location="cpu", weights_only=False)["model"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        if hasattr(train_loader.sampler, "rng") and saved["sampler_rng"] is not None:
            train_loader.sampler.rng.set_state(saved["sampler_rng"])
        LOG.info("Resuming at epoch %d from %s", start_epoch, resume_path)

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="Training HybridBathNet v3", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "ext": 0.0, "depth": 0.0, "recon": 0.0}
        n_batches = 0

        # Curriculum NLL annealing
        curr_nll_w = base_nll_weight * min(1.0, epoch / max(1, nll_warmup_epochs))
        criterion.w_nll = curr_nll_w

        current_lr = optimizer.param_groups[0]["lr"]

        for x, y_patch, m_patch, raw in train_loader:
            x      = x.to(device)
            y_patch = y_patch.to(device).unsqueeze(1)   # (B, 1, H, W)
            m_patch = m_patch.to(device).unsqueeze(1)   # (B, 1, H, W)

            optimizer.zero_grad()

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                # Use forward_scalar for compatibility with scalar CompositePhysicsNLLLoss
                if hasattr(criterion, 'forward_joint'):
                    outputs = model(x, mask=m_patch, raw_reflectance=raw.to(device))
                    cy, cx = y_patch.shape[-2] // 2, y_patch.shape[-1] // 2
                    pred_norm = (outputs['depth'][:,0,cy,cx]-y_mean)/(y_std+1e-6)
                    log_var_sc, z_ext = outputs['log_var'][:,0,cy,cx], outputs['z_ext']
                else:
                    pred_norm, log_var_sc, kd_avg, z_ext = model.forward_scalar(
                        x, y_mean, y_std, mask=m_patch, raw_reflectance=raw.to(device))
                # Compute centre-pixel ground truth in normalised form
                cy, cx = y_patch.shape[2] // 2, y_patch.shape[3] // 2
                y_norm_sc = (y_patch[:, 0, cy, cx] - y_mean) / (y_std + 1e-6)

                if hasattr(criterion, 'forward_joint'):
                    loss, loss_dict = criterion.forward_joint(
                        outputs, raw.to(device), m_patch, y_norm_sc, y_mean, y_std)
                else:
                    loss, loss_dict = criterion(
                        pred_norm, log_var_sc, y_norm_sc, z_ext,
                        y_mean=y_mean, y_std=y_std)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss_dict}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_hybrid_gradients(model, grad_clip_norm, separate_uncertainty_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update()    # §4.5

            totals["loss"] += loss_dict["loss_total"]
            totals["nll"]  += loss_dict["loss_nll"]
            totals["ext"]  += loss_dict.get("loss_ext", 0.0)
            totals['depth'] += loss_dict.get('loss_depth', 0.0)
            totals['recon'] += loss_dict.get('loss_recon', 0.0)
            n_batches      += 1

        scheduler.step()

        if n_batches > 0:
            for k in totals:
                totals[k] /= n_batches

        # §5.2 — save snapshot at end of each cosine cycle
        # A cycle ends when (epoch % T_0 == 0)
        if epoch % lr_restart_period == 0:
            snap_path = Path(checkpoint_dir) / f"hybridbathnet_snapshot_ep{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "y_mean": y_mean, "y_std": y_std}, snap_path)
            snapshots.append(str(snap_path))
            LOG.info("Snapshot saved → %s", snap_path)

        # Evaluate EMA model on validation (§4.5)
        backup = copy.deepcopy(model.state_dict())
        ema.apply_shadow()
        val_metrics = evaluate_hybridbathnet(model, val_loader, device, y_mean, y_std)
        candidate_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        ema.restore(backup)
        if not np.isfinite(val_metrics["rmse"]):
            raise FloatingPointError(f"Non-finite validation RMSE at epoch {epoch}")
        history.append({"epoch": epoch, "learning_rate": current_lr,
                        "loss": totals["loss"], "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"], "val_r2": val_metrics["r2"],
                        "loss_depth": totals['depth'], "loss_nll": totals['nll'],
                        "loss_ext": totals['ext'], "loss_recon": totals['recon']})
        (Path(checkpoint_dir) / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

        epoch_bar.set_postfix(
            loss=f"{totals['loss']:.3f}",
            lr=f"{current_lr:.2e}",
            val_rmse=f"{val_metrics['rmse']:.2f}m",
        )
        LOG.info(
            "Epoch %3d/%d | L=%.4f NLL[w=%.2f]=%.4f Ext=%.4f | LR=%.2e | "
            "EMA Val → RMSE=%.3fm R²=%.3f Acc±1m=%.1f%% Acc±2m=%.1f%% "
            "Aleatoric_std=%.3fm",
            epoch, epochs,
            totals["loss"], curr_nll_w, totals["nll"], totals["ext"],
            current_lr,
            val_metrics["rmse"], val_metrics["r2"],
            val_metrics["acc_within_1m_%"], val_metrics["acc_within_2m_%"],
            val_metrics["mean_aleatoric_std_m"],
        )

        # Early stopping
        if val_metrics["rmse"] < best_val:
            best_val   = val_metrics["rmse"]
            best_state = candidate_state
            torch.save({"model": best_state, "epoch": epoch,
                        "val_rmse": best_val}, Path(checkpoint_dir) / "best_ema.pt")
            no_improve = 0
        else:
            no_improve += 1
        resume_tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "ema": ema.shadow,
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "history": history, "best_val": best_val, "no_improve": no_improve,
                    "snapshots": snapshots, "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "sampler_rng": train_loader.sampler.rng.get_state() if hasattr(train_loader.sampler, "rng") else None}, resume_tmp)
        resume_tmp.replace(resume_path)
        if no_improve >= early_stop_patience:
            LOG.info(
                "Early stopping triggered at epoch %d (no improvement for %d epochs).",
                epoch, early_stop_patience
            )
            break

    # Restore best EMA weights
    if best_state is not None:
        model.load_state_dict(best_state)
        LOG.info("Loaded best checkpoint (val RMSE: %.3f m)", best_val)

    return model, snapshots


# ---------------------------------------------------------------------------
# Evaluation  (§1.1 σ fix, §5.4 calibration metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hybridbathnet(
    model, loader, device, y_mean, y_std,
    compute_calibration: bool = False,
    predictions_path=None,
) -> dict:
    """Evaluate model on a DataLoader.

    §1.1 fix: mean_aleatoric_std_m is in metres (depth output is already in
    metres because the model uses softplus; log_var is on the same scale).
    """
    model.eval()
    all_pred, all_true, all_sigma = [], [], []
    all_ext, all_kd = [], []

    for x, y_patch, m_patch, raw in loader:
        x       = x.to(device)
        m_patch = m_patch.to(device).unsqueeze(1)

        out = model(x, mask=m_patch, raw_reflectance=raw.to(device))
        depth   = out["depth"].squeeze(1).cpu()    # (B, H, W) metres
        log_var = out["log_var"].squeeze(1).cpu()  # (B, H, W)

        # Centre pixel for scalar comparison
        cy, cx = depth.shape[1] // 2, depth.shape[2] // 2
        depth_sc   = depth[:, cy, cx].numpy()
        log_var_sc = log_var[:, cy, cx].numpy()

        # §1.1: σ is in metres because depth output is in metres
        sigma_sc = np.sqrt(np.exp(log_var_sc))

        # Ground truth: centre-pixel depth (metres, positive below sea level)
        y_centre = y_patch[:, cy, cx].numpy()

        all_pred.append(depth_sc)
        all_true.append(y_centre)
        all_sigma.append(sigma_sc)
        all_ext.append(out["z_ext"].flatten().cpu().numpy())
        all_kd.append(out["kd_avg"].flatten().cpu().numpy())

    all_pred  = np.concatenate(all_pred)
    all_true  = np.concatenate(all_true)
    all_sigma = np.concatenate(all_sigma)
    all_ext = np.concatenate(all_ext)
    all_kd = np.concatenate(all_kd)
    if not all(np.isfinite(a).all() for a in (all_pred, all_true, all_sigma, all_ext, all_kd)):
        raise FloatingPointError("Non-finite prediction, target, uncertainty, or physics bound")
    if predictions_path is not None:
        np.savez(predictions_path, target=all_true, prediction=all_pred,
                 sigma=all_sigma, z_ext=all_ext, kd=all_kd)

    metrics = compute_metrics(all_true, all_pred)

    # §1.1 fix — report σ in metres
    metrics["mean_aleatoric_std_m"] = float(np.mean(all_sigma))
    metrics["mean_z_ext_m"] = float(all_ext.mean())
    metrics["reference_above_bound_pct"] = float(100 * np.mean(all_true > all_ext))
    metrics["mean_extinction_excess_m"] = float(np.maximum(all_pred-all_ext-1, 0).mean())
    unbounded = np.log(model.i0_over_epsilon) / (2 * all_kd + 1e-6)
    metrics["bound_clipped_pct"] = float(100 * np.mean(np.abs(unbounded-all_ext) > 1e-4))

    # Depth-bin breakdown
    metrics.update(compute_depth_bin_metrics(all_true, all_pred))

    if compute_calibration:
        calib = calibration_metrics(all_true, all_pred, all_sigma)
        metrics.update(calib)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--seed",      type=int, default=42, help="Random seed")
    parser.add_argument("--no-amp",    action="store_true",  help="Disable AMP")
    parser.add_argument("--no-rotations", action="store_true",
                        help="Disable 90° rotation augmentation (ablation §3.8)")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg      = load_config(args.config)
    torch.set_num_threads(cfg["training"].get("cpu_threads", 4))
    feat_dir = cfg["features"]["features_dir"]
    data     = np.load(Path(feat_dir) / "patches.npz")
    X, Y     = data["X"], data["Y"]
    expected_channels = cfg.get("hybridbathnet", {}).get("in_channels", 16)
    if X.shape[1] != expected_channels:
        raise ValueError(f"Stale feature data: expected {expected_channels} channels, got {X.shape[1]}. Rebuild features.")
    M        = data.get("M", np.ones_like(Y))   # fall back to all-water if no mask saved
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    if any(len(idx) == 0 for idx in (train_idx, val_idx, test_idx)):
        raise ValueError("Train, validation, and test splits must all be nonempty")

    LOG.info("Patches — X: %s  Y: %s  channels: %d", X.shape, Y.shape, X.shape[1])

    train_cfg = cfg["training"]
    hnet_cfg  = cfg.get("hybridbathnet", {})
    device    = train_cfg["device"] if torch.cuda.is_available() else "cpu"
    LOG.info("Device: %s  |  Seed: %d  |  AMP: %s", device, args.seed, not args.no_amp)

    # Datasets
    base_train_ds = PatchDataset(X, Y, M, train_idx)
    base_val_ds   = PatchDataset(
        X, Y, M, val_idx if len(val_idx) else train_idx,
        mean=base_train_ds.mean, std=base_train_ds.std,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std,
    )
    base_test_ds  = PatchDataset(
        X, Y, M, test_idx if len(test_idx) else train_idx,
        mean=base_train_ds.mean, std=base_train_ds.std,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std,
    )

    noise_std      = hnet_cfg.get("aug_noise_std", 0.02)
    use_rotations  = not args.no_rotations
    train_ds = SpectralAugDataset(base_train_ds, augment=True,
                                  noise_std=noise_std, use_rotations=use_rotations)
    val_ds   = SpectralAugDataset(base_val_ds,   augment=False)
    test_ds  = SpectralAugDataset(base_test_ds,  augment=False)
    LOG.info("SpectralAug: noise_std=%.3f  rotations=%s", noise_std, use_rotations)

    # §4.1 — use sampler OR loss weighting, not both
    use_sampler = hnet_cfg.get("use_depth_sampler", True)
    use_depth_w = hnet_cfg.get("use_depth_weighting", False) and not use_sampler

    n_strata       = hnet_cfg.get("depth_strata", 5)
    cy_tr, cx_tr   = base_train_ds.Y.shape[1] // 2, base_train_ds.Y.shape[2] // 2
    train_depths   = base_train_ds.Y[:, cy_tr, cx_tr]
    batch_size     = train_cfg["batch_size"]

    if use_sampler:
        sampler      = DepthStratifiedSampler(train_depths, n_strata=n_strata, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        LOG.info("DepthStratifiedSampler: %d strata  %d samples/epoch", n_strata, len(sampler))
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model (§1.2 — pass mean/std so model can de-normalise reflectance)
    mu  = base_train_ds.mean.squeeze().astype(np.float32)
    sig = base_train_ds.std.squeeze().astype(np.float32)

    model = HybridBathNetModel(
        in_channels     = X.shape[1],
        n_spectral      = hnet_cfg.get("n_spectral", 3),
        mean            = mu,
        std             = sig,
        feature_dim     = hnet_cfg.get("feature_dim", FEATURE_DIM),
        dropout_rate    = hnet_cfg.get("dropout_rate", 0.1),
        i0_over_epsilon = hnet_cfg.get("i0_over_epsilon", 100.0),
        use_red_kd      = hnet_cfg.get("use_red_kd", True),
        use_recon_head  = hnet_cfg.get("use_recon_head", True),
        n_recon_bands   = hnet_cfg.get("n_recon_bands", 3),
        kd_aggregation = hnet_cfg.get("kd_aggregation", "mean"),
        physics_conditioning = hnet_cfg.get("physics_conditioning", "spatial"),
        z_ext_min = hnet_cfg.get("z_ext_min", 1.0),
        z_ext_max = hnet_cfg.get("z_ext_max", 50.0),
        physics_enabled = hnet_cfg.get("physics_enabled", True),
        nonnegative_depth = hnet_cfg.get("nonnegative_depth", True),
        use_explicit_reflectance = hnet_cfg.get("use_explicit_reflectance", True),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("HybridBathNet v3 | Trainable params: %s", f"{n_params:,}")

    loss_weights = hnet_cfg.get("loss_weights", {})
    criterion    = CompositePhysicsNLLLoss(
        w_mse               = loss_weights.get("w_mse", 1.0),
        w_nll               = loss_weights.get("w_nll", 1.0),
        w_phys              = loss_weights.get("w_phys", 0.3),
        w_ext               = loss_weights.get("w_ext", 0.3),
        depth_scale         = hnet_cfg.get("depth_weight_scale", 10.0),
        use_depth_weighting = use_depth_w,
        beta_nll            = hnet_cfg.get("beta_nll", 0.5),
        huber_delta         = hnet_cfg.get("huber_delta", 0.5),
    )

    epochs       = train_cfg["epochs"]
    lr           = train_cfg["learning_rate"]
    lr_restart   = hnet_cfg.get("lr_restart_period", 30)
    grad_clip    = hnet_cfg.get("grad_clip_norm", 1.0)
    nll_warmup   = hnet_cfg.get("nll_warmup_epochs", 20)
    ema_decay    = hnet_cfg.get("ema_decay", 0.999)
    patience     = hnet_cfg.get("early_stop_patience", 15)
    ckpt_dir     = train_cfg["checkpoint_dir"]

    LOG.info("Normalization → y_mean: %.3f  y_std: %.3f", base_train_ds.y_mean, base_train_ds.y_std)
    LOG.info("Loss weights  → MSE=%.2f NLL=%.2f PhysNonNeg=%.2f Ext=%.2f",
             loss_weights.get("w_mse", 1.0), loss_weights.get("w_nll", 1.0),
             loss_weights.get("w_phys", 0.3), loss_weights.get("w_ext", 0.3))

    if args.evaluate_only:
        checkpoint = torch.load(Path(ckpt_dir) / "hybridbathnet.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        snapshots = checkpoint.get("snapshots", [])
    else:
        model, snapshots = train_hybridbathnet(
            model, criterion, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device,
            y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std,
            nll_warmup_epochs=nll_warmup,
            base_nll_weight=loss_weights.get("w_nll", 1.0),
            grad_clip_norm=grad_clip,
            lr_restart_period=lr_restart,
            ema_decay=ema_decay,
            early_stop_patience=patience,
            checkpoint_dir=ckpt_dir,
            use_amp=(not args.no_amp),
        )

    # Save final checkpoint with all metadata
    ensure_dir(ckpt_dir)
    if not args.evaluate_only:
        torch.save({
        "model":    model.state_dict(),
        "y_mean":   base_train_ds.y_mean,
        "y_std":    base_train_ds.y_std,
        "ch_mean":  mu.tolist(),
        "ch_std":   sig.tolist(),
        "version":  "v3",
        "n_channels": X.shape[1],
        "snapshots": snapshots,
        "seed":     args.seed,
        "configuration": cfg,
        "runtime_seconds": time.perf_counter() - started,
    }, Path(ckpt_dir) / "hybridbathnet.pt")
    LOG.info("Saved final checkpoint → %s/hybridbathnet.pt", ckpt_dir)

    # Test evaluation with calibration metrics
    results_dir = ensure_dir(cfg["evaluation"]["results_dir"])
    test_metrics = evaluate_hybridbathnet(
        model, val_loader if args.validation_only else test_loader, device,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std,
        compute_calibration=True,
        predictions_path=results_dir / ("validation_predictions.npz" if args.validation_only else "test_predictions.npz"),
    )
    LOG.info(
        "HybridBathNet v3 (EVALUATION) → RMSE %.3f m  MAE %.3f m  R² %.3f  "
        "Acc±1m %.1f%%  Acc±2m %.1f%%  σ_ale %.3f m  "
        "NLL %.3f  Cov68 %.1f%%  Cov90 %.1f%%  σ-|e|corr %.3f",
        test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"],
        test_metrics["acc_within_1m_%"], test_metrics["acc_within_2m_%"],
        test_metrics["mean_aleatoric_std_m"],
        test_metrics.get("gaussian_nll", float("nan")),
        test_metrics.get("coverage_68pct", float("nan")),
        test_metrics.get("coverage_90pct", float("nan")),
        test_metrics.get("sigma_error_corr", float("nan")),
    )

    # §5.5 Conformal recalibration: compute k on val set
    val_metrics = evaluate_hybridbathnet(
        model, val_loader, device,
        y_mean=base_train_ds.y_mean, y_std=base_train_ds.y_std,
        compute_calibration=True,
    )
    # (For full conformal we need per-sample errors and sigmas — simplified here
    # using coverage_68pct to report whether calibration is needed)
    LOG.info(
        "Val calibration → NLL %.3f  Cov68 %.1f%%  Cov90 %.1f%%",
        val_metrics.get("gaussian_nll", float("nan")),
        val_metrics.get("coverage_68pct", float("nan")),
        val_metrics.get("coverage_90pct", float("nan")),
    )

    # Write results file (also write the cnn_baseline_results.json for compare_baselines.py)
    ensure_dir(cfg["evaluation"]["results_dir"])
    test_metrics["evaluation_split"] = "validation" if args.validation_only else "test"
    test_metrics["runtime_seconds"] = time.perf_counter() - started
    results_path = Path(cfg["evaluation"]["results_dir"]) / (
        "validation_results.json" if args.validation_only else "hybridbathnet_results.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    LOG.info("Saved results → %s", results_path)

    # §7 — also write cnn_baseline_results.json so compare_baselines.py works
    cnn_compat = {k: v for k, v in test_metrics.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
    with open(Path(cfg["evaluation"]["results_dir"]) / "cnn_baseline_results.json", "w") as f:
        json.dump(cnn_compat, f, indent=2)


if __name__ == "__main__":
    main()
