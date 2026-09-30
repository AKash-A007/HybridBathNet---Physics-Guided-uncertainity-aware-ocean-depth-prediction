"""
HybridBathNet Architecture
==========================
Physics-Guided + Uncertainty-Aware Ocean Depth Prediction Model.

Components:
1. PhysicsGuidedEncoder       : 3-block spatial Conv2D encoder extracting 128-D features.
2. BeerLambertPrior           : Calculates attenuation prior K_d and extinction depth limit z_ext.
3. PhysicsConstrainedDecoder  : Physics-guided decoder layer that fuses 128-D spatial features
                                with K_d & Beer-Lambert extinction depth bounds.
4. DepthHead                  : MLP predicting mean water depth \\hat{y} (meters).
5. AleatoricUncertaintyHead   : MLP predicting scalar log-variance s = ln(\\sigma^2).
6. CompositePhysicsNLLLoss    : Composite loss combining Gaussian NLL + non-negativity penalty
                                + extinction depth bound penalty.
7. predict_with_uncertainty   : MC Dropout inference for epistemic & total uncertainty.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 128


# ---------------------------------------------------------------------------
# 1. Physics Prior: Beer-Lambert Light Extinction Calculation
# ---------------------------------------------------------------------------

def compute_kd_and_extinction(
    blue: torch.Tensor,
    green: torch.Tensor,
    i0_over_epsilon: float = 100.0,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the diffuse attenuation coefficient (K_d) and maximum physical
    extinction depth limit (z_extinction) using the Beer-Lambert optical law.

    Beer-Lambert Law:
        I(z) = I_0 * exp(-2 * K_d * z)
        => z_extinction = ln(I_0 / epsilon) / (2 * K_d)

    K_d proxy (QAA / Morel-based Blue-to-Green reflectance ratio):
        K_d(490) \\approx 0.0166 + 0.156 * (R_blue / R_green)^(-1.14)

    Args:
        blue: (B, 1, H, W) or (B, H, W) Blue reflectance (B02)
        green: (B, 1, H, W) or (B, H, W) Green reflectance (B03)
        i0_over_epsilon: Signal-to-noise ratio threshold (I_0 / \\epsilon)
        eps: Small stability constant

    Returns:
        K_d: (B, 1) average diffuse attenuation coefficient across patch
        z_ext: (B, 1) maximum optical extinction depth limit (meters)
    """
    b = torch.clamp(blue, min=eps)
    g = torch.clamp(green, min=eps)

    bg_ratio = b / g  # Blue/Green ratio
    # QAA optical water clarity proxy formula for K_d (m^-1)
    kd_map = 0.0166 + 0.156 * torch.pow(bg_ratio, -1.14)
    kd_map = torch.clamp(kd_map, min=0.01, max=5.0)

    # Average K_d over the spatial patch
    if kd_map.dim() == 4:
        kd_avg = kd_map.mean(dim=(-2, -1))  # (B, 1)
    elif kd_map.dim() == 3:
        kd_avg = kd_map.mean(dim=(-2, -1), keepdim=True)  # (B, 1)
    else:
        kd_avg = kd_map

    # Beer-Lambert maximum depth bound (meters)
    # 2 * K_d accounts for 2-way path (down to seabed and back to satellite)
    z_ext = math.log(i0_over_epsilon) / (2.0 * kd_avg + eps)
    z_ext = torch.clamp(z_ext, min=1.0, max=50.0)

    return kd_avg, z_ext


# ---------------------------------------------------------------------------
# 2. Physics-Guided Encoder
# ---------------------------------------------------------------------------

class PhysicsGuidedEncoder(nn.Module):
    """Spatial feature encoder compressing multi-band patch (B, C, H, W) into
    a 128-dimensional latent feature vector (B, FEATURE_DIM).
    Includes Dropout for MC-Dropout uncertainty estimation.

    Args:
        in_channels: 6 (4 spectral bands + 1 Stumpf ratio + 1 Beer-Lambert z_prior channel)
    """

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.1):

        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = block(64, FEATURE_DIM)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            f_spatial: (B, FEATURE_DIM)
        """
        x = self.enc1(x)
        x = self.enc2(self.pool1(x))
        x = self.bottleneck(self.pool2(x))
        x = self.gap(x)
        return self.flatten(x)


# ---------------------------------------------------------------------------
# 3. Physics-Constrained Decoder
# ---------------------------------------------------------------------------

class PhysicsConstrainedDecoder(nn.Module):
    """Physics-constrained feature decoder layer implementing Beer-Lambert optical depth bounding.

    Stage 2 Physics Integration:
    -----------------------------
    Takes the CNN spatial feature vector f_spatial (128-D) and physical attenuation bounds:
      1. Computes maximum optical extinction depth limit:
            z_extinction = ln(I_0 / epsilon) / (2 * K_d)
      2. Applies non-negative activation + physical extinction bounding:
            z_bounded = min(Softplus(f_spatial), z_extinction)

      Physics Rationale for Project Presentation:
      -------------------------------------------
      - Softplus(f_spatial): Enforces strictly non-negative depth predictions (depth >= 0 meters),
        reflecting the physical reality that bathymetry is measured below sea level.
      - min(..., z_extinction): Implements a physical optical ceiling. According to the Beer-Lambert
        law, light cannot penetrate water beyond its extinction limit z_extinction for a given
        turbidity Kd. If the spatial CNN encoder attempts to predict a depth deeper than z_extinction,
        the min() operation caps the representation, preventing unphysical depth extrapolations in
        opaque/turbid waters.

      3. Fuses z_bounded with physical priors [K_d, z_extinction] into refined feature vector f_physics
         which feeds both the Depth Regression Head and Aleatoric Uncertainty Head.

    Args:
        feature_dim: Dimension of spatial feature vector (128)
        physics_dim: Output physics feature dimension (128)
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, physics_dim: int = FEATURE_DIM):
        super().__init__()
        self.phys_embed = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Gate to balance learned spatial features vs physics prior embedding
        self.gate_fc = nn.Sequential(
            nn.Linear(feature_dim + 32, physics_dim),
            nn.Sigmoid(),
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim + 32, physics_dim),
            nn.LayerNorm(physics_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        f_spatial: torch.Tensor,
        kd: torch.Tensor,
        z_ext: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_spatial: (B, 128) encoder spatial feature vector
            kd: (B, 1) diffuse attenuation coefficient prior
            z_ext: (B, 1) optical extinction depth bound

        Returns:
            f_physics: (B, 128) physically-bounded feature vector
        """
        # Step 2a & 2b: Apply Softplus + min() physical bounding constraint
        f_softplus = F.softplus(f_spatial)                      # (B, 128) — depth >= 0m
        z_bounded = torch.minimum(f_softplus, z_ext)           # (B, 128) — depth <= z_extinction

        # Step 2c: Physics parameter embedding & dynamic feature fusion
        phys_params = torch.cat([kd, z_ext], dim=-1)             # (B, 2)
        phys_emb = self.phys_embed(phys_params)                  # (B, 32)

        concat_feat = torch.cat([z_bounded, phys_emb], dim=-1)   # (B, 160)

        # Dynamic physics gating mechanism
        gate = self.gate_fc(concat_feat)                         # (B, 128)
        f_fused = self.fusion_fc(concat_feat)                    # (B, 128)

        # Gated integration of spatial features and physical constraints
        f_physics = gate * f_fused + (1.0 - gate) * z_bounded
        return f_physics


# ---------------------------------------------------------------------------
# 4. Prediction Heads (Dual Head)
# ---------------------------------------------------------------------------

class DepthHead(nn.Module):
    """Predicts scalar water depth \\hat{y} in meters from physics-regularized features."""

    def __init__(self, feature_dim: int = FEATURE_DIM, dropout_rate: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, f_physics: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            depth_pred: (B,) predicted depth in meters
        """
        return self.mlp(f_physics).squeeze(-1)


class AleatoricUncertaintyHead(nn.Module):
    """Predicts log variance s = ln(\\sigma^2) representing aleatoric data uncertainty."""

    def __init__(self, feature_dim: int = FEATURE_DIM, dropout_rate: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, f_physics: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            log_var: (B,) predicted log variance s = ln(\\sigma^2)
        """
        log_var = self.mlp(f_physics).squeeze(-1)
        # Clamped for numerical stability during exponentiation in NLL loss
        return torch.clamp(log_var, min=-10.0, max=10.0)


# ---------------------------------------------------------------------------
# 5. Full HybridBathNet Model Assembly
# ---------------------------------------------------------------------------

class HybridBathNetModel(nn.Module):
    """Complete HybridBathNet pipeline integrating:
    - Physics-Guided Encoder
    - Beer-Lambert Physical Prior Computation
    - Physics-Constrained Decoder Layer (Softplus + min(..., z_ext))
    - Dual Heads (DepthHead + AleatoricUncertaintyHead)
    """

    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = FEATURE_DIM,
        dropout_rate: float = 0.1,
        i0_over_epsilon: float = 100.0,
    ):

        super().__init__()
        self.i0_over_epsilon = i0_over_epsilon
        self.encoder = PhysicsGuidedEncoder(in_channels=in_channels, dropout_rate=dropout_rate)
        self.physics_decoder = PhysicsConstrainedDecoder(feature_dim=feature_dim)
        self.depth_head = DepthHead(feature_dim=feature_dim, dropout_rate=dropout_rate)
        self.uncertainty_head = AleatoricUncertaintyHead(feature_dim=feature_dim, dropout_rate=dropout_rate)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input image patch stack.
               Bands assumed: B02 (ch 0), B03 (ch 1), B04 (ch 2), B08 (ch 3), ...

        Returns:
            depth_pred: (B,) predicted mean depth \\hat{y} (m)
            log_var: (B,) predicted log-variance s = ln(\\sigma^2)
            kd: (B, 1) diffuse attenuation coefficient K_d (m^-1)
            z_ext: (B, 1) Beer-Lambert optical extinction depth limit (m)
        """
        # Channel 0: Blue (B02), Channel 1: Green (B03)
        blue = x[:, 0:1, :, :]
        green = x[:, 1:2, :, :]

        # 1. Compute Beer-Lambert attenuation coefficient and extinction depth bound
        kd, z_ext = compute_kd_and_extinction(blue, green, i0_over_epsilon=self.i0_over_epsilon)

        # 2. Extract deep spatial latent features via Encoder
        f_spatial = self.encoder(x)  # (B, 128)

        # 3. Fuse spatial features with optical attenuation prior via Physics Decoder
        f_physics = self.physics_decoder(f_spatial, kd, z_ext)  # (B, 128)

        # 4. Predict via Dual Heads
        depth_pred = self.depth_head(f_physics)       # (B,)
        log_var = self.uncertainty_head(f_physics)   # (B,)

        return depth_pred, log_var, kd, z_ext


# ---------------------------------------------------------------------------
# 6. Composite Physics-Informed Gaussian NLL Loss
# ---------------------------------------------------------------------------

class CompositePhysicsNLLLoss(nn.Module):
    """Composite Physics-Informed Gaussian Negative Log-Likelihood Loss.

    Loss Formulation:
        L_total = w_nll * L_NLL + w_phys * L_nonneg + w_ext * L_extinction

    1. Gaussian NLL Loss (Heteroscedastic Aleatoric Uncertainty):
        L_NLL = 0.5 * exp(-s) * (y - \\hat{y})^2 + 0.5 * s
        where s = ln(\\sigma^2).
        Minimizing L_NLL automatically balances residual error with uncertainty estimates.

    2. Non-Negativity Physics Loss:
        L_nonneg = mean(ReLU(-\\hat{y}))
        Penalizes invalid negative depth predictions above sea level.

    3. Beer-Lambert Extinction Limit Physics Loss:
        L_extinction = mean(ReLU(\\hat{y} - z_ext))
        Penalizes depth predictions exceeding physical light penetration limits for given K_d.

    Args:
        w_nll: Weight for Gaussian NLL loss term (default: 1.0)
        w_phys: Weight for non-negativity physical constraint (default: 0.5)
        w_ext: Weight for Beer-Lambert extinction bound constraint (default: 0.5)
    """

    def __init__(self, w_mse: float = 1.0, w_nll: float = 1.0, w_phys: float = 0.5, w_ext: float = 0.5):
        super().__init__()
        self.w_mse = w_mse
        self.w_nll = w_nll
        self.w_phys = w_phys
        self.w_ext = w_ext


    def forward(
        self,
        pred_depth_norm: torch.Tensor,
        log_var: torch.Tensor,
        target_depth_norm: torch.Tensor,
        z_ext: torch.Tensor,
        y_mean: float = 0.0,
        y_std: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred_depth_norm: (B,) normalized predicted depth
            log_var: (B,) predicted log variance s = ln(\\sigma^2)
            target_depth_norm: (B,) normalized ground truth depth
            z_ext: (B, 1) or (B,) optical extinction depth limit (m)
            y_mean: dataset mean depth (m)
            y_std: dataset std depth (m)

        Returns:
            total_loss: scalar torch.Tensor for loss.backward()
            loss_components: dict of individual loss values for logging
        """
        # 1. Direct Depth Regression MSE Loss
        diff_sq_norm = (pred_depth_norm - target_depth_norm) ** 2
        loss_mse = torch.mean(diff_sq_norm)

        # 2. Heteroscedastic Gaussian NLL Loss on normalized depths
        inv_var = torch.exp(-log_var)
        l_nll = 0.5 * (inv_var * diff_sq_norm + log_var)
        loss_nll = torch.mean(l_nll)

        # Reconstruct actual meters for physics constraints
        pred_depth_m = pred_depth_norm * y_std + y_mean
        target_depth_m = target_depth_norm * y_std + y_mean
        diff_sq_m = (pred_depth_m - target_depth_m) ** 2

        # 3. Physics Constraint 1: Non-negativity (bathymetry depth >= 0m)
        loss_nonneg = torch.mean(F.relu(-pred_depth_m))

        # 4. Physics Constraint 2: Extinction depth limit bound (\\hat{y} <= z_ext)
        z_ext_flat = z_ext.squeeze(-1)
        loss_ext = torch.mean(F.relu(pred_depth_m - z_ext_flat))

        # Composite total loss calculation (MSE + NLL + Physics non-neg + Physics extinction)
        total_loss = (
            self.w_mse * loss_mse +
            self.w_nll * loss_nll +
            self.w_phys * (loss_nonneg / (y_std + 1e-6)) +
            self.w_ext * (loss_ext / (y_std + 1e-6))
        )

        loss_components = {
            "loss_total": float(total_loss.item()),
            "loss_mse": float(loss_mse.item()),
            "loss_nll": float(loss_nll.item()),
            "loss_nonneg": float(loss_nonneg.item()),
            "loss_ext": float(loss_ext.item()),
            "rmse": float(torch.sqrt(torch.mean(diff_sq_m)).item()),
        }

        return total_loss, loss_components


# ---------------------------------------------------------------------------
# 7. MC Dropout Inference for Epistemic & Total Uncertainty
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20
) -> dict[str, torch.Tensor]:
    """Performs Monte Carlo (MC) Dropout stochastic inference to quantify:
    - Epistemic uncertainty (model uncertainty from dropout variance)
    - Aleatoric uncertainty (data noise from uncertainty head)
    - Total predictive uncertainty (aleatoric + epistemic)

    Args:
        model: Trained HybridBathNetModel instance
        x: (B, C, H, W) input image tensor
        n_samples: Number of MC stochastic forward passes (T = 20)

    Returns:
        dict containing:
            'mean_depth': (B,) predictive mean depth \\bar{y}
            'aleatoric_var': (B,) expected aleatoric variance E[\\sigma_a^2]
            'epistemic_var': (B,) epistemic variance Var(\\hat{y}_t)
            'total_var': (B,) total variance \\sigma_{total}^2
            'total_std': (B,) total predictive standard deviation \\sigma_{total}
    """
    model.eval()
    # Enable dropout layers for stochastic sampling
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    preds = []
    log_vars = []

    with torch.no_grad():
        for _ in range(n_samples):
            depth_pred, log_var, _, _ = model(x)
            preds.append(depth_pred.unsqueeze(0))      # (1, B)
            log_vars.append(log_var.unsqueeze(0))    # (1, B)

    # Stack stochastic passes: (T, B)
    preds = torch.cat(preds, dim=0)
    log_vars = torch.cat(log_vars, dim=0)

    # Calculate statistics
    mean_depth = torch.mean(preds, dim=0)                             # (B,)
    epistemic_var = torch.var(preds, dim=0, unbiased=True)            # (B,)
    aleatoric_var = torch.mean(torch.exp(log_vars), dim=0)           # (B,)
    total_var = aleatoric_var + epistemic_var                        # (B,)
    total_std = torch.sqrt(total_var)                                 # (B,)

    return {
        "mean_depth": mean_depth,
        "aleatoric_var": aleatoric_var,
        "epistemic_var": epistemic_var,
        "total_var": total_var,
        "total_std": total_std,
    }
