"""Spatial correction of total predictive variance from held-region residuals.

Inputs are physical/semantic quantities shared across independently trained
models. Learned encoder coordinates are deliberately not exchanged across them.
Raw aleatoric/epistemic source components remain separate diagnostics.
"""
import torch
from torch import nn


class SpatialVarianceCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.image=nn.Sequential(nn.Conv2d(9,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.AvgPool2d(2),
            nn.Conv2d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Linear(16+9+4,24),nn.SiLU(),nn.Linear(24,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,patch,semantic,base_variance):
        patch=patch.detach();semantic=semantic.detach();base_variance=base_variance.detach()
        pooled=self.image(patch).flatten(1)
        centre=patch[:,:,patch.shape[-2]//2,patch.shape[-1]//2]
        correction=self.head(torch.cat([pooled,centre,semantic],dim=1)).squeeze(1).clamp(-3,3)
        return (base_variance.float().clamp_min(1e-6).log()+correction.float()).clamp(-8,8)


def variance_nll(log_variance,residual,weight=None):
    loss=.5*(log_variance.float()+residual.float().square()*torch.exp(-log_variance.float()))
    if weight is not None:loss=loss*weight
    return loss.mean()
