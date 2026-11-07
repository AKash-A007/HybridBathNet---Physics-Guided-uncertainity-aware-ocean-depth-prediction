"""Composite depth, aleatoric uncertainty and optical-physics objectives."""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Sampler
from .hybridbathnet import CompositePhysicsNLLLoss


class CoverageSampler(Sampler):
    """Seeded full coverage without replacement; RNG state supports resume."""
    def __init__(self, size, seed=42):
        self.size=size
        self.rng=np.random.RandomState(seed)
    def __len__(self): return self.size
    def __iter__(self): return iter(self.rng.permutation(self.size).tolist())


class JointCompositeLoss(CompositePhysicsNLLLoss):
    def __init__(self, w_recon=.1, reflectance_scale=.05, **kwargs):
        super().__init__(**kwargs)
        self.w_recon=w_recon
        self.reflectance_scale=reflectance_scale

    def forward_joint(self, out, reflectance, mask, target_norm, y_mean, y_std):
        cy,cx=out['depth'].shape[-2]//2,out['depth'].shape[-1]//2
        pred=(out['depth'][:,0,cy,cx].float()-y_mean)/(y_std+1e-6)
        total,parts=super().forward(pred,out['log_var'][:,0,cy,cx],target_norm,
                                    out['z_ext'],y_mean,y_std)
        parts['loss_depth']=float(F.smooth_l1_loss(pred,target_norm.float(),beta=self.huber_delta).detach())
        if self.w_recon:
            valid=mask.bool().expand_as(reflectance)
            if not valid.any(): raise ValueError('No water pixels for reconstruction')
            recon=F.smooth_l1_loss(out['r_hat'].float()/self.reflectance_scale,
                                  reflectance.float()/self.reflectance_scale,reduction='none')[valid].mean()
            if not torch.isfinite(recon): raise FloatingPointError('Nonfinite reconstruction')
        else:
            recon=pred.new_zeros(())
        total=total+self.w_recon*recon
        parts.update(loss_recon=float(recon.detach()),loss_total=float(total.detach()))
        return total,parts


class SquaredDepthJointLoss(JointCompositeLoss):
    """Align the supervised mean term with RMSE while retaining joint losses."""
    def forward_joint(self, out, reflectance, mask, target_norm, y_mean, y_std):
        total, parts = super().forward_joint(out, reflectance, mask, target_norm, y_mean, y_std)
        cy,cx=out['depth'].shape[-2]//2,out['depth'].shape[-1]//2
        prediction=(out['depth'][:,0,cy,cx].float()-y_mean)/(y_std+1e-6)
        target=target_norm.float()
        huber=F.smooth_l1_loss(prediction,target,beta=self.huber_delta)
        squared=F.mse_loss(prediction,target)
        total=total+self.w_mse*(squared-huber)
        if not torch.isfinite(total): raise FloatingPointError('Nonfinite squared-depth joint objective')
        parts['loss_depth']=float(squared.detach())
        parts['loss_total']=float(total.detach())
        return total,parts


def initialize_optical_head(model):
    """Avoid exp(-2 K z) saturation from the default K~0.7 at depth~25 m."""
    with torch.no_grad():
        model.recon_head.kl_head[-1].weight.zero_()
        model.recon_head.kl_head[-1].bias.fill_(-3.)
        model.recon_head.al_head[-1].weight.zero_()
        model.recon_head.al_head[-1].bias.fill_(-2.5)
