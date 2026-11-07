import unittest
import torch
from src.cnn_baseline import clip_hybrid_gradients
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import JointCompositeLoss,initialize_optical_head


class RecoveryUncertaintyTests(unittest.TestCase):
    def test_spatial_variance_does_not_backpropagate_to_shared_features(self):
        torch.set_num_threads(2)
        model=HybridBathNetModel(in_channels=16,uncertainty_head='spatial',
                                detach_uncertainty_features=True,mask_aware_physics=True).eval()
        initialize_optical_head(model)
        raw=torch.rand(2,3,32,32)*.04+.01
        mask=torch.ones(2,1,32,32)
        out=model(torch.randn(2,16,32,32),mask,raw)
        loss,_=JointCompositeLoss(w_mse=0,w_phys=0,w_ext=0,w_recon=0,w_nll=1,
            beta_nll=0,detach_nll_mean=True,use_depth_weighting=False).forward_joint(
            out,raw,mask,torch.zeros(2),10.,5.)
        loss.backward()
        shared=[p.grad for n,p in model.named_parameters() if n.startswith('encoder.')]
        self.assertTrue(all(g is None or torch.count_nonzero(g)==0 for g in shared))
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
            if n.startswith('decoder.uncertainty_head.') and p.grad is not None),0)

    def test_variance_magnitude_cannot_rescale_depth_clipping(self):
        model=HybridBathNetModel(in_channels=16)
        mean=model.decoder.depth_conv.weight
        variance=model.decoder.logvar_conv.weight
        mean.grad=torch.ones_like(mean)*.01
        variance.grad=torch.ones_like(variance)*1000
        before=mean.grad.clone()
        clip_hybrid_gradients(model,1.,separate_uncertainty=True)
        torch.testing.assert_close(mean.grad,before)
        self.assertLessEqual(float(variance.grad.norm()),1.0001)


if __name__=='__main__': unittest.main()
