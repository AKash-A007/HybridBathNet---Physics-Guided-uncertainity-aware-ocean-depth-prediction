import unittest
import numpy as np
import torch
from src.hybridbathnet import HybridBathNetModel
from src.joint_objective import CoverageSampler, JointCompositeLoss, SquaredDepthJointLoss, initialize_optical_head


class JointTests(unittest.TestCase):
    def test_squared_depth_only_replaces_huber_term(self):
        depth=torch.ones(2,1,4,4,requires_grad=True)
        raw=torch.full((2,3,4,4),.03)
        out={'depth':depth,'log_var':torch.zeros_like(depth),'z_ext':torch.full((2,1),10.),
             'r_hat':raw}
        args=(out,raw,torch.ones(2,1,4,4),torch.zeros(2),0.,1.)
        common=dict(w_mse=1,w_nll=1,w_ext=.03,w_phys=0,w_recon=.1,
                    use_depth_weighting=False,beta_nll=.5,huber_delta=.5)
        baseline,_=JointCompositeLoss(**common).forward_joint(*args)
        squared,parts=SquaredDepthJointLoss(**common).forward_joint(*args)
        self.assertAlmostEqual(float((squared-baseline).detach()),.25,places=5)
        self.assertAlmostEqual(parts['loss_depth'],1.,places=5)
        squared.backward()
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_coverage_and_resume_rng(self):
        s=CoverageSampler(101,42)
        first=list(s); state=s.rng.get_state(); second=list(s)
        self.assertEqual(sorted(first),list(range(101)))
        s.rng.set_state(state); self.assertEqual(list(s),second)
        self.assertNotEqual(first,second)

    def test_invalid_pixels_cannot_change_physics_conditioning(self):
        torch.set_num_threads(2)
        m=HybridBathNetModel(in_channels=16,mask_aware_physics=True).eval()
        x=torch.zeros(1,16,32,32); raw=torch.full((1,3,32,32),.03)
        mask=torch.ones(1,1,32,32); mask[:,:,:,24:]=0
        seen=[]
        hook=m.cond_proj.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
        with torch.no_grad():
            m(x,mask,raw); raw[:,:,:,24:]=100; m(x,mask,raw)
        hook.remove(); torch.testing.assert_close(seen[0],seen[1])
        with self.assertRaises(ValueError): m(x,torch.zeros_like(mask),raw)

    def test_composite_backward_reaches_all_heads(self):
        torch.manual_seed(42); torch.set_num_threads(2)
        model=HybridBathNetModel(in_channels=16,mask_aware_physics=True).train()
        initialize_optical_head(model)
        x=torch.randn(2,16,32,32)*.1; mask=torch.ones(2,1,32,32)
        raw=torch.rand(2,3,32,32)*.05+.01
        out=model(x,mask,raw)
        objective=JointCompositeLoss(w_recon=.1,w_ext=.03,use_depth_weighting=False)
        loss,parts=objective.forward_joint(out,raw,mask,torch.zeros(2),20.,10.)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        for name in ('encoder','decoder.depth_conv','decoder.logvar_conv','cond_proj','recon_head.kl_head','recon_head.al_head'):
            grads=[p.grad for n,p in model.named_parameters() if n.startswith(name) and p.grad is not None]
            self.assertTrue(grads,name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads),name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0,name)
        self.assertGreater(parts['loss_recon'],0)
        self.assertGreater(parts['loss_nll'],0)


if __name__=='__main__': unittest.main()
