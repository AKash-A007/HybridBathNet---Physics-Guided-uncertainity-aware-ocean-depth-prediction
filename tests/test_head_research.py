import unittest
import torch
from src.hybridbathnet import HybridBathNetModel, conformal_recalibrate
from src.joint_objective import JointCompositeLoss, initialize_optical_head
from src.optical_limit import multiband_bottom_detection_limit, conditional_depth_precision


class HeadResearchTests(unittest.TestCase):
    def test_calibration_uses_finite_sample_rank(self):
        self.assertEqual(conformal_recalibrate([1]*9, list(range(1,10)), .8), 8.)
        self.assertEqual(conformal_recalibrate([1]*3, [1,2,3], .9), float('inf'))
        with self.assertRaises(ValueError):
            conformal_recalibrate([0], [1])

    def test_variance_training_cannot_change_depth_or_physics_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        model = HybridBathNetModel(in_channels=16, detach_uncertainty_features=True,
            reconstruction_attenuation='patch', mask_aware_physics=True).eval()
        initialize_optical_head(model)
        raw = torch.rand(2, 3, 32, 32) * .03 + .01
        mask = torch.ones(2, 1, 32, 32)
        out = model(torch.randn(2, 16, 32, 32), mask, raw)
        args = (out, raw, mask, torch.zeros(2), 10., 5.)
        common = dict(w_mse=1, w_ext=.03, w_phys=0, w_recon=.1,
                      use_depth_weighting=False, detach_nll_mean=True, beta_nll=0)
        loss, _ = JointCompositeLoss(w_nll=1, **common).forward_joint(*args)
        control, _ = JointCompositeLoss(w_nll=0, **common).forward_joint(*args)
        selected = [p for n, p in model.named_parameters()
                    if n.startswith(('encoder.', 'decoder.depth_conv.', 'cond_proj.', 'recon_head.'))]
        first = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
        second = torch.autograd.grad(control, selected, retain_graph=True, allow_unused=True)
        for a, b in zip(first, second):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                torch.testing.assert_close(a, b)
        variance_grad = torch.autograd.grad(loss, model.decoder.logvar_conv.weight)[0]
        self.assertGreater(float(variance_grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(variance_grad).all())
        torch.testing.assert_close(out['kl_map'], out['kl_map'][:, :, :1, :1].expand_as(out['kl_map']))

    def test_multiband_limit_requires_stated_number_of_bands(self):
        k = torch.tensor([[.1, .2, .4]])
        contrast = torch.ones_like(k) * .1
        noise = torch.ones_like(k) * .001
        result = multiband_bottom_detection_limit(k, k, contrast, noise, required_bands=2)
        expected = torch.log(torch.tensor(.1 / .003)) / .4
        torch.testing.assert_close(result['z_limit_m'][0], expected)
        self.assertFalse(result['operationally_calibrated'])
        with self.assertRaises(ValueError):
            multiband_bottom_detection_limit(k, k, contrast, noise, required_bands=4)

    def test_depth_precision_degrades_with_depth_and_noise(self):
        k = torch.tensor([[.2, .3, .8]])
        c = torch.ones_like(k) * .1
        noise = torch.ones_like(k) * .001
        shallow = conditional_depth_precision(torch.tensor([[2.]]), k, c, noise)
        deep = conditional_depth_precision(torch.tensor([[15.]]), k, c, noise)
        noisy = conditional_depth_precision(torch.tensor([[2.]]), k, c, 2*noise)
        self.assertTrue((deep['conditional_sigma_lower_bound_m'] > shallow['conditional_sigma_lower_bound_m']).all())
        torch.testing.assert_close(noisy['conditional_sigma_lower_bound_m'], 2*shallow['conditional_sigma_lower_bound_m'])
        self.assertFalse(deep['operationally_calibrated'])


if __name__ == '__main__':
    unittest.main()
