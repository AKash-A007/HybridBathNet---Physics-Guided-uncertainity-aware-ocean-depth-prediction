"""Regression checks for failures found during the real-data rerun."""
import unittest

import numpy as np
import torch

from src.cnn_baseline import EMA, PatchDataset, SpectralAugDataset, compute_depth_bin_metrics
from src.hybridbathnet import CompositePhysicsNLLLoss, HybridBathNetModel, compute_kd_and_extinction
from src.features import stumpf_ratio


class StabilityTests(unittest.TestCase):
    def test_kd_matches_independent_reference(self):
        blue = np.array([[[[.05, .04], [.02, .03]]]], dtype=np.float32)
        green = np.full_like(blue, .025)
        red = np.full_like(blue, .01)
        reference = np.clip(.0166 + .156 * (blue/green)**(-1.14), .01, 5)
        reference = np.clip(reference + np.clip(.430*red/(green+1e-6), 0, 3), .01, 6)
        for mode in ('mean', 'center'):
            kd, selected, bound = compute_kd_and_extinction(
                torch.from_numpy(blue), torch.from_numpy(green), torch.from_numpy(red),
                aggregation=mode)
            expected = reference.mean((-2,-1)) if mode == 'mean' else reference[:,:,1,1]
            np.testing.assert_allclose(kd.numpy(), reference, rtol=1e-6)
            np.testing.assert_allclose(selected.numpy(), expected, rtol=1e-6)
            np.testing.assert_allclose(bound.numpy(), np.clip(np.log(100)/(2*expected+1e-6),1,50), rtol=1e-6)

    def test_invalid_pixels_do_not_change_masked_kd_mean(self):
        b = torch.tensor([[[[.05, .01], [.05, .05]]]])
        g = torch.full_like(b, .03)
        mask = torch.tensor([[[[1.,0.],[1.,1.]]]])
        _, selected, _ = compute_kd_and_extinction(b,g,mask=mask)
        expected = .0166 + .156 * (.05/.03)**(-1.14)
        self.assertAlmostEqual(selected.item(), expected, places=6)

    def test_stumpf_singularity_is_finite(self):
        self.assertEqual(stumpf_ratio(np.array([.05]), np.array([.001]))[0], 0)

    def test_spectral_jitter_does_not_change_physics_reflectance(self):
        x = np.full((2,16,8,8), .05, dtype=np.float32)
        mask = np.ones((2,8,8),dtype=np.float32)
        x[:,11] = mask
        base = PatchDataset(x, mask*10, mask, np.arange(2))
        augmented = SpectralAugDataset(base, noise_std=.2, use_rotations=True)
        _, _, _, raw = augmented[0]
        np.testing.assert_array_equal(raw.numpy(), x[0,:3])

    def test_cnn_control_ignores_physics_channels(self):
        torch.set_num_threads(2)
        model = HybridBathNetModel(in_channels=16,physics_enabled=False).eval()
        x = torch.zeros(1,16,16,16)
        changed = x.clone()
        changed[:,4:11] = 100
        with torch.no_grad():
            a, b = model(x)['depth'], model(changed)['depth']
        torch.testing.assert_close(a,b)

    def test_sparse_depth_bins_are_explicitly_unavailable(self):
        values = np.array([1.0, 2.0, 3.0])
        metrics = compute_depth_bin_metrics(values, values)
        self.assertEqual(metrics['n_0-5m'], 3)
        self.assertIsNone(metrics['rmse_0-5m'])

    def test_ema_tracks_batchnorm_buffers(self):
        model = torch.nn.BatchNorm2d(2)
        ema = EMA(model)
        model(torch.full((2, 2, 4, 4), 5.0))
        ema.update()
        self.assertTrue(torch.equal(ema.shadow['running_mean'], model.running_mean))
        self.assertTrue(torch.equal(ema.shadow['num_batches_tracked'], model.num_batches_tracked))

    def test_mask_stays_binary_after_normalization(self):
        x = np.ones((2, 16, 8, 8), dtype=np.float32)
        mask = np.ones((2, 8, 8), dtype=np.float32)
        mask[:, :, 0] = 0
        x[:, 11] = mask
        dataset = PatchDataset(x, np.ones_like(mask), mask, np.arange(2))
        np.testing.assert_array_equal(dataset[0][0][11].numpy(), mask[0])

    def test_variance_loss_is_finite_and_in_metres(self):
        criterion = CompositePhysicsNLLLoss(w_mse=0, w_phys=0, w_ext=0, beta_nll=0)
        pred = torch.tensor([2.0], requires_grad=True)
        loss, _ = criterion(pred, torch.zeros(1), torch.zeros(1), torch.ones(1, 1), y_std=3)
        self.assertAlmostEqual(loss.item(), 18.0, delta=1e-4)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        loss, _ = criterion(torch.tensor([100.], dtype=torch.float16),
                            torch.tensor([-10.], dtype=torch.float16),
                            torch.zeros(1), torch.ones(1, 1))
        self.assertTrue(torch.isfinite(loss))

    def test_reconstruction_forward(self):
        torch.set_num_threads(2)
        model = HybridBathNetModel(in_channels=16).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 16, 16, 16))
        self.assertEqual(out['r_hat'].shape, (1, 3, 16, 16))
        self.assertTrue(all(torch.isfinite(value).all() for value in out.values()))


if __name__ == '__main__':
    unittest.main()
