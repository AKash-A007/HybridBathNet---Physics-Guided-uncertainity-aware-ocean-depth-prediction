import unittest
import numpy as np
from scripts.gebco_oof_common import spatial_folds
import torch
from src.oof_uncertainty import SpatialVarianceCorrection,variance_nll


class OofTests(unittest.TestCase):
    def test_initial_correction_preserves_variance_and_learns(self):
        torch.set_num_threads(2)
        model=SpatialVarianceCorrection();base=torch.tensor([1.,4.],requires_grad=True)
        logvar=model(torch.randn(2,9,32,32),torch.randn(2,4),base)
        torch.testing.assert_close(logvar.exp(),base)
        loss=variance_nll(logvar,torch.tensor([3.,3.]));loss.backward()
        self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0)
        self.assertIsNone(base.grad)

    def test_spatial_folds_cover_points_once_and_exclude_support(self):
        centres=np.array([[r*300,c*300] for r in range(18) for c in range(18)])
        support=np.arange(len(centres)*4).reshape(-1,4)
        train=np.arange(len(centres));folds=spatial_folds(train,centres,support)
        np.testing.assert_array_equal(np.sort(np.concatenate([held for _,held in folds])),train)
        for fit,held in folds:
            self.assertFalse(set(fit)&set(held))
            self.assertFalse(set(support[fit].ravel())&set(support[held].ravel()))


if __name__=='__main__':unittest.main()
