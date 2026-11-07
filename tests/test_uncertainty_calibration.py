import unittest
import numpy as np
from src.uncertainty_calibration import ResidualCalibrator,group_weights,interval_score


class CalibrationTests(unittest.TestCase):
    def test_equal_weight_per_group(self):
        groups=np.array([0,0,0,1]);w=group_weights(groups)
        self.assertAlmostEqual(w[:3].sum(),.5);self.assertAlmostEqual(w[3],.5)

    def test_scale_analytic_mle(self):
        c=ResidualCalibrator('gaussian_scale').fit(np.array([2.,-2.]),np.zeros(2),np.ones(2),np.arange(2))
        self.assertAlmostEqual(c.scale,2.)
        with self.assertRaises(ValueError):c.interval(np.zeros(1),np.zeros(1))

    def test_hcp_infinity_atom_and_single_point_reduction(self):
        y=np.arange(1.,16.);c=ResidualCalibrator('hierarchical_conformal').fit(y,np.zeros(15),np.ones(15),np.arange(15))
        lo,hi=c.interval(np.zeros(1),np.ones(1),.9)
        self.assertEqual(float(hi[0]),15.)
        lo,hi=c.interval(np.zeros(1),np.ones(1),.95)
        self.assertTrue(np.isneginf(lo[0]) and np.isposinf(hi[0]))

    def test_empirical_residual_sign_and_crps(self):
        c=ResidualCalibrator('empirical_group_crude').fit(np.array([2.,2.]),np.zeros(2),np.ones(2),np.arange(2))
        lo,hi=c.interval(np.array([10.]),np.array([3.]),.9)
        np.testing.assert_array_equal(lo,[16.]);np.testing.assert_array_equal(hi,[16.])
        np.testing.assert_allclose(c.crps(np.array([15.]),np.array([10.]),np.array([3.])),[1.])

    def test_interval_score_penalizes_width_and_misses(self):
        score=interval_score(np.array([0.,3.]),np.array([-1.,-1.]),np.array([1.,1.]),.9)
        np.testing.assert_allclose(score,[2.,42.])

    def test_saved_calibrator_round_trip_without_targets(self):
        y=np.array([-1.,2.,3.,8.]);mean=np.zeros(4);sigma=np.ones(4);groups=np.array([0,0,1,2])
        for method in ('raw','gaussian_scale','empirical_group_crude','hierarchical_conformal','group_max_conformal'):
            c=ResidualCalibrator(method).fit(y,mean,sigma,groups)
            loaded=ResidualCalibrator.from_state(c.state())
            for first,second in zip(c.interval(mean,sigma,.68),loaded.interval(mean,sigma,.68)):
                np.testing.assert_array_equal(first,second)
            if c.crps(y,mean,sigma) is not None:
                np.testing.assert_allclose(c.crps(y,mean,sigma),loaded.crps(y,mean,sigma))


if __name__=='__main__':unittest.main()
