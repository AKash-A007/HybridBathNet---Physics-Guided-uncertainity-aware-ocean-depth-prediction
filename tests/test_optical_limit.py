import unittest
import math
import torch
from src.optical_limit import bottom_detection_limit,conservative_bottom_detection_limit,optical_support_flags


class OpticalLimitTests(unittest.TestCase):
    def test_attenuated_signal_equals_threshold(self):
        kd=torch.tensor(.1); ku=torch.tensor(.2); contrast=torch.tensor(.05); noise=torch.tensor(.001)
        d=bottom_detection_limit(kd,ku,contrast,noise,detection_snr=3)
        signal=contrast*torch.exp(-(kd+ku)*d['z_limit_m'])
        self.assertAlmostEqual(float(signal),.003,places=7)
        self.assertAlmostEqual(float(d['z_limit_m']),math.log(.05/.003)/.3,places=5)

    def test_invalid_and_undetectable(self):
        d=bottom_detection_limit(torch.tensor([.1,0.,.1]),torch.tensor([.1,.1,.1]),
            torch.tensor([.001,.05,.05]),torch.tensor([.001,.001,0.]))
        self.assertEqual(float(d['z_limit_m'][0]),0.)
        self.assertTrue(torch.isnan(d['z_limit_m'][1:]).all())
        with self.assertRaises(ValueError): optical_support_flags(torch.ones(3),d)
        with self.assertRaises(ValueError): optical_support_flags(torch.ones(3),d,calibrated=True)

    def test_deeper_support_in_clearer_water_and_lower_noise(self):
        t=lambda v:torch.tensor(v)
        d1=bottom_detection_limit(t(.1),t(.1),t(.05),t(.001))
        d2=bottom_detection_limit(t(.2),t(.2),t(.05),t(.001))
        d3=bottom_detection_limit(t(.1),t(.1),t(.05),t(.002))
        self.assertGreater(float(d1['z_limit_m']),float(d2['z_limit_m']))
        self.assertGreater(float(d1['z_limit_m']),float(d3['z_limit_m']))

    def test_conservative_parameter_bounds_can_only_shrink_limit(self):
        t=lambda v:torch.tensor(v)
        point=bottom_detection_limit(t(.1),t(.1),t(.05),t(.001))
        lower=conservative_bottom_detection_limit(t(.12),t(.13),t(.04),t(.002))
        self.assertLess(float(lower['z_limit_m']),float(point['z_limit_m']))
        self.assertFalse(lower['operationally_calibrated'])


if __name__=='__main__': unittest.main()
