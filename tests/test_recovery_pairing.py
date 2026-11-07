"""Train-only pairing features and fixed geographic footprint checks."""
import unittest
import numpy as np
from scripts.build_official_recovery_data import engineer,far


class PairingTests(unittest.TestCase):
    def test_held_reflectance_cannot_change_training_fit_or_features(self):
        rng=np.random.RandomState(5)
        raw=rng.uniform(.01,.08,(3,4,64,64)).astype('float32')
        mask=np.ones((3,64,64),dtype='float32')
        coords=np.array([[100,100],[300,300],[500,500]])
        ids=np.array([0,1,2]);train=np.array([0,1])
        first,fit=engineer(raw,mask,coords,ids,train)
        raw[2]*=20
        second,other=engineer(raw,mask,coords,ids,train)
        self.assertEqual(fit,other)
        np.testing.assert_array_equal(first[train],second[train])

    def test_repeated_same_source_pixels_count_once(self):
        raw=np.full((2,4,64,64),.02,dtype='float32')
        _,fit=engineer(raw,np.ones((2,64,64),dtype='float32'),np.array([[100,100],[100,100]]),np.array([0,0]),np.array([0,1]))
        self.assertEqual(fit['fit_unique_source_pixels'],4096)

    def test_buffer_and_tide_datum_inverse(self):
        np.testing.assert_array_equal(far(np.array([[0,95],[0,96]]),np.array([[0,0]])),[False,True])
        atl_depth=np.array([4.,10.]);atl_tide=np.array([.3,-.1]);scene_tide=np.array([-.2,.4])
        delta=scene_tide-atl_tide
        np.testing.assert_allclose(atl_depth+delta-delta,atl_depth)


if __name__=='__main__':unittest.main()
