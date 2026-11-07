import unittest
import torch
from torch.utils.data import Dataset
from scripts.run_nearshore_hybrid import ShallowAugDataset


class TinyBase(Dataset):
    def __len__(self):return 1
    def __getitem__(self,index):
        return torch.zeros(16,2,2),torch.tensor([[50.,50.],[50.,10.]]),torch.ones(2,2),torch.ones(3,2,2)*.03


class ShallowAugmentationTests(unittest.TestCase):
    def test_flips_do_not_move_supervision_to_deep_neighbour(self):
        torch.manual_seed(42)
        data=ShallowAugDataset(TinyBase(),augment=True)
        for _ in range(50):
            _,depth,mask,_=data[0]
            self.assertEqual(float(depth[1,1]),10.)
            self.assertEqual(float(mask[1,1]),1.)


if __name__=='__main__': unittest.main()
