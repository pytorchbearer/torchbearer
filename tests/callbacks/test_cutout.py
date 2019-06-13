from unittest import TestCase

import torch

import torchbearer
from torchbearer.callbacks.cutout import CutOut


class TestCutOut(TestCase):
    def test_cutout(self):
        random_image = torch.rand(2, 3, 100, 100)
        co = CutOut(1, 10, seed=7)
        state = {torchbearer.X: random_image}
        co.on_sample(state)

        x = [25, 67]
        y = [47, 68]

        known_cut = random_image.numpy()
        known_cut[0, :, y[0]-10//2:y[0]+10//2, x[0]-10//2:x[0]+10//2] = 0
        known_cut[1, :, y[1]-10//2:y[1]+10//2, x[1]-10//2:x[1]+10//2] = 0
        known_cut = torch.from_numpy(known_cut)

        diff = (known_cut-state[torchbearer.X] > 1e-4).any()
        self.assertTrue(diff.item() == 0)
