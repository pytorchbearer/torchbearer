from unittest import TestCase

import torch
import numpy as np

import torchbearer
from torchbearer.callbacks.cutout import Cutout, RandomErase


class TestCutOut(TestCase):
    def test_cutout(self):
        random_image = torch.rand(2, 3, 100, 100)
        co = Cutout(1, 10, seed=7)
        state = {torchbearer.X: random_image}
        co.on_sample(state)
        reg_img = state[torchbearer.X].view(-1)

        x = [25, 67]
        y = [47, 68]

        known_cut = random_image.clone().numpy()
        known_cut[0, :, y[0]-10//2:y[0]+10//2, x[0]-10//2:x[0]+10//2] = 0
        known_cut[1, :, y[1]-10//2:y[1]+10//2, x[1]-10//2:x[1]+10//2] = 0
        known_cut = torch.from_numpy(known_cut)
        known_cut = known_cut.view(-1)

        diff = (torch.abs(known_cut-reg_img) > 1e-4).any()
        self.assertTrue(diff.item() == 0)

    def test_cutout_constant(self):
        random_image = torch.rand(2, 3, 100, 100)
        co = Cutout(1, 10, constant=0.5, seed=7)
        state = {torchbearer.X: random_image}
        co.on_sample(state)
        reg_img = state[torchbearer.X].view(-1)

        x = [25, 67]
        y = [47, 68]

        known_cut = random_image.clone().numpy()
        known_cut[0, :, y[0]-10//2:y[0]+10//2, x[0]-10//2:x[0]+10//2] = 0.5
        known_cut[1, :, y[1]-10//2:y[1]+10//2, x[1]-10//2:x[1]+10//2] = 0.5
        known_cut = torch.from_numpy(known_cut)
        known_cut = known_cut.view(-1)

        diff = (torch.abs(known_cut-reg_img) > 1e-4).any()
        self.assertTrue(diff.item() == 0)

    # TODO: Find a better test for this
    def test_random_erase(self):
        random_image = torch.rand(2, 3, 100, 100)
        co = RandomErase(1, 10, seed=7)
        state = {torchbearer.X: random_image}
        co.on_sample(state)
        reg_img = state[torchbearer.X].view(-1)

        x = [25, 67]
        y = [47, 68]

        known_cut = random_image.clone().numpy()
        known_cut[0, :, y[0]-10//2:y[0]+10//2, x[0]-10//2:x[0]+10//2] = 0
        known_cut[1, :, y[1]-10//2:y[1]+10//2, x[1]-10//2:x[1]+10//2] = 0
        known_cut = torch.from_numpy(known_cut)

        known_cut = known_cut.view(-1)
        masked_pix = known_cut == 0

        diff = (torch.abs(known_cut[masked_pix]-reg_img[masked_pix]) > 1e-4).any()
        self.assertTrue(diff.item() > 0)
