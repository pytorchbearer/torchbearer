from unittest import TestCase

import torch

import torchbearer
from torchbearer.callbacks.cutout import Cutout, RandomErase, CutMix


class TestCutOut(TestCase):
    def test_cutout(self):
        random_image = torch.rand(2, 3, 100, 100)
        co = Cutout(1, 10, seed=7)
        state = {torchbearer.X: random_image}
        co.on_sample(state)
        reg_img = state[torchbearer.X].view(-1)

        x = [21, 86]
        y = [15, 92]

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

        x = [21, 86]
        y = [15, 92]

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

        x = [21, 86]
        y = [15, 92]

        known_cut = random_image.clone().numpy()
        known_cut[0, :, y[0]-10//2:y[0]+10//2, x[0]-10//2:x[0]+10//2] = 0
        known_cut[1, :, y[1]-10//2:y[1]+10//2, x[1]-10//2:x[1]+10//2] = 0
        known_cut = torch.from_numpy(known_cut)

        known_cut = known_cut.view(-1)
        masked_pix = known_cut == 0

        diff = (torch.abs(known_cut[masked_pix]-reg_img[masked_pix]) > 1e-4).any()
        self.assertTrue(diff.item() > 0)

    def test_cutmix(self):
        random_image = torch.rand(5, 3, 100, 100)
        state = {torchbearer.X: random_image, torchbearer.Y_TRUE: torch.randint(10, (5,)).long(), torchbearer.DEVICE: 'cpu'}
        co = CutMix(0.25, classes=10, seed=7)
        co.on_sample(state)
        reg_img = state[torchbearer.X].view(-1)

        x = [72, 83, 18, 96, 40]
        y = [8, 17, 62, 30, 66]
        perm = [0, 4, 3, 2, 1]
        sz = 3

        rnd = random_image.clone().numpy()
        known_cut = random_image.clone().numpy()
        known_cut[0, :, y[0]-sz//2:y[0]+sz//2, x[0]-sz//2:x[0]+sz//2] = rnd[perm[0], :, y[0]-sz//2:y[0]+sz//2, x[0]-sz//2:x[0]+sz//2]
        known_cut[1, :, y[1]-sz//2:y[1]+sz//2, x[1]-sz//2:x[1]+sz//2] = rnd[perm[1], :, y[1]-sz//2:y[1]+sz//2, x[1]-sz//2:x[1]+sz//2]
        known_cut[2, :, y[2]-sz//2:y[2]+sz//2, x[2]-sz//2:x[2]+sz//2] = rnd[perm[2], :, y[2]-sz//2:y[2]+sz//2, x[2]-sz//2:x[2]+sz//2]
        known_cut[3, :, y[3]-sz//2:y[3]+sz//2, x[3]-sz//2:x[3]+sz//2] = rnd[perm[3], :, y[3]-sz//2:y[3]+sz//2, x[3]-sz//2:x[3]+sz//2]
        known_cut[4, :, y[4]-sz//2:y[4]+sz//2, x[4]-sz//2:x[4]+sz//2] = rnd[perm[4], :, y[4]-sz//2:y[4]+sz//2, x[4]-sz//2:x[4]+sz//2]
        known_cut = torch.from_numpy(known_cut)
        known_cut = known_cut.view(-1)

        diff = (torch.abs(known_cut-reg_img) > 1e-4).any()
        self.assertTrue(diff.item() == 0)

    def test_cutmix_targets(self):
        random_image = torch.rand(2, 3, 100, 100)
        co = CutMix(1.0, classes=4, seed=7)
        target = torch.tensor([
            [0., 1., 0., 0.],
            [0., 0., 0., 1.]
        ])
        state = {torchbearer.X: random_image, torchbearer.Y_TRUE: torch.tensor([1, 3]).long(), torchbearer.DEVICE: 'cpu'}
        co.on_sample(state)
        self.assertTrue(((state[torchbearer.TARGET] - target).abs() < 0.00001).all())
        state = {torchbearer.X: random_image, torchbearer.Y_TRUE: torch.tensor([1, 3]).long()}
        co.on_sample_validation(state)
        self.assertTrue(((state[torchbearer.TARGET] - target).abs() < 0.00001).all())
        state = {torchbearer.X: random_image, torchbearer.Y_TRUE: target.long()}
        co.on_sample_validation(state)
        self.assertTrue(((state[torchbearer.TARGET] - target).abs() < 0.00001).all())
