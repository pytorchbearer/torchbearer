import torchbearer
from torchbearer import Callback
import torch
import torch.nn.functional as F
from torch.distributions import Beta

from torchbearer.bases import cite

me = """
@inproceedings{summers2019improved,
  title={Improved mixed-example data augmentation},
  author={Summers, Cecilia and Dinneen, Michael J},
  booktitle={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1262--1270},
  year={2019},
  organization={IEEE}
}
"""


class MixedExampleBase(Callback):
    def __init__(self, alpha=1):
        self.dist = Beta(torch.tensor([float(alpha)]), torch.tensor([float(alpha)]))

    def _get_lam(self, device):
        return self.dist.sample().to(device)

    def mix(self, batch, target, permutation):
        raise NotImplementedError

    def on_sample(self, state):
        permutation = torch.randperm(state[torchbearer.X].size(0))

        batch, lam = self.mix(state[torchbearer.INPUT], state[torchbearer.TARGET], permutation)

        state[torchbearer.INPUT] = batch
        state[torchbearer.MIXUP_LAMBDA] = lam
        state[torchbearer.MIXUP_PERMUTATION] = permutation
        state[torchbearer.Y_TRUE] = (
            state[torchbearer.Y_TRUE],
            state[torchbearer.Y_TRUE][state[torchbearer.MIXUP_PERMUTATION]]
        )


class VerticalConcat(MixedExampleBase):
    def mix(self, batch, target, permutation):
        lam = self._get_lam(batch.device)
        H = batch.size(2)
        dim = torch.round(H * lam)
        return torch.cat((
            batch[:, :, :dim],
            batch[permutation][:, :, dim:],
        ), dim=2), lam


class HorizontalConcat(MixedExampleBase):
    def mix(self, batch, target, permutation):
        lam = self._get_lam(batch.device)
        W = batch.size(3)
        dim = torch.round(W * lam)
        return torch.cat((
            batch[:, :, :, dim],
            batch[permutation][:, :, :, dim:],
        ), dim=3), lam


class MixedConcat(MixedExampleBase):
    def mix(self, batch, target, permutation):
        lam_h = self._get_lam(batch.device)
        lam_w = self._get_lam(batch.device)
        H, W = batch.size(2), batch.size(3)

        dim_h = torch.round(H * lam_h)
        dim_w = torch.round(W * lam_w)

        batch_p = batch[permutation]

        batch_1 = torch.cat((
            batch[:, :, :dim_h],
            batch_p[:, :, dim_h:],
        ), dim=2)

        batch_2 = torch.cat((
            batch_p[:, :, :dim_h],
            batch[:, :, dim_h:],
        ), dim=2)

        batch = torch.cat((
            batch_1[:, :, :, dim_w],
            batch_2[:, :, :, dim_w:],
        ), dim=3)

        lam = lam_h * lam_w + (1. - lam_h) * (1. - lam_w)

        return batch, lam


class RandomRows(MixedExampleBase):
    def __init__(self):
        super(MixedExampleBase, self).__init__()

    def mix(self, batch, target, permutation):
        H = batch.size(2)
        mask = (torch.rand(1, 1, H, 1) < torch.rand(1)).float()
        batch = mask * batch + (1 - mask) * batch[permutation]
        lam = mask.mean()
        return batch, lam


class RandomCols(MixedExampleBase):
    def __init__(self):
        super(MixedExampleBase, self).__init__()

    def mix(self, batch, target, permutation):
        W = batch.size(3)
        mask = (torch.rand(1, 1, 1, W) < torch.rand(1)).float()
        batch = mask * batch + (1 - mask) * batch[permutation]
        lam = mask.mean()
        return batch, lam


class RandomPixels(MixedExampleBase):
    def __init__(self):
        super(MixedExampleBase, self).__init__()

    def mix(self, batch, target, permutation):
        H, W = batch.size(2), batch.size(3)
        mask = (torch.rand(1, 1, H, W) < torch.rand(1)).float()
        batch = mask * batch + (1 - mask) * batch[permutation]
        lam = mask.mean()
        return batch, lam


class RandomElements(MixedExampleBase):
    def __init__(self):
        super(MixedExampleBase, self).__init__()

    def mix(self, batch, target, permutation):
        C, H, W = batch.size(1), batch.size(2), batch.size(3)
        mask = (torch.rand(1, C, H, W) < torch.rand(1)).float()
        batch = mask * batch + (1 - mask) * batch[permutation]
        lam = mask.mean()
        return batch, lam
