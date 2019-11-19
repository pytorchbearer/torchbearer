import torchbearer
from torchbearer import Callback
import torch
from torch.distributions import Beta

from torchbearer.bases import cite

cutout = """
@article{devries2017improved,
  title={Improved regularization of convolutional neural networks with Cutout},
  author={DeVries, Terrance and Taylor, Graham W},
  journal={arXiv preprint arXiv:1708.04552},
  year={2017}
}
"""

random_erase = """
@article{zhong2017random,
  title={Random erasing data augmentation},
  author={Zhong, Zhun and Zheng, Liang and Kang, Guoliang and Li, Shaozi and Yang, Yi},
  journal={arXiv preprint arXiv:1708.04896},
  year={2017}
}
"""


cutmix = """
@article{yun2019cutmix,
  title={Cutmix: Regularization strategy to train strong classifiers with localizable features},
  author={Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsuk and Yoo, Youngjoon},
  journal={arXiv preprint arXiv:1905.04899},
  year={2019}
}
"""


@cite(cutout)
class Cutout(Callback):
    """ Cutout callback which randomly masks out patches of image data. Implementation a modified version of the code
    found `here <https://github.com/uoguelph-mlrg/Cutout>`_.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import Cutout

        # Example Trial which does Cutout regularisation
        >>> cutout = Cutout(1, 10)
        >>> trial = Trial(None, callbacks=[cutout], metrics=['acc'])

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        constant (float): Constant value for each square patch

    State Requirements:
        - :attr:`torchbearer.state.X`: State should have the current data stored
    """
    def __init__(self, n_holes, length, constant=0.):
        super(Cutout, self).__init__()
        self.constant = constant
        self.cutter = BatchCutout(n_holes, length, length)

    def on_sample(self, state):
        super(Cutout, self).on_sample(state)
        mask = self.cutter(state[torchbearer.X])
        erase_locations = mask == 0
        constant = torch.ones_like(state[torchbearer.X]) * self.constant
        state[torchbearer.X][erase_locations] = constant[erase_locations]


@cite(random_erase)
class RandomErase(Callback):
    """ Random erase callback which replaces random patches of image data with random noise.
    Implementation a modified version of the cutout code found
    `here <https://github.com/uoguelph-mlrg/Cutout>`_.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import RandomErase

        # Example Trial which does Cutout regularisation
        >>> erase = RandomErase(1, 10)
        >>> trial = Trial(None, callbacks=[erase], metrics=['acc'])

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.

    State Requirements:
        - :attr:`torchbearer.state.X`: State should have the current data stored
    """
    def __init__(self, n_holes, length):
        super(RandomErase, self).__init__()
        self.cutter = BatchCutout(n_holes, length, length)

    def on_sample(self, state):
        super(RandomErase, self).on_sample(state)
        mask = self.cutter(state[torchbearer.X])
        erase_locations = mask == 0
        random = torch.rand_like(state[torchbearer.X])
        state[torchbearer.X][erase_locations] = random[erase_locations]


@cite(cutmix)
class CutMix(Callback):
    """ Cutmix callback which replaces a random patch of image data with the corresponding patch from another image.
    This callback also converts labels to one hot before combining them according to the lambda parameters, sampled from
    a beta distribution as is done in the paper.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import CutMix

        # Example Trial which does CutMix regularisation
        >>> cutmix = CutMix(1, classes=10)
        >>> trial = Trial(None, callbacks=[cutmix], metrics=['acc'])

    Args:
        alpha (float): The alpha value for the beta distribution.
        classes (int): The number of classes for conversion to one hot.

    State Requirements:
        - :attr:`torchbearer.state.X`: State should have the current data stored
        - :attr:`torchbearer.state.Y_TRUE`: State should have the current data stored
    """
    def __init__(self, alpha, classes=-1, mixup_loss=False):
        super(CutMix, self).__init__()
        self.classes = classes
        self.dist = Beta(torch.tensor([float(alpha)]), torch.tensor([float(alpha)]))
        self.mixup_loss = mixup_loss

    def _to_one_hot(self, target):
        if target.dim() == 1:
            target = target.unsqueeze(1)
            one_hot = torch.zeros_like(target).repeat(1, self.classes)
            one_hot.scatter_(1, target, 1)
            return one_hot
        return target

    def on_sample(self, state):
        super(CutMix, self).on_sample(state)

        lam = self.dist.sample().to(state[torchbearer.DEVICE])
        length = (1 - lam).sqrt()
        cutter = BatchCutout(1, (length * state[torchbearer.X].size(-1)).round().item(), (length * state[torchbearer.X].size(-2)).round().item())
        mask = cutter(state[torchbearer.X])
        erase_locations = mask == 0

        permutation = torch.randperm(state[torchbearer.X].size(0))
        if self.mixup_loss:
            state[torchbearer.MIXUP_PERMUTATION] = permutation
            state[torchbearer.MIXUP_LAMBDA] = lam

        state[torchbearer.X][erase_locations] = state[torchbearer.X][permutation][erase_locations]

        if self.mixup_loss:
            state[torchbearer.TARGET] = (state[torchbearer.TARGET], state[torchbearer.TARGET][state[torchbearer.MIXUP_PERMUTATION]])
        else:
            target = self._to_one_hot(state[torchbearer.TARGET]).float()
            state[torchbearer.TARGET] = lam * target + (1 - lam) * target[permutation]


    def on_sample_validation(self, state):
        super(CutMix, self).on_sample_validation(state)
        if not self.mixup_loss:
            state[torchbearer.TARGET] = self._to_one_hot(state[torchbearer.TARGET]).float()


class BatchCutout(object):
    """Randomly mask out one or more patches from a batch of images.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        width (int): The width (in pixels) of each square patch.
        height (int): The height (in pixels) of each square patch.
    """
    def __init__(self, n_holes, width, height):
        self.n_holes = n_holes
        self.width = width
        self.height = height

    def __call__(self, img):
        """

        Args:
            img (Tensor): Tensor image of size (B, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        b = img.size(0)
        c = img.size(1)
        h = img.size(-2)
        w = img.size(-1)

        mask = torch.ones((b, h, w), device=img.device)

        for n in range(self.n_holes):
            y = torch.randint(h, (b,)).long()
            x = torch.randint(w, (b,)).long()

            y1 = (y - self.height // 2).clamp(0, h).int()
            y2 = (y + self.height // 2).clamp(0, h).int()
            x1 = (x - self.width // 2).clamp(0, w).int()
            x2 = (x + self.width // 2).clamp(0, w).int()

            for batch in range(b):
                mask[batch, y1[batch]: y2[batch], x1[batch]: x2[batch]] = 0

        mask = mask.unsqueeze(1).repeat(1, c, 1, 1)

        return mask
