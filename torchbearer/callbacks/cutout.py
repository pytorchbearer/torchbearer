import torchbearer
from torchbearer import Callback
import torch
import numpy as np
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


@cite(cutout)
class Cutout(Callback):
    """ Cutout callback which randomly masks out patches of image data. Implementation a modified version of the code
    found `here <https://github.com/uoguelph-mlrg/Cutout/blob/master/util/Cutout.py>`_.

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
        seed: Random seed

    State Requirements:
        - :attr:`torchbearer.state.X`: State should have the current data stored
    """
    def __init__(self, n_holes, length, constant=0., seed=None):
        super(Cutout, self).__init__()
        self.cutter = BatchCutout(n_holes, length, constant=constant, seed=seed)

    def on_sample(self, state):
        super(Cutout, self).on_sample(state)
        state[torchbearer.X] = self.cutter(state[torchbearer.X])


@cite(random_erase)
class RandomErase(Callback):
    """ Random erase callback which replaces random patches of image data with random noise.
    Implementation a modified version of the cutout code found
    `here <https://github.com/uoguelph-mlrg/Cutout/blob/master/util/Cutout.py>`_.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import RandomErase

        # Example Trial which does Cutout regularisation
        >>> erase = RandomErase(1, 10)
        >>> trial = Trial(None, callbacks=[erase], metrics=['acc'])

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        seed: Random seed

    State Requirements:
        - :attr:`torchbearer.state.X`: State should have the current data stored
    """
    def __init__(self, n_holes, length, seed=None):
        super(RandomErase, self).__init__()
        self.cutter = BatchCutout(n_holes, length, seed=seed, random_erase=True)

    def on_sample(self, state):
        super(RandomErase, self).on_sample(state)
        state[torchbearer.X] = self.cutter(state[torchbearer.X])


class BatchCutout(object):
    """Randomly mask out one or more patches from a batch of images.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        seed: Random seed
    """
    def __init__(self, n_holes, length, constant=0., random_erase=False, seed=None):
        self.n_holes = n_holes
        self.length = length
        self.random_erasing = random_erase
        self.constant = constant
        np.random.seed(seed)

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

        mask = np.ones((b, h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h, size=b)
            x = np.random.randint(w, size=b)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            for batch in range(b):
                mask[batch, y1[batch]: y2[batch], x1[batch]: x2[batch]] = 0

        mask = torch.from_numpy(mask).unsqueeze(1).repeat(1, c, 1, 1)

        erase_locations = mask == 0

        if self.random_erasing:
            random = torch.from_numpy(np.random.rand(*img.shape)).to(torch.float)
        else:
            random = torch.from_numpy(np.ones_like(img)).to(torch.float) * self.constant

        img[erase_locations] = random[erase_locations]

        return img
