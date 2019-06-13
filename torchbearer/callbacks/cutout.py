import torchbearer
from torchbearer import Callback
import torch
import numpy as np


class CutOut(Callback):
    """ Cutout callback which randomly masks out patches of image data. Implementation a modified version of the code
    found `here <https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py>`_.

    Example::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import CutOut

        # Example Trial which does cutout regularisation
        >>> cutout = CutOut
        >>> trial = Trial(None, callbacks=[cutout], metrics=['acc'])

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, seed=None):
        super(CutOut, self).__init__()
        self.cutter = BatchCutout(n_holes, length, seed)

    def on_sample(self, state):
        super().on_sample(state)
        state[torchbearer.X] = self.cutter(state[torchbearer.X])


class BatchCutout(object):
    """Randomly mask out one or more patches from a batch of images.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, seed):
        self.n_holes = n_holes
        self.length = length
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
                mask[batch, y1[batch]: y2[batch], x1[batch]: x2[batch]] = 0.

        mask = torch.from_numpy(mask).unsqueeze(1).repeat(1, c, 1, 1)
        img = img * mask.to(img.device)

        return img
