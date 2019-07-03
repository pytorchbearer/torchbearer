import torch

import torchbearer
from torchbearer import cite
from torchbearer.callbacks import Callback


bibtex = """
@article{szegedy2015rethinking,
  title={Rethinking the inception architecture for computer vision. arXiv 2015},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jonathon and Wojna, Zbigniew},
  journal={arXiv preprint arXiv:1512.00567},
  volume={1512},
  year={2015}
}
"""


@cite(bibtex)
class LabelSmoothingRegularisation(Callback):
    """Perform Label Smoothing Regularisation (LSR) on the targets during training. This involves converting the target
    to a one-hot vector and smoothing according to the value epsilon.

    .. note::

        Requires a multi-label loss, such as nn.BCELoss

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import LabelSmoothingRegularisation

        # Example Trial which does label smoothing regularisation
        >>> smoothing = LabelSmoothingRegularisation()
        >>> trial = Trial(None, criterion=nn.BCELoss(), callbacks=[smoothing], metrics=['acc'])

    Args:
        epsilon (float): The epsilon parameter from the paper
        classes (int): The number of target classes, not required if the target is already one-hot encoded
    """
    def __init__(self, epsilon, classes=-1):
        self.epsilon = epsilon
        self.classes = classes

    def to_one_hot(self, state):
        target = state[torchbearer.TARGET]

        if target.dim() == 1:
            target = target.unsqueeze(1)
            one_hot = torch.zeros_like(target).repeat(1, self.classes)
            one_hot.scatter_(1, target, 1)
            target = one_hot
        return target

    def on_sample(self, state):
        target = self.to_one_hot(state)
        target = (1 - self.epsilon) * target.float() + (self.epsilon / target.size(1))
        state[torchbearer.TARGET] = target

    def on_sample_validation(self, state):
        target = self.to_one_hot(state)
        state[torchbearer.TARGET] = target.float()
