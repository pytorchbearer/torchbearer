import torchbearer
from torchbearer import Callback
import torch
import torch.nn.functional as F
from torch.distributions import Beta

from torchbearer.bases import cite

bc = """
@inproceedings{tokozume2018between,
  title={Between-class learning for image classification},
  author={Tokozume, Yuji and Ushiku, Yoshitaka and Harada, Tatsuya},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5486--5494},
  year={2018}
}
"""


@cite(bc)
class BCPlus(Callback):
    """BC+ callback which mixes images by treating them as waveforms. For standard BC, see :class:`.Mixup`.
    This callback can optionally convert labels to one hot before combining them according to the lambda parameters,
    sampled from a beta distribution, use alpha=1 to replicate the paper. Use with :meth:`BCPlus.bc_loss` or set
    `mixup_loss = True` and use :meth:`.Mixup.mixup_loss`.

    .. note::

       This callback first sets all images to have zero mean. Consider adding an offset (e.g. 0.5) back before
       visualising.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import BCPlus

        # Example Trial which does BCPlus regularisation
        >>> bcplus = BCPlus(classes=10)
        >>> trial = Trial(None, criterion=BCPlus.bc_loss, callbacks=[bcplus], metrics=['acc'])

    Args:
        mixup_loss (bool): If True, the lambda and targets will be stored for use with the mixup loss function.
        alpha (float): The alpha value for the beta distribution.
        classes (int): The number of classes for conversion to one hot.

    State Requirements:
        - :attr:`torchbearer.state.X`: State should have the current data stored and correctly normalised
        - :attr:`torchbearer.state.Y_TRUE`: State should have the current data stored
    """

    def __init__(self, mixup_loss=False, alpha=1, classes=-1):
        super(BCPlus, self).__init__()
        self.mixup_loss = mixup_loss
        self.classes = classes
        self.dist = Beta(torch.tensor([float(alpha)]), torch.tensor([float(alpha)]))

    @staticmethod
    def bc_loss(state):
        """The KL divergence between the outputs of the model and the ratio labels. Model ouputs should be un-normalised
        logits as this function performs a log_softmax.

        Args:
            state: The current :class:`Trial` state.
        """
        prediction, target = state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE]

        entropy = - (target[target.nonzero().split(1, dim=1)] * target[target.nonzero().split(1, dim=1)].log()).sum()
        cross = - (target * F.log_softmax(prediction, dim=1)).sum()

        return (cross - entropy) / prediction.size(0)

    def _to_one_hot(self, target):
        if target.dim() == 1:
            target = target.unsqueeze(1)
            one_hot = torch.zeros_like(target).repeat(1, self.classes)
            one_hot.scatter_(1, target, 1)
            return one_hot
        return target.float()

    def on_sample(self, state):
        super(BCPlus, self).on_sample(state)

        lam = self.dist.sample().to(state[torchbearer.DEVICE])

        permutation = torch.randperm(state[torchbearer.X].size(0))

        batch1 = state[torchbearer.X]
        batch1 = batch1 - batch1.view(batch1.size(0), -1).mean(1, keepdim=True).view(*tuple([batch1.size(0)] + [1] * (batch1.dim() - 1)))
        g1 = batch1.view(batch1.size(0), -1).std(1, keepdim=True).view(*tuple([batch1.size(0)] + [1] * (batch1.dim() - 1)))

        batch2 = batch1[permutation]
        g2 = g1[permutation]

        p = 1. / (1 + ((g1 / g2) * ((1 - lam) / lam)))

        state[torchbearer.X] = (batch1 * p + batch2 * (1 - p)) / (p.pow(2) + (1 - p).pow(2)).sqrt()

        if not self.mixup_loss:
            target = self._to_one_hot(state[torchbearer.TARGET]).float()
            state[torchbearer.Y_TRUE] = lam * target + (1 - lam) * target[permutation]
        else:
            state[torchbearer.MIXUP_LAMBDA] = lam
            state[torchbearer.MIXUP_PERMUTATION] = permutation
            state[torchbearer.Y_TRUE] = (state[torchbearer.Y_TRUE], state[torchbearer.Y_TRUE][state[torchbearer.MIXUP_PERMUTATION]])

    def on_sample_validation(self, state):
        super(BCPlus, self).on_sample_validation(state)
        if not self.mixup_loss:
            state[torchbearer.TARGET] = self._to_one_hot(state[torchbearer.TARGET]).float()
