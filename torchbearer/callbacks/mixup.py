import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

import torchbearer
from torchbearer import cite
from torchbearer.callbacks import Callback
from torchbearer.metrics import CategoricalAccuracy, AdvancedMetric, running_mean, mean, super


mixup= """
@inproceedings{zhang2018mixup,
  title={mixup: Beyond Empirical Risk Minimization},
  author={Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
"""


@running_mean
@mean
class MixupAcc(AdvancedMetric):
    def __init__(self):
        super(MixupAcc, self).__init__('mixup_acc')
        self.cat_acc = CategoricalAccuracy().root

    def process_train(self, *args):
        super(MixupAcc, self).process_train(*args)
        state = args[0]
        
        target1, target2 = state[torchbearer.Y_TRUE]
        _state = args[0].copy()
        _state[torchbearer.Y_TRUE] = target1
        acc1 = self.cat_acc.process(_state)

        _state = args[0].copy()
        _state[torchbearer.Y_TRUE] = target2
        acc2 = self.cat_acc.process(_state)

        return acc1 * state[torchbearer.MIXUP_LAMBDA] + acc2 * (1-state[torchbearer.MIXUP_LAMBDA])

    def process_validate(self, *args):
        super(MixupAcc, self).process_validate(*args)

        return self.cat_acc.process(*args)

    def reset(self, state):
        self.cat_acc.reset(state)


@cite(mixup)
class Mixup(Callback):
    """Perform mixup on the model inputs. Requires use of :meth:`MixupInputs.loss`, otherwise lambdas can be found in
    state under :attr:`.MIXUP_LAMBDA`. Model targets will be a tuple containing the original target and permuted target.

    .. note::

        The accuracy metric for mixup is different on training to deal with the different targets,
    but for validation it is exactly the categorical accuracy, despite being called "val_mixup_acc"

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import Mixup

        # Example Trial which does Mixup regularisation
        >>> mixup = Mixup(0.9)
        >>> trial = Trial(None, criterion=Mixup.loss, callbacks=[mixup], metrics=['acc'])

    Args:
        alpha (float): The alpha value to use in the beta distribution.
    """
    RANDOM = -10.0

    def __init__(self, alpha=1.0, lam=RANDOM):
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.lam = lam
        self.distrib = Beta(self.alpha, self.alpha)

    @staticmethod
    def mixup_loss(state):
        """The standard cross entropy loss formulated for mixup (weighted combination of `F.cross_entropy`).

        Args:
            state: The current :class:`Trial` state.
        """
        input, target = state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE]

        if state[torchbearer.DATA] is torchbearer.TRAIN_DATA:
            y1, y2 = target
            return F.cross_entropy(input, y1) * state[torchbearer.MIXUP_LAMBDA] + F.cross_entropy(input, y2) * (1-state[torchbearer.MIXUP_LAMBDA])
        else:
            return F.cross_entropy(input, target)

    def on_sample(self, state):
        if self.lam is Mixup.RANDOM:
            if self.alpha > 0:
                lam = self.distrib.sample()
            else:
                lam = 1.0
        else:
            lam = self.lam

        state[torchbearer.MIXUP_LAMBDA] = lam

        state[torchbearer.MIXUP_PERMUTATION] = torch.randperm(state[torchbearer.X].size(0))
        state[torchbearer.X] = state[torchbearer.X] * state[torchbearer.MIXUP_LAMBDA] + state[torchbearer.X][state[torchbearer.MIXUP_PERMUTATION], :] * (1-state[torchbearer.MIXUP_LAMBDA])
        state[torchbearer.Y_TRUE] = (state[torchbearer.Y_TRUE], state[torchbearer.Y_TRUE][state[torchbearer.MIXUP_PERMUTATION]])


from torchbearer.metrics import default as d
d.__loss_map__[Mixup.mixup_loss.__name__] = MixupAcc
