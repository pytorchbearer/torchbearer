import torch
import torch.nn.functional as F

import torchbearer
from torchbearer import cite
from torchbearer.callbacks import Callback


mixup= """
@inproceedings{zhang2018mixup,
  title={mixup: Beyond Empirical Risk Minimization},
  author={Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
"""


@cite(mixup)
class Mixup(Callback):
    """Perform mixup on the model inputs. Requires use of :meth:`MixupInputs.loss`, otherwise lambdas can be found in
    state under :attr:`.MIXUP_LAMBDA`. Model targets will be a tuple containing the original target and permuted target.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import Mixup

        # Example Trial which does Mixup regularisation
        >>> mixup = Mixup(0.9)
        >>> trial = Trial(None, criterion=Mixup.loss, callbacks=[mixup], metrics=['acc'])

    Args:
        alpha (float): The alpha value to use in the beta distribution.
    """
    def __init__(self, alpha=1.0):
        super(Mixup, self).__init__()
        self.alpha = alpha

    @staticmethod
    def loss(state):
        """The standard cross entropy loss formulated for mixup (weighted combination of `F.cross_entropy`).

        Args:
            state: The current :class:`Trial` state.
        """
        input, target = state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE]
        y1, y2 = target
        return F.cross_entropy(input, y1) * state[torchbearer.MIXUP_LAMBDA] + F.cross_entropy(input, y2) * (1-state[torchbearer.MIXUP_LAMBDA])

    def on_sample(self, state, lam=1.0):
        import numpy as np
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        state[torchbearer.MIXUP_LAMBDA] = lam

        state[torchbearer.MIXUP_PERMUTATION] = torch.randperm(state[torchbearer.X].size(0))
        state[torchbearer.X] = state[torchbearer.X] * state[torchbearer.MIXUP_LAMBDA] + state[torchbearer.X][state[torchbearer.MIXUP_PERMUTATION], :] * (1-state[torchbearer.MIXUP_LAMBDA])
        state[torchbearer.Y_TRUE] = (state[torchbearer.Y_TRUE], state[torchbearer.Y_TRUE][state[torchbearer.MIXUP_PERMUTATION]])
