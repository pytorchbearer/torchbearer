import torchbearer
from torchbearer import cite
from torchbearer.callbacks import Callback
import torch.nn.init as init

_mixup= """
@inproceedings{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz},
booktitle={International Conference on Learning Representations},
year={2018}
}
"""

MIXUP_LAMBDA = torchbearer.state_key('mixup_lambda')


# @torchbearer.cite(_mixup)
class MixupInputs(Callback):

    def __init__(self, alpha=1.0):
        super(MixupInput, self).__init__()

        self.alpha = alpha

    @staticmethod
    def loss(state):
        input, target = state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE]
        y1, y2, lam = target
        return F.cross_entropy(input, y1) * state[torchbearer.MIXUP_LAMBDA] + F.cross_entropy(input, y2) * (1-state[torchbearer.MIXUP_LAMBDA])

    def on_sample(self, state, lam=1.0):
        import numpy as np
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)

        permutation = torch.randperm(state[torchbearer.BATCH])
        state[torchbearer.BATCH] = state[torchbearer.BATCH] * lam + state[torchbearer.BATCH][permutation, :] * (1-lam)
        state[torchbearer.Y_TRUE] = (state[torchbearer.Y_TRUE], state[torchbearer.Y_TRUE][permutation])
        state[torchbearer.MIXUP_LAMBDA] = lam
        print("doua fire doua paie")
