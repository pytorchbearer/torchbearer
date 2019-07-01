import torch
import torch.nn.init as init
import torch.nn.functional as F

import torchbearer
from torchbearer import cite
from torchbearer.callbacks import Callback

# mixup= """
# @inproceedings{zhang2018mixup,
# title={mixup: Beyond Empirical Risk Minimization},
# author={Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz},
# booktitle={International Conference on Learning Representations},
# year={2018}
# }
# """

# @torchbearer.cite(mixup)
class MixupInputs(Callback):

    def __init__(self, alpha=1.0):
        super(MixupInputs, self).__init__()
        self.alpha = alpha

    @staticmethod
    def loss(state):
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
