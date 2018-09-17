"""
    .. autoclass:: LR()
"""

import torchbearer
from torchbearer import metrics


def _get_lr(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        lrs.append(group['lr'])

    if len(lrs) == 1:
        return lrs[0]
    return lrs


@metrics.default_for_key('lr')
@metrics.to_dict
class LR(metrics.AdvancedMetric):
    """Returns the learning rate/s from the optimizer group/s. Use this to log the current LR when using decay.
    Default for key 'lr'
    """

    def __init__(self):
        super().__init__('lr')

    def process_train(self, *args):
        state = args[0]
        return _get_lr(state[torchbearer.OPTIMIZER])

    def process_final_train(self, *args):
        state = args[0]
        return _get_lr(state[torchbearer.OPTIMIZER])
