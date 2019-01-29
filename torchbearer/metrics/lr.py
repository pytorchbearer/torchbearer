"""
    .. autoclass:: LR()
"""
import torchbearer
from .metrics import AdvancedMetric
from .decorators import default_for_key, to_dict

old_super = super


def super(_, obj):
    return old_super(obj.__class__, obj)


def _get_lr(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        lrs.append(group['lr'])

    if len(lrs) == 1:
        return lrs[0]
    return lrs


@default_for_key('lr')
@to_dict
class LR(AdvancedMetric):
    """Returns the learning rate/s from the optimizer group/s. Use this to log the current LR when using decay.
    Default for key 'lr'

    State Requirements:
        - :attr:`torchbearer.state.OPTIMIZER`: The optimizer in state will be used to retrieve the learning rate.
    """

    def __init__(self):
        super(LR, self).__init__('lr')

    def process_train(self, *args):
        state = args[0]
        return _get_lr(state[torchbearer.OPTIMIZER])

    def process_final_train(self, *args):
        state = args[0]
        return _get_lr(state[torchbearer.OPTIMIZER])
