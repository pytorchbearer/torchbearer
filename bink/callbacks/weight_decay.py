from bink.callbacks import Callback

import torch


class WeightDecay(Callback):
    def __init__(self, rate=5e-4, p=2, params=None):
        super(WeightDecay, self).__init__()

        self.p = p
        self.params = params
        self.rate = rate / self.p

    def on_start(self, state):
        if self.params is None:
            self.params = state['model'].parameters()

    def on_forward_criterion(self, state):
        for param in self.params:
            state['loss'] += self.rate * torch.pow(param, self.p).sum()


class L1WeightDecay(WeightDecay):
    def __init__(self, rate=5e-4, params=None):
        super(L1WeightDecay, self).__init__(rate=rate, p=1, params=params)


class L2WeightDecay(WeightDecay):
    def __init__(self, rate=5e-4, params=None):
        super(L2WeightDecay, self).__init__(rate=rate, p=2, params=params)
