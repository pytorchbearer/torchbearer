import torchbearer

from torchbearer.callbacks import Callback

import torch


class WeightDecay(Callback):
    """Create a WeightDecay callback which uses the given norm on the given parameters and with the given decay rate. If params is None (default) then the parameters will be retrieved from the model.

    :param rate: The decay rate
    :type rate: float
    :param p: The norm level
    :type p: int
    :param params: The parameters to use (or None)
    :type params: list
    """
    def __init__(self, rate=5e-4, p=2, params=None):
        super(WeightDecay, self).__init__()

        self.p = p
        self.params = params
        self.rate = rate / self.p

    def on_start(self, state):
        """Retrieve params from state['model'] if required.

        :param state: The Model state
        :type state: dict
        """
        if self.params is None:
            self.params = state[torchbearer.MODEL].parameters()

    def on_criterion(self, state):
        """Calculate the decay term and add to state['loss'].

        :param state: The Model state
        :type state: dict
        """
        for param in self.params:
            state[torchbearer.LOSS] += self.rate * torch.pow(param, self.p).sum()


class L1WeightDecay(WeightDecay):
    """WeightDecay callback which uses an L1 norm with the given rate and parameters. If params is None (default) then the parameters will be retrieved from the model.

    :param rate: The decay rate
    :type rate: float
    :param params: The parameters to use (or None)
    :type params: list
    """
    def __init__(self, rate=5e-4, params=None):
        super(L1WeightDecay, self).__init__(rate=rate, p=1, params=params)


class L2WeightDecay(WeightDecay):
    """WeightDecay callback which uses an L2 norm with the given rate and parameters. If params is None (default) then the parameters will be retrieved from the model.

    :param rate: The decay rate
    :type rate: float
    :param params: The parameters to use (or None)
    :type params: list
    """
    def __init__(self, rate=5e-4, params=None):
        super(L2WeightDecay, self).__init__(rate=rate, p=2, params=params)
