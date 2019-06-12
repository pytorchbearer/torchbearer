import torchbearer

from torchbearer.callbacks import Callback

import torch


class WeightDecay(Callback):
    """Create a WeightDecay callback which uses the given norm on the given parameters and with the given decay rate.
    If params is None (default) then the parameters will be retrieved from the model.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import WeightDecay

        # Example Trial which runs a trial with weight decay on the model
        >>> decay = WeightDecay()
        >>> trial = Trial(None, callbacks=[decay], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        rate (float): The decay rate or lambda
        p (int): The norm level
        params (Iterable[Tensor] or Tensor, optional): an iterable of Tensors or a
            single Tensor that will have gradients normalized, otherwise this is retrieved from state

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: Model should have the `parameters` method
        - :attr:`torchbearer.state.LOSS`: Loss should be a tensor that can be incremented
    """
    def __init__(self, rate=5e-4, p=2, params=None):
        super(WeightDecay, self).__init__()

        self.p = p
        self.params = params
        self.rate = rate

    def on_start(self, state):
        """Retrieve params from state['model'] if required.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if self.params is None:
            self.params = state[torchbearer.MODEL].parameters()

    def on_criterion(self, state):
        """Calculate the decay term and add to state['loss'].

        Args:
            state (dict): The :class:`.Trial` state
        """
        for param in self.params:
            state[torchbearer.LOSS] += self.rate * torch.norm(param, self.p)


class L1WeightDecay(WeightDecay):
    """WeightDecay callback which uses an L1 norm with the given rate and parameters. If params is None (default) then
    the parameters will be retrieved from the model.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import L1WeightDecay

        # Example Trial which runs a trial with weight decay on the model using an L1 norm
        >>> decay = L1WeightDecay()
        >>> trial = Trial(None, callbacks=[decay], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        rate (float): The decay rate or lambda
        params (Iterable[Tensor] or Tensor, optional): an iterable of Tensors or a
            single Tensor that will have gradients normalized, otherwise this is retrieved from state

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: Model should have the `parameters` method
        - :attr:`torchbearer.state.LOSS`: Loss should be a tensor that can be incremented
    """
    def __init__(self, rate=5e-4, params=None):
        super(L1WeightDecay, self).__init__(rate=rate, p=1, params=params)


class L2WeightDecay(WeightDecay):
    """WeightDecay callback which uses an L2 norm with the given rate and parameters. If params is None (default) then
    the parameters will be retrieved from the model.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import L2WeightDecay

        # Example Trial which runs a trial with weight decay on the model using an L2 norm
        >>> decay = L2WeightDecay()
        >>> trial = Trial(None, callbacks=[decay], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        rate (float): The decay rate or lambda
        params (Iterable[Tensor] or Tensor, optional): an iterable of Tensors or a
            single Tensor that will have gradients normalized, otherwise this is retrieved from state

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: Model should have the `parameters` method
        - :attr:`torchbearer.state.LOSS`: Loss should be a tensor that can be incremented
    """
    def __init__(self, rate=5e-4, params=None):
        super(L2WeightDecay, self).__init__(rate=rate, p=2, params=params)
