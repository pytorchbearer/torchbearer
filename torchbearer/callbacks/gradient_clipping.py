import torchbearer

from torchbearer.callbacks import Callback

import torch


class GradientNormClipping(Callback):
    """GradientNormClipping callback, uses 'torch.nn.utils.clip_grad_norm\_'
    """

    def __init__(self, max_norm, norm_type=2, params=None):
        """Clip the given norm level to the given value. If params is None they will be retrieved from state.

        :param max_norm: The max norm value
        :param norm_type: The norm type to use
        :param params: The parameters to clip or None
        """
        super(GradientNormClipping, self).__init__()

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.params = params

    def on_start(self, state):
        """If params is None then retrieve from the model.

        :param state: The Model state
        :type state: dict
        """
        if self.params is None:
            self.params = filter(lambda p: p.requires_grad, state[torchbearer.MODEL].parameters())

    def on_backward(self, state):
        """Between the backward pass (which computes the gradients) and the step call (which updates the parameters),
        clip the gradient.

        :param state: The Model state
        :type state: dict
        """
        torch.nn.utils.clip_grad_norm_(self.params, self.max_norm, norm_type=self.norm_type)


class GradientClipping(Callback):
    """GradientClipping callback, uses 'torch.nn.utils.clip_grad_value\_'
    """

    def __init__(self, clip_value, params=None):
        """Clip the gradients of the given parameters to the given value. If params is None they will be retrieved from
        state.

        :param clip_value: The maximum absolute value of the gradient
        :param params: The parameters to clip or None
        """
        super(GradientClipping, self).__init__()

        self.clip_value = clip_value
        self.params = params

    def on_start(self, state):
        """If params is None then retrieve from the model.

        :param state: The Model state
        :type state: dict
        """
        if self.params is None:
            self.params = filter(lambda p: p.requires_grad, state[torchbearer.MODEL].parameters())

    def on_backward(self, state):
        """Between the backward pass (which computes the gradients) and the step call (which updates the parameters),
        clip the gradient.

        :param state: The Model state
        :type state: dict
        """
        torch.nn.utils.clip_grad_value_(self.params, self.clip_value)
