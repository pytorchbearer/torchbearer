import torchbearer

from torchbearer.callbacks import Callback

import torch


class GradientNormClipping(Callback):
    """GradientNormClipping callback, which uses 'torch.nn.utils.clip_grad_norm\_' to clip the gradient norms to the
    given value. If params is None they will be retrieved from state.

    Example: ::

        >>> import torch.nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import GradientNormClipping

        # Example Trial which clips all model gradients norms at 2 under the L1 norm.
        >>> model = torch.nn.Linear(1,1)
        >>> clip = GradientNormClipping(2, 1)
        >>> trial = Trial(model, callbacks=[clip], metrics=['acc'])

    Args:
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        params (Iterable[Tensor] or Tensor, optional): an iterable of Tensors or a
            single Tensor that will have gradients normalized, otherwise this is retrieved from state

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: Model should have the `parameters` method
    """
    def __init__(self, max_norm, norm_type=2, params=None):
        super(GradientNormClipping, self).__init__()

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.params = params

    def on_start(self, state):
        """If params is None then retrieve from the model.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if self.params is None:
            self.params = filter(lambda p: p.requires_grad, state[torchbearer.MODEL].parameters())

    def on_backward(self, state):
        """Between the backward pass (which computes the gradients) and the step call (which updates the parameters),
        clip the gradient.

        Args:
            state (dict): The :class:`.Trial` state
        """
        torch.nn.utils.clip_grad_norm_(self.params, self.max_norm, norm_type=self.norm_type)


class GradientClipping(Callback):
    """GradientClipping callback, which uses 'torch.nn.utils.clip_grad_value\_' to clip the gradients of the given
    parameters to the given value. If params is None they will be retrieved from state.

    Example: ::

        >>> import torch.nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import GradientClipping

        # Example Trial which clips all model gradients at 2 under the L1 norm.
        >>> model = torch.nn.Linear(1,1)
        >>> clip = GradientNormClipping(2, 1)
        >>> trial = Trial(model, callbacks=[clip], metrics=['acc'])

    Args:
        clip_value (float or int): maximum allowed value of the gradients
            The gradients are clipped in the range [-clip_value, clip_value]
        params (Iterable[Tensor] or Tensor, optional): an iterable of Tensors or a
            single Tensor that will have gradients normalized, otherwise this is retrieved from state

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: Model should have the `parameters` method
    """
    def __init__(self, clip_value, params=None):

        super(GradientClipping, self).__init__()

        self.clip_value = clip_value
        self.params = params

    def on_start(self, state):
        """If params is None then retrieve from the model.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if self.params is None:
            self.params = filter(lambda p: p.requires_grad, state[torchbearer.MODEL].parameters())

    def on_backward(self, state):
        """Between the backward pass (which computes the gradients) and the step call (which updates the parameters),
        clip the gradient.

        Args:
            state (dict): The :class:`.Trial` state
        """
        torch.nn.utils.clip_grad_value_(self.params, self.clip_value)
