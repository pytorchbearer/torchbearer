import torch
import torchbearer


class RedirectReLUs(torchbearer.Callback):
    """Callback that replaces all ReLU or ReLU6 modules in the model with
    `redirected ReLU <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__
    versions for the first 16 iterations. Note that this doesn't apply to nn.functional ReLUs.

    Example::

        >>> import torchbearer
        >>> from torchbearer import Trial
        >>> from torchbearer.imaging.models.redirect_relu import RedirectReLUs
        >>> model = torch.nn.Sequential(torch.nn.ReLU())
        >>> @torchbearer.callbacks.on_sample
        ... def input_data(state):
        ...     state[torchbearer.X] = torch.rand(1, 1)
        >>> trial = Trial(model, callbacks=[RedirectReLUs(), input_data]).for_steps(1).run()
        >>> print(model)
        Sequential(
          (0): RedirectedReLU()
        )
        >>> model = torch.nn.Sequential(torch.nn.ReLU())
        >>> trial = Trial(model, callbacks=[RedirectReLUs(), input_data]).for_steps(17).run()
        >>> print(model)
        Sequential(
          (0): ReLU()
        )

    """
    def __init__(self):
        super().__init__()
        self.relu_types = [torch.nn.ReLU]
        self.old_modules = {}

    def on_start(self, state):
        super().on_start(state)
        for i, m in enumerate(state[torchbearer.MODEL].children()):
            if type(m) == torch.nn.ReLU:
                self.old_modules[i] = m
                state[torchbearer.MODEL]._modules[str(i)] = RedirectedReLU()
            elif type(m) == torch.nn.ReLU6:
                self.old_modules[i] = m
                state[torchbearer.MODEL]._modules[str(i)] = RedirectedReLU6()

    def on_sample(self, state):
        if state[torchbearer.BATCH] == 16:
            for i, m in enumerate(state[torchbearer.MODEL].children()):
                if type(m) == RedirectedReLU:
                    state[torchbearer.MODEL]._modules[str(i)] = self.old_modules[i]


class RedirectedReLU(torch.nn.Module):
    def forward(self, x):
        return RedirectedReLUFunction.apply(x)


class RedirectedReLU6(torch.nn.Module):
    def forward(self, x):
        return RedirectedReLU6Function.apply(x)


class RedirectedReLUFunction(torch.autograd.Function):
    """Reimplementation of the redirected ReLU from
    `tensorflows lucid library <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__.

    """
    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        relu_grad = grad_input.clone()
        relu_grad[input < 0] = 0

        neg_pushing_lower = torch.lt(input, 0) & torch.gt(grad_input, 0)
        redirected_grad = grad_input
        redirected_grad[neg_pushing_lower] = 0

        batch = grad_input.shape[0]
        reshaped_relu_grad = relu_grad.view(batch, -1)
        relu_grad_mag = torch.norm(reshaped_relu_grad, p=2, dim=1)

        result_grad = relu_grad
        result_grad[relu_grad_mag == 0, :] = redirected_grad[relu_grad_mag == 0, :]

        return result_grad


class RedirectedReLU6Function(torch.autograd.Function):
    """Reimplementation of the redirected ReLU6 from
    `tensorflows lucid library <https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py>`__.

    """
    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return input.clamp(min=0, max=6)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        relu_grad = grad_input.clone()
        relu_grad[input < 0] = 0
        relu_grad[input > 6] = 0

        neg_pushing_lower = torch.lt(input, 0) & torch.gt(grad_input, 0)
        pos_pushing_higher = torch.gt(input, 6) & torch.lt(grad_input, 0)

        redirected_grad = grad_input
        redirected_grad[neg_pushing_lower] = 0
        redirected_grad[pos_pushing_higher] = 0

        batch = grad_input.shape[0]
        reshaped_relu_grad = relu_grad.view(batch, -1)
        relu_grad_mag = torch.norm(reshaped_relu_grad, p=2, dim=1)

        result_grad = relu_grad
        result_grad[relu_grad_mag == 0, :] = redirected_grad[relu_grad_mag == 0, :]

        return result_grad
