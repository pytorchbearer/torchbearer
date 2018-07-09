from bink.callbacks import Callback

import torch


class GradientNormClipping(Callback):
    def __init__(self, max_norm, norm_type=2, params=None):
        super(GradientNormClipping, self).__init__()

        self.max_norm = max_norm
        self.norm_type = norm_type
        self.params = params

    def on_start(self, state):
        if self.params is None:
            self.params = filter(lambda p: p.requires_grad, state['model'].parameters())

    def on_backward(self, state):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_norm, norm_type=self.norm_type)


class GradientClipping(Callback):
    def __init__(self, max_abs, params=None):
        super(GradientClipping, self).__init__()

        self.max_abs = max_abs
        self.params = params

    def on_start(self, state):
        if self.params is None:
            self.params = filter(lambda p: p.requires_grad, state['model'].parameters())

    def on_backward(self, state):
        for p in self.params:
            p.grad.data.clamp_(-self.max_abs, self.max_abs)
