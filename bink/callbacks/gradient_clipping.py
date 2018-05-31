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
            self.params = state['model'].parameters()

    def on_backward(self, state):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_norm, norm_type=self.norm_type)
