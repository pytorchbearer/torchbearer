import torch
import torch.nn as nn
import torch.optim as optim

import torchbearer
from . import ImagingCallback


class _Wrapper(nn.Module):
    def __init__(self, image, base_model):
        super(_Wrapper, self).__init__()
        self.base_model = base_model
        self.image = image

    def forward(self, _, state):
        x = self.image(_, state)
        try:
            return self.base_model(x, state)
        except TypeError:
            return self.base_model(x)


class BasicAscent(ImagingCallback):
    def __init__(self, image, criterion, transform=None, verbose=0,
                 optimizer=None, steps=256):
        super(BasicAscent, self).__init__(transform=transform)

        self.image = image
        self.criterion = criterion
        self.verbose = verbose
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, image.parameters()), lr=0.05) if optimizer is None else optimizer
        self.steps = steps

    @torchbearer.enable_grad()
    def on_batch(self, state):
        training = state[torchbearer.MODEL].training

        @torchbearer.callbacks.on_sample
        def make_eval(_):
            state[torchbearer.MODEL].eval()

        @torchbearer.callbacks.add_to_loss
        def loss(state):
            return - self.criterion(state)

        model = _Wrapper(self.image, state[torchbearer.MODEL])
        trial = torchbearer.Trial(model, self.optimizer, callbacks=[make_eval, loss])
        trial.for_train_steps(self.steps).to(state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
        trial.run(verbose=self.verbose)

        if training:
            state[torchbearer.MODEL].train()

        return model.image.get_valid_image()

    def run(self, model, verbose=2, device='cpu', dtype=torch.float32):
        old_verbose = self.verbose
        self.verbose = verbose

        state = torchbearer.State()
        state.update({torchbearer.MODEL: model, torchbearer.DEVICE: device, torchbearer.DATA_TYPE: dtype})
        self.process(state)
        self.verbose = old_verbose
