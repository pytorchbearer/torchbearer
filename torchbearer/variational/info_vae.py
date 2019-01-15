import torchbearer
from torchbearer.callbacks import Callback

from .auto_encoder import AutoEncoderBase


class InfoVAEDivergence(Callback):
    """Implements the divergence rule from InfoVAE. Needs to wrap its own data loader to get an unbiased sample. Expects
    Model to be an instance of AutoEncoderBase.
    """
    def __init__(self, generator, divergence):
        super().__init__()

        self.generator = generator
        self.iterator = iter(self.generator)
        self.divergence = divergence

    def compute(self, state):
        model = state[torchbearer.MODEL]
        if isinstance(model, AutoEncoderBase):
            done = False
            x = None
            while not done:
                try:
                    x, _ = next(self.iterator)
                    done = True
                except StopIteration:
                    self.iterator = iter(self.generator)
            x = x.to(device=state[torchbearer.DEVICE], dtype=state[torchbearer.DATA_TYPE])
            state = state.copy()
            model.encode(x, state)
            return state
        else:
            print('error')  # TODO: Make an actual error!

    def on_criterion(self, state):
        self.divergence.on_criterion(self.compute(state))

    def on_criterion_validation(self, state):
        self.divergence.on_criterion_validation(self.compute(state))
