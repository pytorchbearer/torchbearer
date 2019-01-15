from abc import ABC, abstractmethod

import torch.nn.modules as nn

import torchbearer


class AutoEncoderBase(ABC, nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims

    @abstractmethod
    def encode(self, x, state=None):
        """Encode the given batch of images and return latent space sample for each.

        :param x: Batch of images to encode
        :param state: The trial state
        :return: Encoded samples / tuple of samples for different spaces
        """
        return

    @abstractmethod
    def decode(self, sample, state=None):
        """Decode the given latent space sample batch to images.

        :param sample: The latent space samples
        :param state: The trial state
        :return: Decoded images
        """
        return

    def forward(self, x, state=None):
        """Encode then decode the inputs, returning the result. Also binds the target as the input images in state.

        :param x: Model input batch
        :param state: The trial state
        :return: Auto-Encoded images
        """
        if state is not None:
            state[torchbearer.Y_TRUE] = x

        return self.decode(self.encode(x, state), state)
