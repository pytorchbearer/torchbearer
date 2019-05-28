import torch.nn.modules as nn

import torchbearer


class AutoEncoderBase(nn.Module):
    def __init__(self, latent_dims):
        super(AutoEncoderBase, self).__init__()
        self.latent_dims = latent_dims

    def encode(self, x, state=None):
        """Encode the given batch of images and return latent space sample for each.

        Args:
            x: Batch of images to encode
            state: The trial state

        Returns:
            Encoded samples / tuple of samples for different spaces
        """
        raise NotImplementedError

    def decode(self, sample, state=None):
        """Decode the given latent space sample batch to images.

        Args:
            sample: The latent space samples
            state: The trial state

        Returns:
            Decoded images
        """
        raise NotImplementedError

    def forward(self, x, state=None):
        """Encode then decode the inputs, returning the result. Also binds the target as the input images in state.

        Args:
            x: Model input batch
            state: The trial state

        Returns:
            Auto-Encoded images
        """
        if state is not None:
            state[torchbearer.Y_TRUE] = x

        return self.decode(self.encode(x, state), state)
