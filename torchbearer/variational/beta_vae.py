from .auto_encoder import AutoEncoderBase
from .distributions import SimpleNormal

import torch.nn.modules as nn

from torchbearer import state_key

bibtex = """
@article{higgins2016beta,
  title={beta-vae: Learning basic visual concepts with a constrained variational framework},
  author={Higgins, Irina and Matthey, Loic and Pal, Arka and Burgess, Christopher and Glorot, Xavier and Botvinick, Matthew and Mohamed, Shakir and Lerchner, Alexander},
  year={2016}
}
"""

MU = state_key('mu')
LOGVAR = state_key('logvar')


class BetaVAE2DShapes(AutoEncoderBase):
    """2D Shapes model from beta VAE paper. Input size: 1 x 64 x 64. Use with a Bernoulli error distribution
    (BinaryCrossEntropy loss).

    :param latents: Latent space size (10 in the paper)
    """
    def __init__(self, latents=10):
        super().__init__(latent_dims=[latents])
        self.latent_dim = latents

        self.encoder = nn.Sequential(
            nn.Linear(4096, 1200),
            nn.ReLU(True),
            nn.Linear(1200, 1200),
            nn.ReLU(True)
        )

        self.mu = nn.Linear(1200, latents)
        self.logvar = nn.Linear(1200, latents)

        self.decoder = nn.Sequential(
            nn.Linear(latents, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def encode(self, x, state=None):
        x = self.encoder(x.view(x.size(0), -1))
        mu = self.mu(x)
        logvar = self.logvar(x)
        sample = SimpleNormal(mu, logvar).rsample()

        if state is not None:
            state[MU] = mu
            state[LOGVAR] = logvar

        return [sample]

    def decode(self, sample, state=None):
        x = self.decoder(sample[0]).view(-1, 1, 64, 64)
        return x.sigmoid()  # Bernoulli


class BetaVAECNN(AutoEncoderBase):
    """CNN model from beta VAE paper. Input size: nc x 64 x 64.

    :param latents: Latent space size
    :param nc: number of channels in input / output
    :param sigmoid_output: If True, sigmoid the output, otherwise do nothing
    """
    def __init__(self, latents, nc, sigmoid_output=False):
        super().__init__(latent_dims=[latents])

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True)
        )

        self.fc = nn.Linear(1024, 256)
        self.mu = nn.Linear(256, latents)
        self.logvar = nn.Linear(256, latents)

        self.fct = nn.Sequential(
            nn.Linear(latents, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1)
        )

        self._post_fcn = (lambda x: x.sigmoid()) if sigmoid_output else (lambda x: x)

    def encode(self, x, state=None):
        x = self.encoder(x)
        x = self.fc(x.view(x.size(0), -1))
        mu = self.mu(x)
        logvar = self.logvar(x)
        sample = SimpleNormal(mu, logvar).rsample()

        if state is not None:
            state[MU] = mu
            state[LOGVAR] = logvar

        return sample

    def decode(self, sample, state=None):
        x = self.fct(sample).view(-1, 64, 4, 4)
        x = self.decoder(x)
        return self._post_fcn(x)


class BetaVAEChairs(BetaVAECNN):
    """Chairs model from beta VAE paper. Input size: 1 x 64 x 64. Use with a Bernoulli error distribution
    (BinaryCrossEntropy loss).

    :param latents: Latent space size (32 in the paper)
    """
    def __init__(self, latents=32):
        super().__init__(latents, 1, True)


class BetaVAECelebA(BetaVAECNN):
    """CelebA model from beta VAE paper. Input size: 3 x 64 x 64. Use with a Gaussian error distribution
    (MeanSquaredError loss).

    :param latents: Latent space size (32 in the paper)
    """
    def __init__(self, latents=32):
        super().__init__(latents, 3, False)


class BetaVAE3DFaces(BetaVAECNN):
    """3DFaces model from beta VAE paper. Input size: 1 x 64 x 64. Use with a Bernoulli error distribution
    (BinaryCrossEntropy loss).

    :param latents: Latent space size (32 in the paper)
    """
    def __init__(self, latents=32):
        super().__init__(latents, 1, True)
