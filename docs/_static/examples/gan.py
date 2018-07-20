import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

import torchbearer as tb
from torchbearer.callbacks import Callback

os.makedirs('images', exist_ok=True)

# Define constants
epochs = 200
batch_size = 64
lr = 0.0002
nworkers = 8
latent_dim = 100
sample_interval = 400
img_shape = (1, 28, 28)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()

    def forward(self, real_imgs, state):
        # Generator Forward
        z = Variable(torch.Tensor(np.random.normal(0, 1, (real_imgs.shape[0], latent_dim)))).to(state[tb.DEVICE])
        state['gen_imgs'] = self.generator(z)
        state['disc_gen'] = self.discriminator(state['gen_imgs'])
        # We don't want to discriminator gradients on the generator forward pass
        self.discriminator.zero_grad()

        # Discriminator Forward
        state['disc_gen_det'] = self.discriminator(state['gen_imgs'].detach())
        state['disc_real'] = self.discriminator(real_imgs)


class LossCallback(Callback):
    def on_start(self, state):
        super().on_start(state)
        self.adversarial_loss = torch.nn.BCELoss()
        state['valid'] = torch.ones(batch_size, 1, device=state[tb.DEVICE])
        state['fake'] = torch.zeros(batch_size, 1, device=state[tb.DEVICE])

    def on_criterion(self, state):
        super().on_criterion(state)
        fake_loss = self.adversarial_loss(state['disc_gen_det'], state['fake'])
        real_loss = self.adversarial_loss(state['disc_real'], state['valid'])
        state['g_loss'] = self.adversarial_loss(state['disc_gen'], state['valid'])
        state['d_loss'] = (real_loss + fake_loss) / 2
        # This is the loss that backward is called on.
        state[tb.LOSS] = state['g_loss'] + state['d_loss']


class SaverCallback(Callback):
    def on_step_training(self, state):
        super().on_step_training(state)
        batches_done = state[tb.EPOCH] * len(state[tb.GENERATOR]) + state[tb.BATCH]
        if batches_done % sample_interval == 0:
            save_image(state['gen_imgs'].data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)


# Configure data loader
os.makedirs('./data/mnist', exist_ok=True)
transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])

dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# Model and optimizer
model = GAN()
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))


@tb.metrics.running_mean
@tb.metrics.mean
class g_loss(tb.metrics.Metric):
    def __init__(self):
        super().__init__('g_loss')

    def process(self, state):
        return state['g_loss']


@tb.metrics.running_mean
@tb.metrics.mean
class d_loss(tb.metrics.Metric):
    def __init__(self):
        super().__init__('d_loss')

    def process(self, state):
        return state['d_loss']


def zero_loss(y_pred, y_true):
    return torch.zeros(y_true.shape[0], 1)


torchbearermodel = tb.Model(model, optim, zero_loss, ['loss', g_loss(), d_loss()])
torchbearermodel.to('cuda')
torchbearermodel.fit_generator(dataloader, epochs=200, pass_state=True, callbacks=[LossCallback(), SaverCallback()])
