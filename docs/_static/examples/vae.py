import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import save_image

import torchbearer
from torchbearer.callbacks import Callback


class AutoEncoderMNIST(Dataset):
    def __init__(self, mnist_dataset):
        super().__init__()
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        character, label = self.mnist_dataset.__getitem__(index)
        return character, character

    def __len__(self):
        return len(self.mnist_dataset)


BATCH_SIZE = 128

transform = transforms.Compose([transforms.ToTensor()])

# Define standard classification mnist dataset

basetrainset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)

basetestset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)

# Wrap base classification mnist dataset to return the image as the target

trainset = AutoEncoderMNIST(basetrainset)

testset = AutoEncoderMNIST(basetestset)

traingen = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

testgen = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x, state):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        state['mu'] = mu
        state['logvar'] = logvar
        return self.decode(z)


def bce_loss(y_pred, y_true):
    BCE = F.binary_cross_entropy(y_pred, y_true.view(-1, 784), size_average=False)
    return BCE


class AddKLDLoss(Callback):
    def on_criterion(self, state):
        super().on_criterion(state)
        KLD = self.KLD_Loss(state['mu'], state['logvar'])
        state[torchbearer.LOSS] = state[torchbearer.LOSS] + KLD

    def KLD_Loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD


class SaveReconstruction(Callback):
    def __init__(self, num_images=8, folder='results/'):
        super().__init__()
        self.num_images = num_images
        self.folder = folder

    def on_step_validation(self, state):
        super().on_step_validation(state)
        if state[torchbearer.BATCH] == 0:
            data = state[torchbearer.X]
            recon_batch = state[torchbearer.Y_PRED]
            comparison = torch.cat([data[:self.num_images],
                                    recon_batch.view(128, 1, 28, 28)[:self.num_images]])
            save_image(comparison.cpu(),
                       str(self.folder) + 'reconstruction_' + str(state[torchbearer.EPOCH]) + '.png', nrow=self.num_images)


model = VAE()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
loss = bce_loss

from torchbearer import Model

torchbearer_model = Model(model, optimizer, loss, metrics=['loss']).to('cuda')
torchbearer_model.fit_generator(traingen, epochs=10, validation_generator=testgen, callbacks=[AddKLDLoss(), SaveReconstruction()], pass_state=True)
