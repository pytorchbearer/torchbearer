import torch
import numpy as np

def gaussian_rsample(mu, logvar):
    std = logvar.div(2).exp_()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


def weibull_rsample():
    ...


def uniform_rsample():
    ...


class DistributionBase(object):
    def rsample(self, sample_shape=torch.Size()):
        ...
    
    def log_prob(self, value):
        ...


class SimpleGaussian(DistributionBase):
    def __init__(self, mu, logvar):
        super().__init__()
        self.mu = mu
        self.logvar = logvar

    def rsample(self, sample_shape=torch.Size()):
        std = self.logvar.div(2).exp_()
        eps = std.data.new(std.size()).normal_()
        return self.mu + std * eps

    def log_prob(self, value):
        var = self.logvar.exp()
        return (value - self.mu)**2/(2*var) - ((2*np.pi*var)**0.5).log()

