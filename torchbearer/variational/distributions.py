import torch
import numpy as np
from abc import ABC


class DistributionBase(ABC):
    """
    Abstract class for a simple distribution which must implement rsample and log_prob
    """
    def rsample(self, sample_shape=torch.Size()):
        """
        Returns a reparameterized sample or batch of reparameterized samples if the distribution parameters are batched.
        """
        ...
    
    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        :param value: Value at which to evaluate log probabilty
        :type value: Tensor
        """
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

