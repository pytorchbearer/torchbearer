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


class SimpleUniform(DistributionBase):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def rsample(self, sample_shape=torch.Size()):
        rand = torch.rand(sample_shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        lb = value.ge(self.low).type_as(self.low)
        ub = value.lt(self.high).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)


class SimpleExponential(DistributionBase):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        return self.rate.log() - self.rate * value

