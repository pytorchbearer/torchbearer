import math
from numbers import Number

import torch
from torch.distributions import Distribution


class SimpleDistribution(Distribution):
    """
    abstract base class for a simple distribution which only implements rsample and log_prob
    """
    has_rsample = True

    def __init__(self, batch_shape=torch.Size(), event_shape=torch.Size()):
        super().__init__(batch_shape, event_shape)

    @property
    def support(self):
        pass

    @property
    def arg_constraints(self):
        pass

    def expand(self, batch_shape, _instance=None):
        pass

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def cdf(self, value):
        pass

    def icdf(self, value):
        pass

    def enumerate_support(self, expand=True):
        pass

    def entropy(self):
        pass

    def rsample(self, sample_shape=torch.Size()):
        """
        Returns a reparameterized sample or batch of reparameterized samples if the distribution parameters are batched.
        """
        raise NotImplementedError
    
    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        :param value: Value at which to evaluate log probabilty
        :type value: Tensor
        """
        raise NotImplementedError


class SimpleNormal(SimpleDistribution):
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        if isinstance(mu, Number) and isinstance(logvar, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = mu.size()
        super().__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        std = math.exp(self.logvar / 2.0) if isinstance(self.logvar, Number) else self.logvar.div(2).exp_()
        eps = torch.normal(torch.zeros(shape, dtype=self.mu.dtype, device=self.mu.device),
                           torch.ones(shape, dtype=self.mu.dtype, device=self.mu.device))
        return self.mu + std * eps

    def log_prob(self, value):
        var = math.exp(self.logvar) if isinstance(self.logvar, Number) else self.logvar.exp()
        return - ((value - self.mu) ** 2) / (2.0 * var) - (self.logvar / 2.0) - math.log(math.sqrt(2.0 * math.pi))


class SimpleUniform(SimpleDistribution):
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


class SimpleExponential(SimpleDistribution):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        return self.rate.log() - self.rate * value

