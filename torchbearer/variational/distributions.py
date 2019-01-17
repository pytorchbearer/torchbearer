import math
from numbers import Number

import torch
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all


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
    """The SimpleNormal class is a :class:`SimpleDistribution` which implements a straight forward Normal / Gaussian
    distribution. This performs significantly fewer checks than `torch.distributions.Normal`, but should be sufficient
    for the purpose of implementing a VAE.

    Args:
        mu (torch.Tensor, Number): The mean of the distribution, numbers will be cast to tensors
        logvar (torch.Tensor, Number): The log variance of the distribution, numbers will be cast to tensors
    """
    def __init__(self, mu, logvar):
        self.mu, self.logvar = broadcast_all(mu, logvar)
        if isinstance(mu, Number) and isinstance(logvar, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super().__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Simple rsample for a Normal distribution.

        Args:
            sample_shape: Shape of the sample (per mean / variance given)

        Returns:
            A reparameterized sample with gradient with respect to the distribution parameters
        """
        shape = self._extended_shape(sample_shape)
        std = self.logvar.div(2).exp_()
        eps = torch.normal(torch.zeros(shape, dtype=self.mu.dtype, device=self.mu.device),
                           torch.ones(shape, dtype=self.mu.dtype, device=self.mu.device))
        return self.mu + std * eps

    def log_prob(self, value):
        """Calculates the log probability that the given value was drawn from this distribution. Since the density of a
        Gaussian is differentiable, this function is differentiable.

        Args:
            value (torch.Tensor, Number): The sampled value

        Returns:
            The log probability that the given value was drawn from this distribution
        """
        var = self.logvar.exp()
        return - ((value - self.mu) ** 2) / (2.0 * var) - (self.logvar / 2.0) - math.log(math.sqrt(2.0 * math.pi))


class SimpleUniform(SimpleDistribution):
    """The SimpleUniform class is a :class:`SimpleDistribution` which implements a straight forward Uniform distribution
    in the interval ``[low, high)``. This performs significantly fewer checks than `torch.distributions.Uniform`, but
    should be sufficient for the purpose of implementing a VAE.

    Args:
        low (torch.Tensor, Number): The lower range of the distribution (inclusive), numbers will be cast to tensors
        high (torch.Tensor, Number): The upper range of the distribution (exclusive), numbers will be cast to tensors
    """
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = broadcast_all(low, high)
        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Simple rsample for a Uniform distribution.

        Args:
            sample_shape: Shape of the sample (per low / high given)

        Returns:
            A reparameterized sample with gradient with respect to the distribution parameters
        """
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        """Calculates the log probability that the given value was drawn from this distribution. Since this distribution
        is uniform, the log probability is zero for all values in the range ``[low, high)`` and -inf elsewhere. This
        function is therefore non-differentiable.

        Args:
            value (torch.Tensor, Number): The sampled value

        Returns:
            The log probability that the given value was drawn from this distribution
        """
        value = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.get_default_dtype())
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

