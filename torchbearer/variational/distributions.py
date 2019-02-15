"""
The distributions module is an extension of the `torch.distributions` package intended to facilitate implementations
required for specific variational approaches through the :class:`.SimpleDistribution` class. Generally, using a
:class:`torch.distributions.Distribution` object should be preferred over a :class:`SimpleDistribution`, for better
argument validation and more complete implementations. However, if you need to implement something new for a specific
variational approach, then a :class:`.SimpleDistribution` may be more forgiving. Furthermore, you may find it easier
to understand the function of the implementations here.
"""
import math
from numbers import Number

import torch
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all
from torchbearer import cite

steve = """
@article{squires2019a,
title={A Variational Autoencoder for Probabilistic Non-Negative Matrix Factorisation},
author={Steven Squires and Adam Prugel-Bennett and Mahesan Niranjan},
year={2019}
}
"""


class SimpleDistribution(Distribution):
    """Abstract base class for a simple distribution which only implements rsample and log_prob. If the log_prob
    function is not differentiable with respect to the distribution parameters or the given value, then this should be
    mentioned in the documentation.
    """
    has_rsample = True

    def __init__(self, batch_shape=torch.Size(), event_shape=torch.Size()):
        super(SimpleDistribution, self).__init__(batch_shape, event_shape)

    @property
    def support(self):
        return None

    @property
    def arg_constraints(self):
        return None

    def expand(self, batch_shape, _instance=None):
        pass

    @property
    def mean(self):
        return None

    @property
    def variance(self):
        return None

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
        """Returns the log of the probability density/mass function evaluated at `value`.
        Args:
            value (torch.Tensor, Number): Value at which to evaluate log probabilty
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
        super(SimpleNormal, self).__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Simple rsample for a Normal distribution.

        Args:
            sample_shape (torch.Size, tuple): Shape of the sample (per mean / variance given)
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
        self.low, self.high = broadcast_all(low, high)
        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super(SimpleUniform, self).__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Simple rsample for a Uniform distribution.

        Args:
            sample_shape (torch.Size, tuple): Shape of the sample (per low / high given)
        Returns:
            A reparameterized sample with gradient with respect to the distribution parameters
        """
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        """Calculates the log probability that the given value was drawn from this distribution. Since this distribution
        is uniform, the log probability is ``-log(high - low)`` for all values in the range ``[low, high)`` and -inf
        elsewhere. This function is therefore only piecewise differentiable.

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
    """The SimpleExponential class is a :class:`SimpleDistribution` which implements a straight forward Exponential
    distribution with the given lograte. This performs significantly fewer checks than `torch.distributions.Exponential`
    , but should be sufficient for the purpose of implementing a VAE. By using a lograte, the log_prob can be computed
    in a stable fashion, without taking a logarithm.

    Args:
        lograte (torch.Tensor, Number): The natural log of the rate of the distribution, numbers will be cast to tensors
    """

    def __init__(self, lograte):
        self.lograte, = broadcast_all(lograte)
        batch_shape = torch.Size() if isinstance(lograte, Number) else self.lograte.size()
        super(SimpleExponential, self).__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Simple rsample for an Exponential distribution.

        Args:
            sample_shape (torch.Size, tuple): Shape of the sample (per lograte given)
        Returns:
            A reparameterized sample with gradient with respect to the distribution parameters
        """
        shape = self._extended_shape(sample_shape)
        return self.lograte.new(shape).exponential_() / self.lograte.exp()

    def log_prob(self, value):
        """Calculates the log probability that the given value was drawn from this distribution. The log_prob for this
        distribution is fully differentiable and has stable gradient since we use the lograte here.

        Args:
            value (torch.Tensor, Number): The sampled value
        Returns:
            The log probability that the given value was drawn from this distribution
        """
        return self.lograte - self.lograte.exp() * value


@cite(steve)
class SimpleWeibull(SimpleDistribution):
    """The SimpleWeibull class is a :class:`SimpleDistribution` which implements a straight forward Weibull
    distribution. This performs significantly fewer checks than `torch.distributions.Weibull`, but should be sufficient
    for the purpose of implementing a VAE.

    Args:
        l (torch.Tensor, Number): The scale parameter of the distribution, numbers will be cast to tensors
        k (torch.Tensor, Number): The shape parameter of the distribution, numbers will be cast to tensors
    """

    def __init__(self, l, k):
        self.l, self.k = broadcast_all(l, k)
        self.const=1e-8
        if isinstance(k, Number) and isinstance(l, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.k.size()

        super(SimpleWeibull, self).__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Simple rsample for a Weibull distribution.

        Args:
            sample_shape (torch.Size, tuple): Shape of the sample (per k / lambda given)
        Returns:
            A reparameterized sample with gradient with respect to the distribution parameters
        """
        shape = self._extended_shape(sample_shape)
        eps = torch.rand(shape, dtype=self.k.dtype, device=self.k.device)
        return self.l * torch.pow((-torch.log(eps)), (1/self.k))

    def log_prob(self, value):
        """Calculates the log probability that the given value was drawn from this distribution.  This function is differentiable
        and its log probability is -inf for values less than 0.

        Args:
            value (torch.Tensor, Number): The sampled value
        Returns:
            The log probability that the given value was drawn from this distribution
        """
        value = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.get_default_dtype())
        lb=value.ge(torch.zeros(value.shape, dtype=self.k.dtype, device=self.k.device)).float()
        return torch.log(lb) + torch.log(self.k/self.l) + (self.k - torch.ones(self.k.shape, dtype=self.k.dtype, device=self.k.device))*torch.log((lb*value+self.const)/self.l) - torch.pow(value/self.l, self.k)
