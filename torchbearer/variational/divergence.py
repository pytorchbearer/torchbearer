import functools

import torchbearer
from torchbearer import cite
import torchbearer.callbacks as callbacks

from .beta_vae import bibtex as beta_vae


understanding_beta_vae = """
@article{burgess2018understanding,
  title={Understanding disentangling in beta-vae},
  author={Burgess, Christopher P and Higgins, Irina and Pal, Arka and Matthey, Loic and Watters, Nick and Desjardins, Guillaume and Lerchner, Alexander},
  journal={arXiv preprint arXiv:1804.03599},
  year={2018}
}
"""


class DivergenceBase(callbacks.Callback):
    """The :class:`DivergenceBase` class is an abstract base class which defines a series of useful methods for dealing
    with divergences. The keys dict given on init is used to map objects in state to kwargs in the compute function.

    Args:
        keys (dict): Dictionary which maps kwarg names to :class:`.StateKey` objects. When :meth:`compute` is called,
            the given kwargs are mapped to their associated values in state.
        state_key: If not None, the value outputted by :meth:`compute` is stored in state with the given key.
    """

    def __init__(self, keys, state_key=None):
        self.keys = keys
        self.state_key = state_key

        self._post = lambda loss: loss
        self._reduce = lambda x: x.sum(1).mean(0)

        def store(state, val):
            state[state_key] = val.detach()

        self._store = store if state_key is not None else (lambda state, val: ...)

    def with_post_function(self, post_fcn):
        """Register the given post function, to be applied after to loss after reduction.

        Args:
            post_fcn: A function of loss which applies some operation (e.g. multiplying by beta)

        Returns:
            :class:`.Divergence`: self
        """
        old_post = self._post
        self._post = lambda loss: post_fcn(old_post(loss))
        return self

    def compute(self, **kwargs):
        """Compute the loss with the given kwargs defined in the constructor.

        Args:
            kwargs: The bound kwargs, taken from state with the keys given in the constructor

        Returns:
            The calculated divergence as a two dimensional tensor (batch, distribution dimensions)
        """
        raise NotImplementedError

    def loss(self, state):
        kwargs = dict([(name, state[self.keys[name]]) for name in self.keys.keys()])
        return self.compute(**kwargs)

    def on_criterion(self, state):
        div = self._reduce(self.loss(state))
        self._store(state, div)
        state[torchbearer.LOSS] = state[torchbearer.LOSS] + self._post(div)

    def on_criterion_validation(self, state):
        div = self._reduce(self.loss(state))
        self._store(state, div)
        state[torchbearer.LOSS] = state[torchbearer.LOSS] + self._post(div)

    def with_reduction(self, reduction_fcn):
        """Override the reduction operation with the given function, use this if your divergence doesn't output a two
        dimensional tensor.

        Args:
            reduction_fcn: The function to be applied to the divergence output and return a single value

        Returns:
            :class:`.Divergence`: self
        """
        self._reduce = reduction_fcn
        return self

    def with_sum_mean_reduction(self):
        """Override the reduction function to take a sum over dimension one and a mean over dimension zero. (default)

        Returns:
            :class:`.Divergence`: self
        """
        return self.with_reduction(lambda x: x.sum(1).mean(0))

    def with_sum_sum_reduction(self):
        """Override the reduction function to take a sum over all dimensions.

        Returns:
            :class:`.Divergence`: self
        """
        return self.with_reduction(lambda x: x.sum())

    @cite(beta_vae)
    def with_beta(self, beta):
        """Multiply the divergence by the given beta, as introduced by beta-vae.
        :bib:

        Args:
            beta (float): The beta (> 1) to multiply by.

        Returns:
            :class:`.Divergence`: self
        """
        def beta_div(loss):
            return beta * loss
        return self.with_post_function(beta_div)

    @cite(understanding_beta_vae)
    def with_linear_capacity(self, min_c=0, max_c=25, steps=100000, gamma=1000):
        """Limit divergence by capacity, linearly increased from min_c to max_c for steps, as introduced in
        `Understanding disentangling in beta-VAE`.
        :bib:

        Args:
            min_c (float): Minimum capacity
            max_c (float): Maximum capacity
            steps (int): Number of steps to increase over
            gamma (float): Multiplicative gamma, usually a high number

        Returns:
            :class:`.Divergence`: self
        """
        inc = steps / (max_c - min_c)
        c = min_c
        old_callback = self.on_step_training

        @functools.wraps(old_callback)
        def step_c(self, state):
            nonlocal c
            if c < max_c:
                c += inc
            return old_callback(self, state)
        self.on_step_training = step_c

        def limit_div(loss):
            return gamma * (loss - c).abs()
        return self.with_post_function(limit_div)


class SimpleNormalUnitNormalKLDivergence(DivergenceBase):
    """A KL divergence between a SimpleNormal (or similar) instance and a fixed unit normal (N[0, 1]) target.

    .. note::

       The distribution object must have mu and logvar attributes

    Args:
        distribution_key: :class:`.StateKey` instance which will be mapped to the distribution object.
        state_key: If not None, the value outputted by :meth:`compute` is stored in state with the given key.
    """
    def __init__(self, distribution_key, state_key=None):
        super().__init__({'distribution': distribution_key}, state_key=state_key)

    def compute(self, distribution):
        mu, logvar = distribution.mu, distribution.logvar
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


class SimpleNormalSimpleNormalKLDivergence(DivergenceBase):
    """A KL divergence between two SimpleNormal (or similar) distributions.

    .. note::

       The distribution objects must have mu and logvar attributes

    Args:
        distribution_key_1: :class:`.StateKey` instance which will be mapped to the first distribution object.
        distribution_key_2: :class:`.StateKey` instance which will be mapped to the second distribution object.
        state_key: If not None, the value outputted by :meth:`compute` is stored in state with the given key.
    """
    def __init__(self, distribution_key_1, distribution_key_2, state_key=None):
        super().__init__({'distribution_1': distribution_key_1, 'distribution_2': distribution_key_2}, state_key=state_key)

    def compute(self, distribution_1, distribution_2):
        mu_1, logvar_1 = distribution_1.mu, distribution_1.logvar
        mu_2, logvar_2 = distribution_2.mu, distribution_2.logvar
        return -0.5 * (logvar_1.exp() / logvar_2.exp() + (mu_2 - mu_1).pow(2) / logvar_2.exp() + logvar_2 - logvar_1 - 1)
