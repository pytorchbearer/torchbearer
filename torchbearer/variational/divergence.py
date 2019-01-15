from abc import ABC, abstractmethod

import torch

import torchbearer
import torchbearer.callbacks as callbacks


class DivergenceBase(ABC, callbacks.Callback):
    def __init__(self, keys, state_key=None):
        self.keys = keys
        self.state_key = state_key

        self._post = lambda loss: loss
        self._reduce = lambda x: x.mean(0)

        def store(state, val):
            state[state_key] = val.detach()

        self._store = store if state_key is not None else (lambda state, val: ...)

    @abstractmethod
    def compute(self, **kwargs):
        return

    def loss(self, state):
        kwargs = dict([(name, state[self.keys[name]]) for name in self.keys.keys()])
        return self.compute(**kwargs)

    def on_criterion(self, state):
        div = self.loss(state)
        self._store(state, self._reduce(div))
        state[torchbearer.LOSS] = state[torchbearer.LOSS] + self._reduce(self._post(div))

    def on_criterion_validation(self, state):
        div = self.loss(state)
        self._store(state, self._reduce(div))
        state[torchbearer.LOSS] = state[torchbearer.LOSS] + self._reduce(self._post(div))

    def beta(self, beta):
        def beta_div(loss):
            return beta * loss
        self._post = beta_div
        return self

    def limit_capacity(self, gamma, c):
        def limit_div(loss):
            return gamma * (loss - c).relu()
        self._post = limit_div
        return self


class GaussianKLDivergence(DivergenceBase):
    def __init__(self, mu_key, logvar_key, state_key=None):
        super().__init__({'mu': mu_key, 'logvar': logvar_key}, state_key=state_key)

    def compute(self, mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(1)  # Conditionally independent


class GaussianVsGaussianKLDivergence(DivergenceBase):
    def __init__(self, mu_key_0, logvar_key_0, mu_key_1, logvar_key_1, state_key=None):
        super().__init__({'mu_0': mu_key_0, 'logvar_0': logvar_key_0, 'mu_1': mu_key_1, 'logvar_1': logvar_key_1}, state_key=state_key)

    def compute(self, mu_0, logvar_0, mu_1, logvar_1):
        kl = logvar_0.exp() / logvar_1.exp() + (mu_1 - mu_0).pow(2) / logvar_1.exp() + logvar_1 - logvar_0 - 1
        kl = -0.5 * (kl)
        return kl.sum(1)


