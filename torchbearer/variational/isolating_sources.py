import torch
import torchbearer as tb
from torchbearer.callbacks import Callback
from .divergence import DivergenceBase
from .distributions import SimpleGaussian
import math
from numbers import Number


class IsolatingSourcesLoss(Callback):
    def __init__(self, pz, qz_x, expanded_qz):
        """
        p distributions are priors, q distributions are posteriors

        :param pz: Distribution p(z)
        :param qz_x: Distribution q(z|x)
        :param expanded_qz: Distribution q(z|x) where parameters are unsqueezed in first dim. this is used to calculate q(z)


        """
        super().__init__()
        self.pz = pz
        self.qz_x = qz_x
        self.expanded_qz = expanded_qz

    def compute(self, state):
        bs, ds = 0, 0
        sample_key = tb.X

        sample = state[sample_key]

        logpz = self.pz.log_prob(sample)
        logqz_x = self.qz_x.log_prob(sample)
        _logqz = self.expanded_qz.log_prob(sample.unsqueeze(1))

        logqz_prodmarginals = (self.logsumexp(_logqz, dim=1, keepdim=False)-(bs*ds).log()).sum(1)
        logqz = (self.logsumexp(_logqz.sum(2), dim=1, keepdim=False)-(bs*ds).log())

    def logsumexp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                           dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)


class MutualInformationKL(DivergenceBase):
    def __init__(self, mu_key, logvar_key, state_key=None):
        super().__init__({'mu': mu_key, 'logvar': logvar_key}, state_key)

    def compute(self, mu, logvar):
        ...
        # lnqz_ni - lnqz
