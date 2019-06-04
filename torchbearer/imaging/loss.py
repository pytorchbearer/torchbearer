import torch
import torch.nn.functional as F

from .images import IMAGE
import torchbearer

LAYER_DICT = torchbearer.state_key('layer_dict')


def _evaluate_target(state, target, channels=lambda x: x[:]):
    if isinstance(target, torchbearer.StateKey):
        return channels(state[target])
    else:
        return channels(state[LAYER_DICT][target])


class Criterion(object):
    def process(self, state):
        raise NotImplementedError

    def __call__(self, state):
        return self.process(state)

    def __add__(self, other):
        if callable(other):
            return LambdaCriterion(lambda state: self(state) + other(state))
        else:
            return LambdaCriterion(lambda state: self(state) + other)

    def __mul__(self, other):
        if callable(other):
            return LambdaCriterion(lambda state: self(state) * other(state))
        else:
            return LambdaCriterion(lambda state: self(state) * other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-1 * other)

    def __neg__(self):
        return -1 * self


class LambdaCriterion(Criterion):
    def __init__(self, function):
        self._function = function

    def process(self, state):
        return self._function(state)


criterion = LambdaCriterion


class Channel(Criterion):
    def __init__(self, channel, target):
        super(Channel, self).__init__()
        self._channel = channel
        self._target = target

    def process(self, state):
        return _evaluate_target(state, self._target)[0, self._channel].mean()


class TotalVariation(Criterion):
    def __init__(self, target=IMAGE):
        super(TotalVariation, self).__init__()
        self._target = target

    def process(self, state):
        target = _evaluate_target(state, self._target)
        if target.dim() == 4:
            target = target[0]
        return torch.sum(torch.abs(target[:, :, :-1] - target[:, :, 1:])) + torch.sum(torch.abs(target[:, :-1, :] - target[:, 1:, :]))


class DeepDream(Criterion):
    def __init__(self, target):
        super(DeepDream, self).__init__()
        self._target = target

    def process(self, state):
        return _evaluate_target(state, self._target).pow(2).mean()


class L1(Criterion):
    def __init__(self, constant=0, target=IMAGE, channels=lambda x: x[:]):
        super(L1, self).__init__()
        self._constant = constant
        self._target = target
        self._channels = channels

    def process(self, state):
        return (_evaluate_target(state, self._target, self._channels) - self._constant).abs().sum()


class L2(Criterion):
    def __init__(self, constant=0, eps=1e-6, target=IMAGE, channels=lambda x: x[:]):
        super(L2, self).__init__()
        self._constant = constant
        self._eps = eps
        self._target = target
        self._channels = channels

    def process(self, state):
        return ((_evaluate_target(state, self._target, self._channels) - self._constant).pow(2).sum() + self._eps).sqrt()


class Blur(Criterion):
    def __init__(self, target=IMAGE, channels=lambda x: x[:]):
        self._target = target
        self._channels = channels

    @staticmethod
    def _blur(x):
        depth = x.size(1)
        k = torch.zeros(3, 3, depth, depth)
        for ch in range(depth):
            k_ch = k[:, :, ch, ch]
            k_ch[:, :] = 0.5
            k_ch[1:-1, 1:-1] = 1.0
        k = k.permute(3, 2, 0, 1).to(x.device)
        conv_k = lambda x: F.conv2d(x, k, padding=1)
        return conv_k(x) / conv_k(torch.ones_like(x))

    def process(self, state):
        x = _evaluate_target(state, self._target, channels=self._channels)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x_blur = Blur._blur(x).detach()

        return 0.5 * (x - x_blur).pow(2).sum()


class BlurAlpha(Blur):
    def __init__(self, target=IMAGE):
        super(BlurAlpha, self).__init__(target=target, channels=lambda x: x[-1].unsqueeze(0))
