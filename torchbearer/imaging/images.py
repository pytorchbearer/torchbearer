import math

import torch
import torch.nn as nn

import torchbearer


def _correlate_color(image, correlation, max_norm):
    if image.size(0) == 4:
        alpha = image[-1].unsqueeze(0)
        image = image[:-1]
    else:
        alpha = None
    shape = image.shape
    image = image.view(3, -1).permute(1, 0)
    color_correlation_normalized = correlation / max_norm
    image = image.matmul(color_correlation_normalized.t())
    image = image.permute(1, 0).contiguous().view(shape)
    if alpha is not None:
        image = torch.cat((image, alpha), dim=0)
    return image


class Image(nn.Module):
    def __init__(self, transform=None, decorrelate=True):
        super(Image, self).__init__()

        self.color_correlation_svd_sqrt = nn.Parameter(
            torch.tensor([[0.26, 0.09, 0.02],
                          [0.27, 0.00, -0.05],
                          [0.27, -0.09, 0.03]], dtype=torch.float32),
            requires_grad=False)
        self.max_norm_svd_sqrt = self.color_correlation_svd_sqrt.norm(dim=0).max()

        self.transform = transform if transform is not None else lambda x: x

        self.activation = lambda x: x
        self.correlate = (lambda x: _correlate_color(x, self.color_correlation_svd_sqrt,
                                                     self.max_norm_svd_sqrt)) if decorrelate else (lambda x: x)

    @property
    def image(self):
        raise NotImplementedError

    def get_rgb_image(self):
        return self.activation(self.correlate(self.image))

    def forward(self, _, state):
        image = self.get_rgb_image()
        x = self.transform(image).unsqueeze(0)
        state[torchbearer.INPUT] = x
        return x

    def with_activation(self, function):
        self.activation = function
        return self

    def sigmoid(self):
        return self.with_activation(torch.sigmoid)

    def tanh(self):
        return self.with_activation(torch.tanh)

    def clip(self, floor=0., ceil=1.):
        scale = ceil - floor
        return self.with_activation(lambda x: ((x.tanh() + 1.) / 2.) * scale + floor)

    # def __add__(self, other):
    #     if torch.is_tensor(other):
    #         other = Image.from_tensor(other, sigmoid=False, decorrelate=False, requires_grad=False)
    #     return AddImage(self, other, transform=self.transform)
    #
    # def __radd__(self, other):
    #     return self.__add__(other)

    @staticmethod
    def from_tensor(tensor, transform=None, decorrelate=True, requires_grad=True):
        return TensorImage(tensor, transform=transform, decorrelate=decorrelate, requires_grad=requires_grad)


# class AddImage(Image):
#     def __init__(self, a, b, transform=None):
#         super(AddImage, self).__init__(transform=transform, sigmoid=False, decorrelate=False)
#
#         self.a = a
#         self.b = b
#
#     @property
#     def image(self):
#         return (self.a.get_rgb_image() - .5) + self.b.get_rgb_image()


class TensorImage(Image):
    def __init__(self, tensor, transform=None, decorrelate=True, requires_grad=True):
        super(TensorImage, self).__init__(transform=transform, decorrelate=decorrelate)

        self.tensor = nn.Parameter(tensor, requires_grad=requires_grad)

    @property
    def image(self):
        return self.tensor


def fftfreq2d(w, h):
    import numpy as np
    fy = np.fft.fftfreq(h)[:, None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return torch.from_numpy(np.sqrt(fx * fx + fy * fy)).float()


class FFTImage(Image):
    def __init__(self, shape, sd=0.01, decay_power=1, transform=None, decorrelate=True, requires_grad=True):
        super(FFTImage, self).__init__(transform=transform, decorrelate=decorrelate)

        ch, h, w = shape
        freqs = fftfreq2d(w, h)
        scale = torch.ones(1) / torch.max(freqs, torch.tensor([1. / max(w, h)], dtype=torch.float32)).pow(decay_power)
        self.scale = nn.Parameter(scale * math.sqrt(w * h), requires_grad=False)

        param_size = [ch] + list(freqs.shape) + [2]
        param = torch.randn(param_size) * sd
        self.param = nn.Parameter(param, requires_grad=requires_grad)

        self._shape = shape

    @property
    def image(self):
        ch, h, w = self._shape
        spectrum = self.scale.unsqueeze(0).unsqueeze(3) * self.param
        image = torch.irfft(spectrum, 2)
        image = image[:ch, :h, :w] / 4.0
        return image
