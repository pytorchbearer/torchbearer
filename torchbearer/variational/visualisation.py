import numpy as np
from abc import ABC, abstractmethod

import torch
from torchvision.utils import save_image

import torchbearer as tb
import torchbearer.callbacks as c
from torchbearer.trial import fluent


class LatentWalker(ABC, c.Callback):
    def __init__(self, same_image, row_size):
        """
        :param same_image: If True, use the same image for all latent dimension walks. Else each dimension has different image
        :type same_image: bool
        """
        super(LatentWalker, self).__init__()
        self.data_key = None
        self.same_image = same_image
        self.row_size = row_size
        self.model = None
        self.data = None
        self.dev = None
        self.file = None
        self.store_key = None
        self.variable_space = 0

    @fluent
    def on_train(self):
        """
        Sets the walker to run during training
        """
        self.on_step_training = c.once_per_epoch(self._vis)

    @fluent
    def on_val(self):
        """
        Sets the walker to run during validation
        """
        self.on_step_validation = c.once_per_epoch(self._vis)

    @fluent
    def for_space(self, space_id):
        """
        Sets the ID for which latent space to vary when model outputs [latent_space_0, latent_space_1, ...]
        :param space_id: ID of the latent space to vary
        :return:
        """
        self.variable_space = space_id

    @fluent
    def data(self, data_key):
        """
        :param data_key: State key which will contain data to act on
        :type data_key: tb.StateKey
        """
        self.data_key = data_key

    @fluent
    def to_key(self, state_key):
        """
        :param state_key: State key under which to store result
        :type state_key: tb.StateKey
        """
        self.store_key = state_key

    @fluent
    def to_file(self, file):
        """
        :param file: File to save result to
        """
        self.file = file

    def _vis(self, state):
        self.model = state[tb.MODEL]
        self.data = state[self.data_key] if self.data_key is not None else state[tb.X]
        self.dev = state[tb.DEVICE]

        with torch.no_grad():
            result = self.vis(state)

        if self.file is not None:
            self._save_walk(result)
        if self.store_key is not None:
            state[self.store_key] = result

    @abstractmethod
    def vis(self, state):
        """
        Create the tensor of images to be displayed
        """
        return

    def _save_walk(self, tensor):
        save_image(tensor, self.file, self.row_size, normalize=True, pad_value=1)


class ReconstructionViewer(LatentWalker):
    def __init__(self, row_size=8, recon_key=tb.Y_PRED):
        super(ReconstructionViewer, self).__init__(False, row_size)
        self.recon_key = recon_key

    def vis(self, state):
        data = self.data[:self.row_size]
        recons = state[self.recon_key][:self.row_size]
        return torch.cat([data, recons])


class LinSpaceWalker(LatentWalker):
    def __init__(self, lin_start=-1, lin_end=1, lin_steps=8, num_dims=10, zero_init=False, same_image=True):
        """
        Latent space walker that explores each dimension linearly from start to end points
        :param lin_start: Starting point of linspace
        :param lin_end: End point of linspace
        :param lin_steps: Number of steps to take in linspace
        :param num_dims: Number of dimensions to walk
        :param zero_init: If True, dimensions not being walked are 0. Else, they are obtained from encoder
        :param same_image: If True, use same image for each dimension walked. Else, use different images
        """
        super(LinSpaceWalker, self).__init__(same_image, lin_steps)
        self.num_dims = num_dims
        self.zero_init = zero_init
        self.linspace = torch.linspace(lin_start, lin_end, lin_steps)

    def vis(self, state):
        self.linspace = self.linspace.to(self.dev)
        num_images = self.row_size * self.num_dims
        num_spaces = len(self.model.latent_dims)

        if self.zero_init:
            sample = []
            for i in range(num_spaces):
                sample.append(torch.zeros(num_images, self.model.latent_dims[i], device=self.dev).unsqueeze(1).repeat(1, self.row_size, 1))
        else:
            sample = list(self.model.encode(self.data, state))
            for i in range(len(sample)):
                sample[i] = sample[i].unsqueeze(1).repeat(1, self.row_size, 1)

        dims = np.random.permutation(sample[self.variable_space].shape[-1])[:self.num_dims]

        for dim in list(dims):
            sample[self.variable_space][dim, :, dim] = self.linspace

        for i in range(num_spaces):
            sample[i] = sample[i].view(-1, self.model.latent_dims[i])[:num_images]

        result = self.model.decode(sample).view(sample[self.variable_space].shape[0], -1, self.data.shape[-2], self.data.shape[-1])
        return result


class RandomWalker(LatentWalker):
    def __init__(self, var=1, num_images=32, uniform=False, same_image=True, row_size=8):
        """
        Latent space walker that shows random samples from latent space

        :param var: Variance of random sample
        :param num_images: Number of random images to sample
        :param uniform: If True, sample uniform distribution [-v, v). If False, sample normal distribution with var v
        :param same_image:
        :param row_size:
        """
        super(RandomWalker, self).__init__(same_image, row_size)
        self.num_images = num_images
        self.uniform = uniform
        self.var = var

    def vis(self, state):
        num_spaces = len(self.model.latent_dims)

        sample = []
        for i in range(num_spaces):
            sample.append(torch.zeros(self.num_images, self.model.latent_dims[i], device=self.dev))

        if self.uniform:
            sample[self.variable_space] = (torch.rand(self.num_images, self.model.latent_dims[self.variable_space], device=self.dev)*2-1)*self.var
        else:
            sample[self.variable_space] = (torch.randn(self.num_images, self.model.latent_dims[self.variable_space], device=self.dev)*2-1)*self.var

        result = self.model.decode(sample).view(sample[0].shape[0], -1, self.data.shape[-2], self.data.shape[-1])
        return result


class CodePathWalker(LatentWalker):
    def __init__(self, num_steps, p1, p2):
        """
        Latent space walker that walks between two specified codes p1 and p2
        :param num_steps: Number of steps to take between points
        :param p1: Batch of codes
        :param p2: Batch of codes
        """
        super().__init__(True, num_steps)
        self.p1 = p1
        self.p2 = p2
        self.num_steps = num_steps

    def vis(self, state):
        step_sizes = (self.p1 - self.p2)/(self.num_steps-1)

        codes = torch.zeros(self.p1.shape[0], self.num_steps, self.p1.shape[1]).to(self.dev)
        for i in range(self.num_steps):
            codes[:, i] = self.p1 - step_sizes*i

        result = self.model.decode(codes).view(codes.shape[0]*self.num_steps, -1, self.data.shape[-2], self.data.shape[-1])
        return result


class ImagePathWalker(CodePathWalker):
    """
    Latent space walker that walks between two specified images im1 and im2
    :param num_steps: Number of steps to take between points
    :param im1: Batch of images
    :param im2: Batch of images
    """
    def __init__(self, num_steps, im1, im2):
        super().__init__(num_steps, None, None)
        self.im1, self.im2 = im1, im2

    def vis(self, state):
        if len(self.im1.shape) == 1:
            self.im1.unsqueeze(0)
            self.im2.unsqueeze(0)

        self.p1 = self.model.encode(self.im1.to(self.dev), state)
        self.p2 = self.model.encode(self.im2.to(self.dev), state)

        return super().vis(state)
