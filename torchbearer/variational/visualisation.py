import torch

import torchbearer
import torchbearer.callbacks as c


class LatentWalker(c.Callback):
    def __init__(self, same_image, row_size):
        """
        Args:
            same_image (bool): If True, use the same image for all latent dimension walks. Else each dimension has different image
            row_size (int): Number of images displayed in each row of the grid.
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

    def on_train(self):
        """
        Sets the walker to run during training

        Returns:
            LatentWalker: self
        """
        self.on_step_training = c.once_per_epoch(self._vis)
        return self

    def on_val(self):
        """
        Sets the walker to run during validation

        Returns:
            LatentWalker: self
        """
        self.on_step_validation = c.once_per_epoch(self._vis)
        return self

    def for_space(self, space_id):
        """
        Sets the ID for which latent space to vary when model outputs [latent_space_0, latent_space_1, ...]

        Args:
            space_id (int): ID of the latent space to vary

        Returns:
            LatentWalker: self
        """
        self.variable_space = space_id
        return self

    def for_data(self, data_key):
        """
        Args:
            data_key (:class:`.StateKey`): State key which will contain data to act on

        Returns:
            LatentWalker: self
        """
        self.data_key = data_key
        return self

    def to_key(self, state_key):
        """

        Args:
            state_key (:class:`.StateKey`): State key under which to store result

        Returns:
            LatentWalker: self
        """
        self.store_key = state_key
        return self

    def to_file(self, file):
        """
        Args:
            file (string, pathlib.Path object or file object): File in which result is saved

        Returns:
            LatentWalker: self
        """
        self.file = file
        return self

    def _vis(self, state):
        self.model = state[torchbearer.MODEL]
        self.data = state[self.data_key] if self.data_key is not None else state[torchbearer.X]
        self.dev = state[torchbearer.DEVICE]

        with torch.no_grad():
            result = self.vis(state)

        if self.file is not None:
            self._save_walk(result)
        if self.store_key is not None:
            state[self.store_key] = result

    def vis(self, state):
        """
        Create the tensor of images to be displayed
        """
        raise NotImplementedError

    def _save_walk(self, tensor):
        from torchvision.utils import save_image
        save_image(tensor, self.file, self.row_size, normalize=True, pad_value=1)


class ReconstructionViewer(LatentWalker):
    def __init__(self, row_size=8, recon_key=torchbearer.Y_PRED):
        """
        Latent space walker that just returns the reconstructed images for the batch

        Args:
            row_size (int): Number of images displayed in each row of the grid.
            recon_key (StateKey): :class:`.StateKey` of the reconstructed images
        """
        super(ReconstructionViewer, self).__init__(False, row_size)
        self.recon_key = recon_key

    def vis(self, state):
        data = self.data[:self.row_size]
        recons = state[self.recon_key][:self.row_size]
        return torch.cat([data, recons])


class LinSpaceWalker(LatentWalker):
    def __init__(self, lin_start=-1, lin_end=1, lin_steps=8, dims_to_walk=[0], zero_init=False, same_image=False):
        """
        Latent space walker that explores each dimension linearly from start to end points

        Args:
            lin_start (float): Starting point of linspace
            lin_end (float): End point of linspace
            lin_steps (int): Number of steps to take in linspace
            dims_to_walk (list of int): List of dimensions to walk
            zero_init (bool): If True, dimensions not being walked are 0. Else, they are obtained from encoder
            same_image (bool): If True, use same image for each dimension walked. Else, use different images
        """
        super(LinSpaceWalker, self).__init__(same_image, lin_steps)
        self.dims_to_walk = dims_to_walk
        self.zero_init = zero_init
        self.linspace = torch.linspace(lin_start, lin_end, lin_steps)

    def vis(self, state):
        self.linspace = self.linspace.to(self.dev)
        num_images = self.row_size * len(self.dims_to_walk)
        num_spaces = len(self.model.latent_dims)

        if self.zero_init:
            sample = []
            for i in range(num_spaces):
                sample.append(torch.zeros(num_images, self.model.latent_dims[i], device=self.dev).unsqueeze(1).repeat(1, self.row_size, 1))
        elif self.same_image:
            sample = list(self.model.encode(self.data[0], state))
            for i in range(len(sample)):
                sample[i] = sample[i][0].unsqueeze(0).unsqueeze(1).repeat(sample[i].shape[0], self.row_size, 1)
        else:
            sample = list(self.model.encode(self.data, state))
            for i in range(len(sample)):
                sample[i] = sample[i].unsqueeze(1).repeat(1, self.row_size, 1)

        dims = self.dims_to_walk

        i = 0
        for dim in list(dims):
            sample[self.variable_space][i, :, dim] = self.linspace
            i += 1

        for i in range(num_spaces):
            sample[i] = sample[i].view(-1, self.model.latent_dims[i])[:num_images]

        result = self.model.decode(sample).view(num_images, -1, self.data.shape[-2], self.data.shape[-1])
        return result


class RandomWalker(LatentWalker):
    def __init__(self, var=1, num_images=32, uniform=False, row_size=8):
        """
        Latent space walker that shows random samples from latent space

        Args:
            var (float or torch.Tensor): Variance of random sample
            num_images (int): Number of random images to sample
            uniform (bool): If True, sample uniform distribution [-v, v). If False, sample normal distribution with var v
            row_size (int): Number of images displayed in each row of the grid.
        """
        super(RandomWalker, self).__init__(False, row_size)
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

        Args:
            num_steps (int): Number of steps to take between points
            p1 (torch.Tensor): Batch of codes
            p2 (torch.Tensor): Batch of codes
        """
        super(CodePathWalker, self).__init__(True, num_steps)
        self.p1 = p1
        self.p2 = p2
        self.num_steps = num_steps

    def vis(self, state):
        step_sizes = (self.p1 - self.p2)/(self.num_steps-1)

        codes = torch.zeros(self.p1.shape[0], self.num_steps, self.p1.shape[1]).to(self.dev)
        for i in range(self.num_steps):
            codes[:, i] = self.p1 - step_sizes*i
        codes = codes.view(-1, self.p1.shape[1])

        result = self.model.decode(codes).view(codes.shape[0], -1, self.data.shape[-2], self.data.shape[-1])
        return result


class ImagePathWalker(CodePathWalker):
    def __init__(self, num_steps, im1, im2):
        """
        Latent space walker that walks between two specified images im1 and im2

        Args:
            num_steps (int): Number of steps to take between points
            im1 (torch.Tensor): Batch of images
            im2 (torch.Tensor): Batch of images
        """
        super(ImagePathWalker, self).__init__(num_steps, None, None)
        self.im1, self.im2 = im1, im2

    def vis(self, state):
        self.p1 = self.model.encode(self.im1.to(self.dev), state)
        self.p2 = self.model.encode(self.im2.to(self.dev), state)

        return super(ImagePathWalker, self).vis(state)
