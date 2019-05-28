import torchbearer
from torchbearer import Callback
from torchbearer.callbacks.decorators import once_per_epoch

import torch


def _to_file(filename):
    from PIL import Image

    def handler(image, _):
        ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(filename)
    return handler


def _to_pyplot():
    import matplotlib.pyplot as plt

    def handler(image, _):
        ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        plt.imshow(ndarr)
        plt.show()
    return handler


def _to_tensorboard(name='Image', log_dir='./logs', comment='torchbearer'):
    import torchbearer.callbacks.tensor_board as tb
    import os
    log_dir = os.path.join(log_dir, comment)

    def handler(image, state):
        writer = tb.get_writer(log_dir, _to_tensorboard)
        writer.add_image(name, image.clamp(0, 1), state[torchbearer.EPOCH])
        tb.close_writer(log_dir, _to_tensorboard)
    return handler


def _to_visdom(name='Image', log_dir='./logs', comment='torchbearer', visdom_params=None):
    import torchbearer.callbacks.tensor_board as tb
    import os
    log_dir = os.path.join(log_dir, comment)

    def handler(image, state):
        writer = tb.get_writer(log_dir, _to_visdom, visdom=True, visdom_params=visdom_params)
        writer.add_image(name + str(state[torchbearer.EPOCH]), image.clamp(0, 1), state[torchbearer.EPOCH])
        tb.close_writer(log_dir, _to_visdom)

    return handler


class ImagingCallback(Callback):
    """The :class:`ImagingCallback` provides a generic interface for callbacks which yield images that should be sent to
    a file, tensorboard, visdom etc. without needing bespoke code. This allows the user to easily define custom
    visualisations by only writing the code to produce the image.

    Args:
        transform (callable, optional): A function/transform that  takes in a Tensor and returns a transformed version.
            This will be applied to the image before it is sent to output.
    """
    def __init__(self, transform=None):
        self._handlers = []
        self.transform = (lambda img: img) if transform is None else transform

    def on_batch(self, state):
        raise NotImplementedError

    def process(self, state):
        img = self.on_batch(state)
        if img is not None:
            img = self.transform(img)
            for handler in self._handlers:
                handler(img, state)

    def on_train(self):
        """Process this callback for training batches

        Returns:
            ImagingCallback: self
        """
        _old_step_training = self.on_step_training

        def wrapper(state):
            _old_step_training(state)
            self.process(state)

        self.on_step_training = wrapper
        return self

    def on_val(self):
        """Process this callback for validation batches

        Returns:
            ImagingCallback: self
        """
        _old_step_validation = self.on_step_validation

        def wrapper(state):
            _old_step_validation(state)
            if state[torchbearer.DATA] is torchbearer.VALIDATION_DATA:
                self.process(state)

        self.on_step_validation = wrapper
        return self

    def on_test(self):
        """Process this callback for test batches

        Returns:
            ImagingCallback: self
        """
        _old_step_validation = self.on_step_validation

        def wrapper(state):
            _old_step_validation(state)
            if state[torchbearer.DATA] is torchbearer.TEST_DATA:
                self.process(state)

        self.on_step_validation = wrapper
        return self

    def with_handler(self, handler):
        """Append the given output handler to the list of handlers

        Args:
            handler: A function of image and state which stores the given image in some way

        Returns:
            ImagingCallback: self
        """
        self._handlers.append(handler)
        return self

    def to_file(self, filename):
        """Send images from this callback to the given file

        Args:
            filename (str): the filename to store the image to

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_file(filename))

    def to_pyplot(self):
        """Show images from this callback with pyplot

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_pyplot())

    def to_state(self, key):
        """Put images from this callback in state with the given key

        Args:
            key (StateKey): The state key to use for the image

        Returns:
            ImagingCallback: self
        """
        def handler(img, state):
            state[key] = img
        return self.with_handler(handler)

    def to_tensorboard(self, name='Image', log_dir='./logs', comment='torchbearer'):
        """Direct images from this callback to tensorboard with the given parameters

        Args:
            name (str): The name of the image
            log_dir (str): The tensorboard log path for output
            comment (str): Descriptive comment to append to path

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_tensorboard(name=name, log_dir=log_dir, comment=comment))

    def to_visdom(self, name='Image', log_dir='./logs', comment='torchbearer', visdom_params=None):
        """Direct images from this callback to visdom with the given parameters

        Args:
            name (str): The name of the image
            log_dir (str): The visdom log path for output
            comment (str): Descriptive comment to append to path
            visdom_params (:class:`.VisdomParams`): Visdom parameter settings object, uses default if None

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_visdom(name=name, log_dir=log_dir, comment=comment, visdom_params=visdom_params))


class CachingImagingCallback(ImagingCallback):
    """The :class:`CachingImagingCallback` is an :class:`ImagingCallback` which caches batches of images from the given
    state key up to the required amount before passing this along with state to the implementing class, once per epoch.

    Args:
        key (StateKey): The :class:`.StateKey` containing image data (tensor of size [b, c, w, h])
        transform (callable, optional): A function/transform that  takes in a Tensor and returns a transformed version.
            This will be applied to the image before it is sent to output.
        num_images: The number of images to cache
    """
    def __init__(self,
                 key=torchbearer.INPUT,
                 transform=None,
                 num_images=16):
        super(CachingImagingCallback, self).__init__(transform=transform)
        self.key = key
        self.num_images = num_images

        self._data = None
        self._done = False

    def on_cache(self, cache, state):
        """This method should be implemented by the overriding class to return an image from the cache.

        Args:
            cache (tensor): The collected cache of size (num_images, C, W, H)
            state (dict): The trial state dict

        Returns:
            The processed image
        """
        raise NotImplementedError

    def on_batch(self, state):
        if not self._done:
            data = state[self.key].detach()

            if self._data is None:
                remaining = self.num_images if self.num_images < data.size(0) else data.size(0)

                self._data = data[:remaining]
            else:
                remaining = self.num_images - self._data.size(0)

                if remaining > data.size(0):
                    remaining = data.size(0)

                self._data = torch.cat((self._data, data[:remaining]), dim=0)

            if self._data.size(0) >= self.num_images:
                image = self.on_cache(self._data, state)
                self._done = True
                self._data = None
                return image

    def on_end_epoch(self, state):
        super(CachingImagingCallback, self).on_end_epoch(state)
        self._done = False


class MakeGrid(CachingImagingCallback):
    """The :class:`MakeGrid` callback is a :class:`CachingImagingCallback` which calls make grid on the cache with the
    provided parameters.

    Args:
        key (StateKey): The :class:`.StateKey` containing image data (tensor of size [b, c, w, h])
        transform (callable, optional): A function/transform that  takes in a Tensor and returns a transformed version.
            This will be applied to the image before it is sent to output.
        num_images: The number of images to cache
        nrow: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        padding: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        normalize: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        norm_range: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        scale_each: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        pad_value: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
    """
    def __init__(self,
                 key=torchbearer.INPUT,
                 transform=None,
                 num_images=16,
                 nrow=8,
                 padding=2,
                 normalize=False,
                 norm_range=None,
                 scale_each=False,
                 pad_value=0):
        super(MakeGrid, self).__init__(transform=transform)
        self.key = key
        self.num_images = num_images
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_cache(self, cache, state):
        import torchvision.utils as utils
        return utils.make_grid(
            cache,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value
        )


class FromState(ImagingCallback):
    """The :class:`FromState` callback is an :class:`ImagingCallback` which retrieves and image from state when called.
    The number of times the function is called can be controlled with a provided decorator (once_per_epoch, only_if
    etc.)

    Args:
        key (StateKey): The :class:`.StateKey` containing the image (tensor of size [c, w, h])
        transform (callable, optional): A function/transform that  takes in a Tensor and returns a transformed version.
            This will be applied to the image before it is sent to output.
        decorator: A function which will be used to wrap the callback function. once_per_epoch by default
    """
    def __init__(self, key, transform=None, decorator=once_per_epoch):
        super(FromState, self).__init__(transform=transform)
        self.key = key
        self.on_batch = decorator(self.on_batch)

    def on_batch(self, state):
        try:
            return state[self.key]
        except KeyError:
            return None
