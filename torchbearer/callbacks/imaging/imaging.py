import torchbearer
from torchbearer import Callback

import torch


def _to_file(filename):
    from PIL import Image

    def handler(image, index, _):
        ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(filename.format(index=str(index)))

    return handler


def _to_pyplot(title=None):
    import matplotlib.pyplot as plt

    def handler(image, index, _):
        ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        plt.imshow(ndarr)
        if title is not None:
            plt.title(title.format(index=str(index)))
        plt.show()

    return handler


def _to_tensorboard(name='Image', log_dir='./logs', comment='torchbearer'):
    import torchbearer.callbacks.tensor_board as tb
    import os
    log_dir = os.path.join(log_dir, comment)

    def handler(image, index, state):
        writer = tb.get_writer(log_dir, _to_tensorboard)
        writer.add_image(name.format(index=str(index)), image.clamp(0, 1), state[torchbearer.EPOCH])
        tb.close_writer(log_dir, _to_tensorboard)

    return handler


def _to_visdom(name='Image', log_dir='./logs', comment='torchbearer', visdom_params=None):
    import torchbearer.callbacks.tensor_board as tb
    import os
    log_dir = os.path.join(log_dir, comment)

    def handler(image, index, state):
        writer = tb.get_writer(log_dir, _to_visdom, visdom=True, visdom_params=visdom_params)
        writer.add_image(name.format(index=str(index)) + '_' + str(state[torchbearer.EPOCH]), image.clamp(0, 1), state[torchbearer.EPOCH])
        tb.close_writer(log_dir, _to_visdom)

    return handler


def _cache_images(num_images):
    cache = {'images': None, 'done': False}

    def decorator(fun):
        def step(state):
            if state[torchbearer.BATCH] == 0:
                cache['done'] = False

            if not cache['done']:
                data = fun(state)

                if cache['images'] is None:
                    remaining = num_images if num_images < data.size(0) else data.size(0)

                    cache['images'] = data[:remaining]
                else:
                    remaining = num_images - cache['images'].size(0)

                    if remaining > data.size(0):
                        remaining = data.size(0)

                    cache['images'] = torch.cat((cache['images'], data[:remaining]), dim=0)

                if cache['images'].size(0) >= num_images:
                    res = cache['images']
                    cache['done'] = True
                    cache['images'] = None
                    return res
        return step
    return decorator


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
            for handler, index in self._handlers:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                rng = range(img.size(0)) if index is None else index
                try:
                    for i in rng:
                        handler(img[i], i, state)
                except TypeError:
                    handler(img[rng], rng, state)

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

    def with_handler(self, handler, index=None):
        """Append the given output handler to the list of handlers

        Args:
            handler: A function of image and state which stores the given image in some way
            index (int or list or None): if not None, only apply the handler on this index / list of indices

        Returns:
            ImagingCallback: self
        """
        self._handlers.append((handler, index))
        return self

    def to_file(self, filename, index=None):
        """Send images from this callback to the given file

        Args:
            filename (str): the filename to store the image to
            index (int or list or None): if not None, only apply the handler on this index / list of indices

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_file(filename), index=index)

    def to_pyplot(self, index=None):
        """Show images from this callback with pyplot

        Args:
            index (int or list or None): if not None, only apply the handler on this index / list of indices

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_pyplot(), index=index)

    def to_state(self, keys, index=None):
        """Put images from this callback in state with the given key

        Args:
            keys (StateKey or list[StateKey]): The state key or keys to use for the images
            index (int or list or None): if not None, only apply the handler on this index / list of indices

        Returns:
            ImagingCallback: self
        """
        if str(keys) == keys:
            keys = [keys]

        try:
            _ = (key for key in keys)
        except TypeError:
            keys = [keys]

        def handler(img, i, state):
            state[keys[i]] = img
        return self.with_handler(handler, index=index)

    def to_tensorboard(self, name='Image', log_dir='./logs', comment='torchbearer', index=None):
        """Direct images from this callback to tensorboard with the given parameters

        Args:
            name (str): The name of the image
            log_dir (str): The tensorboard log path for output
            comment (str): Descriptive comment to append to path
            index (int or list or None): if not None, only apply the handler on this index / list of indices

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_tensorboard(name=name, log_dir=log_dir, comment=comment), index=index)

    def to_visdom(self, name='Image', log_dir='./logs', comment='torchbearer', visdom_params=None, index=None):
        """Direct images from this callback to visdom with the given parameters

        Args:
            name (str): The name of the image
            log_dir (str): The visdom log path for output
            comment (str): Descriptive comment to append to path
            visdom_params (:class:`.VisdomParams`): Visdom parameter settings object, uses default if None
            index (int or list or None): if not None, only apply the handler on this index / list of indices

        Returns:
            ImagingCallback: self
        """
        return self.with_handler(_to_visdom(name=name, log_dir=log_dir, comment=comment, visdom_params=visdom_params), index=index)

    def cache(self, num_images):
        """Cache images **before** they are passed to handlers. Once per epoch, a single cache will be returned,
        containing the first `num_images` to be returned.

        Args:
            num_images (int): The number of images to cache

        Returns:
            ImagingCallback: self
        """
        self.on_batch = _cache_images(num_images)(self.on_batch)
        return self

    def make_grid(self, nrow=8, padding=2, normalize=False, norm_range=None, scale_each=False, pad_value=0):
        """Use `torchvision.utils.make_grid` to make a grid of the images being returned by this callback. Recommended
        for use alongside `cache`.

        Args:
            nrow: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
            padding: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
            normalize: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
            norm_range: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
            scale_each: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
            pad_value: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_

        Returns:
            ImagingCallback: self
        """
        import torchvision.utils as utils

        def decorator(func):
            def wrapper(state):
                cache = func(state)
                if cache is not None:
                    return utils.make_grid(cache, nrow=nrow, padding=padding, normalize=normalize, range=norm_range,
                                           scale_each=scale_each, pad_value=pad_value)
            return wrapper

        self.on_batch = decorator(self.on_batch)
        return self


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
    def __init__(self, key, transform=None, decorator=None):
        super(FromState, self).__init__(transform=transform)
        self.key = key

        if decorator is not None:
            self.on_batch = decorator(self.on_batch)

    def on_batch(self, state):
        try:
            return state[self.key]
        except KeyError:
            return None


class CachingImagingCallback(FromState):
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
        super(CachingImagingCallback, self).__init__(key=key, transform=transform, decorator=_cache_images(num_images))

        def decorator(func):
            def wrapper(state):
                res = func(state)
                if res is not None:
                    return self.on_cache(res, state)
            return wrapper

        self.on_batch = decorator(self.on_batch)

    def on_cache(self, cache, state):
        """This method should be implemented by the overriding class to return an image from the cache.

        Args:
            cache (tensor): The collected cache of size (num_images, C, W, H)
            state (dict): The trial state dict

        Returns:
            The processed image
        """
        raise NotImplementedError


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
        super(MakeGrid, self).__init__(transform=transform, num_images=num_images, key=key)
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
