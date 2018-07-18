import copy

import torchvision.utils as utils

from tensorboardX import SummaryWriter

import torch.nn.functional as F

import torchbearer

from torchbearer.callbacks import Callback

import os

import torch


class TensorBoard(Callback):
    """The TensorBoard callback is used to write metric graphs to tensorboard. Requires the TensorboardX library for
    python.
    """

    def __init__(self, log_dir='./logs',
                 write_graph=True,
                 write_batch_metrics=False,
                 batch_step_size=10,
                 write_epoch_metrics=True,
                 comment='torchbearer'):
        """TensorBoard callback which writes metrics to the given log directory.

        :param log_dir: The tensorboard log path for output
        :type log_dir: str
        :param write_graph: If True, the model graph will be written using the TensorboardX library
        :type write_graph: bool
        :param write_batch_metrics: If True, batch metrics will be written
        :type write_batch_metrics: bool
        :param batch_step_size: The step size to use when writing batch metrics, make this larger to reduce latency
        :type batch_step_size: int
        :param write_epoch_metrics: If True, metrics from the end of the epoch will be written
        :type write_epoch_metrics: True
        :param comment: Descriptive comment to append to path
        :type comment: str
        """
        super(TensorBoard, self).__init__()

        self.log_dir = log_dir
        self.write_graph = write_graph
        self.write_batch_metrics = write_batch_metrics
        self.batch_step_size = batch_step_size
        self.write_epoch_metrics = write_epoch_metrics
        self.comment = comment

        if self.write_graph:
            def handle_graph(state):
                dummy = torch.rand(state[torchbearer.X].size(), requires_grad=False)
                model = copy.deepcopy(state[torchbearer.MODEL]).to('cpu')
                self._writer.add_graph(model, (dummy, ))
                self._handle_graph = lambda _: ...
            self._handle_graph = handle_graph
        else:
            self._handle_graph = lambda _: ...

        self._writer = None
        self._batch_writer = None

    def on_start(self, state):
        self.log_dir = os.path.join(self.log_dir, state[torchbearer.MODEL].__class__.__name__ + '_' + self.comment)
        self._writer = SummaryWriter(log_dir=self.log_dir)

    def on_start_epoch(self, state):
        if self.write_batch_metrics:
            log_dir = os.path.join(self.log_dir, 'epoch-' + str(state[torchbearer.EPOCH]))
            self._batch_writer = SummaryWriter(log_dir=log_dir)

    def on_end(self, state):
        self._writer.close()

    def on_sample(self, state):
        self._handle_graph(state)

    def on_step_training(self, state):
        if self.write_batch_metrics and state[torchbearer.BATCH] % self.batch_step_size == 0:
            for metric in state[torchbearer.METRICS]:
                self._batch_writer.add_scalar('batch/' + metric, state[torchbearer.METRICS][metric], state[torchbearer.BATCH])

    def on_step_validation(self, state):
        if self.write_batch_metrics and state[torchbearer.BATCH] % self.batch_step_size == 0:
            for metric in state[torchbearer.METRICS]:
                self._batch_writer.add_scalar('batch/' + metric, state[torchbearer.METRICS][metric], state[torchbearer.BATCH])

    def on_end_epoch(self, state):
        if self.write_batch_metrics:
            self._batch_writer.close()

        if self.write_epoch_metrics:
            for metric in state[torchbearer.METRICS]:
                self._writer.add_scalar('epoch/' + metric, state[torchbearer.METRICS][metric], state[torchbearer.EPOCH])


class TensorBoardImages(Callback):
    """The TensorBoardImages callback will write a selection of images from the validation pass to tensorboard using the
    TensorboardX library and torchvision.utils.make_grid
    """

    def __init__(self, log_dir='./logs',
                 comment='torchbearer',
                 name='Image',
                 key=torchbearer.Y_PRED,
                 write_each_epoch=True,
                 num_images=16,
                 nrow=8,
                 padding=2,
                 normalize=False,
                 range=None,
                 scale_each=False,
                 pad_value=0):
        """Create TensorBoardImages callback which writes images from the given key to the given path. Full name of
        image sub directory will be model name + _ + comment.

        :param log_dir: The tensorboard log path for output
        :type log_dir: str
        :param comment: Descriptive comment to append to path
        :type comment: str
        :param name: The name of the image
        :type name: str
        :param key: The key in state containing image data (tensor of size [c, w, h] or [b, c, w, h])
        :type key: str
        :param write_each_epoch: If True, write data on every epoch, else write only for the first epoch.
        :type write_each_epoch: bool
        :param num_images: The number of images to write
        :type num_images: int
        :param nrow: See `torchvision.utils.make_grid https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid`
        :param padding: See `torchvision.utils.make_grid https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid`
        :param normalize: See `torchvision.utils.make_grid https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid`
        :param range: See `torchvision.utils.make_grid https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid`
        :param scale_each: See `torchvision.utils.make_grid https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid`
        :param pad_value: See `torchvision.utils.make_grid https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid`
        """
        self.log_dir = log_dir
        self.comment = comment
        self.name = name
        self.key = key
        self.write_each_epoch = write_each_epoch
        self.num_images = num_images
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.range = range
        self.scale_each = scale_each
        self.pad_value = pad_value

        self._writer = None
        self._data = None
        self.done = False

    def on_start(self, state):
        log_dir = os.path.join(self.log_dir, state[torchbearer.MODEL].__class__.__name__ + '_' + self.comment)
        self._writer = SummaryWriter(log_dir=log_dir)

    def on_step_validation(self, state):
        if not self.done:
            data = state[self.key].clone()

            if len(data.size()) == 3:
                data = data.unsqueeze(1)

            if self._data is None:
                remaining = self.num_images if self.num_images < data.size(0) else data.size(0)

                self._data = data[:remaining].to('cpu')
            else:
                remaining = self.num_images - self._data.size(0)

                if remaining > data.size(0):
                    remaining = data.size(0)

                self._data = torch.cat((self._data, data[:remaining].to('cpu')), dim=0)

            if self._data.size(0) >= self.num_images:
                image = utils.make_grid(
                    self._data,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    range=self.range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value
                )
                self._writer.add_image(self.name, image, state[torchbearer.EPOCH])
                self.done = True
                self._data = None

    def on_end_epoch(self, state):
        if self.write_each_epoch:
            self.done = False

    def on_end(self, state):
        self._writer.close()


class TensorBoardProjector(Callback):
    """The TensorBoardProjector callback is used to write images from the validation pass to Tensorboard using the
    TensorboardX library.
    """

    def __init__(self, log_dir='./logs',
                 comment='torchbearer',
                 num_images=100,
                 avg_pool_size=1,
                 avg_data_channels=True,
                 write_data=True,
                 write_features=True,
                 features_key=torchbearer.Y_PRED):
        """Construct a TensorBoardProjector callback which writes images to the given directory and, if required,
        associated features.

        :param log_dir: The tensorboard log path for output
        :type log_dir: str
        :param comment: Descriptive comment to append to path
        :type comment: str
        :param num_images: The number of images to write
        :type num_images: int
        :param avg_pool_size: Size of the average pool to perform on the image. This is recommended to reduce the
        overall image sizes and improve latency
        :type avg_pool_size: int
        :param avg_data_channels: If True, the image data will be averaged in the channel dimension
        :type avg_data_channels: bool
        :param write_data: If True, the raw data will be written as an embedding
        :type write_data: bool
        :param write_features: If True, the image features will be written as an embedding
        :type write_features: bool
        :param features_key: The key in state to use for the embedding. Typically model output but can be used to show
        features from any layer of the model.
        :type features_key: str
        """
        self.log_dir = log_dir
        self.comment = comment
        self.num_images = num_images
        self.avg_pool_size = avg_pool_size
        self.avg_data_channels = avg_data_channels
        self.write_data = write_data
        self.write_features = write_features
        self.features_key = features_key

        self._writer = None

        self.done = False

    def on_start(self, state):
        log_dir = os.path.join(self.log_dir, state[torchbearer.MODEL].__class__.__name__ + '_' + self.comment)
        self._writer = SummaryWriter(log_dir=log_dir)

    def on_step_validation(self, state):
        if not self.done:
            x = state[torchbearer.X].data.clone()

            if len(x.size()) == 3:
                x = x.unsqueeze(1)

            x = F.avg_pool2d(x, self.avg_pool_size).data

            data = None

            if state[torchbearer.EPOCH] == 0 and self.write_data:
                if self.avg_data_channels:
                    data = torch.mean(x, 1)
                else:
                    data = x

                data = data.view(data.size(0), -1)

            feature = None

            if self.write_features:
                feature = state[self.features_key].data.clone()
                feature = feature.view(feature.size(0), -1)

            label = state[torchbearer.Y_TRUE].data.clone()

            if state[torchbearer.BATCH] == 0:
                remaining = self.num_images if self.num_images < label.size(0) else label.size(0)

                self._images = x[:remaining].to('cpu')
                self._labels = label[:remaining].to('cpu')

                if data is not None:
                    self._data = data[:remaining].to('cpu')

                if feature is not None:
                    self._features = feature[:remaining].to('cpu')
            else:
                remaining = self.num_images - self._labels.size(0)

                if remaining > label.size(0):
                    remaining = label.size(0)

                self._images = torch.cat((self._images, x[:remaining].to('cpu')), dim=0)
                self._labels = torch.cat((self._labels, label[:remaining].to('cpu')), dim=0)

                if data is not None:
                    self._data = torch.cat((self._data, data[:remaining].to('cpu')), dim=0)

                if feature is not None:
                    self._features = torch.cat((self._features, feature[:remaining].to('cpu')), dim=0)

            if self._labels.size(0) >= self.num_images:
                if state[torchbearer.EPOCH] == 0 and self.write_data:
                    self._writer.add_embedding(self._data, metadata=self._labels, label_img=self._images, tag='data', global_step=-1)
                if self.write_features:
                    self._writer.add_embedding(self._features, metadata=self._labels, label_img=self._images, tag='features', global_step=state[torchbearer.EPOCH])
                self.done = True

    def on_end_epoch(self, state):
        if self.write_features:
            self.done = False

    def on_end(self, state):
        self._writer.close()
