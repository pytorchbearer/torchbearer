import copy
import os
import warnings

import torch
import torch.nn.functional as F

import torchbearer
from torchbearer.callbacks import Callback

__writers__ = dict()


class VisdomParams:
    """
    Class to hold visdom client arguments. Modify member variables before initialising tensorboard callbacks for custom
    arguments. See: `visdom <https://github.com/facebookresearch/visdom#visdom-arguments-python-only>`_
    """
    SERVER = 'http://localhost'
    ENDPOINT = 'events'
    PORT = 8097
    IPV6 = True
    HTTP_PROXY_HOST = None
    HTTP_PROXY_PORT = None
    ENV = 'main'
    SEND = True
    RAISE_EXCEPTIONS = None
    USE_INCOMING_SOCKET = True
    LOG_TO_FILENAME = None

    def __to_dict__(self):
        base_params = {e.lower(): VisdomParams.__dict__[e] for e in VisdomParams.__dict__ if '__' not in e}
        new_params = {e.lower(): self.__dict__[e] for e in self.__dict__ if '__' not in e}
        base_params.update(new_params)
        return base_params


def get_writer(log_dir, logger, visdom=False, visdom_params=None):
    """
    Get the writer assigned to the given log directory.
    If the writer doesn't exist it will be created, and a reference to the logger added.

    Args:
        log_dir (str): the log directory
        logger: the object requesting the writer. That object should call `close_writer` when its finished
        visdom (bool): if true VisdomWriter is returned instead of tensorboard SummaryWriter
        visdom_params (VisdomParams): Visdom parameter settings object, uses default if None

    Returns:
        the `SummaryWriter` or `VisdomWriter` object
    """
    import tensorboardX
    from tensorboardX import SummaryWriter
    import sys
    import errno
    kwargs = {}
    if sys.version_info[0] >= 3:
        kwargs['exist_ok'] = True

    writer_key = 'writer'
    if visdom:
        writer_key = 'writer_visdom'

    if log_dir not in __writers__ or writer_key not in __writers__[log_dir]:
        if visdom:
            w = tensorboardX.torchvis.VisdomWriter()
            from visdom import Visdom
            try:
                os.makedirs(log_dir, **kwargs)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(log_dir):
                    pass
                else:
                    raise exc
            if visdom_params is None:
                visdom_params = VisdomParams()
                visdom_params.LOG_TO_FILENAME = os.path.join(log_dir, 'log.log')
            w.vis = Visdom(**visdom_params.__to_dict__())
        else:
            w = SummaryWriter(log_dir=log_dir)
        __writers__[log_dir] = {writer_key: w, 'references': set()}

    __writers__[log_dir]['references'].add(logger)
    return __writers__[log_dir][writer_key]


def close_writer(log_dir, logger):
    """
    Decrement the reference count for a writer belonging to a specific log directory.
    If the reference count gets to zero, the writer will be closed and removed.

    Args:
        log_dir (str): the log directory
        logger: the object releasing the writer
    """
    if log_dir in __writers__:
        __writers__[log_dir]['references'].discard(logger)

        if len(__writers__[log_dir]['references']) is 0:
            if 'writer' in __writers__[log_dir]:
                __writers__[log_dir]['writer'].close()

            if 'writer_visdom' in __writers__[log_dir]:
                __writers__[log_dir]['writer_visdom'].close()

        del __writers__[log_dir]


class AbstractTensorBoard(Callback):
    """TensorBoard callback which writes metrics to the given log directory. Requires the TensorboardX library for python.

    Args:
        log_dir (str): The tensorboard log path for output
        comment (str): Descriptive comment to append to path
        visdom (bool): If true, log to visdom instead of tensorboard
        visdom_params (VisdomParams): Visdom parameter settings object, uses default if None

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: PyTorch model
    """

    def __init__(self, log_dir='./logs',
                 comment='torchbearer', visdom=False, visdom_params=None):
        super(AbstractTensorBoard, self).__init__()

        self.raw_log_dir = log_dir
        self.log_dir = log_dir
        self.comment = comment
        self.writer = None
        self.visdom = visdom
        self.visdom_params = visdom_params

    def get_writer(self, log_dir=None, visdom=False, visdom_params=None):
        """
        Get a SummaryWriter for the given directory (or the default writer if the directory is not given).
        If you are getting a `SummaryWriter` for a custom directory, it is your responsibility to close
        it using `close_writer`.

        Args:
            log_dir (str): the (optional) directory
            visdom (bool): If true, return VisdomWriter, if false return tensorboard SummaryWriter
            visdom_params (VisdomParams): Visdom parameter settings object, uses default if None

        Returns:
            the `SummaryWriter` or `VisdomWriter`
        """
        if log_dir is None:
            self.writer = get_writer(self.log_dir, self, visdom=visdom, visdom_params=visdom_params)
            return self.writer
        else:
            return get_writer(log_dir, self, visdom=visdom, visdom_params=visdom_params)

    def close_writer(self, log_dir=None):
        """
        Decrement the reference count for a writer belonging to the given log directory
        (or the default writer if the directory is not given). If the reference count gets to zero,
        the writer will be closed and removed.

        Args:
            log_dir (str): the (optional) directory
        """
        if log_dir is None:
            close_writer(self.log_dir, self)
        else:
            close_writer(log_dir, self)

    def on_start(self, state):
        self.log_dir = os.path.join(self.log_dir, state[torchbearer.MODEL].__class__.__name__ + '_' + self.comment)
        self.writer = self.get_writer(visdom=self.visdom, visdom_params=self.visdom_params)

    @staticmethod
    def add_metric(add_fn, tag, metric, *args, **kwargs):
        """ Static method that recurses through `metric` until the `add_fn` can be applied. Useful when metric is an
        iterable of tensors so that the tensors can  all be passed to an `add_fn` such as writer.add_scalar.
        For example, if passed `metric` as [[A, B], [C, ], D, {'E': E}] then `add_fn` would be called on A, B, C, D and
        E and the respective tags (with base tag 'met') would be: met_0_0, met_0_1, met_1_0, met_2, met_E. Throws a
        warning if `add_fn` fails to parse a metric.

        Args:
            add_fn: Function to be called to log a metric, e.g. SummaryWriter.add_scalar
            tag: Tag under which to log the metric
            metric: Iterable of metrics.
            *args: Args for `add_fn`
            **kwargs: Keyword args for `add_fn`

        Returns:

        """
        try:
            add_fn(tag, metric, *args, **kwargs)
        except NotImplementedError:
            try:
                for key, met in enumerate(metric):
                    if isinstance(metric, dict):
                        key, met = met, metric[met]

                    AbstractTensorBoard.add_metric(add_fn, tag+'_{}'.format(key), met, *args, **kwargs)
            except TypeError as e:
                warnings.warn('Failed to log metric to tensorboard with error: {}'.format(e))
        except Exception as e:
            warnings.warn('Failed to log metric to tensorboard with error: {}'.format(e))

    def on_end(self, state):
        self.close_writer()


class TensorBoard(AbstractTensorBoard):
    """TensorBoard callback which writes metrics to the given log directory. Requires the TensorboardX library for python.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import TensorBoard
        >>> import datetime
        >>> current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # Callback that will log to tensorboard under "(model name)_(current time)"
        >>> tb = TensorBoard(log_dir='./logs', write_graph=False, comment=current_time)
        # Trial that will run the callback and log accuracy and loss metrics
        >>> t = Trial(None, callbacks=[tb], metrics=['acc', 'loss'])

    Args:
        log_dir (str): The tensorboard log path for output
        write_graph (bool): If True, the model graph will be written using the TensorboardX library
        write_batch_metrics (bool): If True, batch metrics will be written
        batch_step_size (int): The step size to use when writing batch metrics, make this larger to reduce latency
        write_epoch_metrics (bool): If True, metrics from the end of the epoch will be written
        comment (str): Descriptive comment to append to path
        visdom (bool): If true, log to visdom instead of tensorboard
        visdom_params (VisdomParams): Visdom parameter settings object, uses default if None

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: PyTorch model
        - :attr:`torchbearer.state.EPOCH`: State should have the current epoch stored
        - :attr:`torchbearer.state.X`: State should have the current data stored if a model graph is to be built
        - :attr:`torchbearer.state.BATCH`: State should have the current batch number stored if logging batch metrics
        - :attr:`torchbearer.state.TRAIN_STEPS`: State should have the number of training steps stored
        - :attr:`torchbearer.state.METRICS`: State should have a dictionary of metrics stored
    """

    def __init__(self, log_dir='./logs',
                 write_graph=True,
                 write_batch_metrics=False,
                 batch_step_size=10,
                 write_epoch_metrics=True,
                 comment='torchbearer',
                 visdom=False,
                 visdom_params=None):
        super(TensorBoard, self).__init__(log_dir, comment, visdom, visdom_params)

        self.write_graph = write_graph
        self.write_batch_metrics = write_batch_metrics
        self.batch_step_size = batch_step_size
        self.write_epoch_metrics = write_epoch_metrics
        self.visdom = visdom

        if self.write_graph and not visdom:
            def handle_graph(state):
                dummy = torch.rand(state[torchbearer.X].size(), requires_grad=False)
                model = copy.deepcopy(state[torchbearer.MODEL]).to('cpu')
                self.writer.add_graph(model, (dummy,))
                self._handle_graph = lambda _: None

            self._handle_graph = handle_graph
        else:
            self._handle_graph = lambda _: None

        self.batch_log_dir = None
        self.batch_writer = None

    def on_start_epoch(self, state):
        if self.write_batch_metrics:
            if self.visdom:
                self.batch_log_dir = os.path.join(self.log_dir, 'epoch/')
            else:
                self.batch_log_dir = os.path.join(self.log_dir, 'epoch-' + str(state[torchbearer.EPOCH]))
            self.batch_writer = self.get_writer(self.batch_log_dir, visdom=self.visdom)

    def on_sample(self, state):
        self._handle_graph(state)

    def on_step_training(self, state):
        if self.write_batch_metrics and state[torchbearer.BATCH] % self.batch_step_size == 0:
            for metric in state[torchbearer.METRICS]:
                if self.visdom:
                    self.add_metric(self.batch_writer.add_scalar, metric, state[torchbearer.METRICS][metric],
                                                 state[torchbearer.EPOCH] * state[torchbearer.TRAIN_STEPS] + state[
                                                     torchbearer.BATCH], main_tag='batch')
                else:
                    self.add_metric(self.batch_writer.add_scalar, 'batch/' + metric, state[torchbearer.METRICS][metric], state[torchbearer.BATCH])

    def on_step_validation(self, state):
        if self.write_batch_metrics and state[torchbearer.BATCH] % self.batch_step_size == 0:
            for metric in state[torchbearer.METRICS]:
                if self.visdom:
                    self.add_metric(self.batch_writer.add_scalar, metric, state[torchbearer.METRICS][metric],
                                                 state[torchbearer.EPOCH] * state[torchbearer.TRAIN_STEPS] + state[
                                                     torchbearer.BATCH], main_tag='batch')
                else:
                    self.add_metric(self.batch_writer.add_scalar, 'batch/' + metric, state[torchbearer.METRICS][metric], state[torchbearer.BATCH])

    def on_end_epoch(self, state):
        if self.write_batch_metrics and not self.visdom:
            self.close_writer(self.batch_log_dir)

        if self.write_epoch_metrics:
            for metric in state[torchbearer.METRICS]:
                if self.visdom:
                    self.add_metric(self.writer.add_scalar, metric, state[torchbearer.METRICS][metric], state[torchbearer.EPOCH],
                                           main_tag='epoch')
                else:
                    self.add_metric(self.writer.add_scalar, 'epoch/' + metric, state[torchbearer.METRICS][metric], state[torchbearer.EPOCH])

    def on_end(self, state):
        super(TensorBoard, self).on_end(state)
        if self.write_batch_metrics and self.visdom:
            self.close_writer(self.batch_log_dir)


class TensorBoardText(AbstractTensorBoard):
    """TensorBoard callback which writes metrics as text to the given log directory. Requires the TensorboardX library
    for python.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import TensorBoardText
        >>> import datetime
        >>> current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # Callback that will log to tensorboard under "(model name)_(current time)"
        >>> tb = TensorBoardText(comment=current_time)
        # Trial that will run the callback and log accuracy and loss metrics as text to tensorboard
        >>> t = Trial(None, callbacks=[tb], metrics=['acc', 'loss'])

    Args:
        log_dir (str): The tensorboard log path for output
        write_epoch_metrics (bool): If True, metrics from the end of the epoch will be written
        log_trial_summary (bool): If True logs a string summary of the Trial
        batch_step_size (int): The step size to use when writing batch metrics, make this larger to reduce latency
        comment (str): Descriptive comment to append to path
        visdom (bool): If true, log to visdom instead of tensorboard
        visdom_params (VisdomParams): Visdom parameter settings object, uses default if None

    State Requirements:
        - :attr:`torchbearer.state.SELF`: The :attr:`torchbearer.Trial` running this callback
        - :attr:`torchbearer.state.EPOCH`: State should have the current epoch stored
        - :attr:`torchbearer.state.BATCH`: State should have the current batch number stored if logging batch metrics
        - :attr:`torchbearer.state.METRICS`: State should have a dictionary of metrics stored
    """

    def __init__(self, log_dir='./logs',
                 write_epoch_metrics=True,
                 write_batch_metrics=False,
                 log_trial_summary=True,
                 batch_step_size=100,
                 comment='torchbearer',
                 visdom=False,
                 visdom_params=None):
        super(TensorBoardText, self).__init__(log_dir, comment, visdom, visdom_params)
        self.write_epoch_metrics = write_epoch_metrics
        self.write_batch_metrics = write_batch_metrics
        self.log_trial_summary = log_trial_summary
        self.batch_step_size = batch_step_size
        self.visdom = visdom
        self.batch_log_dir = None
        self.batch_writer = None
        self.logged_summary = False

    @staticmethod
    def table_formatter(string):
        table = '<table><th>Metric</th><th>Value</th>'
        string = string.replace('{', '').replace('}', '').replace("'", "")  # TODO: Replace this with single pass regex

        def cell(string):
            return '<td>' + string + '</td>'

        def row(string):
            return '<tr>' + string + '</tr>'

        metrics = string.split(',')
        for _, metric in enumerate(metrics):
            items = metric.split(':')
            name, value = items[0], items[1]
            table = table + row(cell(name) + cell(value))

        return table + '</table>'

    def on_start(self, state):
        super(TensorBoardText, self).on_start(state)
        if self.log_trial_summary and not self.logged_summary:
            self.logged_summary = True
            self.writer.add_text('trial', str(state[torchbearer.SELF]).replace('\n', '\n \n'), 1)

    def on_start_epoch(self, state):
        if self.write_batch_metrics:
            if self.visdom:
                self.batch_log_dir = os.path.join(self.log_dir, 'epoch/')
                batch_params = self.visdom_params if self.visdom_params is not None else VisdomParams()
                batch_params.ENV = batch_params.ENV + '-batch'
                self.batch_writer = self.get_writer(self.batch_log_dir, visdom=self.visdom, visdom_params=batch_params)
            else:
                self.batch_log_dir = os.path.join(self.log_dir, 'epoch-' + str(state[torchbearer.EPOCH]))
                self.batch_writer = self.get_writer(self.batch_log_dir)

    def on_step_training(self, state):
        if self.write_batch_metrics and state[torchbearer.BATCH] % self.batch_step_size == 0:
            if self.visdom:
                self.batch_writer.add_text('batch', '<h3>Epoch {} - Batch {}</h3>'.format(state[torchbearer.EPOCH], state[torchbearer.BATCH])+self.table_formatter(str(state[torchbearer.METRICS])), 1)
            else:
                self.batch_writer.add_text('batch', self.table_formatter(str(state[torchbearer.METRICS])), state[torchbearer.BATCH])

    def on_end_epoch(self, state):
        if self.write_epoch_metrics:
            if self.visdom:
                self.writer.add_text('epoch', '<h4>Epoch {}</h4>'.format(state[torchbearer.EPOCH])+self.table_formatter(str(state[torchbearer.METRICS])), 1)
            else:
                self.writer.add_text('epoch', self.table_formatter(str(state[torchbearer.METRICS])), state[torchbearer.EPOCH])

    def on_end(self, state):
        super(TensorBoardText, self).on_end(state)
        if self.write_batch_metrics and self.visdom:
            self.close_writer(self.batch_log_dir)


class TensorBoardImages(AbstractTensorBoard):
    """The TensorBoardImages callback will write a selection of images from the validation pass to tensorboard using the
    TensorboardX library and torchvision.utils.make_grid (requires torchvision). Images are selected from the given key and saved to the given
    path. Full name of image sub directory will be model name + _ + comment.

    Example: ::

        >>> from torchbearer import Trial, state_key
        >>> from torchbearer.callbacks import TensorBoardImages
        >>> import datetime
        >>> current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        >>> IMAGE_KEY = state_key('image_key')

        >>> # Callback that will log to tensorboard under "(model name)_(current time)"
        >>> tb = TensorBoardImages(comment=current_time, name='test_image', key=IMAGE_KEY)
        >>> # Trial that will run log to tensorboard images stored under IMAGE_KEY
        >>> t = Trial(None, callbacks=[tb], metrics=['acc', 'loss'])

    Args:
        log_dir (str): The tensorboard log path for output
        comment (str): Descriptive comment to append to path
        name (str): The name of the image
        key (StateKey): The key in state containing image data (tensor of size [c, w, h] or [b, c, w, h])
        write_each_epoch (bool): If True, write data on every epoch, else write only for the first epoch.
        num_images (int): The number of images to write
        nrow: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        padding: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        normalize: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        norm_range: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        scale_each: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        pad_value: See `torchvision.utils.make_grid <https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid>`_
        visdom (bool): If true, log to visdom instead of tensorboard
        visdom_params (VisdomParams): Visdom parameter settings object, uses default if None

    State Requirements:
        - :attr:`torchbearer.state.EPOCH`: State should have the current epoch stored
        - `key`: State should have images stored under the given state key
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
                 norm_range=None,
                 scale_each=False,
                 pad_value=0,
                 visdom=False,
                 visdom_params=None):
        super(TensorBoardImages, self).__init__(log_dir, comment, visdom=visdom, visdom_params=visdom_params)
        self.name = name
        self.key = key
        self.write_each_epoch = write_each_epoch
        self.num_images = num_images
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

        self._data = None
        self.done = False

    def on_step_validation(self, state):
        if not self.done:
            import torchvision.utils as utils
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
                    range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value
                )
                if self.visdom:
                    name = self.name + str(state[torchbearer.EPOCH])
                else:
                    name = self.name
                self.writer.add_image(name, image, state[torchbearer.EPOCH])
                self.done = True
                self._data = None

    def on_end_epoch(self, state):
        if self.write_each_epoch:
            self.done = False


class TensorBoardProjector(AbstractTensorBoard):
    """The TensorBoardProjector callback is used to write images from the validation pass to Tensorboard using the
    TensorboardX library. Images are written to the given directory and, if required, so are associated features.

    Args:
        log_dir (str): The tensorboard log path for output
        comment (str): Descriptive comment to append to path
        num_images (int): The number of images to write
        avg_pool_size (int): Size of the average pool to perform on the image. This is recommended to reduce the overall
            image sizes and improve latency
        avg_data_channels (bool): If True, the image data will be averaged in the channel dimension
        write_data (bool): If True, the raw data will be written as an embedding
        write_features (bool): If True, the image features will be written as an embedding
        features_key (StateKey): The key in state to use for the embedding. Typically model output but can be used to show
            features from any layer of the model.

    State Requirements:
        - :attr:`torchbearer.state.EPOCH`: State should have the current epoch stored
        - :attr:`torchbearer.state.X`: State should have the current data stored
        - :attr:`torchbearer.state.Y_TRUE`: State should have the current targets stored
    """

    def __init__(self, log_dir='./logs',
                 comment='torchbearer',
                 num_images=100,
                 avg_pool_size=1,
                 avg_data_channels=True,
                 write_data=True,
                 write_features=True,
                 features_key=torchbearer.Y_PRED):
        super(TensorBoardProjector, self).__init__(log_dir, comment)
        self.num_images = num_images
        self.avg_pool_size = avg_pool_size
        self.avg_data_channels = avg_data_channels
        self.write_data = write_data
        self.write_features = write_features
        self.features_key = features_key
        self.done = False
        self._images = None
        self._labels = None
        self._features = None
        self._data = None

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
                    self.writer.add_embedding(self._data, metadata=self._labels, label_img=self._images, tag='data',
                                              global_step=-1)
                if self.write_features:
                    self.writer.add_embedding(self._features, metadata=self._labels, label_img=self._images,
                                              tag='features', global_step=state[torchbearer.EPOCH])
                self.done = True

    def on_end_epoch(self, state):
        if self.write_features:
            self.done = False
