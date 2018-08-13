import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchbearer
from torchbearer.metrics import MetricList
from torchbearer.callbacks import CallbackList


def fluent(func):
    """Decorator for class methods which forces return of self.
    """
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return wrapper


def update_device_and_dtype(state, *args, **kwargs):
    """Function get data type and device values from the args / kwargs and update state.

    :param state: The dict to update
    :type state: dict
    :param args: Arguments to the :func:`Trial.to` function
    :param kwargs: Keyword arguments to the :func:`Trial.to` function
    :return: device, dtype pair
    :rtype: tuple
    """
    for key, _ in kwargs.items():
        if key == torchbearer.DATA_TYPE:
            state[torchbearer.DATA_TYPE] = kwargs['dtype']
        elif torchbearer.DEVICE in kwargs:
            state[torchbearer.DEVICE] = kwargs['device']

    for arg in args:
        if isinstance(arg, torch.dtype):
            state[torchbearer.DATA_TYPE] = arg
        else:
            state[torchbearer.DEVICE] = arg

    return state


class Trial(object):
    """ The trial class contains all of the required hyper-parameters for model running in torchbearer and presents an
    API for model fitting, evaluating and predicting.

    :param model: The base pytorch model
    :type model: torch.nn.Module
    :param criterion: The final loss criterion that provides a loss value to the optimizer
    :type criterion: function or None
    :param optimizer: The optimizer used for pytorch model weight updates
    :type optimizer: torch.optim.Optimizer
    :param metrics: The list of :class:`torchbearer.Metric <.Metric>` instances to process during fitting
    :type metrics: list
    :param callbacks: The list of :class:`torchbearer.Callback <.Callback>` instances to call during fitting
    :type callbacks: list
    :param pass_state: If True, the torchbearer state will be passed to the model during fitting
    :type pass_state: bool
    """
    def __init__(self, model, criterion=None, optimizer=None, metrics=[], callbacks=[], pass_state=False):
        if criterion is None:
            def criterion(_, y_true):
                torch.zeros(1, device=y_true.device)

        self.pass_state = pass_state

        self.state = {
            torchbearer.MODEL: model,
            torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: MetricList(metrics),
            torchbearer.CALLBACK_LIST: CallbackList(callbacks),
            torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float32,
            torchbearer.SELF: self
        }

    @fluent
    def with_train_generator(self, generator, steps=None):
        """Use this trial with the given train generator. Returns self so that methods can be chained for convenience.

        :param generator: The train data generator to use during calls to :meth:`.fit`
        :type generator: DataLoader
        :param steps: The number of steps per epoch to take when using this generator
        :type steps: int
        :return: self
        :rtype: Trial
        """
        self.state[torchbearer.GENERATOR] = generator

        steps = len(generator) if steps is None else steps
        if not isinstance(steps, int):
            warnings.warn("Number of training steps is not an int, casting to int")
            steps = int(steps)
        if steps > len(generator):
            warnings.warn("Number of training steps exceeds number of data items, limiting to number of items")
            steps = len(generator)
        self.state[torchbearer.TRAIN_STEPS] = steps

    @fluent
    def with_train_data(self, x, y, batch_size=1, shuffle=True, num_workers=1, steps=None):
        """Use this trial with the given train data. Returns self so that methods can be chained for convenience.

        :param x: The train x data to use during calls to :meth:`.fit`
        :type x: torch.Tensor
        :param y: The train labels to use during calls to :meth:`.fit`
        :type y: torch.Tensor
        :param batch_size: The size of each batch to sample from the data
        :type batch_size: int
        :param shuffle: If True, then data will be shuffled each epoch
        :type shuffle: bool
        :param num_workers: Number of worker threads to use in the data loader
        :type num_workers: int
        :param steps: The number of steps per epoch to take when using this data
        :type steps: int
        :return: self
        :rtype: Trial
        """
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
        self.with_train_generator(dataloader, steps=steps)

    @fluent
    def with_val_generator(self, generator, steps=None):
        """Use this trial with the given validation generator. Returns self so that methods can be chained for
        convenience.

        :param generator: The validation data generator to use during calls to :meth:`.fit` and :meth:`.evaluate`
        :type generator: DataLoader
        :param steps: The number of steps per epoch to take when using this generator
        :type steps: int
        :return: self
        :rtype: Trial
        """
        self.state[torchbearer.VALIDATION_GENERATOR] = generator

        steps = len(generator) if steps is None else steps
        if not isinstance(steps, int):
            warnings.warn("Number of validation steps is not an int, casting to int")
            steps = int(steps)
        if steps > len(generator):
            warnings.warn("Number of validation steps exceeds number of data items, limiting to number of items")
            steps = len(generator)
        self.state[torchbearer.VALIDATION_STEPS] = steps

    @fluent
    def with_val_data(self, x, y, batch_size=1, shuffle=True, num_workers=1, steps=None):
        """Use this trial with the given validation data. Returns self so that methods can be chained for convenience.

        :param x: The validation x data to use during calls to :meth:`.fit` and :meth:`.evaluate`
        :type x: torch.Tensor
        :param y: The validation labels to use during calls to :meth:`.fit` and :meth:`.evaluate`
        :type y: torch.Tensor
        :param batch_size: The size of each batch to sample from the data
        :type batch_size: int
        :param shuffle: If True, then data will be shuffled each epoch
        :type shuffle: bool
        :param num_workers: Number of worker threads to use in the data loader
        :type num_workers: int
        :param steps: The number of steps per epoch to take when using this data
        :type steps: int
        :return: self
        :rtype: Trial
        """
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
        self.with_val_generator(dataloader, steps=steps)

    @fluent
    def with_test_generator(self, generator, steps=None):
        """Use this trial with the given test generator. Returns self so that methods can be chained for convenience.

        :param generator: The test data generator to use during calls to :meth:`.predict`
        :type generator: DataLoader
        :param steps: The number of steps per epoch to take when using this generator
        :type steps: int
        :return: self
        :rtype: Trial
        """
        self.state[torchbearer.TEST_GENERATOR] = generator

        steps = len(generator) if steps is None else steps
        if not isinstance(steps, int):
            warnings.warn("Number of test steps is not an int, casting to int")
            steps = int(steps)
        if steps > len(generator):
            warnings.warn("Number of test steps exceeds number of data items, limiting to number of items")
            steps = len(generator)
        self.state[torchbearer.TEST_STEPS] = steps

    @fluent
    def with_test_data(self, x, batch_size=1, num_workers=1, steps=None):
        """Use this trial with the given test data. Returns self so that methods can be chained for convenience.

        :param x: The test x data to use during calls to :meth:`.predict`
        :type x: torch.Tensor
        :param batch_size: The size of each batch to sample from the data
        :type batch_size: int
        :param num_workers: Number of worker threads to use in the data loader
        :type num_workers: int
        :param steps: The number of steps per epoch to take when using this data
        :type steps: int
        :return: self
        :rtype: Trial
        """
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
        self.with_test_generator(dataloader, steps=steps)

    def fit(self, epochs=1, verbose=2):
        """Fit this trial for the given number of epochs, starting from the last trained epoch.

        :param epochs: The number of epochs to fit for
        :type epochs: int
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :return: The model history (dict of epoch metrics)
        :rtype: dict
        """
        pass

    def evaluate(self, verbose=2):
        """Evaluate this trial on the validation data.

        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :return: The final metric values
        :rtype: dict
        """
        pass

    def predict(self, verbose=2):
        """Determine predictions for this trial on the test data.

        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :return: Model outputs as a list
        :rtype: list
        """
        pass

    @fluent
    def train(self):
        """Set model and metrics to training mode.

        :return: self
        :rtype: Trial
        """
        self.state[torchbearer.MODEL].train()
        self.state[torchbearer.METRIC_LIST].train()

    @fluent
    def eval(self):
        """Set model and metrics to evaluation mode

        :return: self
        :rtype: Trial
        """
        self.state[torchbearer.MODEL].eval()
        self.state[torchbearer.METRIC_LIST].eval()

    @fluent
    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers.

        :param args: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :param kwargs: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :return: self
        :rtype: Trial
        """
        self.state[torchbearer.MODEL].to(*args, **kwargs)

        for state in self.state[torchbearer.OPTIMIZER].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(*args, **kwargs)

        self.state = update_device_and_dtype(self.state, *args, **kwargs)

    @fluent
    def cuda(self, device=None):
        """ Moves all model parameters and buffers to the GPU.

        :param device: if specified, all parameters will be copied to that device
        :type device: int, optional
        :return: self
        :rtype: Trial
        """
        if device is None:
            device = torch.cuda.current_device()
        self.to('cuda:' + str(device))

    @fluent
    def cpu(self):
        """ Moves all model parameters and buffers to the CPU.

        :return: self
        :rtype: Trial
        """
        self.to('cpu')

    def state_dict(self, **kwargs):
        """Get a dict containing the model and optimizer states, as well as the model history.

        :param kwargs: See: `torch.nn.Module.state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.state_dict>`_
        :return: A dict containing parameters and persistent buffers.
        :rtype: dict
        """
        state_dict = {
            torchbearer.MODEL: self.state[torchbearer.MODEL].state_dict(**kwargs),
            torchbearer.OPTIMIZER: self.state[torchbearer.OPTIMIZER].state_dict(),
            torchbearer.HISTORY: self.state[torchbearer.HISTORY]
        }
        return state_dict

    @fluent
    def load_state_dict(self, state_dict, resume=True, **kwargs):
        """Resume this trial from the given state. Expects that this trial was constructed in the same way. Optionally,
        just load the model state.

        :param state_dict: The state dict to reload
        :type state_dict: dict
        :param resume: If True, resume from the given state. Else, just load in the model weights.
        :param kwargs: See: `torch.nn.Module.load_state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.load_state_dict>`_
        :return: self
        :rtype: Trial
        """
        if resume:
            self.state[torchbearer.MODEL].load_state_dict(state_dict[torchbearer.MODEL], **kwargs)
            self.state[torchbearer.OPTIMIZER].load_state_dict(state_dict[torchbearer.OPTIMIZER])
            self.state[torchbearer.HISTORY] = state_dict[torchbearer.HISTORY]
        else:
            self.state[torchbearer.MODEL].load_state_dict(state_dict[torchbearer.MODEL], **kwargs)
