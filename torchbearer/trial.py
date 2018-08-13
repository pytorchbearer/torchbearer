def fluent(func):
    def decorator(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return decorator


class Trial:
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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    @fluent
    def with_split(self, split, val_steps=None):
        """Use this trial with a validation data split from the train data.

        :param split: Fraction of the training data to set aside for validation
        :type split: float
        :param val_steps: Number of validation steps to take or None to default to the length of the split
        :type val_steps: int
        :return: self
        :rtype: Trial
        """
        pass

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

    def train(self):
        """Set model and metrics to training mode.
        """
        pass

    def eval(self):
        """Set model and metrics to evaluation mode
        """
        pass

    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers.

        :param args: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :param kwargs: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :return: self
        :rtype: Trial
        """
        pass

    def cuda(self, device=None):
        """ Moves all model parameters and buffers to the GPU.

        :param device: if specified, all parameters will be copied to that device
        :type device: int, optional
        :return: self
        :rtype: Trial
        """
        pass

    def cpu(self):
        """ Moves all model parameters and buffers to the CPU.

        :return: self
        :rtype: Trial
        """
        pass

    def state_dict(self, **kwargs):
        """Get a dict containing the model and optimizer states, as well as the model history.

        :param kwargs: See: `torch.nn.Module.state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.state_dict>`_
        :return: A dict containing parameters and persistent buffers.
        :rtype: dict
        """
        pass

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
        pass
