import warnings

import torch
from torch.utils.data import DataLoader

import torchbearer
from torchbearer import Trial
from torchbearer import metrics as torchbearer_metrics


class Model:
    """ Create torchbearermodel which wraps a base torchmodel and provides a training environment surrounding it

    :param model: The base pytorch model
    :type model: torch.nn.Module
    :param optimizer: The optimizer used for pytorch model weight updates
    :type optimizer: torch.optim.Optimizer
    :param criterion: The final loss criterion that provides a loss value to the optimizer
    :type criterion: function or None
    :param metrics: Additional metrics for display and use within callbacks
    :type metrics: list
    """
    def __init__(self, model, optimizer, criterion=None, metrics=[]):
        super().__init__()
        if criterion is None:
            criterion = lambda y_pred, y_true: torch.zeros(1, device=y_true.device)

        self.main_state = {
            torchbearer.MODEL: model,
            torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float32,
            torchbearer.METRIC_LIST: torchbearer_metrics.MetricList(metrics),
            torchbearer.SELF: self,
            torchbearer.CALLBACK_LIST: torchbearer.callbacks.CallbackList([])
        }
        self.trial = None

    def fit(self, x, y, batch_size=None, epochs=1, verbose=2, callbacks=[], validation_split=None,
            validation_data=None, shuffle=True, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, workers=1, pass_state=False):
        """ Perform fitting of a model to given data and label tensors

        :param x: The input data tensor
        :type x: torch.Tensor
        :param y: The target labels for data tensor x
        :type y: torch.Tensor
        :param batch_size: The mini-batch size (number of samples processed for a single weight update)
        :type batch_size: int
        :param epochs: The number of training epochs to be run (each sample from the dataset is viewed exactly once)
        :type epochs: int
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :param callbacks: The list of torchbearer callbacks to be called during training and validation
        :type callbacks: list
        :param validation_split: Fraction of the training dataset to be set aside for validation testing
        :type validation_split: float
        :param validation_data: Optional validation data tensor
        :type validation_data: (torch.Tensor, torch.Tensor)
        :param shuffle: If True mini-batches of training/validation data are randomly selected, if False mini-batches samples are selected in order defined by dataset
        :type shuffle: bool
        :param initial_epoch: The integer value representing the first epoch - useful for continuing training after a number of epochs
        :type initial_epoch: int
        :param steps_per_epoch: The number of training mini-batches to run per epoch
        :type steps_per_epoch: int
        :param validation_steps: The number of validation mini-batches to run per epoch
        :type validation_steps: int
        :param workers: The number of cpu workers devoted to batch loading and aggregating
        :type workers: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: The final state context dictionary
        :rtype: dict[str,any]
        """
        warnings.warn('Model.fit is deprecated and will be removed in the next release, use Trial.fit instead', DeprecationWarning)
        self.trial = Trial(self.main_state[torchbearer.MODEL],
                           criterion=self.main_state[torchbearer.CRITERION],
                           optimizer=self.main_state[torchbearer.OPTIMIZER],
                           metrics=self.main_state[torchbearer.METRIC_LIST],
                           callbacks=callbacks,
                           pass_state=pass_state)
        self.trial.with_train_data(x, y, batch_size=batch_size, shuffle=shuffle, num_workers=workers, steps=steps_per_epoch)

        if validation_split is not None:
            self.trial.with_split(validation_split, val_steps=validation_steps)
        elif validation_data is not None:
            self.trial.with_val_data(validation_data[0], validation_data[1], batch_size=batch_size, shuffle=shuffle, num_workers=workers, steps=validation_steps)

        return self.trial.fit(epochs=epochs, verbose=verbose)

    def fit_generator(self, generator, train_steps=None, epochs=1, verbose=2, callbacks=[],
                      validation_generator=None, validation_steps=None, initial_epoch=0, pass_state=False):
        """ Perform fitting of a model to given data generator

        :param generator: The training data generator (usually a pytorch DataLoader)
        :type generator: DataLoader
        :param train_steps: The number of training mini-batches to run per epoch
        :type train_steps: int
        :param epochs: The number of training epochs to be run (each sample from the dataset is viewed exactly once)
        :type epochs: int
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :param callbacks: The list of torchbearer callbacks to be called during training and validation
        :type callbacks: list
        :param validation_generator: The validation data generator (usually a pytorch DataLoader)
        :type validation_generator: DataLoader
        :param validation_steps: The number of validation mini-batches to run per epoch
        :type validation_steps: int
        :param initial_epoch: The integer value representing the first epoch - useful for continuing training after a number of epochs
        :type initial_epoch: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: The final state context dictionary
        :rtype: dict[str,any]
        """
        warnings.warn('Model.fit_generator is deprecated and will be removed in the next release, use Trial.fit instead', DeprecationWarning)
        self.trial = Trial(self.main_state[torchbearer.MODEL],
                           criterion=self.main_state[torchbearer.CRITERION],
                           optimizer=self.main_state[torchbearer.OPTIMIZER],
                           metrics=self.main_state[torchbearer.METRIC_LIST],
                           callbacks=callbacks,
                           pass_state=pass_state)
        self.trial.with_train_data(generator, steps=train_steps)
        self.trial.with_val_generator(validation_generator, steps=validation_steps)

        return self.trial.fit(epochs=epochs, verbose=verbose)

    def evaluate(self, x=None, y=None, batch_size=32, verbose=2, steps=None, pass_state=False):
        """ Perform an evaluation loop on given data and label tensors to evaluate metrics

        :param x: The input data tensor
        :type x: torch.Tensor
        :param y: The target labels for data tensor x
        :type y: torch.Tensor
        :param batch_size: The mini-batch size (number of samples processed for a single weight update)
        :type batch_size: int
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: The dictionary containing final metrics
        :rtype: dict[str,any]
        """
        warnings.warn('Model.evaluate is deprecated and will be removed in the next release, use Trial.evaluate instead', DeprecationWarning)
        self.trial = Trial(self.main_state[torchbearer.MODEL],
                           criterion=self.main_state[torchbearer.CRITERION],
                           optimizer=self.main_state[torchbearer.OPTIMIZER],
                           metrics=self.main_state[torchbearer.METRIC_LIST],
                           pass_state=pass_state)
        self.trial.with_val_data(x, y, batch_size=batch_size, steps=steps)

        return self.trial.evaluate(verbose)

    def evaluate_generator(self, generator, verbose=2, steps=None, pass_state=False):
        """ Perform an evaluation loop on given data generator to evaluate metrics

        :param generator: The evaluation data generator (usually a pytorch DataLoader)
        :type generator: DataLoader
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: The dictionary containing final metrics
        :rtype: dict[str,any]
        """
        warnings.warn(
            'Model.evaluate_generator is deprecated and will be removed in the next release, use Trial.evaluate instead',
            DeprecationWarning)
        self.trial = Trial(self.main_state[torchbearer.MODEL],
                           criterion=self.main_state[torchbearer.CRITERION],
                           optimizer=self.main_state[torchbearer.OPTIMIZER],
                           metrics=self.main_state[torchbearer.METRIC_LIST],
                           pass_state=pass_state)
        self.trial.with_val_generator(generator, steps=steps)

        return self.trial.evaluate(verbose)

    def predict(self, x=None, batch_size=32, verbose=2, steps=None, pass_state=False):
        """ Perform a prediction loop on given data tensor to predict labels

        :param x: The input data tensor
        :type x: torch.Tensor
        :param batch_size: The mini-batch size (number of samples processed for a single weight update)
        :type batch_size: int
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: Tensor of final predicted labels
        :rtype: torch.Tensor
        """
        warnings.warn(
            'Model.predict is deprecated and will be removed in the next release, use Trial.predict instead',
            DeprecationWarning)
        self.trial = Trial(self.main_state[torchbearer.MODEL],
                           criterion=self.main_state[torchbearer.CRITERION],
                           optimizer=self.main_state[torchbearer.OPTIMIZER],
                           metrics=self.main_state[torchbearer.METRIC_LIST],
                           pass_state=pass_state)
        self.trial.with_test_data(x, batch_size=batch_size, steps=steps)

        return self.trial.predict(verbose)

    def predict_generator(self, generator, verbose=2, steps=None, pass_state=False):
        """Perform a prediction loop on given data generator to predict labels

        :param generator: The prediction data generator (usually a pytorch DataLoader)
        :type generator: DataLoader
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: Tensor of final predicted labels
        :rtype: torch.Tensor
        """
        warnings.warn(
            'Model.predict_generator is deprecated and will be removed in the next release, use Trial.predict instead',
            DeprecationWarning)
        self.trial = Trial(self.main_state[torchbearer.MODEL],
                           criterion=self.main_state[torchbearer.CRITERION],
                           optimizer=self.main_state[torchbearer.OPTIMIZER],
                           metrics=self.main_state[torchbearer.METRIC_LIST],
                           pass_state=pass_state)
        self.trial.with_test_generator(generator, steps=steps)

        return self.trial.predict(verbose)

    def train(self):
        """ Set model and metrics to training mode
        """
        self.trial.train()

    def eval(self):
        """ Set model and metrics to evaluation mode
        """
        self.trial.eval()

    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers.

        :param args: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :param kwargs: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :return: Self torchbearermodel
        :rtype: Model
        """
        self.trial.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """ Moves all model parameters and buffers to the GPU.

        :param device: if specified, all parameters will be copied to that device
        :type device: int, optional
        :return: Self torchbearermodel
        :rtype: Model
        """
        self.trial.cuda(device)
        return self

    def cpu(self):
        """ Moves all model parameters and buffers to the CPU.

        :return: Self torchbearermodel
        :rtype: Model
        """
        self.trial.cpu()
        return self

    def load_state_dict(self, state_dict, **kwargs):
        """ Copies parameters and buffers from :func:`state_dict` into this module and its descendants.

        :param state_dict: A dict containing parameters and persistent buffers.
        :type state_dict: dict
        :param kwargs: See: `torch.nn.Module.load_state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.load_state_dict>`_
        """
        self.trial.load_state_dict(state_dict, **kwargs)

    def state_dict(self, **kwargs):
        """
        :param kwargs: See: `torch.nn.Module.state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.state_dict>`_

        :return: A dict containing parameters and persistent buffers.
        :rtype: dict
        """
        return self.trial.state_dict(**kwargs)
