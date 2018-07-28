import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchbearer
from torchbearer import metrics as torchbearer_metrics
from torchbearer.callbacks.aggregate_predictions import AggregatePredictions
from torchbearer.callbacks.callbacks import CallbackList
from torchbearer.callbacks.printer import Tqdm


class Model:
    """ Torchbearermodel to wrap base torch model and provide training environment around it
    """
    def __init__(self, model, optimizer, loss_criterion, metrics=[]):
        """ Create torchbearermodel which wraps a base torchmodel and provides a training environment surrounding it

        :param model: The base pytorch model
        :type model: torch.nn.Module
        :param optimizer: The optimizer used for pytorch model weight updates
        :type optimizer: torch.optim.Optimizer
        :param loss_criterion: The final loss criterion that provides a loss value to the optimizer
        :type loss_criterion: function
        :param metrics: Additional metrics for display and use within callbacks
        :type metrics: list
        """
        super().__init__()
        self.main_state = {
            torchbearer.MODEL: model,
            torchbearer.CRITERION: loss_criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float32,
            torchbearer.METRIC_LIST: torchbearer_metrics.MetricList(metrics),
            torchbearer.SELF: self
        }

    def fit(self, x, y, batch_size=None, epochs=1, verbose=1, callbacks=[], validation_split=None,
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
        :param verbose: If 1 use tqdm progress frontend, else display no training progress
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
        
        trainset, valset = torchbearer.cv_utils.get_train_valid_sets(x, y, validation_data, validation_split, shuffle=shuffle)
        trainloader = DataLoader(trainset, batch_size, shuffle=shuffle, num_workers=workers)

        if valset is not None:
            valloader = DataLoader(valset, batch_size, shuffle=shuffle, num_workers=workers)
        else:
            valloader = None

        return self.fit_generator(trainloader, train_steps=steps_per_epoch, epochs=epochs, verbose=verbose,
                                  callbacks=callbacks, validation_generator=valloader, validation_steps=validation_steps,
                                  initial_epoch=initial_epoch, pass_state=pass_state)

    def fit_generator(self, generator, train_steps=None, epochs=1, verbose=1, callbacks=[],
                      validation_generator=None, validation_steps=None, initial_epoch=0, pass_state=False):
        """ Perform fitting of a model to given data generator

        :param generator: The training data generator (usually a pytorch DataLoader)
        :type generator: DataLoader
        :param train_steps: The number of training mini-batches to run per epoch
        :type train_steps: int
        :param epochs: The number of training epochs to be run (each sample from the dataset is viewed exactly once)
        :type epochs: int
        :param verbose: If 1 use tqdm progress frontend, else display no training progress
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
        if verbose == 1:
            callbacks = [Tqdm()] + callbacks
        _callbacks = CallbackList(callbacks)

        # Get train and validation steps
        if validation_steps is None and validation_generator is not None:
            validation_steps = len(validation_generator)
        if train_steps is None:
            train_steps = len(generator)
        if generator is not None and train_steps > len(generator):
            train_steps = len(generator)
        if not isinstance(train_steps, int):
            train_steps = int(train_steps)
            warnings.warn("Number of training steps is not an int, converting to int")

        if not isinstance(epochs, int):
            if isinstance(epochs, float):
                epochs = int(epochs)
                warnings.warn("Number of epochs is a float, converting to int")
            else:
                warnings.warn("Number of epochs is neither float nor int, setting to 0")
                epochs = 0

        # Init state
        state = {
            torchbearer.MAX_EPOCHS: epochs,
            torchbearer.TRAIN_STEPS: train_steps,
            torchbearer.BATCH: 0,
            torchbearer.GENERATOR: generator,
            torchbearer.STOP_TRAINING: False
        }
        state.update(self.main_state)

        _callbacks.on_start(state)

        for state[torchbearer.EPOCH] in range(initial_epoch, epochs):
            _callbacks.on_start_epoch(state)

            if state[torchbearer.GENERATOR] is not None:
                state[torchbearer.TRAIN_ITERATOR] = iter(state[torchbearer.GENERATOR])
            self.train()

            _callbacks.on_start_training(state)
            state[torchbearer.METRIC_LIST].reset(state)
            state[torchbearer.METRICS] = {}

            for state[torchbearer.BATCH] in range(0, state[torchbearer.TRAIN_STEPS]):
                # Extract batch
                if state[torchbearer.GENERATOR] is None: # TODO: Replace with flag check
                    self._load_batch_none('train', state)
                else:
                    self._load_batch_standard('train', state)

                _callbacks.on_sample(state)

                # Zero grads
                state[torchbearer.OPTIMIZER].zero_grad()

                # Forward pass
                if pass_state:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X], state=state)
                else:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X])
                _callbacks.on_forward(state)

                # Loss Calculation
                state[torchbearer.LOSS] = state[torchbearer.CRITERION](state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE])

                _callbacks.on_criterion(state)
                state[torchbearer.METRICS] = state[torchbearer.METRIC_LIST].process(state)

                # Backwards pass
                state[torchbearer.LOSS].backward()
                _callbacks.on_backward(state)

                # Update parameters
                state[torchbearer.OPTIMIZER].step()
                _callbacks.on_step_training(state)

                if state[torchbearer.STOP_TRAINING]:
                    break

            state[torchbearer.METRICS].update(state[torchbearer.METRIC_LIST].process_final(state))
            final_metrics = state[torchbearer.METRICS]

            _callbacks.on_end_training(state)

            # Validate
            if validation_generator is not None or validation_steps is not None:
                state[torchbearer.VALIDATION_GENERATOR] = validation_generator
                state[torchbearer.VALIDATION_STEPS] = validation_steps
                self.eval()
                self._validate(state, _callbacks, pass_state)

            final_metrics.update(state[torchbearer.METRICS])
            state[torchbearer.METRICS] = final_metrics
            _callbacks.on_end_epoch(state)

            if state[torchbearer.STOP_TRAINING]:
                break
        _callbacks.on_end(state)

        return state

    def _test_loop(self, state, callbacks, pass_state, batch_loader, num_steps=None):
        """ The generic testing loop used for validation, evaluation and prediction

        :param state: The current state context dictionary
        :type state: dict[str,any]
        :param callbacks: The list of torchbearer callbacks to be called during training and validation
        :type callbacks: CallbackList
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :param batch_loader: The batch loader to use
        :type batch_loader: function
        :param num_steps: The number of testing mini-batches to run
        :return: The state context dictionary
        :rtype: dict[str,any]
        """
        with torch.no_grad():
            state[torchbearer.METRIC_LIST].reset(state)
            state[torchbearer.METRICS] = {}

            if num_steps is None:
                num_steps = len(state[torchbearer.VALIDATION_GENERATOR])
            if state[torchbearer.VALIDATION_GENERATOR] is not None and num_steps > len(state[torchbearer.VALIDATION_GENERATOR]):
                num_steps = len(state[torchbearer.VALIDATION_GENERATOR])
            if not isinstance(num_steps, int):
                num_steps = int(num_steps)
                warnings.warn('Num test steps is not an int, converting to int.', Warning)

            state[torchbearer.VALIDATION_STEPS] = num_steps
            if state[torchbearer.VALIDATION_GENERATOR] is not None:
                state[torchbearer.VALIDATION_ITERATOR] = iter(state[torchbearer.VALIDATION_GENERATOR])

            callbacks.on_start_validation(state)

            for state[torchbearer.BATCH] in range(state[torchbearer.VALIDATION_STEPS]):
                # Load batch
                batch_loader('validation', state)
                callbacks.on_sample_validation(state)

                # Forward pass
                if pass_state:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X], state=state)
                else:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X])
                callbacks.on_forward_validation(state)

                # Loss and metrics
                if torchbearer.Y_TRUE in state:
                    state[torchbearer.LOSS] = state[torchbearer.CRITERION](state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE])
                    callbacks.on_criterion_validation(state)
                    state[torchbearer.METRICS] = state[torchbearer.METRIC_LIST].process(state)

                callbacks.on_step_validation(state)
                if state[torchbearer.STOP_TRAINING]:
                    break

            if torchbearer.Y_TRUE in state:
                state[torchbearer.METRICS].update(state[torchbearer.METRIC_LIST].process_final(state))
            callbacks.on_end_validation(state)

        return state

    def _validate(self, state, _callbacks, pass_state):
        """ Perform a validation loop

        :param state: The current context state dictionary
        :param _callbacks: The list of torchbearer callbacks to be called during validation loop
        :type callbacks: CallbackList
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: None
        :rtype: None
        """
        self._test_loop(state, _callbacks, pass_state, self._load_batch_standard, state[torchbearer.VALIDATION_STEPS])

    def evaluate(self, x=None, y=None, batch_size=32, verbose=1, steps=None, pass_state=False):
        """ Perform an evaluation loop on given data and label tensors to evaluate metrics

        :param x: The input data tensor
        :type x: torch.Tensor
        :param y: The target labels for data tensor x
        :type y: torch.Tensor
        :param batch_size: The mini-batch size (number of samples processed for a single weight update)
        :type batch_size: int
        :param verbose: If 1 use tqdm progress frontend, else display no training progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: The dictionary containing final metrics
        :rtype: dict[str,any]
        """
        trainset = DataLoader(TensorDataset(x, y), batch_size, steps)
        return self.evaluate_generator(trainset, verbose, pass_state=pass_state)

    def evaluate_generator(self, generator, verbose=1, steps=None, pass_state=False):
        """ Perform an evaluation loop on given data generator to evaluate metrics

        :param generator: The evaluation data generator (usually a pytorch DataLoader)
        :type generator: DataLoader
        :param verbose: If 1 use tqdm progress frontend, else display no training progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: The dictionary containing final metrics
        :rtype: dict[str,any]
        """

        state = {torchbearer.EPOCH: 0, torchbearer.MAX_EPOCHS: 1, torchbearer.STOP_TRAINING: False, torchbearer.VALIDATION_GENERATOR: generator}
        state.update(self.main_state)

        _callbacks = []
        if verbose == 1:
            _callbacks.append(Tqdm('e'))

        if state[torchbearer.VALIDATION_GENERATOR] is None:
            batch_loader = self._load_batch_none
        else:
            batch_loader = self._load_batch_standard

        self._test_loop(state, CallbackList(_callbacks), pass_state, batch_loader, steps)

        return state[torchbearer.METRICS]

    def predict(self, x=None, batch_size=32, verbose=1, steps=None, pass_state=False):
        """ Perform a prediction loop on given data tensor to predict labels

        :param x: The input data tensor
        :type x: torch.Tensor
        :param batch_size: The mini-batch size (number of samples processed for a single weight update)
        :type batch_size: int
        :param verbose: If 1 use tqdm progress frontend, else display no training progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: Tensor of final predicted labels
        :rtype: torch.Tensor
        """
        pred_set = DataLoader(TensorDataset(x), batch_size, steps)
        return self.predict_generator(pred_set, verbose, pass_state=pass_state)

    def predict_generator(self, generator, verbose=1, steps=None, pass_state=False):
        """Perform a prediction loop on given data generator to predict labels

        :param generator: The prediction data generator (usually a pytorch DataLoader)
        :type generator: DataLoader
        :param verbose: If 1 use tqdm progress frontend, else display no training progress
        :type verbose: int
        :param steps: The number of evaluation mini-batches to run
        :type steps: int
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: Tensor of final predicted labels
        :rtype: torch.Tensor
        """
        state = {torchbearer.EPOCH: 0, torchbearer.MAX_EPOCHS: 1, torchbearer.STOP_TRAINING: False, torchbearer.VALIDATION_GENERATOR: generator}
        state.update(self.main_state)

        _callbacks = [AggregatePredictions()]
        if verbose == 1:
            _callbacks.append(Tqdm('p'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, self._load_batch_predict, steps)

        return state[torchbearer.FINAL_PREDICTIONS]

    def train(self):
        """ Set model and metrics to training mode
        """
        self.main_state[torchbearer.MODEL].train()
        self.main_state[torchbearer.METRIC_LIST].train()

    def eval(self):
        """ Set model and metrics to evaluation mode
        """
        self.main_state[torchbearer.MODEL].eval()
        self.main_state[torchbearer.METRIC_LIST].eval()

    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers.

        :param args: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :param kwargs: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
        :return: Self torchbearermodel
        :rtype: Model
        """
        self.main_state[torchbearer.MODEL].to(*args, **kwargs)

        for state in self.main_state[torchbearer.OPTIMIZER].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(*args, **kwargs)

        self.main_state = Model._update_device_and_dtype_from_args(self.main_state, *args, **kwargs)

        return self

    def cuda(self, device=None):
        """ Moves all model parameters and buffers to the GPU.

        :param device: if specified, all parameters will be copied to that device
        :type device: int, optional
        :return: Self torchbearermodel
        :rtype: Model
        """
        if device is None:
            device = torch.cuda.current_device()
        return self.to('cuda:' + str(device))

    def cpu(self):
        """ Moves all model parameters and buffers to the CPU.

        :return: Self torchbearermodel
        :rtype: Model
        """
        return self.to('cpu')

    def load_state_dict(self, state_dict, **kwargs):
        """ Copies parameters and buffers from :func:`state_dict` into this module and its descendants.

        :param state_dict: A dict containing parameters and persistent buffers.
        :type state_dict: dict
        :param kwargs: See: `torch.nn.Module.load_state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.load_state_dict>`_
        """
        self.main_state[torchbearer.MODEL].load_state_dict(state_dict[torchbearer.MODEL], **kwargs)
        self.main_state[torchbearer.OPTIMIZER].load_state_dict(state_dict[torchbearer.OPTIMIZER])

    def state_dict(self, **kwargs):
        """
        :param kwargs: See: `torch.nn.Module.state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.state_dict>`_

        :return: A dict containing parameters and persistent buffers.
        :rtype: dict
        """
        state_dict = {
            torchbearer.MODEL: self.main_state[torchbearer.MODEL].state_dict(**kwargs),
            torchbearer.OPTIMIZER: self.main_state[torchbearer.OPTIMIZER].state_dict()
        }
        return state_dict

    @staticmethod
    def _deep_to(batch, device, dtype):
        """ Static method to call :func:`to` on tensors or tuples. All items in tuple will have :func:_deep_to called

        :param batch: The mini-batch which requires a :func:`to` call
        :type batch: tuple, list, torch.Tensor
        :param device: The desired device of the batch
        :type device: torch.device
        :param dtype: The desired datatype of the batch
        :type dtype: torch.dtype
        :return: The moved or casted batch
        :rtype: tuple, list, torch.Tensor
        """
        is_tuple = isinstance(batch, tuple)

        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = list(batch)
            for i in range(len(batch)):
                batch[i] = Model._deep_to(batch[i], device, dtype)
            batch = tuple(batch) if is_tuple else batch
        else:
            if batch.dtype.is_floating_point:
                batch = batch.to(device, dtype)
            else:
                batch = batch.to(device)

        return batch

    @staticmethod
    def _load_batch_standard(iterator, state):
        """ Static method to load a standard (input data, target) tuple mini-batch from iterator into state

        :param iterator: Training or validation data iterator
        :type iterator: iterable
        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]
        """
        state[torchbearer.X], state[torchbearer.Y_TRUE] = Model._deep_to(next(state[iterator + '_iterator']), state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])

    @staticmethod
    def _load_batch_none(_, state):
        """Static method to load a none (none, none) tuple mini-batch into state

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]
        """
        state[torchbearer.X], state[torchbearer.Y_TRUE] = None, None

    @staticmethod
    def _load_batch_predict(iterator, state):
        """ Static method to load a prediction (input data, target) or (input data) mini-batch from iterator into state

        :param iterator: Training or validation data iterator
        :type iterator: iterable
        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]
        """
        data = Model._deep_to(next(state[iterator + '_iterator']), state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
        if isinstance(data, list) or isinstance(data, tuple):
            state[torchbearer.X], state[torchbearer.Y_TRUE] = data
        else:
            state[torchbearer.X] = data

    @staticmethod
    def _update_device_and_dtype_from_args(main_state, *args, **kwargs):
        """ static method to update a main state dictionary with new data type and device values
        
        :param main_state: The main state to update
        :type main_state: dict[str,any]
        :param args: Arguments to the :func:`Model.to` function
        :param kwargs: Keyword arguments to the :func:`Model.to` function
        :return: Updated main state dictionary
        :rtype: dict[str,any]
        """
        for key, val in kwargs.items():
            if key == torchbearer.DATA_TYPE:
                main_state[torchbearer.DATA_TYPE] = kwargs[torchbearer.DATA_TYPE]
            elif torchbearer.DEVICE in kwargs:
                main_state[torchbearer.DEVICE] = kwargs[torchbearer.DEVICE]

        for arg in args:
            if isinstance(arg, torch.dtype):
                main_state[torchbearer.DATA_TYPE] = arg
            else:
                main_state[torchbearer.DEVICE] = arg

        return main_state