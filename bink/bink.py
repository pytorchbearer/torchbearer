import torch
from torch.utils.data import DataLoader, TensorDataset

import bink
from bink.cv_utils import get_train_valid_sets
from bink.callbacks.callbacks import CallbackList
from bink.callbacks.printer import Tqdm
from bink.callbacks.aggregate_predictions import AggregatePredictions
from bink import metrics as bink_metrics


class Model:
    ''' Binkmodel to wrap base torch model and provide training environment around it
    '''
    def __init__(self, model, optimizer, loss_criterion, metrics=[]):
        ''' Create binkmodel which wraps a base torchmodel and provides a training environment surrounding it

        :param model: The base pytorch model
        :type model: torch.nn.Module
        :param optimizer: The optimizer used for pytorch model weight updates
        :type optimizer: torch.optim.Optimizer
        :param loss_criterion: The final loss criterion that provides a loss value to the optimizer
        :type loss_criterion: function
        :param metrics: Additional metrics for display and use within callbacks
        :type metrics: list
        '''
        super().__init__()
        self.main_state = {
            bink.MODEL: model,
            bink.CRITERION: loss_criterion,
            bink.OPTIMIZER: optimizer,
            bink.DEVICE: 'cpu',
            bink.DATA_TYPE: torch.float32,
            bink.METRIC_LIST: bink_metrics.MetricList(metrics),
            bink.SELF: self
        }

    def fit(self, x, y, batch_size=None, epochs=1, verbose=1, callbacks=[], validation_split=0.0,
            validation_data=None, shuffle=True, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, workers=1, pass_state=False):
        ''' Perform fitting of a model to given data and label tensors

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
        :param callbacks: The list of bink callbacks to be called during training and validation
        :type callbacks: list
        :param validation_split: Fraction of the training dataset to be set aside for validation testing
        :type validation_split: float
        :param validation_data: Optional validation data tensor
        :type validation_data: torch.Tensor
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
        '''
        
        trainset, valset = get_train_valid_sets(x, y, validation_data, validation_split, shuffle=shuffle)
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
        ''' Perform fitting of a model to given data generator

        :param generator: The training data generator (usually a pytorch DataLoader)
        :type generator: DataLoader
        :param train_steps: The number of training mini-batches to run per epoch
        :type train_steps: int
        :param epochs: The number of training epochs to be run (each sample from the dataset is viewed exactly once)
        :type epochs: int
        :param verbose: If 1 use tqdm progress frontend, else display no training progress
        :type verbose: int
        :param callbacks: The list of bink callbacks to be called during training and validation
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
        '''
        if verbose == 1:
            callbacks = [Tqdm()] + callbacks
        _callbacks = CallbackList(callbacks)

        # Get train and validation steps
        if validation_steps is None and validation_generator is not None:
            validation_steps = len(validation_generator)
        if train_steps is None:
            train_steps = len(generator)

        # Init state
        state = {
            bink.MAX_EPOCHS: epochs,
            bink.TRAIN_STEPS: train_steps,
            bink.BATCH: 0,
            bink.GENERATOR: generator,
            bink.STOP_TRAINING: False
        }
        state.update(self.main_state)

        _callbacks.on_start(state)

        for state[bink.EPOCH] in range(initial_epoch, epochs):
            _callbacks.on_start_epoch(state)

            state[bink.TRAIN_ITERATOR] = iter(state[bink.GENERATOR])
            self.train()

            _callbacks.on_start_training(state)
            state[bink.METRIC_LIST].reset(state)
            state[bink.METRICS] = {}

            for state[bink.BATCH] in range(0, state[bink.TRAIN_STEPS]):
                # Extract batch
                self._load_batch_standard('train', state)
                _callbacks.on_sample(state)

                # Zero grads
                state[bink.OPTIMIZER].zero_grad()

                # Forward pass
                if pass_state:
                    state[bink.Y_PRED] = state[bink.MODEL](state[bink.X], state=state)
                else:
                    state[bink.Y_PRED] = state[bink.MODEL](state[bink.X])
                _callbacks.on_forward(state)

                # Loss Calculation
                state[bink.LOSS] = state[bink.CRITERION](state[bink.Y_PRED], state[bink.Y_TRUE])

                _callbacks.on_criterion(state)
                state[bink.METRICS] = state[bink.METRIC_LIST].process(state)

                # Backwards pass
                state[bink.LOSS].backward()
                _callbacks.on_backward(state)

                # Update parameters
                state[bink.OPTIMIZER].step()
                _callbacks.on_step_training(state)

                if state[bink.STOP_TRAINING]:
                    break

            state[bink.METRICS].update(state[bink.METRIC_LIST].process_final(state))
            final_metrics = state[bink.METRICS]

            _callbacks.on_end_training(state)

            # Validate
            if validation_generator is not None:
                state[bink.VALIDATION_GENERATOR] = validation_generator
                state[bink.VALIDATION_STEPS] = validation_steps
                self.eval()
                self._validate(state, _callbacks, pass_state)

            final_metrics.update(state[bink.METRICS])
            state[bink.METRICS] = final_metrics
            _callbacks.on_end_epoch(state)

            if state[bink.STOP_TRAINING]:
                break
        _callbacks.on_end(state)

        return state

    def _test_loop(self, state, callbacks, pass_state, batch_loader, num_steps=None):
        ''' The generic testing loop used for validation, evaluation and prediction

        :param state: The current state context dictionary
        :type state: dict[str,any]
        :param callbacks: The list of bink callbacks to be called during training and validation
        :type callbacks: CallbackList
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :param batch_loader: The type of batch loader to use
        :type batch_loader: function
        :param num_steps: The number of testing mini-batches to run
        :return: The state context dictionary
        :rtype: dict[str,any]
        '''
        with torch.no_grad():
            state[bink.METRIC_LIST].reset(state)
            state[bink.METRICS] = {}

            if num_steps is None:
                num_steps = len(state[bink.VALIDATION_GENERATOR])

            state[bink.VALIDATION_STEPS] = num_steps
            state[bink.VALIDATION_ITERATOR] = iter(state[bink.VALIDATION_GENERATOR])

            callbacks.on_start_validation(state)

            for state[bink.BATCH] in range(num_steps):
                # Load batch
                batch_loader('validation', state)
                callbacks.on_sample_validation(state)

                # Forward pass
                if pass_state:
                    state[bink.Y_PRED] = state[bink.MODEL](state[bink.X], state=state)
                else:
                    state[bink.Y_PRED] = state[bink.MODEL](state[bink.X])
                callbacks.on_forward_validation(state)

                # Loss and metrics
                if bink.Y_TRUE in state:
                    state[bink.LOSS] = state[bink.CRITERION](state[bink.Y_PRED], state[bink.Y_TRUE])
                    state[bink.METRICS] = state[bink.METRIC_LIST].process(state)

                callbacks.on_step_validation(state)
                if state[bink.STOP_TRAINING]:
                    break

            if bink.Y_TRUE in state:
                state[bink.METRICS].update(state[bink.METRIC_LIST].process_final(state))
            callbacks.on_end_validation(state)

        return state

    def _validate(self, state, _callbacks, pass_state):
        ''' Perform a validation loop

        :param state: The current context state dictionary
        :param _callbacks: The list of bink callbacks to be called during validation loop
        :type callbacks: CallbackList
        :param pass_state: If True the state dictionary is passed to the torch model forward method, if False only the input data is passed
        :type pass_state: bool
        :return: None
        :rtype: None
        '''
        self._test_loop(state, _callbacks, pass_state, self._load_batch_standard, state[bink.VALIDATION_STEPS])

    def evaluate(self, x=None, y=None, batch_size=32, verbose=1, steps=None, pass_state=False):
        ''' Perform an evaluation loop on given data and label tensors to evaluate metrics

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
        '''
        trainset = DataLoader(TensorDataset(x, y), batch_size, steps)
        return self.evaluate_generator(trainset, verbose, pass_state=pass_state)

    def evaluate_generator(self, generator, verbose=1, steps=None, pass_state=False):
        ''' Perform an evaluation loop on given data generator to evaluate metrics

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
        '''

        state = {bink.EPOCH: 0, bink.MAX_EPOCHS: 1, bink.STOP_TRAINING: False, bink.VALIDATION_GENERATOR: generator}
        state.update(self.main_state)

        _callbacks = []
        if verbose == 1:
            _callbacks.append(Tqdm('e'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, self._load_batch_standard, steps)

        return state[bink.METRICS]

    def predict(self, x=None, batch_size=None, verbose=1, steps=None, pass_state=False):
        ''' Perform a prediction loop on given data tensor to predict labels

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
        '''
        pred_set = DataLoader(TensorDataset(x, None), batch_size, steps)
        return self.predict_generator(pred_set, verbose, pass_state=pass_state)

    def predict_generator(self, generator, verbose=1, steps=None, pass_state=False):
        '''Perform a prediction loop on given data generator to predict labels

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
        '''
        state = {bink.EPOCH: 0, bink.MAX_EPOCHS: 1, bink.STOP_TRAINING: False, bink.VALIDATION_GENERATOR: generator}
        state.update(self.main_state)

        _callbacks = [AggregatePredictions()]
        if verbose == 1:
            _callbacks.append(Tqdm('p'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, self._load_batch_predict, steps)

        return state[bink.FINAL_PREDICTIONS]

    def train(self):
        ''' Set model and metrics to training mode
        '''
        self.main_state[bink.MODEL].train()
        self.main_state[bink.METRIC_LIST].train()

    def eval(self):
        ''' Set model and metrics to evaluation mode
        '''
        self.main_state[bink.MODEL].eval()
        self.main_state[bink.METRIC_LIST].eval()

    def to(self, *args, **kwargs):
        ''' Moves and/or casts the parameters and buffers.

        :param args: See `torch.nn.Module.to https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to`
        :param kwargs: See `torch.nn.Module.to https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to`
        :return: Self binkmodel
        :rtype: Model
        '''
        self.main_state[bink.MODEL].to(*args, **kwargs)

        for state in self.main_state[bink.OPTIMIZER].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(*args, **kwargs)

        for key, val in kwargs.items():
            if key == bink.DATA_TYPE:
                self.main_state[bink.DATA_TYPE] = kwargs[bink.DATA_TYPE]
            elif bink.DEVICE in kwargs:
                self.main_state[bink.DEVICE] = kwargs[bink.DEVICE]
        for arg in args:
            if isinstance(arg, torch.dtype):
                self.main_state[bink.DATA_TYPE] = arg
            else:
                self.main_state[bink.DEVICE] = arg

        return self

    def cuda(self, device=torch.cuda.current_device()):
        ''' Moves all model parameters and buffers to the GPU.

        :param device: if specified, all parameters will be copied to that device
        :type device: int, optional
        :return: Self binkmodel
        :rtype: Model
        '''
        return self.to('cuda:' + str(device))

    def cpu(self):
        ''' Moves all model parameters and buffers to the CPU.

        :return: Self binkmodel
        :rtype: Model
        '''
        return self.to('cpu')

    def load_state_dict(self, state_dict, **kwargs):
        ''' Copies parameters and buffers from :func:`state_dict` into this module and its descendants.

        :param state_dict: A dict containing parameters and persistent buffers.
        :type state_dict: dict
        :param kwargs: `See torch.nn.Module.load_state_dict https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.load_state_dict`
        '''
        self.main_state[bink.MODEL].load_state_dict(state_dict[bink.MODEL], **kwargs)
        self.main_state[bink.OPTIMIZER].load_state_dict(state_dict[bink.OPTIMIZER])

    def state_dict(self, **kwargs):
        '''
        :param kwargs: See `torch.nn.Module.state_dict https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.state_dict`

        :return: A dict containing parameters and persistent buffers.
        :rtype: dict
        '''
        state_dict = {
            bink.MODEL: self.main_state[bink.MODEL].state_dict(**kwargs),
            bink.OPTIMIZER: self.main_state[bink.OPTIMIZER].state_dict()
        }
        return state_dict

    @staticmethod
    def _deep_to(batch, device, dtype):
        ''' Static method to call :func:`to` on tensors or tuples. All items in tuple will have :func:_deep_to called

        :param batch: The mini-batch which requires a :func:`to` call
        :type batch: tuple, list, torch.Tensor
        :param device: The desired device of the batch
        :type device: torch.device
        :param dtype: The desired datatype of the batch
        :type dtype: torch.dtype
        :return: The moved or casted batch
        :rtype: tuple, list, torch.Tensor
        '''
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
        ''' Static method to load a standard (input data, target) tuple mini-batch from iterator into state

        :param iterator: Training or validation data iterator
        :type iterator: iterable
        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]
        '''
        state[bink.X], state[bink.Y_TRUE] = Model._deep_to(next(state[iterator + '_iterator']), state[bink.DEVICE], state[bink.DATA_TYPE])

    @staticmethod
    def _load_batch_predict(iterator, state):
        ''' Static method to load a prediction (input data, target) or (input data) mini-batch from iterator into state

        :param iterator: Training or validation data iterator
        :type iterator: iterable
        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]
        '''
        data = Model._deep_to(next(state[iterator + '_iterator']), state[bink.DEVICE], state[bink.DATA_TYPE])
        if isinstance(data, list) or isinstance(data, tuple):
            state[bink.X], state[bink.Y_TRUE] = data
        else:
            state[bink.X] = data
