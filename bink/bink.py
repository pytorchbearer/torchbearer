import torch
from torch.utils.data import DataLoader, TensorDataset
from bink.cv_utils import get_train_valid_sets
from bink.callbacks.callbacks import CallbackList
from bink.callbacks.printer import Tqdm
from bink.callbacks.aggregate_predictions import AggregatePredictions
from bink import metrics as bink_metrics
from types import *

class Model:
    '''

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
            'model': model,
            'criterion': loss_criterion,
            'optimizer': optimizer,
            'device': 'cpu',
            'dtype': torch.float32,
            'metric_list': bink_metrics.MetricList(metrics),
            'self': self
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
            'max_epochs': epochs,
            'train_steps': train_steps,
            't': 0,
            'generator': generator,
            'stop_training': False
        }
        state.update(self.main_state)

        _callbacks.on_start(state)

        for state['epoch'] in range(initial_epoch, epochs):
            _callbacks.on_start_epoch(state)

            state['train_iterator'] = iter(state['generator'])
            self.train()

            _callbacks.on_start_training(state)
            state['metric_list'].reset(state)
            state['metrics'] = {}

            for state['t'] in range(0, state['train_steps']):
                # Extract batch
                self._load_batch_standard('train', state)
                _callbacks.on_sample(state)

                # Zero grads
                state['optimizer'].zero_grad()

                # Forward pass
                if pass_state:
                    state['y_pred'] = state['model'](state['x'], state=state)
                else:
                    state['y_pred'] = state['model'](state['x'])
                _callbacks.on_forward(state)

                # Loss Calculation
                state['loss'] = state['criterion'](state['y_pred'], state['y_true'])

                _callbacks.on_criterion(state)
                state['metrics'] = state['metric_list'].process(state)

                # Backwards pass
                state['loss'].backward()
                _callbacks.on_backward(state)

                # Update parameters
                state['optimizer'].step()
                _callbacks.on_step_training(state)

                if state['stop_training']:
                    break

            state['metrics'].update(state['metric_list'].process_final(state))
            final_metrics = state['metrics']

            _callbacks.on_end_training(state)

            # Validate
            if validation_generator is not None:
                state['validation_generator'] = validation_generator
                state['validation_steps'] = validation_steps
                self.eval()
                self._validate(state, _callbacks, pass_state)

            final_metrics.update(state['metrics'])
            state['metrics'] = final_metrics
            _callbacks.on_end_epoch(state)

            if state['stop_training']:
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
            state['metric_list'].reset(state)
            state['metrics'] = {}

            if num_steps is None:
                num_steps = len(state['validation_generator'])

            state['validation_steps'] = num_steps
            state['validation_iterator'] = iter(state['validation_generator'])

            callbacks.on_start_validation(state)

            for state['t'] in range(num_steps):
                # Load batch
                batch_loader('validation', state)
                callbacks.on_sample_validation(state)

                # Forward pass
                if pass_state:
                    state['y_pred'] = state['model'](state['x'], state=state)
                else:
                    state['y_pred'] = state['model'](state['x'])
                callbacks.on_forward_validation(state)

                # Loss and metrics
                if 'y_true' in state:
                    state['loss'] = state['criterion'](state['y_pred'], state['y_true'])
                    state['metrics'] = state['metric_list'].process(state)

                callbacks.on_step_validation(state)
                if state['stop_training']:
                    break

            if 'y_true' in state:
                state['metrics'].update(state['metric_list'].process_final(state))
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
        self._test_loop(state, _callbacks, pass_state, self._load_batch_standard, state['validation_steps'])

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
        state = {'epoch': 0, 'max_epochs': 1, 'stop_training': False, 'validation_generator': generator}
        state.update(self.main_state)

        _callbacks = []
        if verbose == 1:
            _callbacks.append(Tqdm('e'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, self._load_batch_standard, steps)

        return state['metrics']

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
        state = {'epoch': 0, 'max_epochs': 1, 'stop_training': False, 'validation_generator': generator}
        state.update(self.main_state)

        _callbacks = [AggregatePredictions()]
        if verbose == 1:
            _callbacks.append(Tqdm('p'))
        self._test_loop(state, CallbackList(_callbacks), pass_state, self._load_batch_predict, steps)

        return state['final_predictions']

    def train(self):
        ''' Set model and metrics to training mode
        '''
        self.main_state['model'].train()
        self.main_state['metric_list'].train()

    def eval(self):
        ''' Set model and metrics to evaluation mode
        '''
        self.main_state['model'].eval()
        self.main_state['metric_list'].eval()

    def to(self, *args, **kwargs):
        ''' Moves and/or casts the parameters and buffers.

        :param args: See `torch.nn.Module.to https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to`
        :param kwargs: See `torch.nn.Module.to https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to`
        :return: Self binkmodel
        :rtype: Model
        '''
        self.main_state['model'].to(*args, **kwargs)

        for state in self.main_state['optimizer'].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(*args, **kwargs)

        for key, val in kwargs.items():
            if key == 'dtype':
                self.main_state['dtype'] = kwargs['dtype']
            elif 'device' in kwargs:
                self.main_state['device'] = kwargs['device']
        for arg in args:
            if isinstance(arg, torch.dtype):
                self.main_state['dtype'] = arg
            else:
                self.main_state['device'] = arg

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
        self.main_state['model'].load_state_dict(state_dict['model'], kwargs)
        self.main_state['optimizer'].load_state_dict(state_dict['optimizer'])

    def state_dict(self, **kwargs):
        '''
        :param kwargs: See `torch.nn.Module.state_dict https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.state_dict`

        :return: A dict containing parameters and persistent buffers.
        :rtype: dict
        '''
        state_dict = {
            'model': self.main_state['model'].state_dict(kwargs),
            'optimizer': self.main_state['optimizer'].state_dict()
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
        state['x'], state['y_true'] = Model._deep_to(next(state[iterator + '_iterator']), state['device'], state['dtype'])

    @staticmethod
    def _load_batch_predict(iterator, state):
        ''' Static method to load a prediction (input data, target) or (input data) mini-batch from iterator into state

        :param iterator: Training or validation data iterator
        :type iterator: iterable
        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]
        '''
        data = Model._deep_to(next(state[iterator + '_iterator']), state['device'], state['dtype'])
        if isinstance(data, list) or isinstance(data, tuple):
            state['x'], state['y_true'] = data
        else:
            state['x'] = data
