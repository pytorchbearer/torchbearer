import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchbearer
from torchbearer import State
from torchbearer.metrics import MetricList
from torchbearer.callbacks import Callback, CallbackList, Tqdm, AggregatePredictions


class CallbackListInjection(CallbackList):
    """This class allows for an callback to be injected into a callback list, without masking the methods available for
    mutating the list. In this way, callbacks (such as printers) can be injected seamlessly into the methods of the
    trial class.

    :param callback: The callback to inject
    :param callback_list: The underlying callback list
    """
    def __init__(self, callback, callback_list):
        super(CallbackListInjection, self).__init__([])

        self.callback = callback
        self.callback_list = callback_list

    def __iter__(self):
        return self.callback_list.__iter__()

    def __copy__(self):
        return self.callback_list.copy()

    def copy(self):
        return self.__copy__()

    def append(self, callback_list):
        self.callback_list.append(callback_list)

    def _for_list(self, function):
        function(self.callback)  # Call injected callback BEFORE the callback list
        function(self.callback_list)


def inject_printer(validation_label_letter='v'):
    """The inject printer decorator is used to inject the appropriate printer callback, according to the verbosity level.

    :param validation_label_letter: The validation label letter to use
    :return: A decorator
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            import inspect
            verbose = kwargs['verbose'] if 'verbose' in kwargs else inspect.signature(func).parameters['verbose'].default  # Populate default value

            if verbose >= 2:
                printer = Tqdm(validation_label_letter=validation_label_letter)
            elif verbose >= 1:
                printer = Tqdm(validation_label_letter=validation_label_letter, on_epoch=True)
            else:
                printer = Callback()

            callback_list_old = self.state[torchbearer.CALLBACK_LIST]

            self.state[torchbearer.CALLBACK_LIST] = CallbackListInjection(printer, callback_list_old)

            res = func(self, *args, **kwargs)

            self.state[torchbearer.CALLBACK_LIST] = callback_list_old
            return res
        return wrapper
    return decorator


def deep_to(batch, device, dtype):
    """ Static method to call :func:`to` on tensors or tuples. All items in tuple will have :func:`deep_to` called
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
            batch[i] = deep_to(batch[i], device, dtype)
        batch = tuple(batch) if is_tuple else batch
    elif isinstance(batch, dict):
        for key in batch:
            batch[key] = deep_to(batch[key], device, dtype)
    else:
        if batch.dtype.is_floating_point:
            batch = batch.to(device, dtype)
        else:
            batch = batch.to(device)

    return batch


def load_batch_standard(state):
    """ Callback to load a standard (input data, target) tuple mini-batch from iterator into state

    :param state: The current state dict of the :class:`Model`.
    :type state: dict[str,any]
    """
    state[torchbearer.X], state[torchbearer.Y_TRUE] = deep_to(next(state[torchbearer.ITERATOR]),
                                                              state[torchbearer.DEVICE],
                                                              state[torchbearer.DATA_TYPE])


def load_batch_none(state):
    """Callback to load a none (none, none) tuple mini-batch into state

    :param state: The current state dict of the :class:`Model`.
    :type state: dict[str,any]
    """
    state[torchbearer.X], state[torchbearer.Y_TRUE] = None, None


def load_batch_predict(state):
    """ Callback to load a prediction (input data, target) or (input data) mini-batch from iterator into state

    :param state: The current state dict of the :class:`Model`.
    :type state: dict[str,any]
    """
    data = deep_to(next(state[torchbearer.ITERATOR]), state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
    if isinstance(data, list) or isinstance(data, tuple):
        state[torchbearer.X], state[torchbearer.Y_TRUE] = data
    else:
        state[torchbearer.X] = data
        
        
class Sampler:
    def __init__(self, batch_loader):
        super().__init__()
        self.batch_loader = batch_loader

    def sample(self, state):
        self.batch_loader(state)


def inject_sampler(generator, predict=False):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.state[generator] is None:
                loader = load_batch_none
            elif predict:
                loader = load_batch_predict
            else:
                loader = load_batch_standard

            self.state[torchbearer.SAMPLER] = Sampler(loader)

            res = func(self, *args, **kwargs)

            return res
        return wrapper
    return decorator


def inject_callback(callback):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            callback_list_old = self.state[torchbearer.CALLBACK_LIST]

            self.state[torchbearer.CALLBACK_LIST] = CallbackListInjection(callback, callback_list_old)

            res = func(self, *args, **kwargs)

            self.state[torchbearer.CALLBACK_LIST] = callback_list_old
            return res
        return wrapper
    return decorator


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
    :type state: State
    :param args: Arguments to the :func:`Trial.to` function
    :param kwargs: Keyword arguments to the :func:`Trial.to` function
    :return: device, dtype pair
    :rtype: tuple
    """
    for key, _ in kwargs.items():
        if key == str(torchbearer.DATA_TYPE):
            state[torchbearer.DATA_TYPE] = kwargs['dtype']
        elif str(torchbearer.DEVICE) in kwargs:
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
    def __init__(self, model, optimizer=None, criterion=None, metrics=[], callbacks=[], pass_state=False):
        if criterion is None:
            def criterion(_, y_true):
                return torch.zeros(1, device=y_true.device)

        self.pass_state = pass_state

        self.state = State()
        self.state.update({
            torchbearer.MODEL: model,
            torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: MetricList(metrics),
            torchbearer.CALLBACK_LIST: CallbackList(callbacks),
            torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float32,
            torchbearer.SELF: self,
            torchbearer.HISTORY: []
        })

    @fluent
    def with_train_generator(self, generator, steps=None):
        """Use this trial with the given train generator. Returns self so that methods can be chained for convenience.

        :param generator: The train data generator to use during calls to :meth:`.run`
        :type generator: DataLoader
        :param steps: The number of steps per epoch to take when using this generator
        :type steps: int
        :return: self
        :rtype: Trial
        """
        self.state[torchbearer.TRAIN_GENERATOR] = generator

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

        :param x: The train x data to use during calls to :meth:`.run`
        :type x: torch.Tensor
        :param y: The train labels to use during calls to :meth:`.run`
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

        :param generator: The validation data generator to use during calls to :meth:`.run` and :meth:`.evaluate`
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

        :param x: The validation x data to use during calls to :meth:`.run` and :meth:`.evaluate`
        :type x: torch.Tensor
        :param y: The validation labels to use during calls to :meth:`.run` and :meth:`.evaluate`
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

    @fluent
    def with_generators(self, train_generator=None, train_steps=None, val_generator=None, val_steps=None, test_generator=None, test_steps=None):
        """Use this trial with the given generators. Returns self so that methods can be chained for convenience.

        :param train_generator: The training data generator to use during calls to :meth:`.run`
        :type train_generator: DataLoader
        :param train_steps: The number of steps per epoch to take when using the training generator
        :type train_steps: int
        :param val_generator: The validation data generator to use during calls to :meth:`.run` and :meth:`.evaluate`
        :type val_generator: DataLoader
        :param val_steps: The number of steps per epoch to take when using the validation generator
        :type val_steps: int
        :param test_generator: The testing data generator to use during calls to :meth:`.predict`
        :type test_generator: DataLoader
        :param test_steps: The number of steps per epoch to take when using the testing generator
        :type test_steps: int
        :return: self
        :rtype: Trial
        """
        if train_generator is not None:
            self.with_train_generator(train_generator, train_steps)
        if val_generator is not None:
            self.with_val_generator(val_generator, val_steps)
        if test_generator is not None:
            self.with_test_generator(test_generator, test_steps)

    @inject_printer()
    def run(self, epochs=1, verbose=2):
        """Run this trial for the given number of epochs, starting from the last trained epoch.

        :param epochs: The number of epochs to run for
        :type epochs: int
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :return: The model history (dict of epoch metrics)
        :rtype: dict
        """
        state = State()
        state.update({
            torchbearer.MAX_EPOCHS: epochs,
            torchbearer.STOP_TRAINING: False
        })

        state.update(self.state)  # TODO: Swap this for something which makes `self.state` still mutable

        state[torchbearer.CALLBACK_LIST].on_start(state)

        for state[torchbearer.EPOCH] in range(len(state[torchbearer.HISTORY]), state[torchbearer.MAX_EPOCHS]):
            state[torchbearer.CALLBACK_LIST].on_start_epoch(state)

            final_metrics = self._fit_pass(state)[torchbearer.METRICS]

            if state[torchbearer.STOP_TRAINING]:
                break

            final_metrics.update(self._validation_pass(state))
            state[torchbearer.METRICS] = final_metrics
            state[torchbearer.CALLBACK_LIST].on_end_epoch(state)
            self.state[torchbearer.HISTORY].append(state[torchbearer.METRICS])

            if state[torchbearer.STOP_TRAINING]:
                break

        state[torchbearer.CALLBACK_LIST].on_end(state)

        return self.state[torchbearer.HISTORY]

    @inject_sampler(torchbearer.TRAIN_GENERATOR)
    def _fit_pass(self, state):
        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable

        self.train()

        state[torchbearer.STEPS] = state[torchbearer.TRAIN_STEPS]
        state[torchbearer.GENERATOR] = state[torchbearer.TRAIN_GENERATOR]
        state[torchbearer.ITERATOR] = iter(state[torchbearer.GENERATOR]) if state[torchbearer.GENERATOR] is not None else None  # TODO: Inject this?

        state[torchbearer.METRIC_LIST].reset(state)
        state[torchbearer.METRICS] = {}

        state[torchbearer.CALLBACK_LIST].on_start_training(state)

        for state[torchbearer.BATCH] in range(0, state[torchbearer.STEPS]):
            state[torchbearer.SAMPLER].sample(state)
            state[torchbearer.CALLBACK_LIST].on_sample(state)

            # Zero grads
            state[torchbearer.OPTIMIZER].zero_grad()

            # Forward Pass
            if self.pass_state:
                state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X], state=state)
            else:
                state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X])

            state[torchbearer.CALLBACK_LIST].on_forward(state)

            # Loss Calculation
            state[torchbearer.LOSS] = state[torchbearer.CRITERION](state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE])

            state[torchbearer.CALLBACK_LIST].on_criterion(state)
            state[torchbearer.METRICS] = state[torchbearer.METRIC_LIST].process(state)

            # Backwards pass
            state[torchbearer.LOSS].backward()
            state[torchbearer.CALLBACK_LIST].on_backward(state)

            # Update parameters
            state[torchbearer.OPTIMIZER].step()
            state[torchbearer.CALLBACK_LIST].on_step_training(state)

            if state[torchbearer.STOP_TRAINING]:
                break

        state[torchbearer.METRICS].update(state[torchbearer.METRIC_LIST].process_final(state))

        state[torchbearer.CALLBACK_LIST].on_end_training(state)
        return state

    def _test_pass(self, state):
        with torch.no_grad():
            state[torchbearer.ITERATOR] = iter(state[torchbearer.GENERATOR]) if state[torchbearer.GENERATOR] is not None else None  # TODO: Inject this?

            state[torchbearer.METRIC_LIST].reset(state)
            state[torchbearer.METRICS] = {}

            state[torchbearer.CALLBACK_LIST].on_start_validation(state)

            for state[torchbearer.BATCH] in range(state[torchbearer.STEPS]):
                state[torchbearer.SAMPLER].sample(state)
                state[torchbearer.CALLBACK_LIST].on_sample_validation(state)

                # Forward Pass
                if self.pass_state:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X], state=state)
                else:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X])

                state[torchbearer.CALLBACK_LIST].on_forward_validation(state)

                # Loss and metrics
                if torchbearer.Y_TRUE in state:
                    state[torchbearer.LOSS] = state[torchbearer.CRITERION](state[torchbearer.Y_PRED],
                                                                           state[torchbearer.Y_TRUE])
                    state[torchbearer.CALLBACK_LIST].on_criterion_validation(state)
                    state[torchbearer.METRICS] = state[torchbearer.METRIC_LIST].process(state)

                state[torchbearer.CALLBACK_LIST].on_step_validation(state)
                if state[torchbearer.STOP_TRAINING]:
                    break

            if torchbearer.Y_TRUE in state:
                state[torchbearer.METRICS].update(state[torchbearer.METRIC_LIST].process_final(state))
            state[torchbearer.CALLBACK_LIST].on_end_validation(state)
        return state

    @inject_sampler(torchbearer.VALIDATION_GENERATOR)
    def _validation_pass(self, state):
        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable

        if state[torchbearer.VALIDATION_GENERATOR] is not None or state[torchbearer.VALIDATION_STEPS] is not None:
            self.eval()

            state[torchbearer.STEPS] = state[torchbearer.VALIDATION_STEPS]
            state[torchbearer.GENERATOR] = state[torchbearer.VALIDATION_GENERATOR]

            self._test_pass(state)
        return state[torchbearer.METRICS]

    @inject_sampler(torchbearer.VALIDATION_GENERATOR)
    @inject_printer(validation_label_letter='e')
    def evaluate(self, verbose=2):
        """Evaluate this trial on the validation data.

        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :return: The final metric values
        :rtype: dict
        """
        state = State()
        state.update({
            torchbearer.MAX_EPOCHS: 1,
            torchbearer.EPOCH: 0,
            torchbearer.STOP_TRAINING: False
        })

        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable

        if state[torchbearer.VALIDATION_GENERATOR] is not None or state[torchbearer.VALIDATION_STEPS] is not None:
            self.eval()

            state[torchbearer.STEPS] = state[torchbearer.VALIDATION_STEPS]
            state[torchbearer.GENERATOR] = state[torchbearer.VALIDATION_GENERATOR]

            return self._test_pass(state)[torchbearer.METRICS]
        return {}

    @inject_callback(AggregatePredictions())
    @inject_sampler(torchbearer.TEST_GENERATOR, predict=True)
    @inject_printer(validation_label_letter='p')
    def predict(self, verbose=2):
        """Determine predictions for this trial on the test data.

        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, Else: display no training progress
        :type verbose: int
        :return: Model outputs as a list
        :rtype: list
        """
        state = {
            torchbearer.MAX_EPOCHS: 1,
            torchbearer.EPOCH: 0,
            torchbearer.STOP_TRAINING: False
        }

        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable

        if state[torchbearer.TEST_GENERATOR] is not None or state[torchbearer.TEST_STEPS] is not None:
            self.eval()

            state[torchbearer.STEPS] = self.state[torchbearer.TEST_STEPS]
            state[torchbearer.GENERATOR] = self.state[torchbearer.TEST_GENERATOR]

            return self._test_pass(state)[torchbearer.FINAL_PREDICTIONS]
        return []

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
            torchbearer.HISTORY: self.state[torchbearer.HISTORY],
            torchbearer.CALLBACK_LIST: self.state[torchbearer.CALLBACK_LIST].state_dict()
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
            self.state[torchbearer.CALLBACK_LIST].load_state_dict(state_dict[torchbearer.CALLBACK_LIST])
        else:
            self.state[torchbearer.MODEL].load_state_dict(state_dict[torchbearer.MODEL], **kwargs)
