import functools
import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer

import torchbearer
from torchbearer import State
from torchbearer.metrics import MetricList
from torchbearer.callbacks import Callback, CallbackList, Tqdm, AggregatePredictions


class MockOptimizer(Optimizer):
    """The Mock Optimizer will be used inplace of an optimizer in the event that none is passed to the Trial class.
    """
    def __init__(self):
        super(MockOptimizer, self).__init__([torch.zeros(1)], [])

    def add_param_group(self, param_group):
        pass  # Do Nothing

    def load_state_dict(self, state_dict):
        pass  # Do Nothing

    def state_dict(self):
        return {}  # Return Empty

    def step(self, closure=None):
        pass  # Do Nothing

    def zero_grad(self):
        pass  # Do Nothing


class CallbackListInjection(CallbackList):
    """This class allows for an callback to be injected into a callback list, without masking the methods available for
    mutating the list. In this way, callbacks (such as printers) can be injected seamlessly into the methods of the
    trial class.

    :param callback: The callback to inject
    :param callback_list: The underlying callback list
    :type callback_list: CallbackList
    """
    def __init__(self, callback, callback_list):
        super(CallbackListInjection, self).__init__([])

        self.callback = callback
        self.callback_list = callback_list

    def state_dict(self):
        return self.callback_list.state_dict()

    def load_state_dict(self, state_dict):
        self.callback_list.load_state_dict(state_dict)
        return self

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
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            import inspect
            verbose = kwargs['verbose'] if 'verbose' in kwargs else inspect.signature(func).parameters['verbose'].default  # Populate default value
            verbose = self.verbose if verbose == -1 else verbose

            printer = get_printer(verbose=verbose, validation_label_letter=validation_label_letter)

            callback_list_old = self.state[torchbearer.CALLBACK_LIST]

            self.state[torchbearer.CALLBACK_LIST] = CallbackListInjection(printer, callback_list_old)

            res = func(self, *args, **kwargs)

            self.state[torchbearer.CALLBACK_LIST] = callback_list_old
            return res
        return wrapper
    return decorator


def get_printer(verbose, validation_label_letter):
    if verbose >= 2:
        printer = Tqdm(validation_label_letter=validation_label_letter)
    elif verbose >= 1:
        printer = Tqdm(validation_label_letter=validation_label_letter, on_epoch=True)
    else:
        printer = Callback()

    return printer


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
    """ Load a standard (input data, target) tuple mini-batch from iterator into state

    :param state: The current state dict of the :class:`Trial`.
    :type state: dict[str,any]
    """
    state[torchbearer.X], state[torchbearer.Y_TRUE] = deep_to(next(state[torchbearer.ITERATOR]),
                                                              state[torchbearer.DEVICE],
                                                              state[torchbearer.DATA_TYPE])


def load_batch_none(state):
    """ Load a none (none, none) tuple mini-batch into state

    :param state: The current state dict of the :class:`Trial`.
    :type state: dict[str,any]
    """
    state[torchbearer.X], state[torchbearer.Y_TRUE] = None, None


def load_batch_predict(state):
    """ Load a prediction (input data, target) or (input data) mini-batch from iterator into state

    :param state: The current state dict of the :class:`Trial`.
    :type state: dict[str,any]
    """
    data = deep_to(next(state[torchbearer.ITERATOR]), state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
    if isinstance(data, list) or isinstance(data, tuple):
        state[torchbearer.X], state[torchbearer.Y_TRUE] = data
    else:
        state[torchbearer.X] = data
        
        
class Sampler:
    """
    Sampler wraps a batch loader function and executes it when :meth:`Sampler.sample` is called

    :param batch_loader: The batch loader to execute
    :type batch_loader: function
    """
    def __init__(self, batch_loader):
        super().__init__()
        self.batch_loader = batch_loader

    def sample(self, state):
        self.batch_loader(state)


def inject_sampler(data_key, predict=False):
    """ Decorator to inject a :class:`Sampler` into state[torchbearer.SAMPLER] along with the specified \
        generator into state[torchbearer.GENERATOR] and number of steps into state[torchbearer.STEPS]
    :param data_key: Key for the data to inject
    :type data_key: StateKey
    :param predict: If true, the prediction batch loader is used, if false the standard data loader is used
    :type predict: bool
    :return: the decorator
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            key = kwargs['data_key'] if 'data_key' in kwargs else data_key  # Populate default value
            generator, steps = self.state[key] if self.state[key] is not None else (None, None)

            if generator is None:
                loader = load_batch_none
            elif predict:
                loader = load_batch_predict
            else:
                loader = load_batch_standard

            self.state[torchbearer.DATA] = key
            self.state[torchbearer.SAMPLER] = Sampler(loader)
            self.state[torchbearer.GENERATOR] = generator
            self.state[torchbearer.STEPS] = steps

            res = func(self, *args, **kwargs)

            return res
        return wrapper
    return decorator


def inject_callback(callback):
    """ Decorator to inject a callback into the callback list and remove the callback after the decorated function has executed
    
    :param callback: Callback to be injected
    :type callback: Callback
    :return: the decorator
    """
    def decorator(func):
        @functools.wraps(func)
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
    @functools.wraps(func)
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
    :param optimizer: The optimizer used for pytorch model weight updates
    :type optimizer: torch.optim.Optimizer
    :param criterion: The final loss criterion that provides a loss value to the optimizer
    :type criterion: function or None
    :param metrics: The list of :class:`torchbearer.Metric <.Metric>` instances to process during fitting
    :type metrics: list
    :param callbacks: The list of :class:`torchbearer.Callback <.Callback>` instances to call during fitting
    :type callbacks: list
    :param pass_state: If True, the torchbearer state will be passed to the model during fitting
    :type pass_state: bool
    :param verbose: Global verbosity .If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training progress
    :type verbose: int
    """
    def __init__(self, model, optimizer=None, criterion=None, metrics=[], callbacks=[], pass_state=False, verbose=2):
        if criterion is None:
            def criterion(_, __):
                return torch.zeros(1, device=self.state[torchbearer.DEVICE], dtype=self.state[torchbearer.DATA_TYPE], requires_grad=True)

        self.pass_state = pass_state
        self.verbose = verbose

        self.state = State()
        self.state.update({
            torchbearer.MODEL: model,
            torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer if optimizer is not None else MockOptimizer(),
            torchbearer.METRIC_LIST: MetricList(metrics),
            torchbearer.CALLBACK_LIST: CallbackList(callbacks),
            torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float32,
            torchbearer.SELF: self,
            torchbearer.HISTORY: [],
            torchbearer.BACKWARD_ARGS: {},
            torchbearer.TRAIN_GENERATOR: None,
            torchbearer.VALIDATION_GENERATOR: None,
            torchbearer.TEST_GENERATOR: None,
            torchbearer.TRAIN_STEPS: None,
            torchbearer.VALIDATION_STEPS: None,
            torchbearer.TEST_STEPS: None,
            torchbearer.TRAIN_DATA: None,
            torchbearer.VALIDATION_DATA: None,
            torchbearer.TEST_DATA: None,
        })

    def __str__(self):
        def state_string(name, state_key):
            import math
            N = (50-len(name))/2
            return "-"*math.floor(N) + " " + name.upper()+ " " + "-"*math.ceil(N) + "\n" + str(self.state[state_key]) + "\n\n"

        optim_str = state_string('Optimzer', torchbearer.OPTIMIZER)
        crit_str = state_string("Criterion", torchbearer.CRITERION)
        metrics_str = state_string("Metrics", torchbearer.METRIC_LIST)
        callbacks_str = state_string("Callbacks", torchbearer.CALLBACK_LIST)
        model_str = state_string("Model", torchbearer.MODEL)

        return optim_str + crit_str + metrics_str + callbacks_str + model_str

    def __repr__(self):
        return str(self)

    @fluent
    def for_train_steps(self, steps):
        """Run this trial for the given number of training steps. Note that the generator will output (None, None) if it
        has not been set. Useful for differentiable programming. Returns self so that methods can be chained for
        convenience.

        :param steps: The number of training steps per epoch to run
        :type steps: int
        :return: self
        :rtype: Trial
        """
        if not isinstance(steps, int):
            warnings.warn("Number of training steps is not an int, casting to int")
            steps = int(steps)
        generator = self.state[torchbearer.TRAIN_GENERATOR]
        if generator is not None and steps > len(generator):
            warnings.warn("Number of training steps exceeds number of data items, limiting to number of items")
            steps = len(generator)
        self.state[torchbearer.TRAIN_STEPS] = steps
        self.state[torchbearer.TRAIN_DATA] = (self.state[torchbearer.TRAIN_GENERATOR], self.state[torchbearer.TRAIN_STEPS])

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
        self.for_train_steps(steps)

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
    def for_val_steps(self, steps):
        """Run this trial for the given number of validation steps. Note that the generator will output (None, None) if
        it has not been set. Useful for differentiable programming. Returns self so that methods can be chained for
        convenience.

        :param steps: The number of validation steps per epoch to run
        :type steps: int
        :return: self
        :rtype: Trial
        """
        if not isinstance(steps, int):
            warnings.warn("Number of validation steps is not an int, casting to int")
            steps = int(steps)
        generator = self.state[torchbearer.VALIDATION_GENERATOR]
        if generator is not None and steps > len(generator):
            warnings.warn("Number of validation steps exceeds number of data items, limiting to number of items")
            steps = len(generator)
        self.state[torchbearer.VALIDATION_STEPS] = steps
        self.state[torchbearer.VALIDATION_DATA] = (self.state[torchbearer.VALIDATION_GENERATOR], self.state[torchbearer.VALIDATION_STEPS])

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
        self.for_val_steps(steps)

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
    def for_test_steps(self, steps):
        """Run this trial for the given number of test steps. Note that the generator will output (None, None) if
        it has not been set. Useful for differentiable programming. Returns self so that methods can be chained for
        convenience.

        :param steps: The number of test steps per epoch to run (when using :meth:`.predict`)
        :type steps: int
        :return: self
        :rtype: Trial
        """
        if not isinstance(steps, int):
            warnings.warn("Number of test steps is not an int, casting to int")
            steps = int(steps)
        generator = self.state[torchbearer.TEST_GENERATOR]
        if generator is not None and steps > len(generator):
            warnings.warn("Number of test steps exceeds number of data items, limiting to number of items")
            steps = len(generator)
        self.state[torchbearer.TEST_STEPS] = steps
        self.state[torchbearer.TEST_DATA] = (self.state[torchbearer.TEST_GENERATOR], self.state[torchbearer.TEST_STEPS])

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
        self.for_test_steps(steps)

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
    def for_steps(self, train_steps=None, val_steps=None, test_steps=None):
        """Use this trial for the given number of train, val and test steps. Returns self so that methods can be chained
        for convenience.

        :param train_steps: The number of training steps per epoch to run
        :type train_steps: int, optional
        :param val_steps: The number of validation steps per epoch to run
        :type val_steps: int, optional
        :param test_steps: The number of test steps per epoch to run (when using :meth:`.predict`)
        :type test_steps: int, optional
        :return: self
        :rtype: Trial
        """
        if train_steps is not None:
            self.for_train_steps(train_steps)
        if val_steps is not None:
            self.for_val_steps(val_steps)
        if test_steps is not None:
            self.for_test_steps(test_steps)

    @fluent
    def with_generators(self, train_generator=None, val_generator=None, test_generator=None, train_steps=None, val_steps=None, test_steps=None):
        """Use this trial with the given generators. Returns self so that methods can be chained for convenience.

        :param train_generator: The training data generator to use during calls to :meth:`.run`
        :type train_generator: DataLoader
        :param val_generator: The validation data generator to use during calls to :meth:`.run` and :meth:`.evaluate`
        :type val_generator: DataLoader
        :param test_generator: The testing data generator to use during calls to :meth:`.predict`
        :type test_generator: DataLoader
        :param train_steps: The number of steps per epoch to take when using the training generator
        :type train_steps: int
        :param val_steps: The number of steps per epoch to take when using the validation generator
        :type val_steps: int
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
    def run(self, epochs=1, verbose=-1):
        r"""Run this trial for the given number of epochs, starting from the last trained epoch.

        Args:
            epochs (int, optional): The number of epochs to run for
            verbose (int, optional): If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training
            progress, If -1: Automatic

        State Requirements:
            - :attr:`torchbearer.state.MODEL`: Model should be callable and not none, set on Trial init

        Returns:
            list: The model history (list of tuple of steps summary and epoch metric dicts)
        """
        state = State()
        state.update({
            torchbearer.MAX_EPOCHS: epochs,
            torchbearer.STOP_TRAINING: False,
        })

        state.update(self.state)  # TODO: Swap this for something which makes `self.state` still mutable

        if state[torchbearer.MODEL] is None or not callable(state[torchbearer.MODEL]):
            warnings.warn('The Model is None or not callable which may cause issues if not deliberate')
            state[torchbearer.MODEL] = lambda *args, **kwargs: None

        if state[torchbearer.TRAIN_GENERATOR] is not None \
                or state[torchbearer.TRAIN_STEPS] is not None \
                or state[torchbearer.VALIDATION_GENERATOR] is not None \
                or state[torchbearer.VALIDATION_STEPS] is not None:

            state[torchbearer.CALLBACK_LIST].on_start(state)

            for state[torchbearer.EPOCH] in range(len(state[torchbearer.HISTORY]), state[torchbearer.MAX_EPOCHS]):
                state[torchbearer.CALLBACK_LIST].on_start_epoch(state)

                final_metrics = self._fit_pass(state)[torchbearer.METRICS]

                if state[torchbearer.STOP_TRAINING]:
                    break

                final_metrics.update(self._validation_pass(state))
                state[torchbearer.METRICS] = final_metrics
                state[torchbearer.CALLBACK_LIST].on_end_epoch(state)
                steps_summary = (state[torchbearer.TRAIN_STEPS], state[torchbearer.VALIDATION_STEPS])
                self.state[torchbearer.HISTORY].append((steps_summary, state[torchbearer.METRICS]))
                state[torchbearer.CALLBACK_LIST].on_checkpoint(state)

                if state[torchbearer.STOP_TRAINING]:
                    break

            state[torchbearer.CALLBACK_LIST].on_end(state)

        return self.state[torchbearer.HISTORY]

    @inject_sampler(torchbearer.TRAIN_DATA)
    def _fit_pass(self, state):
        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable
        self.train()

        state[torchbearer.ITERATOR] = iter(state[torchbearer.GENERATOR]) if state[torchbearer.GENERATOR] is not None else None  # TODO: Inject this?

        state[torchbearer.METRIC_LIST].reset(state)
        state[torchbearer.METRICS] = {}

        state[torchbearer.CALLBACK_LIST].on_start_training(state)

        for state[torchbearer.BATCH] in range(state[torchbearer.STEPS]):
            state[torchbearer.SAMPLER].sample(state)
            state[torchbearer.CALLBACK_LIST].on_sample(state)

            def closure():
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

                # Backwards pass
                state[torchbearer.LOSS].backward(**state[torchbearer.BACKWARD_ARGS])
                state[torchbearer.CALLBACK_LIST].on_backward(state)

            # Update parameters
            state[torchbearer.OPTIMIZER].step(closure)
            state[torchbearer.METRICS] = state[torchbearer.METRIC_LIST].process(state)
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

    @inject_sampler(torchbearer.VALIDATION_DATA)
    def _validation_pass(self, state):
        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable

        if state[torchbearer.VALIDATION_GENERATOR] is not None or state[torchbearer.VALIDATION_STEPS] is not None:
            self.eval()

            self._test_pass(state)
        return state[torchbearer.METRICS]

    @inject_sampler(torchbearer.VALIDATION_DATA)
    @inject_printer(validation_label_letter='e')
    def evaluate(self, verbose=-1, data_key=None):  # Note: kwargs appear unused but are inspected in inject_sampler
        """Evaluate this trial on the validation data.

        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training progress, If -1: Automatic
        :type verbose: int
        :param data_key: Optional key for the data to evaluate on. Default: torchbearer.VALIDATION_DATA
        :type data_key: StateKey
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

        if state[torchbearer.GENERATOR] is not None or state[torchbearer.STEPS] is not None:
            state[torchbearer.CALLBACK_LIST].on_start(state)
            state[torchbearer.CALLBACK_LIST].on_start_epoch(state)

            self.eval()
            state = self._test_pass(state)

            state[torchbearer.CALLBACK_LIST].on_end_epoch(state)

            if len(self.state[torchbearer.HISTORY]) != 0:
                self.state[torchbearer.HISTORY][-1][1].update(state[torchbearer.METRICS])

            state[torchbearer.CALLBACK_LIST].on_end(state)
            return state[torchbearer.METRICS]
        return {}

    @inject_callback(AggregatePredictions())
    @inject_sampler(torchbearer.TEST_DATA, predict=True)
    @inject_printer(validation_label_letter='p')
    def predict(self, verbose=-1, data_key=None):  # Note: kwargs appear unused but are inspected in inject_sampler
        """Determine predictions for this trial on the test data.

        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training progress, If -1: Automatic
        :type verbose: int
        :param data_key: Optional key for the data to predict on. Default: torchbearer.TEST_DATA
        :type data_key: StateKey
        :return: Model outputs as a list
        :rtype: list
        """
        state = {
            torchbearer.MAX_EPOCHS: 1,
            torchbearer.EPOCH: 0,
            torchbearer.STOP_TRAINING: False
        }

        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable

        if state[torchbearer.GENERATOR] is not None or state[torchbearer.STEPS] is not None:
            state[torchbearer.CALLBACK_LIST].on_start(state)
            state[torchbearer.CALLBACK_LIST].on_start_epoch(state)

            self.eval()
            res = self._test_pass(state)[torchbearer.FINAL_PREDICTIONS]

            state[torchbearer.CALLBACK_LIST].on_end_epoch(state)
            state[torchbearer.CALLBACK_LIST].on_end(state)
            return res
        return []

    @fluent
    def replay(self, callbacks=[], verbose=2):  # TODO: Should we track if testing passes have happened?
        """ Replay the fit passes stored in history with given callbacks, useful when reloading a saved Trial. Note that only progress and metric information is populated in state during a replay.

        :param callbacks: List of callbacks to be run during the replay
        :type callbacks: list
        :param verbose: If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training progress
        :type verbose: int
        :return: self
        :rtype: Trial
        """
        history = self.state[torchbearer.HISTORY]
        callbacks.append(get_printer(verbose=verbose, validation_label_letter='v'))
        callbacks = CallbackList(callbacks)

        state = State()
        state.update(self.state)
        state[torchbearer.STOP_TRAINING] = False
        state[torchbearer.MAX_EPOCHS] = len(history)

        callbacks.on_start(state)
        for i in range(len(history)):
            state[torchbearer.EPOCH] = i
            state[torchbearer.TRAIN_STEPS], state[torchbearer.VALIDATION_STEPS] = history[i][0]
            state[torchbearer.METRICS] = history[i][1]

            self._replay_pass(state, callbacks)
        callbacks.on_end(state)

    @fluent
    def _replay_pass(self, state, callback_list):
        callback_list.on_start_epoch(state)
        all_metrics = state[torchbearer.METRICS]

        # Training pass
        state[torchbearer.STEPS] = state[torchbearer.TRAIN_STEPS]
        state[torchbearer.METRICS] = {key: all_metrics[key] for key in all_metrics.keys() if "val_" not in key}
        callback_list.on_start_training(state)
        for state[torchbearer.BATCH] in range(state[torchbearer.STEPS]):
            callback_list.on_sample(state)
            callback_list.on_forward(state)
            callback_list.on_criterion(state)
            callback_list.on_backward(state)
            callback_list.on_step_training(state)
            if state[torchbearer.STOP_TRAINING]:
                break
        callback_list.on_end_training(state)

        # Validation pass
        if not state[torchbearer.STOP_TRAINING]:
            state[torchbearer.STEPS] = state[torchbearer.VALIDATION_STEPS]
            state[torchbearer.METRICS] = {key: all_metrics[key] for key in all_metrics.keys() if "val_" in key}
            callback_list.on_start_validation(state)
            for state[torchbearer.BATCH] in range(state[torchbearer.STEPS]):
                callback_list.on_sample_validation(state)
                callback_list.on_forward_validation(state)
                callback_list.on_criterion_validation(state)
                callback_list.on_step_validation(state)
                if state[torchbearer.STOP_TRAINING]:
                    break
            callback_list.on_end_validation(state)

        state[torchbearer.METRICS] = all_metrics
        callback_list.on_end_epoch(state)

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
        if torchbearer.DATA in self.state:
            self.state[torchbearer.METRIC_LIST].eval(data_key=self.state[torchbearer.DATA])
        else:
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
            torchbearer.VERSION: torchbearer.__version__.replace('.dev', ''),
            torchbearer.MODEL: self.state[torchbearer.MODEL].state_dict(**kwargs),
            torchbearer.OPTIMIZER: self.state[torchbearer.OPTIMIZER].state_dict(),
            torchbearer.HISTORY: self.state[torchbearer.HISTORY],
            torchbearer.CALLBACK_LIST: self.state[torchbearer.CALLBACK_LIST].state_dict()
        }
        return state_dict

    @fluent
    def load_state_dict(self, state_dict, resume=True, **kwargs):
        """Resume this trial from the given state. Expects that this trial was constructed in the same way. Optionally,
        just load the model state when resume=False.

        :param state_dict: The state dict to reload
        :type state_dict: dict
        :param resume: If True, resume from the given state. Else, just load in the model weights.
        :param kwargs: See: `torch.nn.Module.load_state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.load_state_dict>`_
        :return: self
        :rtype: Trial
        """
        if resume and torchbearer.MODEL in state_dict:  # torchbearer dict
            if torchbearer.VERSION in state_dict and state_dict[torchbearer.VERSION] != torchbearer.__version__.replace('.dev', ''):
                warnings.warn('This state dict was saved with a different torchbearer version, loading available keys. Consider setting resume=False')

            if torchbearer.MODEL in state_dict:
                self.state[torchbearer.MODEL].load_state_dict(state_dict[torchbearer.MODEL], **kwargs)

            if torchbearer.OPTIMIZER in state_dict:
                self.state[torchbearer.OPTIMIZER].load_state_dict(state_dict[torchbearer.OPTIMIZER])

            if torchbearer.HISTORY in state_dict:
                self.state[torchbearer.HISTORY] = state_dict[torchbearer.HISTORY]

            if torchbearer.CALLBACK_LIST in state_dict:
                self.state[torchbearer.CALLBACK_LIST].load_state_dict(state_dict[torchbearer.CALLBACK_LIST])
        elif torchbearer.MODEL in state_dict:
            self.state[torchbearer.MODEL].load_state_dict(state_dict[torchbearer.MODEL], **kwargs)
        else:  # something else
            warnings.warn('Not a torchbearer state dict, passing to model')
            self.state[torchbearer.MODEL].load_state_dict(state_dict, **kwargs)
