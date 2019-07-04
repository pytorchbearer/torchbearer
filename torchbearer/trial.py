import sys

if sys.version_info[0] < 3:
    import inspect
    def get_default(fcn, arg):
        a = inspect.getargspec(fcn)
        return dict(zip(a.args[-len(a.defaults):], a.defaults))[arg]
else:
    from inspect import signature
    def get_default(fcn, arg):
        return signature(fcn).parameters[arg].default

import functools
import warnings
import itertools

import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer

import torchbearer
from torchbearer import cite
from torchbearer import State
from torchbearer.metrics import MetricList
from torchbearer.callbacks import Callback, CallbackList, Tqdm, AggregatePredictions
from torchbearer.bases import base_closure

bibtex = """
@article{2018torchbearer,
  title={Torchbearer: A Model Fitting Library for PyTorch},
  author={Harris, Ethan and Painter, Matthew and Hare, Jonathon},
  journal={arXiv preprint arXiv:1809.03363},
  year={2018}
}
"""


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
        if closure is not None:
            closure()

    def zero_grad(self):
        pass  # Do Nothing


class MockModel(torch.nn.Module):
    def forward(self, x, state=None):
        return None


class CallbackListInjection(CallbackList):
    """This class allows for an callback to be injected into a callback list, without masking the methods available for
    mutating the list. In this way, callbacks (such as printers) can be injected seamlessly into the methods of the
    trial class.

    Args:
        callback (Callback): The :class:`.Callback` to inject
        callback_list (CallbackList): The underlying :class:`.CallbackList`
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

    Args:
        validation_label_letter (str): The validation label letter to use

    Returns:
        A decorator
    """
    from inspect import getcallargs

    def decorator(func):
        root = func if not hasattr(func, 'root') else func.root
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            call_args = getcallargs(root, self, *args, **kwargs)
            verbose = call_args['verbose'] if 'verbose' in call_args else get_default(func, 'verbose')  # Populate default value
            verbose = self.verbose if verbose == -1 else verbose

            printer = get_printer(verbose=verbose, validation_label_letter=validation_label_letter)

            callback_list_old = self.state[torchbearer.CALLBACK_LIST]

            self.state[torchbearer.CALLBACK_LIST] = CallbackListInjection(printer, callback_list_old)

            res = func(self, *args, **kwargs)

            self.state[torchbearer.CALLBACK_LIST] = callback_list_old
            return res
        wrapper.root = root
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
    """ Static method to call :func:`to` on tensors, tuples or dicts. All items will have :func:`deep_to` called

    Example: ::

        >>> import torch
        >>> from torchbearer import deep_to
        >>> example_dict = {'a': torch.ones(5)*2.1, 'b': torch.ones(1)*5.9}
        >>> deep_to(example_dict, device='cpu', dtype=torch.int)
        {'a': tensor([2, 2, 2, 2, 2], dtype=torch.int32), 'b': tensor([5], dtype=torch.int32)}

    Args:
        batch (tuple / list / torch.Tensor / dict): The mini-batch which requires a :func:`to` call
        device (torch.device): The desired device of the batch
        dtype (torch.dtype): The desired datatype of the batch

    Returns:
        tuple / list / torch.Tensor: The moved or casted batch
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
    elif torch.is_tensor(batch):
        if batch.dtype.is_floating_point:
            batch = batch.to(device, dtype)
        else:
            batch = batch.to(device)

    return batch


def load_batch_infinite(loader):
    """ Wraps a batch loader and refreshes the iterator once it has been completed.

    Args:
        loader: batch loader to wrap
    """

    def call(state):
        try:
            loader(state)
        except StopIteration:
            state[torchbearer.ITERATOR] = iter(state[torchbearer.GENERATOR])
            loader(state)

    return call


def load_batch_standard(state):
    """ Load a standard (input data, target) tuple mini-batch from iterator into state

    Args:
        state (dict): The current state dict of the :class:`Trial`.
    """
    state[torchbearer.X], state[torchbearer.Y_TRUE] = deep_to(next(state[torchbearer.ITERATOR]),
                                                              state[torchbearer.DEVICE],
                                                              state[torchbearer.DATA_TYPE])


def load_batch_none(state):
    """ Load a none (none, none) tuple mini-batch into state

    Args:
        state (dict): The current state dict of the :class:`Trial`.
    """
    state[torchbearer.X], state[torchbearer.Y_TRUE] = None, None


def load_batch_predict(state):
    """ Load a prediction (input data, target) or (input data) mini-batch from iterator into state

    Args:
        state (dict): The current state dict of the :class:`Trial`.
    """
    data = deep_to(next(state[torchbearer.ITERATOR]), state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
    if isinstance(data, list) or isinstance(data, tuple):
        try:
            state[torchbearer.X], state[torchbearer.Y_TRUE] = data
        except ValueError:
            state[torchbearer.X] = data[0]
    else:
        state[torchbearer.X] = data


def inject_sampler(data_key, batch_sampler):
    """ Decorator to inject a :class:`Sampler` into state[torchbearer.SAMPLER] along with the specified \
        generator into state[torchbearer.GENERATOR] and number of steps into state[torchbearer.STEPS]

    Args:
        data_key (:class:`.StateKey`): Key for the data to inject
        batch_sampler (function): Batch sampler function that extracts batch from data loader, stores in state and sends
        data to correct device

    Returns:
        The decorator
    """
    from inspect import getcallargs

    def decorator(func):
        root = func if not hasattr(func, 'root') else func.root

        def infinite_wrapper(self, key, generator, steps, sampler):
            if generator is not None and steps is not None:
                over_steps = steps > len(generator)
                inf_steps = steps == -1
                inf_train_loader = key == torchbearer.TRAIN_DATA and self.state[torchbearer.INF_TRAIN_LOADING]

                if over_steps or inf_steps or inf_train_loader:  # Want iterator to refresh at end not per epoch
                    if steps == -1: warnings.warn("Trial is set to run indefinitely. "
                                              "Make sure you have some method to terminate safely.")
                    sampler = load_batch_infinite(sampler)

                # Want iterator to run until end before refreshing regardless of number of train/val steps
                if inf_train_loader and not hasattr(generator, 'tb_iter'):
                    generator.tb_iter = iter(generator)

            return generator, sampler

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            sampler = batch_sampler

            call_args = getcallargs(root, self, *args, **kwargs)
            key = call_args['data_key'] if ('data_key' in call_args and call_args['data_key'] is not None) else data_key  # Populate default value
            generator, steps = self.state[key] if self.state[key] is not None else (None, None)

            if self.state[torchbearer.LOADER] is not None:
                sampler = self.state[torchbearer.LOADER]
            elif generator is None:
                sampler = load_batch_none

            generator, sampler = infinite_wrapper(self, key, generator, steps, sampler)

            self.state[torchbearer.DATA] = key
            self.state[torchbearer.SAMPLER] = sampler
            self.state[torchbearer.GENERATOR] = generator
            self.state[torchbearer.STEPS] = steps

            res = func(self, *args, **kwargs)

            return res
        wrapper.root = root
        return wrapper
    return decorator


def inject_callback(callback):
    """ Decorator to inject a callback into the callback list and remove the callback after the decorated function has executed

    Args:
        callback (Callback): :class:`.Callback` to be injected

    Returns:
        The decorator
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


def update_device_and_dtype(state, *args, **kwargs):
    """Function gets data type and device values from the args / kwargs and updates state.

    Args:
        state (State): The :class:`.State` to update
        args: Arguments to the :func:`Trial.to` function
        kwargs: Keyword arguments to the :func:`Trial.to` function

    Returns:
        state
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


@cite(bibtex)
class Trial(object):
    """
    The trial class contains all of the required hyper-parameters for model running in torchbearer and presents an
    API for model fitting, evaluating and predicting.

    Example: ::

        >>> import torch
        >>> from torchbearer import Trial

        # Example trial that attempts to aims the output of a linear layer.
        # Makes use of a callback to input the random data at each batch and a loss that is the absolute value of the
        # linear layer output. Runs for 10 iterations and a single epoch.
        >>> model = torch.nn.Linear(2,1)
        >>> optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)

        >>> @torchbearer.callbacks.on_sample
        ... def initial_data(state):
        ...     state[torchbearer.X] = torch.rand(1, 2)*10
        >>> def minimise_output_loss(y_pred, y_true):
        ...     return torch.abs(y_pred)
        >>> trial = Trial(model, optimiser, minimise_output_loss, ['loss'], [initial_data]).for_steps(10).run(1)

    Args:
        model (torch.nn.Module): The base pytorch model
        optimizer (torch.optim.Optimizer): The optimizer used for pytorch model weight updates
        criterion (func / None): The final loss criterion that provides a loss value to the optimizer
        metrics (list): The list of :class:`torchbearer.Metric <.Metric>` instances to process during fitting
        callbacks (list): The list of :class:`torchbearer.Callback <.Callback>` instances to call during fitting
        verbose (int): Global verbosity .If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training
            progress
    """
    def __init__(self, model, optimizer=None, criterion=None, metrics=[], callbacks=[], verbose=2):
        if criterion is None:
            def criterion(_, __):
                return torch.zeros(1, device=self.state[torchbearer.DEVICE], dtype=self.state[torchbearer.DATA_TYPE], requires_grad=True)

        self.verbose = verbose

        self.closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE, torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        self.state = State()
        self.state.update({
            torchbearer.MODEL: model if model is not None else MockModel(),
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
            torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.LOADER: None
        })

        self.state[torchbearer.CALLBACK_LIST].on_init(self.state)

    def __str__(self):
        def state_string(name, state_key):
            import math
            N = (50-len(name))/2
            res = "-" * int(math.floor(N)) + " " + name.upper() + " " + "-" * int(math.ceil(N))
            res = res + "-" if len(res) < 52 else res
            return res + "\n" + str(self.state[state_key]) + "\n\n"

        optim_str = state_string('Optimzer', torchbearer.OPTIMIZER)
        crit_str = state_string("Criterion", torchbearer.CRITERION)
        metrics_str = state_string("Metrics", torchbearer.METRIC_LIST)
        callbacks_str = state_string("Callbacks", torchbearer.CALLBACK_LIST)
        model_str = state_string("Model", torchbearer.MODEL)

        return optim_str + crit_str + metrics_str + callbacks_str + model_str

    def __repr__(self):
        return str(self)

    # Data addition

    def for_train_steps(self, steps):
        """Run this trial for the given number of training steps. Note that the generator will output (None, None) if it
        has not been set. Useful for differentiable programming. Returns self so that methods can be chained for
        convenience. If steps is larger than dataset size then loader will be refreshed like if it was a new epoch. If
        steps is -1 then loader will be refreshed until stopped by STOP_TRAINING flag or similar.

        Example: ::

            # Simple trial that runs for 100 training iterations, in this case optimising nothing
            >>> from torchbearer import Trial
            >>> trial = Trial(None).for_train_steps(100)

        Args:
            steps (int): The number of training steps per epoch to run.

        Returns:
            Trial: self
        """
        if not isinstance(steps, int):
            warnings.warn("Number of training steps is not an int, casting to int")
            steps = int(steps)
        self.state[torchbearer.TRAIN_STEPS] = steps
        self.state[torchbearer.TRAIN_DATA] = (self.state[torchbearer.TRAIN_GENERATOR], self.state[torchbearer.TRAIN_STEPS])

        return self

    def with_train_generator(self, generator, steps=None):
        """Use this trial with the given train generator. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs for 100 training iterations on the MNIST dataset
            >>> from torchbearer import Trial
            >>> from torchvision.datasets import MNIST
            >>> from torch.utils.data import DataLoader
            >>> dataloader = DataLoader(MNIST('./data/', train=True))
            >>> trial = Trial(None).with_train_generator(dataloader).for_steps(100).run(1)

        Args:
            generator: The train data generator to use during calls to :meth:`.run`
            steps (int): The number of steps per epoch to take when using this generator.

        Returns:
            Trial: self
        """
        self.state[torchbearer.TRAIN_GENERATOR] = generator
        steps = self.state[torchbearer.TRAIN_STEPS] if steps is None else steps
        steps = len(generator) if steps is None else steps
        self.for_train_steps(steps)

        return self

    def with_train_data(self, x, y, batch_size=1, shuffle=True, num_workers=1, steps=None):
        """Use this trial with the given train data. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs for 10 training iterations on some random data
            >>> from torchbearer import Trial
            >>> data = torch.rand(10, 1)
            >>> targets = torch.rand(10, 1)
            >>> trial = Trial(None).with_val_data(data, targets).for_steps(10).run(1)

        Args:
            x (torch.Tensor): The train x data to use during calls to :meth:`.run`
            y (torch.Tensor): The train labels to use during calls to :meth:`.run`
            batch_size (int): The size of each batch to sample from the data
            shuffle (bool): If True, then data will be shuffled each epoch
            num_workers (int): Number of worker threads to use in the data loader
            steps (int): The number of steps per epoch to take when using this data

        Returns:
            Trial: self
        """
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
        self.with_train_generator(dataloader, steps=steps)

        return self

    def for_val_steps(self, steps):
        """Run this trial for the given number of validation steps. Note that the generator will output (None, None) if
        it has not been set. Useful for differentiable programming. Returns self so that methods can be chained for
        convenience. If steps larger than dataset size then loader will be refreshed like if it was a new epoch. If
        steps -1 then loader will be refreshed until stopped by STOP_TRAINING flag or similar.

        Example: ::

            # Simple trial that runs for 10 validation iterations on no data
            >>> from torchbearer import Trial
            >>> data = torch.rand(10, 1)
            >>> trial = Trial(None).for_val_steps(10).run(1)

        Args:
            steps (int): The number of validation steps per epoch to run

        Returns:
            Trial: self
        """
        if not isinstance(steps, int):
            warnings.warn("Number of validation steps is not an int, casting to int")
            steps = int(steps)
        self.state[torchbearer.VALIDATION_STEPS] = steps
        self.state[torchbearer.VALIDATION_DATA] = (self.state[torchbearer.VALIDATION_GENERATOR], self.state[torchbearer.VALIDATION_STEPS])

        return self

    def with_val_generator(self, generator, steps=None):
        """Use this trial with the given validation generator. Returns self so that methods can be chained for
        convenience.

        Example: ::

            # Simple trial that runs for 100 validation iterations on the MNIST dataset
            >>> from torchbearer import Trial
            >>> from torchvision.datasets import MNIST
            >>> from torch.utils.data import DataLoader
            >>> dataloader = DataLoader(MNIST('./data/', train=False))
            >>> trial = Trial(None).with_val_generator(dataloader).for_steps(100).run(1)

        Args:
            generator: The validation data generator to use during calls to :meth:`.run` and :meth:`.evaluate`
            steps (int): The number of steps per epoch to take when using this generator

        Returns:
            Trial: self
        """
        self.state[torchbearer.VALIDATION_GENERATOR] = generator
        steps = self.state[torchbearer.VALIDATION_STEPS] if steps is None else steps
        steps = len(generator) if steps is None else steps
        self.for_val_steps(steps)

        return self

    def with_val_data(self, x, y, batch_size=1, shuffle=True, num_workers=1, steps=None):
        """Use this trial with the given validation data. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs for 10 validation iterations on some random data
            >>> from torchbearer import Trial
            >>> data = torch.rand(10, 1)
            >>> targets = torch.rand(10, 1)
            >>> trial = Trial(None).with_val_data(data, targets).for_steps(10).run(1)

        Args:
            x (torch.Tensor): The validation x data to use during calls to :meth:`.run` and :meth:`.evaluate`
            y (torch.Tensor): The validation labels to use during calls to :meth:`.run` and :meth:`.evaluate`
            batch_size (int): The size of each batch to sample from the data
            shuffle (bool): If True, then data will be shuffled each epoch
            num_workers (int): Number of worker threads to use in the data loader
            steps (int): The number of steps per epoch to take when using this data

        Returns:
            Trial: self
        """
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
        self.with_val_generator(dataloader, steps=steps)

        return self

    def for_test_steps(self, steps):
        """Run this trial for the given number of test steps. Note that the generator will output (None, None) if
        it has not been set. Useful for differentiable programming. Returns self so that methods can be chained for
        convenience. If steps larger than dataset size then loader will be refreshed like if it was a new epoch. If
        steps -1 then loader will be refreshed until stopped by STOP_TRAINING flag or similar.

        Example: ::

            # Simple trial that runs for 10 test iterations on some random data
            >>> from torchbearer import Trial
            >>> data = torch.rand(10, 1)
            >>> trial = Trial(None).with_test_data(data).for_test_steps(10).run(1)

        Args:
            steps (int): The number of test steps per epoch to run (when using :meth:`.predict`)

        Returns:
            Trial: self
        """
        if not isinstance(steps, int):
            warnings.warn("Number of test steps is not an int, casting to int")
            steps = int(steps)
        self.state[torchbearer.TEST_STEPS] = steps
        self.state[torchbearer.TEST_DATA] = (self.state[torchbearer.TEST_GENERATOR], self.state[torchbearer.TEST_STEPS])

        return self

    def with_test_generator(self, generator, steps=None):
        """Use this trial with the given test generator. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs for 10 test iterations on no data
            >>> from torchbearer import Trial
            >>> data = torch.rand(10, 1)
            >>> trial = Trial(None).with_test_data(data).for_test_steps(10).run(1)

        Args:
            generator: The test data generator to use during calls to :meth:`.predict`
            steps (int): The number of steps per epoch to take when using this generator

        Returns:
            Trial: self
        """
        self.state[torchbearer.TEST_GENERATOR] = generator
        steps = self.state[torchbearer.TEST_STEPS] if steps is None else steps
        steps = len(generator) if steps is None else steps
        self.for_test_steps(steps)

        return self

    def with_test_data(self, x, batch_size=1, num_workers=1, steps=None):
        """Use this trial with the given test data. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs for 10 test iterations on some random data
            >>> from torchbearer import Trial
            >>> data = torch.rand(10, 1)
            >>> trial = Trial(None).with_test_data(data).for_test_steps(10).run(1)

        Args:
            x (torch.Tensor): The test x data to use during calls to :meth:`.predict`
            batch_size (int): The size of each batch to sample from the data
            num_workers (int): Number of worker threads to use in the data loader
            steps (int): The number of steps per epoch to take when using this data

        Returns:
            Trial: self
        """
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
        self.with_test_generator(dataloader, steps=steps)

        return self

    def for_steps(self, train_steps=None, val_steps=None, test_steps=None):
        """Use this trial for the given number of train, val and test steps. Returns self so that methods can be chained
        for convenience. If steps larger than dataset size then loader will be refreshed like if it was a new epoch. If
        steps -1 then loader will be refreshed until stopped by STOP_TRAINING flag or similar.

        Example: ::

            # Simple trial that runs for 10 training, validation and test iterations on some random data
            >>> from torchbearer import Trial
            >>> train_data = torch.rand(10, 1)
            >>> val_data = torch.rand(10, 1)
            >>> test_data = torch.rand(10, 1)
            >>> trial = Trial(None).with_train_data(train_data).with_val_data(val_data).with_test_data(test_data)
            >>> trial.for_steps(10, 10, 10).run(1)

        Args:
            train_steps (int): The number of training steps per epoch to run
            val_steps (int): The number of validation steps per epoch to run
            test_steps (int): The number of test steps per epoch to run (when using :meth:`.predict`)

        Returns:
            Trial: self
        """
        if train_steps is not None:
            self.for_train_steps(train_steps)
        if val_steps is not None:
            self.for_val_steps(val_steps)
        if test_steps is not None:
            self.for_test_steps(test_steps)

        return self

    def with_generators(self, train_generator=None, val_generator=None, test_generator=None, train_steps=None, val_steps=None, test_steps=None):
        """Use this trial with the given generators. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs for 100 steps from a training and validation data generator
            >>> from torchbearer import Trial
            >>> from torchvision.datasets import MNIST
            >>> from torch.utils.data import DataLoader
            >>> trainloader = DataLoader(MNIST('./data/', train=True))
            >>> valloader = DataLoader(MNIST('./data/', train=False))
            >>> trial = Trial(None).with_generators(trainloader, valloader, train_steps=100, val_steps=100).run(1)

        Args:
            train_generator: The training data generator to use during calls to :meth:`.run`
            val_generator: The validation data generator to use during calls to :meth:`.run` and :meth:`.evaluate`
            test_generator: The testing data generator to use during calls to :meth:`.predict`
            train_steps (int): The number of steps per epoch to take when using the training generator
            val_steps (int): The number of steps per epoch to take when using the validation generator
            test_steps (int): The number of steps per epoch to take when using the testing generator

        Returns:
            Trial: self
        """
        if train_generator is not None:
            self.with_train_generator(train_generator, train_steps)
        if val_generator is not None:
            self.with_val_generator(val_generator, val_steps)
        if test_generator is not None:
            self.with_test_generator(test_generator, test_steps)

        return self

    def with_data(self, x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, batch_size=1,
                  num_workers=1, train_steps=None, val_steps=None, test_steps=None, shuffle=True):
        """Use this trial with the given data. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs for 10 test iterations on some random data
            >>> from torchbearer import Trial
            >>> data = torch.rand(10, 1)
            >>> targets = torch.rand(10, 1)
            >>> test_data = torch.rand(10, 1)
            >>> trial = Trial(None).with_data(x_train=data, y_train=targets, x_test=test_data)
            >>> trial.for_test_steps(10).run(1)

        Args:
            x_train (torch.Tensor): The training data to use
            y_train (torch.Tensor): The training targets to use
            x_val (torch.Tensor): The validation data to use
            y_val (torch.Tensor): The validation targets to use
            x_test (torch.Tensor): The test data to use
            batch_size (int): Batch size to use in mini-batching
            num_workers (int): Number of workers to use for data loading and batching
            train_steps (int): Number of steps for each training pass
            val_steps (int): Number of steps for each validation pass
            test_steps (int): Number of steps for each test pass
            shuffle (bool): If True, shuffle training and validation data.

        Returns:
            Trial: self
        """
        self.with_train_data(x_train, y_train, batch_size, shuffle, num_workers, train_steps)
        self.with_val_data(x_val, y_val, batch_size, shuffle, num_workers, val_steps)
        self.with_test_data(x_test, batch_size, num_workers, test_steps)

    # Infinite steps and loading

    def for_inf_train_steps(self):
        """Use this trial with an infinite number of training steps (until stopped via STOP_TRAINING flag or similar). 
        Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs training data until stopped
            >>> from torchbearer import Trial
            >>> from torchvision.datasets import MNIST
            >>> from torch.utils.data import DataLoader
            >>> trainloader = DataLoader(MNIST('./data/', train=True))
            >>> trial = Trial(None).with_train_generator(trainloader).for_inf_train_steps()
            >>> trial.run(1)

        Returns:
            Trial: self
        """
        self.for_train_steps(-1)
        return self

    def for_inf_val_steps(self):
        """Use this trial with an infinite number of validation steps (until stopped via STOP_TRAINING flag or similar).
        Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs validation data until stopped
            >>> from torchbearer import Trial
            >>> from torchvision.datasets import MNIST
            >>> from torch.utils.data import DataLoader
            >>> valloader = DataLoader(MNIST('./data/', train=False))
            >>> trial = Trial(None).with_val_generator(valloader).for_inf_val_steps()
            >>> trial.run(1)

        Returns:
            Trial: self
        """
        self.for_val_steps(-1)
        return self

    def for_inf_test_steps(self):
        """Use this trial with an infinite number of test steps (until stopped via STOP_TRAINING flag or similar). 
        Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs test data until stopped
            >>> from torchbearer import Trial
            >>> test_data = torch.rand(1000, 10)
            >>> trial = Trial(None).with_test_data(test_data).for_inf_test_steps()
            >>> trial.run(1)

        Returns:
            Trial: self
        """
        self.for_test_steps(-1)
        return self

    def for_inf_steps(self, train=True, val=True, test=True):
        """Use this trail with infinite steps. Returns self so that methods can be chained for convenience.

        Example: ::

            # Simple trial that runs training and test data until stopped
            >>> from torchbearer import Trial
            >>> from torchvision.datasets import MNIST
            >>> from torch.utils.data import DataLoader
            >>> trainloader = DataLoader(MNIST('./data/', train=True))
            >>> valloader = DataLoader(MNIST('./data/', train=False))
            >>> trial = Trial(None).with_train_generator(trainloader).for_inf_steps(valloader)
            >>> trial.with_inf_test_loader(True, False, True).run(1)

        Args:
            train (bool): Use an infinite number of training steps
            val (bool): Use an infinite number of validation steps
            test (bool): Use an infinite number of test steps

        Returns:
            Trial: self
        """
        if train: self.for_inf_train_steps()
        if val: self.for_inf_val_steps()
        if test: self.for_inf_test_steps()

        return self

    def with_inf_train_loader(self):
        """Use this trial with a training iterator that refreshes when it finishes instead of each epoch. 
        This allows for setting training steps less than the size of the generator and model will still be trained on 
        all training samples if enough "epochs" are run.

        Example: ::

            # Simple trial that runs 10 epochs of 100 iterations of a training generator without reshuffling until all data has been seen
            >>> from torchbearer import Trial
            >>> from torchvision.datasets import MNIST
            >>> from torch.utils.data import DataLoader
            >>> trainloader = DataLoader(MNIST('./data/', train=True))
            >>> trial = Trial(None).with_train_generator(trainloader).with_inf_train_loader()
            >>> trial.run(10)

        Returns:
            Trial: self:
        """
        self.state[torchbearer.INF_TRAIN_LOADING] = True

        return self

    # Customise training loop

    def with_loader(self, batch_loader):
        """Use this trial with custom batch loader. Usually calls next on state[torchbearer.ITERATOR] and populates
        state[torchbearer.X] and state[torchbearer.Y_TRUE]

        Example: ::

            # Simple trial that runs with a custom loader function that populates X and Y_TRUE in state with random data
            >>> from torchbearer import Trial
            >>> def custom_loader(state):
            ...     state[X], state[Y_TRUE] = torch.rand(5, 5), torch.rand(5, 5)
            >>> trial = Trial(None).with_loader(custom_loader)
            >>> trial.run(10)

        Args:
            batch_loader (function): Function of state that extracts data from data loader (stored under
                torchbearer.ITERATOR), stores it in state and sends it to the correct device

        Returns:
            Trial: self:
        """
        self.state[torchbearer.LOADER] = batch_loader
        return self

    def with_closure(self, closure):
        """Use this trial with custom closure

        Example: ::

            # Simple trial that runs with a custom closure
            >>> from torchbearer import Trial
            >>> def custom_closure(state):
            ...     print(state[torchbearer.BATCH])
            >>> trial = Trial(None).with_closure(custom_closure).for_steps(3)
            >>> _ = trial.run(1)
            0
            1
            2

        Args:
            closure (function): Function of state that defines the custom closure

        Returns:
            Trial: self:
        """
        self.closure = closure

        return self

    # Run

    @inject_printer()
    def run(self, epochs=1, verbose=-1):
        r"""Run this trial for the given number of epochs, starting from the last trained epoch.

        Example: ::

            # Simple trial that runs with a custom closure
            >>> from torchbearer import Trial
            >>> trial = Trial(None).for_steps(100)
            >>> _ = trial.run(1)

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

        if self.state[torchbearer.MODEL] is None or not callable(self.state[torchbearer.MODEL]):
            warnings.warn('The Model is None or not callable which may cause issues if not deliberate')
            self.state[torchbearer.MODEL] = MockModel()

        state.update(self.state)  # TODO: Swap this for something which makes `self.state` still mutable

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
                steps_summary = {str(torchbearer.TRAIN_STEPS): state[torchbearer.TRAIN_STEPS], str(torchbearer.VALIDATION_STEPS): state[torchbearer.VALIDATION_STEPS]}
                self.state[torchbearer.HISTORY].append(dict(state[torchbearer.METRICS], **steps_summary))
                state[torchbearer.CALLBACK_LIST].on_checkpoint(state)

                if state[torchbearer.STOP_TRAINING]:
                    break

            state[torchbearer.CALLBACK_LIST].on_end(state)

        return self.state[torchbearer.HISTORY]

    @staticmethod
    def _new_iter(generator):
        if generator is None:
            return None
        if hasattr(generator, 'inf') and generator.inf:  # Inf train loader deals with the iterator itself
            return generator.tb_iter
        else:
            return iter(generator)

    @inject_sampler(torchbearer.TRAIN_DATA, load_batch_standard)
    def _fit_pass(self, state):
        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable
        self.train()

        state[torchbearer.ITERATOR] = Trial._new_iter(state[torchbearer.GENERATOR])

        state[torchbearer.METRIC_LIST].reset(state)
        state[torchbearer.METRICS] = {}

        state[torchbearer.STEPS] = 0 if state[torchbearer.STEPS] is None else state[torchbearer.STEPS]
        state[torchbearer.CALLBACK_LIST].on_start_training(state)
        for state[torchbearer.BATCH] in (range(state[torchbearer.STEPS]) if state[torchbearer.STEPS] != -1 else itertools.count()):
            state[torchbearer.SAMPLER](state)
            state[torchbearer.CALLBACK_LIST].on_sample(state)

            # Update parameters
            state[torchbearer.OPTIMIZER].step(lambda: self.closure(state))

            state[torchbearer.METRICS] = state[torchbearer.METRIC_LIST].process(state.data)
            state[torchbearer.CALLBACK_LIST].on_step_training(state)

            if state[torchbearer.STOP_TRAINING]:
                break

        state[torchbearer.METRICS].update(state[torchbearer.METRIC_LIST].process_final(state.data))

        state[torchbearer.CALLBACK_LIST].on_end_training(state)
        return state

    def _test_pass(self, state):
        with torch.no_grad():
            state[torchbearer.ITERATOR] = Trial._new_iter(state[torchbearer.GENERATOR])

            state[torchbearer.METRIC_LIST].reset(state)
            state[torchbearer.METRICS] = {}

            state[torchbearer.CALLBACK_LIST].on_start_validation(state)

            state[torchbearer.STEPS] = 0 if state[torchbearer.STEPS] is None else state[torchbearer.STEPS]
            for state[torchbearer.BATCH] in range(state[torchbearer.STEPS]):
                state[torchbearer.SAMPLER](state)
                state[torchbearer.CALLBACK_LIST].on_sample_validation(state)

                # Forward Pass
                try:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X], state=state)
                except TypeError:
                    state[torchbearer.Y_PRED] = state[torchbearer.MODEL](state[torchbearer.X])

                state[torchbearer.CALLBACK_LIST].on_forward_validation(state)

                # Loss and metrics
                if torchbearer.Y_TRUE in state:
                    # Loss Calculation
                    try:
                        state[torchbearer.LOSS] = state[torchbearer.CRITERION](state)
                    except TypeError:
                        state[torchbearer.LOSS] = state[torchbearer.CRITERION](state[torchbearer.Y_PRED],
                                                                           state[torchbearer.Y_TRUE])
                    state[torchbearer.CALLBACK_LIST].on_criterion_validation(state)
                    state[torchbearer.METRICS] = state[torchbearer.METRIC_LIST].process(state.data)

                state[torchbearer.CALLBACK_LIST].on_step_validation(state)
                if state[torchbearer.STOP_TRAINING]:
                    break

            if torchbearer.Y_TRUE in state:
                state[torchbearer.METRICS].update(state[torchbearer.METRIC_LIST].process_final(state.data))
            state[torchbearer.CALLBACK_LIST].on_end_validation(state)
        return state

    @inject_sampler(torchbearer.VALIDATION_DATA, load_batch_standard)
    def _validation_pass(self, state):
        state.update(self.state)  # TODO: Hack to make injection work, should be removed if `self.state` is mutable

        if state[torchbearer.VALIDATION_GENERATOR] is not None or state[torchbearer.VALIDATION_STEPS] is not None:
            self.eval()

            self._test_pass(state)
        return state[torchbearer.METRICS]

    @inject_sampler(torchbearer.VALIDATION_DATA, load_batch_standard)
    @inject_printer(validation_label_letter='e')
    def evaluate(self, verbose=-1, data_key=None):  # Note: kwargs appear unused but are inspected in inject_sampler
        """Evaluate this trial on the validation data.

        Example: ::

            # Simple trial to evaluate on both validation and test data
            >>> from torchbearer import Trial
            >>> test_data = torch.rand(5, 5)
            >>> val_data = torch.rand(5, 5)
            >>> t = Trial(None).with_val_data(val_data).with_test_data(test_data)
            >>> t.evaluate(data_key=torchbearer.VALIDATION_DATA).evaluate(data_key=torchbearer.TEST_DATA)

        Args:
            verbose (int): If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training progress, If -1: Automatic
            data_key (StateKey): Optional :class:`.StateKey` for the data to evaluate on. Default: torchbearer.VALIDATION_DATA

        Returns:
            dict: The final metric values
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
                self.state[torchbearer.HISTORY][-1].update(state[torchbearer.METRICS])

            state[torchbearer.CALLBACK_LIST].on_end(state)
            return state[torchbearer.METRICS]
        return {}

    @inject_callback(AggregatePredictions())
    @inject_sampler(torchbearer.TEST_DATA, load_batch_predict)
    @inject_printer(validation_label_letter='p')
    def predict(self, verbose=-1, data_key=None):  # Note: kwargs appear unused but are inspected in inject_sampler
        """Determine predictions for this trial on the test data.

        Example: ::

            # Simple trial to predict on some validation and test data
            >>> from torchbearer import Trial
            >>> val_data = torch.rand(5, 5)
            >>> test_data = torch.rand(5, 5)
            >>> t = Trial(None).with_val_data(val_data).with_test_data(test_data)
            >>> val_predictions = t.predict(data_key=torchbearer.VALIDATION_DATA)
            >>> test_predictions = t.predict(data_key=torchbearer.TEST_DATA)

        Args:
            verbose (int): If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training progress, If -1: Automatic
            data_key (StateKey): Optional :class:`.StateKey` for the data to predict on. Default: torchbearer.TEST_DATA

        Returns:
            list: Model outputs as a list
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
            res = self._test_pass(state)[torchbearer.FINAL_PREDICTIONS]

            state[torchbearer.CALLBACK_LIST].on_end_epoch(state)
            state[torchbearer.CALLBACK_LIST].on_end(state)
            return res
        return []

    def replay(self, callbacks=None, verbose=2, one_batch=False):  # TODO: Should we track if testing passes have happened?
        """ Replay the fit passes stored in history with given callbacks, useful when reloading a saved Trial. Note that only progress and metric information is populated in state during a replay.

        Example: ::

            >>> from torchbearer import Trial
            >>> state = torch.load('some_state.pt')
            >>> t = Trial(None).load_state_dict(state)
            >>> t.replay()

        Args:
            callbacks (list): List of callbacks to be run during the replay
            verbose (int): If 2: use tqdm on batch, If 1: use tqdm on epoch, If 0: display no training progress
            one_batch (bool): If True, only one batch per epoch is replayed. If False, all batches are replayed

        Returns:
            Trial: self
        """
        if callbacks is None:
            callbacks = []
        history = self.state[torchbearer.HISTORY]
        callbacks.append(get_printer(verbose=verbose, validation_label_letter='v'))
        callbacks = CallbackList(callbacks)

        state = State()
        state.update(self.state)
        state[torchbearer.STOP_TRAINING] = False
        state[torchbearer.MAX_EPOCHS] = len(history)

        callbacks.on_start(state)
        for i in range(len(history)):
            metrics = dict(history[i])
            state[torchbearer.EPOCH] = i
            if not one_batch:
                state[torchbearer.TRAIN_STEPS], state[torchbearer.VALIDATION_STEPS] = metrics[str(torchbearer.TRAIN_STEPS)], metrics[str(torchbearer.VALIDATION_STEPS)]
            else:
                state[torchbearer.TRAIN_STEPS], state[torchbearer.VALIDATION_STEPS] =\
                    1 if metrics[str(torchbearer.TRAIN_STEPS)] is not None else None,\
                    1 if metrics[str(torchbearer.VALIDATION_STEPS)] is not None else None
            del metrics[str(torchbearer.TRAIN_STEPS)]
            del metrics[str(torchbearer.VALIDATION_STEPS)]
            state[torchbearer.METRICS] = metrics

            self._replay_pass(state, callbacks)
        callbacks.on_end(state)

        return self

    def _replay_pass(self, state, callback_list):
        callback_list.on_start_epoch(state)
        all_metrics = state[torchbearer.METRICS]

        if state[torchbearer.TRAIN_STEPS] is not None:
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

        if state[torchbearer.VALIDATION_STEPS] is not None:
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

        return self

    # Device management

    def train(self):
        """Set model and metrics to training mode.

        Example: ::
            >>> from torchbearer import Trial
            >>> t = Trial(None).train()

        Returns:
            Trial: self
        """
        self.state[torchbearer.MODEL].train()
        self.state[torchbearer.METRIC_LIST].train()

        return self

    def eval(self):
        """Set model and metrics to evaluation mode

        Example: ::
            >>> from torchbearer import Trial
            >>> t = Trial(None).eval()

        Returns:
            Trial: self
        """
        self.state[torchbearer.MODEL].eval()
        if torchbearer.DATA in self.state:
            self.state[torchbearer.METRIC_LIST].eval(data_key=self.state[torchbearer.DATA])
        else:
            self.state[torchbearer.METRIC_LIST].eval()

        return self

    def to(self, *args, **kwargs):
        """ Moves and/or casts the parameters and buffers.

        Example: ::
            >>> from torchbearer import Trial
            >>> t = Trial(None).to('cuda:1')

        Args:
            args: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_
            kwargs: See: `torch.nn.Module.to <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.to>`_

        Returns:
            Trial: self
        """
        self.state[torchbearer.MODEL].to(*args, **kwargs)

        for state in self.state[torchbearer.OPTIMIZER].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(*args, **kwargs)

        self.state = update_device_and_dtype(self.state, *args, **kwargs)

        return self

    def cuda(self, device=None):
        """ Moves all model parameters and buffers to the GPU.

        Example: ::
            >>> from torchbearer import Trial
            >>> t = Trial(None).cuda()

        Args:
            device (int): if specified, all parameters will be copied to that device

        Returns:
            Trial: self
        """
        if device is None:
            device = torch.cuda.current_device()
        self.to('cuda:' + str(device))

        return self

    def cpu(self):
        """ Moves all model parameters and buffers to the CPU.

        Example: ::
            >>> from torchbearer import Trial
            >>> t = Trial(None).cpu()

        Returns:
            Trial: self
        """
        self.to('cpu')

        return self

    # States

    def state_dict(self, **kwargs):
        """Get a dict containing the model and optimizer states, as well as the model history.

        Example: ::
            >>> from torchbearer import Trial
            >>> t = Trial(None)
            >>> state = t.state_dict() # State dict that can now be saved with torch.save

        Args:
            kwargs: See: `torch.nn.Module.state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.state_dict>`_

        Returns:
            dict: A dict containing parameters and persistent buffers.
        """
        state_dict = {
            torchbearer.VERSION: torchbearer.__version__.replace('.dev', ''),
            torchbearer.MODEL: self.state[torchbearer.MODEL].state_dict(**kwargs),
            torchbearer.OPTIMIZER: self.state[torchbearer.OPTIMIZER].state_dict(),
            torchbearer.HISTORY: self.state[torchbearer.HISTORY],
            torchbearer.CALLBACK_LIST: self.state[torchbearer.CALLBACK_LIST].state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict, resume=True, **kwargs):
        """Resume this trial from the given state. Expects that this trial was constructed in the same way. Optionally,
        just load the model state when resume=False.

        Example: ::
            >>> from torchbearer import Trial
            >>> t = Trial(None)
            >>> state = torch.load('some_state.pt')
            >>> t.load_state_dict(state)

        Args:
            state_dict (dict): The state dict to reload
            resume (bool): If True, resume from the given state. Else, just load in the model weights.
            kwargs: See: `torch.nn.Module.load_state_dict <https://pytorch.org/docs/stable/nn.html?highlight=#torch.nn.Module.load_state_dict>`_

        Returns:
            Trial: self
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

        return self
