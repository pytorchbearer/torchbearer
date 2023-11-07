import functools
import traceback
import warnings

import torch
from packaging import version

import torchbearer

import sys

if sys.version_info[0] < 3:
    def set_doc(inner, doc):
        return None  # Not simple to do in Python 2.7 so we can leave it for now, just build docs with Python 3+
else:
    def set_doc(inner, doc):
        inner.__doc__ = doc


def _pytorch_version_lt(version_string):
    ver = torch.__version__ if 'TorchVersion' in str(type(torch.__version__)) or str(
        torch.__version__) is torch.__version__ else "0.4.0"

    return version.parse(ver) < version.parse(version_string)


def _pytorch_version_gt(version_string):
    ver = torch.__version__ if 'TorchVersion' in str(type(torch.__version__)) or str(
        torch.__version__) is torch.__version__ else "0.4.0"

    return version.parse(ver) > version.parse(version_string)


class no_grad(torch.no_grad):
    """ Context-manager and decorator that disables gradient calculation.
    See `torch.autograd.no_grad <https://pytorch.org/docs/stable/autograd.html#torch.autograd.no_grad>`_
    """

    def __init__(self):
        super(no_grad, self).__init__()
        if _pytorch_version_lt("0.4.1"):  # No grad is not a decorator
            _patch_call(self, self.call)

    def call(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


def _patch_call(instance, func):
    class _(type(instance)):
        def __call__(self, *arg, **kwarg):
            return func(*arg, **kwarg)

    instance.__class__ = _


class enable_grad(torch.enable_grad):
    """ Context-manager and decorator that enables gradient calculation.
    See `torch.autograd.enable_grad <https://pytorch.org/docs/stable/autograd.html#torch.autograd.enable_grad>`_
    """

    def __init__(self):
        super(enable_grad, self).__init__()
        if _pytorch_version_lt("0.4.1"):  # Enable grad is not a decorator
            _patch_call(self, self.call)

    def call(self, func):
        @functools.wraps(func)
        def decorate_enable_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_enable_grad


class Metric(object):
    """Base metric class. Process will be called on each batch, process-final at the end of each epoch.
    The metric contract allows for metrics to take any args but not kwargs. The initial metric call will be given state,
    however, subsequent metrics can pass any values desired.

    .. note::

        All metrics must extend this class.

    Args:
        name (str): The name of the metric
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def process(self, *args):
        """Process the state and update the metric for one iteration.

        Args:
            args: Arguments given to the metric. If this is a root level metric, will be given state

        Returns:
            None, or the value of the metric for this batch
        """
        pass

    def process_final(self, *args):
        """Process the terminal state and output the final value of the metric.

        Args:
            args: Arguments given to the metric. If this is a root level metric, will be given state

        Returns:
            None or the value of the metric for this epoch
        """
        pass

    def eval(self, data_key=None):
        """Put the metric in eval mode during model validation.
        """
        pass

    def train(self):
        """Put the metric in train mode during model training.
        """
        pass

    def reset(self, state):
        """Reset the metric, called before the start of an epoch.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass


class Callback(object):
    """Base callback class.

    .. note::

        All callbacks should override this class.

    """

    def state_dict(self):
        """Get a dict containing the callback state.

        Returns:
            dict: A dict containing parameters and persistent buffers.
        """
        return {}

    def __str__(self):
        return str(self.__class__).replace('<class ', '').replace('>', '').replace("'", "")

    def load_state_dict(self, state_dict):
        """Resume this callback from the given state. Expects that this callback was constructed in the same way.

        Args:
            state_dict (dict): The state dict to reload

        Returns:
            Callback: self
        """
        return self

    def on_init(self, state):
        """Perform some action with the given state as context at the init of a trial instance

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_start(self, state):
        """Perform some action with the given state as context at the start of a model fit.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_start_epoch(self, state):
        """Perform some action with the given state as context at the start of each epoch.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_start_training(self, state):
        """Perform some action with the given state as context at the start of the training loop.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_sample(self, state):
        """Perform some action with the given state as context after data has been sampled from the generator.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_forward(self, state):
        """Perform some action with the given state as context after the forward pass (model output) has been completed.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_criterion(self, state):
        """Perform some action with the given state as context after the criterion has been evaluated.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_backward(self, state):
        """Perform some action with the given state as context after backward has been called on the loss.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_step_training(self, state):
        """Perform some action with the given state as context after step has been called on the optimiser.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_end_training(self, state):
        """Perform some action with the given state as context after the training loop has completed.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_start_validation(self, state):
        """Perform some action with the given state as context at the start of the validation loop.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_sample_validation(self, state):
        """Perform some action with the given state as context after data has been sampled from the validation generator.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_forward_validation(self, state):
        """Perform some action with the given state as context after the forward pass (model output) has been completed
        with the validation data.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_criterion_validation(self, state):
        """Perform some action with the given state as context after the criterion evaluation has been completed
        with the validation data.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_step_validation(self, state):
        """Perform some action with the given state as context at the end of each validation step.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_end_validation(self, state):
        """Perform some action with the given state as context at the end of the validation loop.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_end_epoch(self, state):
        """Perform some action with the given state as context at the end of each epoch.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_checkpoint(self, state):
        """Perform some action with the state after all other callbacks have completed at the end of an epoch and the
        history has been updated. Should only be used for taking checkpoints or snapshots and will only be called by the
        run method of Trial.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass

    def on_end(self, state):
        """Perform some action with the given state as context at the end of the model fitting.

        Args:
            state (dict): The current state dict of the :class:`.Trial`.
        """
        pass


def _get_param_list(param):
    if isinstance(param, list):
        return param
    if isinstance(param, tuple):
        return list(param)
    return [param]


def _forward_with_exceptions(x, model, y_pred, state):
    dx = state[x]

    # Forward Pass
    try:
        exc_info = sys.exc_info()
        state[y_pred] = state[model](*_get_param_list(dx), state=state)
    except Exception as e:
        error = []
        try:
            state[y_pred] = state[model](*_get_param_list(dx))
        except TypeError as e2:
            if isinstance(e, TypeError):  # If both are type errors, show both.
                error.append(e2)
            error.append(e)
            raise Exception(error)
        except Exception as e2:
            if not isinstance(e, TypeError):
                error.append(e)
            error.append(e2)
            raise Exception(error)
    finally:
        print_trace = False
        for exc in exc_info:
            if exc is not None:
                print_trace = True

        traceback.print_exception(*exc_info) if print_trace else None


def base_closure(x, model, y_pred, y_true, crit, loss, opt):
    """Function to create a standard pytorch closure using objects taken from state under the given keys.

    Args:
        x: State key under which the input data is stored
        model: State key under which the pytorch model is stored
        y_pred: State key under which the predictions will be stored
        y_true: State key under which the targets are stored
        crit: State key under which the criterion function is stored (function of state or (y_pred, y_true))
        loss: State key under which the loss will be stored
        opt: State key under which the optimsiser is stored

    Returns:
        function: Standard closure function
    """

    def closure(state):
        # Zero grads
        state[opt].zero_grad()

        _forward_with_exceptions(x, model, y_pred, state)

        state[torchbearer.CALLBACK_LIST].on_forward(state)

        # Loss Calculation
        try:
            state[loss] = state[crit](state)
        except TypeError:
            loss_function_params = _get_param_list(state[y_pred]) + _get_param_list(state[y_true])
            state[loss] = state[crit](*loss_function_params)

        state[torchbearer.CALLBACK_LIST].on_criterion(state)

        # Backwards pass
        state[loss].backward(**state[torchbearer.BACKWARD_ARGS])

        state[torchbearer.CALLBACK_LIST].on_backward(state)

    return closure


standard_closure = lambda: base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                                        torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)


def apex_closure():
    from apex import amp

    def _apex_closure(state):
        # Zero grads
        state[torchbearer.OPTIMIZER].zero_grad()

        _forward_with_exceptions(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, state)

        state[torchbearer.CALLBACK_LIST].on_forward(state)

        # Loss Calculation
        try:
            state[torchbearer.LOSS] = state[torchbearer.CRITERION](state)
        except TypeError:
            loss_function_params = _get_param_list(state[torchbearer.Y_PRED]) + _get_param_list(
                state[torchbearer.Y_TRUE])
            state[torchbearer.LOSS] = state[torchbearer.CRITERION](*loss_function_params)

        state[torchbearer.CALLBACK_LIST].on_criterion(state)

        # Backwards pass
        with amp.scale_loss(state[torchbearer.LOSS], state[torchbearer.OPTIMIZER]) as scaled_loss:
            scaled_loss.backward(**state[torchbearer.BACKWARD_ARGS])

        state[torchbearer.CALLBACK_LIST].on_backward(state)

    return _apex_closure


def cite(bibtex):
    """A decorator which adds a reference to the **Google style** docstring of the given object. The ``Args:`` or
    ``Returns:`` line is then prepended with the given bibtex string at runtime. Otherwise, the last line is used.

    Args:
        bibtex (str): The bibtex string to insert

    Returns:
        The decorator
    """

    def decorator(inner):
        doc = inner.__doc__.split('\n')
        i = 0
        s = 0
        for line in doc:
            sline = line.strip()
            if sline == 'Args:' or sline == 'Returns:':
                for char in line:
                    if char == ' ':
                        s += 1
                break
            i += 1

        spaces = ' ' * (s + 4)
        to_insert = ' ' * s + '::\n\n' + spaces
        to_insert += bibtex.strip().replace('\n', '\n' + spaces).rstrip()

        doc.insert(i, '')
        doc.insert(i, to_insert)
        set_doc(inner, '\n'.join(doc))
        return inner

    return decorator


def get_metric(self_tag, state, metric_key):
    if torchbearer.DATA in state and state[torchbearer.DATA] == 'test_data' and 'val_' in metric_key:
        return None

    if metric_key in state[torchbearer.METRICS]:
        return state[torchbearer.METRICS][metric_key]
    else:
        warnings.warn('{}: Failed to retrieve key `{}` from metrics. '.format(self_tag, metric_key))
