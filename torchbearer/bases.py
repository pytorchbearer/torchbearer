from distutils.version import LooseVersion
import functools

import torch
import torchbearer


class no_grad(torch.no_grad):
    def __init__(self):
        super(no_grad, self).__init__()
        version = torch.__version__ if str(torch.__version__) is torch.__version__ else "0.4.1"
        if LooseVersion(version) >= LooseVersion("0.4.1"):  # No grad isn't a decorator
            self.__call__ = torch.no_grad.__call__

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class enable_grad(torch.enable_grad):
    def __init__(self):
        super(enable_grad, self).__init__()
        version = torch.__version__ if str(torch.__version__) is torch.__version__ else "0.4.1"
        if LooseVersion(version) >= LooseVersion("0.4.1"):  # Enable grad isn't a decorator
            self.__call__ = torch.enable_grad.__call__

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_enable_grad(*args, **kwargs):
            with torch.enable_grad():
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
            :class:`.Callback`: self
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

    def on_end_validation(self, state):
        """Perform some action with the given state as context at the end of the validation loop.

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

        # Forward Pass
        try:
            state[y_pred] = state[model](state[x], state=state)
        except TypeError:
            state[y_pred] = state[model](state[x])

        state[torchbearer.CALLBACK_LIST].on_forward(state)

        # Loss Calculation
        try:
            state[loss] = state[crit](state)
        except TypeError:
            state[loss] = state[crit](state[y_pred], state[y_true])

        state[torchbearer.CALLBACK_LIST].on_criterion(state)

        # Backwards pass
        state[loss].backward(**state[torchbearer.BACKWARD_ARGS])

        state[torchbearer.CALLBACK_LIST].on_backward(state)
    return closure


def fluent(func):
    """Decorator for class methods which forces return of self.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return wrapper
