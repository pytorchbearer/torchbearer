from distutils.version import LooseVersion
import functools

import torch


def no_grad():
    version = torch.__version__ if str(torch.__version__) is torch.__version__ else "0.4.1"
    if LooseVersion(version) < LooseVersion("0.4.1"):  # No grad isn't a decorator
        def decorator(func):
            @functools.wraps(func)
            def wrap_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return func(*args, **kwargs)
            return wrap_no_grad
        return decorator
    else:
        return torch.no_grad()


class Metric(object):
    """Base metric class. Process will be called on each batch, process-final at the end of each epoch.
    The metric contract allows for metrics to take any args but not kwargs. The initial metric call will be given state,
    however, subsequent metrics can pass any values desired.

    .. note::

        All metrics must extend this class.

    :param name: The name of the metric
    :type name: str

    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @no_grad()
    def process(self, *args):
        """Process the state and update the metric for one iteration.

        :param args: Arguments given to the metric. If this is a root level metric, will be given state
        :return: None, or the value of the metric for this batch

        """
        pass

    @no_grad()
    def process_final(self, *args):
        """Process the terminal state and output the final value of the metric.

        :param args: Arguments given to the metric. If this is a root level metric, will be given state
        :return: None or the value of the metric for this epoch

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

        :param state: The current state dict of the :class:`.Model`.

        """
        pass


class Callback(object):
    """Base callback class.

    .. note::

        All callbacks should override this class.

    """

    def state_dict(self):
        """Get a dict containing the callback state.

        :return: A dict containing parameters and persistent buffers.
        :rtype: dict
        """
        return {}

    def __str__(self):
        return str(self.__class__).replace('<class ', '').replace('>', '').replace("'", "")

    def load_state_dict(self, state_dict):
        """Resume this callback from the given state. Expects that this callback was constructed in the same way.

        :param state_dict: The state dict to reload
        :type state_dict: dict
        :return: self
        :rtype: Callback
        """
        return self

    def on_start(self, state):
        """Perform some action with the given state as context at the start of a model fit.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_start_epoch(self, state):
        """Perform some action with the given state as context at the start of each epoch.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_start_training(self, state):
        """Perform some action with the given state as context at the start of the training loop.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_sample(self, state):
        """Perform some action with the given state as context after data has been sampled from the generator.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_forward(self, state):
        """Perform some action with the given state as context after the forward pass (model output) has been completed.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_criterion(self, state):
        """Perform some action with the given state as context after the criterion has been evaluated.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_backward(self, state):
        """Perform some action with the given state as context after backward has been called on the loss.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_step_training(self, state):
        """Perform some action with the given state as context after step has been called on the optimiser.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end_training(self, state):
        """Perform some action with the given state as context after the training loop has completed.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end_epoch(self, state):
        """Perform some action with the given state as context at the end of each epoch.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_checkpoint(self, state):
        """Perform some action with the state after all other callbacks have completed at the end of an epoch and the
        history has been updated. Should only be used for taking checkpoints or snapshots and will only be called by the
        run method of Trial.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end(self, state):
        """Perform some action with the given state as context at the end of the model fitting.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_start_validation(self, state):
        """Perform some action with the given state as context at the start of the validation loop.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_sample_validation(self, state):
        """Perform some action with the given state as context after data has been sampled from the validation generator.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_forward_validation(self, state):
        """Perform some action with the given state as context after the forward pass (model output) has been completed
        with the validation data.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_criterion_validation(self, state):
        """Perform some action with the given state as context after the criterion evaluation has been completed
        with the validation data.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end_validation(self, state):
        """Perform some action with the given state as context at the end of the validation loop.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass

    def on_step_validation(self, state):
        """Perform some action with the given state as context at the end of each validation step.

        :param state: The current state dict of the :class:`.Model`.
        :type state: dict[str,any]

        """
        pass
