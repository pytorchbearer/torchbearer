from torchbearer.callbacks import Callback
import torchbearer
from torchbearer.state import StateKey

from inspect import signature


class LambdaCallback(Callback):
    def __init__(self, func):
        self.func = func

    def on_lambda(self, state):
        return self.func(state)


def bind_to(target):
    def decorator(func):
        if isinstance(func, LambdaCallback):
            callback = func
        else:
            callback = LambdaCallback(func)
        setattr(callback, target.__name__, lambda state: callback.on_lambda(state))
        return callback
    return decorator


def on_start(func):
    """ The :func:`on_start` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`~.Callback.on_start` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_start)(func)


def on_start_epoch(func):
    """ The :func:`on_start_epoch` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_epoch`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`~.Callback.on_start_epoch` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_start_epoch)(func)


def on_start_training(func):
    """ The :func:`on_start_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_training`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_start_training` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_start_training)(func)


def on_sample(func):
    """ The :func:`on_sample` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_sample`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_sample` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_sample)(func)


def on_forward(func):
    """ The :func:`on_forward` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_forward`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_forward` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_forward)(func)


def on_criterion(func):
    """ The :func:`on_criterion` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_criterion`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_criterion` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_criterion)(func)


def on_backward(func):
    """ The :func:`on_backward` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_backward`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_backward` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_backward)(func)


def on_step_training(func):
    """ The :func:`on_step_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_step_training`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_step_training` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_step_training)(func)


def on_end_training(func):
    """ The :func:`on_end_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_training`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end_training` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_end_training)(func)


def on_end_epoch(func):
    """ The :func:`on_end_epoch` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_epoch`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end_epoch` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_end_epoch)(func)


def on_end(func):
    """ The :func:`on_end` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_end)(func)


def on_start_validation(func):
    """ The :func:`on_start_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_start_validation` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_start_validation)(func)


def on_sample_validation(func):
    """ The :func:`on_sample_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_sample_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_sample_validation` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_sample_validation)(func)


def on_forward_validation(func):
    """ The :func:`on_forward_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_forward_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_forward_validation` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_forward_validation)(func)


def on_criterion_validation(func):
    """ The :func:`on_criterion_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_criterion_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_criterion_validation` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_criterion_validation)(func)


def on_end_validation(func):
    """ The :func:`on_end_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end_validation` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_end_validation)(func)


def on_step_validation(func):
    """ The :func:`on_step_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_step_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_step_validation` calling func
    :rtype: :class:`.Callback`
    """
    return bind_to(Callback.on_step_validation)(func)


def add_to_loss(func):
    """ The :func:`add_to_loss` decorator is used to initialise a :class:`.Callback` with the value returned from func
    being added to the loss

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback which adds the returned value from func to the loss
    :rtype: :class:`.Callback`
    """
    @on_criterion
    @on_criterion_validation
    def add_to_loss_callback(state):
        state[torchbearer.LOSS] = state[torchbearer.LOSS] + func(state)

    return add_to_loss_callback


def once(fcn):
    """
    Decorator to fire a callback once in the entire fitting procedure.
    :param fcn: the `torchbearer callback` function to decorate.
    :return: the decorator
    """
    done = False

    def _once(_):
        nonlocal done
        if not done:
            done = True
            return True
        return False

    return only_if(_once)(fcn)


def once_per_epoch(fcn):
    """
    Decorator to fire a callback once (on the first call) in any given epoch.
    :param fcn: the `torchbearer callback` function to decorate.
    :return: the decorator
    """
    last_epoch = None

    def ope(state):
        nonlocal last_epoch
        if state[torchbearer.EPOCH] != last_epoch:
            last_epoch = state[torchbearer.EPOCH]
            return True
        return False

    return only_if(ope)(fcn)


def only_if(condition_expr):
    """
    Decorator to fire a callback only if the given conditional expression function returns True.
    :param condition_expr: a function/lambda that must evaluate to true for the\
                           decorated `torchbearer callback` to be called. The `state`\
                           object passed to the callback will be passed as an argument\
                           to the condition function.
    :return: the decorator
    """
    def condition_decorator(fcn):
        if isinstance(fcn, LambdaCallback):
            fcn.on_lambda = condition_decorator(fcn.on_lambda)
            return fcn
        else:
            params = signature(fcn).parameters
            if len(params) == 2:  # Assume Class method
                def decfcn(o, state):
                    if condition_expr(state):
                        fcn(o, state)
            else:  # Assume function of state
                def decfcn(state):
                    if condition_expr(state):
                        fcn(state)
            return decfcn
    return condition_decorator
