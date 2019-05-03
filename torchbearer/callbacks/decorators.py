import sys
if sys.version_info[0] < 3:
    import inspect

    def count_args(fcn):
        return len(inspect.getargspec(fcn).args)
else:
    from inspect import signature

    def count_args(fcn):
        return len(signature(fcn).parameters)

import types

import torchbearer
from torchbearer.callbacks import Callback


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
        setattr(callback, target.__name__, types.MethodType(lambda self, state: self.on_lambda(state), callback))
        return callback
    return decorator


def on_start(func):
    """ The :func:`on_start` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_start` calling func
    """
    return bind_to(Callback.on_start)(func)


def on_start_epoch(func):
    """ The :func:`on_start_epoch` decorator is used to initialise a :class:`.Callback` with
    :meth:`~.Callback.on_start_epoch` calling the decorated function

        Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_start_epoch` calling func
    """
    return bind_to(Callback.on_start_epoch)(func)


def on_start_training(func):
    """ The :func:`on_start_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_training`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_start_training` calling func
    """
    return bind_to(Callback.on_start_training)(func)


def on_sample(func):
    """ The :func:`on_sample` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_sample`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_sample` calling func
    """
    return bind_to(Callback.on_sample)(func)


def on_forward(func):
    """ The :func:`on_forward` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_forward`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_forward` calling func
    """
    return bind_to(Callback.on_forward)(func)


def on_criterion(func):
    """ The :func:`on_criterion` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_criterion`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_criterion` calling func
    """
    return bind_to(Callback.on_criterion)(func)


def on_backward(func):
    """ The :func:`on_backward` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_backward`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_backward` calling func
    """
    return bind_to(Callback.on_backward)(func)


def on_step_training(func):
    """ The :func:`on_step_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_step_training`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_step_training` calling func
    """
    return bind_to(Callback.on_step_training)(func)


def on_end_training(func):
    """ The :func:`on_end_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_training`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_end_training` calling func
    """
    return bind_to(Callback.on_end_training)(func)


def on_end_epoch(func):
    """ The :func:`on_end_epoch` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_epoch`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_end_epoch` calling func
    """
    return bind_to(Callback.on_end_epoch)(func)


def on_end(func):
    """ The :func:`on_end` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_end` calling func
    """
    return bind_to(Callback.on_end)(func)


def on_start_validation(func):
    """ The :func:`on_start_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_validation`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_start_validation` calling func
    """
    return bind_to(Callback.on_start_validation)(func)


def on_sample_validation(func):
    """ The :func:`on_sample_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_sample_validation`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_sample_validation` calling func
    """
    return bind_to(Callback.on_sample_validation)(func)


def on_forward_validation(func):
    """ The :func:`on_forward_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_forward_validation`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_forward_validation` calling func
    """
    return bind_to(Callback.on_forward_validation)(func)


def on_criterion_validation(func):
    """ The :func:`on_criterion_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_criterion_validation`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_criterion_validation` calling func
    """
    return bind_to(Callback.on_criterion_validation)(func)


def on_end_validation(func):
    """ The :func:`on_end_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_validation`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_end_validation` calling func
    """
    return bind_to(Callback.on_end_validation)(func)


def on_step_validation(func):
    """ The :func:`on_step_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_step_validation`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_step_validation` calling func
    """
    return bind_to(Callback.on_step_validation)(func)


def on_checkpoint(func):
    """ The :func:`on_checkpoint` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_checkpoint`
    calling the decorated function

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback with :meth:`~.Callback.on_checkpoint` calling func
    """
    return bind_to(Callback.on_checkpoint)(func)


def add_to_loss(func):
    """ The :func:`add_to_loss` decorator is used to initialise a :class:`.Callback` with the value returned from func
    being added to the loss

    Args:
        func (function): The function(state) to *decorate*

    Returns:
        Callback: Initialised callback which adds the returned value from func to the loss
    """
    @on_criterion
    @on_criterion_validation
    def add_to_loss_callback(state):
        state[torchbearer.LOSS] = state[torchbearer.LOSS] + func(state)

    return add_to_loss_callback


def once(fcn):
    """
    Decorator to fire a callback once in the lifetime of the callback. If the callback is a class method, each
    instance of the class will fire only once. For functions, only the first instance will fire (even if more than
    one function is present in the callback list).

    Args:
        fcn (function): the `torchbearer callback` function to decorate.

    Returns:
        the decorator
    """
    def _once(self, _):
        try:
            return not self.__done__
        except AttributeError:
            self.__done__ = True
            return True

    return only_if(_once)(fcn)


def once_per_epoch(fcn):
    """Decorator to fire a callback once (on the first call) in any given epoch. If the callback is a class method, each
    instance of the class will fire once per epoch. For functions, only the first instance will fire (even if more than
    one function is present in the callback list).

    .. note::
        The decorated callback may exhibit unusual behaviour if it is reused

    Args:
        fcn (function): the `torchbearer callback` function to decorate.

    Returns:
        the decorator
    """
    def ope(self, state):
        try:
            if state[torchbearer.EPOCH] != self.__last_epoch__:
                self.__last_epoch__ = state[torchbearer.EPOCH]
                return True
            return False
        except AttributeError:
            self.__last_epoch__ = state[torchbearer.EPOCH]
            return True

    return only_if(ope)(fcn)


def only_if(condition_expr):
    """
    Decorator to fire a callback only if the given conditional expression function returns True. The conditional
    expression can be a function of state or self and state. If the decorated function is not a class method (i.e. it
    does not take state) the decorated function will be passed instead. This enables the storing of temporary variables.

    Args:
        condition_expr (function(self, state) or function(self)): a function/lambda which takes state and optionally\
                        self that must evaluate to true for the decorated `torchbearer callback` to be called. The\
                        `state` object passed to the callback will be passed as an argument to the condition function.

    Returns:
        the decorator
    """
    def condition_decorator(fcn):
        if isinstance(fcn, LambdaCallback):
            fcn.func = condition_decorator(fcn.func)
            return fcn
        else:
            count = count_args(fcn)
            if count == 2 and not hasattr(fcn, '__self__'):  # Assume Class method
                def decfcn(o, state):
                    try:
                        res = condition_expr(o, state)
                    except TypeError:
                        res = condition_expr(state)
                    if res:
                        return fcn(o, state)
            else:  # Assume function of state
                def id_fcn(state):
                    return fcn(state)  # Hack to allow setting attributes of bound methods

                def decfcn(state):
                    try:
                        res = condition_expr(id_fcn, state)
                    except TypeError:
                        res = condition_expr(state)
                    if res:
                        return id_fcn(state)
            return decfcn
    return condition_decorator
