from torchbearer.callbacks import Callback
import torchbearer


def on_start(func):
    """ The :func:`on_start` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`~.Callback.on_start` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_start = func
    return callback


def on_start_epoch(func):
    """ The :func:`on_start_epoch` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_epoch`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`~.Callback.on_start_epoch` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_start_epoch = func
    return callback


def on_start_training(func):
    """ The :func:`on_start_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_training`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_start_training` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_start_training = func
    return new_callback


def on_sample(func):
    """ The :func:`on_sample` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_sample`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_sample` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_sample = func
    return callback


def on_forward(func):
    """ The :func:`on_forward` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_forward`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_forward` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_forward = func
    return new_callback


def on_criterion(func):
    """ The :func:`on_criterion` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_criterion`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_criterion` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_criterion = func
    return new_callback


def on_backward(func):
    """ The :func:`on_backward` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_backward`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_backward` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_backward = func
    return callback


def on_step_training(func):
    """ The :func:`on_step_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_step_training`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_step_training` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_step_training = func
    return callback


def on_end_training(func):
    """ The :func:`on_end_training` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_training`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end_training` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_end_training = func
    return callback


def on_end_epoch(func):
    """ The :func:`on_end_epoch` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_epoch`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end_epoch` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_end_epoch = func
    return callback


def on_end(func):
    """ The :func:`on_end` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end` calling func
    :rtype: :class:`.Callback`
    """
    callback = Callback()
    callback.on_end = func
    return callback


def on_start_validation(func):
    """ The :func:`on_start_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_start_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_start_validation` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_start_validation = func
    return new_callback


def on_sample_validation(func):
    """ The :func:`on_sample_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_sample_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_sample_validation` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_sample_validation = func
    return new_callback


def on_forward_validation(func):
    """ The :func:`on_forward_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_forward_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_forward_validation` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_forward_validation = func
    return new_callback


def on_criterion_validation(func):
    """ The :func:`on_criterion_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_criterion_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_criterion_validation` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_criterion_validation = func
    return new_callback


def on_end_validation(func):
    """ The :func:`on_end_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_end_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_end_validation` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_end_validation = func
    return new_callback


def on_step_validation(func):
    """ The :func:`on_step_validation` decorator is used to initialise a :class:`.Callback` with :meth:`~.Callback.on_step_validation`
    calling the decorated function

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback with :meth:`.Callback.on_step_validation` calling func
    :rtype: :class:`.Callback`
    """
    new_callback = Callback()
    new_callback.on_step_validation = func
    return new_callback


def add_to_loss(func):
    """ The :func:`add_to_loss` decorator is used to initialise a :class:`.Callback` with the value returned from func
    being added to the loss

    :param func: The function(state) to *decorate*
    :type func: function
    :return: Initialised callback which adds the returned value from func to the loss
    :rtype: :class:`.Callback`
    """
    def add_to_loss_func(state):
        state[torchbearer.LOSS] += func(state)

    new_callback = Callback()
    new_callback.on_criterion = add_to_loss_func
    new_callback.on_criterion_validation = add_to_loss_func
    return new_callback


def once(fcn):
    """
    Decorator to fire a callback once in the entire fitting procedure.
    :param fcn: the `torchbearer callback` function to decorate.
    :return: the decorator
    """
    done = False

    def _once(_, __):
        nonlocal done
        if not done:
            done = True
        return done

    return only_if(_once)(fcn)


def once_per_epoch(fcn):
    """
    Decorator to fire a callback once (on the first call) in any given epoch.
    :param fcn: the `torchbearer callback` function to decorate.
    :return: the decorator
    """
    last_epoch = None

    def ope(_, state):
        nonlocal last_epoch
        if state[torchbearer.EPOCH] != last_epoch:
            last_epoch = state[torchbearer.EPOCH]
            return True
        return False

    return only_if(ope)(fcn)


def only_if(condition_expr):
    """
    Decorator to fire a callback only if the given conditional expression function returns True.
    :param condition_expr: a function/lambda that must evaluate to true for the
                           decorated `torchbearer callback` to be called. The `state`
                           object passed to the callback will be passed as an argument
                           to the condition function.
    :return: the decorator
    """
    def condition_decorator(fcn):
        def decfcn(*args, **kwargs):
            if condition_expr(*args, **kwargs):
                fcn(*args, **kwargs)
        return decfcn
    return condition_decorator
